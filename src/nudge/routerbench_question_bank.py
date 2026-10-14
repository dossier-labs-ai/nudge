"""Saved-answer question-bank evidence; independent of regressor policy scoring."""

import argparse
import json
from pathlib import Path

from .discovery import answer_statistics
from .math import fit, policy_probabilities, reward_scale
from .question_evidence import update_evidence
from .routerbench import write_json
from .routerbench_data import MODELS, aggregate, evaluate
from .routerbench_retention import feedback_rows
from .types import Expert


def measure(data, measurements, rubric, decisions):
    chosen, fresh, test, questions = feedback_rows(data, measurements, rubric, decisions)
    expert = Expert()
    results = {}
    for name, rows in [('archived_chosen_feedback', chosen), ('unrestricted_exploration_feedback', fresh)]:
        # Fixed existing hypotheses, not a live LLM selection experiment.
        retained = [q for q in questions if answer_statistics(q, rows)['answer_std'] > expert.min_feature_std]
        diagnostics = update_evidence(rows, retained, MODELS, {}, {'penalty': 10})
        boundary = len(rows) // 2
        first = update_evidence(rows[:boundary], retained, MODELS, {}, {'penalty': 10})
        subsequent = update_evidence(rows, retained, MODELS, first, {'penalty': 10})
        scale = reward_scale([r['reward'] for r in rows])
        metrics, policies = {}, {}
        for label, selected in [('no_questions', []), ('question_bank', retained)]:
            critics = fit(rows, MODELS, selected, scale)
            p = policy_probabilities(critics, [r for r, _ in test], selected, scale, expert)
            policies[label] = p.tolist()
            metrics[label] = aggregate([
                {'family': record['family'], 'rune': evaluate(record, row['jev'], 10),
                 'nudge': evaluate(record, dict(zip(MODELS, probabilities)), 10)}
                for (row, record), probabilities in zip(test, p)])
        results[name] = {'retained_questions': [q.name for q in retained],
                         'all_800_evidence': list(diagnostics.values()),
                         'retrospective_first_half_next_half': list(subsequent.values()),
                         'test': metrics, 'test_policies': policies}
    return {'results': results, 'train_count': len(chosen), 'test_count': len(test),
            'prepared_fingerprint': measurements['prepared_fingerprint'],
            'rubric_fingerprint': measurements['rubric_fingerprint'],
            'scope': ['Fixed original 16 Rune questions; no new LLM proposals.',
                      'Retained as hypotheses based on saved-answer variation; paraphrases not remeasured.',
                      'All 800 chosen outcomes inform question evidence; no regressor retention gate.',
                      'First-half/next-half comparison is retrospective: original questions were designed after inspecting this sample.',
                      'Positive replication is descriptive, not proof of useful routing or independent validation.',
                      'Separate policy fit uses 800 rows, final evaluation uses 200; test outcomes never select questions.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('prepared', 'measurements', 'semantic_report', 'decisions', 'out'):
        parser.add_argument('--' + key.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    result = measure(json.loads(args.prepared.read_text()), json.loads(args.measurements.read_text()),
                     json.loads(args.semantic_report.read_text())['rubric'],
                     [json.loads(line) for line in args.decisions.read_text().splitlines()])
    write_json(args.out, result)
    print(json.dumps({name: {'retained': r['retained_questions'],
                            'replication': {d['question']['name']: d['replication']
                                            for d in r['retrospective_first_half_next_half']},
                            'test_utility': {k: v['nudge']['utility'] for k, v in r['test'].items()}}
                      for name, r in result['results'].items()}, indent=2))


if __name__ == '__main__':
    main()
