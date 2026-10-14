"""Offline known-question retention diagnostic; no invented paraphrase measurements."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .math import fit, policy_probabilities, reward_scale
from .routerbench import write_json
from .routerbench_data import MODELS, aggregate, evaluate
from .routerbench_semantic_regressor import prepare
from .types import Expert
from .utility import question_utility, retention_status


def select(rows, questions, expert):
    training = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'selection']
    scale = reward_scale([r['reward'] for r in training])
    retained, history = [], []
    for round_number in range(expert.max_rounds):
        remaining = [q for q in questions if q not in retained]
        if not remaining:
            break
        proposals = [[q] for q in remaining]
        if len(remaining) > 1:
            proposals.append(remaining)
        current = fit(training, MODELS, retained, scale)
        measured = []
        for additions in proposals:
            candidate_questions = retained + additions
            candidate = fit(training, MODELS, candidate_questions, scale)
            result = question_utility(current, candidate, validation, validation,
                                      retained, candidate_questions, scale, expert)
            result.update(round=round_number + 1, additions=[q.name for q in additions],
                          status=retention_status(result, expert))
            measured.append((result, additions))
        history.extend(result for result, _ in measured)
        accepted = [(result, additions) for result, additions in measured if result['status'] == 'kept']
        if not accepted:
            break
        winner, additions = max(accepted, key=lambda pair: pair[0]['gain_lower_bound'])
        winner['selected'] = True
        retained += additions
    return retained, history


def feedback_rows(data, measurements, rubric, decisions, *, seed=20261013):
    chosen, _, test, questions = prepare(data, measurements, rubric, decisions, 10)
    expert = Expert()
    for i, row in enumerate(chosen):
        identifier = f'routerbench-{seed}-{i + 1}'
        row.update(id=identifier, decision_order=i + 1, policy=decisions[i]['nudge_probabilities'],
                   propensity=decisions[i]['propensity'],
                   split='selection' if int(hashlib.sha256(identifier.encode()).hexdigest()[:16], 16)
                   / 2**64 < expert.selection_fraction else 'train')
    # New-policy replay reveals exactly one outcome per prompt. Full vectors stay here.
    rng = np.random.default_rng(seed)
    fresh = []
    scale = 1.0
    critics = fit([], MODELS, [], scale)
    for i, row in enumerate(chosen):
        p = policy_probabilities(critics, [row], [], scale, expert)[0]
        j = int(rng.choice(len(MODELS), p=p))
        recorded = data['train'][i]
        fresh.append({**row, 'action': MODELS[j], 'policy': dict(zip(MODELS, map(float, p))),
                      'propensity': float(p[j]),
                      'reward': recorded['quality'][j] - 10 * recorded['cost'][j]})
        if (i + 1) % 200 == 0:
            scale = reward_scale([r['reward'] for r in fresh])
            critics = fit(fresh, MODELS, [], scale)
    return chosen, fresh, test, questions


def measure(data, measurements, rubric, decisions, *, seed=20261013):
    chosen, fresh, test, questions = feedback_rows(data, measurements, rubric, decisions, seed=seed)
    expert = Expert()
    results = {}
    for name, rows in [('archived_chosen_feedback', chosen), ('unrestricted_exploration_feedback', fresh)]:
        retained, history = select(rows, questions, expert)
        scale = reward_scale([r['reward'] for r in rows])
        metrics, policies = {}, {}
        for label, selected in [('no_questions', []), ('retained', retained), ('all_questions', questions)]:
            critics = fit(rows, MODELS, selected, scale)
            probabilities = policy_probabilities(critics, [r for r, _ in test], selected, scale, expert)
            policies[label] = probabilities.tolist()
            metrics[label] = aggregate([
                {'family': record['family'], 'rune': evaluate(record, row['jev'], 10),
                 'nudge': evaluate(record, dict(zip(MODELS, p)), 10)}
                for (row, record), p in zip(test, probabilities)])
        results[name] = {'retained_questions': [q.name for q in retained], 'selection': history,
                         'test': metrics, 'test_policies': policies,
                         'training_action_counts': {a: sum(r['action'] == a for r in rows) for a in MODELS}}
    return {'train_count': len(chosen), 'test_count': len(test), 'seed': seed, 'results': results,
            'expert': asdict(expert), 'prepared_fingerprint': measurements['prepared_fingerprint'],
            'rubric_fingerprint': measurements['rubric_fingerprint'],
            'scope': ['Known original 16 questions; saved Rune probabilities.',
                      'Utility-retention diagnostic; paraphrase stability not remeasured or fabricated.',
                      'Selection uses training-only fit/selection partition; final test never selects.',
                      'Exploration feedback uses unrestricted no-question policy, one outcome per prompt.',
                      'No live discovery or all-outcome training. Reused exploratory test split.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('prepared', 'measurements', 'semantic_report', 'decisions', 'out'):
        parser.add_argument('--' + key.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    result = measure(json.loads(args.prepared.read_text()), json.loads(args.measurements.read_text()),
                     json.loads(args.semantic_report.read_text())['rubric'],
                     [json.loads(line) for line in args.decisions.read_text().splitlines()])
    write_json(args.out, result)
    print(json.dumps({k: {'retained': v['retained_questions'],
                         'test_utility': {p: m['nudge']['utility'] for p, m in v['test'].items()}}
                      for k, v in result['results'].items()}, indent=2))


if __name__ == '__main__':
    main()
