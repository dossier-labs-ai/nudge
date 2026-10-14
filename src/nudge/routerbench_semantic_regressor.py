"""Frozen-split linear-critic evaluation using archived Rune semantic probabilities."""

import argparse
import json
from pathlib import Path

import numpy as np

from .math import fit, reward_scale, vector
from .historical import tilted, bounded_expert
from .routerbench import write_json
from .routerbench_data import MODELS, aggregate, evaluate, fingerprint
from .routerbench_live import choice_probabilities
from .types import Question


def prepare(data, measurements, rubric, decisions, penalty):
    if data['models'] != list(MODELS):
        raise ValueError('model ordering differs from evaluator')
    if measurements['prepared_fingerprint'] != fingerprint(data):
        raise ValueError('Rune measurements belong to another sample')
    if measurements['rubric_fingerprint'] != fingerprint(rubric):
        raise ValueError('archived rubric does not match Rune measurements')
    questions = [Question(name, definition, 'categorical', categories=tuple(levels))
                 for name, (definition, levels) in rubric.items()]
    rows = data['train'] + data['test']
    if len({r['id'] for r in rows}) != len(rows) or len({r['state'] for r in rows}) != len(rows):
        raise ValueError('duplicate prompts or partition overlap')
    if len(measurements['records']) != len(rows) or len(decisions) != len(rows):
        raise ValueError('incomplete saved observations')
    logged = {r['id']: r for r in decisions}
    if len(logged) != len(rows) or set(logged) != {r['id'] for r in rows}:
        raise ValueError('decision identity mismatch')
    chosen, full, test = [], [], []
    for i, (recorded, measured) in enumerate(zip(rows, measurements['records'])):
        if measured['prompt_sha256'] != fingerprint(recorded['state']):
            raise ValueError('Rune feature alignment mismatch')
        if set(measured['probabilities']) != set(rubric):
            raise ValueError('incomplete semantic questions')
        features = {q.name: choice_probabilities(
            {'type': 'choice', 'probabilities': measured['probabilities'][q.name]}, q.categories)
            for q in questions}
        decision = logged[recorded['id']]
        partition = 'train' if i < len(data['train']) else 'test'
        if decision['partition'] != partition:
            raise ValueError('saved split differs from prepared split')
        baseline = choice_probabilities({'type': 'choice', 'probabilities': decision['rune_probabilities']}, MODELS)
        row = {'state': recorded['state'], 'features': features, 'jev': baseline, 'weight': 1.0}
        if partition == 'train':
            selected = MODELS.index(decision['action'])
            chosen.append({**row, 'action': decision['action'],
                           'reward': recorded['quality'][selected] - penalty * recorded['cost'][selected]})
            full.extend({**row, 'action': model,
                         'reward': recorded['quality'][j] - penalty * recorded['cost'][j]}
                        for j, model in enumerate(MODELS))
        else:
            test.append((row, recorded))
    return chosen, full, test, questions


def compare(data, measurements, rubric, decisions, expert, *, penalty=10.0):
    chosen, full, test, questions = prepare(data, measurements, rubric, decisions, penalty)
    # Match the existing critic's scale/prior/regularization, with training feedback only.
    scale = reward_scale([r['reward'] for r in chosen])
    results, predictions = {}, {}
    for name, train in (('chosen_feedback', chosen), ('all_training_outcomes', full)):
        critics = fit(train, MODELS, questions, scale)
        scores = np.array([[float(np.array(critics[m]['mean']) @
                                  vector(row['jev'][m], row['features'], questions))
                            for m in MODELS] for row, _ in test])
        predictions[name] = scores.tolist()
        for policy_name in ('bounded_tilt', 'greedy'):
            records = []
            for (row, recorded), values in zip(test, scores):
                policy = (tilted(row['jev'], dict(zip(MODELS, values)), scale, expert)
                          if policy_name == 'bounded_tilt' else
                          {m: float(j == int(values.argmax())) for j, m in enumerate(MODELS)})
                records.append({'family': recorded['family'],
                                'rune': evaluate(recorded, row['jev'], penalty),
                                'nudge': evaluate(recorded, policy, penalty)})
            results[f'{name}_{policy_name}'] = aggregate(records)
    # A fixed-model comparator is selected on training utility, never test outcomes.
    fixed_index = int(np.array([[r['quality'][j] - penalty * r['cost'][j] for j in range(len(MODELS))]
                               for r in data['train']]).mean(axis=0).argmax())
    fixed_policy = {m: float(j == fixed_index) for j, m in enumerate(MODELS)}
    fixed_metrics = [evaluate(r, fixed_policy, penalty) for r in data['test']]
    return {'training_prompts': len(chosen), 'test_prompts': len(test),
            'questions': [q.to_dict() for q in questions], 'rubric_fingerprint': fingerprint(rubric),
            'prepared_fingerprint': fingerprint(data), 'heads_per_call': measurements['heads_per_call'],
            'regressor': 'existing per-action Gaussian linear critic; posterior means; ridge alpha 16',
            'feature_representation': 'intercept, per-model baseline logit, all Rune category probabilities',
            'scale_from_training_chosen_feedback': scale, 'penalty': penalty,
            'results': results, 'test_predictions': predictions,
            'test_prompt_ids': [r['id'] for _, r in test],
            'training_selected_fixed_model': {'model': MODELS[fixed_index],
                **{key: float(np.mean([m[key] for m in fixed_metrics])) for key in fixed_metrics[0]}},
            'scope': ['Offline refit on the existing 800/200 partition; no new inference or exploration.',
                      'Saved original 16-question measurements; not the revised 17-question rubric.',
                      'Full outcome training is a stronger information setting than logged chosen feedback.',
                      'Questions were designed after inspecting this sample; test split is reused/exploratory.',
                      'No test rewards used for fitting, regularization selection, or reward scaling.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'measurements', 'semantic_report', 'decisions', 'manifest', 'out'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.prepared.read_text())
    manifest = json.loads(args.manifest.read_text())
    if manifest['data_sha256'] != fingerprint(data):
        parser.error('manifest sample mismatch')
    result = compare(data, json.loads(args.measurements.read_text()),
                     json.loads(args.semantic_report.read_text())['rubric'],
                     [json.loads(line) for line in args.decisions.read_text().splitlines()],
                     bounded_expert(manifest['expert']), penalty=manifest['penalty_per_usd'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(json.dumps({k: v['nudge'] for k, v in result['results'].items()}, indent=2))


if __name__ == '__main__':
    main()
