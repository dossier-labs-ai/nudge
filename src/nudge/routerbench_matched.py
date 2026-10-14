"""Matched question-set ablations on one frozen chosen-feedback trajectory."""

import argparse
import json
from pathlib import Path

import numpy as np

from .math import fit, policy_probabilities, reward_scale, vector
from .routerbench import write_json
from .routerbench_data import MODELS, aggregate, evaluate, fingerprint
from .routerbench_diagnose import load_recorded_run
from .routerbench_semantic_regressor import prepare
from .types import Expert


def compare_sets(train, test, feature_sets, expert, *, penalty=10, full_information=False,
                 scale=None):
    """Only feature representations vary; feedback, baselines and fit stay fixed.

    Each representation contains questions and state-keyed measured features.
    Final outcomes are used only after all fitting and policy construction.
    """
    states = [r['state'] for r in train] + [r['state'] for r, _ in test]
    training_states = [r['state'] for r in train]
    testing_states = [r['state'] for r, _ in test]
    if (set(training_states) & set(testing_states)
            or len(testing_states) != len(set(testing_states))
            or (not full_information and len(training_states) != len(set(training_states)))):
        raise ValueError('duplicate states or train/test overlap')
    if full_information:
        pairs = {(r['state'], r['action']) for r in train}
        if len(pairs) != len(train) or len(pairs) != len(set(training_states)) * len(MODELS):
            raise ValueError('full-information rows must contain every action exactly once')
    scale = reward_scale([r['reward'] for r in train]) if scale is None else scale
    policies, fits, scores = {}, {}, {}
    for name, (questions, measured) in feature_sets.items():
        if questions and set(measured) != set(states):
            raise ValueError('incomplete or misaligned question measurements')
        training = [{**r, 'features': measured.get(r['state'], {})} for r in train]
        testing = [{**r, 'features': measured.get(r['state'], {})} for r, _ in test]
        critics = fit(training, MODELS, questions, scale)
        fits[name] = fingerprint(critics)
        values = np.array([[np.array(critics[m]['mean']) @
                            vector(r['jev'][m], r['features'], questions)
                            for m in MODELS] for r in testing])
        scores[name] = values.tolist()
        policies[name + '/greedy'] = np.eye(len(MODELS))[values.argmax(axis=1)]
        policies[name + '/posterior'] = policy_probabilities(
            critics, testing, questions, scale, expert)
    metrics, evaluated = {}, {}
    for name, probabilities in policies.items():
        records = [{'family': record['family'],
                    'rune': evaluate(record, row['jev'], penalty),
                    'nudge': evaluate(record, dict(zip(MODELS, p)), penalty)}
                   for (row, record), p in zip(test, probabilities)]
        evaluated[name] = records
        metrics[name] = aggregate(records)
        metrics[name]['mean_action_probability'] = dict(zip(
            MODELS, map(float, probabilities.mean(axis=0))))
    paired, against_reference = {}, {}
    for rule in ('greedy', 'posterior'):
        control = evaluated['no_questions/' + rule]
        for name in feature_sets:
            if name == 'no_questions':
                continue
            records = evaluated[name + '/' + rule]
            paired[name + '/' + rule] = aggregate([
                {'family': a['family'], 'rune': b['nudge'], 'nudge': a['nudge']}
                for a, b in zip(records, control)])['paired_nudge_minus_rune']
        if 'generated' in feature_sets and 'reference' in feature_sets:
            against_reference[rule] = aggregate([
                {'family': a['family'], 'rune': b['nudge'], 'nudge': a['nudge']}
                for a, b in zip(evaluated['generated/' + rule],
                                evaluated['reference/' + rule])])['paired_nudge_minus_rune']
    return {'training_count': len(train), 'training_prompt_count': len(set(training_states)),
            'test_count': len(test), 'scale': scale,
            'training_feedback_fingerprint': fingerprint([
                {k: r[k] for k in ('state', 'action', 'reward', 'jev', 'weight')}
                for r in train]),
            'training_action_counts': {m: sum(r['action'] == m for r in train) for m in MODELS},
            'questions': {name: [q.to_dict() for q in qs]
                          for name, (qs, _) in feature_sets.items()},
            'critic_fingerprints': fits, 'test_scores': scores,
            'test_policies': {name: value.tolist() for name, value in policies.items()},
            'metrics': metrics, 'paired_against_no_questions': paired,
            'generated_minus_reference': against_reference}


def matched(data, measurements, rubric, decisions, *, database=None, expert=None,
            all_training_outcomes=False):
    train, full, test, reference = prepare(data, measurements, rubric, decisions, 10)
    sets = {'no_questions': ([], {}),
            'reference': (reference, {r['state']: r['features']
                                     for r in train + [r for r, _ in test]})}
    if database is not None:
        live, _, heldout, generated, _, _, penalty, snapshot = load_recorded_run(data, database)
        if penalty != 10 or len(heldout) != len(test):
            raise ValueError('live run objective or test completion mismatch')
        # Require exact replay order and chosen feedback. Never silently compare
        # reference features fitted on one trajectory with another live trajectory.
        for recorded, current in zip(train, live):
            if (any(recorded[k] != current[k] for k in ('state', 'action', 'reward'))
                    or not all(abs(recorded['jev'][m] - current['jev'][m]) <= 1e-12
                               for m in MODELS)):
                raise ValueError('live run training feedback or baseline mismatch')
        for (recorded, _), (current, _) in zip(test, heldout):
            if (recorded['state'] != current['state']
                    or not all(abs(recorded['jev'][m] - current['jev'][m]) <= 1e-12
                               for m in MODELS)):
                raise ValueError('live run test alignment or baseline mismatch')
        sets['generated'] = (generated, {r['state']: r['features']
                                        for r in live + [r for r, _ in heldout]})
        expert = Expert(**snapshot['expert'])
    result = compare_sets(train, test, sets, expert or Expert())
    if all_training_outcomes:
        result['all_training_outcomes_diagnostic'] = compare_sets(
            full, test, sets, expert or Expert(), full_information=True, scale=result['scale'])
    result.update(prepared_fingerprint=fingerprint(data),
                  reference_heads_per_call=measurements.get('heads_per_call'),
                  generated_heads_per_call=1 if database is not None else None,
                  scope=['One identical chosen-feedback trajectory across all question sets.',
                         'Same per-action Gaussian linear fit, scale, prior and baseline features.',
                         'Greedy and posterior routing evaluated separately; no all-outcome fitting.',
                         'Reference measurements are cached; generated measurements come from live run.',
                         'Reference Rune measurements used batched heads; generated questions use single heads.',
                         'Reused exploratory 200-prompt test; no tuning or selection on its outcomes.'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('prepared', 'measurements', 'semantic_report', 'decisions', 'out'):
        parser.add_argument('--' + key.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--database', type=Path,
                        help='Completed fresh live policy.db; omit for reference/no-question control.')
    parser.add_argument('--all-training-outcomes', action='store_true',
                        help='Separate full-information fit diagnostic; never used by discovery.')
    args = parser.parse_args()
    result = matched(json.loads(args.prepared.read_text()),
                     json.loads(args.measurements.read_text()),
                     json.loads(args.semantic_report.read_text())['rubric'],
                     [json.loads(line) for line in args.decisions.read_text().splitlines()],
                     database=args.database, all_training_outcomes=args.all_training_outcomes)
    write_json(args.out, result)
    print(json.dumps({k: v['nudge'] for k, v in result['metrics'].items()}, indent=2))


if __name__ == '__main__':
    main()
