"""Descriptive action-by-answer evidence, independent of the routing regressor.

Retention is a proposer decision, not a significance test. These summaries use
only chosen outcomes and known logging probabilities; unseen outcomes stay unknown.
"""

from itertools import combinations

import numpy as np


def bucket(question, value):
    if question.kind == 'categorical':
        return question.categories[int(np.argmax(question.feature_vector(value)))]
    return str(min(3, int(question.normalize(value) * 4)))


def evidence(rows, question, actions):
    rows = [r for r in rows if r['weight'] > 0]
    cells = {}
    support_gaps = set()
    for row in rows:
        propensity = row['propensity']
        if (not np.isfinite(propensity) or not 0 < propensity <= 1
                or not np.isclose(propensity, row['policy'][row['action']])):
            raise ValueError('invalid logged propensity')
        key = (bucket(question, row['features'][question.name]), row['action'])
        support_gaps.update((key[0], a) for a in actions if row['policy'][a] <= 0)
        cells.setdefault(key, []).append((row['weight'] / propensity, row['reward']))
    summaries = {}
    for key, values in cells.items():
        weights, rewards = np.array(values).T
        ess = float(weights.sum() ** 2 / (weights @ weights))
        summaries[key] = {'mean': float(np.average(rewards, weights=weights)),
                          'effective_count': ess, 'count': len(values)}
    groups = sorted({key[0] for key in cells})
    contrasts = []
    for a, b in combinations(actions, 2):
        supported = [g for g in groups if all(
            summaries.get((g, action), {}).get('effective_count', 0) >= 2
            and (g, action) not in support_gaps for action in (a, b))]
        if len(supported) < 2:
            continue
        gaps = [summaries[g, a]['mean'] - summaries[g, b]['mean'] for g in supported]
        center = float(np.mean(gaps))
        for group, gap in zip(supported, gaps):
            contrasts.append({'actions': [a, b], 'answer': group,
                              'effective_counts': [summaries[group, action]['effective_count']
                                                   for action in (a, b)],
                              'advantage': gap, 'centered_advantage': gap - center})
    return {'observation_count': len(rows),
            'unsupported_cells': [{'answer': g, 'action': a} for g, a in sorted(support_gaps)],
            'cells': [{'answer': g, 'action': a, **v} for (g, a), v in sorted(summaries.items())],
            'contrasts': contrasts,
            'discrimination_rms': float(np.sqrt(np.mean([
                c['centered_advantage'] ** 2 for c in contrasts]))) if contrasts else None,
            'interpretation': 'descriptive, propensity-weighted; not certified usefulness'}


def replication(original, newer):
    def index(profile):
        return {(tuple(c['actions']), c['answer']): c['advantage'] for c in profile['contrasts']}
    left, right = index(original), index(newer)
    # Recenter using only shared answer groups for each pair.
    products, agreements = [], []
    for pair in sorted({key[0] for key in left.keys() & right.keys()}):
        keys = sorted(k for k in left.keys() & right.keys() if k[0] == pair)
        if len(keys) < 2:
            continue
        x, y = np.array([left[k] for k in keys]), np.array([right[k] for k in keys])
        x, y = x - x.mean(), y - y.mean()
        for a, b in zip(x, y):
            if abs(a) > 1e-12 and abs(b) > 1e-12:
                products.append(float(a * b))
                agreements.append(bool(a * b > 0))
    return {'comparable_contrasts': len(products),
            'mean_cross_product': float(np.mean(products)) if products else None,
            'direction_agreement': float(np.mean(agreements)) if agreements else None}


def update_evidence(rows, questions, actions, previous, objective):
    result = {}
    for question in questions:
        from .discovery import question_key
        key = question_key(question)
        current = evidence(rows, question, actions)
        prior = previous.get(key)
        if prior is None or prior['objective'] != objective:
            prior = {'objective': objective,
                     'decision_boundary': max((r.get('observation_boundary', r['decision_order'])
                                               for r in rows), default=0),
                     'initial': current}
        newer = evidence([r for r in rows if r['decision_order'] > prior['decision_boundary']],
                         question, actions)
        result[key] = {**prior, 'question': question.to_dict(), 'current': current,
                       'subsequent': newer, 'replication': replication(prior['initial'], newer),
                       'status': 'retained_hypothesis'}
    return result


def summarize(record):
    """Bound prompt size; complete cell-level evidence remains in the snapshot."""
    def profile(value):
        return {key: value[key] for key in ('observation_count', 'discrimination_rms')} | {
            'supported_contrasts': len(value['contrasts']),
            'unsupported_cells': len(value['unsupported_cells']),
            'largest_contrasts': sorted(value['contrasts'],
                                        key=lambda c: -abs(c['centered_advantage']))[:6]}
    return {**{k: record[k] for k in ('question', 'status', 'decision_boundary', 'replication')},
            **{k: profile(record[k]) for k in ('initial', 'current', 'subsequent')}}
