"""General chosen-feedback policy comparison for exploratory question discovery."""

import numpy as np

from .math import vector, policy_probabilities

POLICY_DRAWS = 64


def effective_size(weights):
    return float(weights.sum() ** 2 / (weights @ weights)) if np.any(weights) else 0.0


def compare_policies(rows, current, candidate, predictions, *, actions, scale, min_support):
    """Paired DR value estimates using one training-only reward model for both.

    predictions[i,a] is the common control variate, not an observed outcome.
    Selection outcomes enter only the chosen-action residual correction.
    """
    actions = list(actions)
    current, candidate, predictions = map(np.asarray, (current, candidate, predictions))
    shape = (len(rows), len(actions))
    if any(p.shape != shape or not np.isfinite(p).all()
           for p in (current, candidate, predictions)):
        raise ValueError('invalid policy comparison dimensions or values')
    for p in (current, candidate):
        if np.any(p < 0) or not np.allclose(p.sum(axis=1), 1):
            raise ValueError('invalid target policy')
    w = np.array([r['weight'] for r in rows])
    if not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('invalid evaluation weights')
    selected = np.array([actions.index(r['action']) for r in rows])
    logged = np.array([[r['policy'][a] for a in actions] for r in rows])
    mu = np.array([r['propensity'] for r in rows])
    if (not np.isfinite(logged).all() or np.any(logged < 0)
            or not np.allclose(logged.sum(axis=1), 1)
            or not np.isfinite(mu).all() or np.any(mu <= 0)
            or not np.allclose(mu, logged[np.arange(len(rows)), selected])):
        raise ValueError('invalid logged action propensity')
    reward = np.array([r['reward'] for r in rows])
    if not np.isfinite(reward).all():
        raise ValueError('invalid evaluation reward')
    residual = reward - predictions[np.arange(len(rows)), selected]
    estimates, ess, max_ratios = [], [], []
    for policy in (current, candidate):
        ratio = policy[np.arange(len(rows)), selected] / mu
        estimates.append(np.sum(policy * predictions, axis=1) + ratio * residual)
        ess.append(effective_size(w * ratio))
        max_ratios.append(float(ratio[w > 0].max()))
    delta = estimates[1] - estimates[0]
    normalized = w / w.sum()
    gain = float(normalized @ delta)
    # Paired, weighted independent-observation SE; adaptive reuse is not covered.
    correction = 1 - float(normalized @ normalized)
    se = (float(np.sqrt(np.sum(normalized**2 * (delta - gain)**2) / correction))
          if correction > 0 else 0.0)
    coverage = {a: effective_size(w * (selected == j)) for j, a in enumerate(actions)}
    changed = np.sum(normalized[:, None] * np.abs(candidate - current), axis=0)
    unsupported = [a for j, a in enumerate(actions)
                   if np.any((w > 0) & (logged[:, j] == 0) &
                             (np.abs(candidate[:, j] - current[:, j]) > 1e-12))]
    uncovered = [a for j, a in enumerate(actions)
                 if changed[j] > 1e-10 and coverage[a] < max(2, min_support)]
    change_weights = w * np.abs(candidate[np.arange(len(rows)), selected]
                                - current[np.arange(len(rows)), selected]) / mu
    change_ess = effective_size(change_weights)
    missing_change = sum(changed[j] for j, a in enumerate(actions) if coverage[a] < 2)
    missing_fraction = float(missing_change / changed.sum()) if changed.sum() > 1e-12 else 0.0
    supported = (not unsupported and missing_fraction <= 0.1
                 and effective_size(w) >= max(2, min_support)
                 and (change_ess >= min_support or changed.sum() <= 1e-12))
    return {
        'estimator': 'paired_doubly_robust_v1', 'policy_draws': POLICY_DRAWS,
        'reward_scale': scale,
        'current_utility': float(normalized @ estimates[0]),
        'candidate_utility': float(normalized @ estimates[1]),
        'utility_gain': gain, 'gain': gain / scale,
        'standard_error': se, 'gain_lower_bound': (gain - 1.96 * se) / scale,
        'approximate_interval_95': [gain - 1.96 * se, gain + 1.96 * se],
        'current_effective_sample_size': ess[0], 'candidate_effective_sample_size': ess[1],
        'max_importance_ratios': max_ratios, 'action_effective_sample_sizes': coverage,
        'policy_change_effective_sample_size': change_ess,
        'unobserved_change_fraction': missing_fraction,
        'unsupported_actions': unsupported, 'underobserved_changed_actions': uncovered,
        'supported': bool(supported), 'selection_count': int(np.count_nonzero(w)),
        'mean_policy_change': float(np.sum(normalized[:, None] * np.abs(candidate-current)) / 2),
    }


def question_utility(critics, candidate, rows, candidate_rows, questions,
                     candidate_questions, scale, expert):
    if [r['id'] for r in rows] != [r['id'] for r in candidate_rows]:
        raise ValueError('policy comparison requires aligned decision IDs')
    actions = list(critics)
    predictions = np.array([
        [np.array(critics[a]['mean']) @ vector(r['jev'][a], r['features'], questions)
         for a in actions] for r in rows])
    result = compare_policies(
        rows, policy_probabilities(critics, rows, questions, scale, expert),
        policy_probabilities(candidate, candidate_rows, candidate_questions, scale, expert),
        predictions, actions=actions, scale=scale, min_support=expert.min_split)
    result['policy_draws'] = expert.posterior_draws
    return result


def retention_status(result, expert):
    if not result['supported']:
        return 'deferred_coverage'
    if result['gain_lower_bound'] > expert.min_gain:
        return 'kept'
    if result['approximate_interval_95'][1] / result['reward_scale'] > expert.min_gain:
        return 'deferred_uncertainty'
    return 'no_utility_gain'
