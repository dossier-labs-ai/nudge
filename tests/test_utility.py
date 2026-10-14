import copy
import unittest

import numpy as np

from nudge import Expert, Question
from nudge.discovery import answer_statistics
from nudge.math import fit, mse
from nudge.utility import compare_policies, policy_probabilities, question_utility, retention_status


class UtilityTests(unittest.TestCase):
    def test_uncertainty_is_pending_and_clear_failures_are_distinct(self):
        result = {'supported': True, 'gain_lower_bound': -0.01,
                  'approximate_interval_95': [-0.01, 0.02], 'reward_scale': 1}
        self.assertEqual(retention_status(result, Expert()), 'deferred_uncertainty')
        self.assertEqual(retention_status({**result, 'supported': False}, Expert()), 'deferred_coverage')
        self.assertEqual(retention_status({**result, 'gain_lower_bound': 0.01}, Expert()), 'kept')
        self.assertEqual(retention_status({**result, 'approximate_interval_95': [-0.02, -0.01]}, Expert()),
                         'no_utility_gain')

    def balanced(self):
        # Each context has both actions represented, but every row has only one outcome.
        return [dict(id=str(i), action='a' if i % 2 == 0 else 'b',
                     jev={'b': 0.5, 'a': 0.5}, policy={'a': 0.5, 'b': 0.5},
                     propensity=0.5, reward=float((i // 2) % 2), weight=1.0,
                     features={'difficulty': float((i // 2) % 2)}) for i in range(200)]

    def compare(self, rows, current, candidate, predictions):
        return compare_policies(rows, current, candidate, predictions,
                                actions=['a', 'b'], scale=1.0, min_support=10)

    def test_logged_propensities_recover_known_utility_with_wrong_reward_model(self):
        rows = self.balanced()
        for i, row in enumerate(rows):
            # Deterministic 3:1 allocation; known logged propensity, arbitrary reward units.
            row['action'] = 'a' if i % 4 < 3 else 'b'
            row['policy'] = {'a': 0.75, 'b': 0.25}
            row['propensity'] = row['policy'][row['action']]
            row['reward'] = 2.0 if row['action'] == 'a' else 6.0
        current = np.tile([0.5, 0.5], (len(rows), 1))
        candidate = np.tile([0.25, 0.75], (len(rows), 1))
        result = self.compare(rows, current, candidate, np.zeros_like(current))
        self.assertAlmostEqual(result['current_utility'], 4.0)
        self.assertAlmostEqual(result['candidate_utility'], 5.0)
        self.assertAlmostEqual(result['utility_gain'], 1.0)
        self.assertGreater(result['gain_lower_bound'], 0)
        self.assertTrue(result['supported'])

    def test_identical_policies_have_zero_paired_gain_and_uncertainty(self):
        rows = self.balanced()
        p = np.tile([0.4, 0.6], (len(rows), 1))
        result = self.compare(rows, p, p, np.ones_like(p) * 123)
        self.assertEqual(result['utility_gain'], 0)
        self.assertEqual(result['standard_error'], 0)

    def test_better_reward_mse_with_no_action_discrimination_has_no_utility_gain(self):
        rows = self.balanced()
        q = Question('difficulty', 'Is this difficult?')
        current = fit(rows[:100], ['a', 'b'], [], 1)
        candidate = fit(rows[:100], ['a', 'b'], [q], 1)
        self.assertLess(mse(candidate, rows[100:], [q]), mse(current, rows[100:], []))
        result = question_utility(current, candidate, rows[100:], rows[100:], [], [q],
                                  1, Expert(explore_floor=1))
        self.assertAlmostEqual(result['utility_gain'], 0)
        self.assertAlmostEqual(result['mean_policy_change'], 0)

    def test_uncovered_actions_are_not_silently_extrapolated(self):
        rows = self.balanced()
        for row in rows:
            row.update(action='a', policy={'a': 1.0, 'b': 0.0}, propensity=1.0)
        current = np.tile([1.0, 0.0], (len(rows), 1))
        candidate = np.tile([0.5, 0.5], (len(rows), 1))
        result = self.compare(rows, current, candidate, np.tile([0, 100], (len(rows), 1)))
        self.assertFalse(result['supported'])
        self.assertEqual(result['unsupported_actions'], ['b'])
        self.assertEqual(result['underobserved_changed_actions'], ['b'])

    def test_weak_effective_coverage_and_invalid_propensities(self):
        rows = self.balanced()
        for row in rows[1:]:
            row['weight'] = 0.00001
        p = np.tile([0.5, 0.5], (len(rows), 1))
        self.assertFalse(self.compare(rows, p, p, p)['supported'])
        for invalid in (0, -1, float('nan'), 0.25):
            rows[0]['propensity'] = invalid
            with self.assertRaises(ValueError):
                self.compare(rows, p, p, p)

    def test_policy_is_batch_invariant_and_ignores_selection_outcomes(self):
        rows = self.balanced()
        q = Question('difficulty', 'Is this difficult?')
        critics = fit(rows[:100], ['a', 'b'], [q], 1)
        expert = Expert()
        p = policy_probabilities(critics, rows[100:], [q], 1, expert)
        r = rows[100]
        expected = policy_probabilities(critics, [r], [q], 1, expert)[0]
        np.testing.assert_allclose(p[0], expected)
        reversed_critics = dict(reversed(list(critics.items())))
        reversed_policy = policy_probabilities(reversed_critics, [r], [q], 1, expert)[0]
        np.testing.assert_allclose(p[0], reversed_policy[::-1])
        changed = copy.deepcopy(rows[100:])
        for row in changed:
            row['reward'] = 10000
        np.testing.assert_array_equal(p, policy_probabilities(critics, changed, [q], 1, expert))
        sampled = policy_probabilities(critics, rows[100:], [q], 1, Expert())
        np.testing.assert_array_equal(sampled, policy_probabilities(
            critics, rows[100:], [q], 1, Expert()))
        np.testing.assert_allclose(sampled.sum(axis=1), 1)

    def test_reward_rescaling_preserves_normalized_gain(self):
        rows = self.balanced()
        for row in rows:
            row['reward'] = float(row['action'] == 'b')
        p = np.tile([0.5, 0.5], (len(rows), 1))
        q = np.tile([0.25, 0.75], (len(rows), 1))
        first = self.compare(rows, p, q, p)
        for row in rows:
            row['reward'] *= 100
        second = compare_policies(rows, p, q, p * 100, actions=['a', 'b'],
                                  scale=100, min_support=10)
        self.assertAlmostEqual(first['gain'], second['gain'])
        self.assertAlmostEqual(first['gain_lower_bound'], second['gain_lower_bound'])

    def test_categorical_variation_is_not_collapsed_to_mean_category_index(self):
        q = Question('kind', 'Kind?', 'categorical', categories=('a', 'b', 'c'))
        rows = [{'weight': 1, 'features': {'kind': value}} for value in
                ({'a': 0.5, 'b': 0, 'c': 0.5}, {'a': 0, 'b': 1, 'c': 0})]
        self.assertEqual(q.normalize(rows[0]['features']['kind']),
                         q.normalize(rows[1]['features']['kind']))
        self.assertGreater(answer_statistics(q, rows)['answer_std'], 0.5)
