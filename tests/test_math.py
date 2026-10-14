import unittest

import numpy as np

from nudge.math import coefficients, fit, policy_probabilities, reward_scale, vector
from nudge import Expert, Question


class MathTests(unittest.TestCase):
    def test_learned_utility_overrides_baseline_zero_and_always_explores(self):
        row = {'jev': {'a': 1.0, 'b': 0.0}, 'features': {}}
        critics = fit([], ['a', 'b'], [], 1)
        prior = policy_probabilities(critics, [row], [], 1, Expert())[0]
        self.assertTrue(np.all(prior > 0))
        critics['a']['mean'] = [-100.0, 0.0]
        critics['b']['mean'] = [100.0, 0.0]
        policy = policy_probabilities(critics, [row], [], 1, Expert())[0]
        np.testing.assert_allclose(policy, [0.025, 0.975])

    def test_posterior_matches_analytic_intercept(self):
        rows = [
            {"action": "a", "jev": {"a": 0.5}, "features": {}, "reward": 1.0, "weight": 1.0}
            for _ in range(16)
        ]
        critics = fit(rows, ["a", "b"], [], 1)
        self.assertAlmostEqual(critics["a"]["mean"][0], 0.5)
        self.assertEqual(critics["b"]["mean"], [0, 0])
        self.assertEqual(critics["a"]["precision"][0][0], 128)

    def test_posterior_samples_have_correct_covariance(self):
        critic = {"mean": [0.3, -0.2], "precision": [[100, 40], [40, 80]], "n": 20}
        rng = np.random.default_rng(21)
        samples = np.array([coefficients(critic, 2, Expert(), rng) for _ in range(12000)])
        np.testing.assert_allclose(samples.mean(axis=0), critic["mean"], atol=0.008)
        np.testing.assert_allclose(
            np.cov(samples.T), 4 * np.linalg.inv(critic["precision"]), atol=0.003
        )
        critic["n"] = 0
        self.assertFalse(np.array_equal(coefficients(critic, 2, Expert(), rng), critic["mean"]))

    def test_zero_spread_and_ordinal_normalization(self):
        self.assertEqual(reward_scale([]), 1)
        self.assertEqual(reward_scale([0, 0]), 1)
        self.assertEqual(reward_scale([-3, -3]), 3)
        q = Question("level", "How urgent?", "ordinal")
        np.testing.assert_equal(vector(0, {"level": 3}, [q]), [1, -4, 0.5])
        with self.assertRaises(ValueError):
            reward_scale([-1e308, 1e308])

    def test_categorical_question_uses_probability_vector(self):
        q = Question(
            "task_family",
            "What kind of task is this?",
            "categorical",
            categories=("math", "code", "reading"),
        )
        probabilities = {"math": 0.2, "code": 0.7, "reading": 0.1}
        self.assertAlmostEqual(q.normalize(probabilities), 0.45)
        self.assertEqual(q.feature_vector(probabilities), (0.2, 0.7, 0.1))


if __name__ == "__main__":
    unittest.main()
