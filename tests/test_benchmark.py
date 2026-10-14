import unittest

from nudge.historical import bounded_expert
from nudge.benchmark import expected_reward, run_one, tilt_oracle


class BenchmarkTests(unittest.TestCase):
    def test_oracle_closed_form_two_actions(self):
        baseline = {"a": 0.75, "b": 0.25}
        rewards = {"a": 0.1, "b": 0.9}
        result = tilt_oracle(baseline, rewards, bounded_expert())
        expected_b = 0.95 * (0.25 * 4 / (0.75 + 0.25 * 4)) + 0.05 * 0.25
        self.assertAlmostEqual(result["b"], expected_b)
        self.assertGreater(expected_reward(result, rewards), expected_reward(baseline, rewards))

    def test_no_signal_has_zero_true_regret(self):
        run = run_one(
            scenario="no_signal",
            method="discovery",
            seed=0,
            steps=200,
            learn_every=100,
            evaluation_steps=32,
        )
        self.assertAlmostEqual(run["cumulative_pseudo_regret"], 0, places=10)
        self.assertAlmostEqual(run["test_gain"], 0, places=10)

    def test_calibration_improves_and_is_reproducible(self):
        args = dict(
            scenario="calibration",
            method="no_features",
            seed=4,
            steps=300,
            learn_every=100,
            evaluation_steps=64,
        )
        first, second = run_one(**args), run_one(**args)
        self.assertEqual(first, second)
        self.assertGreater(first["test_gain"], 0.02)
        self.assertGreaterEqual(first["cumulative_pseudo_regret"], -1e-9)

    def test_decay_adapts_after_reward_reversal(self):
        args = dict(
            scenario="nonstationary", method="discovery", seed=0, steps=1000, learn_every=200
        )
        uniform = run_one(**args, half_life_seconds=None)
        decayed = run_one(**args, half_life_seconds=7 * 86400)
        self.assertLess(decayed["second_half_regret"], uniform["second_half_regret"])
        self.assertGreater(decayed["test_gain"], uniform["test_gain"] + 0.02)
        self.assertLess(decayed["learned_weight"], uniform["learned_weight"])


if __name__ == "__main__":
    unittest.main()
