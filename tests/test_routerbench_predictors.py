import unittest

import numpy as np

from nudge.routerbench_predictors import polynomial, predict


class PredictorTests(unittest.TestCase):
    def test_polynomial_adds_symmetric_quadratic_terms(self):
        values = np.array([[1.0, 2.0, 3.0], [1.0, -1.0, 4.0]])
        expanded = polynomial(values)
        self.assertEqual(expanded.shape, (2, 6))
        np.testing.assert_allclose(expanded[0], [1, 2, 3, 4, 6, 9])

    def test_shared_rbf_returns_finite_predictions_for_each_action(self):
        features = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 0.1], [1.0, 0.9]])
        actions = np.array([0, 1, 0, 1])
        rewards = np.array([0.1, 0.9, 0.2, 0.8])
        prediction = predict(
            features,
            actions,
            rewards,
            np.array([[1.0, 0.05], [1.0, 0.95]]),
            np.array([0, 1]),
            family="shared_rbf",
            alpha=1.0,
            bandwidth=0.5,
        )
        self.assertEqual(prediction.shape, (2,))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertLess(prediction[0], prediction[1])


if __name__ == "__main__":
    unittest.main()
