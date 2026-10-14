import json
import unittest
from pathlib import Path

import numpy as np

from nudge.routerbench_factors import audit, extract, fit, metrics, prune, task_text


class FactorTests(unittest.TestCase):
    def test_saved_sample_policy_replays_from_prompt_factors(self):
        root = Path(__file__).resolve().parents[1] / 'benchmark-results/routerbench'
        result_path = root / 'oracle-factors-lexical.json'
        if not result_path.exists():
            self.skipTest('local full-sample audit artifact not present')
        data = json.loads((root / 'prepared.json').read_text())
        result = json.loads(result_path.read_text())
        rows = data['train'] + data['test']
        names, values = extract([r['state'] for r in rows], lexical=True)
        columns = [names.index(name) for name in result['selected_factors']]
        routing = {tuple(cell['factor_bits']): data['models'].index(cell['model'])
                   for cell in result['cells']}
        actions = np.array([routing[tuple(bits)] for bits in values[:, columns]])
        measured = metrics(np.array([r['quality'] for r in rows]),
                           np.array([r['cost'] for r in rows]), actions, result['penalty'])
        for key, value in measured.items():
            self.assertAlmostEqual(value, result['oracle'][key], places=12)

    def test_prune_removes_redundant_but_preserves_interaction(self):
        values = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=bool)
        rewards = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
        self.assertEqual(prune(values, [0, 1, 2], rewards), [0, 1])

    def test_interaction_can_require_zero_gain_first_step(self):
        values = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=bool)
        rewards = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
        selected, history, _ = fit(values, rewards)
        self.assertEqual(len(selected), 2)
        self.assertEqual(history[-1]["mean_utility"], 1.0)

    def test_shared_successes_do_not_require_matching_oracle_label(self):
        values = np.array([[0], [1]], dtype=bool)
        rewards = np.array([[1, 1, 0], [0, 1, 1]])
        selected, history, _ = fit(values, rewards)
        self.assertEqual(selected, [])
        np.testing.assert_array_equal(history[-1]["actions"], [1, 1])

    def test_identical_features_cannot_resolve_conflicting_outcomes(self):
        values = np.zeros((2, 1), dtype=bool)
        rewards = np.eye(2)
        selected, history, _ = fit(values, rewards)
        self.assertEqual(selected, [])
        self.assertEqual(history[-1]["mean_utility"], 0.5)

    def test_math_examples_excluded_from_topic_indicators(self):
        state = ('TASK: provide a numerical answer.\nQuestion: price dollars\n'
                 '\n\n\nWhat is the area of the triangle?')
        self.assertEqual(task_text(state), 'What is the area of the triangle?')
        names, values = extract([state, 'How much does this cost?'])
        index = names.index('money or prices')
        np.testing.assert_array_equal(values[:, index], [False, True])

    def test_cost_sensitive_selection_and_replayable_cells(self):
        data = {"models": ["cheap", "expensive"], "train": [
            {"state": "alpha", "quality": [1, 1], "cost": [0.01, 0.02]},
            {"state": "beta", "quality": [0, 1], "cost": [0.01, 0.02]},
        ], "test": []}
        result = audit(data, lexical=True)
        self.assertEqual(result['unresolved_prompts'], 0)
        self.assertAlmostEqual(result['steps'][-1]['quality'], 1)
        self.assertAlmostEqual(result['steps'][-1]['total_cost_usd'], 0.03)
        self.assertEqual({c['model'] for c in result['cells']}, {'cheap', 'expensive'})
        self.assertEqual(sum(c['prompts'] for c in result['cells']), 2)
        changed_metadata = {**data, 'train': [
            {**r, 'id': str(i), 'family': 'unavailable'} for i, r in enumerate(data['train'])]}
        self.assertEqual(audit(changed_metadata, lexical=True)['selected_factors'],
                         result['selected_factors'])


if __name__ == '__main__':
    unittest.main()
