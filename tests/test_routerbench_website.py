import json
import tempfile
import unittest
from pathlib import Path

from nudge.routerbench_report import online_curve
from nudge.routerbench_website import cumulative_scores, render


class WebsiteTests(unittest.TestCase):
    def snapshot(self):
        rows = [{'partition': 'train', 'nudge': {'utility': x}, 'rune': {'utility': .5}}
                for x in (.1, .8, .3, .9, .7, .2, .6)]
        return {'status': {'window': 3, 'penalty_per_usd': 10, 'updated_at': '2026-10-15'},
                'curve': online_curve(rows, window=3)}

    def test_prefix_recovery_matches_original_scores_without_future_data(self):
        snapshot = self.snapshot()
        scores = cumulative_scores(snapshot, 5)
        self.assertAlmostEqual(scores[-1]['nudge'], 2.8 / 5)
        self.assertAlmostEqual(scores[-1]['rune'], .5)
        snapshot['curve'][5]['nudge'] = 999
        self.assertEqual(scores, cumulative_scores(snapshot, 5))
        with self.assertRaises(ValueError):
            cumulative_scores(snapshot, 10)

    def test_assets_include_readable_labels_and_exact_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            render(self.snapshot(), output, 5)
            svg = (output / 'comparison.svg').read_text()
            for label in ('Prompts processed', 'Average answer quality', 'adjusted for cost',
                          'Higher is better', 'Nudge + Rune', 'Rune alone'):
                self.assertIn(label, svg)
            self.assertEqual(json.loads((output / 'summary.json').read_text())['prompts'], 5)
            self.assertEqual(len(json.loads((output / 'source.json').read_text())['curve']), 5)
            self.assertTrue((output / 'comparison-mobile.png').exists())
