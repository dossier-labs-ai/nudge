import csv
import json
import tempfile
import unittest
from pathlib import Path

from nudge.routerbench_exports import export_utility


class ExportTests(unittest.TestCase):
    def test_live_utility_exports_are_numeric_and_include_learning_state(self):
        rows = [{'partition': 'train', 'version': 0,
                 'nudge': {'utility': value}, 'rune': {'utility': 0.5}}
                for value in (0.2, 0.8, 0.9)]
        with tempfile.TemporaryDirectory() as directory:
            export_utility(rows, directory, stage='learning', penalty=10)
            root = Path(directory)
            status = json.loads((root / 'status.json').read_text())
            curve = list(csv.DictReader((root / 'utility.csv').read_text().splitlines()))
            self.assertEqual(status['stage'], 'learning')
            self.assertEqual(status['prompts'], 3)
            self.assertAlmostEqual(status['mean_utility']['nudge'], 1.9 / 3)
            self.assertAlmostEqual(float(curve[-1]['cumulative_gain']), 0.4)
            self.assertEqual((root / 'utility.png').read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            export_utility(rows, directory, stage='failed', penalty=10, error='TimeoutError')
            self.assertEqual(json.loads((root / 'status.json').read_text())['error'], 'TimeoutError')
            self.assertFalse(list(root.glob('*.pending*')))

    def test_empty_stream_exports_status_without_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            status = export_utility([], directory, stage='starting', penalty=10)
            self.assertEqual(status['mean_utility'], {})
            self.assertIsNone(status['latest'])
