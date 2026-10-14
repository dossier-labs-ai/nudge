import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from nudge.adapters import InferenceError
from nudge.routerbench_data import fingerprint
from nudge.routerbench_live import Gateway
from nudge.routerbench_factors import metrics
from nudge.routerbench_semantic import (
    RUBRIC, analyze, annotate, encoded, payload, schema, tree_fit, tree_predict, validate,
)


def labels():
    return {name: levels[0] for name, (_, levels) in RUBRIC.items()}


class SemanticTests(unittest.TestCase):
    def test_saved_full_sample_semantic_policy_replays(self):
        root = Path(__file__).resolve().parents[1] / 'benchmark-results/routerbench'
        if not (root / 'semantic-analysis.json').exists():
            self.skipTest('local semantic artifacts not present')
        data = json.loads((root / 'prepared.json').read_text())
        annotations = json.loads((root / 'semantic-annotations.json').read_text())
        report = json.loads((root / 'semantic-analysis.json').read_text())
        if annotations['rubric_fingerprint'] != fingerprint(RUBRIC):
            with self.assertRaisesRegex(ValueError, 'rubric changed'):
                encoded(data, annotations)
            return  # Historical artifacts must never be reinterpreted under a new rubric.
        encoded(data, annotations)
        policy = report['combination_policy']
        routing = {tuple(c['values']): data['models'].index(c['model']) for c in policy['cells']}
        actions = np.array([routing[tuple(a['factors'][k] for k in policy['factors'])]
                            for a in annotations['annotations']])
        rows = data['train'] + data['test']
        measured = metrics(np.array([r['quality'] for r in rows]),
                           np.array([r['cost'] for r in rows]), actions, report['penalty'])
        for key, value in measured.items():
            self.assertAlmostEqual(value, report['combination_trajectory'][-1][key], places=12)
        for tree in report['supported_trees']:
            self.assertEqual(sum(r['prompts'] for r in tree['rules']), len(rows))
            self.assertGreaterEqual(min(r['prompts'] for r in tree['rules']), 20)

    def test_analysis_finds_semantic_model_interaction_end_to_end(self):
        rows = [{'state': str(i), 'quality': [1, 0] if i < 40 else [0, 1], 'cost': [0, 0]}
                for i in range(80)]
        data = {'train': rows[:60], 'test': rows[60:], 'models': ['a', 'b']}
        annotations = {
            'prepared_fingerprint': fingerprint(data), 'rubric_fingerprint': fingerprint(RUBRIC),
            'annotations': [{'prompt_sha256': fingerprint(row['state']), 'factors':
                             labels() | {'reasoning_steps': 'one' if i < 40 else 'three_plus'}}
                            for i, row in enumerate(rows)]}
        result = analyze(data, annotations)
        self.assertEqual(result['individual_factors']['reasoning_steps']['quality'], 1)
        self.assertEqual(result['individual_factors']['task_type']['quality'], 0.5)
        self.assertEqual(result['supported_trees'][0]['factors'], ['reasoning_steps'])
        self.assertEqual(result['supported_trees'][0]['sample_fit']['quality'], 1)
        self.assertEqual(result['supported_trees'][0]['five_fold_out_of_fold']['quality'], 1)
        self.assertEqual(result['combination_trajectory'][0]['factors'], ['reasoning_steps'])

    def test_schema_matches_rubric_and_checked_in_gateway_schema(self):
        path = Path(__file__).resolve().parents[1] / 'gateway/schemas/semantic.json'
        self.assertEqual(json.loads(path.read_text()), schema())
        self.assertEqual(set(schema()['properties']['annotations']['items']['required']), set(RUBRIC))

    def test_annotation_rejects_missing_rows_fields_or_unknown_labels(self):
        for invalid in ({'annotations': []}, {'annotations': [{}]},
                        {'annotations': [labels() | {'task_type': 'mmlu'}]}):
            with self.assertRaises(InferenceError):
                validate(invalid, 1)
        self.assertEqual(validate({'annotations': [labels()]}, 1), [labels()])

    def test_option_difficulty_requires_supplied_options(self):
        for changes in ({'option_difficulty': 'obvious'}, {'has_answer_options': 'yes'}):
            with self.assertRaisesRegex(InferenceError, 'difficulty must apply'):
                validate({'annotations': [labels() | changes]}, 1)
        valid = labels() | {'has_answer_options': 'yes', 'option_difficulty': 'subtle'}
        self.assertEqual(validate({'annotations': [valid]}, 1), [valid])

    def test_payload_is_text_only_and_rejects_record_objects(self):
        self.assertEqual(payload(['hello']), {'prompts': ['hello']})
        with self.assertRaises(ValueError):
            payload([{'state': 'hello', 'quality': [1]}])

    def test_tree_selects_utility_and_enforces_minimum_support(self):
        x = np.array([[0]] * 20 + [[1]] * 20 + [[2]])
        rewards = np.array([[1, 0]] * 20 + [[0, 1]] * 20 + [[0, 1]])
        tree = tree_fit(x, rewards, min_leaf=20, max_leaves=16)
        self.assertEqual(len(tree), 3)
        self.assertTrue(all(n['count'] >= 20 for n in tree if 'feature' not in n))
        np.testing.assert_array_equal(tree_predict(tree, x), [0] * 20 + [1] * 21)
        self.assertEqual(len(tree_fit(x, rewards, max_leaves=1)), 1)

    def test_shared_winner_does_not_require_splitting_oracle_ties(self):
        tree = tree_fit(np.array([[0], [1]]), np.array([[1, 1, 0], [0, 1, 1]]), min_leaf=1)
        self.assertEqual(len(tree), 1)
        self.assertEqual(tree[0]['action'], 1)

    def test_encoded_checks_rubric_and_prompt_alignment(self):
        data = {'train': [{'state': 'a'}], 'test': []}
        annotations = {'prepared_fingerprint': fingerprint(data), 'rubric_fingerprint': fingerprint(RUBRIC),
                       'annotations': [{'prompt_sha256': fingerprint('a'), 'factors': labels()}]}
        self.assertEqual(encoded(data, annotations).shape, (1, len(RUBRIC)))
        annotations['annotations'][0]['prompt_sha256'] = fingerprint('different')
        with self.assertRaises(ValueError):
            encoded(data, annotations)

    def test_real_gateway_transport_sends_no_outcomes_and_resumes_cache(self):
        captured = []
        class Fixture(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                sent = json.loads(body['input']['messages'][0]['content'])
                captured.append(sent)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({'output': {'parsed': {
                    'annotations': [] if len(captured) == 1 else
                    [labels() for _ in sent['prompts']]}}}).encode())

        server = ThreadingHTTPServer(('127.0.0.1', 0), Fixture)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            data = {'train': [{'state': 'Solve this.', 'quality': [1], 'cost': [0], 'id': 'secret-id'}],
                    'test': [{'state': 'Another question.', 'quality': [0]}]}
            with tempfile.TemporaryDirectory() as directory:
                gateway = Gateway(f'http://127.0.0.1:{server.server_port}', {}, directory, 'semantic-test')
                result = annotate(data, gateway, batch_size=2, concurrency=1)
                cached = annotate(data, gateway, batch_size=2, concurrency=1)
                self.assertEqual(result, cached)
                self.assertEqual(encoded(data, result).shape, (2, len(RUBRIC)))
                self.assertEqual(len(result['annotation_validation_retries']), 1)
            self.assertEqual(captured, [{'prompts': ['Solve this.', 'Another question.']}] * 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)


if __name__ == '__main__':
    unittest.main()
