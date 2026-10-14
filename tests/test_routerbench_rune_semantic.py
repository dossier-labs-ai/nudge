import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from nudge.adapters import InferenceError
from nudge.routerbench_live import Gateway
from nudge.routerbench_rune_semantic import agreement, head_levels, heads, measure, parse, summarize
from nudge.routerbench_semantic import RUBRIC, analyze
from nudge.routerbench_data import fingerprint
from nudge.rune_gateway import native_request


class RuneSemanticTests(unittest.TestCase):
    def test_report_alignment_agreement_and_frozen_policy_transfer(self):
        rows = [{'state': str(i), 'quality': [1, 0], 'cost': [0, 0]} for i in range(32)]
        data = {'train': rows, 'test': [], 'models': ['a', 'b']}
        reference = {'prepared_fingerprint': fingerprint(data), 'rubric_fingerprint': fingerprint(RUBRIC),
                     'annotations': [{'prompt_sha256': fingerprint(r['state']), 'factors': {
                         k: levels[0] for k, (_, levels) in RUBRIC.items()}} for r in rows]}
        measurements = {'prepared_fingerprint': fingerprint(data), 'rubric_fingerprint': fingerprint(RUBRIC),
                        'heads_per_call': 16, 'records': [
                            {'prompt_sha256': fingerprint(r['state']), 'probabilities': {
                                k: {level: float(i == 0) for i, level in enumerate(levels)}
                                for k, (_, levels) in RUBRIC.items()}} for r in rows]}
        measurements['records'][0]['probabilities']['subject']['everyday_life'] = 0
        measurements['records'][0]['probabilities']['subject']['mathematics'] = 1
        report = summarize(data, reference, measurements, analyze(data, reference))
        self.assertEqual(report['all_labels_match_prompts'], 31)
        self.assertEqual(report['mean_factor_agreement'], 1 - 1 / (32 * len(RUBRIC)))
        self.assertEqual(report['factors']['subject']['agreement'], 31 / 32)
        self.assertEqual(report['frozen_policy_transfer'][0]['rune_labels_frozen_policy']['quality'], 1)
        stale = analyze(data, reference) | {'rubric': {'option_distinction': ['old combined question']}}
        with self.assertRaisesRegex(ValueError, 'policy rubric mismatch'):
            summarize(data, reference, measurements, stale)
        measurements['records'][0]['prompt_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            summarize(data, reference, measurements, analyze(data, reference))

    def test_native_heads_use_semantic_definitions_and_complete_categories(self):
        questions = heads(list(RUBRIC))
        native_request({'messages': [{'role': 'user', 'content': json.dumps({
            'state': 'Question', 'questions': questions})}]})
        for name, head in questions.items():
            self.assertIn(RUBRIC[name][0], head['instructions'])
            self.assertEqual(len(head['criteria']), len(head_levels(name)))
        self.assertNotIn('not applicable', questions['option_difficulty']['criteria'].values())

    def test_metrics_expose_majority_dominance(self):
        score = agreement([[1, 0]] * 4, [0, 0, 0, 1])
        self.assertEqual(score['agreement'], 0.75)
        self.assertEqual(score['majority_reference_baseline'], 0.75)
        self.assertEqual(score['macro_recall_present_classes'], 0.5)
        self.assertEqual(score['confusion_rows_reference_columns_rune'], [[3, 0], [1, 0]])

    def test_parse_requires_every_head_and_valid_probability_distribution(self):
        with self.assertRaises(InferenceError):
            parse({'answers': {}}, ['constraint_load'])
        raw = {'answers': {'constraint_load': {'type': 'choice', 'probabilities': {
            '0': 0.1, '1': 0.2, '2': 0.7}}}}
        self.assertEqual(parse(raw, ['constraint_load'])['constraint_load'],
                         {'one': 0.1, 'two': 0.2, 'three_plus': 0.7})
        raw['answers']['constraint_load']['probabilities']['2'] = float('nan')
        with self.assertRaises(InferenceError):
            parse(raw, ['constraint_load'])

    def test_batched_and_single_requests_preserve_all_heads_and_cache(self):
        captured = []
        class Fixture(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                sent = json.loads(body['input']['messages'][0]['content'])
                captured.append(sent)
                answers = {name: {'type': 'choice', 'probabilities': {
                    key: float(key == ('1' if name == 'has_answer_options' and
                                       sent['state'] == 'With options' else '0'))
                    for key in head['criteria']}}
                           for name, head in sent['questions'].items()}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({'output': {'parsed': {'answers': answers}}}).encode())

        server = ThreadingHTTPServer(('127.0.0.1', 0), Fixture)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            data = {'train': [{'state': 'Question', 'quality': [1], 'id': 'private'},
                              {'state': 'With options', 'quality': [1]}], 'test': []}
            with tempfile.TemporaryDirectory() as directory:
                gateway = Gateway(f'http://127.0.0.1:{server.server_port}', {}, directory, 'rune-test')
                batch = measure(data, gateway, concurrency=1)
                single = measure(data, gateway, heads_per_call=1, concurrency=4)
                cached = measure(data, gateway, concurrency=1)
                self.assertEqual(batch['records'], single['records'])
                self.assertEqual(batch, cached)
                self.assertEqual(len(captured), 35)
                self.assertTrue(all(set(p) == {'state', 'questions'} for p in captured))
                difficulty_calls = [p for p in captured if 'option_difficulty' in p['questions']]
                self.assertEqual(len(difficulty_calls), 1)
                self.assertEqual(difficulty_calls[0]['state'], 'With options')
                self.assertEqual(batch['records'][0]['probabilities']['option_difficulty']['not_applicable'], 1)
                self.assertEqual(batch['records'][1]['probabilities']['option_difficulty']['not_applicable'], 0)
                for p in batch['records'][0]['probabilities'].values():
                    self.assertAlmostEqual(np.sum(list(p.values())), 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)


if __name__ == '__main__':
    unittest.main()
