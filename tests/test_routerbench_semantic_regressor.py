import copy
import unittest

from nudge.routerbench_data import MODELS, fingerprint
from nudge.routerbench_semantic_regressor import compare, prepare
from nudge.routerbench_retention import measure
from nudge.routerbench_question_bank import measure as measure_bank
from nudge.historical import bounded_expert


class SemanticRegressorTests(unittest.TestCase):
    def setUp(self):
        rows = [dict(id=str(i), state=f'prompt {i}', family='synthetic',
                     quality=[float((i + j) % 3 == 0) for j in range(len(MODELS))],
                     cost=[0.001] * len(MODELS)) for i in range(26)]
        self.data = dict(models=list(MODELS), train=rows[:22], test=rows[22:])
        self.rubric = {'task': ['What task?', ['a', 'b']]}
        self.measurements = dict(
            prepared_fingerprint=fingerprint(self.data),
            rubric_fingerprint=fingerprint(self.rubric), heads_per_call=1,
            records=[dict(prompt_sha256=fingerprint(r['state']),
                          probabilities={'task': {'a': i % 2, 'b': 1 - i % 2}})
                     for i, r in enumerate(rows)])
        self.decisions = [dict(id=r['id'], partition='train' if i < 22 else 'test',
                               action=MODELS[i % len(MODELS)],
                               rune_probabilities={m: 1 / len(MODELS) for m in MODELS})
                          for i, r in enumerate(rows)]

    def run_comparison(self, data, measurements):
        return compare(data, measurements, self.rubric, self.decisions, bounded_expert())

    def test_test_rewards_cannot_change_predictions_or_scale(self):
        before = self.run_comparison(self.data, self.measurements)
        changed = copy.deepcopy(self.data)
        for row in changed['test']:
            row['quality'] = [0.0] * len(MODELS)
            row['cost'] = [0.2] * len(MODELS)
        measurements = {**self.measurements, 'prepared_fingerprint': fingerprint(changed)}
        after = self.run_comparison(changed, measurements)
        self.assertEqual(before['test_predictions'], after['test_predictions'])
        self.assertEqual(before['scale_from_training_chosen_feedback'],
                         after['scale_from_training_chosen_feedback'])
        self.assertEqual(before['training_prompts'], 22)
        self.assertEqual(before['test_prompts'], 4)
        self.assertEqual(len(before['results']), 4)
        for key, result in before['results'].items():
            self.assertEqual(result['count'], 4)
            self.assertNotEqual(result['nudge']['utility'],
                                after['results'][key]['nudge']['utility'])

    def test_rejects_wrong_alignment_rubric_and_partition(self):
        for kind in ('alignment', 'rubric', 'partition', 'questions'):
            with self.subTest(kind=kind):
                measurements = copy.deepcopy(self.measurements)
                decisions = copy.deepcopy(self.decisions)
                if kind == 'alignment':
                    measurements['records'].reverse()
                elif kind == 'rubric':
                    measurements['rubric_fingerprint'] = 'wrong'
                elif kind == 'partition':
                    decisions[0]['partition'] = 'test'
                else:
                    measurements['records'][0]['probabilities'] = {}
                with self.assertRaises(ValueError):
                    prepare(self.data, measurements, self.rubric, decisions, 10)

    def test_retention_never_uses_final_test_outcomes(self):
        decisions = [{**d, 'nudge_probabilities': d['rune_probabilities'],
                      'propensity': 1 / len(MODELS)} for d in self.decisions]
        before = measure(self.data, self.measurements, self.rubric, decisions)
        changed = copy.deepcopy(self.data)
        for row in changed['test']:
            row['quality'] = [1000.0] * len(MODELS)
        measurements = {**self.measurements, 'prepared_fingerprint': fingerprint(changed)}
        after = measure(changed, measurements, self.rubric, decisions)
        for name, original in before['results'].items():
            updated = after['results'][name]
            self.assertEqual(original['retained_questions'], updated['retained_questions'])
            self.assertEqual(original['selection'], updated['selection'])
            self.assertEqual(original['test_policies'], updated['test_policies'])
            self.assertNotEqual(original['test'], updated['test'])

    def test_question_bank_evidence_ignores_final_test_rewards(self):
        decisions = [{**d, 'nudge_probabilities': d['rune_probabilities'],
                      'propensity': 1 / len(MODELS)} for d in self.decisions]
        before = measure_bank(self.data, self.measurements, self.rubric, decisions)
        changed = copy.deepcopy(self.data)
        for row in changed['test']:
            row['quality'] = [1000.0] * len(MODELS)
        measurements = {**self.measurements, 'prepared_fingerprint': fingerprint(changed)}
        after = measure_bank(changed, measurements, self.rubric, decisions)
        for name, original in before['results'].items():
            updated = after['results'][name]
            for key in ('retained_questions', 'all_800_evidence',
                        'retrospective_first_half_next_half', 'test_policies'):
                self.assertEqual(original[key], updated[key])
            self.assertNotEqual(original['test'], updated['test'])
