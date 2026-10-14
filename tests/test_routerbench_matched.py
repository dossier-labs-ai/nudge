import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from nudge import Answer, Expert, Proposal, Question
from nudge.routerbench import replay
from nudge.routerbench_data import MODELS, fingerprint
from nudge.routerbench_matched import compare_sets, matched


class MeasuredProvider:
    identity = 'external-fixture'

    def ask(self, state, questions):
        return Answer({m: 1 / len(MODELS) for m in MODELS},
                      {q.name: int(state) % 2 for q in questions})


class ProposerProvider:
    identity = 'external-proposer-fixture'

    def __init__(self):
        self.seen_states = []

    def propose(self, context):
        return [] if context.questions else [Proposal('add', Question(
            'odd', 'Is the number odd?', paraphrase='Is this an odd integer?'))]


class MatchedTests(unittest.TestCase):
    def test_discriminating_feature_changes_routing_and_test_rewards_cannot_change_fit(self):
        questions = [Question('odd', 'Is the number odd?')]
        rows = [{'state': str(i), 'jev': {m: 1 / len(MODELS) for m in MODELS},
                 'weight': 1.0, 'action': MODELS[(i // 2) % 2],
                 'reward': float(i % 2 == (i // 2) % 2)} for i in range(400)]
        test = [({'state': str(400 + i), 'jev': rows[0]['jev']},
                 {'family': 'synthetic', 'quality': [float(j == i % 2) for j in range(11)],
                  'cost': [0.] * 11}) for i in range(20)]
        measured = {str(i): {'odd': i % 2} for i in range(420)}
        sets = {'no_questions': ([], {}), 'generated': (questions, measured)}
        before = compare_sets(rows, test, sets, Expert())
        self.assertGreater(before['paired_against_no_questions']['generated/greedy']
                           ['utility']['mean'], 0.4)
        changed = copy.deepcopy(test)
        for _, record in changed:
            record['quality'] = [100.] * 11
        after = compare_sets(rows, changed, sets, Expert())
        for key in ('critic_fingerprints', 'test_scores', 'test_policies', 'scale',
                    'training_feedback_fingerprint'):
            self.assertEqual(before[key], after[key])
        self.assertNotEqual(before['metrics'], after['metrics'])
        del measured['400']
        with self.assertRaisesRegex(ValueError, 'misaligned'):
            compare_sets(rows, test, sets, Expert())

    def test_completed_live_store_aligns_and_rejects_another_feedback_trajectory(self):
        rows = [dict(id=str(i), state=str(i), family='synthetic',
                     quality=[float(j % 2 == i % 2) for j in range(11)], cost=[0.] * 11)
                for i in range(60)]
        data = dict(models=list(MODELS), train=rows[:48], test=rows[48:])
        rubric = {'reference': ['Is the number odd?', ['no', 'yes']]}
        measurements = dict(prepared_fingerprint=fingerprint(data),
                            rubric_fingerprint=fingerprint(rubric), records=[
            dict(prompt_sha256=fingerprint(r['state']), probabilities={
                'reference': {'no': 1 - i % 2, 'yes': i % 2}})
            for i, r in enumerate(rows)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                replay(data, MeasuredProvider(), ProposerProvider(), path,
                       expert=Expert(max_rounds=1))
            decisions = [json.loads(line) for line in
                         (path / 'decisions.jsonl').read_text().splitlines()]
            result = matched(data, measurements, rubric, decisions,
                             database=path / 'policy.db', all_training_outcomes=True)
            self.assertEqual(len(result['questions']['generated']), 1)
            self.assertEqual(len(result['metrics']), 6)
            self.assertAlmostEqual(result['generated_minus_reference']['greedy']['utility']['mean'],
                                   result['metrics']['generated/greedy']['nudge']['utility'] -
                                   result['metrics']['reference/greedy']['nudge']['utility'])
            self.assertEqual(result['training_count'], 48)
            full = result['all_training_outcomes_diagnostic']
            self.assertEqual(full['training_count'], 48 * len(MODELS))
            self.assertEqual(full['training_prompt_count'], 48)
            self.assertEqual(full['scale'], result['scale'])
            changed = copy.deepcopy(data)
            for row in changed['test']:
                row['quality'] = [100.] * len(MODELS)
            changed_measurements = {**measurements, 'prepared_fingerprint': fingerprint(changed)}
            updated = matched(changed, changed_measurements, rubric, decisions,
                              database=path / 'policy.db', all_training_outcomes=True)
            self.assertEqual(full['test_policies'],
                             updated['all_training_outcomes_diagnostic']['test_policies'])
            decisions[0]['action'] = MODELS[(MODELS.index(decisions[0]['action']) + 1) % 11]
            with self.assertRaisesRegex(ValueError, 'feedback or baseline mismatch'):
                matched(data, measurements, rubric, decisions, database=path / 'policy.db')
