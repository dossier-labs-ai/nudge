import copy
import unittest

from nudge import Question
from nudge.question_evidence import evidence, replication, update_evidence


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.q = Question('q', 'Condition?')
        self.rows = [dict(id=str(i), decision_order=i + 1, action=a,
                          policy={'a': .5, 'b': .5}, propensity=.5, weight=1,
                          features={'q': group}, reward=float((a == 'b') == bool(group)))
                     for i, (group, a) in enumerate([(g, a) for _ in range(20)
                                                     for g in (0, 1) for a in ('a', 'b')])]

    def test_relative_advantage_and_common_difficulty_are_distinct(self):
        signal = evidence(self.rows, self.q, ['a', 'b'])
        self.assertAlmostEqual(signal['discrimination_rms'], 1)
        common = [{**r, 'reward': r['features']['q'] + (r['action'] == 'b')} for r in self.rows]
        self.assertAlmostEqual(evidence(common, self.q, ['a', 'b'])['discrimination_rms'], 0)
        self.assertEqual(replication(signal, signal)['direction_agreement'], 1)
        reversed_rows = [{**r, 'reward': 1 - r['reward']} for r in self.rows]
        self.assertEqual(replication(signal, evidence(reversed_rows, self.q, ['a', 'b']))
                         ['direction_agreement'], 0)

    def test_unseen_actions_and_invalid_propensity(self):
        rows = [r for r in self.rows if r['action'] == 'a']
        self.assertIsNone(evidence(rows, self.q, ['a', 'b'])['discrimination_rms'])
        rows[0] = {**rows[0], 'propensity': 0}
        with self.assertRaises(ValueError):
            evidence(rows, self.q, ['a', 'b'])

    def test_inverse_propensity_weights_recover_context_mix(self):
        rows = []
        for g in (0, 1):
            for reward, allocation in ((0, 3), (1, 1)):
                for i in range(4):
                    action = 'a' if i < allocation else 'b'
                    policy = {'a': allocation / 4, 'b': 1 - allocation / 4}
                    rows.append({'action': action, 'policy': policy, 'propensity': policy[action],
                                 'weight': 1, 'features': {'q': g}, 'reward': reward})
        result = evidence(rows, self.q, ['a', 'b'])
        for cell in result['cells']:
            self.assertAlmostEqual(cell['mean'], .5)

    def test_corrections_are_not_new_validation_and_objective_change_resets(self):
        first = update_evidence(self.rows[:40], [self.q], ['a', 'b'], {}, {'r': 1})
        corrected = copy.deepcopy(self.rows[:40])
        corrected[0]['reward'] = -20
        again = update_evidence(corrected, [self.q], ['a', 'b'], first, {'r': 1})
        self.assertEqual(next(iter(again.values()))['subsequent']['observation_count'], 0)
        self.assertEqual(next(iter(first.values()))['initial'], next(iter(again.values()))['initial'])
        later = update_evidence(self.rows, [self.q], ['a', 'b'], first, {'r': 1})
        self.assertEqual(next(iter(later.values()))['subsequent']['observation_count'], 40)
        self.assertEqual(next(iter(later.values()))['replication']['direction_agreement'], 1)
        changed = update_evidence(self.rows, [self.q], ['a', 'b'], first, {'r': -1})
        self.assertEqual(next(iter(changed.values()))['subsequent']['observation_count'], 0)
