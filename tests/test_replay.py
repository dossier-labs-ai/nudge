import unittest

from nudge import Question
from nudge.replay import SamplingReplayBuffer


class ReplayTests(unittest.TestCase):
    def rows(self, n=300):
        return [dict(id=str(i), split='train', weight=1.0, action='a',
                     features={'kind': {'common': 1.0, 'rare': 0.0}}, residual=0.0)
                for i in range(n)]

    def test_recent_historical_distinct_and_reproducible(self):
        rows = self.rows()
        buffer = SamplingReplayBuffer(rows, questions=[])
        sample = buffer.sample(60)
        self.assertEqual(sample, buffer.sample(60))
        self.assertEqual(len({r['id'] for r in sample}), 60)
        self.assertEqual([r['id'] for r in sample[:20]], [str(i) for i in range(20)])
        self.assertEqual(sum(r['replay_source'] == 'representative' for r in sample), 20)
        self.assertEqual(sum(r['replay_source'] == 'informative' for r in sample), 20)
        self.assertTrue(any(int(r['id']) > 100 for r in sample))
        self.assertNotEqual(sample, SamplingReplayBuffer(
            rows, questions=[], round_number=2).sample(60))
        self.assertTrue(all('replay_source' not in r for r in rows))

    def test_selection_and_expired_rows_never_enter_replay(self):
        rows = self.rows(12)
        rows[0]['split'] = 'selection'
        rows[1]['weight'] = 0
        rows[2]['split'] = 'test'
        sample = SamplingReplayBuffer(rows, questions=[]).sample(100)
        self.assertEqual(len(sample), 9)
        self.assertFalse({'0', '1', '2'} & {r['id'] for r in sample})
        self.assertTrue(all(r['weight'] == 1 for r in sample))

    def test_informative_sampling_enriches_rare_actions_and_answer_groups(self):
        for rare_dimension in ('action', 'answer'):
            with self.subTest(dimension=rare_dimension):
                rows = self.rows()
                question = Question('kind', 'Kind?', 'categorical', categories=('common', 'rare'))
                for row in rows[-10:]:
                    if rare_dimension == 'action':
                        row['action'] = 'b'
                    else:
                        row['features']['kind'] = {'common': 0.0, 'rare': 1.0}
                counts = {'representative': 0, 'informative': 0}
                for round_number in range(30):
                    sample = SamplingReplayBuffer(
                        rows, questions=[question], round_number=round_number).sample(60)
                    for row in sample:
                        if int(row['id']) >= 290:
                            counts[row['replay_source']] += 1
                self.assertGreater(counts['informative'], counts['representative'] * 2)

    def test_empty_small_and_duplicate_buffers(self):
        self.assertEqual(SamplingReplayBuffer([], questions=[]).sample(5), [])
        rows = self.rows(2)
        sample = SamplingReplayBuffer(rows, questions=[]).sample(100)
        self.assertEqual([r['id'] for r in sample], ['0', '1'])
        with self.assertRaises(ValueError):
            SamplingReplayBuffer(rows + rows, questions=[])
        for limit in (0, -1, True):
            with self.assertRaises(ValueError):
                SamplingReplayBuffer(rows, questions=[]).sample(limit)
