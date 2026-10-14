import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nudge import Answer, Choice, Expert, Nudge, Proposal, Question
from nudge.store import Store
from nudge.training_replay import priority, select


class RecordingAnswerer:
    identity = 'bounded-replay-test'

    def __init__(self):
        self.feature_states = []

    def ask(self, state, questions):
        if questions:
            self.feature_states.append(state)
        return Answer({'a': 0.5, 'b': 0.5}, {q.name: float(int(state) % 2) for q in questions})


class TrainingReplayTests(unittest.TestCase):
    def test_bounded_learning_restarts_corrections_and_indexed_reads(self):
        choice = Choice('bounded', 'Choose', {'a': 'A', 'b': 'B'}, {'result': 'Outcome'})
        expert = Expert(max_training_rows=24, max_rounds=1, min_split=1, stability_samples=4)
        model = RecordingAnswerer()
        contexts = []

        class Proposer:
            identity = 'bounded-proposer'

            def propose(self, context):
                contexts.append(context)
                return [] if context.questions else [Proposal(
                    'add', Question('odd', 'Odd?', paraphrase='Is it odd?'))]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.db'
            now = datetime(2026, 1, 1, tzinfo=timezone.utc)
            with Nudge(choice, store=path, model=model, proposer=Proposer(), expert=expert,
                       seed=42, clock=lambda: now, id_factory=lambda: str(counter)) as learner:
                for counter in range(240):
                    now += timedelta(hours=1)
                    decision = learner.ask(state=str(counter))
                    learner.feedback(decision.id, result=counter % 3)
                db = learner._store.connection
                before_meta, before_rows = learner._store.training_rows(choice.name)
                identifiers = {r['id'] for r in before_rows}
                self.assertEqual(len(identifiers), 24)
                self.assertTrue(set(map(str, range(232, 240))) <= identifiers)
                self.assertTrue(any(int(i) < 120 for i in identifiers))
                self.assertLessEqual(db.execute('SELECT COUNT(*) FROM training_replay').fetchone()[0], 72)
                # Uniform reservoir is the global bottom-k, not just recent history.
                uniform = {r[0] for r in db.execute(
                    "SELECT decision_id FROM training_replay WHERE lane='uniform'")}
                self.assertEqual(uniform, set(sorted(map(str, range(240)),
                                                    key=lambda i: priority(i, 'uniform'))[:24]))
                trace = []
                db.set_trace_callback(trace.append)
                learned = learner.learn(discover_questions=True, half_life_seconds=None)
                db.set_trace_callback(None)
                self.assertEqual(learned.feedback_count, 240)
                self.assertEqual(learned.training_sample_count, 24)
                self.assertTrue(set(model.feature_states) <= identifiers)
                self.assertLessEqual(len(model.feature_states), 28)
                self.assertTrue(all(len(c.examples) <= 24 for c in contexts))
                self.assertFalse(any('LEFT JOIN feedback' in sql for sql in trace))
                self.assertFalse(any('ORDER BY d.rowid DESC' in sql for sql in trace))
                live = learner._store.live(choice.name)
                self.assertEqual(sum(c['n'] for c in live['critics'].values()), 24)
                self.assertEqual(set(live['training_replay']['decision_ids']), identifiers)
                self.assertEqual(live['question_evidence'][next(iter(live['question_evidence']))]
                                 ['decision_boundary'], 240)
                # Corrections neither count as new observations nor refresh decision age.
                target = before_rows[0]
                learner.feedback(target['id'], result=99)
                meta, corrected = learner._store.training_rows(choice.name)
                self.assertEqual(meta['feedback_count'], before_meta['feedback_count'])
                retained = next(r for r in corrected if r['id'] == target['id'])
                self.assertEqual(retained['fields'], {'result': 99})
                self.assertEqual(retained['created_at'], target['created_at'])
                self.assertEqual(meta['decision_boundary'], before_meta['decision_boundary'])
                saved = learner._store.training_rows(choice.name)
                audit = learner.decision('0')
            with Nudge(choice, store=path, model=model, expert=expert) as reopened:
                self.assertEqual(reopened._store.training_rows(choice.name), saved)
                self.assertEqual(reopened.decision('0'), audit)
                count = len(model.feature_states)
                reopened.learn(half_life_seconds=None)
                self.assertLessEqual(len(model.feature_states) - count, 24)
                self.assertEqual(reopened.report().feedback_count, 240)
                self.assertEqual(reopened.report().training_sample_count, 24)

    def test_training_query_work_does_not_scale_with_audit_history(self):
        choice = Choice('bounded', 'Choose', {'a': 'A', 'b': 'B'}, {'result': 'Outcome'})
        with tempfile.TemporaryDirectory() as directory:
            with Nudge(choice, store=Path(directory) / 'p.db', model=RecordingAnswerer(),
                       expert=Expert(max_training_rows=12), id_factory=lambda: str(i)) as learner:
                work = []
                for i in range(1000):
                    d = learner.ask(state=str(i))
                    learner.feedback(d.id, result=1)
                    if i in (99, 999):
                        steps = [0]

                        def progress():
                            steps[0] += 1
                            return 0

                        db = learner._store.connection
                        db.set_progress_handler(progress, 1)
                        meta, rows = learner._store.training_rows(choice.name)
                        db.set_progress_handler(None, 0)
                        self.assertEqual(len(rows), 12)
                        self.assertEqual(meta['feedback_count'], i + 1)
                        work.append(steps[0])
                self.assertLessEqual(work[1], work[0] * 1.1)

    def test_informative_pool_balances_actions_and_prefers_surprise(self):
        candidates = ([{'id': f'r{i}', 'lane': 'recent', 'priority': i} for i in range(9)] +
                      [{'id': f'u{i}', 'lane': 'uniform', 'priority': i} for i in range(9)] +
                      [{'id': f'a{i}', 'lane': 'action:a', 'priority': i} for i in range(9)] +
                      [{'id': 'rare', 'lane': 'action:b', 'priority': 100}])
        sample = select(candidates, 9)
        self.assertEqual(len(sample), 9)
        self.assertIn('rare', {r['id'] for r in sample})
        self.assertLess(priority('same', 'informative', 11), priority('same', 'informative', 1))

    def test_pending_history_is_bounded_and_objective_scoped(self):
        choice = Choice('bounded', 'Choose', {'a': 'A', 'b': 'B'}, {'result': 'Outcome'})
        with tempfile.TemporaryDirectory() as directory:
            with Nudge(choice, store=Path(directory) / 'p.db', model=RecordingAnswerer()) as learner:
                store = learner._store
                store.start_run('test', choice.name, 0, 0)
                request = {'choice': choice.to_dict()}
                for i in range(100):
                    proposal = Proposal('add', Question(f'q{i}', f'Question {i}?')).to_dict()
                    seq = store.proposal('test', 1, 'test', request, [proposal], [])
                    store.finish_proposal(seq, [{'proposal': proposal, 'status': 'deferred_uncertainty'}])
                pending = store.pending_proposals(choice.to_dict())
                self.assertEqual(len(pending), 64)
                self.assertEqual({r['proposal']['question']['name'] for r in pending},
                                 {f'q{i}' for i in range(36, 100)})
                self.assertEqual(len(store.proposal_history(choice.to_dict())), 16)
                other = {**choice.to_dict(), 'reward': {'result': -1}}
                self.assertEqual(store.pending_proposals(other), [])
                seq = store.proposal('test', 1, 'test', request, [proposal], [])
                store.finish_proposal(seq, [{'proposal': proposal, 'status': 'kept'}])
                self.assertEqual(len(store.pending_proposals(choice.to_dict())), 63)

    def test_legacy_store_is_rejected_without_mutation_and_capacity_validated(self):
        for invalid in (0, 1, 2, True, 1.5):
            with self.assertRaises(ValueError):
                Expert(max_training_rows=invalid)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.db'
            with sqlite3.connect(path) as db:
                db.execute('PRAGMA user_version=1')
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'fresh store'):
                Store(path)
            self.assertEqual(path.read_bytes(), before)
            with sqlite3.connect(path) as db:
                self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 1)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                                 .fetchone()[0], 0)
