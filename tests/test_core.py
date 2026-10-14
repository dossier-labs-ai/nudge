import json
import sqlite3
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path

from nudge import Answer, Choice, ConcurrentLearnError, Expert, Nudge, Question
from nudge.benchmark import SyntheticAnswerer, synthetic_choice
from nudge.math import policy_probabilities


class ConstantAnswerer:
    identity = "test-constant-v1"

    def __init__(self):
        self.calls = 0

    def ask(self, state, questions):
        self.calls += 1
        return Answer({"a": 0.6, "b": 0.4}, {q.name: float(state) for q in questions})


def choice():
    return Choice("test", "A test decision", {"a": "A", "b": "B"}, {"result": "Outcome"})


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "nudge.db"
        self.model = ConstantAnswerer()
        self.policy = Nudge(choice(), store=self.path, model=self.model, seed=4)
        self.addCleanup(self.policy.close)

    def test_cold_start_record_and_restart(self):
        d = self.policy.ask(state="verbatim\n☃")
        self.assertTrue(all(p > 0 for p in d.policy.values()))
        self.assertAlmostEqual(sum(d.policy.values()), 1)
        self.assertEqual(d.propensity, d.policy[d.action])
        self.assertEqual(self.model.calls, 1)
        with Nudge(choice(), store=self.path, model=ConstantAnswerer()) as reopened:
            self.assertEqual(reopened.decision(d.id), d)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT state FROM decisions").fetchone()[0], "verbatim\n☃")
        self.assertFalse(self.policy.learn().activated)

    def test_replacement_feedback_and_refit(self):
        d = self.policy.ask(state="0")
        self.policy.feedback(d.id, result=1)
        first = self.policy.learn()
        self.policy.feedback(d.id, result=-2)
        second = self.policy.learn()
        self.assertEqual((first.version, second.version), (1, 2))
        self.assertEqual(second.feedback_count, 1)
        report = self.policy.report()
        self.assertEqual(report.actions[d.action]["mean_reward"], -2)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM feedback").fetchone()[0], 2)
            old = json.loads(
                db.execute("SELECT payload FROM snapshots WHERE version=1").fetchone()[0]
            )
            self.assertGreater(old["critics"][d.action]["mean"][0], 0)
        self.assertEqual(self.policy.decision(d.id), d)

    def test_validation_and_choice_isolation(self):
        d = self.policy.ask(state="0")
        for fields in ({}, {"result": float("nan")}, {"result": True}, {"other": 1}):
            with self.assertRaises(ValueError):
                self.policy.feedback(d.id, **fields)
        with self.assertRaises(KeyError):
            self.policy.feedback("missing", result=1)
        with Nudge(
            replace(choice(), name="other"), store=self.path, model=ConstantAnswerer()
        ) as other:
            with self.assertRaises(KeyError):
                other.feedback(d.id, result=1)
        with self.assertRaises(ValueError):
            Nudge(
                replace(choice(), actions={"a": "Changed", "b": "B"}),
                store=self.path,
                model=ConstantAnswerer(),
            )
        self.assertEqual(self.policy.report().feedback_count, 0)

    def test_serving_matches_evaluation_after_serialization(self):
        for _ in range(40):
            decision = self.policy.ask(state='0')
            self.policy.feedback(decision.id, result=float(decision.action == 'b'))
        self.policy.learn()
        live = self.policy._store.live(choice().name)
        decision = self.policy.ask(state='0')
        expected = policy_probabilities(live['critics'],
                                       [{'jev': decision.jev, 'features': decision.features}],
                                       [], live['scale'], self.policy.expert)[0]
        self.assertEqual(decision.policy, dict(zip(live['critics'], expected)))

    def test_old_bounded_contract_requires_new_choice_or_store(self):
        with sqlite3.connect(self.path) as db:
            contract = json.loads(db.execute('SELECT contract FROM choices').fetchone()[0])
            contract.pop('policy')
            contract['expert']['max_drift'] = 4
            db.execute('UPDATE choices SET contract=?', (json.dumps(contract),))
        with self.assertRaisesRegex(ValueError, 'new choice name or store'):
            Nudge(choice(), store=self.path, model=ConstantAnswerer())

    def test_reward_changes_and_missing_feedback(self):
        d = self.policy.ask(state="0")
        self.policy.ask(state="0")
        self.policy.feedback(d.id, result=1)
        self.policy.learn(reward={"result": -1})
        report = self.policy.report()
        self.assertEqual(report.feedback_count, 1)
        self.assertEqual(report.decisions, 2)
        self.assertEqual(report.actions[d.action]["mean_reward"], -1)
        with self.assertRaises(ValueError):
            Nudge(choice(), store=self.path, model=ConstantAnswerer())
        with Nudge(
            replace(choice(), reward={"result": -1}), store=self.path, model=ConstantAnswerer()
        ) as reopened:
            self.assertEqual(reopened.report().version, 1)

    def test_no_model_calls_during_report_or_cached_refit(self):
        d = self.policy.ask(state="0")
        self.policy.feedback(d.id, result=1)
        self.policy.learn()
        calls = self.model.calls
        self.policy.report()
        self.policy.learn()
        self.assertEqual(self.model.calls, calls)

    def test_failure_leaves_live_policy_unchanged(self):
        d = self.policy.ask(state="0")
        self.policy.feedback(d.id, result=1e100)
        with self.assertRaises(ValueError):
            self.policy.learn()
        self.assertEqual(self.policy.report().version, 0)
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT status FROM learning_runs").fetchone()[0], "failed")

    def test_new_questions_cache_and_reopen(self):
        path = Path(self.temp.name) / "features.db"
        with Nudge(
            choice(), store=path, model=self.model, questions=[Question("number", "Is it one?")]
        ) as policy:
            d = policy.ask(state="1")
            policy.feedback(d.id, result=1)
            calls = self.model.calls
            policy.learn()
            self.assertEqual(calls, self.model.calls)
        with Nudge(choice(), store=path, model=self.model) as reopened:
            self.assertEqual(reopened.ask(state="0").features, {"number": 0})

    def test_invalid_answer_is_never_logged(self):
        class Bad(ConstantAnswerer):
            def ask(self, state, questions):
                return Answer({"a": 0.1, "b": 0.2})

        with Nudge(choice(), store=Path(self.temp.name) / "bad.db", model=Bad()) as policy:
            with self.assertRaises(ValueError):
                policy.ask(state="x")
            self.assertEqual(policy.report().decisions, 0)

    def test_ordinal_definitions_roundtrip_and_weighted_reward(self):
        definition = Choice(
            "cost",
            "Choose",
            {"a": "A", "b": "B"},
            {"resolved": "Resolution", "refunded": "Refund"},
            {"resolved": 1, "refunded": -0.5},
        )
        q = Question("level", "Urgency?", "ordinal", levels=("a", "b", "c", "d", "e"))
        path = Path(self.temp.name) / "ordinal.db"
        with Nudge(definition, store=path, model=self.model, questions=[q]) as policy:
            d = policy.ask(state="3")
            policy.feedback(d.id, resolved=1, refunded=1)
            policy.learn()
            self.assertEqual(policy.report().actions[d.action]["mean_reward"], 0.5)
        with Nudge(definition, store=path, model=self.model, questions=[q]) as reopened:
            self.assertEqual(reopened.report().questions, (q,))

    def test_concurrent_ask_feedback_and_stale_learner(self):
        entered = threading.Event()
        resume = threading.Event()

        class WaitingProposer:
            identity = "waiting"

            def propose(self, context):
                entered.set()
                if not resume.wait(5):
                    raise RuntimeError("test timeout")
                return []

        path = Path(self.temp.name) / "concurrent.db"
        expert = Expert(min_split=1)
        with Nudge(
            synthetic_choice(),
            store=path,
            model=SyntheticAnswerer("context"),
            proposer=WaitingProposer(),
            expert=expert,
            seed=1,
        ) as first:
            for i in range(40):
                d = first.ask(state=json.dumps({"repeat": i % 2, "urgent": False}))
                first.feedback(d.id, resolved=1)
            errors = []

            def learn():
                try:
                    first.learn(discover_questions=True)
                except BaseException as error:
                    errors.append(error)

            thread = threading.Thread(target=learn)
            thread.start()
            try:
                self.assertTrue(entered.wait(5))
                # The same instance can ask while its learner is inside a model call.
                extra = first.ask(state='{"repeat": true, "urgent": false}')
                first.feedback(extra.id, resolved=0)
                with Nudge(
                    synthetic_choice(),
                    store=path,
                    model=SyntheticAnswerer("context"),
                    expert=expert,
                ) as second:
                    self.assertEqual(second.learn(discover_questions=False).feedback_count, 41)
            finally:
                resume.set()
                thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], ConcurrentLearnError)
            self.assertEqual(first.report().version, 1)


if __name__ == "__main__":
    unittest.main()
