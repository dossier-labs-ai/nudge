import json
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from nudge import Answer, Expert, Nudge, Proposal, Question
from nudge.benchmark import QUESTIONS, SyntheticAnswerer, SyntheticProposer, synthetic_choice
from nudge.discovery import CallBudget, edited, hydrate


class DiscoveryTests(unittest.TestCase):
    def test_parallel_hydration_is_bounded_batched_cached_and_ordered(self):
        class Concurrent(SyntheticAnswerer):
            concurrency = 4
            def __init__(self):
                super().__init__('context')
                self.lock = threading.Lock()
                self.active = self.peak = 0
                self.head_counts = []

            def ask(self, state, questions):
                with self.lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                    self.head_counts.append(len(questions))
                try:
                    time.sleep(0.01)
                    return super().ask(state, questions)
                finally:
                    with self.lock:
                        self.active -= 1

        policy, path = self.run_policy()
        with sqlite3.connect(path) as db:
            rows = [dict(id=i, state=s) for i, s in db.execute(
                'SELECT id,state FROM decisions ORDER BY rowid LIMIT 13')]
        model = Concurrent()
        budget = CallBudget(13)
        kwargs = dict(store=policy._store, answerer=model, choice=synthetic_choice(), budget=budget)
        answers = hydrate(rows, QUESTIONS[:2], **kwargs)
        self.assertEqual([r['id'] for r in answers], [r['id'] for r in rows])
        self.assertGreater(model.peak, 1)
        self.assertLessEqual(model.peak, 4)
        self.assertEqual(model.head_counts, [2] * 13)
        self.assertEqual(budget.used, 13)
        for row in answers:
            self.assertEqual(row['features'], SyntheticAnswerer('context').ask(
                row['state'], QUESTIONS[:2]).features)
        self.assertEqual(hydrate(rows, QUESTIONS[:2], **kwargs), answers)
        self.assertEqual(len(model.head_counts), 13)

    def test_parallel_discovery_preserves_bank_policy_and_budget(self):
        class Concurrent(SyntheticAnswerer):
            concurrency = 4
        serial, _ = self.run_policy()
        parallel, _ = self.run_policy(model=Concurrent('context'))
        a = serial.learn(half_life_seconds=None, discover_questions=True)
        b = parallel.learn(half_life_seconds=None, discover_questions=True)
        self.assertEqual(a.questions, b.questions)
        self.assertEqual(a.model_calls, b.model_calls)
        self.assertEqual([e['status'] for e in a.discovery], [e['status'] for e in b.discovery])
        for state in ('{"repeat": true, "urgent": false}', '{"repeat": false, "urgent": true}'):
            self.assertEqual(serial.ask(state=state).policy, parallel.ask(state=state).policy)
        limited, path = self.run_policy(model=Concurrent('context'), expert=Expert(max_calls=3))
        result = limited.learn(half_life_seconds=None, discover_questions=True)
        self.assertEqual(result.model_calls, 3)
        self.assertEqual(result.questions, ())
        with sqlite3.connect(path) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM feature_answers').fetchone()[0], 2)

    def run_policy(self, *, expert=None, model=None, proposer=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "nudge.db"
        counter = iter(range(10000))
        policy = Nudge(
            synthetic_choice(),
            store=path,
            model=model or SyntheticAnswerer("context"),
            proposer=proposer or SyntheticProposer(),
            expert=expert or Expert(max_calls=3000),
            seed=19,
            id_factory=lambda: f"test-{next(counter)}",
        )
        self.addCleanup(policy.close)
        for i in range(600):
            repeat = bool(i % 2)
            d = policy.ask(state=json.dumps({"ticket": i, "repeat": repeat, "urgent": bool(i % 3)}))
            policy.feedback(d.id, resolved=float((d.action == "escalate") == repeat))
        return policy, path

    def test_discovers_feature_and_proposer_can_use_all_observed_prompts(self):
        contexts = []

        class Capture(SyntheticProposer):
            def propose(self, context):
                contexts.append(context)
                return super().propose(context)

        policy, path = self.run_policy(proposer=Capture())
        result = policy.learn(discover_questions=True)
        self.assertIn("repeat", [q.name for q in result.questions])
        self.assertNotIn("flat", [q.name for q in result.questions])
        self.assertTrue(any(e.get("status") == "flat" for e in result.discovery))
        with sqlite3.connect(path) as db:
            train = db.execute("SELECT state,payload FROM decisions").fetchall()
            allowed = {(state, json.loads(payload)["action"]) for state, payload in train}
            for context in contexts:
                self.assertEqual(len(context.examples), 256)
                for example in context.examples:
                    self.assertIn((example["state"], example["action"]), allowed)
            records = db.execute("SELECT request,response,results FROM proposals").fetchall()
            self.assertTrue(records)
            self.assertTrue(all(json.loads(row[2]) for row in records))
        repeat = policy.ask(state='{"repeat": true, "urgent": false}')
        simple = policy.ask(state='{"repeat": false, "urgent": false}')
        self.assertGreater(repeat.policy["escalate"], simple.policy["escalate"])

    def test_budget_exhaustion_refits_without_partial_question(self):
        policy, _ = self.run_policy(expert=Expert(max_calls=2))
        result = policy.learn(discover_questions=True)
        self.assertEqual(result.model_calls, 2)
        self.assertEqual(result.questions, ())
        self.assertTrue(result.activated)
        self.assertEqual(result.discovery[-1]["status"], "budget_exhausted")

    def test_questions_are_retained_even_when_policy_cannot_change(self):
        policy, _ = self.run_policy(expert=Expert(max_calls=3000, explore_floor=1))
        result = policy.learn(half_life_seconds=None, discover_questions=True)
        self.assertIn('repeat', [q.name for q in result.questions])
        kept = [e for e in result.discovery if e['status'] == 'kept']
        self.assertTrue(kept)
        self.assertTrue(all('utility_gain' not in e for e in kept))

    def test_rejection_feedback_drives_next_round_and_current_question_removal_diagnostic(self):
        contexts = []

        class Refine:
            identity = 'refine-from-feedback'

            def propose(self, context):
                contexts.append(context)
                if context.round == 1:
                    return [Proposal('add', Question('flat', 'Flat?', paraphrase='Flat indeed?'))]
                if context.round == 2:
                    return [Proposal('add', QUESTIONS[0])]
                return []

        policy, _ = self.run_policy(proposer=Refine())
        result = policy.learn(half_life_seconds=None, discover_questions=True)
        self.assertEqual(len(contexts), 3)
        self.assertEqual(contexts[1].history[-1]['results'][0]['status'], 'flat')
        self.assertEqual(contexts[1].history[-1]['results'][0]['retained_question_names'], [])
        self.assertIn('repeat', [q.name for q in result.questions])
        kept = contexts[2].history[-1]['results'][0]
        self.assertEqual(kept['retention_basis'], 'proposer_hypothesis_and_answer_checks')
        self.assertEqual(kept['retained_question_names'], ['repeat'])
        diagnostic = contexts[2].question_diagnostics[0]
        self.assertEqual(diagnostic['question']['name'], 'repeat')
        self.assertGreater(diagnostic['current']['discrimination_rms'], 0.5)
        self.assertEqual(diagnostic['subsequent']['observation_count'], 0)
        self.assertEqual(contexts[2].to_dict()['selection_objective'],
                         'question_action_discrimination_v1')

    def test_replay_history_corrections_and_cache_survive_reopen(self):
        contexts = []

        class Capture(SyntheticProposer):
            def propose(self, context):
                contexts.append(context)
                return super().propose(context)

        expert = Expert(max_calls=3000, proposer_examples=30)
        policy, path = self.run_policy(expert=expert, proposer=Capture())
        policy.learn(half_life_seconds=None, discover_questions=True)
        self.assertTrue(any(context.history for context in contexts[1:]))
        with sqlite3.connect(path) as db:
            latest = db.execute(
                "SELECT id FROM decisions WHERE split='train' ORDER BY rowid DESC LIMIT 1"
            ).fetchone()[0]
            cached = db.execute('SELECT COUNT(*) FROM feature_answers').fetchone()[0]
            saved = json.loads(db.execute(
                'SELECT request FROM proposals ORDER BY sequence LIMIT 1'
            ).fetchone()[0])
        self.assertEqual(len(saved['examples']), 30)
        self.assertEqual(len({e['id'] for e in saved['examples']}), 30)
        self.assertEqual({e['replay_source'] for e in saved['examples']},
                         {'recent', 'representative', 'informative'})
        # Real stored decisions contain serving scores/distributions. Those must
        # not swamp the proposer context, while observed feedback stays exact.
        for example in saved['examples']:
            self.assertFalse(set(example) & {'contributions', 'scores', 'why', 'jev', 'policy'})
            self.assertTrue({'state', 'action', 'fields', 'reward', 'propensity',
                             'features', 'version', 'replay_source'} <= set(example))
            observed = policy.decision(example['id'])
            self.assertEqual(example['propensity'], observed.propensity)
            self.assertEqual(example['action'], observed.action)
        contexts.clear()
        with Nudge(synthetic_choice(), store=path, model=SyntheticAnswerer('context'),
                   proposer=Capture(), expert=expert) as reopened:
            reopened.feedback(latest, resolved=5.0)
            reopened.learn(half_life_seconds=None, discover_questions=True)
            self.assertEqual(reopened.report().learned_feedback_count, 600)
            self.assertEqual(sum(a['learned_weight'] for a in reopened.report().actions.values()),
                             600)
            first = contexts[0]
            self.assertTrue(first.history)
            self.assertTrue(all(h['selection_objective'] ==
                                'question_action_discrimination_v1'
                                for h in first.history))
            self.assertTrue(all(h['run_status'] == 'activated' for h in first.history))
            self.assertTrue(all('examples' not in h for h in first.history))
            example = next(e for e in first.examples if e['id'] == latest)
            self.assertEqual(example['reward'], 5.0)
            self.assertEqual(example['fields'], {'resolved': 5.0})
            self.assertGreater(example['propensity'], 0)
            self.assertEqual(example['version'], 0)
            changed = {**synthetic_choice().to_dict(), 'reward': {'resolved': -1.0}}
            self.assertEqual(reopened._store.proposal_history(changed), [])
        with sqlite3.connect(path) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM feature_answers').fetchone()[0],
                             cached)

    def test_unstable_paraphrase_is_rejected(self):
        class Unstable(SyntheticAnswerer):
            def ask(self, state, questions):
                answer = super().ask(state, questions)
                features = dict(answer.features)
                for question in questions:
                    if question.prompt == QUESTIONS[0].paraphrase:
                        features[question.name] = 1 - features[question.name]
                return Answer(answer.probabilities, features)

        class One:
            identity = "one"

            def propose(self, context):
                return [Proposal("add", QUESTIONS[0])]

        policy, _ = self.run_policy(model=Unstable("context"), proposer=One())
        result = policy.learn(discover_questions=True)
        self.assertEqual(result.questions, ())
        self.assertEqual(result.discovery[0]["status"], "unstable")

    def test_historical_mse_objectives_are_not_relabelled_as_utility(self):
        policy, path = self.run_policy()
        policy.learn(half_life_seconds=None, discover_questions=True)
        with sqlite3.connect(path) as db:
            sequence, request = db.execute(
                'SELECT sequence,request FROM proposals ORDER BY sequence LIMIT 1'
            ).fetchone()
            request = json.loads(request)
            request['selection_objective'] = 'chosen-action time-weighted held-out reward MSE'
            db.execute('UPDATE proposals SET request=? WHERE sequence=?',
                       (json.dumps(request), sequence))
        history = policy._store.proposal_history(synthetic_choice().to_dict())
        self.assertEqual(history[0]['selection_objective'],
                         'chosen-action time-weighted held-out reward MSE')
        self.assertEqual(history[-1]['selection_objective'],
                         'question_action_discrimination_v1')

    def test_revise_drop_contracts(self):
        questions = []
        for i in range(32):
            q = Question(f"feature_{i}", f"Feature {i}?")
            questions = edited(questions, Proposal("add", q), Expert().max_questions)
        self.assertEqual(len(questions), 32)
        with self.assertRaises(ValueError):
            edited(
                questions,
                Proposal("add", Question("overflow", "Overflow?")),
                Expert().max_questions,
            )
        old = Question("old", "Old?", paraphrase="Old indeed?")
        new = Question("new", "New?", paraphrase="New indeed?")
        self.assertEqual(edited([old], Proposal("revise", new, "old"), 3), [new])
        self.assertEqual(edited([new], Proposal("drop", target="new"), 3), [])
        with self.assertRaises(ValueError):
            edited([old], Proposal("add", old), 3)

    def test_bundle_is_measured_and_retained_atomically(self):
        class Bundle:
            identity = 'bundle-test'

            def propose(self, context):
                return [] if context.questions else [Proposal('add_bundle', questions=tuple(QUESTIONS[:2]))]

        policy, _ = self.run_policy(proposer=Bundle())
        result = policy.learn(half_life_seconds=None, discover_questions=True)
        self.assertEqual({q.name for q in result.questions}, {q.name for q in QUESTIONS[:2]})
        kept = next(e for e in result.discovery if e['status'] == 'kept')
        self.assertEqual(len(kept['answer_checks']), 2)
        self.assertTrue(kept['question_evidence'])
        initial = [QUESTIONS[0]]
        with self.assertRaises(ValueError):
            edited(initial, Proposal('add_bundle', questions=(QUESTIONS[1], QUESTIONS[0])), 32)
        self.assertEqual(initial, [QUESTIONS[0]])

    def test_pending_survives_history_limit_and_resolves_after_remeasurement(self):
        policy, path = self.run_policy()
        # Persist actual proposal records; pending must not depend on the history window.
        choice = synthetic_choice().to_dict()
        run = 'pending-test'
        policy._store.start_run(run, choice['name'], 0, 0)
        proposal = Proposal('add', QUESTIONS[0]).to_dict()
        request = {'choice': choice, 'selection_objective': 'unrestricted_posterior_decision_utility_v1'}
        record = policy._store.proposal(run, 1, 'test', request, [proposal], [])
        policy._store.finish_proposal(record, [{'proposal': proposal, 'status': 'deferred_uncertainty'}])
        for i in range(20):
            policy._store.proposal(run, i + 2, 'test', request, [], [])
        with Nudge(synthetic_choice(), store=path, model=SyntheticAnswerer('context'),
                   expert=policy.expert) as reopened:
            self.assertEqual(len(reopened._store.pending_proposals(choice)), 1)
            other = {**choice, 'reward': {'resolved': -1}}
            self.assertEqual(reopened._store.pending_proposals(other), [])
            record = reopened._store.proposal(run, 22, 'test', request, [proposal], [])
            reopened._store.finish_proposal(record, [{'proposal': proposal, 'status': 'kept'}])
            self.assertEqual(reopened._store.pending_proposals(choice), [])

    def test_discovery_only_preserves_policy_and_later_evidence_survives_restart(self):
        policy, path = self.run_policy()
        late = policy.ask(state='{"repeat": true, "urgent": false}')
        before = policy._store.live(synthetic_choice().name)
        with patch('nudge.core.fit', side_effect=AssertionError('discovery must not train routing')):
            result = policy.discover_questions(half_life_seconds=None)
        self.assertFalse(result.activated)
        after = policy._store.live(synthetic_choice().name)
        for key in ('critics', 'questions', 'scale', 'feedback_cutoff', 'feedback_count'):
            self.assertEqual(before[key], after[key])
        self.assertTrue(policy.report().question_bank)
        self.assertEqual(policy.report().questions, ())
        policy.feedback(late.id, resolved=1)
        policy.discover_questions(half_life_seconds=None)
        self.assertTrue(all(d['subsequent']['observation_count'] == 0
                            for d in policy.report().question_evidence))
        extra = policy.ask(state='{"repeat": false, "urgent": true}')
        policy.feedback(extra.id, resolved=0)
        with Nudge(synthetic_choice(), store=path, model=SyntheticAnswerer('context'),
                   expert=policy.expert) as reopened:
            reopened.discover_questions(half_life_seconds=None)
            self.assertTrue(all(d['subsequent']['observation_count'] == 1
                                for d in reopened.report().question_evidence))
            reopened.proposer = policy.proposer
            with patch.object(reopened.proposer, 'propose', side_effect=AssertionError('fit must not discover')):
                reopened.learn(half_life_seconds=None)
            self.assertEqual(reopened.report().questions, reopened.report().question_bank)
            self.assertEqual(reopened.report().learned_feedback_count, 602)


if __name__ == "__main__":
    unittest.main()
