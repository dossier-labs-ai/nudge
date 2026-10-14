import contextlib
import io
import json
import sqlite3
import tempfile
import threading
import tomllib
import unittest
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from nudge import Answer, Expert, Question
from nudge.adapters import InferenceError, JsonHTTP
from nudge.routerbench import replay, routing_choice, reference_baselines
from nudge.routerbench_data import FAMILIES, MODELS, aggregate, evaluate, load_pinned, prepare, fingerprint
from nudge.routerbench_live import Gateway, LiveProposer, RuneAnswerer, choice_probabilities
from nudge.types import ProposalContext
from nudge.routerbench_report import overhead, supervised_zero_router, compare_runs, online_curve
from nudge.routerbench_diagnose import diagnose
from nudge.rune_gateway import NativeProvider, handler


def fixture_records():
    rows = []
    for family in FAMILIES:
        for i in range(12):
            row = {
                "sample_id": f"{family}-{i}",
                "eval_name": family if family != "mmlu" else "mmlu-math",
                "prompt": repr([f"Question {family} {i}"]),
            }
            for j, model in enumerate(MODELS):
                row[model] = float((i + j) % 2)
                row[model + "|total_cost"] = (j + 1) / 10000
            rows.append(row)
    return rows


class ExternalAnswerer:
    identity = "fixture-provider"

    def __init__(self):
        self.states = []

    def ask(self, state, questions):
        self.states.append(state)
        return Answer({model: 1 / len(MODELS) for model in MODELS}, {})


class RouterBenchTests(unittest.TestCase):
    def test_resume_recovers_interruption_before_feedback_without_resampling(self):
        data = prepare(fixture_records(), count=36, holdout=0)
        with tempfile.TemporaryDirectory() as directory:
            plain, recovered = Path(directory) / 'plain', Path(directory) / 'recovered'
            replay(data, ExternalAnswerer(), None, plain, learn_every=12)
            class InterruptedFeedback(ExternalAnswerer):
                def ask(self, state, questions):
                    answer = super().ask(state, questions)
                    if len(self.states) == 7:
                        # Real SQLite write failure at the feedback boundary,
                        # after ask has committed its serving decision.
                        with sqlite3.connect(recovered / 'policy.db') as db:
                            db.execute("CREATE TRIGGER interrupted BEFORE INSERT ON feedback "
                                       "BEGIN SELECT RAISE(ABORT, 'interrupted feedback'); END")
                    return answer
            with self.assertRaisesRegex(sqlite3.IntegrityError, 'interrupted feedback'):
                replay(data, InterruptedFeedback(), None, recovered, learn_every=12)
            with sqlite3.connect(recovered / 'policy.db') as db:
                db.execute('DROP TRIGGER interrupted')
            ledger = recovered / 'decisions.jsonl'
            original = ledger.read_text()
            corrupted = [json.loads(line) for line in original.splitlines()]
            corrupted[0]['partition'] = 'test'
            ledger.write_text('\n'.join(map(json.dumps, corrupted)) + '\n')
            with self.assertRaisesRegex(ValueError, 'partition or policy mismatch'):
                replay(data, ExternalAnswerer(), None, recovered, learn_every=12, resume=True)
            with sqlite3.connect(recovered / 'policy.db') as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM feedback').fetchone()[0], 6)
            ledger.write_text(original)
            result = replay(data, ExternalAnswerer(), None, recovered, learn_every=12, resume=True)
            self.assertEqual(result['resumed_decisions'], 7)
            self.assertEqual((plain / 'decisions.jsonl').read_text(), ledger.read_text())
            event = json.loads((recovered / 'resumes.jsonl').read_text())
            self.assertTrue(event['recovered_export'])
            self.assertTrue(event['recovered_feedback'])

    def test_resume_recovers_one_committed_decision_after_export_interruption(self):
        data = prepare(fixture_records(), count=36, holdout=0)
        with tempfile.TemporaryDirectory() as directory:
            plain, recovered = Path(directory) / 'plain', Path(directory) / 'recovered'
            replay(data, ExternalAnswerer(), None, plain, learn_every=12)
            opened = Path.open
            writes = 0
            def interrupt(path, *args, **kwargs):
                nonlocal writes
                if path.name == 'decisions.jsonl' and args == ('a',):
                    writes += 1
                    if writes == 7:
                        raise OSError('interrupted export')
                return opened(path, *args, **kwargs)
            with patch.object(Path, 'open', interrupt), self.assertRaisesRegex(OSError, 'interrupted export'):
                replay(data, ExternalAnswerer(), None, recovered, learn_every=12)
            model = ExternalAnswerer()
            result = replay(data, model, None, recovered, learn_every=12, resume=True)
            self.assertEqual(result['resumed_decisions'], 7)
            self.assertEqual(len(model.states), 29)
            self.assertEqual((plain / 'decisions.jsonl').read_text(),
                             (recovered / 'decisions.jsonl').read_text())
            event = json.loads((recovered / 'resumes.jsonl').read_text())
            self.assertTrue(event['recovered_export'])
            self.assertFalse(event['recovered_feedback'])

    def test_online_stream_scores_before_feedback_and_keeps_learning(self):
        data = prepare(fixture_records(), count=36, holdout=0)
        self.assertEqual(data['test'], [])
        with tempfile.TemporaryDirectory() as directory:
            result = replay(data, ExternalAnswerer(), None, directory, learn_every=12)
            records = [json.loads(line) for line in
                       (Path(directory) / 'decisions.jsonl').read_text().splitlines()]
        self.assertEqual(result['evaluation'], 'online')
        self.assertEqual(result['test'], {})
        self.assertEqual(result['fixed_model_test'], {})
        self.assertEqual(result['report']['feedback_count'], 36)
        self.assertEqual([f['after'] for f in result['learning']], [12, 24, 36])
        self.assertEqual([r['version'] for r in records], [0] * 12 + [1] * 12 + [2] * 12)
        curve = online_curve(records, window=12)
        self.assertAlmostEqual(curve[-1]['rune'], sum(r['rune']['utility'] for r in records[-12:]) / 12)
        self.assertAlmostEqual(curve[-1]['cumulative_gain'],
                               sum(r['nudge']['utility'] - r['rune']['utility'] for r in records))
        self.assertEqual(online_curve(records[:20], window=12), curve[:20])

    def test_proportional_sampling_preserves_small_families_without_oversampling(self):
        rows = fixture_records()
        rows = [r for r in rows if r['eval_name'] != FAMILIES[0]] + rows[:2]
        data = prepare(rows, count=31, holdout=12)
        counts = Counter(r['family'] for r in data['train'] + data['test'])
        self.assertEqual(counts, {FAMILIES[0]: 1, **{f: 6 for f in FAMILIES[1:]}})
        self.assertEqual(len(data['test']), 12)
        self.assertEqual(len({r['state'] for r in data['train'] + data['test']}), 31)
        self.assertEqual(data, prepare(list(reversed(rows)), count=31, holdout=12))
        self.assertEqual(data['sampling'], 'proportional_without_replacement')
        with self.assertRaisesRegex(ValueError, 'eligible population'):
            prepare(rows, count=63, holdout=12)

    def test_resume_failed_learning_preserves_feedback_rng_and_future_decisions(self):
        class InterruptedProposer:
            identity = 'external-interrupted-proposer'

            def __init__(self):
                self.fail = True

            def propose(self, context):
                if self.fail:
                    self.fail = False
                    raise InferenceError('external interruption')
                return []

        data = prepare(fixture_records(), count=60, holdout=12)
        with tempfile.TemporaryDirectory() as directory:
            plain, recovered = Path(directory) / 'plain', Path(directory) / 'recovered'
            expert = Expert(max_rounds=1)
            with contextlib.redirect_stdout(io.StringIO()):
                expected = replay(data, ExternalAnswerer(), None, plain,
                                  expert=expert, learn_every=12)
                proposer = InterruptedProposer()
                with self.assertRaises(InferenceError):
                    replay(data, ExternalAnswerer(), proposer, recovered,
                           expert=expert, learn_every=12)
                actual = replay(data, ExternalAnswerer(), proposer, recovered,
                                expert=expert, learn_every=12, resume=True)
            self.assertEqual(expected['test'], actual['test'])
            self.assertEqual((plain / 'decisions.jsonl').read_text(),
                             (recovered / 'decisions.jsonl').read_text())
            self.assertGreater(actual['resumed_decisions'], 0)
            records = [json.loads(line) for line in
                       (recovered / 'decisions.jsonl').read_text().splitlines()]
            records[0]['partition'] = 'test'
            (recovered / 'decisions.jsonl').write_text('\n'.join(map(json.dumps, records)))
            with self.assertRaisesRegex(ValueError, 'partition or policy mismatch'):
                replay(data, ExternalAnswerer(), None, recovered, expert=expert,
                       learn_every=12, resume=True)

    def test_live_proposer_retries_missing_json_with_audited_bounded_requests(self):
        requests, always_invalid = [], [False]

        class Fixture(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                parsed = None if always_invalid[0] or len(requests) < 3 else {'proposals': []}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({'output': {'parsed': parsed},
                                            'usage': {'output_tokens': 16384}}).encode())

        server = ThreadingHTTPServer(('127.0.0.1', 0), Fixture)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                gateway = Gateway(f'http://127.0.0.1:{server.server_port}', {}, directory, 'retry')
                choice = routing_choice(10)
                proposer = LiveProposer(gateway)
                context = ProposalContext(choice.to_dict(), (), (), 1)
                self.assertEqual(proposer.propose(context), [])
                self.assertEqual(len(requests), 3)
                self.assertTrue(all(request == requests[0] for request in requests))
                self.assertEqual(proposer.propose(context), [])
                self.assertEqual(len(requests), 3)
                always_invalid[0] = True
                with self.assertRaisesRegex(InferenceError, 'parsed JSON'):
                    proposer.propose(ProposalContext(choice.to_dict(), (), (), 2))
                self.assertEqual(len(requests), 6)
                audit = [json.loads(line) for line in
                         (Path(directory) / 'inferences.jsonl').read_text().splitlines()]
                self.assertEqual(sum('error' in r for r in audit), 5)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_paired_run_comparison_checks_baseline_and_reproduces_zero_change(self):
        data = prepare(fixture_records(), count=60, holdout=12)
        with tempfile.TemporaryDirectory() as directory:
            before, after = Path(directory) / 'before', Path(directory) / 'after'
            for path in (before, after):
                with contextlib.redirect_stdout(io.StringIO()):
                    replay(data, ExternalAnswerer(), None, path, expert=Expert(max_rounds=0))
                (path / 'manifest.json').write_text(json.dumps(
                    dict(data_sha256=fingerprint(data), max_proposals=4)))
            result = compare_runs(before, after)
            self.assertEqual(result['test_count'], 12)
            self.assertEqual(result['current_minus_previous']['utility']['mean'], 0)
            records = [json.loads(line) for line in
                       (after / 'decisions.jsonl').read_text().splitlines()]
            records[-1]['rune_probabilities'][MODELS[0]] += 1e-16
            (after / 'decisions.jsonl').write_text('\n'.join(map(json.dumps, records)) + '\n')
            self.assertLess(compare_runs(before, after)['maximum_baseline_probability_difference'],
                            1e-12)
            records[-1]['rune_probabilities'][MODELS[0]] = 0.99
            (after / 'decisions.jsonl').write_text('\n'.join(map(json.dumps, records)) + '\n')
            with self.assertRaisesRegex(ValueError, 'baseline changed'):
                compare_runs(before, after)

    def test_reference_baselines_require_exact_sample_and_partition(self):
        data = prepare(fixture_records(), count=60, holdout=12)
        probabilities = {m: 1 / len(MODELS) for m in MODELS}
        decisions = [dict(id=r['id'], partition=partition, rune_probabilities=probabilities)
                     for partition in ('train', 'test') for r in data[partition]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'manifest.json').write_text(json.dumps(
                dict(data_sha256=fingerprint(data), penalty_per_usd=10)))
            def save():
                (path / 'decisions.jsonl').write_text('\n'.join(map(json.dumps, decisions)))
            save()
            baselines = reference_baselines(data, path, 10)
            self.assertEqual(len(baselines), 60)
            self.assertEqual(baselines[data['test'][0]['state']], probabilities)
            with self.assertRaises(ValueError):
                reference_baselines(data, path, 20)
            decisions[-1]['partition'] = 'train'
            save()
            with self.assertRaises(ValueError):
                reference_baselines(data, path, 10)

    def test_frozen_baseline_does_not_make_inference_calls(self):
        class ExternalGateway:
            identity = 'fixture'

            def infer(self, *args):
                raise AssertionError('baseline should use recorded probabilities')

        probabilities = {m: 1 / len(MODELS) for m in MODELS}
        answerer = RuneAnswerer(routing_choice(10), ExternalGateway(),
                               baselines={'test': probabilities})
        self.assertEqual(answerer.ask('test', []).probabilities, probabilities)
        with self.assertRaises(KeyError):
            answerer.ask('unknown', [])

    def test_block_prefetch_preserves_sequential_decisions_and_feedback(self):
        data = prepare(fixture_records(), count=60, holdout=12)
        batches = []
        answerer = ExternalAnswerer()

        def prefetch(states, questions):
            batches.append(list(states))
            for state in states:
                answerer.ask(state, questions)

        with tempfile.TemporaryDirectory() as directory:
            plain, prefetched = Path(directory) / "plain", Path(directory) / "prefetched"
            with contextlib.redirect_stdout(io.StringIO()):
                replay(
                    data,
                    ExternalAnswerer(),
                    None,
                    plain,
                    learn_every=12,
                    expert=Expert(max_rounds=0),
                )
                replay(
                    data,
                    answerer,
                    None,
                    prefetched,
                    learn_every=12,
                    expert=Expert(max_rounds=0),
                    prefetch=prefetch,
                )
            self.assertEqual(
                (plain / "decisions.jsonl").read_text(),
                (prefetched / "decisions.jsonl").read_text(),
            )
        self.assertEqual(
            [state for batch in batches for state in batch],
            [r["state"] for r in data["train"] + data["test"]],
        )
        self.assertEqual([len(batch) for batch in batches], [12] * 5)

    def test_prefetch_uses_configured_parallel_http_requests(self):
        barrier = threading.Barrier(4)
        lock = threading.Lock()
        active, peak = [0], [0]

        class Fixture(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                with lock:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                try:
                    barrier.wait(timeout=5)
                    result = {
                        "output": {
                            "parsed": {
                                "answers": {
                                    "action": {
                                        "type": "choice",
                                        "probabilities": {m: 1 / len(MODELS) for m in MODELS},
                                    }
                                }
                            }
                        }
                    }
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())
                finally:
                    with lock:
                        active[0] -= 1

        server = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                gateway = Gateway(
                    f"http://127.0.0.1:{server.server_port}", {}, Path(directory), "concurrency"
                )
                RuneAnswerer(routing_choice(10), gateway, concurrency=4).prefetch(
                    ["a", "b", "c", "d"], []
                )
            self.assertEqual(peak[0], 4)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_offline_diagnostics_refit_real_saved_policy_without_mutation(self):
        data = prepare(fixture_records(), count=60, holdout=12)
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                replay(data, ExternalAnswerer(), None, directory, expert=Expert(max_rounds=0))
            database = Path(directory) / "policy.db"
            original = database.read_bytes()
            result = diagnose(data, database)
            self.assertEqual(database.read_bytes(), original)
            self.assertEqual(result["training_prompts"], 48)
            self.assertEqual(result["completed_test_prompts"], 12)
            self.assertEqual(sum(result["feedback_counts"].values()), 48)
            comparisons = result["comparisons"]
            ceiling = comparisons["hindsight_best_allowed_tilt"]["nudge"]["utility"]
            for name in (
                "logged_nudge",
                "chosen_feedback_mean",
                "all_training_outcomes_mean",
                "perfect_test_rewards_same_tilt",
            ):
                self.assertLessEqual(comparisons[name]["nudge"]["utility"], ceiling + 1e-12)
            self.assertEqual(
                comparisons["chosen_feedback_mean"],
                comparisons["chosen_feedback_no_questions_mean"],
            )

    def test_real_gateway_client_caches_and_keeps_baseline_independent(self):
        requests = []

        class Fixture(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, body))
                payload = json.loads(body["input"]["messages"][0]["content"])
                answers = {}
                for name, q in payload["questions"].items():
                    labels = list(q["criteria"])
                    probabilities = {label: 1 / len(labels) for label in labels}
                    answers[name] = {"type": "choice", "probabilities": probabilities}
                if payload.get("state") == "missing feature" and len(answers) > 1:
                    answers.pop(next(iter(answers)))
                response = {
                    "output": {"parsed": {"answers": answers}},
                    "usage": {"input_tokens": 100, "output_tokens": 1},
                    "inference_id": "fixture-id",
                    "variant_name": "native",
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

        server = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                gateway = Gateway(
                    f"http://127.0.0.1:{server.server_port}",
                    {"rune_api_key": "fixture-secret"},
                    directory,
                    "fixture-v1",
                )
                model = RuneAnswerer(routing_choice(10), gateway)
                original = model.ask("public prompt", [])
                with_feature = model.ask("public prompt", [Question("math", "Is this math?")])
                model.ask("public prompt", [Question("math", "Is this math?")])
                self.assertEqual(original.probabilities, with_feature.probabilities)
                self.assertEqual(with_feature.features, {"math": 0.5})
                self.assertEqual(len(requests), 2)
                questions = [Question(f"feature_{i}", f"Feature {i}?") for i in range(14)] + [
                    Question("level", "Which level?", "ordinal",
                             levels=("Very low", "Low", "Medium", "High", "Very high")),
                    Question("category", "Which category?", "categorical",
                             categories=("alpha", "beta", "gamma")),
                ]
                batched = model.ask("public prompt", questions)
                self.assertEqual(len(requests), 3)
                payload = json.loads(requests[-1][1]["input"]["messages"][0]["content"])
                self.assertEqual(payload["state"], "public prompt")
                self.assertEqual(len(payload["questions"]), 16)
                self.assertEqual(batched.probabilities, original.probabilities)
                self.assertEqual(batched.features["feature_0"], 0.5)
                self.assertAlmostEqual(batched.features["level"], 3)
                self.assertEqual(batched.features["category"],
                                 {name: 1 / 3 for name in ("alpha", "beta", "gamma")})
                self.assertEqual(model.ask("public prompt", questions), batched)
                self.assertEqual(len(requests), 3)
                with self.assertRaisesRegex(ValueError, "at most 32"):
                    model.ask("public prompt", questions * 3)
                self.assertEqual(len(requests), 3)
                frozen = RuneAnswerer(routing_choice(10), gateway,
                                      baselines={"another prompt": original.probabilities})
                frozen.ask("another prompt", questions)
                self.assertEqual(len(requests), 4)
                with self.assertRaisesRegex(InferenceError, "feature head mismatch"):
                    model.ask("missing feature", questions)
                self.assertTrue(all(path == "/inference" for path, _ in requests))
                self.assertEqual(requests[0][1]["credentials"]["rune_api_key"], "fixture-secret")
                gateway.infer('nudge_propose', 'public system', {'questions': {}})
                audit = [json.loads(line) for line in
                         (Path(directory) / 'inferences.jsonl').read_text().splitlines()]
                self.assertEqual(audit[-1]['input'], {
                    'system': 'public system', 'payload': {'questions': {}}})
                with patch('nudge.routerbench_live.JsonHTTP.post', side_effect=KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        gateway.infer('nudge_propose', 'cancelled system', {'questions': {}})
                interrupted = json.loads((Path(directory) / 'inferences.jsonl')
                                         .read_text().splitlines()[-1])
                self.assertEqual(interrupted['error'], 'KeyboardInterrupt')
                legacy = {k: v for k, v in interrupted.items() if k != 'error'}
                counted = overhead([legacy])['nudge_propose']
                self.assertEqual(counted['interrupted_or_incomplete'], 1)
                self.assertIsNone(counted['latency_seconds_p50_p95'])
                self.assertNotIn(
                    "fixture-secret", (Path(directory) / "inferences.jsonl").read_text()
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_cost_accounting_and_zero_router_use_declared_information(self):
        records = [
            {
                "function": "nudge_propose",
                "latency_seconds": 2,
                "usage": {"input_tokens": 1000, "output_tokens": 100},
                "provider_usage": {
                    "prompt_tokens_details": {"cached_tokens": 200},
                    "completion_tokens_details": {"reasoning_tokens": 30},
                },
            }
        ]
        cost = overhead(records)["nudge_propose"]
        self.assertAlmostEqual(cost["estimated_usd"], (800 * 1.4 + 200 * 0.14 + 100 * 4.4) / 1e6)
        self.assertEqual(cost["reasoning_tokens"], 30)
        data = {
            "train": [{"quality": [0, 1] + [0] * 9, "cost": [0, 1] + [2] * 9}],
            "test": [{"quality": [0.2, 0.8] + [0] * 9, "cost": [0, 1] + [2] * 9}],
        }
        mixture = supervised_zero_router(data, 0.5)
        self.assertEqual(mixture["probabilities"], {MODELS[0]: 0.5, MODELS[1]: 0.5})
        self.assertAlmostEqual(mixture["test_quality"], 0.5)

    def test_pinned_data_only_and_disjoint_reproducible_split(self):
        data = prepare(fixture_records(), count=36, holdout=12)
        self.assertEqual(data, prepare(fixture_records(), count=36, holdout=12))
        self.assertEqual(len(data["train"]), 24)
        self.assertEqual(len(data["test"]), 12)
        self.assertFalse({r["state"] for r in data["train"]} & {r["state"] for r in data["test"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "untrusted.pkl"
            path.write_bytes(b"untrusted pickle")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_pinned(path)

    def test_invalid_outcomes_and_duplicate_prompts_are_excluded(self):
        rows = fixture_records()
        rows += [dict(rows[0])]
        rows[1][MODELS[0]] = float("nan")
        data = prepare(rows, count=36, holdout=12)
        self.assertEqual(data["excluded"]["duplicate_prompt"], 1)
        self.assertEqual(data["excluded"]["incomplete_or_invalid_outcomes"], 1)

    def test_expected_regret_and_paired_metrics(self):
        row = {"quality": [1] + [0] * 10, "cost": [0.01] * 11}
        p = {model: float(i == 0) for i, model in enumerate(MODELS)}
        baseline = {model: 1 / 11 for model in MODELS}
        self.assertAlmostEqual(evaluate(row, p, 10)["utility"], 0.9)
        self.assertAlmostEqual(evaluate(row, p, 10)["hindsight_regret"], 0)
        records = [
            {"family": "math", "rune": evaluate(row, baseline, 10), "nudge": evaluate(row, p, 10)}
        ] * 5
        result = aggregate(records)
        self.assertAlmostEqual(result["paired_nudge_minus_rune"]["quality"]["mean"], 10 / 11)

    def test_replay_reveals_only_selected_training_feedback(self):
        data = prepare(fixture_records(), count=36, holdout=12)
        answerer = ExternalAnswerer()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            result = replay(
                data, answerer, None, directory, learn_every=12, expert=Expert(max_rounds=0)
            )
            self.assertEqual(result["report"]["feedback_count"], 24)
            self.assertEqual(result["report"]["decisions"], 36)
            self.assertEqual(result["test"]["count"], 12)
            records = [
                json.loads(line)
                for line in (Path(directory) / "decisions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len({r["version"] for r in records if r["partition"] == "test"}), 1)
            self.assertEqual(answerer.states, [r["state"] for r in data["train"] + data["test"]])
            # Feedback counts are per selected arm, never all 11 rewards per row.
            self.assertEqual(
                sum(x["feedback_count"] for x in result["report"]["actions"].values()), 24
            )

    def test_choice_validation(self):
        self.assertEqual(
            choice_probabilities(
                {"type": "choice", "probabilities": {"x": 0.5, "y": 0.5}}, ["x", "y"]
            ),
            {"x": 0.5, "y": 0.5},
        )
        with self.assertRaises(InferenceError):
            choice_probabilities({"type": "choice", "probabilities": {"x": 2, "y": -1}}, ["x", "y"])

    def test_rune_protocol_bridge_real_http_with_external_transport_fixture(self):
        native = {
            "answers": {"a": {"type": "choice", "probabilities": {"yes": 0.8, "no": 0.2}}},
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
        provider = NativeProvider(
            "https://model-example.api.baseten.co/environments/rune-dst-v1/sync/v1/systemone"
        )
        requested = []

        def upstream_open(opener, request, **kwargs):
            # Intercept only the external Baseten request; local HTTP remains real.
            if request.full_url == provider.endpoint:
                requested.append(request)
                return io.BytesIO(json.dumps(native).encode())
            return original_open(opener, request, **kwargs)

        import urllib.request

        original_open = urllib.request.OpenerDirector.open
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler(provider))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                key = Path(directory) / "key"
                key.write_text("fixture-credential")
                client = JsonHTTP(
                    f"http://127.0.0.1:{server.server_port}/v1/chat/completions", api_key_file=key
                )
                payload = {
                    "state": "public question",
                    "questions": {
                        "a": {
                            "type": "choice",
                            "instructions": "Code?",
                            "criteria": {"yes": "Yes", "no": "No"},
                        }
                    },
                }
                with patch.object(urllib.request.OpenerDirector, "open", upstream_open):
                    response = client.post(
                        {
                            "model": "rune",
                            "messages": [{"role": "user", "content": json.dumps(payload)}],
                        }
                    )
                self.assertEqual(json.loads(response["choices"][0]["message"]["content"]), native)
                self.assertEqual(
                    requested[0].get_header("Authorization"), "Api-Key fixture-credential"
                )
                self.assertEqual(json.loads(requested[0].data), payload)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_gateway_config_thinking_and_native_transport(self):
        config = tomllib.loads((Path(__file__).parents[1] / "gateway/routerbench.toml").read_text())
        variant = config["functions"]["nudge_propose"]["variants"]["glm_thinking"]
        self.assertEqual(variant["model"], "glm-5.2-full")
        self.assertEqual(
            variant["extra_body"],
            [{"pointer": "/chat_template_kwargs", "value": {"thinking": True}}],
        )
        self.assertEqual(
            config["models"]["rune_native"]["providers"]["native_bridge"]["type"], "openai"
        )


if __name__ == "__main__":
    unittest.main()
