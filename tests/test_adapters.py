import json
import math
import threading
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from nudge import Choice, ProposalContext, Question
from nudge.adapters import (
    InferenceError,
    JsonHTTP,
    LogScoreAnswerer,
    TensorZero,
    TensorZeroAnswerer,
    TensorZeroProposer,
    typesafe_answer,
    typesafe_request,
)
from nudge.routerbench_live import Gateway, LiveProposer, RuneAnswerer
from nudge.discovery import question_key


CHOICE = Choice(
    "support",
    "Support ticket",
    {"answer": "Answer", "escalate": "Escalate"},
    {"resolved": "Resolution"},
)
LEVELS = (
    "No urgency",
    "Can wait days",
    "Needs help soon",
    "Needs help today",
    "Immediate emergency",
)


class AdapterTests(unittest.TestCase):
    def test_atomic_bundle_survives_both_gateway_adapters(self):
        questions = [Question('repeat', 'Repeated?', paraphrase='Recurring?'),
                     Question('urgent', 'Urgent?', paraphrase='Needs help today?')]
        proposal = {'operation': 'add_bundle', 'target': None, 'question': None,
                    'questions': [q.to_dict() for q in questions]}
        url, _ = self.serve([(200, {'output': {'parsed': {'proposals': [proposal]}}})])
        context = ProposalContext(CHOICE.to_dict(), (), (), 1)
        generic = TensorZeroProposer(TensorZero(base_url=url, function_name='nudge_propose', revision='test'))
        self.assertEqual(generic.propose(context)[0].questions, tuple(questions))
        with tempfile.TemporaryDirectory() as directory:
            gateway = Gateway(url, {}, Path(directory), 'test-bundle')
            live = LiveProposer(gateway)
            self.assertEqual(live.propose(context)[0].questions, tuple(questions))

    def test_live_proposer_leaves_feature_inference_to_budgeted_core(self):
        q = Question("repeat", "Repeated?", paraphrase="Recurring?")
        alternate = replace(q, prompt=q.paraphrase, paraphrase=None)

        def response(value):
            return (200, {"output": {"parsed": value}})

        def answer(key, labels):
            return response({"answers": {key: {"type": "choice", "probabilities": labels}}})

        url, requests = self.serve(
            [
                response(
                    {"proposals": [{"operation": "add", "target": None, "question": q.to_dict()}]}
                ),
                answer("action", {"answer": 0.5, "escalate": 0.5}),
                answer(question_key(alternate), {"no": 0.25, "yes": 0.75}),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            gateway = Gateway(url, {}, Path(directory), "paraphrase")
            answerer = RuneAnswerer(CHOICE, gateway, concurrency=4)
            proposer = LiveProposer(gateway)
            context = ProposalContext(CHOICE.to_dict(), (), ({"state": "training"},), 1)
            proposer.propose(context)
            self.assertEqual(len(requests), 1)
            self.assertEqual(answerer.ask("training", [alternate]).features["repeat"], 0.75)
        self.assertEqual(len(requests), 3)
        body = json.loads(requests[-1][1]["input"]["messages"][0]["content"])
        self.assertEqual(body["state"], "training")
        self.assertEqual(body["questions"][question_key(alternate)]["instructions"], "Recurring?")

    def test_sixteen_proposals_survive_both_real_gateway_adapters(self):
        proposals = [
            {
                "operation": "add",
                "target": None,
                "question": {
                    "name": f"feature_{i}",
                    "prompt": f"Feature {i}?",
                    "kind": "binary",
                    "paraphrase": f"Does feature {i} apply?",
                    "levels": None,
                },
            }
            for i in range(16)
        ]
        url, requests = self.serve([(200, {"output": {"parsed": {"proposals": proposals}}})])
        context = ProposalContext(CHOICE.to_dict(), (), (), 1)
        generic = TensorZeroProposer(
            TensorZero(base_url=url, function_name="nudge_propose", revision="test")
        )
        self.assertEqual(len(generic.propose(context)), 16)
        with tempfile.TemporaryDirectory() as directory:
            gateway = Gateway(url, {}, Path(directory), "test")
            live = LiveProposer(gateway)
            self.assertEqual(len(live.propose(context)), 16)
            with self.assertRaises(InferenceError):
                LiveProposer(gateway, max_proposals=4).propose(
                    context
                )
        for _, body in requests[:2]:
            self.assertIn("no more than 16 proposals", body["input"]["system"])
        schema = json.loads(
            (Path(__file__).parents[1] / "gateway/schemas/proposals.json").read_text()
        )
        self.assertEqual(schema["properties"]["proposals"]["maxItems"], 16)

    def test_typesafe_wire_shape_and_zero_based_score(self):
        questions = [
            Question("repeat", "Repeated?"),
            Question("urgency", "Urgency?", "ordinal", levels=LEVELS),
        ]
        request = typesafe_request(CHOICE, "ticket", questions, "jev-1.13.0")
        self.assertEqual(request["questions"]["action"]["criteria"], dict(CHOICE.actions))
        self.assertEqual(request["questions"]["feature_0"]["type"], "noul")
        self.assertEqual(request["questions"]["feature_1"]["criteria"], list(LEVELS))
        # Constructed from the documented HTTP shapes, not a captured live response.
        response = {
            "model": "jev-1.13.0",
            "answers": {
                "action": {
                    "type": "choice",
                    "choice": "answer",
                    "confidence": 0.6,
                    "probabilities": {"answer": 0.8, "escalate": 0.2},
                },
                "feature_0": {"type": "noul", "noul": 0.9},
                "feature_1": {
                    "type": "score",
                    "score": 2.4,
                    "confidence": 0.5,
                    "probabilities": {"0": 0, "1": 0, "2": 0.6, "3": 0.4, "4": 0},
                },
            },
        }
        answer = typesafe_answer(response, CHOICE, questions, "jev-1.13.0")
        self.assertEqual(answer.features, {"repeat": 0.9, "urgency": 3.4})
        with self.assertRaises(InferenceError):
            typesafe_answer(response, CHOICE, questions, "jev-1.12.0")
        response["answers"]["feature_1"]["score"] = 5
        with self.assertRaises(InferenceError):
            typesafe_answer(response, CHOICE, questions, "jev-1.13.0")

    def test_batched_log_scores(self):
        calls = []

        def score(state, requests):
            calls.append((state, requests))
            return [
                {"answer": math.log(0.8), "escalate": math.log(0.2)},
                {"no": math.log(0.1), "yes": math.log(0.9)},
                {str(i): 0.0 for i in range(1, 6)},
            ]

        model = LogScoreAnswerer(CHOICE, scorer=score, identity="local-checkpoint-v1")
        answer = model.ask(
            "ticket",
            [
                Question("repeat", "Repeated?"),
                Question("urgency", "Urgency?", "ordinal", levels=LEVELS),
            ],
        )
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(answer.probabilities["answer"], 0.8)
        self.assertAlmostEqual(answer.features["repeat"], 0.9)
        self.assertAlmostEqual(answer.features["urgency"], 3)

    def test_batched_log_scores_keep_categorical_distribution(self):
        def score(state, requests):
            return [
                {"answer": math.log(0.8), "escalate": math.log(0.2)},
                {"0": math.log(0.1), "1": math.log(0.7), "2": math.log(0.2)},
            ]

        question = Question(
            "task_family",
            "What kind of task is this?",
            "categorical",
            categories=("math", "code", "reading"),
        )
        answer = LogScoreAnswerer(CHOICE, scorer=score, identity="categorical-check").ask(
            "prompt", [question]
        )
        self.assertAlmostEqual(answer.features["task_family"]["code"], 0.7)

    def serve(self, responses):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append(
                    (self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                )
                status, response = responses[min(len(requests) - 1, len(responses) - 1)]
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def close():
            server.shutdown()
            server.server_close()
            thread.join(2)

        self.addCleanup(close)
        return f"http://127.0.0.1:{server.server_port}", requests

    def test_zero_probability_log_scores_and_invalid_scores(self):
        model = LogScoreAnswerer(
            CHOICE, scorer=lambda *_: [{"answer": 0, "escalate": -math.inf}], identity="zero"
        )
        self.assertEqual(model.ask("state", []).probabilities, {"answer": 1.0, "escalate": 0.0})
        for scores in ((-math.inf, -math.inf), (math.nan, 0), (math.inf, 0)):
            with self.subTest(scores=scores):
                model.scorer = lambda *_: [dict(zip(CHOICE.actions, scores))]
                with self.assertRaises((InferenceError, ValueError)):
                    model.ask("state", [])

    def test_gateway_answerer_contract(self):
        url, requests = self.serve(
            [
                (
                    200,
                    {
                        "output": {
                            "parsed": {
                                "probabilities": {"answer": 0.7, "escalate": 0.3},
                                "features": {"repeat": 0.9},
                            }
                        }
                    },
                )
            ]
        )
        model = TensorZeroAnswerer(
            CHOICE, TensorZero(base_url=url, function_name="scored_choice", revision="fixture-v1")
        )
        answer = model.ask("my ticket", [Question("repeat", "Repeated?")])
        self.assertEqual(answer.features, {"repeat": 0.9})
        payload = json.loads(requests[0][1]["input"]["messages"][0]["content"][0]["text"])
        self.assertEqual(payload["state"], "my ticket")
        self.assertNotIn("reward", payload["choice"])

    def test_gateway_real_http_retry_and_proposer_parsing(self):
        url, requests = self.serve(
            [
                (503, {"error": "retry"}),
                (
                    200,
                    {
                        "output": {
                            "parsed": {
                                "proposals": [
                                    {
                                        "operation": "add",
                                        "target": None,
                                        "question": {
                                            "name": "repeat",
                                            "prompt": "Repeated?",
                                            "kind": "binary",
                                            "paraphrase": "Recurring?",
                                            "levels": None,
                                        },
                                    }
                                ]
                            }
                        }
                    },
                ),
            ]
        )
        gateway = TensorZero(base_url=url, function_name="nudge_propose", revision="test-v1")
        gateway.http.backoff = 0
        proposals = TensorZeroProposer(gateway).propose(
            ProposalContext(CHOICE.to_dict(), (), (), 1)
        )
        self.assertEqual(proposals[0].question.name, "repeat")
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0][0], "/inference")
        self.assertEqual(requests[0][1]["function_name"], "nudge_propose")

    def test_auth_error_not_retried_or_exposed(self):
        url, requests = self.serve([(401, {"error": "sensitive-provider-body"})])
        with self.assertRaises(InferenceError) as caught:
            JsonHTTP(url + "/inference", backoff=0).post({})
        self.assertNotIn("sensitive", str(caught.exception))
        self.assertEqual(len(requests), 1)

    def test_gateway_rejects_unparsed_output(self):
        url, _ = self.serve([(200, {"output": {"raw": "not json", "parsed": None}})])
        with self.assertRaises(InferenceError):
            TensorZero(base_url=url, function_name="test", revision="v1").infer("system", {})

    def test_invalid_transport_config(self):
        for url in ("https://user:secret@example.com", "http://example.com", "file:///etc/passwd"):
            with self.assertRaises(ValueError):
                JsonHTTP(url)


if __name__ == "__main__":
    unittest.main()
