"""Live TensorZero adapters and persistent inference evidence for RouterBench."""

from __future__ import annotations

import json
import math
import sqlite3
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .adapters import InferenceError, JsonHTTP, proposer_prompt
from .discovery import question_key
from .routerbench_data import fingerprint
from .types import Answer, Proposal, Question, canonical, validate_answer


def secret(project, reference, source="gcloud"):
    name, _, version = reference.partition(":")
    command = [
        "gcloud",
        "secrets",
        "versions",
        "access",
        version or "latest",
        "--project=" + project,
        "--secret=" + name,
    ]
    token = None
    if source == "application-default":
        auth = subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if auth.returncode:
            raise RuntimeError("application-default authentication unavailable")
        token = auth.stdout.strip()
        command.append("--access-token-file=/dev/stdin")
    elif source != "gcloud":
        raise ValueError("unknown credential source")
    completed = subprocess.run(command, input=token, capture_output=True, text=True, timeout=60)
    if completed.returncode or not completed.stdout.strip():
        raise RuntimeError(f"cannot read Secret Manager reference {project}/{reference}")
    return completed.stdout.strip()


class Gateway:
    """Only calls TensorZero /inference; secrets remain in memory.

    Cache identity includes the checked gateway configuration and endpoint/model
    revision manifest. Raw reasoning text is not retained, only presence/length.
    """

    def __init__(self, base_url, credentials, directory, identity, max_calls=10000):
        self.base_url, self._credentials, self.identity = base_url, credentials, identity
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.directory / "inferences.db"
        self.lock, self.calls, self.max_calls = threading.Lock(), 0, max_calls
        with sqlite3.connect(self.cache_path) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    def infer(self, function, system, payload):
        key = fingerprint([self.identity, function, system, payload])
        with sqlite3.connect(self.cache_path, timeout=60) as db:
            hit = db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        if hit:
            return json.loads(hit[0])
        with self.lock:
            if self.calls >= self.max_calls:
                raise InferenceError("pilot gateway call limit reached")
            self.calls += 1
        body = {
            "function_name": function,
            "include_raw_response": True,
            "input": {
                "system": system,
                "messages": [{"role": "user", "content": canonical(payload)}],
            },
            "credentials": self._credentials,
        }
        start = time.monotonic()
        record = {"key": key, "function": function, "started_at": time.time()}
        if function == 'nudge_propose':
            # Public benchmark context only; never persist the credential body.
            # Keep failed requests too, before feature prefetch can fail.
            record['input'] = {'system': system, 'payload': payload}
        try:
            response = JsonHTTP(
                self.base_url.rstrip("/") + "/inference",
                timeout=300 if function in ("nudge_propose", "nudge_semantic_labels") else 45,
                # Rune can return transient 502/503 responses while replicas
                # scale or recover. Keep the request bounded, but do not let
                # one transient gateway response abort a multi-hour pilot.
                attempts=5,
                backoff=0.5,
            ).post(body)
            record.update(
                inference_id=response.get("inference_id"),
                variant=response.get("variant_name"),
                usage=response.get("usage", {}),
            )
            raw_responses = response.get("raw_response", [])
            record["reasoning_characters"] = 0
            for raw in raw_responses:
                native = json.loads(raw["data"])
                for item in native.get("choices", []):
                    message = item.get("message", {})
                    record["reasoning_characters"] += len(message.get("reasoning_content") or "")
                if native.get("usage"):
                    record["provider_usage"] = native["usage"]
                record["provider_model"] = native.get("model")
            parsed = response.get("output", {}).get("parsed")
            if not isinstance(parsed, dict):
                # Keep public benchmark output for diagnosing a schema failure.
                record["invalid_output"] = response.get("output")
                raise InferenceError("gateway did not return parsed JSON")
            record["result"] = parsed
            with sqlite3.connect(self.cache_path, timeout=60) as db:
                db.execute("INSERT OR IGNORE INTO cache VALUES (?, ?)", (key, canonical(parsed)))
            return parsed
        except InferenceError as error:
            record["error"] = str(error)
            raise
        except BaseException as error:
            record['error'] = type(error).__name__
            raise
        finally:
            record["latency_seconds"] = time.monotonic() - start
            with self.lock, (self.directory / "inferences.jsonl").open("a") as stream:
                stream.write(canonical(record) + "\n")


def choice_probabilities(answer, labels):
    if answer["type"] != "choice" or set(answer["probabilities"]) != set(labels):
        raise InferenceError("Rune Choice labels/type mismatch")
    values = {key: float(value) for key, value in answer["probabilities"].items()}
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in values.values()):
        raise InferenceError("invalid Rune probability")
    total = sum(values.values())
    if not math.isclose(total, 1, abs_tol=1e-5, rel_tol=0):
        raise InferenceError("Rune probabilities do not sum to one")
    return {key: value / total for key, value in values.items()}


class RuneAnswerer:
    def __init__(self, choice, gateway, *, concurrency=2, baselines=None):
        if type(concurrency) is not int or not 1 <= concurrency <= 128:
            raise ValueError("concurrency must be between 1 and 128")
        self.choice, self.gateway = choice, gateway
        self.concurrency = concurrency
        self.baselines = baselines
        self.identity = "routerbench-rune-batched-v2:" + fingerprint([gateway.identity, choice.to_dict()])
        if baselines is not None:
            self.identity += ':frozen-baselines:' + fingerprint(baselines)

    def ask(self, state, questions):
        questions = list(questions)
        if len(questions) > 32:
            raise ValueError("Rune supports at most 32 feature questions per request")
        # Baseline is its own immutable request. Adding feature heads cannot
        # silently change its probabilities; gateway results are cached by body.
        action = {
            "type": "choice",
            "instructions": self.choice.situation,
            "criteria": dict(self.choice.actions),
        }
        result = ({'answers': {'action': {'type': 'choice',
                                         'probabilities': self.baselines[state]}}}
                  if self.baselines is not None else self.gateway.infer(
                      "nudge_rune", "Native Rune classification request.",
                      {"state": state, "questions": {"action": action}}))
        if set(result["answers"]) != {"action"}:
            raise InferenceError("Rune baseline head mismatch")
        probabilities = choice_probabilities(result["answers"]["action"], self.choice.actions)
        features = {}
        heads = {}
        for question in questions:
            labels = (
                {"no": "No", "yes": "Yes"}
                if question.kind == "binary"
                else (
                    {str(i + 1): description for i, description in enumerate(question.levels)}
                    if question.kind == "ordinal"
                    else {
                        str(i): category for i, category in enumerate(question.categories or ())
                    }
                )
            )
            heads[question_key(question)] = {
                "type": "choice", "instructions": question.prompt, "criteria": labels
            }
        if heads:
            result = self.gateway.infer(
                "nudge_rune",
                "Native Rune classification request.",
                {"state": state, "questions": heads},
            )
            if set(result["answers"]) != set(heads):
                raise InferenceError("Rune feature head mismatch")
        for question in questions:
            key = question_key(question)
            labels = heads[key]["criteria"]
            distribution = choice_probabilities(result["answers"][key], labels)
            features[question.name] = (
                distribution["yes"]
                if question.kind == "binary"
                else (
                    sum(int(label) * probability for label, probability in distribution.items())
                    if question.kind == "ordinal"
                    else {
                        category: distribution[str(i)]
                        for i, category in enumerate(question.categories or ())
                    }
                )
            )
        return validate_answer(Answer(probabilities, features), self.choice, questions)

    def prefetch(self, states, questions):
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            # Consume every result so errors cannot silently disappear.
            for i, _ in enumerate(pool.map(lambda state: self.ask(state, questions), states), 1):
                if i % 100 == 0:
                    print(
                        canonical(
                            {
                                "stage": "prefetch",
                                "completed": i,
                                "total": len(states),
                                "questions": [q.name for q in questions],
                            }
                        ),
                        flush=True,
                    )


class LiveProposer:
    def __init__(self, gateway, *, max_proposals=16):
        self.gateway = gateway
        self.prompt = proposer_prompt(max_proposals)
        self.max_proposals = max_proposals
        self.identity = (
            f"glm-5.2-thinking:routerbench-proposals-v8:{max_proposals}:" + gateway.identity
        )

    def propose(self, context):
        # A thinking model can spend its output allowance before emitting JSON.
        # Failed outputs are audited, never cached; retry this exact request at
        # most twice without changing examples or fabricating an empty proposal.
        for attempt in range(3):
            try:
                raw = self.gateway.infer("nudge_propose", self.prompt, context.to_dict())["proposals"]
                break
            except InferenceError as error:
                if str(error) != 'gateway did not return parsed JSON' or attempt == 2:
                    raise
        if not isinstance(raw, list) or len(raw) > self.max_proposals:
            raise InferenceError(
                f"pilot proposer must return at most {self.max_proposals} proposals"
            )
        proposals = [
            Proposal(
                operation=p["operation"],
                target=p.get("target"),
                question=Question(**p["question"]) if p.get("question") else None,
                questions=tuple(Question(**q) for q in (p.get('questions') or [])),
            )
            for p in raw
        ]
        # Core owns bounded replay hydration and its model-call budget.
        return proposals
