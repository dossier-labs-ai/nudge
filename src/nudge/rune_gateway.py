"""Local TensorZero provider bridge for native Rune Choice inference.

This process is the provider transport behind TensorZero, never the application
inference entrypoint. Only the fixed operator-supplied native endpoint is used.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .adapters import InferenceError, NoRedirect
from .types import canonical


def native_request(body: dict) -> dict:
    if body.get("stream"):
        raise ValueError("streaming is unsupported")
    messages = [m for m in body["messages"] if m["role"] == "user"]
    if len(messages) != 1:
        raise ValueError("exactly one native request is required")
    content = messages[0]["content"]
    if isinstance(content, list):
        content = "".join(block["text"] for block in content if block["type"] == "text")
    payload = json.loads(content)
    if set(payload) != {"state", "questions"} or not isinstance(payload["state"], str):
        raise ValueError("invalid native request")
    if not 1 <= len(payload["questions"]) <= 32:
        raise ValueError("native request needs 1..32 heads")
    for question in payload["questions"].values():
        if question["type"] != "choice" or not isinstance(question["instructions"], str):
            raise ValueError("only Choice heads are supported")
        if not 2 <= len(question["criteria"]) <= 255:
            raise ValueError("Choice needs 2..255 labels")
    return payload


class NativeProvider:
    def __init__(self, endpoint: str, timeout: float = 30, max_concurrency: int = 2):
        if not re.fullmatch(
            r"https://model-[a-z0-9]+\.api\.baseten\.co/environments/[a-z0-9-]+/sync/v1/systemone",
            endpoint,
        ):
            raise ValueError("expected a Baseten native Rune environment URL")
        self.endpoint, self.timeout = endpoint, timeout
        if type(max_concurrency) is not int or not 1 <= max_concurrency <= 128:
            raise ValueError("max_concurrency must be between 1 and 128")
        self.slots = threading.BoundedSemaphore(max_concurrency)

    def infer(self, payload: dict, authorization: str) -> dict:
        if not authorization.startswith("Bearer ") or not authorization[7:].strip():
            raise ValueError("missing TensorZero dynamic provider credential")
        request = urllib.request.Request(
            self.endpoint,
            data=canonical(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Api-Key " + authorization[7:],
            },
            method="POST",
        )
        if not self.slots.acquire(timeout=self.timeout):
            raise InferenceError("Rune provider capacity timeout")
        try:
            with urllib.request.build_opener(NoRedirect()).open(
                request, timeout=self.timeout
            ) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise InferenceError("Rune response too large")
            return json.loads(raw)
        except urllib.error.HTTPError as error:
            code = error.code
            error.close()
            raise InferenceError(f"Rune HTTP {code}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise InferenceError("Rune network/timeout failure") from None
        finally:
            self.slots.release()


def handler(provider):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log credentials or request/response content.

        def send_json(self, status, value):
            raw = canonical(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self.send_json(200 if self.path == "/health" else 404, {"provider": "rune-native"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_json(404, {"error": "unknown route"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2 * 1024 * 1024:
                    raise ValueError("invalid request size")
                body = json.loads(self.rfile.read(length))
                payload = native_request(body)
                native = provider.infer(payload, self.headers.get("Authorization", ""))
                usage = native.get("usage", {})
                self.send_json(
                    200,
                    {
                        "id": "rune-native",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": canonical(native),
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": usage.get("input_tokens", 0),
                            "completion_tokens": usage.get("output_tokens", 0),
                            "total_tokens": usage.get("input_tokens", 0)
                            + usage.get("output_tokens", 0),
                        },
                    },
                )
            except (ValueError, KeyError, TypeError):
                self.send_json(400, {"error": "invalid Rune provider request/response"})
            except InferenceError as error:
                self.send_json(502, {"error": str(error)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--port", type=int, default=3119)
    parser.add_argument("--max-concurrency", type=int, choices=range(1, 129), default=2)
    args = parser.parse_args()

    class ProviderServer(ThreadingHTTPServer):
        request_queue_size = 128

    server = ProviderServer(
        ("127.0.0.1", args.port),
        handler(NativeProvider(args.endpoint, max_concurrency=args.max_concurrency)),
    )
    server.daemon_threads = True
    print(canonical({"provider": "rune-native", "port": args.port}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
