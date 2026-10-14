# Answerers and proposers

The core accepts any object with a stable `identity` and
`ask(state, questions) -> Answer(probabilities, features)`. Bind the Choice when
constructing the answerer. Action probabilities must cover the declared actions;
feature keys must exactly match the requested questions. No provider name is
special to the learning algorithm. The `Decision.jev` field retains the original
proposal's name but contains whatever baseline the answerer supplies.

## HF and other score-producing models

`LogScoreAnswerer` adapts a callable `scorer(state, requests)` to this interface.
It sends one batch containing a Choice request and the requested binary,
ordinal, or categorical questions. Each request has `instructions` and `labels`
(label -> description).
Return one mapping of comparable label log scores per request.

```python
from nudge.adapters import LogScoreAnswerer

answerer = LogScoreAnswerer(
    choice,
    scorer=my_batched_scorer,
    identity="checkpoint-commit:tokenizer-revision:prompt-v1",
)
```

This fits HF classifier logits or conditional log-probability scorers without
requiring Transformers/Torch in the base package. For a causal LM, the scorer
must correctly score candidate labels, including tokenizer and multi-token
handling; generated numeric confidence is not interchangeable with those scores.
Each question must be scored independently so extra feature questions cannot
change the baseline. Binary features become the probability of yes; ordinal
features become expected levels 1..5; categorical features remain a soft
probability vector over their declared labels. Ordinal questions need five
concrete level descriptions, and categorical questions need 2..32 labels.
Arbitrary checkpoints need their own verified scoring backend;
v0 does not load `hf:<repo>` identifiers automatically.

The synthetic benchmark scorer and adapter tests are executable examples of the
contract, not evidence that a particular HF checkpoint is calibrated or useful.

## TensorZero

`TensorZero(base_url=..., function_name=..., revision=...)` sends bounded JSON
inference requests. `TensorZeroProposer(gateway)` builds and validates proposal
requests. `TensorZeroAnswerer(choice, gateway)` expects a configured classifier
function returning `{ "probabilities": {...}, "features": {...} }`.

Use explicit `api_key_file` and `credential_files={name: path}` parameters when
needed. Read mounted Secret Manager files in the host environment. There are no
environment lookups or credentials in package configuration. HTTP has at most
three attempts by default, a 30-second socket timeout, bounded backoff, an 8 MiB
response limit, no redirects, and retries only for network failures or specified
transient statuses. Nonretryable error response bodies are not exposed.

`gateway/functions.toml` is a function configuration example, not a deployed
service. Choose an existing chat-model route for the proposer. The answerer
requires a real scoring route; pointing a generic chat model at its schema does
not create calibrated action probabilities. Tests exercise the real HTTP client
against a local gateway fixture. No live credentials/checkpoint were provided.

## Native Rune through TensorZero

The RouterBench pilot includes a local TensorZero provider bridge for Rune's
native Choice protocol. Application inference still enters through `/inference`.
See [the pilot runbook](routerbench.md) for isolation, credentials, caching and
the explicitly thinking-enabled GLM proposer configuration.

`RuneAnswerer` sends all requested feature questions (up to 32) in one native
Choice request with a shared state. Binary, ordinal, and categorical answers
retain their existing probability/expectation decoding. The baseline remains a
separate, cached request; frozen baselines require no baseline inference. An
empty feature bank sends no feature request. Gateway caching uses the complete
batch, so changing the bank can require a new request.

The answerer identity is `routerbench-rune-batched-v2`. Start a fresh choice/store
and refit when upgrading from the single-question adapter; preserve existing
stores as historical artifacts. Batched soft probabilities can differ from
single-question probabilities. Discovery still caches measured features per
question, so measurements may come from different discovery batches; batching
does not guarantee invariance to bank membership or measured GPU/latency savings.

`LiveProposer(gateway, max_proposals=16)` only proposes edits. It no longer accepts
an answerer or an all-history `seen_states` list and performs no feature prefetch.
Core hydration answers new questions only on bounded replay, charging its call
budget. RouterBench exposes `--max-training-rows` (default 1024).

## TypeSafe protocol codecs

`typesafe_request` and `typesafe_answer` implement the documented typed contract:
action Choice, binary Noul, ordinal Score, and categorical Choice. Score indices start at zero, so
the decoder adds one. Responses must match the pinned model revision. The codecs
make no network calls and do not hard-code TypeSafe as the baseline provider.

References verified during implementation:
[TypeSafe API](https://docs.typesafe.ai/api),
[Score semantics](https://docs.typesafe.ai/primitives/score), and
[independent question evaluation](https://docs.typesafe.ai/introduction).

The host repository requires model inference through TensorZero. No direct
TypeSafe endpoint adapter is installed here; these pure codecs are available
for a future backend integration. This restriction does not constrain the core's
model-independent API.
