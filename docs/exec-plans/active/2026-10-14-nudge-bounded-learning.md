# Plan: Bound Nudge learning work with one shared sample

Status: Planned — implementation deferred at user request
Owner: Agent
Last updated: 2026-10-14

## Goal

History can grow while expensive answering, question evidence and predictor fitting
remain bounded per learning update. Reuse the existing decision log, replay module,
answer cache and atomic snapshot activation; avoid another training pipeline.

The user approved planning this policy and explicitly said: "We will implement it,
but not now." This document is not a claim that bounded training is implemented.

## Done Criteria

- Select at most 256 eligible observations once per update, before hydration.
  Candidate answering, evidence and fitting may use only those selected rows.
- Refresh the sample across updates without replacement within an update. Retain
  durable history for sampling and audit; do not silently evict feedback.
- Screen candidates on a small subset before filling their remaining answers.
  A failed screening check never triggers full-sample candidate answering.
- One update budget bounds actual inference attempts and proposer input/output;
  cache hits do not consume inference allowance and retries do consume it.
- If the update cannot complete within budget, preserve the last activated policy
  and bank. Keep useful cached measurements and an explicit failure/deferred record.
- Replayed old observations never become new confirming evidence. Evaluation
  outcomes never enter sampling, discovery or fitting.
- Tests, owning docs/spec and a measured bounded-work experiment verify the above.
  Do not require immediate routing gain to retain a question hypothesis.

## Context

Read: repository AGENTS.md, ARCHITECTURE.md, docs/README.md,
docs/AGENTIC_HARNESS.md, docs/product-specs/index.md, specs/nudge-v0-spec.md,
docs/design.md, routerbench.md and experiment-log-2026-10-14.md.
Relevant code: replay.py, core.py, discovery.py, question_evidence.py, store.py,
types.py, adapters.py, routerbench.py and routerbench_live.py in the package.

Verified current behavior:

- SamplingReplayBuffer limits proposer examples, not regression training rows.
- Nudge._learn loads eligible history and hydrates it before discovery and fit.
- LiveProposer.propose independently prefetches candidate answers on seen_states,
  before the core checks proposal validity and paraphrase consistency.
- check_answer uses 16 rows, after the full candidate hydration. Live prefetch and
  core screening can select different paraphrase rows.
- Gateway caching prevented duplicate request keys in v8, but 24,896 Rune calls
  were still made. Ten unstable rejected candidates covered 5,800 historical
  prompt/question pairs before rejection; this is coverage, not an exact count of
  wasted network calls. Three proposer calls failed to emit JSON.
- Current 100,000 learning-call and 200,000 gateway-call ceilings do not provide a
  useful operating budget. Core counts answerer invocations while the live adapter
  can expand them into many requests; HTTP retries are not separate audit calls.
- Completed v8 artifacts remain local under benchmark-results/routerbench/. The
  experiment log records results and limitations; do not overwrite that evidence.

## Policy / Mechanism Boundary

### Policy contract

- User intent: useful observations, bounded work, and simple code. The agreed
  implementation direction is one uniform sample of up to 256 rows for evidence
  and fitting; prioritize proposer examples only within that sample initially.
- Uniform selection avoids adding inverse inclusion-probability machinery for a
  prioritized regression sample. Logged action propensity remains a different
  quantity and retains its existing interpretation.
- Eligibility requires usable feedback and positive recency weight. Preserve
  objective, recency and correction semantics; missing feedback is not zero reward.
- Do not silently enforce action/task quotas while calling the sample uniform.
  Record coverage and insufficient evidence. Stratification is a separate later
  policy if measurements show it is necessary.
- Preserve discovery-only versus fitting-only operations and frozen evaluation.

### Mechanism contract

- Select IDs once in core._learn before answer hydration, through a small helper
  in the existing replay module. Use a reproducible seed tied to the update/history
  boundary; maintain existing decision RNG and resume determinism.
- Pass the bounded rows downstream. Remove LiveProposer's historical prefetch side
  effect rather than maintaining two competing candidate-evaluation paths.
- Validate proposals first. Use one explicitly selected screening subset for
  original/paraphrase answers, then fill only surviving candidates on remaining
  sampled rows. Retain concurrency and caches in the answerer without allowing it
  to choose extra states. Preserve atomic bundle semantics.
- Keep the full observed decision boundary separate from sampled row count.
  Resume, feedback counts, late corrections and evidence replication must not
  mistake a sample maximum or its size for the full history cutoff.
- Preserve frozen initial question evidence and use only genuinely later decision
  IDs for subsequent evidence. Changing sample membership cannot create evidence
  accumulation or count the same decision twice.
- Make one budget owner authoritative for an update. Check/reserve allowance before
  dispatch, safely under concurrency; include per-question expansion and retries.
  Resolve interaction with generic adapters as well as the live gateway, removing
  misleading parallel counters rather than adding another independent budget.
- Bound proposer context before submission and output via request limits. Define
  the token-counting method and treatment of oversized individual prompts; a row
  cap alone is not a token cap. Never silently crop away material task content.
- Exact call/token allowances are implementation-time configuration choices, not
  settled numbers in this plan. Record them explicitly in Expert/CLI configuration
  and manifests, with no environment-variable configuration path.
- Persist sampled IDs/seed, eligible and sampled counts, cutoff, budget consumption,
  screening outcomes and timings in existing run/snapshot reporting. Distinguish
  inference cache hits, transport attempts and failures in accounting.
- Reading/sampling history may initially be O(history). The immediate guarantee
  bounds expensive inference and fit size, not every database scan or CPU operation.

Scope: isolated Nudge package, its tests/docs and Nudge spec. No app
API, generated clients, staff UI/tools, production data or deployment changes.
Inference continues through TensorZero. If persisted contracts change, bump the
contract and document fresh-store requirements; do not add legacy fallback paths.

### Boundary tests

- More than 256 eligible rows: observed answerer states, evidence rows and fit rows
  all belong to one recorded sample of at most 256, even across refinement rounds.
- Unstable candidate: only the screening subset is answered; no full-history live
  prefetch occurs. Invalid proposals cause no candidate inference.
- Refreshing samples, late feedback and corrections preserve chronological evidence
  boundaries; replay alone cannot increase subsequent-validation support.
- Concurrency, retries, multiple heads and cache hits obey the same budget. Failure
  leaves the activated bank and policy intact, including across checked resume.
- Final-test rewards cannot change sample IDs, candidate choices or fitted policies.

## Plan

1. Re-read the listed implementation and tests. Trace every answering path and
   budget counter. Add the shared sampling boundary and tests first; merge serially
   with evidence/cutoff changes because they share core state.
2. Move candidate screening before expansion and remove live prefetch over history.
   Integrate the one budget owner at actual request dispatch, including transport
   retries and proposer size limits. Update persisted contracts and resume checks.
3. Update design/spec/runbook and diagnostics. Measure work and quality with recorded
   seeds/configuration; use a new evaluation split for improvement claims rather
   than optimizing against v8's repeatedly inspected final 200 prompts.

Parallelism pass: batch read-only code/test inspection and baseline checks. Sampling,
screening and budgeting touch shared state, so merge their edits serially. Independent
documentation/measurement review can overlap read-only work; no subagents required.

## Validation

- Run focused co-located Python tests for sampling, discovery, evidence, gateway and
  resume, then the complete Nudge suite and Ruff.
- Run pnpm run docs:check and pnpm run build; build the Python package.
- Exercise captured HTTP fixtures to verify actual retry/request accounting.
- Run a bounded live experiment when implementation is authorized: report calls,
  tokens, elapsed time, selected IDs and rejection work, not just retained questions.
  Verify fixed ceilings as the historical pool grows.
- Compare fresh/reference/no-question sets on identical sampled feedback and routing
  rules; greedy and posterior separately. Keep any all-outcome diagnostic separate.
  Reuse the existing matched evaluator only after verifying it respects the sampled
  training contract; its current all-row fit must not silently undo this policy.

## Ezra Humanization Review

N/A: isolated developer experiment; no staff-facing behavior or Ezra workflow changes.

## Shortcut Audit

- Planning only in this change; no bounded-policy implementation or fresh inference.
- Sampling only proposer context would not satisfy the outcome. Removing prefetch
  alone would not either: core hydration, evidence and fitting must share the bound.
- No claim of constant-time history access, statistically proven utility gain, or
  token-bounded prompts merely from the 256-row limit.
- One persistence path, one sample boundary and one authoritative update budget.

## Progress Log

- 2026-10-14: User approved this direction and requested a deferred implementation
  plan. Saved verified current behavior, boundaries, tests and unresolved numeric
  allowances. Existing completed Nudge work is being committed separately from any
  future implementation of this policy.

## Decisions

- Start with uniform sampling for evidence/fitting; avoid introducing biased
  prioritized fitting and weight corrections without measured need.
- Preserve historical storage and current question-bank semantics; bound work at
  the update boundary instead of adding a second replay store or pipeline.

## Follow-Ups

- Implement only in a later requested work session. Choose and document concrete
  request/token allowances before running the bounded experiment.
