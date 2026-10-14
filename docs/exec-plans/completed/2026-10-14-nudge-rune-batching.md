# Plan: Batch Rune feature questions

Status: Complete
Owner: Agent
Last updated: 2026-10-14

## Goal and done criteria

Send sixteen discovered feature questions in one Rune request, preserve baseline
isolation and typed decoding, test the request path, commit and push.

## Context and boundaries

Read ARCHITECTURE.md, docs/README.md, docs/AGENTIC_HARNESS.md,
docs/product-specs/index.md, specs/nudge-v0-spec.md, adapter docs, Rune bridge,
live adapter and existing HTTP tests. Only the experimental Python adapter,
tests and documentation change. No deployment, production operation, API schema
or staff workflow changes. Unrelated BEV workspace changes remain excluded.

## Plan and parallelism

Batch repository reads and start package baseline checks independently. Edit
adapter and its HTTP regression test serially. Version the answerer identity
to require fresh stores/refits. Run package tests and required root build,
review the diff, then commit and push only this change.

## Policy and mechanism contract

Baseline probabilities remain independent of the feature bank. Feature requests
carry one state and up to 32 Choice heads; missing or extra response heads fail.
Empty banks issue no feature request. Cache whole requests at the gateway.
Per-question discovery caches remain unchanged. Batch membership can affect
soft probabilities; this does not promise mathematical invariance or speedup.

## Validation

HTTP fixture exercises real Gateway and RuneAnswerer: sixteen mixed feature
heads in one request, typed decoding, baseline isolation, repeated-call caching,
frozen baselines and request limit. Run full Python suite and root build.

## Ezra Humanization Review

N/A: isolated developer experiment without staff-facing behavior.

## Shortcut Audit

No provider bypass, temporary scripts, production mutation or local-layer mocks
added. Live latency/GPU performance is unmeasured. Existing stores are preserved;
new identity rejects silently reopening old fitted contracts.

## Progress

- System Python baseline could not import NumPy; switched to package .venv.
- Batched adapter and HTTP regression coverage implemented.
- Fixed the HTTP fixture's missing-state handling for proposer requests.
- All 119 Python tests pass; changed Python files pass Ruff and diff checks.
- Full `pnpm run build` passes: docs, lint, typecheck, 4,593 tests and package
  builds. The first run's calendar timeout did not recur on rerun.
- No live inference or deployment performed; performance remains unmeasured.
