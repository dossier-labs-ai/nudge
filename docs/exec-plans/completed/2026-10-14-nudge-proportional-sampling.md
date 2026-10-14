# Plan: Proportional RouterBench preparation

Status: Complete — package checks pass; full build has test failures
Owner: Agent
Last updated: 2026-10-14

## Goal

Implement the user's instruction to do proportional sampling for the proposed
10k comparison; inspect existing cap, batching and GLM recovery mechanisms.

## Done Criteria

- Deterministic proportional sampling without replacement, configurable total
  and held-out counts, tests and runbook updates.
- Report verified systems limits without launching the costly experiment.

## Context

Read ARCHITECTURE.md, docs/README.md, docs/AGENTIC_HARNESS.md,
docs/product-specs/index.md, specs/nudge-v0-spec.md and Nudge's RouterBench
runbook, learner, discovery, adapter, evidence and sampling code.

## Policy / Mechanism Boundary

User authorized proportional sampling and adjusting the fit cap. Use existing
cap configuration; do not invent a new learning algorithm or select a cap from
test outcomes. Largest-remainder family allocation follows eligible population
sizes. Preserve historical artifacts. No provider, production or deployment work.

## Plan

1. Batch independent code and test reads; inspect existing mechanisms.
2. Update sampler, CLI, focused tests and runbook serially.
3. Run package tests and repository checks; report evidence and limits.

## Validation

Python unit tests, Ruff, docs checks and repository build. Fixtures exercise
unequal family populations, exhaustion, exact sizes and reproducibility.

## Ezra Humanization Review

N/A: isolated developer experiment with no staff workflow.

## Shortcut Audit

No inference launched or utility improvement claimed. Multiple question heads
per prompt are batched; cross-row discovery remains sequential. GLM retry
recovery is implemented but oversized context is unresolved.

## Progress Log

- 2026-10-14: Implemented sampler and CLI flags. All 122 Python tests pass;
  Ruff and git diff checks pass. Full build passed docs, lint (260 warnings)
  and typechecking, then stopped with 4 failed / 4,579 passed tests and three
  worker errors, including worker-start timeouts. Package builds were not reached.
  No baseline comparison establishes whether those failures predate this change.
  No live experiment was launched. Training cap remains explicitly configurable;
  no new cap default or discovery policy was selected.
