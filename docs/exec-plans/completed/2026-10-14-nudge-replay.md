# Plan: Nudge discovery replay

Status: Complete
Owner: Agent
Last updated: 2026-10-14

## Goal and done criteria

User approved using the decision log as a replay store with deliberate sampling,
cached question answers and persistent discovery history: "fantastic, please do
that. In SmartChoices, we used a SamplingReplayBuffer that had these kind of
semantics." Implement recent/representative/informative proposal sampling,
without duplicating regression evidence or exposing selection examples.
Verify through real SQLite learn/reopen cycles, sampling tests and package tests.

## Context and scope

Read Nudge spec, design, discovery/core/store/types and harness. Existing store
already retains decisions, latest corrected outcomes, propensity, snapshot
versions, immutable splits, cached answers and proposal diagnostics. SmartChoices
source not found in checked local repositories; requested its location. The
specific mixture below is our implementation choice, not a verified reproduction.
No storage migration, inference service, deployment or scheduling required.
Existing MSE selection remains explicitly labeled; utility scoring is separate.

## Mechanism

Use an approximately equal recent/uniform/informative proposal sample without
replacement. Informative priority uses training residuals and rare action/answer
groups. Zero-weight and non-training rows are ineligible. Sampling never changes
critic fit rows, evidence weights or the independent selection partition.
Persist sampled IDs/reasons with existing proposal requests; supply bounded
prior proposal diagnostics for the same choice/reward to later rounds and runs.
Existing exact-question cache handles revisions and restart. Update proposer
instructions to reason about relative advantage and refine from measured history.

## Plan and validation

1. Add sampling and history integration; preserve fit/selection boundaries.
2. Test coverage, reproducibility, no duplicates/leakage, correction/restart and
   evidence counts using real store/learning paths.
3. Run package tests, Ruff and docs checks; document semantics and limitations.

Parallelism pass: independent discovery and baseline tests batched; edits serial
because sampler/discovery/context/store changes share integration boundaries.
Baseline: 80 Python tests pass. Ezra review: N/A, standalone Python experiment.
Shortcut audit: no live benefit claim without an experiment; no unobserved rewards
invented; no replay duplication or priority weighting in regression.

## Completion evidence

Implemented replay sampler, proposal-history read path, context metadata and v3
problem-space/refinement prompt. Existing storage/cache and fit paths retained.
85 Python tests pass, including a real 600-decision learn/reopen/correction cycle
whose evidence remains exactly 600. Sampling tests cover historical coverage,
rare actions/answer groups, uniqueness, deterministic rounds and held-out exclusion.
Ruff passes; `pnpm run build` passes docs, lint (260 warnings, no errors), typecheck,
unit tests and package builds. No live routing improvement was measured or claimed.
Source location question remains optional; no SmartChoices equivalence claim.
