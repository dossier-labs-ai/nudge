# Plan: Unrestricted decisions and known-question retention

Status: Completed implementation and measurement; retention result negative
Owner: Agent
Last updated: 2026-10-14

User approved removing Jev's veto, testing saved semantic questions before more
live discovery, supporting combinations and retaining uncertain candidates for
further experiments. Implement in the general Choice library. Jev probabilities
are features only, not support constraints. Use posterior probability of being
the highest-utility action plus a uniform exploration floor. Compute exactly the
same finite posterior ensemble at serving and evaluation, with no score clipping,
drift cap or delayed exploration. Cold-start prior explores every action.

Replace blanket per-action coverage veto with paired-change evidence diagnostics;
retain uncertain proposals in persistent history for reevaluation with new data.
Keep positive uncertainty-adjusted utility acceptance. Evaluate bundles alongside
individual questions. First measurement uses saved original 16 Rune heads, 800
train/200 final test; selection only within training. Report archived logged
chosen feedback and new-policy chosen-feedback replay separately, plus no-question
and fixed-all-question comparators. Do not tune on final test outcomes.

Breaking runtime policy: remove drift/sample-after configuration, require a fresh
named choice/store, retain archived bounded math only in historical analysis tools.
No live calls/deployment. Read core/math/utility/discovery/archive evaluators/tests.
Edits serial across these shared boundaries; baseline tests/build passed previously.
Test zero-Jev override, exact evaluator/serving agreement, bundles and uncertainty,
retention and leakage; run package suite/Ruff/full build and document measured result.
Ezra review N/A: standalone experiment. No synthetic paraphrase measurements.

## Outcome and evidence

Implemented shared unrestricted posterior policy, uniform support from cold start,
atomic question bundles, paired-change coverage diagnostics and persistent pending
hypotheses. Removed drift/sample-after settings and documented fresh-choice/store
migration. Action-name draw seeds prevent dictionary serialization from changing
serving/evaluation probabilities.

Measured original 16 saved Rune questions individually and together, using only
the internal training/selection partition for admission. On 200 final prompts:
archived feedback retained zero, utility 0.651992 (fixed-all 0.616917); new
exploration feedback retained zero, utility 0.653526 (fixed-all 0.611841).
With exploration, 16 candidate edits remain uncertain and one lacks sufficient
coverage. No test-score tuning, new model calls or invented answer-stability data.
Artifact: `benchmark-results/routerbench/unrestricted-retention.json`.
Reproduction and interpretation are in the package RouterBench runbook.

106 Python tests pass through the full `pnpm run build`, including real synthetic
question/bundle retention, pending persistence, baseline-zero override, exact
serving/evaluator agreement and final-test leakage checks. Ruff and `uv build`
pass. Existing repository lint/build warnings remain. No deployment or publication.

The implementation removes the identified policy restrictions, but the experiment
does not establish useful RouterBench question retention. This is an explicit
negative result, not a successful-learning claim. Pending hypotheses are available
to the next proposer round when additional feedback supplies new evidence.
