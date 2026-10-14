# Nudge oracle discriminating factors

User outcome: identify discriminating factors for the existing 1,000 prompts
that approach the measured 91.8% quality / 0.91569349056 utility oracle.

Scope: offline analysis of the complete prepared outcome matrix. Use all 1,000
rows for this explicitly requested sample analysis. No inference, deployments,
changes to the online learner, or customer operations.

Status: complete. All 63 package tests pass, including independent replay of
the full saved policy from prompt factors. Ruff, docs check, and diff whitespace
check pass. No application behavior was shipped.

Read: architecture, agentic harness, Nudge spec, experiment log, RouterBench
ingestion, feature-search and predictor tools. Existing changes are preserved.

Plan: add a reproducible full-sample factor-combination audit, with tests;
compare readable structural/topic indicators with literal text indicators;
report actual selected factors, their combined quality/cost, and remaining gap.
Selection directly maximizes routing utility, not classification accuracy of
one arbitrarily chosen oracle model. Persist aggregate evidence and exact rules.

Parallelism: batched independent source reads and baseline tests; implement the
single owning analysis module serially. No temporary scripts. Validate focused
tests, package suite, Ruff, docs check, and a real offline artifact run.

Done: report the strongest measured factor set and its scope; do not claim a
globally minimal set from greedy search or equate sample fitting with prediction
on new prompts. Do not invent a semantic explanation for literal word splits.

Shortcut audit: metadata IDs and family labels excluded from extracted factors;
outcomes used only to select the full-sample diagnostic policy. No raw prompt
text in published documentation. No production or application behavior changed.

Results: 40 retained factors reproduce all oracle metrics with 987 routing
cells (976 singletons). A 24-factor prefix reaches the quality ceiling; the
37-factor structural-only policy reaches 89.4%. Exact artifact policies are
preserved in the ignored benchmark-results directory. Clearly distinguish this
sample reconstruction from a reusable semantic explanation.
