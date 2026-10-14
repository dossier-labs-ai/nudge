# Semantic routing factors for the existing 1,000 prompts

Status: complete. All 1,000 annotated; semantic analysis and exact model
assignments saved. Owned local gateway stopped. No online learner changes.

User: "Ok, please do that" in response to finding semantic properties (task,
subject, reasoning operations, constraint complexity, required precision) that
explain model performance, instead of literal word checks.

Outcome: label all 1,000 public prompts with a fixed semantic rubric using
GLM-5.2 through a local TensorZero gateway, then report measurable routing
differences and combinations against their recorded outcomes. Classifier inputs
contain prompt text only; no model outcomes, family IDs, or sample IDs.

Scope: standalone experimental code, tests, schema, local gateway and retained
analysis artifacts. Serverless GLM inference uses the existing Secret Manager
reference in memory. No Rune activation, cloud deployment, production data,
or changes to the Nudge learner. Stop the owned local gateway after the run.

Read: architecture, harness, Nudge spec, RouterBench runbook, live Gateway,
earlier factor audit and tests. Existing unrelated edits preserved.

Execution: define semantic rubric and strict validation; batch outcome-blind
annotations with bounded calls, concurrency and resumable cache; evaluate
individual factors and combinations. Report both unrestricted sample grouping
and interpretable routing trees with minimum group support. The latter prevents
the earlier singleton lookup from masquerading as a semantic explanation.
Also measure cross-validated routing as a secondary check of model preferences.

Parallelism: source/config discovery and baseline tests batched; one module
owns annotation/analysis. Run tests while real annotation is in flight, then
merge findings into documentation serially. No temporary scripts.

Validation: payload/outcome separation, categorical validation, tree utility
objective and minimum leaf support, local HTTP fixture, full package suite,
Ruff/docs checks, real complete annotation coverage and saved analysis.

Shortcut audit: semantic labels are model judgments, not regex proxies; never
assert a ceiling is reached unless metrics show it. No arbitrary model-name
target classification: select models by full utility outcomes, preserving
ties. Complexity and group sizes accompany any sample fit. Not a proof of a
globally best semantic vocabulary. No runtime or application shipping.

Results: all 16 semantic dimensions attain 88.275% quality / 0.87634671916
utility at USD 0.640328084 historical downstream cost, with 516 cells (350
singletons). Seven dimensions support a nine-rule tree with at least 22 prompts
per leaf: 81.05% quality / 0.791792478 utility. Its out-of-fold utility is
0.772484208. These results do not establish a compact semantic route to the
full oracle. Documentation records all factor names and concrete model switches.

Execution evidence: 127 real GLM serverless calls, no transport failures, two
validation retries for the final batch's wrong annotation count. Recovery
reused all successful cached work. Classifier inputs exclude outcomes/metadata.
Actual saved-policy replay verifies combination metrics and leaf support.
Package tests, Ruff and documentation checks are the final validation ladder.
