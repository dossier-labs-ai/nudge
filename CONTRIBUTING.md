# Contributing

Keep the core portable: Python 3.11+, NumPy, SQLite, and standard-library modules.
Backend-specific imports belong in optional adapters, never in the core.
Changes are MIT licensed, subject to the included LICENSE.

Run `uv sync --group dev`, `uv run python -m unittest discover -s tests -v`,
`uv run ruff check src tests examples`, and `uv build` from this directory.
This standalone repository runs the Python suite directly; no Node/Vitest
installation or parent application checkout is needed.

Test real SQLite and real math. Replace external model/network boundaries with
deterministic scorers or local HTTP servers. New inference integrations need
documented input/output contracts, malformed-response tests, bounded timeouts,
and explicit model revisions. Never add environment-based secret discovery.

For experiment changes, preserve seeds, action sampling probabilities, and
independent final evaluation. Report unsuccessful scenarios alongside successful
ones. An improvement in a reused selection score is not proof of generalization.
Regret comparisons must identify the comparator and distinguish expected from
realized rewards. Keep synthetic benchmarks distinct from real-model evidence.

Do not commit states, feedback databases, credentials, or customer data. Package
artifacts must not import the host application or include its private files.
Nothing in the build automatically publishes a release.
