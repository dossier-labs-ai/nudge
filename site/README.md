# nudge.dossierlabs.ai

Static site for Nudge, served by Firebase Hosting. Everything under `site/`
is published as-is; there is no build step.

- Hosting site id: `nudge-dossierlabs` (`site` in `../firebase.json`).
- Config: `../firebase.json` (`public: site`, clean URLs).
- Deploy: `firebase deploy --only hosting --project <gcp-project>` from the
  repository root. Preview URL: `https://nudge-dossierlabs.web.app`.
- Custom domain `nudge.dossierlabs.ai` is attached in the Firebase console;
  DNS for `dossierlabs.ai` lives in Squarespace.

`index.html` includes the fixed 5,000-prompt RouterBench result. The page source
is excluded from the Python distribution (see `pyproject.toml` `sdist.include`).

## Benchmark chart

The desktop/mobile SVG and PNG assets under `assets/routerbench/` plot every
prompt from 1 through 5,000 on a 0–5,000 x-axis. The y-axis is the cumulative
average quality-minus-cost score, not accuracy or the last 200 prompts alone.
`source.json` preserves the aggregate input; `scores.csv` contains the plotted
points. No benchmark prompts, credentials or inference logs are published.

Rebuild the exact chart from its frozen public input:

```sh
uv run --extra operations python -m nudge.routerbench_website \
  --input site/assets/routerbench/source.json \
  --out-dir site/assets/routerbench --count 5000
```

The renderer recovers cumulative sums from the exported trailing-window means
and verifies them against the exported cumulative advantage. It never samples,
fits, smooths, or extrapolates scores. At 5,000 prompts, cumulative means are
0.6671034989734627 (Nudge) and 0.5809910636182534 (Rune), matching the live export.
The website's headline is their relative difference, rounded to 14.8%.
