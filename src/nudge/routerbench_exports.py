"""Atomic, live utility artifacts, independent of provider inference."""

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path


def atomic_text(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(value)
    pending.replace(path)


def export_utility(records, directory, *, stage, penalty, window=200, error=None):
    from .routerbench_report import online_curve, plot

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    curve = online_curve(records, window)
    for point in curve:
        point['gain'] = point['nudge'] - point['rune']
    rows = [r for r in records if r['partition'] == 'train']
    latest = curve[-1] if curve else None
    status = {
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'stage': stage, 'error': error, 'prompts': len(rows),
        'policy_version': rows[-1]['version'] if rows else None,
        'penalty_per_usd': penalty, 'window': window, 'latest': latest,
        'mean_utility': {p: sum(r[p]['utility'] for r in rows) / len(rows)
                         for p in ('nudge', 'rune')} if rows else {},
        'metric': 'Expected recorded utility, scored before chosen-action feedback; excludes router overhead',
    }
    output = io.StringIO()
    columns = ['prompts', 'window_count', 'rune', 'nudge', 'gain', 'cumulative_gain']
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(curve)
    atomic_text(directory / 'utility.csv', output.getvalue())
    atomic_text(directory / 'utility.json', json.dumps({'status': status, 'curve': curve}, indent=2))
    if curve:
        pending = directory / 'utility.pending.png'
        plot(records, {}, pending)
        pending.replace(directory / 'utility.png')
    atomic_text(directory / 'status.json', json.dumps(status, indent=2))
    return status
