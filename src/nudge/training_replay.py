"""Bounded replay selection; candidate membership is maintained on feedback."""

import hashlib
import math


def priority(identifier, lane, weight=1.0):
    # Stable pseudorandom exponential race: smaller values win. Equal weights
    # give a uniform bottom-k reservoir without a scan or restart-dependent RNG.
    value = int(hashlib.sha256(f'{lane}:{identifier}'.encode()).hexdigest()[:13], 16)
    uniform = (value + 1) / (2**52 + 1)
    return -math.log(uniform) / weight


def select(candidates, limit):
    """Unique mixture from bounded pools, filling overlap from remaining members."""
    by_id = {r['id']: r for r in candidates}
    if len(by_id) <= limit:
        return list(by_id.values())
    lanes = {}
    for row in candidates:
        lanes.setdefault(row['lane'], []).append(row)
    recent = sorted(lanes.get('recent', []), key=lambda r: (r['priority'], -r.get('decision_order', 0)))
    uniform = sorted(lanes.get('uniform', []), key=lambda r: r['priority'])
    # Round robin action strata prevents common actions consuming informative slots.
    actions = [sorted(rows, key=lambda r: r['priority']) for lane, rows in sorted(lanes.items())
               if lane.startswith('action:')]
    informative = [rows[i] for i in range(max(map(len, actions), default=0))
                   for rows in actions if i < len(rows)]
    chosen = {}
    for pool, count in ((recent, (limit + 2) // 3),
                        (uniform, (limit + 1) // 3),
                        (informative, limit // 3)):
        added = 0
        for row in pool:
            if row['id'] not in chosen:
                chosen[row['id']] = row
                added += 1
                if added == count:
                    break
    for row in uniform + informative + recent:
        if len(chosen) >= limit:
            break
        chosen.setdefault(row['id'], row)
    return list(chosen.values())
