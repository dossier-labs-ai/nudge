"""Proposal sampling within the bounded training replay set.

The caller supplies bounded rows; persistent training membership lives in SQLite.
"""

from collections import Counter
import hashlib

import numpy as np

from .types import canonical


class SamplingReplayBuffer:
    def __init__(self, rows, *, questions, round_number=1, training_only=True):
        # Caller supplies decision-time order from weight_rows (newest first).
        self.rows = [r for r in rows if (not training_only or r['split'] == 'train') and r['weight'] > 0]
        if len({r['id'] for r in self.rows}) != len(self.rows):
            raise ValueError('replay requires unique decision IDs')
        seed = canonical([[r['id'] for r in self.rows],
                          [q.to_dict() for q in questions], round_number])
        self.seed = int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16)
        self.questions = questions

    def sample(self, limit):
        if type(limit) is not int or limit < 1:
            raise ValueError('replay sample limit must be positive')
        n = min(limit, len(self.rows))
        if n == 0:
            return []
        # Small buffers are replayed once in full, preserving recency order.
        if n == len(self.rows):
            return [{**r, 'replay_source': 'all'} for r in self.rows]
        groups = []
        for row in self.rows:
            signature = []
            for question in self.questions:
                value = row['features'][question.name]
                bucket = (int(np.argmax(question.feature_vector(value)))
                          if question.kind == 'categorical' else
                          min(3, int(question.normalize(value) * 4)))
                signature.append((question.name, bucket))
            groups.append(signature)
        actions = Counter(r['action'] for r in self.rows)
        counts = Counter(group for signature in groups for group in signature)
        residuals = np.abs([r['residual'] for r in self.rows])
        if not np.isfinite(residuals).all():
            raise ValueError('replay residuals must be finite')
        # Rank surprise so reward units and single extreme outcomes cannot dominate.
        surprise = np.searchsorted(np.sort(residuals), residuals, side='left') / len(self.rows)
        priorities = np.array([
            1 + surprise[i] + len(self.rows) / actions[row['action']] +
            (sum(len(self.rows) / counts[g] for g in groups[i]) / len(groups[i])
             if groups[i] else 0)
            for i, row in enumerate(self.rows)
        ])
        recent = (n + 2) // 3
        representative = (n + 1) // 3
        rng = np.random.default_rng(self.seed)
        selected = [(i, 'recent') for i in range(recent)]
        remaining = np.arange(recent, len(self.rows))
        uniform = rng.choice(remaining, size=representative, replace=False)
        selected.extend((int(i), 'representative') for i in uniform)
        remaining = np.setdiff1d(remaining, uniform)
        weights = priorities[remaining]
        informative = rng.choice(remaining, size=n - len(selected), replace=False,
                                 p=weights / weights.sum())
        selected.extend((int(i), 'informative') for i in informative)
        return [{**self.rows[i], 'replay_source': source} for i, source in selected]
