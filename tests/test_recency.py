import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from nudge import Answer, Choice, Nudge
from nudge.discovery import examples_for
from nudge.math import fit, mse
from nudge.recency import weight_rows


class RecencyTests(unittest.TestCase):
    def row(self, day, reward=1, weight=1):
        return {
            "id": str(day),
            "split": "train",
            "version": 0,
            "propensity": 0.5,
            "created_at": (
                datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)
            ).isoformat(),
            "state": str(day),
            "action": "a",
            "jev": {"a": 0.5},
            "features": {},
            "fields": {"reward": reward},
            "reward": reward,
            "weight": weight,
        }

    def test_newest_first_exact_half_lives_and_same_time(self):
        rows, reference = weight_rows(
            [self.row(0), self.row(14), self.row(7), self.row(14)], 7 * 86400
        )
        self.assertEqual([r["state"] for r in rows], ["14", "14", "7", "0"])
        self.assertEqual([r["weight"] for r in rows], [1, 1, 0.5, 0.25])
        self.assertEqual(reference, "2026-01-15T00:00:00.000000+00:00")
        uniform, _ = weight_rows(rows, None)
        self.assertEqual([r["weight"] for r in uniform], [1, 1, 1, 1])

    def test_elapsed_time_not_row_position(self):
        rows, _ = weight_rows([self.row(0), self.row(1), self.row(30)], 86400)
        self.assertEqual([r["weight"] for r in rows], [1, 2**-29, 2**-30])
        rows, _ = weight_rows([self.row(0), self.row(10000)], 1)
        self.assertEqual([r["weight"] for r in rows], [1, 0])

    def test_weighted_posterior_and_uncertainty(self):
        rows = [self.row(0, reward=1, weight=0.25), self.row(1, reward=-1, weight=1)]
        critic = fit(rows, ["a"], [], 1)["a"]
        self.assertAlmostEqual(critic["mean"][0], -3 / 69)
        self.assertEqual(critic["precision"][0][0], 69)
        self.assertEqual(critic["n"], 1.25)
        unweighted = fit([{**r, "weight": 1} for r in rows], ["a"], [], 1)["a"]
        self.assertGreater(
            np.linalg.inv(critic["precision"])[0, 0], np.linalg.inv(unweighted["precision"])[0, 0]
        )

    def test_selection_error_uses_weights(self):
        critics = fit([], ["a"], [], 1)
        self.assertAlmostEqual(mse(critics, [self.row(0, 2, 0.25), self.row(1, 0, 1)], []), 0.8)

    def test_small_replay_buffer_retains_recency_order_and_weights(self):
        rows, _ = weight_rows([self.row(0, 100), self.row(1, 1)], 86400)
        examples = examples_for(rows, fit([], ["a"], [], 1), [], 2)
        self.assertEqual([r["state"] for r in examples], ["1", "0"])
        self.assertEqual([r["weight"] for r in examples], [1, 0.5])

    def test_invalid_half_lives_and_naive_dates(self):
        for invalid in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                weight_rows([], invalid)
        with self.assertRaises(ValueError):
            weight_rows([{**self.row(0), "created_at": "2026-01-01T00:00:00"}], 1)

    def test_late_corrected_feedback_does_not_refresh_decision_age(self):
        class Model:
            identity = "recency-test"

            def ask(self, state, questions):
                return Answer({"a": 1.0, "b": 0.0})

        choice = Choice("recency", "test", {"a": "A", "b": "B"}, {"reward": "Observed"})
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "store.db"
            with Nudge(choice, store=path, model=Model(), clock=lambda: now) as policy:
                old = policy.ask(state="old")
                now += timedelta(days=7)
                fresh = policy.ask(state="fresh")
                policy.feedback(fresh.id, reward=1)
                policy.feedback(old.id, reward=-1)
                policy.learn(discover_questions=False)
                self.assertEqual(sum(a['learned_weight'] for a in policy.report().actions.values()), 1.5)
                now += timedelta(days=100)
                policy.feedback(old.id, reward=-2)
                policy.learn(discover_questions=False)
                self.assertEqual(sum(a['learned_weight'] for a in policy.report().actions.values()), 1.5)
                with sqlite3.connect(path) as db:
                    snapshots = [
                        json.loads(r[0])
                        for r in db.execute(
                            "SELECT payload FROM snapshots WHERE version>0 ORDER BY version"
                        )
                    ]
                self.assertEqual(snapshots[0]["recency"], snapshots[1]["recency"])
                self.assertEqual(
                    snapshots[1]["recency"]["reference_time"],
                    fresh_time := "2026-01-08T00:00:00.000000+00:00",
                )
                self.assertEqual(fresh_time, snapshots[0]["recency"]["reference_time"])
                policy.learn(half_life_seconds=None, discover_questions=False)
                self.assertEqual(sum(a['learned_weight'] for a in policy.report().actions.values()), 2)
            with Nudge(choice, store=path, model=Model()) as reopened:
                self.assertEqual(reopened.decision(old.id), old)

    def test_missing_recency_metadata_preserves_snapshot_history(self):
        # Recency metadata alone is optional; the runtime policy contract must match.
        from test_core import ConstantAnswerer, choice

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.db"
            with Nudge(choice(), store=path, model=ConstantAnswerer()) as policy:
                d = policy.ask(state="0")
            with sqlite3.connect(path) as db:
                payload = json.loads(db.execute("SELECT payload FROM snapshots").fetchone()[0])
                payload.pop("recency")
                old = json.dumps(payload)
                db.execute("UPDATE snapshots SET payload=?", (old,))
            with Nudge(choice(), store=path, model=ConstantAnswerer()) as policy:
                policy.feedback(d.id, result=1)
                policy.learn()
                self.assertEqual(policy.report().actions[d.action]["learned_weight"], 1)
            with sqlite3.connect(path) as db:
                self.assertEqual(
                    db.execute("SELECT payload FROM snapshots WHERE version=0").fetchone()[0], old
                )


if __name__ == "__main__":
    unittest.main()
