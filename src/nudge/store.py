"""Local SQLite persistence. No connection is held across model inference."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

from .types import canonical
from .training_replay import priority, select


class ConcurrentLearnError(RuntimeError):
    """Another learner activated first; retry with a fresh snapshot."""


SCHEMA = """
CREATE TABLE IF NOT EXISTS choices (
 name TEXT PRIMARY KEY, contract TEXT NOT NULL, live_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
 choice TEXT NOT NULL, version INTEGER NOT NULL, payload TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
 PRIMARY KEY(choice, version)
);
CREATE TABLE IF NOT EXISTS decisions (
 id TEXT PRIMARY KEY, choice TEXT NOT NULL, version INTEGER NOT NULL,
 split TEXT NOT NULL CHECK(split IN ('train','selection')), state TEXT NOT NULL,
 payload TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
 FOREIGN KEY(choice, version) REFERENCES snapshots(choice, version)
);
CREATE INDEX IF NOT EXISTS decisions_choice ON decisions(choice);
CREATE TABLE IF NOT EXISTS feedback (
 sequence INTEGER PRIMARY KEY AUTOINCREMENT,
 decision_id TEXT NOT NULL REFERENCES decisions(id), fields TEXT NOT NULL,
 received_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS feedback_latest ON feedback(decision_id, sequence DESC);
CREATE TABLE IF NOT EXISTS feature_answers (
 decision_id TEXT NOT NULL REFERENCES decisions(id), question_key TEXT NOT NULL,
 value TEXT NOT NULL, PRIMARY KEY(decision_id, question_key)
);
CREATE TABLE IF NOT EXISTS learning_runs (
 id TEXT PRIMARY KEY, choice TEXT NOT NULL, base_version INTEGER NOT NULL,
 feedback_cutoff INTEGER NOT NULL, status TEXT NOT NULL, details TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS proposals (
 sequence INTEGER PRIMARY KEY AUTOINCREMENT,
 run_id TEXT NOT NULL REFERENCES learning_runs(id), round INTEGER NOT NULL,
 proposer TEXT NOT NULL, request TEXT NOT NULL, response TEXT NOT NULL,
 results TEXT NOT NULL, choice TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS proposals_choice_sequence ON proposals(choice, sequence DESC);
CREATE TABLE IF NOT EXISTS replay_meta (
 choice TEXT PRIMARY KEY, feedback_count INTEGER NOT NULL DEFAULT 0,
 cutoff INTEGER NOT NULL DEFAULT 0, decision_boundary INTEGER NOT NULL DEFAULT 0,
 capacity INTEGER NOT NULL, actions TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS latest_feedback (
 decision_id TEXT PRIMARY KEY REFERENCES decisions(id), fields TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS training_replay (
 choice TEXT NOT NULL, lane TEXT NOT NULL, decision_id TEXT NOT NULL REFERENCES decisions(id),
 priority REAL NOT NULL, decision_order INTEGER NOT NULL, PRIMARY KEY(choice, lane, decision_id)
);
CREATE INDEX IF NOT EXISTS replay_priority ON training_replay(choice,lane,priority,decision_order DESC,decision_id);
CREATE TABLE IF NOT EXISTS pending (
 choice TEXT NOT NULL, proposal TEXT NOT NULL, objective TEXT NOT NULL,
 result TEXT NOT NULL, sequence INTEGER NOT NULL, PRIMARY KEY(choice,proposal)
);
CREATE INDEX IF NOT EXISTS pending_choice_sequence ON pending(choice,sequence DESC);
"""


class Store:
    def __init__(self, path: str | Path):
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(str(path), timeout=10, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 2):
            self.connection.close()
            raise ValueError(f"unsupported Nudge schema version {version}; bounded training requires "
                             "a fresh store; preserve the old store for auditing and refit")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(SCHEMA)
        self.connection.execute("PRAGMA user_version=2")
        self.connection.commit()

    @contextmanager
    def transaction(self):
        with self.lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                yield self.connection
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise

    def initialize(self, choice: str, contract: dict, initial: dict) -> None:
        with self.transaction() as db:
            row = db.execute("SELECT contract FROM choices WHERE name=?", (choice,)).fetchone()
            if row:
                if row["contract"] != canonical(contract):
                    raise ValueError(
                        "incompatible choice/model/expert configuration; use a new choice name or store"
                    )
                return
            db.execute(
                "INSERT INTO snapshots(choice,version,payload) VALUES(?,?,?)",
                (choice, 0, canonical(initial)),
            )
            db.execute("INSERT INTO choices VALUES(?,?,?)", (choice, canonical(contract), 0))
            db.execute('INSERT INTO replay_meta(choice,capacity,actions) VALUES(?,?,?)',
                       (choice, contract['expert']['max_training_rows'],
                        canonical(list(contract['actions']))))

    def live(self, choice: str) -> dict:
        with self.lock:
            row = self.connection.execute(
                "SELECT s.payload FROM snapshots s JOIN choices c "
                "ON s.choice=c.name AND s.version=c.live_version WHERE c.name=?",
                (choice,),
            ).fetchone()
            return json.loads(row["payload"])

    def decision(self, choice: str, identifier: str) -> dict:
        with self.lock:
            row = self.connection.execute(
                "SELECT payload FROM decisions WHERE choice=? AND id=?", (choice, identifier)
            ).fetchone()
            if row is None:
                raise KeyError(identifier)
            return json.loads(row["payload"])

    def insert_decision(
        self,
        choice: str,
        state: str,
        split: str,
        payload: dict,
        features: dict[str, float],
        created_at: str,
    ) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT INTO decisions(id,choice,version,split,state,payload,created_at) VALUES(?,?,?,?,?,?,?)",
                (
                    payload["id"],
                    choice,
                    payload["version"],
                    split,
                    state,
                    canonical(payload),
                    created_at,
                ),
            )
            db.executemany(
                "INSERT INTO feature_answers VALUES(?,?,?)",
                [(payload["id"], key, canonical(value)) for key, value in features.items()],
            )
            db.execute('UPDATE replay_meta SET decision_boundary='
                       '(SELECT rowid FROM decisions WHERE id=?) WHERE choice=?', (payload['id'], choice))

    def feedback(self, choice: str, identifier: str, fields: dict) -> None:
        with self.transaction() as db:
            decision = db.execute(
                "SELECT rowid AS decision_order,payload,created_at,version FROM decisions WHERE choice=? AND id=?",
                (choice, identifier)).fetchone()
            if decision is None:
                raise KeyError(identifier)
            first = db.execute('SELECT 1 FROM latest_feedback WHERE decision_id=?',
                               (identifier,)).fetchone() is None
            cursor = db.execute(
                "INSERT INTO feedback(decision_id,fields) VALUES(?,?)",
                (identifier, canonical(fields)),
            )
            db.execute('INSERT INTO latest_feedback VALUES(?,?) ON CONFLICT(decision_id) '
                       'DO UPDATE SET fields=excluded.fields', (identifier, canonical(fields)))
            db.execute('UPDATE replay_meta SET cutoff=?,feedback_count=feedback_count+? WHERE choice=?',
                       (cursor.lastrowid, int(first), choice))
            meta = db.execute('SELECT * FROM replay_meta WHERE choice=?', (choice,)).fetchone()
            capacity = meta['capacity']
            payload = json.loads(decision['payload'])
            snapshot = json.loads(db.execute('SELECT payload FROM snapshots WHERE choice=? AND version=?',
                                            (choice, decision['version'])).fetchone()[0])
            reward = sum(fields[k] * v for k, v in snapshot['choice']['reward'].items())
            surprise = min(10.0, abs(reward - payload['scores'][payload['action']]) /
                           max(snapshot['scale'], 1e-12))
            lanes = [('recent', -datetime.fromisoformat(decision['created_at']).timestamp(), capacity),
                     ('uniform', priority(identifier, 'uniform'), capacity),
                     ('action:' + payload['action'], priority(identifier, 'informative', 1 + surprise),
                      max(1, capacity // len(json.loads(meta['actions']))))]
            for lane, score, count in lanes:
                db.execute('INSERT INTO training_replay VALUES(?,?,?,?,?) '
                           'ON CONFLICT(choice,lane,decision_id) DO UPDATE SET priority=excluded.priority',
                           (choice, lane, identifier, score, decision['decision_order']))
                db.execute('DELETE FROM training_replay WHERE choice=? AND lane=? AND decision_id IN '
                           '(SELECT decision_id FROM training_replay WHERE choice=? AND lane=? '
                           'ORDER BY priority,decision_order DESC,decision_id LIMIT -1 OFFSET ?)',
                           (choice, lane, choice, lane, count))

    def training_rows(self, choice):
        """Read only bounded membership and indexed decision/feedback rows."""
        with self.transaction() as db:
            meta = dict(db.execute('SELECT * FROM replay_meta WHERE choice=?', (choice,)).fetchone())
            candidates = [dict(r) for r in db.execute(
                'SELECT decision_id AS id,lane,priority,decision_order FROM training_replay WHERE choice=?',
                (choice,))]
            rows = []
            for member in select(candidates, meta['capacity']):
                row = db.execute('SELECT d.rowid AS decision_order,d.state,d.split,d.created_at,d.payload,'
                                 'f.fields FROM decisions d JOIN latest_feedback f ON f.decision_id=d.id '
                                 'WHERE d.id=?', (member['id'],)).fetchone()
                rows.append({**json.loads(row['payload']), 'state': row['state'], 'split': row['split'],
                             'created_at': row['created_at'], 'decision_order': row['decision_order'],
                             'fields': json.loads(row['fields'])})
            return meta, rows

    def rows(self, choice: str) -> tuple[int, list[dict]]:
        # One read transaction provides an atomic cutoff + decision/feedback view.
        with self.lock:
            db = self.connection
            db.execute("BEGIN")
            try:
                cutoff = db.execute("SELECT COALESCE(MAX(sequence),0) FROM feedback").fetchone()[0]
                rows = db.execute(
                    """
                    SELECT d.rowid AS decision_order,d.id,d.state,d.split,d.payload,d.created_at,f.fields FROM decisions d
                    LEFT JOIN feedback f ON f.sequence=(
                      SELECT MAX(f2.sequence) FROM feedback f2
                      WHERE f2.decision_id=d.id AND f2.sequence<=?)
                    WHERE d.choice=? ORDER BY d.rowid DESC
                """,
                    (cutoff, choice),
                ).fetchall()
                result = [
                    {
                        **json.loads(row["payload"]),
                        "state": row["state"],
                        "split": row["split"],
                        "created_at": row["created_at"],
                        "decision_order": row['decision_order'],
                        "fields": json.loads(row["fields"]) if row["fields"] else None,
                    }
                    for row in rows
                ]
                db.commit()
                return cutoff, result
            except BaseException:
                db.rollback()
                raise

    def cached(self, identifier: str, key: str) -> object | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT value FROM feature_answers WHERE decision_id=? AND question_key=?",
                (identifier, key),
            ).fetchone()
            if not row:
                return None
            value = row[0]
            if isinstance(value, str):
                return json.loads(value)
            return value

    def cache(self, identifier: str, answers: dict[str, object]) -> None:
        with self.transaction() as db:
            db.executemany(
                "INSERT OR IGNORE INTO feature_answers VALUES(?,?,?)",
                [(identifier, key, canonical(value)) for key, value in answers.items()],
            )

    def start_run(self, identifier: str, choice: str, base: int, cutoff: int) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT INTO learning_runs VALUES(?,?,?,?,?,?)",
                (identifier, choice, base, cutoff, "running", "{}"),
            )

    def proposal(
        self,
        run: str,
        round_number: int,
        proposer: str,
        request: dict,
        response: list,
        results: list,
    ) -> int:
        with self.transaction() as db:
            cursor = db.execute(
                "INSERT INTO proposals(run_id,round,proposer,request,response,results,choice) VALUES(?,?,?,?,?,?,?)",
                (
                    run,
                    round_number,
                    proposer,
                    canonical(request),
                    canonical(response),
                    canonical(results),
                    request['choice']['name'],
                ),
            )
            return cursor.lastrowid

    def finish_proposal(self, sequence: int, results: list) -> None:
        with self.transaction() as db:
            db.execute(
                "UPDATE proposals SET results=? WHERE sequence=?", (canonical(results), sequence)
            )
            record = db.execute('SELECT choice,request FROM proposals WHERE sequence=?',
                                (sequence,)).fetchone()
            objective = canonical(json.loads(record['request'])['choice'])
            for result in results:
                key = canonical(result['proposal'])
                if result['status'].startswith('deferred_'):
                    db.execute('INSERT OR REPLACE INTO pending VALUES(?,?,?,?,?)',
                               (record['choice'], key, objective, canonical(result), sequence))
                else:
                    db.execute('DELETE FROM pending WHERE choice=? AND proposal=?',
                               (record['choice'], key))
            db.execute('DELETE FROM pending WHERE choice=? AND proposal IN '
                       '(SELECT proposal FROM pending WHERE choice=? ORDER BY sequence DESC,proposal '
                       'LIMIT -1 OFFSET 64)', (record['choice'], record['choice']))

    def proposal_history(self, choice: dict, limit: int = 16) -> list[dict]:
        """Recent diagnostics only: never replay stored selection rows or nested history."""
        with self.lock:
            records = self.connection.execute(
                "SELECT p.sequence,p.run_id,p.round,p.request,p.results,r.status "
                "FROM proposals p JOIN learning_runs r ON p.run_id=r.id "
                "WHERE p.choice=? ORDER BY p.sequence DESC LIMIT ?",
                (choice['name'], limit),
            ).fetchall()
        return [
            {'sequence': r['sequence'], 'run': r['run_id'], 'round': r['round'],
             'run_status': r['status'], 'results': json.loads(r['results']),
             'selection_objective': json.loads(r['request']).get(
                 'selection_objective', 'chosen-action time-weighted held-out reward MSE')}
            for r in reversed(records)
            if json.loads(r['request'])['choice'] == choice and json.loads(r['results'])
        ]

    def fail_run(self, identifier: str, error: str) -> None:
        with self.transaction() as db:
            db.execute(
                "UPDATE learning_runs SET status='failed',details=? WHERE id=?",
                (canonical({"error_type": error}), identifier),
            )

    def pending_proposals(self, choice):
        with self.lock:
            return [json.loads(row['result']) for row in self.connection.execute(
                'SELECT result FROM pending WHERE choice=? AND objective=? ORDER BY sequence',
                (choice['name'], canonical(choice)))]

    def activate(self, choice: str, run: str, base: int, snapshot: dict) -> None:
        with self.transaction() as db:
            current = db.execute(
                "SELECT live_version FROM choices WHERE name=?", (choice,)
            ).fetchone()[0]
            if current != base:
                raise ConcurrentLearnError("live policy changed during learning; retry learn()")
            db.execute(
                "INSERT INTO snapshots(choice,version,payload) VALUES(?,?,?)",
                (choice, snapshot["version"], canonical(snapshot)),
            )
            db.execute(
                "UPDATE choices SET live_version=? WHERE name=?", (snapshot["version"], choice)
            )
            db.execute(
                "UPDATE learning_runs SET status='activated',details=? WHERE id=?",
                (canonical({"version": snapshot["version"]}), run),
            )

    def close(self) -> None:
        with self.lock:
            self.connection.close()
