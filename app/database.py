import json
import os
import sqlite3
from pathlib import Path
from typing import Any


def database_path() -> str:
    return os.getenv("LEAD_DB_PATH", str(Path(__file__).with_name("leads.sqlite3")))


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(database_path(), timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize() -> None:
    with connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lead_json TEXT NOT NULL,
            score REAL NOT NULL CHECK(score >= 0 AND score <= 100),
            priority TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            model_version TEXT NOT NULL
        )""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at)")


def save_prediction(lead: dict[str, Any], score: float, priority: str, model_version: str) -> int:
    with connect() as db:
        cursor = db.execute(
            "INSERT INTO predictions (lead_json, score, priority, model_version) VALUES (?, ?, ?, ?)",
            (json.dumps(lead, sort_keys=True), score, priority, model_version),
        )
        return int(cursor.lastrowid)


def recent_predictions(limit: int = 100) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            "SELECT id, lead_json, score, priority, created_at, model_version "
            "FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [{**dict(row), "lead": json.loads(row["lead_json"])} for row in rows]
