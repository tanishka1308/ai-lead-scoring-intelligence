"""Database access layer for storing and querying lead predictions using SQLite."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
from lead_scoring.config import get_db_path

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id TEXT NOT NULL,
    industry TEXT,
    lead_source TEXT,
    job_level TEXT,
    lead_details TEXT,
    score REAL NOT NULL,
    priority TEXT NOT NULL CHECK(priority IN ('High', 'Medium', 'Low')),
    calibrated_probability REAL,
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _resolve_db_path(db_path: Optional[Path | str] = None) -> Path:
    if db_path is not None:
        p = Path(db_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return get_db_path()


def init_db(db_path: Optional[Path | str] = None) -> None:
    """Initialize SQLite database schema."""
    path = _resolve_db_path(db_path)
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()


def save_predictions(
    predictions: List[Dict[str, Any]], db_path: Optional[Path | str] = None
) -> int:
    """Save prediction records to SQLite database using parameterized queries.

    Returns:
        Number of rows inserted.
    """
    if not predictions:
        return 0

    path = _resolve_db_path(db_path)
    now_utc = datetime.now(timezone.utc).isoformat()

    insert_sql = """
    INSERT INTO predictions (
        lead_id, industry, lead_source, job_level, lead_details,
        score, priority, calibrated_probability, model_version, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    rows = []
    for pred in predictions:
        details = pred.get("details", {})
        lead_details_str = json.dumps(details) if isinstance(details, dict) else str(details)

        # Extract features for indexable columns if present
        industry = details.get("industry") if isinstance(details, dict) else None
        lead_source = details.get("lead_source") if isinstance(details, dict) else None
        job_level = details.get("job_level") if isinstance(details, dict) else None

        rows.append((
            str(pred["lead_id"]),
            str(industry) if industry else None,
            str(lead_source) if lead_source else None,
            str(job_level) if job_level else None,
            lead_details_str,
            float(pred["score"]),
            str(pred["priority"]),
            float(pred["calibrated_probability"]) if pred.get("calibrated_probability") is not None else None,
            str(pred.get("model_version", "v1.0")),
            pred.get("created_at", now_utc),
        ))

    conn = sqlite3.connect(path)
    try:
        cursor = conn.cursor()
        cursor.executemany(insert_sql, rows)
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def fetch_predictions(
    limit: int = 50,
    offset: int = 0,
    priority: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """Fetch paginated predictions with optional priority filter. Parameterized query."""
    path = _resolve_db_path(db_path)

    query = """
    SELECT id, lead_id, industry, lead_source, job_level, lead_details,
           score, priority, calibrated_probability, model_version, created_at
    FROM predictions
    """
    params: List[Any] = []

    if priority:
        query += " WHERE priority = ?"
        params.append(priority)

    query += " ORDER BY id DESC LIMIT ? OFFSET ?;"
    params.extend([max(1, limit), max(0, offset)])

    conn = sqlite3.connect(path)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()

        results: List[Dict[str, Any]] = []
        for r in rows:
            lead_details_parsed = None
            if r["lead_details"]:
                try:
                    lead_details_parsed = json.loads(r["lead_details"])
                except Exception:
                    lead_details_parsed = r["lead_details"]

            results.append({
                "id": r["id"],
                "lead_id": r["lead_id"],
                "industry": r["industry"],
                "lead_source": r["lead_source"],
                "job_level": r["job_level"],
                "lead_details": lead_details_parsed,
                "score": r["score"],
                "priority": r["priority"],
                "calibrated_probability": r["calibrated_probability"],
                "model_version": r["model_version"],
                "created_at": r["created_at"],
            })
        return results
    finally:
        conn.close()


def get_summary(db_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Calculate aggregated prediction summary from the database."""
    path = _resolve_db_path(db_path)
    conn = sqlite3.connect(path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), AVG(score) FROM predictions;")
        row = cursor.fetchone()
        total_leads = row[0] if row else 0
        avg_score = round(float(row[1]), 2) if (row and row[1] is not None) else 0.0

        cursor.execute(
            "SELECT priority, COUNT(*) FROM predictions GROUP BY priority;"
        )
        priority_rows = cursor.fetchall()
        counts_by_priority = {"High": 0, "Medium": 0, "Low": 0}
        for prio, count in priority_rows:
            if prio in counts_by_priority:
                counts_by_priority[prio] = count

        cursor.execute("SELECT MAX(created_at) FROM predictions;")
        latest_row = cursor.fetchone()
        latest_created_at = latest_row[0] if latest_row else None

        return {
            "total_leads": total_leads,
            "average_score": avg_score,
            "counts_by_priority": counts_by_priority,
            "latest_prediction_at": latest_created_at,
        }
    finally:
        conn.close()
