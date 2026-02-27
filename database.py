"""
AgriTrust AI – Database Layer
==============================
SQLite-backed persistence for loan application records.

Cloud Deployment Note:
  In production this module would be replaced by a PostgreSQL / Aurora
  connection pool (via SQLAlchemy + asyncpg).  The SQLite file here is
  a drop-in local substitute that keeps the Streamlit demo self-contained.

Microservices Note:
  A dedicated FastAPI microservice with its own DB cluster would handle
  write-heavy production loads; this module remains for local dev / demo.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join("data", "agritrust.db")


def _get_conn() -> sqlite3.Connection:
    """Return a connection with row_factory set for dict-like access."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create the applications table if it does not exist.
    Safe to call multiple times (idempotent).
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS applications (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        applicant_name  TEXT    NOT NULL DEFAULT 'N/A',
        farm_size       REAL    NOT NULL,
        soil_score      INTEGER NOT NULL,
        rainfall        REAL    NOT NULL,
        previous_loans  INTEGER NOT NULL,
        yield_amount    REAL    NOT NULL,
        trust_score     REAL    NOT NULL,
        risk_category   TEXT    NOT NULL,
        officer_id      TEXT    NOT NULL DEFAULT 'system',
        timestamp       TEXT    NOT NULL
    );
    """
    with _get_conn() as conn:
        conn.execute(ddl)
        conn.commit()


def insert_application(
    applicant_name: str,
    farm_size: float,
    soil_score: int,
    rainfall: float,
    previous_loans: int,
    yield_amount: float,
    trust_score: float,
    risk_category: str,
    officer_id: str = "system",
) -> int:
    """Insert a new application record and return its auto-generated id."""
    sql = """
    INSERT INTO applications
        (applicant_name, farm_size, soil_score, rainfall,
         previous_loans, yield_amount, trust_score,
         risk_category, officer_id, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _get_conn() as conn:
        cur = conn.execute(sql, (
            applicant_name, farm_size, soil_score, rainfall,
            previous_loans, yield_amount, trust_score,
            risk_category, officer_id, ts,
        ))
        conn.commit()
        return cur.lastrowid


def fetch_all_applications(risk_filter: str = "All") -> list[dict]:
    """
    Return all applications as a list of dicts, ordered by timestamp desc.
    Optionally filter by risk_category.

    # TODO: Add advanced filtering and search functionality here
    # (date range, officer, farm size band, trust score range)
    """
    with _get_conn() as conn:
        if risk_filter == "All":
            rows = conn.execute(
                "SELECT * FROM applications ORDER BY timestamp DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM applications WHERE risk_category = ? "
                "ORDER BY timestamp DESC",
                (risk_filter,),
            ).fetchall()
    return [dict(r) for r in rows]


def get_summary_stats() -> dict:
    """Return aggregate statistics for the admin dashboard."""
    with _get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0]
        approved = conn.execute(
            "SELECT COUNT(*) FROM applications WHERE risk_category IN ('Low Risk','Moderate Risk')"
        ).fetchone()[0]
        avg_score = conn.execute(
            "SELECT AVG(trust_score) FROM applications"
        ).fetchone()[0]
        dist = conn.execute(
            "SELECT risk_category, COUNT(*) as cnt FROM applications GROUP BY risk_category"
        ).fetchall()

    return {
        "total": total,
        "approved": approved,
        "rejected": total - approved,
        "avg_score": round(avg_score, 1) if avg_score else 0,
        "distribution": {row["risk_category"]: row["cnt"] for row in dist},
    }
