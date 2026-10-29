"""
SQLite Database helper for storing and retrieving IPL Match Predictions.
Fully compatible with local execution and serverless environments (e.g. Vercel, AWS Lambda)
by dynamically utilizing writable /tmp storage and in-memory fallbacks when deployed.
"""

import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent

# In-memory fallback if disk persistence is restricted
_MEMORY_PREDICTIONS = []


def get_db_path() -> Path:
    """
    Determine the appropriate database file path.
    On Vercel and serverless platforms, the project directory is strictly read-only,
    so we must use the writable temporary directory (/tmp).
    """
    is_serverless = (
        bool(os.environ.get("VERCEL"))
        or bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
        or bool(os.environ.get("LAMBDA_TASK_ROOT"))
        or not os.access(str(ROOT_DIR), os.W_OK)
    )

    if is_serverless:
        temp_dir = Path(tempfile.gettempdir())
        tmp_db = temp_dir / "predictions.db"

        # Copy initial local DB to /tmp if it exists and hasn't been copied yet
        local_db = ROOT_DIR / "database" / "predictions.db"
        if local_db.exists() and not tmp_db.exists():
            try:
                shutil.copy2(str(local_db), str(tmp_db))
            except Exception:
                pass

        return tmp_db

    # Standard local environment path
    local_dir = ROOT_DIR / "database"
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return local_dir / "predictions.db"


def get_db_connection():
    db_path = get_db_path()
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                team1 TEXT NOT NULL,
                team2 TEXT NOT NULL,
                toss_winner TEXT NOT NULL,
                toss_decision TEXT NOT NULL,
                venue TEXT NOT NULL,
                predicted_winner TEXT NOT NULL,
                prob_team1 REAL NOT NULL,
                prob_team2 REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[INFO] SQLite init skipped or read-only ({e}). Fallback active.")


def save_prediction(team1, team2, toss_winner, toss_decision, venue, predicted_winner, prob_team1, prob_team2):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Record in memory so history works seamlessly even without persistent disk
    record = {
        "id": len(_MEMORY_PREDICTIONS) + 1,
        "timestamp": now_str,
        "team1": team1,
        "team2": team2,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "venue": venue,
        "predicted_winner": predicted_winner,
        "prob_team1": prob_team1,
        "prob_team2": prob_team2
    }
    _MEMORY_PREDICTIONS.insert(0, record)

    # Attempt to persist to SQLite
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (timestamp, team1, team2, toss_winner, toss_decision, venue, predicted_winner, prob_team1, prob_team2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (now_str, team1, team2, toss_winner, toss_decision, venue, predicted_winner, prob_team1, prob_team2))
        conn.commit()
        conn.close()
    except Exception as e:
        # Gracefully handle read-only environments without throwing errors to the user
        print(f"[INFO] SQLite write skipped in serverless environment ({e}). Preserved in memory.")


def get_prediction_history(limit=50):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        if rows:
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"[INFO] Using in-memory prediction history ({e}).")

    return _MEMORY_PREDICTIONS[:limit]


def clear_history():
    global _MEMORY_PREDICTIONS
    _MEMORY_PREDICTIONS = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[INFO] Clear SQLite history skipped ({e}).")


if __name__ == "__main__":
    init_db()
    print(f"[OK] Database initialized at: {get_db_path()}")
