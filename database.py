"""
SQLite Database helper for storing and retrieving IPL Match Predictions.
"""

import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent / "database" / "predictions.db"


def get_db_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
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


def save_prediction(team1, team2, toss_winner, toss_decision, venue, predicted_winner, prob_team1, prob_team2):
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO predictions (timestamp, team1, team2, toss_winner, toss_decision, venue, predicted_winner, prob_team1, prob_team2)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (now_str, team1, team2, toss_winner, toss_decision, venue, predicted_winner, prob_team1, prob_team2))
    conn.commit()
    conn.close()


def get_prediction_history(limit=50):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_history():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"[OK] Database initialized at: {DB_PATH}")
