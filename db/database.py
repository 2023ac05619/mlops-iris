import sqlite3
from pathlib import Path
from datetime import datetime

DB_FILE = Path(__file__).parent / "predictions.db"

def initialize_db():
    """Creates the database and the predictions table if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            model_name TEXT NOT NULL,
            prediction_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            features TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    # print("[INFO] Database initialized successfully.")

def log_prediction(model_name: str, prediction_name: str, confidence: float, features: str):
    """Inserts a single prediction log into the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO prediction_logs (timestamp, model_name, prediction_name, confidence, features)
        VALUES (?, ?, ?, ?, ?)
    """, (datetime.now(), model_name, prediction_name, confidence, str(features)))
    conn.commit()
    conn.close()

def get_prediction_counts():
    """Queries the database to get prediction counts per model and class."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT model_name, prediction_name, COUNT(*) as count
        FROM prediction_logs
        GROUP BY model_name, prediction_name
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows
