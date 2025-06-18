import sqlite3
import json
import os
from typing import List, Dict
from datetime import datetime
from .base import BaseStorage


class SQLiteStorage(BaseStorage):
    def __init__(self, db_path: str = "predictions.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_sessions (
                    uid TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    original_image TEXT,
                    predicted_image TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detection_objects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_uid TEXT,
                    label TEXT,
                    score REAL,
                    box TEXT,
                    FOREIGN KEY (prediction_uid) REFERENCES prediction_sessions (uid)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_uid ON detection_objects (prediction_uid)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON detection_objects (label)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_score ON detection_objects (score)")

    def save_prediction(self, uid: str, original_image: str, predicted_image: str) -> None:
        """Save metadata for a prediction session"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO prediction_sessions (uid, original_image, predicted_image)
                VALUES (?, ?, ?)
            """, (uid, original_image, predicted_image))

    def save_detection(self, prediction_uid: str, label: str, score: float, box: List[float]) -> None:
        """Save a single detected object"""
        box_json = json.dumps(box)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO detection_objects (prediction_uid, label, score, box)
                VALUES (?, ?, ?, ?)
            """, (prediction_uid, label, score, box_json))

    def get_prediction(self, uid: str) -> Dict:
        """Retrieve full prediction session including metadata and all detections"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get prediction session
            session = conn.execute(
                "SELECT * FROM prediction_sessions WHERE uid = ?",
                (uid,)
            ).fetchone()

            if not session:
                return None

            # Get detection objects
            objects = conn.execute(
                "SELECT * FROM detection_objects WHERE prediction_uid = ?",
                (uid,)
            ).fetchall()

            return {
                "uid": session["uid"],
                "timestamp": session["timestamp"],
                "original_image": session["original_image"],
                "predicted_image": session["predicted_image"],
                "detection_objects": [
                    {
                        "id": obj["id"],
                        "label": obj["label"],
                        "score": obj["score"],
                        "box": json.loads(obj["box"])
                    } for obj in objects
                ]
            }

    def get_predictions_by_label(self, label: str) -> List[Dict]:
        """Get all prediction sessions that include a detection with a specific label"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT DISTINCT ps.uid, ps.timestamp
                FROM prediction_sessions ps
                JOIN detection_objects do ON ps.uid = do.prediction_uid
                WHERE do.label = ?
                ORDER BY ps.timestamp DESC
            """, (label,)).fetchall()

            return [{"uid": row["uid"], "timestamp": row["timestamp"]} for row in rows]

    def get_predictions_by_score(self, min_score: float) -> List[Dict]:
        """Get all prediction sessions that include detections with score >= min_score"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT DISTINCT ps.uid, ps.timestamp
                FROM prediction_sessions ps
                JOIN detection_objects do ON ps.uid = do.prediction_uid
                WHERE do.score >= ?
                ORDER BY ps.timestamp DESC
            """, (min_score,)).fetchall()

            return [{"uid": row["uid"], "timestamp": row["timestamp"]} for row in rows]

    def get_prediction_image_path(self, uid: str) -> str:
        """Get the path to the predicted image file for a given prediction UID"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT predicted_image FROM prediction_sessions WHERE uid = ?",
                (uid,)
            ).fetchone()

            if not row:
                return None

            return row[0]