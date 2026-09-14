import sqlite3
import json
import os
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

from src.memory.models import WatchEvent, RatingEvent, EventType, FeatureType


class MemoryStorage:
    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at REAL,
                    last_active_at REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watch_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    movie_id INTEGER,
                    event_type TEXT,
                    signal_weight REAL,
                    timestamp REAL,
                    metadata TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    movie_id INTEGER,
                    rating REAL,
                    signal_target REAL,
                    timestamp REAL,
                    review_text TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profile (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT,
                    updated_at REAL,
                    version INTEGER
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT,
                    feature_type TEXT,
                    feature_value TEXT,
                    score REAL,
                    confidence REAL,
                    interaction_count INTEGER,
                    last_updated REAL,
                    PRIMARY KEY(user_id, feature_type, feature_value)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS negative_preferences (
                    user_id TEXT,
                    feature_type TEXT,
                    feature_value TEXT,
                    penalty_weight REAL,
                    interaction_count INTEGER,
                    last_updated REAL,
                    PRIMARY KEY(user_id, feature_type, feature_value)
                )
            """)
            # Indices for fast querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_watch_events_user_ts ON watch_events(user_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_watch_events_user_movie ON watch_events(user_id, movie_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(user_id, timestamp)")
            conn.commit()

    def ensure_user(self, user_id: str, ts: Optional[float] = None):
        ts = ts or time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (user_id, created_at, last_active_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET last_active_at=excluded.last_active_at
            """, (user_id, ts, ts))
            conn.commit()

    def record_event(self, event: WatchEvent) -> int:
        self.ensure_user(event.user_id, event.timestamp)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO watch_events (user_id, movie_id, event_type, signal_weight, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event.user_id,
                event.movie_id,
                event.event_type.value if isinstance(event.event_type, EventType) else str(event.event_type),
                event.signal_weight,
                event.timestamp,
                json.dumps(event.metadata) if event.metadata else "{}"
            ))
            event_id = cursor.lastrowid
            conn.commit()
            return event_id

    def record_rating(self, rating: RatingEvent) -> int:
        self.ensure_user(rating.user_id, rating.timestamp)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ratings (user_id, movie_id, rating, signal_target, timestamp, review_text)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                rating.user_id,
                rating.movie_id,
                rating.rating,
                rating.signal_target,
                rating.timestamp,
                json.dumps(rating.review_text) if isinstance(rating.review_text, (dict, list)) else (str(rating.review_text) if rating.review_text is not None else None)
            ))
            rating_id = cursor.lastrowid
            conn.commit()
            return rating_id

    def get_events(self, user_id: str, since_timestamp: Optional[float] = None) -> List[WatchEvent]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if since_timestamp is not None:
                cursor.execute("""
                    SELECT id, user_id, movie_id, event_type, signal_weight, timestamp, metadata
                    FROM watch_events
                    WHERE user_id = ? AND timestamp >= ?
                    ORDER BY timestamp ASC
                """, (user_id, since_timestamp))
            else:
                cursor.execute("""
                    SELECT id, user_id, movie_id, event_type, signal_weight, timestamp, metadata
                    FROM watch_events
                    WHERE user_id = ?
                    ORDER BY timestamp ASC
                """, (user_id,))
            rows = cursor.fetchall()
            events = []
            for row in rows:
                meta = {}
                if row["metadata"]:
                    try:
                        meta = json.loads(row["metadata"])
                    except:
                        meta = {}
                events.append(WatchEvent(
                    id=row["id"],
                    user_id=row["user_id"],
                    movie_id=row["movie_id"],
                    event_type=EventType(row["event_type"]) if row["event_type"] in [e.value for e in EventType] else EventType.MOVIE_VIEWED,
                    signal_weight=row["signal_weight"],
                    timestamp=row["timestamp"],
                    metadata=meta
                ))
            return events

    def get_ratings(self, user_id: str) -> List[RatingEvent]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, user_id, movie_id, rating, signal_target, timestamp, review_text
                FROM ratings
                WHERE user_id = ?
                ORDER BY timestamp ASC
            """, (user_id,))
            rows = cursor.fetchall()
            ratings = []
            for row in rows:
                ratings.append(RatingEvent(
                    id=row["id"],
                    user_id=row["user_id"],
                    movie_id=row["movie_id"],
                    rating=row["rating"],
                    signal_target=row["signal_target"],
                    timestamp=row["timestamp"],
                    review_text=row["review_text"]
                ))
            return ratings

    def get_recent_movie_events(self, user_id: str, movie_id: int, window_seconds: float = 1800.0, current_time: Optional[float] = None) -> List[WatchEvent]:
        current_time = current_time or time.time()
        since = current_time - window_seconds
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, user_id, movie_id, event_type, signal_weight, timestamp, metadata
                FROM watch_events
                WHERE user_id = ? AND movie_id = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
            """, (user_id, movie_id, since, current_time))
            rows = cursor.fetchall()
            events = []
            for row in rows:
                meta = {}
                if row["metadata"]:
                    try:
                        meta = json.loads(row["metadata"])
                    except:
                        pass
                events.append(WatchEvent(
                    id=row["id"],
                    user_id=row["user_id"],
                    movie_id=row["movie_id"],
                    event_type=EventType(row["event_type"]) if row["event_type"] in [e.value for e in EventType] else EventType.MOVIE_VIEWED,
                    signal_weight=row["signal_weight"],
                    timestamp=row["timestamp"],
                    metadata=meta
                ))
            return events

    def get_movie_metadata_fallback(self, movie_id: int) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT metadata FROM watch_events
                WHERE movie_id = ? AND metadata != '{}' AND metadata IS NOT NULL
                ORDER BY timestamp DESC LIMIT 1
            """, (movie_id,))
            row = cursor.fetchone()
            if row and row["metadata"]:
                try:
                    return json.loads(row["metadata"])
                except:
                    pass
        return {}

    def save_profile_json(self, user_id: str, profile_json: str, version: int = 1, updated_at: Optional[float] = None):
        updated_at = updated_at or time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_profile (user_id, profile_json, updated_at, version)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_json=excluded.profile_json,
                    updated_at=excluded.updated_at,
                    version=excluded.version
            """, (user_id, profile_json, updated_at, version))
            conn.commit()

    def get_profile_json(self, user_id: str) -> Optional[str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT profile_json FROM user_profile WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return row["profile_json"]
            return None
