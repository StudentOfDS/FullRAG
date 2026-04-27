from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass(slots=True)
class CacheItem:
    question: str
    answer: str
    embedding: set[str]
    expires_at: datetime


class SemanticCache:
    def __init__(self, threshold: float = 0.95, ttl_seconds: int = 3600, db_path: str = "./data/cache.db") -> None:
        self.threshold = threshold
        self.ttl = timedelta(seconds=ttl_seconds)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_cache (
                question TEXT PRIMARY KEY,
                answer TEXT NOT NULL,
                token_json TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, query: str) -> str | None:
        tokens = set(query.lower().split())
        now = datetime.utcnow().isoformat()
        self._conn.execute("DELETE FROM semantic_cache WHERE expires_at <= ?", (now,))
        self._conn.commit()
        rows = self._conn.execute("SELECT answer, token_json FROM semantic_cache").fetchall()
        for answer, token_json in rows:
            sim = self._jaccard(tokens, set(json.loads(token_json)))
            if sim >= self.threshold:
                return answer
        return None

    def put(self, query: str, answer: str) -> None:
        expires_at = (datetime.utcnow() + self.ttl).isoformat()
        token_json = json.dumps(sorted(set(query.lower().split())))
        self._conn.execute(
            "INSERT OR REPLACE INTO semantic_cache(question, answer, token_json, expires_at) VALUES (?, ?, ?, ?)",
            (query, answer, token_json, expires_at),
        )
        self._conn.commit()

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 1.0
        return len(a & b) / max(1, len(a | b))
