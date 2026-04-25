from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(slots=True)
class CacheItem:
    question: str
    answer: str
    embedding: set[str]
    expires_at: datetime


class SemanticCache:
    def __init__(self, threshold: float = 0.95, ttl_seconds: int = 3600) -> None:
        self.threshold = threshold
        self.ttl = timedelta(seconds=ttl_seconds)
        self._items: list[CacheItem] = []

    def get(self, query: str) -> str | None:
        tokens = set(query.lower().split())
        now = datetime.utcnow()
        self._items = [item for item in self._items if item.expires_at > now]
        for item in self._items:
            sim = self._jaccard(tokens, item.embedding)
            if sim >= self.threshold:
                return item.answer
        return None

    def put(self, query: str, answer: str) -> None:
        self._items.append(
            CacheItem(
                question=query,
                answer=answer,
                embedding=set(query.lower().split()),
                expires_at=datetime.utcnow() + self.ttl,
            )
        )

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 1.0
        return len(a & b) / max(1, len(a | b))
