from __future__ import annotations

import hashlib
import math
import random
import time

from fullrag.models.entities import Chunk


class DenseIndex:
    """In-memory dense index simulating Pinecone+FAISS fallback."""

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension
        self._vectors: dict[str, list[float]] = {}

    def upsert_with_retry(self, chunks: list[Chunk], max_retries: int = 5, backoff: float = 0.1) -> None:
        for chunk in chunks:
            for attempt in range(max_retries):
                try:
                    self._vectors[chunk.chunk_id] = self._embed(chunk.text)
                    break
                except RuntimeError:
                    if attempt == max_retries - 1:
                        raise
                    jitter = random.uniform(0, backoff)
                    time.sleep(backoff + jitter)
                    backoff *= 2

    def query(self, text: str, candidate_ids: list[str], top_k: int) -> list[tuple[str, float]]:
        q = self._embed(text)
        scored: list[tuple[str, float]] = []
        for chunk_id in candidate_ids:
            vector = self._vectors.get(chunk_id)
            if vector is None:
                continue
            scored.append((chunk_id, self._cosine(q, vector)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _embed(self, text: str) -> list[float]:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        vec = [rng.uniform(-1, 1) for _ in range(self.dimension)]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))
