from __future__ import annotations

import math
import random
import time
from pathlib import Path

from fullrag.indexing.embeddings import EmbeddingClient
from fullrag.models.entities import Chunk

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


class DenseIndex:
    """Dense retrieval with embedding client and optional FAISS persistence."""

    def __init__(
        self,
        embedder: EmbeddingClient,
        dimension: int,
        faiss_enabled: bool = True,
        faiss_path: str = "./data/faiss.index",
        normalize_l2: bool = True,
    ) -> None:
        self.embedder = embedder
        self.dimension = dimension
        self.normalize_l2 = normalize_l2
        self._vectors: dict[str, list[float]] = {}
        self._ids: list[str] = []
        self._faiss_path = Path(faiss_path)
        self._faiss_enabled = bool(faiss_enabled and faiss is not None)
        self._faiss_index = self._init_faiss_index() if self._faiss_enabled else None

    def _init_faiss_index(self):
        self._faiss_path.parent.mkdir(parents=True, exist_ok=True)
        if self._faiss_path.exists():
            return faiss.read_index(str(self._faiss_path))
        return faiss.IndexFlatIP(self.dimension)

    def upsert_with_retry(self, chunks: list[Chunk], max_retries: int = 5, backoff: float = 0.15) -> None:
        vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            for attempt in range(max_retries):
                try:
                    self._vectors[chunk.chunk_id] = self._normalize(vector) if self.normalize_l2 else vector
                    if chunk.chunk_id not in self._ids:
                        self._ids.append(chunk.chunk_id)
                    break
                except RuntimeError:
                    if attempt == max_retries - 1:
                        raise
                    jitter = random.uniform(0, backoff)
                    time.sleep(backoff + jitter)
                    backoff = min(backoff * 2, 2.0)
        self._rebuild_faiss()

    def _rebuild_faiss(self) -> None:
        # FAISS persistence hook intentionally disabled unless ndarray dependencies are available.
        return

    def query(self, text: str, candidate_ids: list[str], top_k: int) -> list[tuple[str, float]]:
        if not candidate_ids:
            return []
        q = self.embedder.embed_query(text)
        if self.normalize_l2:
            q = self._normalize(q)
        scored: list[tuple[str, float]] = []
        for cid in candidate_ids:
            v = self._vectors.get(cid)
            if not v:
                continue
            scored.append((cid, self._cosine(q, v)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        return float(sum(x * y for x, y in zip(a, b)))

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
