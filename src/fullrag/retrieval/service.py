from __future__ import annotations

import asyncio

from fullrag.indexing.dense import DenseIndex
from fullrag.indexing.sparse import BM25Index
from fullrag.models.entities import Chunk
from fullrag.storage.docstore import DocumentStore

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None


class HybridRetriever:
    def __init__(
        self,
        docstore: DocumentStore,
        sparse_index: BM25Index,
        dense_index: DenseIndex,
        stage1_candidates: int = 200,
        short_query_alpha: float = 0.35,
        default_alpha: float = 0.5,
        long_query_alpha: float = 0.75,
        long_query_token_threshold: int = 12,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enable_reranker: bool = True,
    ) -> None:
        self.docstore = docstore
        self.sparse_index = sparse_index
        self.dense_index = dense_index
        self.stage1_candidates = stage1_candidates
        self.short_query_alpha = short_query_alpha
        self.default_alpha = default_alpha
        self.long_query_alpha = long_query_alpha
        self.long_query_token_threshold = long_query_token_threshold
        self._reranker = CrossEncoder(reranker_model) if enable_reranker and CrossEncoder else None

    async def retrieve(self, query: str, top_k: int = 8) -> list[Chunk]:
        sparse_hits = self.sparse_index.search(query, self.stage1_candidates)
        if not sparse_hits:
            candidates = [chunk.chunk_id for chunk in self.docstore.active_chunks()]
            dense_hits = self.dense_index.query(query, candidates, top_k)
            return self.docstore.get_many([cid for cid, _ in dense_hits])

        candidate_ids = [cid for cid, _ in sparse_hits]
        dense_hits = self.dense_index.query(query, candidate_ids, self.stage1_candidates)
        sparse_map = dict(sparse_hits)
        dense_map = dict(dense_hits)
        alpha = self._dynamic_alpha(query)

        def norm(v: float, all_values: list[float]) -> float:
            lo, hi = min(all_values), max(all_values)
            return 0.0 if hi == lo else (v - lo) / (hi - lo)

        s_values = list(sparse_map.values()) or [0.0]
        d_values = list(dense_map.values()) or [0.0]
        scored: list[tuple[str, float]] = []
        for cid in candidate_ids:
            s = norm(sparse_map.get(cid, 0.0), s_values)
            d = norm(dense_map.get(cid, 0.0), d_values)
            scored.append((cid, alpha * d + (1 - alpha) * s))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked_ids = await asyncio.to_thread(self._rerank_ids, query, [cid for cid, _ in scored[: top_k * 3]])
        final = self._dedupe_and_merge(self.docstore.get_many(reranked_ids), top_k)
        return final

    def _dynamic_alpha(self, query: str) -> float:
        tokens = len(query.split())
        if tokens < 5:
            return self.short_query_alpha
        if tokens > self.long_query_token_threshold:
            return self.long_query_alpha
        return self.default_alpha

    def _rerank_ids(self, query: str, ids: list[str]) -> list[str]:
        if not self._reranker:
            return ids
        chunks = self.docstore.get_many(ids)
        pairs = [(query, c.text[:2000]) for c in chunks]
        scores = self._reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        rescored = sorted(zip(chunks, scores), key=lambda item: float(item[1]), reverse=True)
        return [chunk.chunk_id for chunk, _ in rescored]

    @staticmethod
    def _dedupe_and_merge(chunks: list[Chunk], top_k: int) -> list[Chunk]:
        seen = set()
        final: list[Chunk] = []
        for chunk in chunks:
            if chunk.tombstoned or chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            final.append(chunk)
            if len(final) >= top_k:
                break
        return final
