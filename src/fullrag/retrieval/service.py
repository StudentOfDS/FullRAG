from __future__ import annotations

import asyncio

from fullrag.indexing.dense import DenseIndex
from fullrag.indexing.sparse import BM25Index
from fullrag.models.entities import Chunk
from fullrag.storage.docstore import DocumentStore


class HybridRetriever:
    def __init__(
        self,
        docstore: DocumentStore,
        sparse_index: BM25Index,
        dense_index: DenseIndex,
        stage1_candidates: int = 200,
    ) -> None:
        self.docstore = docstore
        self.sparse_index = sparse_index
        self.dense_index = dense_index
        self.stage1_candidates = stage1_candidates

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

        reranked_ids = await asyncio.to_thread(self._rerank_ids, query, [cid for cid, _ in scored[:top_k * 2]])
        final = self._dedupe_and_merge(self.docstore.get_many(reranked_ids), top_k)
        return final

    @staticmethod
    def _dynamic_alpha(query: str) -> float:
        tokens = len(query.split())
        return 0.35 if tokens < 5 else 0.75 if tokens > 12 else 0.5

    @staticmethod
    def _rerank_ids(query: str, ids: list[str]) -> list[str]:
        _ = query
        return ids

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
