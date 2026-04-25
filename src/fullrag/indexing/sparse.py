from __future__ import annotations

from collections import Counter, defaultdict
from math import log

from fullrag.models.entities import Chunk


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_len: dict[str, int] = {}
        self._inverted: dict[str, dict[str, int]] = defaultdict(dict)
        self._avg_len = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            terms = self._terms(chunk)
            self._doc_len[chunk.chunk_id] = len(terms)
            term_counts = Counter(terms)
            for term, tf in term_counts.items():
                self._inverted[term][chunk.chunk_id] = tf
        self._avg_len = sum(self._doc_len.values()) / max(1, len(self._doc_len))

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        q_terms = query.lower().split()
        scores: dict[str, float] = defaultdict(float)
        n_docs = max(1, len(self._doc_len))
        for term in q_terms:
            postings = self._inverted.get(term, {})
            df = len(postings)
            if df == 0:
                continue
            idf = log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for chunk_id, tf in postings.items():
                dl = self._doc_len[chunk_id]
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / max(1e-9, self._avg_len)))
                scores[chunk_id] += idf * (tf * (self.k1 + 1)) / denom
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    @staticmethod
    def _terms(chunk: Chunk) -> list[str]:
        return (chunk.abstract + " " + chunk.text).lower().split()
