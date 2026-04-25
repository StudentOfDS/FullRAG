from __future__ import annotations

from collections import defaultdict

from fullrag.models.entities import Chunk


class DocumentStore:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._by_document: dict[str, list[str]] = defaultdict(list)

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            if chunk.chunk_id not in self._by_document[chunk.document_id]:
                self._by_document[chunk.document_id].append(chunk.chunk_id)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def get_many(self, chunk_ids: list[str]) -> list[Chunk]:
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    def mark_tombstoned(self, document_id: str) -> int:
        count = 0
        for chunk_id in self._by_document.get(document_id, []):
            chunk = self._chunks[chunk_id]
            if not chunk.tombstoned:
                chunk.tombstoned = True
                count += 1
        return count

    def active_chunks(self) -> list[Chunk]:
        return [chunk for chunk in self._chunks.values() if not chunk.tombstoned]
