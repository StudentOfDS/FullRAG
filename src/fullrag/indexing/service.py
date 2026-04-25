from __future__ import annotations

from fullrag.core.interfaces import IndexingService
from fullrag.indexing.dense import DenseIndex
from fullrag.indexing.sparse import BM25Index
from fullrag.models.entities import Chunk
from fullrag.storage.docstore import DocumentStore


class HybridIndexer(IndexingService):
    def __init__(self, docstore: DocumentStore, sparse: BM25Index, dense: DenseIndex) -> None:
        self.docstore = docstore
        self.sparse = sparse
        self.dense = dense

    def upsert(self, chunks: list[Chunk]) -> None:
        self.docstore.upsert_chunks(chunks)
        self.sparse.add(chunks)
        self.dense.upsert_with_retry(chunks)

    def mark_tombstoned(self, document_id: str) -> int:
        return self.docstore.mark_tombstoned(document_id)
