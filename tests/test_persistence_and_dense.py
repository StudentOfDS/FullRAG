from __future__ import annotations

from pathlib import Path

from fullrag.caching.semantic_cache import SemanticCache
from fullrag.indexing.dense import DenseIndex, PineconeClient
from fullrag.models.entities import Chunk
from fullrag.storage.docstore import DocumentStore


class StubEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        base = float(len(text) % 7 + 1)
        return [base, 1.0, 0.5]


def test_docstore_tombstone_and_load(tmp_path: Path):
    db = tmp_path / "docstore.db"
    store = DocumentStore(db_path=str(db))
    c = Chunk(
        chunk_id="c1",
        document_id="d1",
        text="hello",
        abstract="hello",
        section_path=["root"],
        source_file="a.txt",
        page=1,
    )
    store.upsert_chunks([c])
    assert len(store.active_chunks()) == 1
    assert store.mark_tombstoned("d1") == 1
    assert len(store.active_chunks()) == 0


def test_semantic_cache_expiry(tmp_path: Path):
    cache = SemanticCache(threshold=0.5, ttl_seconds=0, db_path=str(tmp_path / "cache.db"))
    cache.put("what is rag", "answer")
    assert cache.get("what is rag") is None


def test_dense_index_persisted_local_fallback(tmp_path: Path):
    dense_path = tmp_path / "dense.json"
    dense = DenseIndex(
        embedder=StubEmbedder(),
        dimension=3,
        pinecone_client=PineconeClient(None, None, "default", enabled=False),
        faiss_enabled=False,
        faiss_path=str(dense_path),
    )
    chunks = [
        Chunk(
            chunk_id="x1",
            document_id="d1",
            text="alpha",
            abstract="alpha",
            section_path=["root"],
            source_file="x.txt",
            page=1,
        )
    ]
    dense.upsert_with_retry(chunks)
    hits = dense.query("alpha", ["x1"], top_k=1)
    assert hits and hits[0][0] == "x1"

    reloaded = DenseIndex(
        embedder=StubEmbedder(),
        dimension=3,
        pinecone_client=PineconeClient(None, None, "default", enabled=False),
        faiss_enabled=False,
        faiss_path=str(dense_path),
    )
    hits2 = reloaded.query("alpha", ["x1"], top_k=1)
    assert hits2 and hits2[0][0] == "x1"
