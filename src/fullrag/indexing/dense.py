from __future__ import annotations

import json
import math
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

from fullrag.indexing.embeddings import EmbeddingClient
from fullrag.models.entities import Chunk

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


class PineconeClient:
    """Minimal Pinecone REST client (serverless API) with retry + jitter."""

    def __init__(
        self,
        api_key: str | None,
        host: str | None,
        namespace: str,
        enabled: bool,
        timeout_seconds: float = 10.0,
        max_retries: int = 4,
        base_backoff_seconds: float = 0.25,
        max_backoff_seconds: float = 4.0,
    ) -> None:
        self.enabled = bool(enabled and api_key and host)
        self.api_key = api_key
        self.host = host.rstrip("/") if host else ""
        self.namespace = namespace
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.base_backoff_seconds = base_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds

    def upsert(self, vectors: list[dict]) -> None:
        if not self.enabled or not vectors:
            return
        self._request("/vectors/upsert", {"vectors": vectors, "namespace": self.namespace})

    def query(self, vector: list[float], top_k: int, candidate_ids: list[str] | None = None) -> list[tuple[str, float]]:
        if not self.enabled:
            return []
        payload: dict[str, object] = {
            "vector": vector,
            "topK": top_k,
            "namespace": self.namespace,
            "includeValues": False,
            "includeMetadata": False,
        }
        if candidate_ids:
            payload["filter"] = {"chunk_id": {"$in": candidate_ids[:1000]}}
        data = self._request("/query", payload)
        return [(m.get("id", ""), float(m.get("score", 0.0))) for m in data.get("matches", []) if m.get("id")]

    def _request(self, path: str, payload: dict) -> dict:
        headers = {"Api-Key": self.api_key or "", "Content-Type": "application/json"}
        req = urllib.request.Request(
            f"{self.host}{path}", data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        backoff = self.base_backoff_seconds
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"Pinecone request failed: {path}") from exc
                sleep = min(backoff + random.uniform(0, backoff), self.max_backoff_seconds)
                time.sleep(sleep)
                backoff = min(backoff * 2, self.max_backoff_seconds)
        return {}


class DenseIndex:
    def __init__(
        self,
        embedder: EmbeddingClient,
        dimension: int,
        pinecone_client: PineconeClient | None = None,
        faiss_enabled: bool = True,
        faiss_path: str = "./data/faiss.index",
        normalize_l2: bool = True,
    ) -> None:
        self.embedder = embedder
        self.dimension = dimension
        self.normalize_l2 = normalize_l2
        self.pinecone = pinecone_client
        self._vectors: dict[str, list[float]] = {}
        self._ids: list[str] = []
        self._faiss_path = Path(faiss_path)
        self._json_path = self._faiss_path.with_suffix(".json")
        self._faiss_enabled = bool(faiss_enabled and faiss is not None)
        self._faiss_index = self._init_faiss_index() if self._faiss_enabled else None
        self._load_local_vectors()

    def _init_faiss_index(self):
        self._faiss_path.parent.mkdir(parents=True, exist_ok=True)
        return faiss.read_index(str(self._faiss_path)) if self._faiss_path.exists() else faiss.IndexFlatIP(self.dimension)

    def _load_local_vectors(self) -> None:
        if self._json_path.exists():
            payload = json.loads(self._json_path.read_text(encoding="utf-8"))
            self._vectors = {k: [float(x) for x in v] for k, v in payload.items()}
            self._ids = list(self._vectors.keys())

    def _persist_local_vectors(self) -> None:
        self._json_path.parent.mkdir(parents=True, exist_ok=True)
        self._json_path.write_text(json.dumps(self._vectors), encoding="utf-8")

    def upsert_with_retry(self, chunks: list[Chunk], max_retries: int = 5, backoff: float = 0.15) -> None:
        vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
        pinecone_vectors: list[dict] = []
        for chunk, vector in zip(chunks, vectors):
            norm_vector = self._normalize(vector) if self.normalize_l2 else vector
            for attempt in range(max_retries):
                try:
                    self._vectors[chunk.chunk_id] = norm_vector
                    if chunk.chunk_id not in self._ids:
                        self._ids.append(chunk.chunk_id)
                    pinecone_vectors.append(
                        {
                            "id": chunk.chunk_id,
                            "values": norm_vector,
                            "metadata": {
                                "chunk_id": chunk.chunk_id,
                                "document_id": chunk.document_id,
                                "source_file": chunk.source_file,
                                "page": chunk.page,
                                "section_path": " > ".join(chunk.section_path),
                                "tombstoned": chunk.tombstoned,
                            },
                        }
                    )
                    break
                except RuntimeError:
                    if attempt == max_retries - 1:
                        raise
                    jitter = random.uniform(0, backoff)
                    time.sleep(backoff + jitter)
                    backoff = min(backoff * 2, 2.0)
        self._persist_local_vectors()
        self._rebuild_faiss()
        if self.pinecone and self.pinecone.enabled:
            self.pinecone.upsert(pinecone_vectors)

    def _rebuild_faiss(self) -> None:
        if not self._faiss_enabled or faiss is None:
            return
        index = faiss.IndexFlatIP(self.dimension)
        if self._ids:
            import numpy as np

            arr = np.asarray([self._vectors[cid] for cid in self._ids], dtype="float32")
            index.add(arr)
        self._faiss_index = index
        faiss.write_index(index, str(self._faiss_path))

    def query(self, text: str, candidate_ids: list[str], top_k: int) -> list[tuple[str, float]]:
        q = self.embedder.embed_query(text)
        if self.normalize_l2:
            q = self._normalize(q)

        if self.pinecone and self.pinecone.enabled:
            try:
                hits = self.pinecone.query(q, top_k=top_k, candidate_ids=candidate_ids or None)
                if hits:
                    return hits
            except Exception:
                pass

        scoped = candidate_ids if candidate_ids else list(self._vectors.keys())
        scored = [(cid, self._cosine(q, self._vectors[cid])) for cid in scoped if cid in self._vectors]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        return float(sum(x * y for x, y in zip(a, b)))

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
