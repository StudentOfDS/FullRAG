from __future__ import annotations

import hashlib
import json
import math
import random
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(slots=True)
class EmbeddingConfig:
    provider: str
    model: str
    dimension: int
    openai_api_key: str | None = None
    huggingface_api_key: str | None = None


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig, timeout_seconds: float = 20.0) -> None:
        self.config = config
        self.timeout_seconds = timeout_seconds

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        provider = self.config.provider.lower()
        if provider == "openai" and self.config.openai_api_key:
            try:
                return self._embed_openai(texts)
            except Exception:
                return [self._stable_hash_embedding(text) for text in texts]
        if provider in {"huggingface", "hf"} and self.config.huggingface_api_key:
            try:
                return self._embed_huggingface(texts)
            except Exception:
                return [self._stable_hash_embedding(text) for text in texts]
        return [self._stable_hash_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _post_json(self, url: str, payload: dict, headers: dict[str, str] | None = None):
        req_headers = {"Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=req_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Embedding endpoint failed: HTTP {exc.code}") from exc

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.config.openai_api_key}"}
        data = self._post_json(
            "https://api.openai.com/v1/embeddings",
            {"model": self.config.model, "input": texts},
            headers=headers,
        )
        return [self._normalize(item["embedding"]) for item in data.get("data", [])]

    def _embed_huggingface(self, texts: list[str]) -> list[list[float]]:
        model = self.config.model or "sentence-transformers/all-MiniLM-L6-v2"
        url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
        headers = {"Authorization": f"Bearer {self.config.huggingface_api_key}"}
        vectors: list[list[float]] = []
        for text in texts:
            payload = self._post_json(url, {"inputs": text}, headers=headers)
            if payload and isinstance(payload[0], list):
                if payload and payload[0] and isinstance(payload[0][0], list):
                    token_vectors = payload[0]
                else:
                    token_vectors = payload
                dim = len(token_vectors[0])
                pooled = [sum(v[i] for v in token_vectors) / len(token_vectors) for i in range(dim)]
                vectors.append(self._normalize(pooled))
            else:
                vectors.append(self._stable_hash_embedding(text))
        return vectors

    def _stable_hash_embedding(self, text: str) -> list[float]:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
        vec = [rng.uniform(-1, 1) for _ in range(self.config.dimension)]
        return self._normalize(vec)

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [float(v / norm) for v in vec]
