from __future__ import annotations

import asyncio
import hashlib
import logging

from fullrag.caching.semantic_cache import SemanticCache
from fullrag.chunking.service import AdaptiveChunker
from fullrag.generation.service import GeneratorPolicy, MultiProviderGenerator
from fullrag.guardrails.service import QueryGuardrails
from fullrag.indexing.dense import DenseIndex, PineconeClient
from fullrag.indexing.embeddings import EmbeddingClient, EmbeddingConfig
from fullrag.indexing.service import HybridIndexer
from fullrag.indexing.sparse import BM25Index
from fullrag.ingestion.service import IngestionPipeline
from fullrag.monitoring.metrics import MetricsRegistry
from fullrag.retrieval.service import HybridRetriever
from fullrag.storage.docstore import DocumentStore
from fullrag.utils.logging import timed_step


class FullRAGOrchestrator:
    def __init__(self, config: dict) -> None:
        self.logger = logging.getLogger("fullrag.orchestrator")
        self.metrics = MetricsRegistry()

        secrets = config.get("secrets", {})
        self.docstore = DocumentStore(db_path=config.get("storage", {}).get("docstore_path", "./data/docstore.db"))
        sparse = BM25Index(
            k1=config["indexing"]["sparse"]["bm25_k1"],
            b=config["indexing"]["sparse"]["bm25_b"],
        )
        embedder = EmbeddingClient(
            EmbeddingConfig(
                provider=config["embedding"].get("provider", "openai"),
                model=config["embedding"].get("model", "text-embedding-3-large"),
                dimension=min(3072, config["embedding"]["dimension"]),
                openai_api_key=secrets.get("openai_api_key"),
                huggingface_api_key=secrets.get("huggingface_api_key"),
            )
        )
        pinecone_cfg = config.get("indexing", {}).get("pinecone", {})
        pinecone = PineconeClient(
            api_key=secrets.get("pinecone_api_key"),
            host=secrets.get("pinecone_host") or pinecone_cfg.get("host"),
            namespace=pinecone_cfg.get("namespace", "default"),
            enabled=pinecone_cfg.get("enabled", False),
            timeout_seconds=pinecone_cfg.get("timeout_seconds", 10.0),
            max_retries=pinecone_cfg.get("max_retries", 4),
            base_backoff_seconds=pinecone_cfg.get("base_backoff_seconds", 0.25),
            max_backoff_seconds=pinecone_cfg.get("max_backoff_seconds", 4.0),
        )
        dense = DenseIndex(
            embedder=embedder,
            dimension=min(3072, config["embedding"]["dimension"]),
            pinecone_client=pinecone,
            faiss_enabled=config["indexing"].get("faiss", {}).get("enabled", True),
            faiss_path=config["indexing"].get("faiss", {}).get("path", "./data/faiss.index"),
            normalize_l2=config["indexing"].get("faiss", {}).get("normalize_l2", True),
        )

        self.ingestion = IngestionPipeline()
        self.chunker = AdaptiveChunker()
        self.indexer = HybridIndexer(self.docstore, sparse, dense)
        hybrid_cfg = config["indexing"].get("hybrid", {})
        self.retriever = HybridRetriever(
            self.docstore,
            sparse,
            dense,
            stage1_candidates=config["indexing"]["sparse"]["stage1_candidates"],
            short_query_alpha=hybrid_cfg.get("short_query_alpha", 0.35),
            default_alpha=hybrid_cfg.get("default_alpha", 0.5),
            long_query_alpha=hybrid_cfg.get("long_query_alpha", 0.75),
            long_query_token_threshold=hybrid_cfg.get("long_query_token_threshold", 12),
            enable_reranker=config.get("feature_flags", {}).get("enable_cross_encoder", True),
        )
        policy = GeneratorPolicy(
            max_prompt_chars=config["llm"].get("max_prompt_chars", 12000),
            max_retries=config["llm"].get("max_retries", 2),
            base_backoff_seconds=config["llm"].get("base_backoff_seconds", 0.15),
            max_backoff_seconds=config["llm"].get("max_backoff_seconds", 2.0),
            circuit_breaker_failure_threshold=config["llm"].get("circuit_breaker_failure_threshold", 3),
            circuit_breaker_cooldown_seconds=config["llm"].get("circuit_breaker_cooldown_seconds", 15.0),
            temperature=config["llm"].get("temperature", 0.2),
            max_output_tokens=config["llm"].get("max_output_tokens", 1000),
        )
        self.generator = MultiProviderGenerator(
            providers=config["llm"]["providers"],
            timeout_seconds=config["llm"]["request_timeout_seconds"],
            max_concurrency=config["llm"]["max_concurrency"],
            max_retries=config["llm"].get("max_retries", 2),
            policy=policy,
            provider_models=config["llm"].get("models", {}),
            secrets=secrets,
        )
        self.guardrails = QueryGuardrails()
        self.cache = SemanticCache(
            threshold=config["cache"]["semantic_threshold"],
            ttl_seconds=config["cache"]["ttl_seconds"],
            db_path=config.get("storage", {}).get("cache_path", "./data/cache.db"),
        )

    def ingest_paths(self, paths: list[str]) -> dict[str, int]:
        with timed_step(self.logger, "ingestion"):
            units = self.ingestion.ingest(paths)
            chunks = self.chunker.chunk(units)
            self.indexer.upsert(chunks)
            self.metrics.inc("ingested_units", len(units))
            self.metrics.inc("ingested_chunks", len(chunks))
            return {"units": len(units), "chunks": len(chunks)}

    def tombstone_document(self, source_file: str) -> int:
        document_id = hashlib.sha256(source_file.encode()).hexdigest()
        count = self.indexer.mark_tombstoned(document_id)
        self.metrics.inc("tombstoned_chunks", count)
        return count

    async def query(self, query: str):
        ok, reason = self.guardrails.validate(query)
        if not ok:
            self.metrics.inc("query_rejected")
            return {"error": reason}

        preprocessed = self.guardrails.preprocess(query)
        cached = self.cache.get(preprocessed)
        if cached:
            self.metrics.inc("cache_hit")
            return {"answer": cached, "sources": [], "provider": "cache", "cached": True}

        chunks = await self.retriever.retrieve(preprocessed)
        result = await self.generator.answer(preprocessed, chunks)
        self.cache.put(preprocessed, result.answer)

        self.metrics.inc("query_success")
        self.metrics.observe("query_latency_ms", result.latency_ms)
        return {
            "answer": result.answer,
            "sources": result.sources,
            "provider": result.provider,
            "cached": result.cached,
        }

    async def nightly_cleanup(self) -> None:
        await asyncio.sleep(0)
        self.logger.info("airflow_cleanup_simulated")
