from __future__ import annotations

import asyncio
import hashlib
import logging

from fullrag.caching.semantic_cache import SemanticCache
from fullrag.chunking.service import AdaptiveChunker
from fullrag.generation.service import MultiProviderGenerator
from fullrag.guardrails.service import QueryGuardrails
from fullrag.indexing.dense import DenseIndex
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

        self.docstore = DocumentStore()
        sparse = BM25Index(
            k1=config["indexing"]["sparse"]["bm25_k1"],
            b=config["indexing"]["sparse"]["bm25_b"],
        )
        dense = DenseIndex(dimension=min(768, config["embedding"]["dimension"]))

        self.ingestion = IngestionPipeline()
        self.chunker = AdaptiveChunker()
        self.indexer = HybridIndexer(self.docstore, sparse, dense)
        self.retriever = HybridRetriever(
            self.docstore,
            sparse,
            dense,
            stage1_candidates=config["indexing"]["sparse"]["stage1_candidates"],
        )
        self.generator = MultiProviderGenerator(
            providers=config["llm"]["providers"],
            timeout_seconds=config["llm"]["request_timeout_seconds"],
            max_concurrency=config["llm"]["max_concurrency"],
        )
        self.guardrails = QueryGuardrails()
        self.cache = SemanticCache(
            threshold=config["cache"]["semantic_threshold"],
            ttl_seconds=config["cache"]["ttl_seconds"],
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
