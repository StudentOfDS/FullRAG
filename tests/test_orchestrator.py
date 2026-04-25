from __future__ import annotations

import asyncio
from pathlib import Path

from fullrag.orchestration.service import FullRAGOrchestrator


def test_ingest_and_query(tmp_path: Path):
    doc = tmp_path / "doc.txt"
    doc.write_text("RAG systems combine retrieval and generation for grounded answers.")

    config = {
        "embedding": {"dimension": 256},
        "indexing": {"sparse": {"bm25_k1": 1.5, "bm25_b": 0.75, "stage1_candidates": 20}},
        "llm": {"providers": ["openai", "ollama"], "request_timeout_seconds": 5, "max_concurrency": 2},
        "cache": {"semantic_threshold": 0.9, "ttl_seconds": 300},
    }
    orchestrator = FullRAGOrchestrator(config)

    stats = orchestrator.ingest_paths([str(doc)])
    assert stats["units"] == 1
    assert stats["chunks"] == 1

    result = asyncio.run(orchestrator.query("What is RAG?"))
    assert "answer" in result
    assert result["provider"] in {"openai", "ollama", "cache"}
