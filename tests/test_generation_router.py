from __future__ import annotations

import asyncio

from fullrag.generation.service import MultiProviderGenerator


def test_generation_local_fallback_when_no_provider_available():
    gen = MultiProviderGenerator(
        providers=["openai"],
        timeout_seconds=1,
        max_concurrency=1,
        max_retries=0,
        secrets={},
    )
    result = asyncio.run(gen.answer("What is RAG?", []))
    assert result.provider == "local_fallback"
    assert "grounded" in result.answer.lower() or "enough grounded context" in result.answer.lower()
