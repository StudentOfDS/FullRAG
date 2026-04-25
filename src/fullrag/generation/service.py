from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from fullrag.models.entities import Chunk, QueryResult


@dataclass(slots=True)
class ProviderClient:
    name: str
    timeout_seconds: float

    async def complete(self, prompt: str) -> str:
        await asyncio.sleep(0)
        return f"[{self.name}] {prompt[:500]}"


class MultiProviderGenerator:
    def __init__(self, providers: list[str], timeout_seconds: int = 30, max_concurrency: int = 8) -> None:
        self._clients = [ProviderClient(name=p, timeout_seconds=timeout_seconds) for p in providers]
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def answer(self, query: str, context: list[Chunk]) -> QueryResult:
        prompt = self._build_prompt(query, context)
        start = time.perf_counter()

        async with self._semaphore:
            for client in self._clients:
                try:
                    text = await asyncio.wait_for(client.complete(prompt), timeout=client.timeout_seconds)
                    return QueryResult(
                        answer=text,
                        sources=[f"{c.source_file}:{c.page}" for c in context],
                        latency_ms=(time.perf_counter() - start) * 1000,
                        provider=client.name,
                        cached=False,
                    )
                except Exception:
                    continue

        return QueryResult(
            answer="No provider available",
            sources=[],
            latency_ms=(time.perf_counter() - start) * 1000,
            provider="none",
            cached=False,
        )

    @staticmethod
    def _build_prompt(query: str, chunks: list[Chunk]) -> str:
        context = "\n\n".join(
            f"SOURCE: {chunk.source_file}#{chunk.page}\nABSTRACT: {chunk.abstract}\nTEXT: {chunk.text}"
            for chunk in chunks
        )
        return (
            "You are a grounded RAG assistant. Use only provided context and cite sources."
            f"\n\nQUESTION:\n{query}\n\nCONTEXT:\n{context}"
        )
