from __future__ import annotations

import asyncio
import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from fullrag.models.entities import Chunk, QueryResult


@dataclass(slots=True)
class ProviderClient:
    name: str
    timeout_seconds: float
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.2
    max_output_tokens: int = 1000

    async def complete(self, prompt: str) -> str:
        return await asyncio.to_thread(self._complete_sync, prompt)

    def _complete_sync(self, prompt: str) -> str:
        provider = self.name.lower()
        if provider == "openai" and self.api_key:
            return self._openai(prompt)
        if provider == "anthropic" and self.api_key:
            return self._anthropic(prompt)
        if provider == "gemini" and self.api_key:
            return self._gemini(prompt)
        if provider == "deepseek" and self.api_key:
            return self._openai(prompt, "https://api.deepseek.com/v1/chat/completions")
        if provider == "huggingface" and self.api_key:
            return self._huggingface(prompt)
        if provider == "ollama":
            return self._ollama(prompt)
        raise RuntimeError(f"provider '{self.name}' unavailable: missing configuration")

    def _post_json(self, url: str, payload: dict, headers: dict[str, str] | None = None) -> dict:
        req_headers = {"Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=req_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} for {url}") from exc

    def _openai(self, prompt: str, url: str = "https://api.openai.com/v1/chat/completions") -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        data = self._post_json(url, payload, headers=headers)
        return data["choices"][0]["message"]["content"]

    def _anthropic(self, prompt: str) -> str:
        headers = {"x-api-key": self.api_key or "", "anthropic-version": "2023-06-01"}
        payload = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = self._post_json("https://api.anthropic.com/v1/messages", payload, headers=headers)
        return data["content"][0]["text"]

    def _gemini(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        data = self._post_json(url, payload)
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def _huggingface(self, prompt: str) -> str:
        model = self.model or "mistralai/Mistral-7B-Instruct-v0.2"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = self._post_json(
            f"https://api-inference.huggingface.co/models/{model}",
            {"inputs": prompt, "parameters": {"max_new_tokens": self.max_output_tokens, "temperature": self.temperature}},
            headers=headers,
        )
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "")
        return str(data)

    def _ollama(self, prompt: str) -> str:
        base = self.base_url or "http://localhost:11434"
        payload = {
            "model": self.model or "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_output_tokens},
        }
        data = self._post_json(f"{base.rstrip('/')}/api/generate", payload)
        return data.get("response", "")


@dataclass(slots=True)
class GeneratorPolicy:
    max_prompt_chars: int = 12000
    max_retries: int = 2
    base_backoff_seconds: float = 0.15
    max_backoff_seconds: float = 2.0
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_cooldown_seconds: float = 15.0
    temperature: float = 0.2
    max_output_tokens: int = 1000


class MultiProviderGenerator:
    def __init__(
        self,
        providers: list[str],
        timeout_seconds: int = 30,
        max_concurrency: int = 8,
        max_retries: int = 2,
        policy: GeneratorPolicy | None = None,
        provider_models: dict[str, str] | None = None,
        secrets: dict[str, str | None] | None = None,
    ) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._policy = policy or GeneratorPolicy(max_retries=max_retries)
        self._max_retries = self._policy.max_retries
        self._provider_failures: dict[str, int] = {}
        self._provider_opened_at: dict[str, float] = {}
        secrets = secrets or {}
        model_defaults = provider_models or {}
        self._clients = [
            ProviderClient(
                name=p,
                timeout_seconds=timeout_seconds,
                model=model_defaults.get(p, "gpt-4o-mini"),
                api_key=secrets.get(f"{p}_api_key"),
                base_url=secrets.get("ollama_base_url"),
                temperature=self._policy.temperature,
                max_output_tokens=self._policy.max_output_tokens,
            )
            for p in providers
        ]

    async def answer(self, query: str, context: list[Chunk]) -> QueryResult:
        prompt = self._enforce_budget(self._build_prompt(query, context))
        start = time.perf_counter()

        async with self._semaphore:
            for client in self._clients:
                if self._is_circuit_open(client.name):
                    continue
                for attempt in range(self._max_retries + 1):
                    try:
                        text = await asyncio.wait_for(client.complete(prompt), timeout=client.timeout_seconds)
                        self._provider_failures[client.name] = 0
                        text = self._enforce_citation_policy(text, context)
                        return QueryResult(
                            answer=text,
                            sources=[f"{c.source_file}:{c.page}" for c in context],
                            latency_ms=(time.perf_counter() - start) * 1000,
                            provider=client.name,
                            cached=False,
                        )
                    except Exception:
                        self._record_failure(client.name)
                        delay = min(self._policy.base_backoff_seconds * (2**attempt), self._policy.max_backoff_seconds)
                        delay += random.uniform(0, delay)
                        await asyncio.sleep(delay)
                        continue

        fallback_text = self._local_fallback(query, context)
        return QueryResult(
            answer=fallback_text,
            sources=[f"{c.source_file}:{c.page}" for c in context],
            latency_ms=(time.perf_counter() - start) * 1000,
            provider="local_fallback",
            cached=False,
        )

    def _enforce_budget(self, prompt: str) -> str:
        if len(prompt) <= self._policy.max_prompt_chars:
            return prompt
        return prompt[: self._policy.max_prompt_chars]

    @staticmethod
    def _enforce_citation_policy(text: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return text
        if "SOURCE:" in text or "[" in text:
            return text
        first = chunks[0]
        return f"{text}\n\nSource: {first.source_file}:{first.page}"

    @staticmethod
    def _local_fallback(query: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return "I do not have enough grounded context to answer this query."
        joined = " ".join(chunk.abstract or chunk.text[:140] for chunk in chunks[:4])
        return f"Grounded fallback answer for: {query}. Context summary: {joined}"

    def _record_failure(self, provider_name: str) -> None:
        failures = self._provider_failures.get(provider_name, 0) + 1
        self._provider_failures[provider_name] = failures
        if failures >= self._policy.circuit_breaker_failure_threshold:
            self._provider_opened_at[provider_name] = time.monotonic()

    def _is_circuit_open(self, provider_name: str) -> bool:
        opened_at = self._provider_opened_at.get(provider_name)
        if opened_at is None:
            return False
        if (time.monotonic() - opened_at) >= self._policy.circuit_breaker_cooldown_seconds:
            self._provider_opened_at.pop(provider_name, None)
            self._provider_failures[provider_name] = 0
            return False
        return True

    @staticmethod
    def _build_prompt(query: str, chunks: list[Chunk]) -> str:
        context = "\n\n".join(
            f"SOURCE: {chunk.source_file}#{chunk.page}\nABSTRACT: {chunk.abstract}\nTEXT: {chunk.text}" for chunk in chunks
        )
        return (
            "You are a grounded RAG assistant. Use only provided context and cite sources. "
            "If context is insufficient, say so explicitly."
            f"\n\nQUESTION:\n{query}\n\nCONTEXT:\n{context}"
        )
