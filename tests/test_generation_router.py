from __future__ import annotations

import asyncio
from unittest.mock import patch

from fullrag.generation.service import GeneratorPolicy, MultiProviderGenerator, ProviderClient


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


def test_provider_payload_uses_policy_values():
    captured: dict = {}

    client = ProviderClient(
        name="openai",
        timeout_seconds=1,
        model="gpt-4o-mini",
        api_key="k",
        temperature=0.7,
        max_output_tokens=321,
    )

    def fake_post(url: str, payload: dict, headers=None):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    with patch.object(ProviderClient, "_post_json", side_effect=fake_post):
        out = client._openai("hello")
    assert out == "ok"
    assert captured["payload"]["temperature"] == 0.7
    assert captured["payload"]["max_tokens"] == 321


def test_generator_policy_is_attached():
    policy = GeneratorPolicy(max_prompt_chars=999, max_retries=1)
    gen = MultiProviderGenerator(providers=["openai"], policy=policy, secrets={})
    assert gen._policy.max_prompt_chars == 999


def test_prompt_budget_preserves_source_headers():
    policy = GeneratorPolicy(max_prompt_chars=220)
    gen = MultiProviderGenerator(providers=["openai"], policy=policy, secrets={})
    chunk_text = " ".join(["word"] * 200)
    from fullrag.models.entities import Chunk

    chunks = [
        Chunk(
            chunk_id="c1",
            document_id="d1",
            text=chunk_text,
            abstract="summary",
            section_path=["root"],
            source_file="doc.pdf",
            page=2,
        )
    ]
    prompt = gen._build_prompt("what?", chunks)
    assert len(prompt) <= 220 + 40  # small overhead for truncation marker/newlines
    assert "SOURCE: doc.pdf#2" in prompt
