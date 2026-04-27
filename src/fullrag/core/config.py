from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class ConfigLoader:
    def __init__(self, config_path: str = "config.json") -> None:
        self._config_path = Path(config_path)

    def load(self) -> dict[str, Any]:
        raw = json.loads(self._config_path.read_text(encoding="utf-8"))
        self._inject_secrets(raw)
        return raw

    @staticmethod
    def _inject_secrets(config: dict[str, Any]) -> None:
        config.setdefault("secrets", {})
        if config.get("indexing", {}).get("pinecone") is not None and os.getenv("PINECONE_HOST"):
            config["indexing"]["pinecone"]["host"] = os.getenv("PINECONE_HOST")
        config["secrets"].update(
            {
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
                "pinecone_host": os.getenv("PINECONE_HOST"),
                "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
                "gemini_api_key": os.getenv("GEMINI_API_KEY"),
                "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY"),
                "huggingface_api_key": os.getenv("HUGGINGFACE_API_KEY"),
                "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            }
        )
