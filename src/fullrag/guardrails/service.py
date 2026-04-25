from __future__ import annotations

import re


class QueryGuardrails:
    def __init__(self) -> None:
        self._injection = [
            re.compile(r"ignore previous instructions", re.IGNORECASE),
            re.compile(r"system prompt", re.IGNORECASE),
            re.compile(r"exfiltrate", re.IGNORECASE),
        ]

    def validate(self, query: str) -> tuple[bool, str | None]:
        if len(query.strip()) < 3:
            return False, "Query is too short"
        for pattern in self._injection:
            if pattern.search(query):
                return False, "Potential prompt injection detected"
        return True, None

    def preprocess(self, query: str) -> str:
        replacements = {"rag": "retrieval augmented generation", "llm": "large language model"}
        normalized = " ".join(query.strip().split())
        for src, dst in replacements.items():
            normalized = re.sub(rf"\b{src}\b", dst, normalized, flags=re.IGNORECASE)
        return normalized
