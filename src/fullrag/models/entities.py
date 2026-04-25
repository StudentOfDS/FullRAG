from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class ExtractedUnit:
    unit_id: str
    text: str
    source_file: str
    page: int | None
    section_path: list[str]
    modality: str
    abstract: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    abstract: str
    section_path: list[str]
    source_file: str
    page: int | None
    tombstoned: bool = False


@dataclass(slots=True)
class RetrievalCandidate:
    chunk_id: str
    sparse_score: float
    dense_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0


@dataclass(slots=True)
class QueryResult:
    answer: str
    sources: list[str]
    latency_ms: float
    provider: str
    cached: bool
    created_at: datetime = field(default_factory=datetime.utcnow)
