from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from fullrag.models.entities import Chunk, ExtractedUnit, QueryResult


class IngestionService(ABC):
    @abstractmethod
    def ingest(self, paths: list[str]) -> list[ExtractedUnit]:
        raise NotImplementedError


class ChunkingService(ABC):
    @abstractmethod
    def chunk(self, units: list[ExtractedUnit]) -> list[Chunk]:
        raise NotImplementedError


class IndexingService(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def mark_tombstoned(self, document_id: str) -> int:
        raise NotImplementedError


class RetrievalService(ABC):
    @abstractmethod
    async def retrieve(self, query: str) -> list[Chunk]:
        raise NotImplementedError


class GenerationService(ABC):
    @abstractmethod
    async def answer(self, query: str, context: Iterable[Chunk]) -> QueryResult:
        raise NotImplementedError
