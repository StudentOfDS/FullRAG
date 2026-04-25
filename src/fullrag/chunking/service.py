from __future__ import annotations

import hashlib
from typing import Iterable

from fullrag.core.interfaces import ChunkingService
from fullrag.models.entities import Chunk, ExtractedUnit


class AdaptiveChunker(ChunkingService):
    def __init__(self, min_tokens: int = 400, max_tokens: int = 900, overlap: int = 80) -> None:
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk(self, units: list[ExtractedUnit]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for unit in units:
            chunks.extend(self._chunk_unit(unit))
        return chunks

    def _chunk_unit(self, unit: ExtractedUnit) -> Iterable[Chunk]:
        words = unit.text.split()
        target = min(max(self.min_tokens, len(words)), self.max_tokens)
        if len(words) <= target:
            yield self._build_chunk(unit, " ".join(words), 0)
            return

        step = max(1, target - self.overlap)
        index = 0
        for i in range(0, len(words), step):
            text = " ".join(words[i : i + target])
            yield self._build_chunk(unit, text, index)
            index += 1

    @staticmethod
    def _build_chunk(unit: ExtractedUnit, text: str, position: int) -> Chunk:
        base = f"{unit.source_file}:{unit.unit_id}:{position}:{text[:80]}"
        chunk_id = hashlib.sha256(base.encode()).hexdigest()
        document_id = hashlib.sha256(unit.source_file.encode()).hexdigest()
        return Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text=text,
            abstract=" ".join(text.split()[:32]),
            section_path=unit.section_path,
            source_file=unit.source_file,
            page=unit.page,
        )
