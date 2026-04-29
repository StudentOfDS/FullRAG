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
        grouped = self._group_by_section(units)
        for _, section_units in grouped.items():
            merged_unit = self._merge_units(section_units)
            chunks.extend(self._chunk_unit(merged_unit))
        return chunks

    @staticmethod
    def _group_by_section(units: list[ExtractedUnit]) -> dict[tuple[str, tuple[str, ...], int | None], list[ExtractedUnit]]:
        grouped: dict[tuple[str, tuple[str, ...], int | None], list[ExtractedUnit]] = {}
        for unit in units:
            key = (unit.source_file, tuple(unit.section_path), unit.page)
            grouped.setdefault(key, []).append(unit)
        return grouped

    @staticmethod
    def _merge_units(units: list[ExtractedUnit]) -> ExtractedUnit:
        if len(units) == 1:
            return units[0]
        head = units[0]
        merged_text = "\n".join(u.text for u in units if u.text.strip())
        merged_abstract = " ".join((u.abstract or "") for u in units).strip()[:400]
        return ExtractedUnit(
            unit_id=hashlib.sha256("|".join(u.unit_id for u in units).encode()).hexdigest(),
            text=merged_text,
            source_file=head.source_file,
            page=head.page,
            section_path=head.section_path,
            modality=head.modality,
            abstract=merged_abstract or merged_text[:200],
            metadata={"merged_units": len(units), **head.metadata},
        )

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
