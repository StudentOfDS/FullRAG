from __future__ import annotations

from fullrag.chunking.service import AdaptiveChunker
from fullrag.models.entities import ExtractedUnit


def test_chunker_merges_units_within_same_section():
    chunker = AdaptiveChunker(min_tokens=4, max_tokens=20, overlap=2)
    units = [
        ExtractedUnit(
            unit_id="u1",
            text="Alpha paragraph for section one.",
            source_file="doc.pdf",
            page=1,
            section_path=["1", "Intro"],
            modality="text",
            abstract="Alpha",
        ),
        ExtractedUnit(
            unit_id="u2",
            text="Beta paragraph continues section one.",
            source_file="doc.pdf",
            page=1,
            section_path=["1", "Intro"],
            modality="text",
            abstract="Beta",
        ),
    ]

    chunks = chunker.chunk(units)
    assert len(chunks) == 1
    assert "Alpha paragraph" in chunks[0].text and "Beta paragraph" in chunks[0].text
    assert chunks[0].section_path == ["1", "Intro"]
