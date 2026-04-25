from __future__ import annotations

from pathlib import Path

from fullrag.core.interfaces import IngestionService
from fullrag.ingestion.loaders import CodebaseLoader, PdfLoader, TextLoader
from fullrag.models.entities import ExtractedUnit


class IngestionPipeline(IngestionService):
    def __init__(self) -> None:
        self._text_loader = TextLoader()
        self._pdf_loader = PdfLoader()
        self._code_loader = CodebaseLoader()

    def ingest(self, paths: list[str]) -> list[ExtractedUnit]:
        units: list[ExtractedUnit] = []
        for path in paths:
            ext = Path(path).suffix.lower()
            if ext in {".txt", ".md"}:
                units.extend(self._text_loader.load(path))
            elif ext == ".pdf":
                units.extend(self._pdf_loader.load(path))
            elif Path(path).is_dir():
                units.extend(self._code_loader.load(path))
        return units
