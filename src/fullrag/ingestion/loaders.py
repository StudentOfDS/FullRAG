from __future__ import annotations

import hashlib
import re
from pathlib import Path

from fullrag.models.entities import ExtractedUnit


TOC_PATTERNS = [
    re.compile(r"\.{3,}\s*\d+$"),
    re.compile(r"^(chapter|section|contents?)\b", re.IGNORECASE),
    re.compile(r"^\d+(?:\.\d+)*\s+.+\s+\d+$"),
]


class BaseLoader:
    def load(self, path: str) -> list[ExtractedUnit]:
        raise NotImplementedError


class TextLoader(BaseLoader):
    def load(self, path: str) -> list[ExtractedUnit]:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        unit_id = hashlib.sha256(f"{path}:{text}".encode()).hexdigest()
        return [
            ExtractedUnit(
                unit_id=unit_id,
                text=text,
                source_file=path,
                page=None,
                section_path=["root"],
                modality="text",
                abstract=text[:200],
            )
        ]


class PdfLoader(BaseLoader):
    """Layout-aware placeholder parser; replace with pymupdf/pdfplumber in production."""

    def load(self, path: str) -> list[ExtractedUnit]:
        content = Path(path).read_text(encoding="utf-8", errors="ignore")
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        units: list[ExtractedUnit] = []
        for idx, line in enumerate(lines, start=1):
            if self._is_toc_line(line):
                continue
            modality = "table" if "|" in line else "text"
            unit_id = hashlib.sha256(f"{path}:{idx}:{line}".encode()).hexdigest()
            units.append(
                ExtractedUnit(
                    unit_id=unit_id,
                    text=line,
                    source_file=path,
                    page=1,
                    section_path=["page_1"],
                    modality=modality,
                    abstract=line[:180],
                )
            )
        return units

    @staticmethod
    def _is_toc_line(line: str) -> bool:
        return any(pattern.search(line) for pattern in TOC_PATTERNS)


class CodebaseLoader(BaseLoader):
    def load(self, path: str) -> list[ExtractedUnit]:
        p = Path(path)
        units: list[ExtractedUnit] = []
        for fp in p.rglob("*.py"):
            text = fp.read_text(encoding="utf-8", errors="ignore")
            uid = hashlib.sha256(f"{fp}:{text}".encode()).hexdigest()
            units.append(
                ExtractedUnit(
                    unit_id=uid,
                    text=text,
                    source_file=str(fp),
                    page=None,
                    section_path=["code", fp.name],
                    modality="code",
                    abstract=(text.splitlines()[0] if text else fp.name)[:200],
                )
            )
        return units
