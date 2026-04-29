from __future__ import annotations

import hashlib
import re
from pathlib import Path

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

try:  # pragma: no cover - optional dependency
    import fitz  # type: ignore
except Exception:  # pragma: no cover
    fitz = None

try:  # pragma: no cover - optional dependency
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None

from fullrag.models.entities import ExtractedUnit


TOC_PATTERNS = [
    re.compile(r"\.{3,}\s*\d+$"),
    re.compile(r"^(chapter|section|contents?)\b", re.IGNORECASE),
    re.compile(r"^\d+(?:\.\d+)*\s+.+\s+\d+$"),
]
HEADING_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")
DOT_LEADER_PATTERN = re.compile(r"\.{3,}\s*\d+$")
PAGE_NUMBER_SUFFIX = re.compile(r"\s\d+$")


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
    """Binary PDF parser with TOC filtering and basic heading-aware section paths."""

    def load(self, path: str) -> list[ExtractedUnit]:
        if fitz is not None:
            units = self._load_with_pymupdf(path)
            if units:
                return units

        if pdfplumber is not None:
            units = self._load_with_pdfplumber(path)
            if units:
                return units

        if PdfReader is None:
            content = Path(path).read_text(encoding="utf-8", errors="ignore")
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            return [
                ExtractedUnit(
                    unit_id=hashlib.sha256(f"{path}:{idx}:{line}".encode()).hexdigest(),
                    text=line,
                    source_file=path,
                    page=1,
                    section_path=["root"],
                    modality="text",
                    abstract=line[:180],
                    metadata={"parser": "text-fallback"},
                )
                for idx, line in enumerate(lines, start=1)
                if not self._is_toc_line(line)
            ]

        reader = PdfReader(path)
        units: list[ExtractedUnit] = []
        current_section = ["root"]

        for page_number, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            lines = [line.strip() for line in raw.splitlines() if line and line.strip()]
            if not lines:
                continue
            if self._is_toc_page(lines):
                continue
            for line_idx, line in enumerate(lines, start=1):
                if self._is_toc_line(line):
                    continue
                heading = HEADING_PATTERN.match(line)
                if heading:
                    current_section = [heading.group(1), heading.group(2)[:100]]
                modality = "table" if "|" in line or "\t" in line else "text"
                unit_id = hashlib.sha256(f"{path}:{page_number}:{line_idx}:{line}".encode()).hexdigest()
                units.append(
                    ExtractedUnit(
                        unit_id=unit_id,
                        text=line,
                        source_file=path,
                        page=page_number,
                        section_path=current_section.copy(),
                        modality=modality,
                        abstract=line[:180],
                        metadata={"parser": "pypdf", "page": page_number},
                    )
                )
        return units

    def _load_with_pymupdf(self, path: str) -> list[ExtractedUnit]:
        units: list[ExtractedUnit] = []
        current_section = ["root"]
        with fitz.open(path) as doc:  # type: ignore[attr-defined]
            for page_number, page in enumerate(doc, start=1):
                blocks = page.get_text("blocks")
                lines = [str(b[4]).strip() for b in blocks if len(b) >= 5 and str(b[4]).strip()]
                if not lines or self._is_toc_page(lines):
                    continue
                for line_idx, line in enumerate(lines, start=1):
                    if self._is_toc_line(line):
                        continue
                    heading = HEADING_PATTERN.match(line)
                    if heading:
                        current_section = [heading.group(1), heading.group(2)[:100]]
                    units.append(
                        ExtractedUnit(
                            unit_id=hashlib.sha256(f"{path}:{page_number}:{line_idx}:{line}".encode()).hexdigest(),
                            text=line,
                            source_file=path,
                            page=page_number,
                            section_path=current_section.copy(),
                            modality="table" if "|" in line or "\t" in line else "text",
                            abstract=line[:180],
                            metadata={"parser": "pymupdf", "page": page_number},
                        )
                    )
        return units

    def _load_with_pdfplumber(self, path: str) -> list[ExtractedUnit]:
        units: list[ExtractedUnit] = []
        current_section = ["root"]
        with pdfplumber.open(path) as pdf:  # type: ignore[attr-defined]
            for page_number, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text() or ""
                lines = [line.strip() for line in raw.splitlines() if line.strip()]
                if not lines or self._is_toc_page(lines):
                    continue
                for line_idx, line in enumerate(lines, start=1):
                    if self._is_toc_line(line):
                        continue
                    heading = HEADING_PATTERN.match(line)
                    if heading:
                        current_section = [heading.group(1), heading.group(2)[:100]]
                    units.append(
                        ExtractedUnit(
                            unit_id=hashlib.sha256(f"{path}:{page_number}:{line_idx}:{line}".encode()).hexdigest(),
                            text=line,
                            source_file=path,
                            page=page_number,
                            section_path=current_section.copy(),
                            modality="table" if "|" in line or "\t" in line else "text",
                            abstract=line[:180],
                            metadata={"parser": "pdfplumber", "page": page_number},
                        )
                    )
        return units

    def _is_toc_page(self, lines: list[str]) -> bool:
        if len(lines) < 6:
            return False
        lowered = [line.lower() for line in lines]
        explicit_contents = any("table of contents" in line or line == "contents" for line in lowered)
        toc_like = sum(1 for line in lines if self._is_toc_line(line))
        dotted = sum(1 for line in lines if DOT_LEADER_PATTERN.search(line))
        numbered_suffix = sum(1 for line in lines if PAGE_NUMBER_SUFFIX.search(line))
        narrative = sum(1 for line in lines if len(line.split()) > 12 and not self._is_toc_line(line))

        score = 0
        if explicit_contents:
            score += 2
        if (toc_like / len(lines)) >= 0.45:
            score += 1
        if (dotted / len(lines)) >= 0.25:
            score += 1
        if (numbered_suffix / len(lines)) >= 0.55:
            score += 1
        if narrative <= max(2, len(lines) // 10):
            score += 1
        return score >= 3

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
