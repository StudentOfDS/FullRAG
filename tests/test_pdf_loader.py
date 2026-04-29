from __future__ import annotations

from fullrag.ingestion.loaders import PdfLoader
import fullrag.ingestion.loaders as loaders


def test_pdf_loader_toc_page_detection():
    loader = PdfLoader()
    lines = [
        "Contents",
        "1 Intro ........ 1",
        "2 Methods ........ 4",
        "3 Results ........ 8",
        "4 Discussion ........ 10",
        "5 Appendix ........ 20",
    ]
    assert loader._is_toc_page(lines) is True


def test_pdf_loader_does_not_flag_narrative_page_as_toc():
    loader = PdfLoader()
    lines = [
        "This chapter explains the overall architecture and rationale for the retrieval pipeline.",
        "We begin with ingestion constraints and discuss reliability practices in production systems.",
        "The next section provides practical examples with implementation details and tradeoffs.",
        "Operationally, we recommend staged rollouts and regression evaluations for every release.",
        "These recommendations are based on observed failure modes and incident analysis.",
        "Finally, we summarize the key findings and future work.",
    ]
    assert loader._is_toc_page(lines) is False


def test_pdf_loader_uses_pymupdf_when_available(monkeypatch):
    class DummyPage:
        def get_text(self, mode):
            return [(0, 0, 10, 10, "1 Intro"), (0, 11, 10, 20, "Hello world")]

    class DummyDoc:
        def __iter__(self):
            return iter([DummyPage()])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyFitz:
        @staticmethod
        def open(path):
            return DummyDoc()

    monkeypatch.setattr(loaders, "fitz", DummyFitz)
    loader = PdfLoader()
    units = loader.load("dummy.pdf")
    assert units and units[0].metadata.get("parser") == "pymupdf"


def test_table_rows_to_markdown():
    loader = PdfLoader()
    rows = [["Name", "Score"], ["Alice", "10"], ["Bob", "9"]]
    md = loader._table_rows_to_markdown(rows)
    assert "| Name | Score |" in md
    assert "| Alice | 10 |" in md


def test_extract_pdfplumber_tables_builds_table_units():
    class DummyPage:
        def extract_tables(self):
            return [[["Col1", "Col2"], ["A", "1"]]]

    loader = PdfLoader()
    units = loader._extract_pdfplumber_tables("doc.pdf", DummyPage(), 2, ["1", "Intro"])
    assert len(units) == 1
    assert units[0].modality == "table"
    assert units[0].metadata.get("table_index") == 1
