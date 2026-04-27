from __future__ import annotations

from fullrag.ingestion.loaders import PdfLoader


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
