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
