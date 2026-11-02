"""Loader tests: text, markdown, PDF (real file), DOCX, HTML, error cases."""

from __future__ import annotations

import io

import pytest

from docrag.loaders import (
    SUPPORTED_EXTENSIONS,
    UnsupportedFileType,
    load_bytes,
    load_document,
    looks_like_heading,
)
from docrag.pdfwrite import build_pdf


def test_markdown_keeps_headings(sample_markdown):
    doc = load_document(sample_markdown)

    assert doc.filename == "aurora-sparrow-field-guide.md"
    assert not doc.warnings
    headings = {block.heading for block in doc.blocks if block.heading}
    assert "Identification" in headings
    assert "Conservation Status" in headings

    # Every body block carries the heading of the section it sits in.
    wingspan = [b for b in doc.blocks if "wingspan of 24" in b.text]
    assert wingspan and wingspan[0].heading == "Identification"
    # Markdown has no pages.
    assert all(block.page is None for block in doc.blocks)


def test_markdown_code_fence_is_one_block():
    raw = "# Title\n\nIntro text.\n\n```python\nx = 1\n\ny = 2\n```\n\nAfter.\n"
    doc = load_bytes(raw.encode("utf-8"), "snippet.md")

    code = [b for b in doc.blocks if b.kind == "code"]
    assert len(code) == 1
    assert "x = 1" in code[0].text and "y = 2" in code[0].text


def test_plain_text_detects_upper_case_headings(sample_text):
    doc = load_document(sample_text)

    headings = [block.text for block in doc.blocks if block.kind == "heading"]
    assert "ELIGIBILITY" in headings
    assert "DATA SECURITY" in headings
    body = [b for b in doc.blocks if "650 dollars" in b.text]
    assert body and body[0].heading == "EQUIPMENT AND EXPENSES"


def test_pdf_pages_and_headings(sample_pdf):
    doc = load_document(sample_pdf)

    assert doc.n_pages and doc.n_pages >= 2
    assert not doc.warnings
    pages = {block.page for block in doc.blocks}
    assert pages == set(range(1, doc.n_pages + 1))

    core_hours = [b for b in doc.blocks if "Core hours are 10:00" in b.text]
    assert core_hours, "expected the core-hours paragraph to survive extraction"
    assert core_hours[0].page == 1
    assert core_hours[0].heading == "CORE HOURS AND AVAILABILITY"


def test_pdf_warns_about_pages_without_text():
    pdf = build_pdf(
        [
            ["Page one has plenty of real words on it."],
            [""],  # an empty page, as a scan would extract
            ["Page three also carries a readable sentence."],
        ]
    )

    doc = load_bytes(pdf, "scanned.pdf")

    assert doc.n_pages == 3
    assert len(doc.warnings) == 1
    assert "page 2" in doc.warnings[0]
    assert "OCR" in doc.warnings[0]
    assert {block.page for block in doc.blocks} == {1, 3}


def test_docx_headings_and_tables():
    docx = pytest.importorskip("docx")

    document = docx.Document()
    document.add_heading("Quarterly Report", level=1)
    document.add_paragraph("Revenue reached 4.2 million euro in Q3.")
    document.add_heading("Risks", level=2)
    document.add_paragraph("Fuel costs rose 12 percent.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Region"
    table.cell(0, 1).text = "Share"
    table.cell(1, 0).text = "North"
    table.cell(1, 1).text = "38%"
    buffer = io.BytesIO()
    document.save(buffer)

    doc = load_bytes(buffer.getvalue(), "report.docx")

    assert doc.n_pages is None  # docx has no fixed pagination
    revenue = [b for b in doc.blocks if "4.2 million" in b.text]
    assert revenue and revenue[0].heading == "Quarterly Report"
    risks = [b for b in doc.blocks if "Fuel costs" in b.text]
    assert risks and risks[0].heading == "Risks"
    tables = [b for b in doc.blocks if b.kind == "table"]
    assert tables and "North | 38%" in tables[0].text


def test_html_headings_and_stripped_scripts():
    html = b"""
    <html><body>
      <h1>Service Levels</h1>
      <script>var noise = 1;</script>
      <p>Support responds within 4 hours.</p>
      <h2>Escalation</h2>
      <ul><li>Call the duty engineer.</li></ul>
    </body></html>
    """

    doc = load_bytes(html, "sla.html")

    texts = [block.text for block in doc.blocks]
    assert "Support responds within 4 hours." in texts
    assert not any("noise" in text for text in texts)
    escalation = [b for b in doc.blocks if "duty engineer" in b.text]
    assert escalation and escalation[0].heading == "Escalation"


def test_empty_file_warns():
    doc = load_bytes(b"", "empty.txt")

    assert doc.blocks == []
    assert doc.warnings and "empty" in doc.warnings[0]


def test_unsupported_extension_raises():
    with pytest.raises(UnsupportedFileType) as excinfo:
        load_bytes(b"data", "archive.zip")

    assert ".pdf" in str(excinfo.value)  # the error lists what is supported
    assert ".zip" not in SUPPORTED_EXTENSIONS


@pytest.mark.parametrize(
    "line, expected",
    [
        ("DATA SECURITY", True),
        ("Range and Habitat", True),
        ("3.1 Scope of the policy", True),
        ("Core hours are 10:00 to 16:00 in the registered time zone.", False),
        ("the journey exceeds 150 kilometres each way,", False),
        ("", False),
    ],
)
def test_heading_heuristic(line, expected):
    assert looks_like_heading(line) is expected
