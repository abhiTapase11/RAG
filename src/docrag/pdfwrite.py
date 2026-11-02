"""A tiny PDF writer.

Just enough PDF to produce a real, multi-page, text-extractable file: it gives
the demo a paged document (so citations look like ``[file.pdf, p.3]``) and gives
the tests a genuine PDF fixture instead of a mock. Helvetica, no images, no
compression.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Sequence

PAGE_WIDTH, PAGE_HEIGHT = 612, 792
MARGIN_X, MARGIN_TOP = 72, 740
FONT_SIZE, LEADING = 11, 15
LINES_PER_PAGE = 44
WRAP_WIDTH = 88


def _escape(text: str) -> str:
    safe = text.encode("latin-1", errors="replace").decode("latin-1")
    return safe.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def paginate(text: str, lines_per_page: int = LINES_PER_PAGE) -> list[list[str]]:
    """Wrap and paginate plain text into pages of lines."""
    lines: list[str] = []
    for paragraph in text.replace("\r\n", "\n").split("\n"):
        stripped = paragraph.rstrip()
        if not stripped:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(stripped, width=WRAP_WIDTH) or [""])
    pages = [lines[i : i + lines_per_page] for i in range(0, len(lines), lines_per_page)]
    return pages or [[""]]


def _content_stream(lines: Sequence[str]) -> bytes:
    parts = [
        "BT",
        f"/F1 {FONT_SIZE} Tf",
        f"{LEADING} TL",
        f"{MARGIN_X} {MARGIN_TOP} Td",
    ]
    for i, line in enumerate(lines):
        if i:
            parts.append("T*")
        parts.append(f"({_escape(line)}) Tj")
    parts.append("ET")
    return ("\n".join(parts) + "\n").encode("latin-1")


def build_pdf(pages: Sequence[Sequence[str]]) -> bytes:
    """Build a PDF from pages of pre-wrapped lines."""
    pages = list(pages) or [[""]]
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    content_ids = [5 + 2 * i for i in range(n)]

    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            "<< /Type /Pages /Count "
            + str(n)
            + " /Kids ["
            + " ".join(f"{pid} 0 R" for pid in page_ids)
            + "] >>"
        ).encode("latin-1"),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    }
    for page_id, content_id, lines in zip(page_ids, content_ids, pages):
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("latin-1")
        stream = _content_stream(lines)
        objects[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"endstream"
        )

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for obj_id in sorted(objects):
        offsets[obj_id] = len(out)
        out += f"{obj_id} 0 obj\n".encode("latin-1") + objects[obj_id] + b"\nendobj\n"

    xref_offset = len(out)
    max_id = max(objects)
    out += f"xref\n0 {max_id + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for obj_id in range(1, max_id + 1):
        out += f"{offsets[obj_id]:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(out)


def text_to_pdf(text: str, path: str | Path) -> Path:
    """Render plain text to a paged PDF on disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_pdf(paginate(text)))
    return path
