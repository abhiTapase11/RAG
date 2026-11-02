"""File loaders.

Every loader returns a :class:`~docrag.schema.LoadedDocument`: a flat list of
blocks that keep the page number and the section heading they belong to, plus a
list of warnings (empty / scanned pages, unreadable tables, ...).
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from .schema import Block, LoadedDocument
from .utils import normalise_ws

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm"}

# A page with fewer characters than this is reported as probably scanned.
MIN_CHARS_PER_PAGE = 20

_SETEXT_RE = re.compile(r"^(=+|-+)$")
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(\.\d+)*)[.)]?\s+\S")
_SMALL_WORDS = {"of", "and", "the", "a", "an", "in", "for", "to", "with", "at", "on"}
_FENCE = "```"


class UnsupportedFileType(ValueError):
    pass


# --------------------------------------------------------------------------- #
# shared heuristics
# --------------------------------------------------------------------------- #
def looks_like_heading(line: str) -> bool:
    """Heuristic heading test for formats that carry no style information."""
    text = line.strip()
    if not text or len(text) > 90:
        return False
    words = text.split()
    if len(words) > 14:
        return False
    if text.endswith((".", ",", ";")):
        return False
    if text.endswith(":"):
        return len(words) <= 8
    if text.isupper():
        return len(words) <= 12
    if _NUMBERED_HEADING_RE.match(text):
        return len(words) <= 12
    # Title Case, no sentence punctuation inside.
    letters = [w for w in words if w[:1].isalpha()]
    if letters and all(w[:1].isupper() or w.lower() in _SMALL_WORDS for w in letters):
        return len(words) <= 10
    return False


def _paragraphs(raw: str) -> list[str]:
    """Split a text blob into paragraphs on blank lines."""
    chunks = re.split(r"\n\s*\n", raw)
    return [c for c in (normalise_ws(part) for part in chunks) if c]


def _blocks_from_plain_text(
    raw: str, page: int | None, heading: str | None
) -> tuple[list[Block], str | None]:
    """Turn a plain-text page or section into heading + paragraph blocks."""
    blocks: list[Block] = []
    current = heading
    for para in _paragraphs(raw):
        lines = [ln for ln in para.split("\n") if ln.strip()]
        # A one-line paragraph that looks like a heading becomes the section title.
        if len(lines) == 1 and looks_like_heading(lines[0]):
            current = lines[0].strip().rstrip(":")
            blocks.append(Block(text=current, page=page, heading=current, kind="heading"))
            continue
        # Heading line followed by body text (common in PDF text extraction).
        if len(lines) > 1 and looks_like_heading(lines[0]) and not looks_like_heading(lines[1]):
            current = lines[0].strip().rstrip(":")
            blocks.append(Block(text=current, page=page, heading=current, kind="heading"))
            body = normalise_ws(" ".join(lines[1:]))
            if body:
                blocks.append(Block(text=body, page=page, heading=current))
            continue
        blocks.append(Block(text=normalise_ws(" ".join(lines)), page=page, heading=current))
    return blocks, current


# PDF text extraction loses blank lines, so paragraphs are rebuilt from lines:
# a paragraph ends at a sentence terminator once it is at least this long.
MIN_PARAGRAPH_WORDS = 30


def _blocks_from_lines(
    raw: str, page: int | None, heading: str | None
) -> tuple[list[Block], str | None]:
    """Rebuild headings and paragraphs from a page that only has line breaks."""
    blocks: list[Block] = []
    current = heading
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        body = normalise_ws(" ".join(buffer))
        if body:
            blocks.append(Block(text=body, page=page, heading=current))
        buffer = []

    lines = [ln.strip() for ln in raw.replace("\r\n", "\n").split("\n")]
    for line in lines:
        if not line:
            flush()
            continue
        starts_fresh = not buffer or buffer[-1].rstrip().endswith((".", "!", "?", ":"))
        if looks_like_heading(line) and (line.isupper() or starts_fresh):
            flush()
            current = line.rstrip(":")
            blocks.append(Block(text=current, page=page, heading=current, kind="heading"))
            continue
        buffer.append(line)
        words = sum(len(part.split()) for part in buffer)
        if line.endswith((".", "!", "?")) and words >= MIN_PARAGRAPH_WORDS:
            flush()
    flush()
    return blocks, current


# --------------------------------------------------------------------------- #
# per-format loaders
# --------------------------------------------------------------------------- #
def load_pdf(data: bytes, filename: str) -> LoadedDocument:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("pypdf is required to read PDF files (pip install pypdf)") from exc

    reader = PdfReader(io.BytesIO(data))
    blocks: list[Block] = []
    warnings: list[str] = []
    heading: str | None = None

    for page_no, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - corrupt page
            warnings.append(f"{filename}: page {page_no} could not be parsed ({exc}).")
            continue
        if len(raw.strip()) < MIN_CHARS_PER_PAGE:
            warnings.append(
                f"{filename}: page {page_no} has little or no extractable text - it is "
                "probably a scan. Run OCR on it to make that page searchable."
            )
            continue
        page_blocks, heading = _blocks_from_lines(raw, page_no, heading)
        blocks.extend(page_blocks)

    if not blocks:
        warnings.append(f"{filename}: no text could be extracted from any page.")
    return LoadedDocument(
        filename=filename, blocks=blocks, n_pages=len(reader.pages), warnings=warnings
    )


def load_docx(data: bytes, filename: str) -> LoadedDocument:
    try:
        import docx  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "python-docx is required to read .docx files (pip install python-docx)"
        ) from exc

    document = docx.Document(io.BytesIO(data))
    blocks: list[Block] = []
    warnings: list[str] = []
    heading: str | None = None

    for para in document.paragraphs:
        text = normalise_ws(para.text)
        if not text:
            continue
        style = (para.style.name or "") if para.style is not None else ""
        if style.lower().startswith("heading") or style in {"Title", "Subtitle"}:
            heading = text
            blocks.append(Block(text=text, heading=heading, kind="heading"))
        elif text.isupper() and len(text.split()) <= 8:
            heading = text
            blocks.append(Block(text=text, heading=heading, kind="heading"))
        else:
            blocks.append(Block(text=text, heading=heading))

    for table_no, table in enumerate(getattr(document, "tables", []), start=1):
        rows = []
        for row in table.rows:
            cells = [normalise_ws(cell.text) for cell in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            blocks.append(
                Block(text="\n".join(rows), heading=heading or f"Table {table_no}", kind="table")
            )

    if not blocks:
        warnings.append(f"{filename}: the document contains no readable text.")
    # .docx has no fixed pages, so the section heading is the citable unit.
    return LoadedDocument(filename=filename, blocks=blocks, n_pages=None, warnings=warnings)


def load_markdown(data: bytes, filename: str) -> LoadedDocument:
    text = data.decode("utf-8", errors="replace")
    blocks: list[Block] = []
    heading: str | None = None
    buffer: list[str] = []
    code: list[str] = []
    in_code = False

    def flush() -> None:
        nonlocal buffer
        body = normalise_ws(" ".join(buffer))
        if body:
            blocks.append(Block(text=body, heading=heading))
        buffer = []

    lines = text.split("\n")
    for i, raw_line in enumerate(lines):
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith(_FENCE):
            if in_code:
                blocks.append(Block(text="\n".join(code), heading=heading, kind="code"))
                code = []
                in_code = False
            else:
                flush()
                in_code = True
            continue
        if in_code:
            code.append(raw_line)
            continue

        if stripped.startswith("#"):
            flush()
            heading = stripped.lstrip("#").strip()
            blocks.append(Block(text=heading, heading=heading, kind="heading"))
            continue

        # Setext heading: a line of text underlined with === or ---.
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if stripped and not buffer and next_line and _SETEXT_RE.match(next_line):
            flush()
            heading = stripped
            blocks.append(Block(text=heading, heading=heading, kind="heading"))
            continue
        if stripped and _SETEXT_RE.match(stripped):
            continue  # the underline itself

        if not stripped:
            flush()
            continue
        buffer.append(stripped)

    if in_code and code:
        blocks.append(Block(text="\n".join(code), heading=heading, kind="code"))
    flush()

    warnings = [] if blocks else [f"{filename}: the file is empty."]
    return LoadedDocument(filename=filename, blocks=blocks, warnings=warnings)


def load_text(data: bytes, filename: str) -> LoadedDocument:
    text = data.decode("utf-8", errors="replace")
    blocks, _ = _blocks_from_plain_text(text, None, None)
    warnings = [] if blocks else [f"{filename}: the file is empty."]
    return LoadedDocument(filename=filename, blocks=blocks, warnings=warnings)


def load_html(data: bytes, filename: str) -> LoadedDocument:
    decoded = data.decode("utf-8", errors="replace")
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        # Fall back to a crude tag strip so HTML never hard-fails.
        doc = load_text(re.sub(r"<[^>]+>", "\n", decoded).encode("utf-8"), filename)
        doc.warnings.append(
            f"{filename}: beautifulsoup4 is not installed, so the HTML structure was ignored."
        )
        return doc

    soup = BeautifulSoup(decoded, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    blocks: list[Block] = []
    heading: str | None = None
    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "td"]):
        text = normalise_ws(element.get_text(" ", strip=True))
        if not text:
            continue
        if element.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            heading = text
            blocks.append(Block(text=text, heading=heading, kind="heading"))
        elif element.name == "pre":
            blocks.append(Block(text=text, heading=heading, kind="code"))
        else:
            blocks.append(Block(text=text, heading=heading))

    warnings = [] if blocks else [f"{filename}: no readable text found in the HTML."]
    return LoadedDocument(filename=filename, blocks=blocks, warnings=warnings)


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
_LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_text,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".html": load_html,
    ".htm": load_html,
}


def load_bytes(data: bytes, filename: str) -> LoadedDocument:
    """Load a document from raw bytes, dispatching on the filename extension."""
    ext = Path(filename).suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise UnsupportedFileType(
            f"{filename}: unsupported file type {ext!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return loader(data, Path(filename).name)


def load_document(path: str | Path) -> LoadedDocument:
    """Load a document from disk."""
    path = Path(path)
    return load_bytes(path.read_bytes(), path.name)
