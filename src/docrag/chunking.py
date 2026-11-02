"""Structure-aware chunking.

Rules, in order of priority:

1. Never merge text from two different section headings into one chunk.
2. Never merge text from two different pages when both pages are known
   (keeps page-level citations honest).
3. Fill a chunk up to ``chunk_target_tokens`` and never exceed
   ``chunk_max_tokens``; paragraphs are the atom, sentences are the fallback
   when a single paragraph is larger than the budget.
4. Carry ``chunk_overlap_ratio`` of the previous chunk's tail (whole sentences)
   into the next chunk so answers that straddle a boundary are still findable.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings, get_settings
from .schema import Block, Chunk, LoadedDocument
from .utils import count_tokens, split_sentences


@dataclass
class _Piece:
    """A unit that is small enough to go into a chunk as-is."""

    text: str
    page: int | None
    heading: str | None
    n_tokens: int


def _heading_tokens(heading: str | None) -> int:
    """Tokens a chunk spends on re-stating its section heading."""
    return count_tokens(heading) + 2 if heading else 0


def _explode(blocks: list[Block], max_tokens: int) -> list[_Piece]:
    """Flatten blocks into pieces, splitting over-long ones on sentences.

    A piece must leave room for the heading prefix the chunk will carry, so the
    finished chunk still fits inside ``max_tokens``.
    """
    pieces: list[_Piece] = []
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        if block.kind == "heading":
            # Headings are carried as chunk metadata and re-emitted as a prefix,
            # so they are not pieces of their own.
            continue
        budget = max(32, max_tokens - _heading_tokens(block.heading))
        n_tokens = count_tokens(text)
        if n_tokens <= budget:
            pieces.append(_Piece(text, block.page, block.heading, n_tokens))
            continue
        # Too big: split on sentence boundaries, packing up to the budget.
        buffer: list[str] = []
        buffer_tokens = 0
        for sentence in split_sentences(text):
            s_tokens = count_tokens(sentence)
            if buffer and buffer_tokens + s_tokens > budget:
                joined = " ".join(buffer)
                pieces.append(_Piece(joined, block.page, block.heading, buffer_tokens))
                buffer, buffer_tokens = [], 0
            buffer.append(sentence)
            buffer_tokens += s_tokens
        if buffer:
            joined = " ".join(buffer)
            pieces.append(_Piece(joined, block.page, block.heading, buffer_tokens))
    return pieces


def _overlap_text(text: str, budget_tokens: int) -> str:
    """Return the trailing whole sentences of ``text`` within the token budget."""
    if budget_tokens <= 0:
        return ""
    tail: list[str] = []
    total = 0
    for sentence in reversed(split_sentences(text)):
        s_tokens = count_tokens(sentence)
        if tail and total + s_tokens > budget_tokens:
            break
        tail.insert(0, sentence)
        total += s_tokens
        if total >= budget_tokens:
            break
    return " ".join(tail).strip()


def chunk_blocks(
    blocks: list[Block],
    doc_id: str,
    filename: str,
    settings: Settings | None = None,
) -> list[Chunk]:
    """Chunk a block list into :class:`Chunk` objects."""
    cfg = settings or get_settings()
    target = max(cfg.chunk_target_tokens, 1)
    hard_max = max(cfg.chunk_max_tokens, target)
    overlap_budget = int(target * cfg.chunk_overlap_ratio)

    pieces = _explode(blocks, hard_max)
    chunks: list[Chunk] = []

    buffer: list[_Piece] = []
    carried = ""  # overlap text pulled from the previous chunk
    carried_tokens = 0

    def flush() -> None:
        nonlocal buffer, carried, carried_tokens
        if not buffer:
            return
        first = buffer[0]
        body = "\n\n".join(piece.text for piece in buffer)
        text = f"{carried}\n\n{body}".strip() if carried else body
        heading = first.heading
        if heading and not text.lower().startswith(heading.lower()):
            text = f"{heading}\n\n{text}"
        chunk = Chunk(
            chunk_id=f"{doc_id}:{len(chunks):04d}",
            text=text,
            doc_id=doc_id,
            filename=filename,
            page=first.page,
            heading=heading,
            index=len(chunks),
            n_tokens=count_tokens(text),
        )
        chunks.append(chunk)
        carried = _overlap_text(body, overlap_budget)
        carried_tokens = count_tokens(carried) if carried else 0
        buffer = []

    def used_tokens() -> int:
        if not buffer:
            return carried_tokens
        return (
            carried_tokens
            + _heading_tokens(buffer[0].heading)
            + sum(p.n_tokens for p in buffer)
        )

    for piece in pieces:
        if buffer:
            prev = buffer[-1]
            boundary = piece.heading != prev.heading or (
                piece.page is not None and prev.page is not None and piece.page != prev.page
            )
            used = used_tokens()
            if boundary:
                flush()
                carried, carried_tokens = "", 0  # do not bleed across sections/pages
            elif used + piece.n_tokens > hard_max:
                flush()
            elif used + piece.n_tokens > target and used >= cfg.chunk_min_tokens:
                flush()
        buffer.append(piece)
    flush()

    # Merge a runt tail chunk into its predecessor when they share a heading/page.
    if len(chunks) >= 2 and chunks[-1].n_tokens < cfg.chunk_min_tokens:
        tail = chunks.pop()
        prev = chunks[-1]
        if prev.heading == tail.heading and prev.page == tail.page:
            prev.text = f"{prev.text}\n\n{tail.text}"
            prev.n_tokens = count_tokens(prev.text)
        else:
            chunks.append(tail)
    return chunks


def chunk_document(
    document: LoadedDocument, doc_id: str, settings: Settings | None = None
) -> list[Chunk]:
    """Chunk a loaded document."""
    return chunk_blocks(document.blocks, doc_id, document.filename, settings)
