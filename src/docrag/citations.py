"""Citation formatting, parsing, and passage highlighting.

Citation format shown to (and required from) the model:

* paged documents   -> ``[annual-report.pdf, p.4]``
* unpaged documents -> ``[handbook.md, SS Leave policy]`` (section heading)

``SS`` is written as the section sign in the real output; it is a single
character so the regex below can treat it as a marker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .schema import Chunk, Retrieved
from .utils import tokenize

SECTION_MARK = "§"  # section sign

_CITATION_RE = re.compile(
    r"\[\s*(?P<file>[^\[\],]+?)\s*"
    r"(?:,\s*(?:p\.?\s*(?P<page>\d+)|" + SECTION_MARK + r"\s*(?P<section>[^\]]+?))\s*)?\]"
)

MAX_HIGHLIGHT_TERMS = 24


@dataclass(frozen=True)
class Citation:
    filename: str
    page: int | None = None
    section: str | None = None
    raw: str = ""

    def matches(self, chunk: Chunk) -> bool:
        if self.filename.lower() != chunk.filename.lower():
            return False
        if self.page is not None:
            return chunk.page == self.page
        if self.section:
            heading = (chunk.heading or "").lower()
            return self.section.lower() in heading or heading in self.section.lower()
        return True


def format_citation(chunk: Chunk) -> str:
    """The exact citation string the model must reproduce for this chunk."""
    if chunk.page is not None:
        return f"[{chunk.filename}, p.{chunk.page}]"
    if chunk.heading:
        return f"[{chunk.filename}, {SECTION_MARK} {chunk.heading}]"
    return f"[{chunk.filename}]"


def parse_citations(text: str) -> list[Citation]:
    """Extract citations from an answer, in order, without duplicates."""
    found: list[Citation] = []
    seen: set[tuple[str, int | None, str | None]] = set()
    for match in _CITATION_RE.finditer(text or ""):
        filename = (match.group("file") or "").strip()
        if not filename or "." not in filename:
            continue  # avoid matching markdown links / bracketed asides
        page = int(match.group("page")) if match.group("page") else None
        section = (match.group("section") or "").strip() or None
        key = (filename.lower(), page, (section or "").lower() or None)
        if key in seen:
            continue
        seen.add(key)
        found.append(Citation(filename=filename, page=page, section=section, raw=match.group(0)))
    return found


def cited_chunk_ids(answer: str, hits: Sequence[Retrieved]) -> set[str]:
    """Which retrieved chunks the answer actually cites."""
    citations = parse_citations(answer)
    return {
        hit.chunk.chunk_id
        for hit in hits
        for citation in citations
        if citation.matches(hit.chunk)
    }


def highlight_terms(query: str, answer: str = "", passage: str = "") -> list[str]:
    """Terms worth bolding inside a passage: query words plus answer overlap."""
    query_terms = tokenize(query)
    passage_terms = set(tokenize(passage)) if passage else set()
    answer_terms = [t for t in tokenize(answer) if t in passage_terms] if answer else []
    terms: list[str] = []
    for term in query_terms + answer_terms:
        if len(term) > 2 and term not in terms:
            terms.append(term)
    return terms[:MAX_HIGHLIGHT_TERMS]


def highlight_passage(passage: str, query: str, answer: str = "") -> str:
    """Bold the parts of a passage that the query/answer actually hit."""
    terms = highlight_terms(query, answer, passage)
    if not terms:
        return passage
    pattern = re.compile(
        r"(?<![\w*])(" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")(?![\w*])",
        re.IGNORECASE,
    )
    return pattern.sub(lambda m: f"**{m.group(1)}**", passage)


def build_context(hits: Iterable[Retrieved], max_chars: int | None = None) -> str:
    """Render retrieved passages as the numbered context block for the prompt."""
    parts: list[str] = []
    used = 0
    for i, hit in enumerate(hits, start=1):
        citation = format_citation(hit.chunk)
        heading = f" - section: {hit.chunk.heading}" if hit.chunk.heading else ""
        body = hit.chunk.text.strip()
        block = f"<passage id=\"{i}\" cite=\"{citation}\"{heading}>\n{body}\n</passage>"
        if max_chars is not None and used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def sources_table(hits: Sequence[Retrieved]) -> list[dict]:
    """Compact, UI-friendly view of the retrieved passages."""
    return [
        {
            "n": i,
            "citation": format_citation(hit.chunk),
            "filename": hit.chunk.filename,
            "page": hit.chunk.page,
            "heading": hit.chunk.heading,
            "score": round(hit.score, 4),
            "rerank_score": None if hit.rerank_score is None else round(hit.rerank_score, 3),
            "why": hit.why(),
            "chunk_id": hit.chunk.chunk_id,
            "text": hit.chunk.text,
        }
        for i, hit in enumerate(hits, start=1)
    ]
