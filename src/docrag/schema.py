"""Dataclasses shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Block:
    """A single structural unit of a document (paragraph, list item, heading)."""

    text: str
    page: int | None = None
    heading: str | None = None
    kind: str = "paragraph"  # paragraph | heading | table | code


@dataclass
class LoadedDocument:
    """Raw extraction result for one file."""

    filename: str
    blocks: list[Block]
    n_pages: int | None = None
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


@dataclass
class Chunk:
    """A retrievable passage plus the metadata needed to cite it."""

    chunk_id: str
    text: str
    doc_id: str
    filename: str
    page: int | None = None
    heading: str | None = None
    index: int = 0
    n_tokens: int = 0

    def metadata(self) -> dict[str, Any]:
        """Chroma metadata must be flat scalars, so ``None`` becomes ``-1``/``""``."""
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page": -1 if self.page is None else int(self.page),
            "heading": self.heading or "",
            "index": int(self.index),
            "n_tokens": int(self.n_tokens),
        }

    @classmethod
    def from_store(cls, chunk_id: str, text: str, meta: dict[str, Any]) -> "Chunk":
        page = meta.get("page", -1)
        return cls(
            chunk_id=chunk_id,
            text=text,
            doc_id=str(meta.get("doc_id", "")),
            filename=str(meta.get("filename", "")),
            page=None if page in (-1, None, "") else int(page),
            heading=str(meta.get("heading") or "") or None,
            index=int(meta.get("index", 0) or 0),
            n_tokens=int(meta.get("n_tokens", 0) or 0),
        )


@dataclass
class Retrieved:
    """A chunk with the scores that got it into the answer."""

    chunk: Chunk
    score: float = 0.0
    vector_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None

    @property
    def citation(self) -> str:
        from .citations import format_citation

        return format_citation(self.chunk)

    def why(self) -> str:
        """Short human-readable explanation of how this passage was found."""
        bits = []
        if self.vector_rank is not None:
            bits.append(f"vector #{self.vector_rank + 1}")
        if self.bm25_rank is not None:
            bits.append(f"keyword #{self.bm25_rank + 1}")
        if self.rerank_score is not None:
            bits.append(f"reranker {self.rerank_score:+.2f}")
        return ", ".join(bits) or "n/a"


@dataclass
class DocumentRecord:
    """Registry entry for one indexed document."""

    doc_id: str
    filename: str
    sha256: str
    n_chunks: int
    n_pages: int | None = None
    bytes: int = 0
    ingested_at: str = ""
    warnings: list[str] = field(default_factory=list)
    suggested_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "sha256": self.sha256,
            "n_chunks": self.n_chunks,
            "n_pages": self.n_pages,
            "bytes": self.bytes,
            "ingested_at": self.ingested_at,
            "warnings": list(self.warnings),
            "suggested_questions": list(self.suggested_questions),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DocumentRecord":
        return cls(
            doc_id=raw["doc_id"],
            filename=raw["filename"],
            sha256=raw.get("sha256", ""),
            n_chunks=int(raw.get("n_chunks", 0)),
            n_pages=raw.get("n_pages"),
            bytes=int(raw.get("bytes", 0)),
            ingested_at=raw.get("ingested_at", ""),
            warnings=list(raw.get("warnings", [])),
            suggested_questions=list(raw.get("suggested_questions", [])),
        )
