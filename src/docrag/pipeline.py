"""The façade the UI, the scripts and the tests all use.

Ingestion:  bytes -> loader -> chunker -> embeddings -> Chroma + registry
Question:   query -> rewrite -> hybrid retrieval -> (rerank) -> grounded answer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .chunking import chunk_document
from .citations import cited_chunk_ids, sources_table
from .config import Settings, get_settings
from .generation import (
    LLMClient,
    get_llm,
    stream_answer,
    stream_summary,
    suggest_questions,
)
from .loaders import SUPPORTED_EXTENSIONS, load_bytes
from .retrieval import RetrievalResult, Retriever
from .schema import Chunk, DocumentRecord, Retrieved
from .store import DocStore
from .utils import doc_id_for, sha256_bytes

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    filename: str
    record: DocumentRecord | None = None
    chunks: list[Chunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False
    reason: str = ""

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)


@dataclass
class Answer:
    """A prepared answer: retrieval is already done, generation is lazy."""

    query: str
    rewritten_query: str
    hits: list[Retrieved]
    _stream_factory: object = None
    text: str = ""

    def stream(self) -> Iterator[str]:
        """Stream the answer, accumulating it into ``self.text``."""
        if self._stream_factory is None:  # pragma: no cover - defensive
            return
        for piece in self._stream_factory():  # type: ignore[operator]
            self.text += piece
            yield piece

    def collect(self) -> str:
        if not self.text:
            for _ in self.stream():
                pass
        return self.text

    def sources(self) -> list[dict]:
        rows = sources_table(self.hits)
        cited = cited_chunk_ids(self.text, self.hits) if self.text else set()
        for row in rows:
            row["cited"] = row["chunk_id"] in cited
        return rows


# --------------------------------------------------------------------------- #
# ingestion
# --------------------------------------------------------------------------- #
def ingest_bytes(
    data: bytes,
    filename: str,
    store: DocStore | None = None,
    settings: Settings | None = None,
    force: bool = False,
    save_copy: bool = True,
) -> IngestResult:
    """Index one document. Re-uploads of the same bytes are skipped."""
    cfg = settings or get_settings()
    store = store or DocStore(cfg)
    name = Path(filename).name

    sha = sha256_bytes(data)
    doc_id = doc_id_for(sha)
    existing = store.find_by_hash(sha)
    if existing is not None and not force:
        return IngestResult(
            filename=name,
            record=existing,
            skipped=True,
            reason=(
                f"{name} is already indexed as {existing.filename} "
                f"({existing.n_chunks} chunks, same SHA-256)."
            ),
        )

    document = load_bytes(data, name)
    chunks = chunk_document(document, doc_id, cfg)
    if not chunks:
        return IngestResult(
            filename=name,
            warnings=document.warnings,
            skipped=True,
            reason=f"{name}: no text could be indexed. {' '.join(document.warnings)}".strip(),
        )

    if save_copy:
        cfg.ensure_dirs()
        try:
            (cfg.uploads_dir / f"{doc_id}_{name}").write_bytes(data)
        except OSError as exc:  # pragma: no cover - disk issues only
            document.warnings.append(f"{name}: could not keep a copy of the upload ({exc}).")

    record = DocumentRecord(
        doc_id=doc_id,
        filename=name,
        sha256=sha,
        n_chunks=len(chunks),
        n_pages=document.n_pages,
        bytes=len(data),
        warnings=list(document.warnings),
        suggested_questions=suggest_questions(chunks),
    )
    if existing is not None and force:
        store.delete_document(existing.doc_id)
    record = store.add_document(chunks, record)
    return IngestResult(
        filename=name, record=record, chunks=chunks, warnings=list(document.warnings)
    )


def ingest_path(
    path: str | Path,
    store: DocStore | None = None,
    settings: Settings | None = None,
    force: bool = False,
) -> IngestResult:
    path = Path(path)
    return ingest_bytes(
        path.read_bytes(), path.name, store=store, settings=settings, force=force, save_copy=False
    )


def ingest_directory(
    directory: str | Path,
    store: DocStore | None = None,
    settings: Settings | None = None,
    force: bool = False,
) -> list[IngestResult]:
    """Ingest every supported file in a directory (non-recursive, sorted)."""
    directory = Path(directory)
    results: list[IngestResult] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            results.append(ingest_path(path, store=store, settings=settings, force=force))
    return results


# --------------------------------------------------------------------------- #
# the app-level object
# --------------------------------------------------------------------------- #
class DocRag:
    """Everything the UI needs, in one object."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: DocStore | None = None,
        llm: LLMClient | None = None,
    ):
        self.cfg = settings or get_settings()
        self.store = store or DocStore(self.cfg)
        self.retriever = Retriever(self.store, self.cfg)
        self._llm = llm

    # -- plumbing ---------------------------------------------------------- #
    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = get_llm(self.cfg)
        return self._llm

    # -- documents --------------------------------------------------------- #
    def documents(self) -> list[DocumentRecord]:
        return self.store.list_documents()

    def document(self, doc_id: str) -> DocumentRecord | None:
        return self.store.get_document(doc_id)

    def ingest(self, data: bytes, filename: str, force: bool = False) -> IngestResult:
        return ingest_bytes(data, filename, store=self.store, settings=self.cfg, force=force)

    def ingest_file(self, path: str | Path, force: bool = False) -> IngestResult:
        return ingest_path(path, store=self.store, settings=self.cfg, force=force)

    def delete(self, doc_id: str) -> bool:
        return self.store.delete_document(doc_id)

    # -- questions --------------------------------------------------------- #
    def ask(
        self,
        query: str,
        history: Sequence[tuple[str, str]] | None = None,
        top_k: int | None = None,
        doc_ids: Iterable[str] | None = None,
        use_reranker: bool | None = None,
    ) -> Answer:
        """Retrieve now, generate when the caller starts streaming."""
        result: RetrievalResult = self.retriever.search(
            query,
            history=history,
            llm=self.llm,
            top_k=top_k,
            doc_ids=doc_ids,
            use_reranker=use_reranker,
        )
        answer = Answer(
            query=query, rewritten_query=result.rewritten_query, hits=list(result.hits)
        )
        answer._stream_factory = lambda: stream_answer(
            query,
            answer.hits,
            llm=self.llm,
            history=history,
            rewritten=result.rewritten_query,
            settings=self.cfg,
            idf=self.retriever.idf_lookup(doc_ids),
        )
        return answer

    def summarize(self, doc_id: str, max_chunks: int = 10) -> Answer:
        """A grounded summary of one document, built from a spread of chunks."""
        record = self.document(doc_id)
        filename = record.filename if record else doc_id
        chunks = self.store.all_chunks([doc_id])
        if chunks and len(chunks) > max_chunks:
            step = len(chunks) / max_chunks
            chunks = [chunks[int(i * step)] for i in range(max_chunks)]
        hits = [Retrieved(chunk=chunk, score=1.0) for chunk in chunks]
        answer = Answer(
            query=f"Summarise {filename}", rewritten_query=f"Summarise {filename}", hits=hits
        )
        answer._stream_factory = lambda: stream_summary(
            filename, hits, llm=self.llm, settings=self.cfg
        )
        return answer

    # -- misc -------------------------------------------------------------- #
    def stats(self) -> dict:
        backend, dim = self.store.embedder_info()
        records = self.documents()
        return {
            "collection": self.store.collection_name,
            "documents": len(records),
            "chunks": sum(r.n_chunks for r in records),
            "embedder": backend or self.cfg.embedding_backend,
            "dim": dim,
            "llm": self.llm.name,
            "mock": self.cfg.mock_llm or self.llm.name == "mock",
        }
