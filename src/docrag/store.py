"""Persistent vector store (Chroma) plus a small JSON document registry.

The registry is what the UI lists: one record per ingested document with its
file hash (the dedup key), chunk count, pages, warnings and suggested
questions. Chroma holds the chunks themselves.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import Settings, get_settings
from .embeddings import Embedder, get_embedder
from .schema import Chunk, DocumentRecord

# Chroma's telemetry is noisy and pointless for a local demo.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "none")


class StoreError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class DocStore:
    """Thin wrapper over a persistent Chroma collection."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        collection: str | None = None,
    ):
        self.cfg = settings or get_settings()
        self.cfg.ensure_dirs()
        self.collection_name = collection or self.cfg.collection
        self._embedder = embedder
        self._client = None
        self._collection = None

    # ------------------------------------------------------------------ #
    # lazily-built resources
    # ------------------------------------------------------------------ #
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder(self.cfg)
        return self._embedder

    @property
    def client(self):
        if self._client is None:
            try:
                import chromadb  # type: ignore
                from chromadb.config import Settings as ChromaSettings  # type: ignore
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise StoreError(
                    "chromadb is required for the vector store (pip install chromadb)"
                ) from exc
            self._client = chromadb.PersistentClient(
                path=str(self.cfg.chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,  # we always pass embeddings ourselves
            )
        return self._collection

    def list_collections(self) -> list[str]:
        out = []
        for item in self.client.list_collections():
            out.append(item if isinstance(item, str) else getattr(item, "name", str(item)))
        return sorted(out)

    # ------------------------------------------------------------------ #
    # registry
    # ------------------------------------------------------------------ #
    def _load_registry(self) -> dict[str, Any]:
        path = self.cfg.registry_path
        if not path.exists():
            return {"version": 0, "collections": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"version": 0, "collections": {}}
        data.setdefault("version", 0)
        data.setdefault("collections", {})
        return data

    def _save_registry(self, data: dict[str, Any]) -> None:
        path = self.cfg.registry_path
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file in the same directory, then replace: never leave a
        # half-written registry behind if the process dies.
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(path.parent), delete=False, suffix=".tmp"
        )
        try:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        os.replace(handle.name, path)

    def _bucket(self, data: dict[str, Any]) -> dict[str, Any]:
        bucket = data["collections"].setdefault(
            self.collection_name, {"documents": {}, "embedder": None, "dim": None}
        )
        bucket.setdefault("documents", {})
        return bucket

    @property
    def version(self) -> int:
        """Bumped on every write; used to invalidate the BM25 cache."""
        return int(self._load_registry().get("version", 0))

    def list_documents(self) -> list[DocumentRecord]:
        bucket = self._bucket(self._load_registry())
        records = [DocumentRecord.from_dict(raw) for raw in bucket["documents"].values()]
        return sorted(records, key=lambda r: (r.filename.lower(), r.doc_id))

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        bucket = self._bucket(self._load_registry())
        raw = bucket["documents"].get(doc_id)
        return DocumentRecord.from_dict(raw) if raw else None

    def find_by_hash(self, sha256: str) -> DocumentRecord | None:
        for record in self.list_documents():
            if record.sha256 == sha256:
                return record
        return None

    def embedder_info(self) -> tuple[str | None, int | None]:
        bucket = self._bucket(self._load_registry())
        return bucket.get("embedder"), bucket.get("dim")

    # ------------------------------------------------------------------ #
    # writes
    # ------------------------------------------------------------------ #
    def add_document(self, chunks: Sequence[Chunk], record: DocumentRecord) -> DocumentRecord:
        """Embed and store a document's chunks, then register it."""
        if not chunks:
            raise StoreError(f"{record.filename}: nothing to index (no chunks produced).")

        stored_backend, stored_dim = self.embedder_info()
        embedder = self.embedder
        if stored_dim is not None and stored_dim != embedder.dim:
            raise StoreError(
                f"Collection {self.collection_name!r} was built with {stored_backend!r} "
                f"({stored_dim} dims) but the current backend is {embedder.name!r} "
                f"({embedder.dim} dims). Re-ingest the documents or change "
                "EMBEDDING_BACKEND / COLLECTION back."
            )

        vectors = embedder.embed_documents([c.text for c in chunks])
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata() for c in chunks],
            embeddings=vectors,
        )

        record.n_chunks = len(chunks)
        record.ingested_at = record.ingested_at or _utc_now()

        data = self._load_registry()
        bucket = self._bucket(data)
        bucket["documents"][record.doc_id] = record.to_dict()
        bucket["embedder"] = embedder.name
        bucket["dim"] = embedder.dim
        data["version"] = int(data.get("version", 0)) + 1
        self._save_registry(data)
        return record

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document's chunks and its registry entry."""
        existing = self.get_document(doc_id)
        try:
            self.collection.delete(where={"doc_id": doc_id})
        except Exception as exc:  # pragma: no cover - chroma edge cases
            raise StoreError(f"Could not delete {doc_id} from the vector store: {exc}") from exc

        data = self._load_registry()
        bucket = self._bucket(data)
        removed = bucket["documents"].pop(doc_id, None) is not None
        data["version"] = int(data.get("version", 0)) + 1
        self._save_registry(data)
        return removed or existing is not None

    def reset(self) -> None:
        """Drop the whole collection (used by tests and ``make reset``)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None
        data = self._load_registry()
        data["collections"].pop(self.collection_name, None)
        data["version"] = int(data.get("version", 0)) + 1
        self._save_registry(data)

    # ------------------------------------------------------------------ #
    # reads
    # ------------------------------------------------------------------ #
    @staticmethod
    def _where(doc_ids: Iterable[str] | None) -> dict[str, Any] | None:
        ids = [d for d in (doc_ids or []) if d]
        if not ids:
            return None
        if len(ids) == 1:
            return {"doc_id": ids[0]}
        return {"doc_id": {"$in": ids}}

    def count(self, doc_ids: Iterable[str] | None = None) -> int:
        if doc_ids is None:
            return int(self.collection.count())
        return len(self.all_chunks(doc_ids))

    def all_chunks(self, doc_ids: Iterable[str] | None = None) -> list[Chunk]:
        """Every chunk (optionally filtered by document) - used to build BM25."""
        result = self.collection.get(
            where=self._where(doc_ids), include=["documents", "metadatas"]
        )
        chunks = [
            Chunk.from_store(cid, text or "", meta or {})
            for cid, text, meta in zip(
                result.get("ids") or [],
                result.get("documents") or [],
                result.get("metadatas") or [],
            )
        ]
        return sorted(chunks, key=lambda c: (c.filename.lower(), c.index))

    def query(
        self,
        query_text: str,
        k: int,
        doc_ids: Iterable[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Vector search. Returns ``(chunk, similarity)`` with 1.0 = identical."""
        if self.collection.count() == 0:
            return []
        embedding = self.embedder.embed_query(query_text)
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=max(1, k),
            where=self._where(doc_ids),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        out: list[tuple[Chunk, float]] = []
        for cid, text, meta, distance in zip(ids, documents, metadatas, distances):
            chunk = Chunk.from_store(cid, text or "", meta or {})
            # cosine distance -> similarity
            out.append((chunk, 1.0 - float(distance)))
        return out


def default_store(settings: Settings | None = None) -> DocStore:
    return DocStore(settings)


def data_root() -> Path:
    return get_settings().data_dir
