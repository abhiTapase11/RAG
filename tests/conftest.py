"""Shared test fixtures.

Tests run fully offline: the dependency-free ``hash`` embedding backend, mock
LLM mode, and a Chroma directory inside ``tmp_path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SAMPLES = ROOT / "data" / "samples"


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Isolated settings: temp data dir, hash embeddings, mock LLM."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setenv("COLLECTION", "test_collection")
    monkeypatch.setenv("USE_RERANKER", "0")

    from docrag import embeddings, retrieval
    from docrag.config import get_settings

    embeddings.reset_embedder_cache()
    retrieval.clear_caches()
    cfg = get_settings(refresh=True)
    cfg.ensure_dirs()
    yield cfg
    embeddings.reset_embedder_cache()
    retrieval.clear_caches()
    get_settings(refresh=True)


@pytest.fixture
def store(settings):
    from docrag.store import DocStore

    return DocStore(settings)


@pytest.fixture
def app(settings, store):
    from docrag.pipeline import DocRag

    return DocRag(settings, store=store)


@pytest.fixture
def sample_markdown() -> Path:
    return SAMPLES / "aurora-sparrow-field-guide.md"


@pytest.fixture
def sample_text() -> Path:
    return SAMPLES / "northwind-remote-work-policy.txt"


@pytest.fixture
def sample_pdf(tmp_path, sample_text) -> Path:
    """The policy sample rendered to a real multi-page PDF."""
    from docrag.pdfwrite import text_to_pdf

    return text_to_pdf(sample_text.read_text(encoding="utf-8"), tmp_path / sample_text.name.replace(".txt", ".pdf"))
