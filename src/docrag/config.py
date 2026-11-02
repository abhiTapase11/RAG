"""Central configuration, read from the environment (``.env`` is honoured)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repo root = two levels up from this file (src/docrag/config.py -> RAG/).
ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """Load ``.env`` if python-dotenv is installed; fall back to a tiny parser."""
    env_path = ROOT / ".env"
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_path, override=False)
        return
    except Exception:
        pass
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env_str(name, "1" if default else "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass
class Settings:
    """Everything tunable in one place."""

    # ---- paths ----
    data_dir: Path = field(default_factory=lambda: Path(_env_str("DATA_DIR", str(ROOT / "data"))))

    # ---- chunking ----
    chunk_target_tokens: int = field(default_factory=lambda: _env_int("CHUNK_TARGET_TOKENS", 650))
    chunk_max_tokens: int = field(default_factory=lambda: _env_int("CHUNK_MAX_TOKENS", 800))
    chunk_min_tokens: int = field(default_factory=lambda: _env_int("CHUNK_MIN_TOKENS", 60))
    chunk_overlap_ratio: float = field(
        default_factory=lambda: _env_float("CHUNK_OVERLAP_RATIO", 0.12)
    )

    # ---- embeddings ----
    embedding_backend: str = field(
        default_factory=lambda: _env_str("EMBEDDING_BACKEND", "sentence-transformers")
    )
    embedding_model: str = field(
        default_factory=lambda: _env_str("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    embedding_batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 32))

    # ---- vector store ----
    collection: str = field(default_factory=lambda: _env_str("COLLECTION", "doc_rag"))

    # ---- retrieval ----
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 6))
    candidate_k: int = field(default_factory=lambda: _env_int("CANDIDATE_K", 24))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    use_reranker: bool = field(default_factory=lambda: _env_bool("USE_RERANKER", False))
    reranker_model: str = field(
        default_factory=lambda: _env_str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    # ---- generation ----
    llm_provider: str = field(default_factory=lambda: _env_str("LLM_PROVIDER", "anthropic"))
    llm_model: str = field(default_factory=lambda: _env_str("LLM_MODEL", "claude-opus-5-5"))
    llm_effort: str = field(default_factory=lambda: _env_str("LLM_EFFORT", "low"))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 4096))
    judge_model: str = field(default_factory=lambda: _env_str("JUDGE_MODEL", "claude-haiku-5-5"))
    mock_llm: bool = field(default_factory=lambda: _env_bool("MOCK_LLM", False))
    # Mock mode has no model to judge groundedness, so it uses a lexical gate.
    mock_min_coverage: float = field(
        default_factory=lambda: _env_float("MOCK_MIN_COVERAGE", 0.75)
    )

    # ---- the sentence shown when the context cannot answer the question ----
    refusal_text: str = "I couldn't find that in the documents."

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "registry.json"

    @property
    def samples_dir(self) -> Path:
        return self.data_dir / "samples"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.chroma_dir, self.uploads_dir):
            path.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings
