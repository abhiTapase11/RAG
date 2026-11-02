"""Hybrid retrieval: vector search + BM25, fused with reciprocal rank fusion,
with optional cross-encoder reranking and follow-up query rewriting.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .config import Settings, get_settings
from .schema import Chunk, Retrieved
from .store import DocStore
from .utils import stem_tokens, tokenize

logger = logging.getLogger(__name__)

_FOLLOWUP_MARKERS = {
    "it", "its", "it's", "they", "them", "their", "that", "this", "those", "these",
    "he", "she", "his", "her", "there", "same", "also", "too", "instead", "one",
}
_FOLLOWUP_OPENERS = ("what about", "how about", "and ", "but ", "why", "who", "when", "where")


# --------------------------------------------------------------------------- #
# reciprocal rank fusion
# --------------------------------------------------------------------------- #
def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Fuse ranked id lists into one score map.

    Each list contributes ``weight / (k + rank)`` per id, rank being 0-based.
    Ids missing from a list simply contribute nothing, which is what makes RRF
    robust to scores that are not comparable across retrievers.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must have the same length as rankings")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item_id in enumerate(ranking):
            if item_id is None:
                continue
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank + 1)
    return scores


def fuse_ranked(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """RRF, returned as a ranked list (highest score first, ties broken by id)."""
    scores = reciprocal_rank_fusion(rankings, k=k, weights=weights)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #
class BM25Index:
    """BM25-Okapi over chunk texts.

    Uses ``rank_bm25`` when installed; otherwise falls back to the small
    implementation below so keyword search always works.
    """

    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks = list(chunks)
        self.corpus = [stem_tokens(c.text) for c in self.chunks]
        self._impl = None
        # Corpus statistics are always built: scoring may come from rank_bm25,
        # but the idf table below is also used as a term-rarity signal.
        self._build_fallback()
        if self.corpus:
            try:
                from rank_bm25 import BM25Okapi  # type: ignore

                self._impl = BM25Okapi(self.corpus)
            except Exception:
                self._impl = None

    # -- fallback implementation ------------------------------------------- #
    def _build_fallback(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.freqs = [Counter(doc) for doc in self.corpus]
        df: Counter = Counter()
        for doc in self.corpus:
            df.update(set(doc))
        n = len(self.corpus)
        self.idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()
        }

    def _fallback_scores(self, terms: Sequence[str]) -> list[float]:
        scores = [0.0] * len(self.corpus)
        for i, freq in enumerate(self.freqs):
            length = self.doc_len[i] or 1
            total = 0.0
            for term in terms:
                tf = freq.get(term, 0)
                if not tf:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * length / (self.avgdl or 1))
                total += idf * (tf * (self.k1 + 1)) / denom
            scores[i] = total
        return scores

    # -- inverse document frequency ---------------------------------------- #
    @property
    def max_idf(self) -> float:
        """Weight for a term that appears nowhere in the corpus."""
        n = len(self.corpus)
        return math.log(1 + (n + 0.5) / 0.5)

    def idf_of(self, term: str) -> float:
        """Rarity weight for a term; unseen terms get the maximum."""
        if not self.corpus:
            return 1.0
        table = self.idf
        if term not in table:
            return self.max_idf
        return max(0.01, float(table[term]))

    # -- public ------------------------------------------------------------ #
    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        terms = stem_tokens(query)
        if not terms:
            return []
        if self._impl is not None:
            scores = list(self._impl.get_scores(terms))
        else:
            scores = self._fallback_scores(terms)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.chunks[i].chunk_id))
        hits = [(self.chunks[i], float(scores[i])) for i in order[: max(1, k)]]
        return [(chunk, score) for chunk, score in hits if score > 0.0]


_bm25_cache: dict[tuple[str, int, tuple[str, ...]], BM25Index] = {}


def get_bm25_index(store: DocStore, doc_ids: Iterable[str] | None = None) -> BM25Index:
    """Build (and cache, per store version) a BM25 index over the stored chunks."""
    ids = tuple(sorted(d for d in (doc_ids or []) if d))
    key = (store.collection_name, store.version, ids)
    index = _bm25_cache.get(key)
    if index is None:
        index = BM25Index(store.all_chunks(ids or None))
        _bm25_cache.clear()  # only the newest version is worth keeping
        _bm25_cache[key] = index
    return index


# --------------------------------------------------------------------------- #
# cross-encoder reranking
# --------------------------------------------------------------------------- #
class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder  # type: ignore

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        return [float(s) for s in self._model.predict(pairs)]


_reranker_cache: dict[str, CrossEncoderReranker | None] = {}


def get_reranker(settings: Settings | None = None) -> CrossEncoderReranker | None:
    """Load the cross-encoder, or return ``None`` if it is unavailable."""
    cfg = settings or get_settings()
    name = cfg.reranker_model
    if name not in _reranker_cache:
        try:
            _reranker_cache[name] = CrossEncoderReranker(name)
        except Exception as exc:
            logger.warning("Cross-encoder reranker %r unavailable: %s", name, exc)
            _reranker_cache[name] = None
    return _reranker_cache[name]


# --------------------------------------------------------------------------- #
# follow-up query rewriting
# --------------------------------------------------------------------------- #
def looks_like_followup(query: str) -> bool:
    text = query.strip().lower()
    if not text:
        return False
    words = tokenize(text, drop_stopwords=False)
    if any(w in _FOLLOWUP_MARKERS for w in words):
        return True
    if len(words) <= 4:
        return True
    return text.startswith(_FOLLOWUP_OPENERS)


def _keywords(text: str, limit: int = 6) -> list[str]:
    seen: list[str] = []
    for token in stem_tokens(text):
        if token not in seen:
            seen.append(token)
    return seen[:limit]


def rewrite_query(
    query: str,
    history: Sequence[tuple[str, str]] | None = None,
    llm=None,
    settings: Settings | None = None,
) -> str:
    """Make a follow-up question standalone.

    ``history`` is a list of ``(role, content)`` pairs, oldest first. With an
    LLM the rewrite is a one-shot call; without one (or in mock mode) the
    keywords of the previous user turn are appended, which is enough to keep
    retrieval on topic.
    """
    cfg = settings or get_settings()
    question = query.strip()
    if not history or not question or not looks_like_followup(question):
        return question

    previous_user = [content for role, content in history if role == "user"]
    previous_assistant = [content for role, content in history if role == "assistant"]
    if not previous_user:
        return question

    if llm is not None and getattr(llm, "supports_rewrite", False):
        turns = "\n".join(
            f"{role.upper()}: {content.strip()}" for role, content in history[-6:] if content.strip()
        )
        prompt = (
            "Rewrite the follow-up question so it can be understood on its own, "
            "without the conversation. Keep it short, keep the user's wording where "
            "possible, resolve pronouns, and output only the rewritten question.\n\n"
            f"Conversation:\n{turns}\n\nFollow-up: {question}\n\nStandalone question:"
        )
        try:
            rewritten = llm.complete(prompt, max_tokens=120).strip().strip('"')
            first_line = rewritten.splitlines()[0].strip() if rewritten else ""
            if 3 <= len(first_line) <= 300:
                return first_line
        except Exception as exc:
            logger.warning("Query rewriting failed, using the heuristic instead: %s", exc)

    context_words = _keywords(previous_user[-1]) + _keywords(
        previous_assistant[-1] if previous_assistant else "", 3
    )
    extra = [w for w in context_words if w not in set(stem_tokens(question))]
    if not extra:
        return question
    return f"{question} ({' '.join(extra[:6])})"


# --------------------------------------------------------------------------- #
# the retriever
# --------------------------------------------------------------------------- #
@dataclass
class RetrievalResult:
    query: str
    rewritten_query: str
    hits: list[Retrieved]
    reranked: bool = False

    def __iter__(self):
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)


class Retriever:
    """Hybrid retriever over one Chroma collection."""

    def __init__(self, store: DocStore | None = None, settings: Settings | None = None):
        self.cfg = settings or get_settings()
        self.store = store or DocStore(self.cfg)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        doc_ids: Iterable[str] | None = None,
        use_reranker: bool | None = None,
        candidate_k: int | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.cfg.top_k
        candidate_k = max(candidate_k or self.cfg.candidate_k, top_k)
        want_rerank = self.cfg.use_reranker if use_reranker is None else use_reranker
        ids = [d for d in (doc_ids or []) if d]

        vector_hits = self.store.query(query, candidate_k, ids or None)
        bm25_hits = get_bm25_index(self.store, ids or None).search(query, candidate_k)

        by_id: dict[str, Chunk] = {}
        vector_rank: dict[str, int] = {}
        bm25_rank: dict[str, int] = {}
        for rank, (chunk, _score) in enumerate(vector_hits):
            by_id.setdefault(chunk.chunk_id, chunk)
            vector_rank.setdefault(chunk.chunk_id, rank)
        for rank, (chunk, _score) in enumerate(bm25_hits):
            by_id.setdefault(chunk.chunk_id, chunk)
            bm25_rank.setdefault(chunk.chunk_id, rank)

        fused = fuse_ranked(
            [[c.chunk_id for c, _ in vector_hits], [c.chunk_id for c, _ in bm25_hits]],
            k=self.cfg.rrf_k,
        )

        hits = [
            Retrieved(
                chunk=by_id[chunk_id],
                score=score,
                vector_rank=vector_rank.get(chunk_id),
                bm25_rank=bm25_rank.get(chunk_id),
            )
            for chunk_id, score in fused
            if chunk_id in by_id
        ]

        reranked = False
        if want_rerank and hits:
            reranker = get_reranker(self.cfg)
            if reranker is not None:
                pool = hits[: max(top_k * 3, 12)]
                scores = reranker.score(query, [h.chunk.text for h in pool])
                for hit, score in zip(pool, scores):
                    hit.rerank_score = score
                pool.sort(key=lambda h: (-(h.rerank_score or 0.0), h.chunk.chunk_id))
                hits = pool + hits[len(pool) :]
                reranked = True

        return RetrievalResult(
            query=query, rewritten_query=query, hits=hits[:top_k], reranked=reranked
        )

    def idf_lookup(self, doc_ids: Iterable[str] | None = None):
        """Term-rarity lookup over the indexed corpus (used by the mock gate)."""
        index = get_bm25_index(self.store, [d for d in (doc_ids or []) if d] or None)
        return index.idf_of

    def search(
        self,
        query: str,
        history: Sequence[tuple[str, str]] | None = None,
        llm=None,
        **kwargs,
    ) -> RetrievalResult:
        """Rewrite a follow-up question, then retrieve."""
        rewritten = rewrite_query(query, history, llm=llm, settings=self.cfg)
        result = self.retrieve(rewritten, **kwargs)
        result.query = query
        result.rewritten_query = rewritten
        return result


def clear_caches() -> None:
    _bm25_cache.clear()
    _reranker_cache.clear()
