"""Small helpers: token counting, hashing, tokenisation, sentence splitting."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
# Sentence end: ., ! or ? followed by whitespace and a capital/quote/digit.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")

STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having", "he",
    "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "just", "me", "more", "most", "my", "no", "nor", "not", "now", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would",
    "you", "your", "yours",
}


@lru_cache(maxsize=1)
def _encoder():
    """Return a tiktoken encoder, or ``None`` when tiktoken is unavailable."""
    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Token count via tiktoken when installed, else a word-based estimate."""
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text))
    # ~1.3 tokens per whitespace word is a decent English approximation.
    words = len(text.split())
    return max(1, int(words * 1.3))


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Lowercase word tokens, used by BM25 and the grounding check."""
    tokens = [m.group(0).lower() for m in _WORD_RE.finditer(text)]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


# Words that carry no topic: they say how a question is phrased, not what it is
# about. The mock-mode grounding gate ignores them, so "How LONG must an
# employee WAIT..." is judged on "employee", "remotely" and "work" instead.
GENERIC_TERMS = {
    "able", "allow", "allowed", "ask", "bad", "begin", "best", "big", "bit", "call",
    "come", "end", "find", "get", "give", "go", "good", "happen", "high", "keep",
    "kind", "know", "large", "larger", "last", "let", "like", "list", "long",
    "longer", "look", "lot", "low", "make", "many", "mind", "more", "most", "much",
    "need", "new", "normal", "old", "possible", "put", "say", "see", "show", "small",
    "sort", "start", "take", "tell", "thing", "think", "time", "type", "typical",
    "usual", "use", "wait", "want", "way",
}

# (suffix, replacement) pairs, longest first: a crude stemmer so that
# "incubates" and "incubation" land on the same term.
_SUFFIXES = [
    ("ational", "ate"),
    ("ation", "ate"),
    ("ities", "ity"),
    ("ments", ""),
    ("ment", ""),
    ("ings", ""),
    ("ing", ""),
    ("edly", ""),
    ("ies", "y"),
    ("ed", ""),
    ("able", ""),
    ("ible", ""),
    ("es", ""),
    ("ly", ""),
    ("s", ""),
]


def stem(word: str) -> str:
    """Suffix-stripping stem. Not linguistics - just enough to match variants."""
    stemmed = word.lower()
    for suffix, replacement in _SUFFIXES:
        if stemmed.endswith(suffix) and len(stemmed) - len(suffix) + len(replacement) >= 4:
            stemmed = stemmed[: len(stemmed) - len(suffix)] + replacement
            break
    if len(stemmed) > 4 and stemmed.endswith("e"):
        stemmed = stemmed[:-1]
    return stemmed


# The gate compares stems, so the generic list is stored stemmed too.
_GENERIC_STEMS = {stem(word) for word in GENERIC_TERMS}


def stem_tokens(text: str, drop_stopwords: bool = True) -> list[str]:
    """Tokenise and stem - the term space used by BM25 and the grounding gate."""
    return [stem(token) for token in tokenize(text, drop_stopwords=drop_stopwords)]


def content_terms(text: str) -> list[str]:
    """Stemmed terms that carry topic: no stopwords, no generic question words."""
    seen: list[str] = []
    for token in stem_tokens(text):
        if token in _GENERIC_STEMS:
            continue
        if token not in seen:
            seen.append(token)
    return seen


def split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter - good enough for chunk boundaries."""
    parts = [p.strip() for p in _SENT_RE.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def doc_id_for(sha256: str) -> str:
    """Short, stable document id derived from the file hash (dedup key)."""
    return sha256[:16]


def normalise_ws(text: str) -> str:
    return re.sub(r"[ \t ]+", " ", text.replace("\r\n", "\n")).strip()
