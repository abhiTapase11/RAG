"""Answer generation.

The LLM sits behind a tiny interface so the provider is swappable, and so the
whole app runs offline in mock mode (``MOCK_LLM=1``), where the answer is the
top retrieved passages stitched together with their citations.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Iterator, Protocol, Sequence, runtime_checkable

from .citations import build_context, format_citation
from .config import Settings, get_settings
from .schema import Chunk, Retrieved
from .utils import content_terms, split_sentences, stem_tokens, tokenize

logger = logging.getLogger(__name__)

class LLMError(RuntimeError):
    pass


SYSTEM_PROMPT = """You are doc-rag, a careful question-answering assistant for a private document library.

Rules you must follow without exception:
1. Answer ONLY from the passages given in <context>. Never use outside knowledge, never guess, never fill gaps with plausible detail.
2. Every factual sentence must end with an inline citation copied verbatim from the passage's cite attribute, for example [report.pdf, p.4]. Cite several sources when several support the sentence.
3. If the passages do not contain the answer, reply with exactly: "{refusal}" - optionally followed by one short sentence saying what the documents do cover. Do not apologise at length and do not speculate.
4. Quote numbers, names and dates exactly as they appear. If passages disagree, say so and cite both.
5. Be concise: a direct answer first, then at most a few supporting sentences. Use short bullets for lists.
"""


def system_prompt(settings: Settings | None = None) -> str:
    cfg = settings or get_settings()
    return SYSTEM_PROMPT.format(refusal=cfg.refusal_text)


def build_user_prompt(query: str, context: str, rewritten: str | None = None) -> str:
    """The per-turn user message: context first, then the question."""
    restated = ""
    if rewritten and rewritten.strip() != query.strip():
        restated = f"\nInterpreted as a standalone question: {rewritten.strip()}\n"
    return (
        "<context>\n"
        f"{context if context.strip() else '(no passages were retrieved)'}\n"
        "</context>\n"
        f"\nQuestion: {query.strip()}\n{restated}"
        "\nAnswer from the context above, with inline citations."
    )


# --------------------------------------------------------------------------- #
# LLM interface
# --------------------------------------------------------------------------- #
@runtime_checkable
class LLMClient(Protocol):
    name: str
    supports_rewrite: bool

    def stream(self, system: str, messages: Sequence[dict]) -> Iterator[str]: ...

    def complete(self, prompt: str, max_tokens: int = 1024, system: str | None = None) -> str: ...


class AnthropicLLM:
    """Anthropic Messages API, streaming."""

    supports_rewrite = True

    def __init__(self, settings: Settings | None = None):
        cfg = settings or get_settings()
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMError("The anthropic package is required (pip install anthropic)") from exc
        if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            # The SDK can also pick up an `ant auth login` profile, so this is a
            # warning rather than a hard failure.
            logger.warning(
                "ANTHROPIC_API_KEY is not set; the SDK will try an `ant auth login` profile. "
                "Set MOCK_LLM=1 to run the demo with no API access at all."
            )
        self.cfg = cfg
        self.model = cfg.llm_model
        self.name = f"anthropic:{cfg.llm_model}"
        self._anthropic = anthropic
        try:
            self._client = anthropic.Anthropic()
        except Exception as exc:  # no key, no auth profile, malformed base URL
            raise LLMError(
                f"Could not create the Anthropic client: {exc} "
                "Put ANTHROPIC_API_KEY in .env (copy .env.example), or set "
                "MOCK_LLM=1 to run offline."
            ) from exc

    def _request_kwargs(self, max_tokens: int) -> dict:
        kwargs: dict = {"model": self.model, "max_tokens": max_tokens}
        if self.cfg.llm_effort:
            kwargs["output_config"] = {"effort": self.cfg.llm_effort}
        return kwargs

    def stream(self, system: str, messages: Sequence[dict]) -> Iterator[str]:
        try:
            with self._client.messages.stream(
                system=system,
                messages=list(messages),
                **self._request_kwargs(self.cfg.llm_max_tokens),
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except self._anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach the Anthropic API: {exc}") from exc

    def complete(self, prompt: str, max_tokens: int = 1024, system: str | None = None) -> str:
        try:
            response = self._client.messages.create(
                system=system or "You are a precise assistant. Answer briefly.",
                messages=[{"role": "user", "content": prompt}],
                **self._request_kwargs(max_tokens),
            )
        except self._anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach the Anthropic API: {exc}") from exc
        return "".join(block.text for block in response.content if block.type == "text")


class MockLLM:
    """Offline stand-in: echoes the retrieved passages as the answer.

    It never invents text, which makes it useful for demoing retrieval quality
    and for tests. ``stream`` reads the ``<passage>`` blocks back out of the
    prompt it was handed, so the plumbing under test is the real one.
    """

    supports_rewrite = False
    name = "mock"

    def __init__(self, settings: Settings | None = None):
        self.cfg = settings or get_settings()

    def stream(self, system: str, messages: Sequence[dict]) -> Iterator[str]:
        last = messages[-1]["content"] if messages else ""
        text = mock_answer_from_prompt(str(last), self.cfg)
        for piece in re.split(r"(\s+)", text):
            if piece:
                yield piece

    def complete(self, prompt: str, max_tokens: int = 1024, system: str | None = None) -> str:
        return mock_answer_from_prompt(prompt, self.cfg)


def get_llm(settings: Settings | None = None) -> LLMClient:
    """Build the configured LLM client, falling back to mock mode when needed."""
    cfg = settings or get_settings()
    if cfg.mock_llm:
        return MockLLM(cfg)
    provider = cfg.llm_provider.strip().lower()
    if provider in {"mock", "none", "offline"}:
        return MockLLM(cfg)
    if provider != "anthropic":
        raise LLMError(
            f"Unknown LLM_PROVIDER {cfg.llm_provider!r}. Supported: 'anthropic', 'mock'."
        )
    try:
        return AnthropicLLM(cfg)
    except LLMError as exc:
        logger.warning("%s - falling back to mock mode.", exc)
        return MockLLM(cfg)


# --------------------------------------------------------------------------- #
# grounding gate
# --------------------------------------------------------------------------- #
def coverage(
    query: str,
    hits: Sequence[Retrieved],
    idf: Callable[[str], float] | None = None,
) -> float:
    """How much of the question's topic vocabulary the passages cover.

    Only topic-bearing stems count (see ``utils.GENERIC_TERMS``), so phrasing
    words like "how long" or "must" cannot drag the score down. With an ``idf``
    lookup the remaining terms are weighted by rarity, so a question hinging on
    a word the corpus has never seen ("vacation", "lifespan") scores far lower
    than one that merely paraphrases.
    """
    terms = set(content_terms(query))
    if not terms:
        return 0.0
    haystack: set[str] = set()
    for hit in hits:
        haystack.update(stem_tokens(hit.chunk.text))
    weight = idf if idf is not None else (lambda _term: 1.0)
    total = sum(weight(term) for term in terms)
    if total <= 0:
        return 0.0
    found = sum(weight(term) for term in terms if term in haystack)
    return found / total


def is_answerable(
    query: str,
    hits: Sequence[Retrieved],
    strict: bool = False,
    min_coverage: float | None = None,
    idf: Callable[[str], float] | None = None,
    settings: Settings | None = None,
) -> bool:
    """Should we even ask the model?

    With a real LLM the model itself decides whether the context answers the
    question, so this only short-circuits when nothing was retrieved. In mock
    mode (``strict=True``) there is no model to judge, so a lexical coverage
    threshold stands in for one.
    """
    if not hits:
        return False
    if not strict:
        return True
    cfg = settings or get_settings()
    threshold = cfg.mock_min_coverage if min_coverage is None else min_coverage
    return coverage(query, hits, idf=idf) >= threshold


# --------------------------------------------------------------------------- #
# mock answers
# --------------------------------------------------------------------------- #
_PASSAGE_RE = re.compile(
    r"<passage id=\"(?P<id>\d+)\" cite=\"(?P<cite>[^\"]+)\"[^>]*>\n(?P<body>.*?)\n</passage>",
    re.DOTALL,
)


def _best_sentences(body: str, terms: set[str], limit: int = 2) -> list[str]:
    scored = []
    for sentence in split_sentences(body):
        overlap = len(terms & set(tokenize(sentence)))
        if overlap:
            scored.append((overlap, len(sentence), sentence))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [sentence for _, _, sentence in scored[:limit]]


def mock_answer_from_prompt(prompt: str, settings: Settings | None = None) -> str:
    """Build the offline answer from the context block inside ``prompt``."""
    cfg = settings or get_settings()
    question_match = re.search(r"\nQuestion: (.+)", prompt)
    question = question_match.group(1).strip() if question_match else ""
    terms = set(tokenize(question))

    passages = list(_PASSAGE_RE.finditer(prompt))
    if not passages:
        return cfg.refusal_text

    lines: list[str] = []
    for match in passages[:3]:
        body = match.group("body").strip()
        cite = match.group("cite").strip()
        picked = _best_sentences(body, terms) or split_sentences(body)[:1]
        for sentence in picked:
            lines.append(f"- {sentence.strip()} {cite}")
        if len(lines) >= 4:
            break

    if not lines:
        return cfg.refusal_text
    header = (
        "Mock mode (no LLM call) - here are the passages that best match your question:"
    )
    return header + "\n" + "\n".join(lines)


# --------------------------------------------------------------------------- #
# the answer stream
# --------------------------------------------------------------------------- #
@dataclass
class AnswerContext:
    query: str
    rewritten_query: str
    hits: list[Retrieved] = field(default_factory=list)
    refused: bool = False
    llm_name: str = ""


def history_messages(
    history: Sequence[tuple[str, str]] | None, limit: int = 6
) -> list[dict]:
    """Previous turns, trimmed, as Anthropic message dicts."""
    messages: list[dict] = []
    for role, content in list(history or [])[-limit:]:
        if role not in {"user", "assistant"} or not str(content).strip():
            continue
        messages.append({"role": role, "content": str(content).strip()})
    # The API requires the first message to be from the user.
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    return messages


def stream_answer(
    query: str,
    hits: Sequence[Retrieved],
    llm: LLMClient | None = None,
    history: Sequence[tuple[str, str]] | None = None,
    rewritten: str | None = None,
    settings: Settings | None = None,
    idf: Callable[[str], float] | None = None,
) -> Iterator[str]:
    """Stream a grounded, cited answer for ``query`` over ``hits``."""
    cfg = settings or get_settings()
    client = llm or get_llm(cfg)
    strict = isinstance(client, MockLLM)

    if not is_answerable(query, hits, strict=strict, idf=idf, settings=cfg):
        yield cfg.refusal_text
        return

    context = build_context(hits)
    messages = history_messages(history)
    messages.append({"role": "user", "content": build_user_prompt(query, context, rewritten)})

    produced = False
    for piece in client.stream(system_prompt(cfg), messages):
        if piece:
            produced = True
            yield piece
    if not produced:
        yield cfg.refusal_text


SUMMARY_INSTRUCTION = (
    "Summarise the document using only the passages below. Produce:\n"
    "1. A one-sentence description of what the document is.\n"
    "2. Three to six bullets with the key facts, each with an inline citation.\n"
    "3. One line listing anything the passages clearly leave open.\n"
    "Cite every bullet. Do not add information that is not in the passages."
)


def stream_summary(
    filename: str,
    hits: Sequence[Retrieved],
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> Iterator[str]:
    """Stream a grounded summary of one document."""
    cfg = settings or get_settings()
    client = llm or get_llm(cfg)
    if not hits:
        yield cfg.refusal_text
        return
    context = build_context(hits)
    prompt = (
        "<context>\n"
        f"{context}\n"
        "</context>\n\n"
        f"Question: Summarise {filename}.\n"
        f"\n{SUMMARY_INSTRUCTION}"
    )
    yield from client.stream(system_prompt(cfg), [{"role": "user", "content": prompt}])


# --------------------------------------------------------------------------- #
# suggested questions (no LLM needed)
# --------------------------------------------------------------------------- #
_NUMBER_RE = re.compile(r"\b\d[\d,.]*\b")


def suggest_questions(chunks: Sequence[Chunk], limit: int = 4) -> list[str]:
    """Heuristic starter questions, from headings and fact-bearing sentences."""
    questions: list[str] = []
    seen_headings: set[str] = set()

    for chunk in chunks:
        heading = (chunk.heading or "").strip(" .:")
        if heading and heading.lower() not in seen_headings and len(heading.split()) <= 10:
            seen_headings.add(heading.lower())
            questions.append(f"What does the section on {heading.lower()} say?")
        if len(questions) >= limit - 1:
            break

    # One question anchored on a concrete number, which exercises exact recall.
    for chunk in chunks:
        for sentence in split_sentences(chunk.text):
            if _NUMBER_RE.search(sentence) and 40 < len(sentence) < 180:
                subject = " ".join(sentence.split()[:9]).rstrip(",.;:")
                questions.append(f"What figure is given for: {subject}?")
                break
        if len(questions) >= limit:
            break

    deduped: list[str] = []
    for question in questions:
        if question not in deduped:
            deduped.append(question)
    return deduped[:limit]


# --------------------------------------------------------------------------- #
# LLM-as-judge (used by scripts/eval.py)
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = (
    "You grade whether an answer is faithful to the given passages. "
    "Faithful means every claim in the answer is supported by the passages, "
    "with no outside knowledge and no invented detail. Reply with JSON only."
)

JUDGE_TEMPLATE = """Passages:
{context}

Question: {question}

Answer to grade:
{answer}

Reply with exactly this JSON and nothing else:
{{"faithful": true|false, "supported_claims": <int>, "unsupported_claims": <int>, "reason": "<one sentence>"}}
"""


@dataclass
class Judgement:
    faithful: bool
    supported: int = 0
    unsupported: int = 0
    reason: str = ""


def judge_faithfulness(
    question: str,
    answer: str,
    hits: Sequence[Retrieved],
    settings: Settings | None = None,
) -> Judgement | None:
    """LLM-as-judge faithfulness check. ``None`` when no judge is available."""
    cfg = settings or get_settings()
    if cfg.mock_llm:
        return None
    try:
        import anthropic  # type: ignore
    except ImportError:
        return None
    client = anthropic.Anthropic()
    prompt = JUDGE_TEMPLATE.format(
        context=build_context(hits, max_chars=12000), question=question, answer=answer
    )
    try:
        response = client.messages.create(
            model=cfg.judge_model,
            max_tokens=512,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        logger.warning("Judge call failed: %s", exc)
        return None

    text = "".join(block.text for block in response.content if block.type == "text")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return Judgement(
        faithful=bool(data.get("faithful")),
        supported=int(data.get("supported_claims", 0) or 0),
        unsupported=int(data.get("unsupported_claims", 0) or 0),
        reason=str(data.get("reason", ""))[:300],
    )


def citation_of(chunk: Chunk) -> str:
    """Re-exported for convenience in the UI layer."""
    return format_citation(chunk)
