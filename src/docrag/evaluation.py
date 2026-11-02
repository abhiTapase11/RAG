"""Retrieval + answer evaluation.

Metrics
-------
hit-rate@k  fraction of answerable questions whose gold chunk is in the top k
MRR         mean reciprocal rank of the first gold chunk
answer-keys fraction of answers containing every expected key fact (string match)
refusals    fraction of unanswerable questions correctly refused
faithful    LLM-as-judge faithfulness (skipped in mock mode)

A "gold chunk" is a retrieved chunk from the expected file whose page matches
(when the case names one) and which contains at least one of the expected key
facts (when the case names any).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .config import Settings, get_settings
from .generation import judge_faithfulness
from .pipeline import DocRag
from .schema import Chunk, Retrieved

logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    id: str
    question: str
    expected_file: str | None = None
    expected_page: int | None = None
    expected_section: str | None = None
    answer_keys: list[str] = field(default_factory=list)
    reference_answer: str = ""
    unanswerable: bool = False
    doc_filter: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], index: int = 0) -> "EvalCase":
        return cls(
            id=str(raw.get("id") or f"q{index + 1}"),
            question=raw["question"],
            expected_file=raw.get("expected_file"),
            expected_page=raw.get("expected_page"),
            expected_section=raw.get("expected_section"),
            answer_keys=[str(k) for k in raw.get("answer_keys", [])],
            reference_answer=str(raw.get("reference_answer", "")),
            unanswerable=bool(raw.get("unanswerable", False)),
            doc_filter=[str(d) for d in raw.get("doc_filter", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "question": self.question}
        if self.unanswerable:
            out["unanswerable"] = True
            return out
        if self.expected_file:
            out["expected_file"] = self.expected_file
        if self.expected_page is not None:
            out["expected_page"] = self.expected_page
        if self.expected_section:
            out["expected_section"] = self.expected_section
        if self.answer_keys:
            out["answer_keys"] = self.answer_keys
        if self.reference_answer:
            out["reference_answer"] = self.reference_answer
        return out


@dataclass
class CaseResult:
    case: EvalCase
    rank: int | None = None  # 1-based rank of the first gold chunk
    retrieved: list[str] = field(default_factory=list)
    answer: str = ""
    answer_ok: bool | None = None
    refused: bool | None = None
    faithful: bool | None = None
    judge_reason: str = ""

    @property
    def hit(self) -> bool:
        return self.rank is not None

    @property
    def rr(self) -> float:
        return 0.0 if self.rank is None else 1.0 / self.rank


@dataclass
class EvalReport:
    k: int
    results: list[CaseResult] = field(default_factory=list)
    judged: bool = False

    # -- aggregate metrics ------------------------------------------------- #
    @property
    def answerable(self) -> list[CaseResult]:
        return [r for r in self.results if not r.case.unanswerable]

    @property
    def unanswerable(self) -> list[CaseResult]:
        return [r for r in self.results if r.case.unanswerable]

    @property
    def hit_rate(self) -> float:
        rows = self.answerable
        return sum(1 for r in rows if r.hit) / len(rows) if rows else 0.0

    @property
    def mrr(self) -> float:
        rows = self.answerable
        return sum(r.rr for r in rows) / len(rows) if rows else 0.0

    @property
    def answer_accuracy(self) -> float | None:
        rows = [r for r in self.answerable if r.answer_ok is not None]
        return (sum(1 for r in rows if r.answer_ok) / len(rows)) if rows else None

    @property
    def refusal_accuracy(self) -> float | None:
        rows = [r for r in self.unanswerable if r.refused is not None]
        return (sum(1 for r in rows if r.refused) / len(rows)) if rows else None

    @property
    def faithfulness(self) -> float | None:
        rows = [r for r in self.results if r.faithful is not None]
        return (sum(1 for r in rows if r.faithful) / len(rows)) if rows else None


# --------------------------------------------------------------------------- #
# gold-chunk matching
# --------------------------------------------------------------------------- #
def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _same_document(filename: str, expected: str) -> bool:
    """Compare by stem, so a .txt sample and its rendered .pdf both match."""
    return Path(filename).stem.lower() == Path(expected).stem.lower()


def chunk_is_gold(chunk: Chunk, case: EvalCase) -> bool:
    if case.expected_file and not _same_document(chunk.filename, case.expected_file):
        return False
    if case.expected_page is not None and chunk.page != case.expected_page:
        return False
    if case.expected_section:
        heading = _normalise(chunk.heading or "")
        section = _normalise(case.expected_section)
        if section and section not in heading and section not in _normalise(chunk.text[:200]):
            return False
    if case.answer_keys:
        haystack = _normalise(chunk.text)
        if not any(_normalise(key) in haystack for key in case.answer_keys):
            return False
    return True


def first_gold_rank(hits: Sequence[Retrieved], case: EvalCase) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if chunk_is_gold(hit.chunk, case):
            return rank
    return None


def answer_contains_keys(answer: str, case: EvalCase) -> bool:
    haystack = _normalise(answer)
    return all(_normalise(key) in haystack for key in case.answer_keys)


def is_refusal(answer: str, settings: Settings | None = None) -> bool:
    cfg = settings or get_settings()
    needle = _normalise(cfg.refusal_text).rstrip(" .")
    return needle in _normalise(answer)


# --------------------------------------------------------------------------- #
# test set I/O
# --------------------------------------------------------------------------- #
def load_testset(path: str | Path) -> list[EvalCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = raw["questions"] if isinstance(raw, dict) else raw
    return [EvalCase.from_dict(item, i) for i, item in enumerate(questions)]


def save_testset(cases: Sequence[EvalCase], path: str | Path) -> None:
    payload = {
        "description": "Q&A test set for doc-rag. Hand-written cases plus any generated ones.",
        "questions": [case.to_dict() for case in cases],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


GENERATE_PROMPT = """Below are passages from the document "{filename}".

Write {n} factual questions a reader might ask, each answerable from these passages alone.

Reply with JSON only, as a list of objects with these keys:
"question" (string), "reference_answer" (string, one sentence),
"answer_keys" (list of 1-3 short exact strings from the passages, e.g. a number or a name).

Passages:
{context}
"""


def generate_testset(
    app: DocRag, per_doc: int = 3, settings: Settings | None = None
) -> list[EvalCase]:
    """Ask the LLM for questions per indexed document (needs a real LLM)."""
    cfg = settings or get_settings()
    if cfg.mock_llm:
        logger.warning("Mock mode: cannot generate a test set, use the bundled one.")
        return []

    from .citations import build_context

    cases: list[EvalCase] = []
    for record in app.documents():
        chunks = app.store.all_chunks([record.doc_id])[:12]
        if not chunks:
            continue
        hits = [Retrieved(chunk=chunk, score=1.0) for chunk in chunks]
        prompt = GENERATE_PROMPT.format(
            filename=record.filename, n=per_doc, context=build_context(hits, max_chars=12000)
        )
        text = app.llm.complete(prompt, max_tokens=1500)
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            logger.warning("Could not parse generated questions for %s", record.filename)
            continue
        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("Generated questions for %s were not valid JSON", record.filename)
            continue
        for i, item in enumerate(items[:per_doc]):
            cases.append(
                EvalCase(
                    id=f"gen-{record.doc_id[:6]}-{i + 1}",
                    question=str(item.get("question", "")).strip(),
                    expected_file=record.filename,
                    answer_keys=[str(k) for k in item.get("answer_keys", [])][:3],
                    reference_answer=str(item.get("reference_answer", "")),
                )
            )
    return [c for c in cases if c.question]


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def run_eval(
    app: DocRag,
    cases: Sequence[EvalCase],
    k: int | None = None,
    use_reranker: bool | None = None,
    judge: bool = True,
    settings: Settings | None = None,
) -> EvalReport:
    cfg = settings or get_settings()
    k = k or cfg.top_k
    report = EvalReport(k=k)
    name_to_id = {r.filename.lower(): r.doc_id for r in app.documents()}

    for case in cases:
        doc_ids = [doc_id for name, doc_id in name_to_id.items()
                   if any(_same_document(name, wanted) for wanted in case.doc_filter)]
        answer = app.ask(
            case.question, top_k=k, doc_ids=doc_ids or None, use_reranker=use_reranker
        )
        text = answer.collect()
        result = CaseResult(
            case=case,
            rank=None if case.unanswerable else first_gold_rank(answer.hits, case),
            retrieved=[hit.chunk.chunk_id for hit in answer.hits],
            answer=text,
        )
        if case.unanswerable:
            result.refused = is_refusal(text, cfg)
        else:
            if case.answer_keys:
                result.answer_ok = answer_contains_keys(text, case)
            if judge and not cfg.mock_llm and not is_refusal(text, cfg):
                judgement = judge_faithfulness(case.question, text, answer.hits, cfg)
                if judgement is not None:
                    report.judged = True
                    result.faithful = judgement.faithful
                    result.judge_reason = judgement.reason
        report.results.append(result)
    return report


# --------------------------------------------------------------------------- #
# printing
# --------------------------------------------------------------------------- #
def _table(rows: list[list[str]], headers: list[str]) -> str:
    try:
        from tabulate import tabulate  # type: ignore

        return tabulate(rows, headers=headers, tablefmt="github")
    except ImportError:
        widths = [
            max(len(str(headers[i])), *(len(str(row[i])) for row in rows)) if rows else len(headers[i])
            for i in range(len(headers))
        ]
        line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        sep = "  ".join("-" * widths[i] for i in range(len(headers)))
        body = [
            "  ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) for row in rows
        ]
        return "\n".join([line, sep, *body])


def _tick(value: bool | None) -> str:
    if value is None:
        return "-"
    return "yes" if value else "NO"


def format_report(report: EvalReport, settings: Settings | None = None) -> str:
    cfg = settings or get_settings()
    rows = []
    for result in report.results:
        case = result.case
        rows.append(
            [
                case.id,
                (case.question[:58] + "...") if len(case.question) > 61 else case.question,
                "unanswerable" if case.unanswerable else (case.expected_file or "-"),
                "-" if case.unanswerable else (str(result.rank) if result.rank else "miss"),
                _tick(result.refused) if case.unanswerable else _tick(result.answer_ok),
                _tick(result.faithful),
            ]
        )
    table = _table(
        rows, ["id", "question", "expected", f"rank@{report.k}", "answer ok", "faithful"]
    )

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"

    summary_rows = [
        [f"hit-rate@{report.k}", pct(report.hit_rate), f"{len(report.answerable)} answerable"],
        ["MRR", f"{report.mrr:.3f}", f"{len(report.answerable)} answerable"],
        [
            "answer key facts",
            pct(report.answer_accuracy),
            "exact string match on expected facts",
        ],
        [
            "correct refusals",
            pct(report.refusal_accuracy),
            f"{len(report.unanswerable)} out-of-scope questions",
        ],
        [
            "faithfulness (judge)",
            pct(report.faithfulness),
            "skipped in mock mode" if cfg.mock_llm else f"judge: {cfg.judge_model}",
        ],
    ]
    summary = _table(summary_rows, ["metric", "value", "notes"])
    return f"{table}\n\n{summary}"
