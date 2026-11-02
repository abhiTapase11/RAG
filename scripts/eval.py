"""Evaluate retrieval and answer quality on the sample Q&A test set.

Usage:
    python scripts/eval.py                     # bundled test set, current settings
    python scripts/eval.py --k 8 --rerank      # more passages + cross-encoder
    python scripts/eval.py --generate 3        # add LLM-generated cases (needs an API key)
    python scripts/eval.py --no-judge          # skip the faithfulness judge

Metrics: hit-rate@k, MRR, exact-fact answer accuracy, correct refusals and
LLM-as-judge faithfulness (automatically skipped in mock mode).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from docrag.config import get_settings
from docrag.generation import LLMError
from docrag.evaluation import (
    format_report,
    generate_testset,
    load_testset,
    run_eval,
    save_testset,
)
from docrag.pipeline import DocRag, ingest_directory


def main(argv: list[str] | None = None) -> int:
    cfg = get_settings()
    default_testset = cfg.samples_dir / "qa_testset.json"

    parser = argparse.ArgumentParser(description="Evaluate doc-rag.")
    parser.add_argument("--testset", default=str(default_testset))
    parser.add_argument("--k", type=int, default=cfg.top_k, help="passages per question")
    parser.add_argument("--rerank", action="store_true", help="enable the cross-encoder reranker")
    parser.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge faithfulness")
    parser.add_argument(
        "--generate",
        type=int,
        default=0,
        metavar="N",
        help="also generate N questions per document with the LLM",
    )
    parser.add_argument(
        "--save-generated",
        default="",
        help="write the generated cases to this JSON file",
    )
    parser.add_argument(
        "--ingest-if-empty",
        action="store_true",
        help="index data/samples first when the store is empty",
    )
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        default=0.0,
        help="exit non-zero if hit-rate@k falls below this (0-1)",
    )
    args = parser.parse_args(argv)

    app = DocRag(cfg)
    if not app.documents():
        if args.ingest_if_empty:
            print(f"Store is empty - indexing {cfg.samples_dir} first...")
            for result in ingest_directory(cfg.samples_dir, store=app.store, settings=cfg):
                state = "skip" if result.skipped else "ok"
                print(f"  [{state}] {result.filename}: {result.n_chunks} chunks")
        else:
            print(
                "No documents indexed. Run 'make ingest-samples' first "
                "(or pass --ingest-if-empty).",
                file=sys.stderr,
            )
            return 2

    path = Path(args.testset)
    if not path.exists():
        print(f"Test set not found: {path}", file=sys.stderr)
        return 2
    cases = load_testset(path)

    if args.generate:
        generated = generate_testset(app, per_doc=args.generate, settings=cfg)
        print(f"Generated {len(generated)} additional case(s) with {app.llm.name}")
        cases = list(cases) + generated
        if generated and args.save_generated:
            save_testset(generated, args.save_generated)
            print(f"Wrote generated cases to {args.save_generated}")

    stats = app.stats()
    print(
        f"\ndoc-rag eval - {stats['documents']} document(s), {stats['chunks']} chunks, "
        f"embeddings {stats['embedder']}, llm {stats['llm']}"
        f"{' (mock mode)' if stats['mock'] else ''}"
    )
    print(f"k={args.k}, reranker={'on' if args.rerank else 'off'}, cases={len(cases)}\n")

    try:
        report = run_eval(
            app,
            cases,
            k=args.k,
            use_reranker=args.rerank or None,
            judge=not args.no_judge,
            settings=cfg,
        )
    except LLMError as exc:
        print(
            f"\nThe language model could not be reached: {exc}\n"
            "Set MOCK_LLM=1 to run offline, or configure ANTHROPIC_API_KEY.",
            file=sys.stderr,
        )
        return 2
    print(format_report(report, cfg))

    misses = [r for r in report.answerable if not r.hit]
    if misses:
        print("\nRetrieval misses:")
        for result in misses:
            print(f"  {result.case.id}: {result.case.question}")

    if args.min_hit_rate and report.hit_rate < args.min_hit_rate:
        print(
            f"\nFAIL: hit-rate@{args.k} {report.hit_rate:.2f} < {args.min_hit_rate:.2f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
