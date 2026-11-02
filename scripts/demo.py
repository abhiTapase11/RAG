"""Headless demo: ingest the samples, ask a grounded question, ask an unknown one.

This is the fastest way to check an install, and it is the script the README's
demo section follows.

Usage:
    python scripts/demo.py --mock      # no API key needed
    python scripts/demo.py             # uses LLM_PROVIDER / ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import os
import sys

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

QUESTIONS = [
    "What is the wingspan of an Aurora Sparrow?",
    "How large is the home office allowance, and when must it be claimed?",
    "How many vacation days do Northwind employees get each year?",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the doc-rag demo.")
    parser.add_argument("--mock", action="store_true", help="force mock mode (no LLM call)")
    parser.add_argument("--rerank", action="store_true", help="enable the cross-encoder")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("question", nargs="*", help="ask your own question(s) instead")
    args = parser.parse_args(argv)

    if args.mock:
        os.environ["MOCK_LLM"] = "1"

    from docrag.config import get_settings
    from docrag.generation import LLMError
    from docrag.pipeline import DocRag, ingest_directory

    cfg = get_settings(refresh=True)
    app = DocRag(cfg)

    if not app.documents():
        print(f"Indexing samples from {cfg.samples_dir} ...")
        try:
            from docrag.pdfwrite import text_to_pdf

            source = cfg.samples_dir / "northwind-remote-work-policy.txt"
            if source.exists():
                text_to_pdf(source.read_text(encoding="utf-8"), source.with_suffix(".pdf"))
        except Exception as exc:  # pragma: no cover - cosmetic step
            print(f"  (could not build the sample PDF: {exc})")
        for result in ingest_directory(cfg.samples_dir, store=app.store, settings=cfg):
            if result.filename.endswith(".txt"):
                continue
            state = "skip" if result.skipped else "ok"
            print(f"  [{state}] {result.filename}: {result.n_chunks} chunks")

    stats = app.stats()
    print(
        f"\n{stats['documents']} document(s), {stats['chunks']} chunks | "
        f"embeddings {stats['embedder']} | llm {stats['llm']}"
        f"{' (mock mode)' if stats['mock'] else ''}\n"
    )

    for question in args.question or QUESTIONS:
        print("=" * 78)
        print(f"Q: {question}")
        answer = app.ask(question, top_k=args.k, use_reranker=args.rerank or None)
        print("A: ", end="", flush=True)
        try:
            for piece in answer.stream():
                print(piece, end="", flush=True)
        except LLMError as exc:
            print(
                f"\n\nThe language model could not be reached: {exc}\n"
                "Run with --mock to stay offline, or configure ANTHROPIC_API_KEY.",
                file=sys.stderr,
            )
            return 2
        print("\n")
        for row in answer.sources():
            mark = "*" if row["cited"] else " "
            print(f"  {mark} {row['n']}. {row['citation']}  score={row['score']:.4f}  ({row['why']})")
        print()
    print("=" * 78)
    print("* = cited in the answer. Run 'make app' for the chat UI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
