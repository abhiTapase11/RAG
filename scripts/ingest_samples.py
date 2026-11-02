"""Index the bundled sample documents.

By default this renders ``northwind-remote-work-policy.txt`` to a real
multi-page PDF first and indexes the PDF instead of the .txt, so the demo shows
page-level citations like ``[northwind-remote-work-policy.pdf, p.2]``. Pass
``--all`` to index every supported file in the directory instead.

Usage:
    python scripts/ingest_samples.py [--dir DIR] [--force] [--all] [--no-pdf]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from docrag.config import get_settings
from docrag.loaders import SUPPORTED_EXTENSIONS
from docrag.pdfwrite import text_to_pdf
from docrag.pipeline import ingest_path
from docrag.store import DocStore

PDF_SOURCE = "northwind-remote-work-policy.txt"


def main(argv: list[str] | None = None) -> int:
    cfg = get_settings()
    parser = argparse.ArgumentParser(description="Index the sample documents.")
    parser.add_argument("--dir", default=str(cfg.samples_dir), help="directory to ingest")
    parser.add_argument("--force", action="store_true", help="re-index even if already present")
    parser.add_argument("--all", action="store_true", help="ingest every supported file")
    parser.add_argument("--no-pdf", action="store_true", help="do not build the sample PDF")
    args = parser.parse_args(argv)

    directory = Path(args.dir)
    if not directory.is_dir():
        print(f"No such directory: {directory}", file=sys.stderr)
        return 2

    pdf_path = (directory / PDF_SOURCE).with_suffix(".pdf")
    if not args.no_pdf and (directory / PDF_SOURCE).exists():
        text_to_pdf((directory / PDF_SOURCE).read_text(encoding="utf-8"), pdf_path)
        print(f"rendered {pdf_path.name} from {PDF_SOURCE}")

    paths = [
        p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not args.all and pdf_path.exists():
        # Skip the .txt whose content is already in the PDF.
        paths = [p for p in paths if p.name != PDF_SOURCE]

    if not paths:
        print(f"Nothing to ingest in {directory}", file=sys.stderr)
        return 1

    store = DocStore(cfg)
    print(f"Indexing {len(paths)} file(s) into collection {store.collection_name!r}")
    print(f"  embeddings: {store.embedder.name} ({store.embedder.dim} dims)")

    failures = 0
    for path in paths:
        try:
            result = ingest_path(path, store=store, settings=cfg, force=args.force)
        except Exception as exc:
            failures += 1
            print(f"  [fail] {path.name}: {exc}", file=sys.stderr)
            continue
        if result.skipped:
            print(f"  [skip] {result.reason}")
            continue
        record = result.record
        pages = f", {record.n_pages} pages" if record and record.n_pages else ""
        print(f"  [ok]   {path.name}: {result.n_chunks} chunks{pages}")
        for warning in result.warnings:
            print(f"         warning: {warning}")

    records = store.list_documents()
    print(
        f"\nCollection now holds {len(records)} document(s), "
        f"{sum(r.n_chunks for r in records)} chunks."
    )
    for record in records:
        print(f"  - {record.filename}  ({record.n_chunks} chunks, doc_id {record.doc_id})")
    print("\nNext:  make app      (or: streamlit run app.py)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
