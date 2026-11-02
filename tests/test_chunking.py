"""Chunking tests, focused on where the boundaries land."""

from __future__ import annotations

from docrag.chunking import chunk_blocks, chunk_document
from docrag.config import get_settings
from docrag.loaders import load_document
from docrag.schema import Block
from docrag.utils import count_tokens


def _cfg(**overrides):
    cfg = get_settings(refresh=True)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_heading_change_forces_a_boundary():
    blocks = [
        Block(text="Alpha section body.", heading="Alpha"),
        Block(text="More about alpha.", heading="Alpha"),
        Block(text="Beta section body.", heading="Beta"),
    ]

    chunks = chunk_blocks(blocks, "doc1", "x.md", _cfg(chunk_min_tokens=1))

    assert [c.heading for c in chunks] == ["Alpha", "Beta"]
    assert "Beta section body." not in chunks[0].text
    assert "alpha" not in chunks[1].text.lower()


def test_page_change_forces_a_boundary():
    blocks = [
        Block(text="Sentence on page one.", page=1, heading="Shared"),
        Block(text="Sentence on page two.", page=2, heading="Shared"),
    ]

    chunks = chunk_blocks(blocks, "doc1", "x.pdf", _cfg(chunk_min_tokens=1))

    assert [c.page for c in chunks] == [1, 2]
    assert "page two" not in chunks[0].text


def test_chunks_respect_the_token_budget():
    paragraph = "The quick brown fox jumps over the lazy dog. " * 12  # ~110 tokens
    blocks = [Block(text=paragraph, heading="Body") for _ in range(12)]
    cfg = _cfg(chunk_target_tokens=200, chunk_max_tokens=260, chunk_overlap_ratio=0.0)

    chunks = chunk_blocks(blocks, "doc1", "x.md", cfg)

    assert len(chunks) > 1
    assert all(count_tokens(c.text) <= cfg.chunk_max_tokens for c in chunks)
    # Chunks are filled, not one-paragraph-each.
    assert max(count_tokens(c.text) for c in chunks) > cfg.chunk_target_tokens / 2


def test_oversized_paragraph_is_split_on_sentences():
    sentences = [f"Fact number {i} is worth knowing in detail." for i in range(60)]
    blocks = [Block(text=" ".join(sentences), heading="Facts")]
    cfg = _cfg(chunk_target_tokens=80, chunk_max_tokens=100, chunk_overlap_ratio=0.0)

    chunks = chunk_blocks(blocks, "doc1", "x.md", cfg)

    assert len(chunks) > 3
    assert all(count_tokens(c.text) <= cfg.chunk_max_tokens for c in chunks)
    # No sentence was cut in half.
    for chunk in chunks:
        body = chunk.text.replace("Facts\n\n", "").strip()
        assert body.endswith(".")


def test_overlap_carries_the_previous_tail_within_a_section():
    sentences = [f"Fact number {i} is worth knowing in detail." for i in range(40)]
    blocks = [Block(text=s, heading="Facts") for s in sentences]
    cfg = _cfg(chunk_target_tokens=60, chunk_max_tokens=90, chunk_overlap_ratio=0.15)

    chunks = chunk_blocks(blocks, "doc1", "x.md", cfg)

    assert len(chunks) > 2
    tail_of_first = chunks[0].text.strip().split("\n\n")[-1]
    assert tail_of_first in chunks[1].text, "expected the previous tail to be carried over"


def test_overlap_does_not_leak_across_sections():
    blocks = [
        Block(text="Alpha has a secret token zebra.", heading="Alpha"),
        Block(text="Beta is about something else entirely.", heading="Beta"),
    ]

    chunks = chunk_blocks(blocks, "doc1", "x.md", _cfg(chunk_min_tokens=1))

    assert "zebra" not in chunks[1].text


def test_ids_metadata_and_heading_prefix(sample_markdown):
    doc = load_document(sample_markdown)

    chunks = chunk_document(doc, "deadbeef")

    assert chunks
    assert [c.chunk_id for c in chunks] == [f"deadbeef:{i:04d}" for i in range(len(chunks))]
    assert all(c.filename == "aurora-sparrow-field-guide.md" for c in chunks)
    assert all(c.doc_id == "deadbeef" for c in chunks)
    # The section heading is part of the chunk text, so it is searchable.
    breeding = [c for c in chunks if c.heading == "Breeding"]
    assert breeding and breeding[0].text.startswith("Breeding")
    # Pageless document -> page is None in the metadata.
    assert breeding[0].metadata()["page"] == -1


def test_tiny_trailing_chunk_is_merged():
    blocks = [
        Block(text="Main body sentence number one. " * 20, heading="Only"),
        Block(text="Tiny.", heading="Only"),
    ]
    cfg = _cfg(chunk_target_tokens=120, chunk_max_tokens=160, chunk_min_tokens=40)

    chunks = chunk_blocks(blocks, "doc1", "x.md", cfg)

    assert "Tiny." in chunks[-1].text
    assert count_tokens(chunks[-1].text) > 40
