"""Citation formatting, parsing and highlighting."""

from __future__ import annotations

from docrag.citations import (
    SECTION_MARK,
    build_context,
    cited_chunk_ids,
    format_citation,
    highlight_passage,
    parse_citations,
    sources_table,
)
from docrag.schema import Chunk, Retrieved


def _hit(chunk_id="c1", filename="report.pdf", page=4, heading=None, text="Body text.", score=0.5):
    chunk = Chunk(
        chunk_id=chunk_id,
        text=text,
        doc_id="d1",
        filename=filename,
        page=page,
        heading=heading,
    )
    return Retrieved(chunk=chunk, score=score)


def test_format_citation_for_paged_document():
    assert format_citation(_hit().chunk) == "[report.pdf, p.4]"


def test_format_citation_falls_back_to_section_then_filename():
    section = _hit(filename="handbook.md", page=None, heading="Leave policy").chunk
    bare = _hit(filename="notes.txt", page=None, heading=None).chunk

    assert format_citation(section) == f"[handbook.md, {SECTION_MARK} Leave policy]"
    assert format_citation(bare) == "[notes.txt]"


def test_parse_citations_reads_pages_sections_and_bare_files():
    answer = (
        "Revenue rose [report.pdf, p.4] and leave is 25 days "
        f"[handbook.md, {SECTION_MARK} Leave policy]. See also [notes.txt]."
    )

    citations = parse_citations(answer)

    assert [(c.filename, c.page, c.section) for c in citations] == [
        ("report.pdf", 4, None),
        ("handbook.md", None, "Leave policy"),
        ("notes.txt", None, None),
    ]


def test_parse_citations_ignores_markdown_links_and_duplicates():
    answer = "See [the docs](https://example.com) and [report.pdf, p.4] twice [report.pdf, p. 4]."

    citations = parse_citations(answer)

    assert len(citations) == 1
    assert citations[0].filename == "report.pdf"


def test_cited_chunk_ids_matches_only_the_cited_passage():
    hits = [
        _hit("c1", page=4),
        _hit("c2", page=9),
        _hit("c3", filename="handbook.md", page=None, heading="Leave policy"),
    ]
    answer = f"Revenue rose [report.pdf, p.4]; leave is 25 days [handbook.md, {SECTION_MARK} Leave]."

    assert cited_chunk_ids(answer, hits) == {"c1", "c3"}


def test_cited_chunk_ids_is_empty_without_citations():
    assert cited_chunk_ids("No citations here.", [_hit()]) == set()


def test_build_context_labels_passages_with_their_citation():
    hits = [_hit("c1", text="Revenue rose 12 percent."), _hit("c2", page=9, text="Costs fell.")]

    context = build_context(hits)

    assert 'id="1" cite="[report.pdf, p.4]"' in context
    assert 'id="2" cite="[report.pdf, p.9]"' in context
    assert "Revenue rose 12 percent." in context


def test_build_context_respects_a_character_budget():
    hits = [_hit(f"c{i}", text="x" * 400) for i in range(5)]

    context = build_context(hits, max_chars=1000)

    assert context.count("<passage") == 2


def test_highlight_passage_bolds_query_and_answer_overlap():
    passage = "The wingspan is 24 centimetres and the mass is 31 grams."

    highlighted = highlight_passage(passage, "What is the wingspan?", "It is 24 centimetres.")

    assert "**wingspan**" in highlighted
    assert "**centimetres**" in highlighted
    assert "mass" in highlighted and "**mass**" not in highlighted


def test_highlight_passage_leaves_text_alone_without_terms():
    passage = "Nothing to match here."

    assert highlight_passage(passage, "") == passage


def test_sources_table_shape():
    rows = sources_table([_hit("c1", score=0.0321)])

    assert rows[0]["n"] == 1
    assert rows[0]["citation"] == "[report.pdf, p.4]"
    assert rows[0]["score"] == 0.0321
    assert rows[0]["chunk_id"] == "c1"
