"""End-to-end tests: ingest -> retrieve -> answer, offline.

Embeddings use the ``hash`` backend and the LLM is either doc-rag's mock client
or the scripted ``FakeLLM`` below, so these tests need no network and no keys.
"""

from __future__ import annotations

import pytest
from conftest import SAMPLES

from docrag.citations import format_citation
from docrag.evaluation import load_testset, run_eval
from docrag.generation import coverage, is_answerable
from docrag.pipeline import DocRag


class FakeLLM:
    """A scripted LLM that records what it was asked."""

    name = "fake"
    supports_rewrite = False

    def __init__(self, reply: str = "Scripted answer."):
        self.reply = reply
        self.calls: list[dict] = []

    def stream(self, system, messages):
        self.calls.append({"system": system, "messages": list(messages)})
        for word in self.reply.split(" "):
            yield word + " "

    def complete(self, prompt, max_tokens=1024, system=None):
        self.calls.append({"system": system, "messages": [{"role": "user", "content": prompt}]})
        return self.reply


@pytest.fixture
def indexed(app, sample_markdown, sample_pdf):
    """A store holding the markdown field guide and the paged policy PDF."""
    md = app.ingest_file(sample_markdown)
    pdf = app.ingest_file(sample_pdf)
    assert not md.skipped and not pdf.skipped
    return app, md.record, pdf.record


def test_ingest_indexes_both_samples(indexed):
    app, md_record, pdf_record = indexed

    assert md_record.n_chunks > 3
    assert pdf_record.n_chunks > 3
    assert pdf_record.n_pages and pdf_record.n_pages >= 2
    assert md_record.suggested_questions, "ingestion should propose starter questions"
    assert app.stats()["chunks"] == md_record.n_chunks + pdf_record.n_chunks
    assert {r.filename for r in app.documents()} == {
        "aurora-sparrow-field-guide.md",
        "northwind-remote-work-policy.pdf",
    }


def test_factual_question_is_answered_with_a_citation(indexed):
    app, md_record, _ = indexed

    answer = app.ask("What is the wingspan of an Aurora Sparrow?", top_k=5)
    text = answer.collect()

    assert "24" in text, text
    assert "aurora-sparrow-field-guide.md" in text  # mock mode cites its passages
    top_files = [hit.chunk.filename for hit in answer.hits]
    assert top_files[0] == "aurora-sparrow-field-guide.md"
    sources = answer.sources()
    assert any(row["cited"] for row in sources)
    assert all(row["citation"].startswith("[") for row in sources)


def test_paged_document_citations_carry_page_numbers(indexed):
    app, _, _ = indexed

    answer = app.ask("What is the home office allowance?", top_k=5)
    answer.collect()

    pdf_hits = [h for h in answer.hits if h.chunk.filename.endswith(".pdf")]
    assert pdf_hits, "the PDF should be retrieved for a policy question"
    citation = format_citation(pdf_hits[0].chunk)
    assert citation.startswith("[northwind-remote-work-policy.pdf, p.")


def test_out_of_scope_question_is_refused(indexed):
    app, _, _ = indexed

    answer = app.ask("How many vacation days do Northwind employees get each year?", top_k=5)
    text = answer.collect()

    assert text.strip() == app.cfg.refusal_text


def test_grounding_gate_separates_in_and_out_of_scope(indexed):
    app, _, _ = indexed
    idf = app.retriever.idf_lookup()

    in_scope = app.ask("What is the mean clutch size?", top_k=5)
    out_of_scope = app.ask("What is the dividend yield of the company?", top_k=5)

    assert coverage(in_scope.query, in_scope.hits, idf=idf) > coverage(
        out_of_scope.query, out_of_scope.hits, idf=idf
    )
    assert is_answerable(in_scope.query, in_scope.hits, strict=True, idf=idf)
    assert not is_answerable(out_of_scope.query, out_of_scope.hits, strict=True, idf=idf)


def test_reupload_of_the_same_bytes_is_deduplicated(indexed, sample_markdown):
    app, md_record, _ = indexed
    before = app.stats()["chunks"]

    again = app.ingest_file(sample_markdown)

    assert again.skipped
    assert "already indexed" in again.reason
    assert app.stats()["chunks"] == before
    assert len(app.documents()) == 2


def test_force_reingest_replaces_the_document(indexed, sample_markdown):
    app, md_record, _ = indexed

    again = app.ingest_file(sample_markdown, force=True)

    assert not again.skipped
    assert again.record is not None
    assert len(app.documents()) == 2
    assert app.store.count([md_record.doc_id]) == again.record.n_chunks


def test_delete_removes_chunks_and_registry_entry(indexed):
    app, md_record, pdf_record = indexed

    assert app.delete(md_record.doc_id) is True

    assert [r.doc_id for r in app.documents()] == [pdf_record.doc_id]
    assert app.store.count([md_record.doc_id]) == 0
    answer = app.ask("What is the wingspan of an Aurora Sparrow?", top_k=5)
    assert all(hit.chunk.filename != "aurora-sparrow-field-guide.md" for hit in answer.hits)


def test_document_filter_restricts_retrieval(indexed):
    app, md_record, _ = indexed

    answer = app.ask("What are the core hours?", top_k=5, doc_ids=[md_record.doc_id])

    assert answer.hits
    assert {hit.chunk.filename for hit in answer.hits} == {"aurora-sparrow-field-guide.md"}


def test_mocked_llm_receives_grounded_prompt_and_history(settings, store, sample_markdown):
    llm = FakeLLM("The wingspan is 24 centimetres [aurora-sparrow-field-guide.md, p.1].")
    app = DocRag(settings, store=store, llm=llm)
    app.ingest_file(sample_markdown)

    answer = app.ask(
        "What is its wingspan?",
        history=[("user", "Tell me about the Aurora Sparrow."), ("assistant", "It is a songbird.")],
        top_k=4,
    )
    text = answer.collect()

    assert text.strip() == llm.reply
    assert len(llm.calls) == 1
    call = llm.calls[0]
    # The system prompt carries the grounding rules and the exact refusal string.
    assert settings.refusal_text in call["system"]
    # History is replayed, and the last message holds the retrieved context.
    assert [m["role"] for m in call["messages"]] == ["user", "assistant", "user"]
    final = call["messages"][-1]["content"]
    assert "<context>" in final and "<passage" in final
    assert "wingspan of 24 centimetres" in final
    # The follow-up was rewritten using the previous turn before retrieval.
    assert answer.rewritten_query != answer.query
    assert "sparrow" in answer.rewritten_query.lower()


def test_summarize_uses_a_spread_of_chunks(settings, store, sample_markdown):
    llm = FakeLLM("Summary of the field guide [aurora-sparrow-field-guide.md].")
    app = DocRag(settings, store=store, llm=llm)
    record = app.ingest_file(sample_markdown).record

    answer = app.summarize(record.doc_id, max_chunks=4)
    text = answer.collect()

    assert text.strip() == llm.reply
    assert len(answer.hits) == 4
    prompt = llm.calls[0]["messages"][-1]["content"]
    assert "Summarise aurora-sparrow-field-guide.md" in prompt
    assert prompt.count("<passage") == 4


def test_eval_on_the_bundled_testset(indexed):
    app, _, _ = indexed
    cases = load_testset(SAMPLES / "qa_testset.json")

    report = run_eval(app, cases, k=5, judge=False)

    assert len(report.results) == len(cases)
    assert report.hit_rate >= 0.8, f"hit-rate@5 was {report.hit_rate:.2f}"
    assert report.mrr >= 0.6, f"MRR was {report.mrr:.2f}"
    assert report.refusal_accuracy == 1.0
    assert report.faithfulness is None  # judge skipped in mock mode
