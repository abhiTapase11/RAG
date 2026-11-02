"""Smoke test for the Streamlit UI using Streamlit's own AppTest harness.

It runs app.py headlessly: the page must render without an exception, and a
question typed into the chat input must produce a cited answer.
"""

from __future__ import annotations

import pytest
from conftest import ROOT

st = pytest.importorskip("streamlit")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(ROOT / "app.py")


@pytest.fixture
def harness(settings, store, sample_markdown):
    """AppTest bound to the isolated store, with one document indexed."""
    from docrag.pipeline import ingest_path

    ingest_path(sample_markdown, store=store, settings=settings)
    st.cache_resource.clear()  # app.py caches its DocRag instance
    at = AppTest.from_file(APP, default_timeout=120)
    yield at
    st.cache_resource.clear()


def test_app_renders_with_a_document(harness):
    at = harness.run()

    assert not at.exception, at.exception
    assert "doc-rag" in " ".join(h.value for h in at.title)
    # Mock mode is announced, and the document appears in the sidebar.
    assert any("Mock mode" in info.value for info in at.info)
    sidebar_text = " ".join(md.value for md in at.sidebar.markdown)
    assert "aurora-sparrow-field-guide.md" in sidebar_text


def test_app_answers_a_question_with_sources(harness):
    at = harness.run()

    at.chat_input[0].set_value("What is the wingspan of an Aurora Sparrow?").run()

    assert not at.exception, at.exception
    replies = [m for m in at.chat_message if m.name == "assistant"]
    assert replies, "expected an assistant reply"
    answer = " ".join(md.value for md in replies[-1].markdown)
    assert "24" in answer
    assert "aurora-sparrow-field-guide.md" in answer


def test_app_refuses_out_of_scope_question(harness):
    at = harness.run()

    at.chat_input[0].set_value("What is the companys dividend policy?").run()

    assert not at.exception, at.exception
    replies = [m for m in at.chat_message if m.name == "assistant"]
    answer = " ".join(md.value for md in replies[-1].markdown)
    assert "couldn't find that in the documents" in answer
