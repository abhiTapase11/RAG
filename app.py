"""doc-rag - chat with your documents (Streamlit UI).

Run with:  streamlit run app.py      (or: make app)
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from docrag.citations import highlight_passage  # noqa: E402
from docrag.config import get_settings  # noqa: E402
from docrag.generation import LLMError  # noqa: E402
from docrag.loaders import SUPPORTED_EXTENSIONS  # noqa: E402
from docrag.pipeline import Answer, DocRag  # noqa: E402

st.set_page_config(page_title="doc-rag", page_icon="\U0001f4c4", layout="wide")

UPLOAD_TYPES = sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)


@st.cache_resource(show_spinner="Loading embeddings model...")
def get_app() -> DocRag:
    return DocRag(get_settings())


def init_state() -> None:
    st.session_state.setdefault("messages", [])  # list[dict]
    st.session_state.setdefault("pending", None)  # queued question
    st.session_state.setdefault("pending_summary", None)  # doc_id to summarise
    st.session_state.setdefault("ingest_log", [])


def history_pairs() -> list[tuple[str, str]]:
    return [(m["role"], m["content"]) for m in st.session_state["messages"]]


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(app: DocRag) -> dict:
    cfg = app.cfg
    with st.sidebar:
        st.header("Documents")

        uploads = st.file_uploader(
            "Upload files",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            help="PDF, DOCX, TXT, MD and HTML. Re-uploading the same file is detected by hash.",
        )
        if uploads and st.button("Index uploaded files", type="primary", width="stretch"):
            log: list[str] = []
            progress = st.progress(0.0)
            for i, upload in enumerate(uploads, start=1):
                try:
                    result = app.ingest(upload.getvalue(), upload.name)
                except Exception as exc:  # surfaced in the UI, not swallowed
                    log.append(f"[error] {upload.name}: {exc}")
                else:
                    if result.skipped:
                        log.append(f"[skipped] {result.reason}")
                    else:
                        log.append(f"[indexed] {upload.name}: {result.n_chunks} chunks")
                    log.extend(f"[warning] {w}" for w in result.warnings)
                progress.progress(i / len(uploads))
            progress.empty()
            st.session_state["ingest_log"] = log
            st.rerun()

        for line in st.session_state.get("ingest_log", []):
            if line.startswith("[error]"):
                st.error(line)
            elif line.startswith("[warning]"):
                st.warning(line)
            elif line.startswith("[skipped]"):
                st.info(line)
            else:
                st.success(line)

        records = app.documents()
        if not records:
            st.caption(
                "No documents indexed yet. Upload a file, or run "
                "`make ingest-samples` for the bundled samples."
            )
        for record in records:
            pages = f" - {record.n_pages} pages" if record.n_pages else ""
            with st.container(border=True):
                st.markdown(f"**{record.filename}**")
                st.caption(f"{record.n_chunks} chunks{pages} - indexed {record.ingested_at}")
                if record.warnings:
                    with st.expander(f"{len(record.warnings)} warning(s)"):
                        for warning in record.warnings:
                            st.warning(warning)
                left, right = st.columns(2)
                if left.button("Summarize", key=f"sum-{record.doc_id}", width="stretch"):
                    st.session_state["pending_summary"] = record.doc_id
                    st.rerun()
                if right.button("Delete", key=f"del-{record.doc_id}", width="stretch"):
                    app.delete(record.doc_id)
                    st.session_state["ingest_log"] = [f"[indexed] deleted {record.filename}"]
                    st.rerun()

        st.divider()
        st.header("Settings")
        top_k = st.slider("Passages to retrieve (top-k)", 2, 15, cfg.top_k)
        use_reranker = st.toggle(
            "Cross-encoder reranking",
            value=cfg.use_reranker,
            help="Slower, usually more precise. Downloads a small model on first use.",
        )
        options = {record.filename: record.doc_id for record in records}
        chosen = st.multiselect(
            "Only search these documents",
            options=list(options),
            default=[],
            help="Leave empty to search everything.",
        )
        doc_ids = [options[name] for name in chosen]

        if st.button("Clear chat", width="stretch"):
            st.session_state["messages"] = []
            st.rerun()

        st.divider()
        stats = app.stats()
        st.caption(
            f"collection `{stats['collection']}` - {stats['documents']} docs, "
            f"{stats['chunks']} chunks\nembeddings `{stats['embedder']}` - llm `{stats['llm']}`"
        )
    return {"top_k": top_k, "use_reranker": use_reranker, "doc_ids": doc_ids}


# --------------------------------------------------------------------------- #
# sources panel
# --------------------------------------------------------------------------- #
def render_sources(rows: list[dict], query: str, answer_text: str) -> None:
    if not rows:
        st.caption("No passages were retrieved.")
        return
    cited = sum(1 for row in rows if row.get("cited"))
    label = f"Sources - {len(rows)} passages retrieved, {cited} cited"
    with st.expander(label):
        for row in rows:
            mark = " **(cited)**" if row.get("cited") else ""
            score_bits = [f"fused score {row['score']:.4f}"]
            if row["rerank_score"] is not None:
                score_bits.append(f"reranker {row['rerank_score']:+.2f}")
            score_bits.append(row["why"])
            st.markdown(f"**{row['n']}. {row['citation']}**{mark}")
            st.caption(" - ".join(score_bits))
            st.markdown(
                f"> {highlight_passage(row['text'], query, answer_text).replace(chr(10), chr(10) + '> ')}"
            )
            st.divider()


def render_history() -> None:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if message.get("rewritten") and message["rewritten"] != message.get("query"):
                    st.caption(f"Searched for: {message['rewritten']}")
                render_sources(
                    message.get("sources", []), message.get("query", ""), message["content"]
                )


# --------------------------------------------------------------------------- #
# answering
# --------------------------------------------------------------------------- #
def run_answer(answer: Answer, query: str) -> None:
    with st.chat_message("assistant"):
        if answer.rewritten_query and answer.rewritten_query != query:
            st.caption(f"Searched for: {answer.rewritten_query}")
        try:
            text = st.write_stream(answer.stream())
        except LLMError as exc:
            text = f"The language model could not be reached: {exc}"
            st.error(text)
        if isinstance(text, list):  # write_stream returns a list for non-str chunks
            text = "".join(str(part) for part in text)
        sources = answer.sources()
        render_sources(sources, query, text or "")
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": text or "",
            "sources": sources,
            "query": query,
            "rewritten": answer.rewritten_query,
        }
    )


def main() -> None:
    init_state()
    app = get_app()
    options = render_sidebar(app)

    st.title("doc-rag")
    st.caption(
        "Ask questions about your documents. Answers are generated only from the "
        "retrieved passages, with inline citations like `[file.pdf, p.4]`."
    )
    if app.stats()["mock"]:
        st.info(
            "**Mock mode** - no LLM is called. Answers are the top retrieved passages with "
            "their citations. Unset `MOCK_LLM` and set `ANTHROPIC_API_KEY` for generated answers.",
            icon="⚠️",
        )

    records = app.documents()
    if records:
        suggestions: list[tuple[str, str]] = []
        for record in records:
            for question in record.suggested_questions[:2]:
                suggestions.append((record.filename, question))
        if suggestions:
            st.markdown("**Try one of these**")
            columns = st.columns(min(len(suggestions), 4))
            for i, (filename, question) in enumerate(suggestions[:4]):
                if columns[i].button(
                    question, key=f"suggest-{i}", help=filename, width="stretch"
                ):
                    st.session_state["pending"] = question
                    st.rerun()

    render_history()

    # Queued work from the sidebar / suggestion buttons.
    pending_summary = st.session_state.pop("pending_summary", None)
    if pending_summary:
        record = app.document(pending_summary)
        prompt = f"Summarize {record.filename if record else pending_summary}"
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        run_answer(app.summarize(pending_summary), prompt)
        return

    question = st.chat_input("Ask a question about your documents")
    queued = st.session_state.pop("pending", None)
    question = question or queued
    if not question:
        return

    history = history_pairs()
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not records:
        with st.chat_message("assistant"):
            text = "No documents are indexed yet - upload a file in the sidebar first."
            st.warning(text)
        st.session_state["messages"].append({"role": "assistant", "content": text})
        return

    with st.spinner("Retrieving passages..."):
        answer = app.ask(
            question,
            history=history,
            top_k=options["top_k"],
            doc_ids=options["doc_ids"],
            use_reranker=options["use_reranker"],
        )
    run_answer(answer, question)


main()
