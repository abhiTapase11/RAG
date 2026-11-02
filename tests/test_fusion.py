"""Reciprocal rank fusion and BM25 tests."""

from __future__ import annotations

import pytest

from docrag.retrieval import BM25Index, fuse_ranked, reciprocal_rank_fusion
from docrag.schema import Chunk


def test_rrf_scores_match_the_formula():
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)

    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_rewards_agreement_between_retrievers():
    vector = ["x", "shared", "y"]
    keyword = ["z", "shared", "w"]

    ranked = fuse_ranked([vector, keyword], k=60)

    assert ranked[0][0] == "shared"
    # Top-of-one-list items still beat mid-list singletons.
    assert [item for item, _ in ranked[1:3]] == ["x", "z"]


def test_rrf_handles_missing_and_empty_lists():
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[], []]) == {}

    scores = reciprocal_rank_fusion([["only"], []], k=10)
    assert scores == {"only": pytest.approx(1 / 11)}


def test_rrf_k_controls_how_much_top_ranks_dominate():
    small_k = reciprocal_rank_fusion([["first", "second"]], k=1)
    large_k = reciprocal_rank_fusion([["first", "second"]], k=1000)

    assert small_k["first"] / small_k["second"] > large_k["first"] / large_k["second"]


def test_rrf_weights_shift_the_ranking():
    vector = ["v", "k"]
    keyword = ["k", "v"]

    vector_heavy = fuse_ranked([vector, keyword], weights=[2.0, 1.0])
    keyword_heavy = fuse_ranked([vector, keyword], weights=[1.0, 2.0])

    assert vector_heavy[0][0] == "v"
    assert keyword_heavy[0][0] == "k"


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_rrf_is_deterministic_for_ties():
    ranked = fuse_ranked([["b", "a"], ["a", "b"]])

    assert [item for item, _ in ranked] == ["a", "b"]  # equal scores -> id order


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, text=text, doc_id="d", filename="f.md")


def test_bm25_finds_the_rare_term():
    index = BM25Index(
        [
            _chunk("c1", "The crowberry layer supports nesting sparrows."),
            _chunk("c2", "Remote employees receive a home office allowance."),
            _chunk("c3", "Core hours are ten to four in the local time zone."),
        ]
    )

    hits = index.search("home office allowance", k=2)

    assert hits
    assert hits[0][0].chunk_id == "c2"
    assert hits[0][1] > 0


def test_bm25_returns_nothing_for_unknown_terms():
    index = BM25Index([_chunk("c1", "Crowberry and sedge seed.")])

    assert index.search("quarterly dividend policy", k=3) == []
    assert index.search("", k=3) == []


def test_bm25_idf_marks_unseen_terms_as_rarest():
    index = BM25Index(
        [_chunk("c1", "sparrow nest crowberry"), _chunk("c2", "sparrow policy office")]
    )

    assert index.idf_of("vacation") == pytest.approx(index.max_idf)
    assert index.idf_of("crowberry") < index.max_idf
    assert index.idf_of("crowberry") > index.idf_of("sparrow")  # rarer term scores higher
