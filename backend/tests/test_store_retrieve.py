import asyncio

import pytest

from pathakam.retrieve import Retriever, rrf
from pathakam.store import Document

from .conftest import FakeEmbedder


def test_trigram_matches_telugu_inflections(filled_store):
    # Query uses the plural dative "రైతులకు" (to the farmers); the text has "రైతు"/"రైతుకు" forms.
    hits = filled_store.get_hits(filled_store.keyword_search("రైతులకు ఏ పత్రాలు కావాలి", 5))
    assert hits and hits[0].doc_id == "rythu"


def test_keyword_search_english_and_short_terms(filled_store):
    assert filled_store.keyword_search("ration card", 5)
    assert filled_store.keyword_search("a ?", 5) == []  # nothing >= 3 chars


def test_keyword_query_with_quotes_is_safe(filled_store):
    filled_store.keyword_search('white "ration" card OR NOT', 5)


def test_vector_search_and_reindex(filled_store):
    [q] = asyncio.run(FakeEmbedder().embed(["white ration card eligible"]))
    top = filled_store.get_hits(filled_store.vector_search(q, 1))
    assert top[0].doc_id == "health" and top[0].page == 3

    # Re-ingesting a document replaces its chunks instead of duplicating them.
    before = filled_store.stats()["chunks"]
    filled_store.replace_document(Document("health", "Health v2"), [(1, 0, "new text only")], [q], "fake:embed")
    assert filled_store.stats()["chunks"] == before - 1
    assert not filled_store.keyword_search("cashless", 5)


def test_embedding_model_switch_is_rejected(filled_store):
    with pytest.raises(ValueError, match="Rebuild the index"):
        filled_store.replace_document(Document("x", "x"), [(1, 0, "t")], [[0.1] * 8], "other-model")


def test_rrf_rewards_agreement():
    fused = dict(rrf([[1, 2, 3], [3, 1, 4]]))
    assert fused[1] > fused[3] > fused[2] and fused[3] > fused[4]


def test_hybrid_retriever(filled_store):
    res = asyncio.run(Retriever(filled_store, FakeEmbedder()).search("ఆధార్ కార్డు బ్యాంకు ఖాతా", 2))
    assert res.hits[0].doc_id == "rythu" and res.hits[0].page == 2
    assert res.keyword_count > 0 and res.vector_count > 0
