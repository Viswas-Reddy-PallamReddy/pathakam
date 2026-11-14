import asyncio

from pathakam.rag import Answerer
from pathakam.retrieve import Retriever
from pathakam.evaluate import run

from .conftest import FakeChat, FakeEmbedder

QUESTIONS = [
    {"id": "q1", "question": "రైతులకు ఏ పత్రాలు కావాలి?", "doc_id": "rythu", "pages": [2]},
    {"id": "q2", "question": "Who is eligible for the health insurance scheme?", "doc_id": "health"},
    {"id": "q3", "question": "What is the bus pass fare?", "answerable": False},
]


def test_retrieval_metrics(filled_store):
    report = asyncio.run(run(QUESTIONS, Retriever(filled_store, FakeEmbedder()), None, k=3))
    s = report["summary"]
    assert s["questions"] == 3 and s["hit_at_k"] == 1.0 and 0 < s["mrr"] <= 1.0


def test_answer_checks(filled_store):
    retriever = Retriever(filled_store, FakeEmbedder())
    answerer = Answerer(retriever, FakeChat("Answer from the sources [1] [9]"), top_k=3)
    s = asyncio.run(run(QUESTIONS, retriever, answerer, k=3))["summary"]
    assert s["cited_rate"] == 1.0
    assert s["citations_valid_rate"] < 1.0  # [9] does not exist
    assert s["errors"] == 0 and s["latency_p50_ms"] is not None
