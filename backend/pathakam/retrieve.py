"""Hybrid retrieval: BM25 (trigram) + dense vectors, fused with Reciprocal Rank Fusion."""

from dataclasses import dataclass

from .llm import Embedder
from .store import Hit, Store


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank). Ranks start at 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


@dataclass
class RetrievalResult:
    hits: list[Hit]
    keyword_count: int
    vector_count: int


class Retriever:
    def __init__(self, store: Store, embedder: Embedder | None, candidates: int = 24):
        self.store = store
        self.embedder = embedder
        self.candidates = candidates

    async def search(self, query: str, k: int) -> RetrievalResult:
        kw = self.store.keyword_search(query, self.candidates)
        vec: list[tuple[int, float]] = []
        if self.embedder is not None and self.store.has_vectors():
            [qv] = await self.embedder.embed([query])
            vec = self.store.vector_search(qv, self.candidates)
        fused = rrf([[cid for cid, _ in kw], [cid for cid, _ in vec]])[:k]
        return RetrievalResult(self.store.get_hits(fused), len(kw), len(vec))
