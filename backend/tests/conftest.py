import hashlib

import pytest

from pathakam.llm import Message
from pathakam.store import Document, Store


class FakeEmbedder:
    """Deterministic bag-of-trigram hashing vectors: similar strings get similar vectors."""

    name = "fake:embed"
    dim = 64

    async def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for i in range(len(t) - 2):
                h = int(hashlib.md5(t[i : i + 3].encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            out.append(v)
        return out


class FakeChat:
    name = "fake:chat"

    def __init__(self, reply="పథకం గురించి సమాధానం [1]"):
        self.reply = reply
        self.calls: list[list[Message]] = []

    async def stream(self, messages, temperature=0.1):
        self.calls.append(list(messages))
        for word in self.reply.split(" "):
            yield word + " "


DOCS = {
    "rythu": ("Farmer support scheme", [
        (1, "రైతు భరోసా పథకం కింద అర్హులైన ప్రతి రైతు కుటుంబానికి సంవత్సరానికి ఆర్థిక సహాయం అందుతుంది."),
        (2, "దరఖాస్తుకు ఆధార్ కార్డు, భూమి పట్టాదారు పాసు పుస్తకం, బ్యాంకు ఖాతా వివరాలు అవసరం."),
    ]),
    "health": ("Health insurance scheme", [
        (1, "The health insurance scheme covers cashless treatment in network hospitals for eligible families."),
        (3, "Families must hold a white ration card to be eligible for the health insurance scheme."),
    ]),
}


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture
def filled_store(store):
    import asyncio

    emb = FakeEmbedder()
    for doc_id, (title, pages) in DOCS.items():
        chunks = [(p, i, t) for i, (p, t) in enumerate(pages)]
        vecs = asyncio.run(emb.embed([t for _, _, t in chunks]))
        store.replace_document(Document(doc_id, title, source_url=f"https://example.gov.in/{doc_id}.pdf"),
                               chunks, vecs, emb.name)
    return store
