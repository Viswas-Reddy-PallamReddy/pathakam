import json

from fastapi.testclient import TestClient

from pathakam.api import create_app
from pathakam.config import Settings
from pathakam.llm import LLMError

from .conftest import FakeChat, FakeEmbedder


def parse_sse(body: str) -> list[dict]:
    return [json.loads(line[5:]) for line in body.splitlines() if line.startswith("data:")]


def client(store, chat):
    return TestClient(create_app(Settings(), chat=chat, embedder=FakeEmbedder(), store=store))


def test_health_and_documents(filled_store):
    c = client(filled_store, FakeChat())
    h = c.get("/api/health").json()
    assert h["status"] == "ok" and h["documents"] == 2 and h["llm"] == "fake:chat"
    assert {d["id"] for d in c.get("/api/documents").json()} == {"rythu", "health"}


def test_chat_streams_sources_then_tokens(filled_store):
    chat = FakeChat()
    r = client(filled_store, chat).post("/api/chat", json={"question": "రైతు భరోసాకు ఏ పత్రాలు కావాలి?"})
    events = parse_sse(r.text)
    types = [e["type"] for e in events]
    assert types[0] == "meta" and types[1] == "sources" and types[-1] == "done"
    assert events[0]["lang"] == "te"
    assert events[1]["sources"][0]["doc_id"] == "rythu"
    assert "".join(e["text"] for e in events if e["type"] == "token").strip() == chat.reply
    # The prompt sent to the model contains numbered sources and the Telugu reply rule.
    system, user = chat.calls[0][0].content, chat.calls[0][-1].content
    assert "Telugu script" in system and "[1]" in user


def test_no_hits_skips_the_model(store):
    chat = FakeChat()
    r = client(store, chat).post("/api/chat", json={"question": "What is the weather?", "lang": "en"})
    events = parse_sse(r.text)
    assert events[-1]["type"] == "done" and events[-1]["grounded"] is False
    assert chat.calls == []


def test_history_is_passed_and_trimmed(filled_store):
    chat = FakeChat()
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(30)]
    client(filled_store, chat).post("/api/chat", json={"question": "ration card", "history": history})
    sent = chat.calls[0]
    assert 1 < len(sent) - 2 <= 2 * Settings().max_history_turns  # minus system + current question


def test_provider_error_becomes_error_event(filled_store):
    class Broken(FakeChat):
        async def stream(self, messages, temperature=0.1):
            raise LLMError("model not found")
            yield  # pragma: no cover

    events = parse_sse(client(filled_store, Broken()).post("/api/chat", json={"question": "ration card"}).text)
    assert events[-1] == {"type": "error", "message": "model not found"}


def test_validation(filled_store):
    c = client(filled_store, FakeChat())
    assert c.post("/api/chat", json={"question": ""}).status_code == 422
    assert c.post("/api/chat", json={"question": "x", "lang": "hi"}).status_code == 422
