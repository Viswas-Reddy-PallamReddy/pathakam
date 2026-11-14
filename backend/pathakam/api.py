"""HTTP API. Run with: uvicorn pathakam.api:create_app --factory --reload"""

import json
import logging
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings
from .llm import ChatModel, Embedder, LLMError, Message, make_chat, make_embedder
from .rag import Answerer
from .retrieve import Retriever
from .store import Store

log = logging.getLogger("pathakam")
_DEFAULT = object()


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = []
    lang: Literal["auto", "te", "en"] = "auto"


def _sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def create_app(
    settings: Settings | None = None,
    chat: ChatModel | None = None,
    embedder: Embedder | None | object = _DEFAULT,
    store: Store | None = None,
) -> FastAPI:
    s = settings or Settings()
    store = store or Store(s.db_path)
    embedder = make_embedder(s) if embedder is _DEFAULT else embedder
    try:
        chat = chat or make_chat(s)
        chat_error = None
    except LLMError as e:  # e.g. OpenRouter selected without a key: keep serving /health
        chat, chat_error = None, str(e)

    retriever = Retriever(store, embedder)
    answerer = Answerer(retriever, chat, s.top_k, s.temperature) if chat else None

    app = FastAPI(title="Pathakam", version=__version__)
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["*"], allow_headers=["*"])
    app.state.store = store

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok" if answerer else "degraded",
            "version": __version__,
            "llm": chat.name if chat else None,
            "llm_error": chat_error,
            "embedder": embedder.name if embedder else None,
            **store.stats(),
        }

    @app.get("/api/documents")
    def documents() -> list[dict]:
        return [
            {k: d[k] for k in ("id", "title", "scheme", "language", "source_url", "pages", "chunks")}
            for d in store.documents()
        ]

    @app.post("/api/chat")
    async def chat_endpoint(req: ChatRequest) -> StreamingResponse:
        history = [Message(t.role, t.content) for t in req.history[-2 * s.max_history_turns :]]

        async def events():
            if answerer is None:
                yield _sse({"type": "error", "message": chat_error or "no chat model configured"})
                return
            try:
                async for ev in answerer.stream(req.question, history, req.lang):
                    if ev["type"] == "done":
                        log.info("chat %s", {k: v for k, v in ev.items() if k != "type"})
                    yield _sse(ev)
            except LLMError as e:  # e.g. the embedder is unreachable during retrieval
                yield _sse({"type": "error", "message": str(e)})

        return StreamingResponse(
            events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
