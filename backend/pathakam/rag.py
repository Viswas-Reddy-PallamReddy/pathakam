"""Grounded answering: retrieve, build a cited prompt, stream the reply with timing events."""

import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from .llm import ChatModel, LLMError, Message
from .retrieve import Retriever
from .store import Hit
from .text import detect_language

SYSTEM_PROMPT = """You are Pathakam, a helpdesk assistant for Andhra Pradesh and Telangana government welfare schemes.

Rules:
1. Answer ONLY from the numbered sources provided. Never use outside knowledge for amounts, dates, eligibility, or documents.
2. Put the source number in square brackets after every fact, like [1] or [2][3].
3. If the sources do not answer the question, say so plainly and suggest the official scheme helpline or the nearest village/ward secretariat or MeeSeva centre. Do not guess.
4. Sources can be outdated. If a source shows a year or GO number, mention it.
5. Be short and practical: at most 6 sentences, or a short list of steps or documents.
6. {language_rule}"""

LANGUAGE_RULES = {
    "te": "Reply in simple Telugu, in Telugu script. Keep scheme names and official terms as they appear in the sources.",
    "en": "Reply in simple English.",
}

NO_SOURCES = {
    "te": "క్షమించండి, ఈ ప్రశ్నకు సమాధానం నా దగ్గర ఉన్న అధికారిక పత్రాల్లో లేదు. దయచేసి సంబంధిత పథకం హెల్ప్‌లైన్‌ను లేదా మీ దగ్గరి గ్రామ/వార్డు సచివాలయం లేదా మీసేవ కేంద్రాన్ని సంప్రదించండి.",
    "en": "Sorry, I could not find this in the official documents I have. Please contact the scheme's helpline or your nearest village/ward secretariat or MeeSeva centre.",
}


def format_sources(hits: Sequence[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] {h.title}, page {h.page}\n{h.text}" for i, h in enumerate(hits, start=1)
    )


def build_messages(question: str, hits: Sequence[Hit], history: Sequence[Message], lang: str) -> list[Message]:
    system = SYSTEM_PROMPT.format(language_rule=LANGUAGE_RULES[lang])
    user = f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"
    return [Message("system", system), *history, Message("user", user)]


def source_payload(hits: Sequence[Hit]) -> list[dict[str, Any]]:
    return [
        {"n": i, "doc_id": h.doc_id, "title": h.title, "page": h.page, "url": h.source_url,
         "snippet": h.text[:400]}
        for i, h in enumerate(hits, start=1)
    ]


class Answerer:
    def __init__(self, retriever: Retriever, chat: ChatModel, top_k: int = 6, temperature: float = 0.1):
        self.retriever = retriever
        self.chat = chat
        self.top_k = top_k
        self.temperature = temperature

    async def stream(
        self, question: str, history: Sequence[Message] = (), lang: str = "auto"
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield events: meta, sources, token*, done | error."""
        t0 = time.perf_counter()
        lang = detect_language(question) if lang == "auto" else lang
        result = await self.retriever.search(question, self.top_k)
        t_retrieved = time.perf_counter()
        yield {"type": "meta", "lang": lang, "model": self.chat.name}
        yield {"type": "sources", "sources": source_payload(result.hits)}

        timing = {"retrieval_ms": round((t_retrieved - t0) * 1000, 1),
                  "keyword_candidates": result.keyword_count, "vector_candidates": result.vector_count}

        if not result.hits:
            # Nothing to ground on: answer with a fixed message instead of letting the model guess.
            yield {"type": "token", "text": NO_SOURCES[lang]}
            yield {"type": "done", "grounded": False, **timing,
                   "total_ms": round((time.perf_counter() - t0) * 1000, 1)}
            return

        messages = build_messages(question, result.hits, history, lang)
        first = None
        chars = 0
        try:
            async for piece in self.chat.stream(messages, self.temperature):
                if first is None:
                    first = time.perf_counter()
                chars += len(piece)
                yield {"type": "token", "text": piece}
        except LLMError as e:
            yield {"type": "error", "message": str(e)}
            return
        end = time.perf_counter()
        yield {"type": "done", "grounded": True, **timing,
               "first_token_ms": round(((first or end) - t0) * 1000, 1),
               "total_ms": round((end - t0) * 1000, 1), "answer_chars": chars}
