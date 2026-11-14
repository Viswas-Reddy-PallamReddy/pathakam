import json
from collections.abc import AsyncIterator, Sequence

import httpx

from .base import LLMError, Message


class OllamaChat:
    def __init__(self, base_url: str, model: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"ollama:{model}"
        self.timeout = timeout

    async def stream(self, messages: Sequence[Message], temperature: float = 0.1) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {"temperature": temperature},
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode(errors="replace")[:300]
                        raise LLMError(f"Ollama returned {resp.status_code}: {body}")
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        if "error" in obj:
                            raise LLMError(f"Ollama: {obj['error']}")
                        piece = obj.get("message", {}).get("content", "")
                        if piece:
                            yield piece
                        if obj.get("done"):
                            break
        except httpx.ConnectError as e:
            raise LLMError(f"Cannot reach Ollama at {self.base_url}. Is `ollama serve` running?") from e


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, batch_size: int = 16, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"ollama:{model}"
        self.batch_size = batch_size
        self.timeout = timeout

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for i in range(0, len(texts), self.batch_size):
                    batch = list(texts[i : i + self.batch_size])
                    resp = await client.post(
                        f"{self.base_url}/api/embed", json={"model": self.model, "input": batch}
                    )
                    if resp.status_code != 200:
                        raise LLMError(f"Ollama embed returned {resp.status_code}: {resp.text[:300]}")
                    out.extend(resp.json()["embeddings"])
        except httpx.ConnectError as e:
            raise LLMError(f"Cannot reach Ollama at {self.base_url}. Is `ollama serve` running?") from e
        return out
