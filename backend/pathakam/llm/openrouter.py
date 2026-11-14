import json
from collections.abc import AsyncIterator, Sequence

import httpx

from .base import LLMError, Message


class OpenRouterChat:
    """Any chat model on OpenRouter (OpenAI-compatible streaming API)."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0):
        if not api_key:
            raise LLMError("PATHAKAM_OPENROUTER_API_KEY is not set")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.name = f"openrouter:{model}"
        self.timeout = timeout

    async def stream(self, messages: Sequence[Message], temperature: float = 0.1) -> AsyncIterator[str]:
        headers = {"Authorization": f"Bearer {self.api_key}", "X-Title": "Pathakam"}
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=headers
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")[:300]
                    raise LLMError(f"OpenRouter returned {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    # SSE: "data: {...}"; lines starting with ":" are keep-alive comments
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    if "error" in obj:
                        raise LLMError(f"OpenRouter: {obj['error'].get('message', obj['error'])}")
                    choices = obj.get("choices") or []
                    piece = (choices[0].get("delta") or {}).get("content") if choices else None
                    if piece:
                        yield piece
