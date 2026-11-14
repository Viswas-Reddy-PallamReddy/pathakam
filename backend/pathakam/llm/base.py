from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMError(RuntimeError):
    """A provider failed (unreachable, bad key, unknown model, ...). The message is safe to show."""


class ChatModel(Protocol):
    name: str

    def stream(self, messages: Sequence[Message], temperature: float) -> AsyncIterator[str]:
        """Yield the reply as text fragments."""
        ...


class Embedder(Protocol):
    name: str

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text."""
        ...
