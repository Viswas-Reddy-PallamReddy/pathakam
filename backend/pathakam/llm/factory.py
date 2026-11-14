from ..config import Settings
from .base import ChatModel, Embedder
from .ollama import OllamaChat, OllamaEmbedder
from .openrouter import OpenRouterChat


def make_chat(s: Settings) -> ChatModel:
    if s.llm_provider == "ollama":
        return OllamaChat(s.ollama_url, s.llm_model)
    if s.llm_provider == "openrouter":
        return OpenRouterChat(s.openrouter_url, s.openrouter_api_key, s.llm_model)
    raise ValueError(f"unknown llm_provider {s.llm_provider!r}")


def make_embedder(s: Settings) -> Embedder | None:
    if s.embed_provider == "none":
        return None
    if s.embed_provider == "ollama":
        return OllamaEmbedder(s.ollama_url, s.embed_model)
    raise ValueError(f"unknown embed_provider {s.embed_provider!r}")
