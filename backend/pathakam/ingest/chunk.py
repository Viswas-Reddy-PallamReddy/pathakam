"""Split pages into overlapping chunks. Chunks never cross pages, so every citation has one page number."""

import re
from collections.abc import Iterator
from dataclasses import dataclass

from .extract import Page

# Sentence ends: Latin punctuation, the Devanagari danda (also used in some Telugu text), and "?".
_SENTENCE_END = re.compile(r"(?<=[.!?।॥])\s+")


@dataclass(frozen=True)
class Chunk:
    page: int
    ordinal: int  # position within the document
    text: str


def _units(text: str, max_chars: int) -> Iterator[str]:
    """Paragraphs; paragraphs that are too long are split into sentences, then hard-wrapped."""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            yield para
            continue
        for sent in _SENTENCE_END.split(para):
            sent = sent.strip()
            while len(sent) > max_chars:
                cut = sent.rfind(" ", 0, max_chars)
                cut = cut if cut > max_chars // 2 else max_chars
                yield sent[:cut].strip()
                sent = sent[cut:].strip()
            if sent:
                yield sent


def chunk_pages(pages: list[Page], max_chars: int = 900, overlap_chars: int = 150) -> list[Chunk]:
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")
    chunks: list[Chunk] = []
    for page in pages:
        buf: list[str] = []
        size = 0
        for unit in _units(page.text, max_chars):
            if buf and size + len(unit) + 1 > max_chars:
                chunks.append(Chunk(page.number, len(chunks), "\n".join(buf)))
                # Carry the trailing units (up to overlap_chars) into the next chunk.
                tail: list[str] = []
                tail_size = 0
                for u in reversed(buf):
                    if tail_size + len(u) + 1 > overlap_chars:
                        break
                    tail.insert(0, u)
                    tail_size += len(u) + 1
                buf, size = tail, tail_size
            buf.append(unit)
            size += len(unit) + 1
        if buf:
            chunks.append(Chunk(page.number, len(chunks), "\n".join(buf)))
    return chunks
