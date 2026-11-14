from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from ..text import normalize

SUPPORTED = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class Page:
    number: int  # 1-based
    text: str


def extract_pages(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [Page(i + 1, normalize(p.extract_text() or "")) for i, p in enumerate(reader.pages)]
    if suffix in {".txt", ".md"}:
        # Plain text: a form feed (\f) marks a page break; otherwise the whole file is page 1.
        raw = path.read_text(encoding="utf-8")
        return [Page(i + 1, normalize(t)) for i, t in enumerate(raw.split("\f"))]
    raise ValueError(f"unsupported file type: {path.name}")
