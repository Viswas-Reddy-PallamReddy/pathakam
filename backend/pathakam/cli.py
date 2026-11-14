"""Command line: fetch | ingest | stats | ask | eval.

  python -m pathakam fetch                     download files listed in data/sources.yaml
  python -m pathakam ingest ../data/raw        extract, chunk, embed and index documents
  python -m pathakam stats                     what is in the index
  python -m pathakam ask "question"            answer from the terminal (streams)
  python -m pathakam eval ../eval/questions.jsonl [--answers]
"""

import argparse
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path

import httpx
import yaml

from .config import DATA_DIR, Settings
from .ingest.chunk import chunk_pages
from .ingest.extract import SUPPORTED, extract_pages
from .llm import make_chat, make_embedder
from .rag import Answerer
from .retrieve import Retriever
from .store import Document, Store
from .text import script_profile


def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {d["file"]: d for d in data.get("documents", [])}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "doc"


def cmd_fetch(args, s: Settings) -> int:
    manifest = load_manifest(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    failed = 0
    with httpx.Client(follow_redirects=True, timeout=60, headers={"User-Agent": "pathakam-fetch/0.1"}) as client:
        for name, entry in manifest.items():
            url = entry.get("download_url") or entry.get("source_url")
            dest = args.out / name
            if dest.exists() or not url:
                print(f"skip  {name}" + ("" if url else " (no url)"))
                continue
            try:
                resp = client.get(url)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                print(f"ok    {name}  {len(resp.content) // 1024} KB")
            except httpx.HTTPError as e:
                failed += 1
                print(f"FAIL  {name}: {e}", file=sys.stderr)
    return 1 if failed else 0


async def _ingest(args, s: Settings) -> int:
    manifest = load_manifest(args.manifest)
    store = Store(s.db_path)
    embedder = make_embedder(s)
    files = []
    for p in args.paths:
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED) if p.is_dir() else [p]
    if not files:
        print("no .pdf/.txt/.md files found", file=sys.stderr)
        return 1

    for f in files:
        entry = manifest.get(f.name, {})
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        doc_id = entry.get("id") or slug(f.stem)
        if not args.force and store.document_sha(doc_id) == sha:
            print(f"same  {f.name}")
            continue

        pages = extract_pages(f)
        usable, flagged = [], []
        for page in pages:
            (flagged if script_profile(page.text).needs_ocr else usable).append(page)
        chunks = chunk_pages(usable, args.max_chars, args.overlap)
        if not chunks:
            print(f"EMPTY {f.name}: no extractable text on {len(pages)} pages (scanned or legacy font, needs OCR)")
            continue

        texts = [c.text for c in chunks]
        vectors = await embedder.embed(texts) if embedder else None
        doc = Document(
            id=doc_id, title=entry.get("title") or f.stem, scheme=entry.get("scheme"),
            language=entry.get("language"), source_url=entry.get("source_url"), path=str(f),
            sha256=sha, pages=len(pages),
        )
        store.replace_document(doc, [(c.page, c.ordinal, c.text) for c in chunks], vectors,
                               s.embed_model if embedder else None)
        te = sum(script_profile(c.text).telugu for c in chunks) / len(chunks)
        note = f"  pages needing OCR: {[p.number for p in flagged]}" if flagged else ""
        no_url = "" if doc.source_url else "  (no source_url in manifest)"
        print(f"ok    {f.name}: {len(pages)} pages, {len(chunks)} chunks, Telugu {te:.0%}{note}{no_url}")
    print(store.stats())
    return 0


async def _ask(args, s: Settings) -> int:
    store = Store(s.db_path)
    answerer = Answerer(Retriever(store, make_embedder(s)), make_chat(s), s.top_k, s.temperature)
    sources = []
    async for ev in answerer.stream(args.question, lang=args.lang):
        if ev["type"] == "sources":
            sources = ev["sources"]
        elif ev["type"] == "token":
            print(ev["text"], end="", flush=True)
        elif ev["type"] == "error":
            print(f"\nerror: {ev['message']}", file=sys.stderr)
            return 1
        elif ev["type"] == "done":
            print("\n")
            for src in sources:
                print(f"  [{src['n']}] {src['title']}, p.{src['page']}")
            print(f"  retrieval {ev['retrieval_ms']} ms, total {ev['total_ms']} ms")
    return 0


async def _eval(args, s: Settings) -> int:
    from .evaluate import load_questions, run

    retriever = Retriever(Store(s.db_path), make_embedder(s))
    answerer = Answerer(retriever, make_chat(s), s.top_k, s.temperature) if args.answers else None
    report = await run(load_questions(args.questions), retriever, answerer, s.top_k)
    print(json.dumps(report["summary"], indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    s = Settings()
    ap = argparse.ArgumentParser(prog="pathakam")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download documents listed in the manifest")
    f.add_argument("--manifest", type=Path, default=DATA_DIR / "sources.yaml")
    f.add_argument("--out", type=Path, default=DATA_DIR / "raw")

    i = sub.add_parser("ingest", help="index documents")
    i.add_argument("paths", type=Path, nargs="+")
    i.add_argument("--manifest", type=Path, default=DATA_DIR / "sources.yaml")
    i.add_argument("--force", action="store_true", help="re-index unchanged files")
    i.add_argument("--max-chars", type=int, default=900)
    i.add_argument("--overlap", type=int, default=150)

    sub.add_parser("stats", help="show index contents")

    a = sub.add_parser("ask", help="ask a question from the terminal")
    a.add_argument("question")
    a.add_argument("--lang", choices=["auto", "te", "en"], default="auto")

    e = sub.add_parser("eval", help="retrieval metrics (and answer checks with --answers)")
    e.add_argument("questions", type=Path)
    e.add_argument("--answers", action="store_true", help="also generate answers with the LLM")
    e.add_argument("--out", type=Path, help="write the full JSON report here")

    args = ap.parse_args(argv)
    if args.cmd == "eval":
        return asyncio.run(_eval(args, s))
    if args.cmd == "fetch":
        return cmd_fetch(args, s)
    if args.cmd == "ingest":
        return asyncio.run(_ingest(args, s))
    if args.cmd == "stats":
        store = Store(s.db_path)
        print(store.stats())
        for d in store.documents():
            print(f"  {d['id']}: {d['title']} ({d['pages']} pages, {d['chunks']} chunks)")
        return 0
    return asyncio.run(_ask(args, s))
