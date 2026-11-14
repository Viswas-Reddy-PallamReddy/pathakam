"""Evaluation over a JSONL question set.

Each line: {"id": "q1", "question": "...", "doc_id": "rythu-bharosa", "pages": [3], "answerable": true}
  doc_id/pages: where the answer is (pages optional). answerable=false: the documents do not contain it.

Retrieval metrics (always): hit@k and MRR over answerable questions.
Answer checks (--answers, calls the LLM): every answer cites a source, cited numbers exist,
reply language matches the question, unanswerable questions are declined; latency p50/p95.
"""

import json
import re
import statistics
from pathlib import Path

from .rag import NO_SOURCES, Answerer
from .retrieve import Retriever
from .text import detect_language

_CITE = re.compile(r"\[(\d+)\]")
# Phrases that indicate the model declined. Kept loose on purpose; review failures by hand.
_DECLINE = re.compile(r"could not find|not (?:in|mentioned|available)|do not (?:contain|have)|"
                      r"లేదు|లభించలేదు|సంప్రదించండి", re.I)


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _relevant(hit: dict, q: dict) -> bool:
    return hit["doc_id"] == q["doc_id"] and (not q.get("pages") or hit["page"] in q["pages"])


async def run(questions: list[dict], retriever: Retriever, answerer: Answerer | None, k: int) -> dict:
    rows = []
    for q in questions:
        res = await retriever.search(q["question"], k)
        hits = [{"doc_id": h.doc_id, "page": h.page} for h in res.hits]
        row = {"id": q["id"], "answerable": q.get("answerable", True), "retrieved": hits}
        if row["answerable"]:
            rank = next((i for i, h in enumerate(hits, 1) if _relevant(h, q)), None)
            row.update(hit=rank is not None, rr=1 / rank if rank else 0.0)

        if answerer is not None:
            text, done, n_sources, error = "", {}, 0, None
            async for ev in answerer.stream(q["question"]):
                if ev["type"] == "sources":
                    n_sources = len(ev["sources"])
                elif ev["type"] == "token":
                    text += ev["text"]
                elif ev["type"] == "done":
                    done = ev
                elif ev["type"] == "error":
                    error = ev["message"]
            cites = [int(c) for c in _CITE.findall(text)]
            declined = bool(_DECLINE.search(text)) or text.strip() in NO_SOURCES.values()
            row.update(
                answer=text, error=error, total_ms=done.get("total_ms"), first_token_ms=done.get("first_token_ms"),
                cited=bool(cites), citations_valid=all(1 <= c <= n_sources for c in cites),
                lang_ok=detect_language(text) == detect_language(q["question"]) if text else False,
                declined=declined,
            )
        rows.append(row)

    ans = [r for r in rows if r["answerable"]]
    summary: dict = {"questions": len(rows), "k": k}
    if ans:
        summary.update(hit_at_k=sum(r["hit"] for r in ans) / len(ans), mrr=sum(r["rr"] for r in ans) / len(ans))
    if answerer is not None:
        ok = [r for r in rows if not r.get("error")]
        un = [r for r in ok if not r["answerable"]]
        an = [r for r in ok if r["answerable"]]
        lat = sorted(r["total_ms"] for r in ok if r["total_ms"] is not None)
        summary.update(
            errors=len(rows) - len(ok),
            cited_rate=sum(r["cited"] for r in an) / len(an) if an else None,
            citations_valid_rate=sum(r["citations_valid"] for r in ok) / len(ok) if ok else None,
            lang_match_rate=sum(r["lang_ok"] for r in ok) / len(ok) if ok else None,
            decline_rate_unanswerable=sum(r["declined"] for r in un) / len(un) if un else None,
            false_decline_rate=sum(r["declined"] for r in an) / len(an) if an else None,
            latency_p50_ms=statistics.median(lat) if lat else None,
            latency_p95_ms=lat[max(0, round(0.95 * len(lat)) - 1)] if lat else None,
        )
    return {"summary": summary, "rows": rows}
