"""SQLite store: documents, chunks, a trigram FTS5 index (BM25) and embeddings as float32 blobs.

Trigram tokenisation is used because Telugu is agglutinative: "రైతు" (farmer) must match
"రైతులకు" (to the farmers). Vector search is brute-force cosine in NumPy, which is fast enough
for tens of thousands of chunks and keeps the stack free of a separate vector database.
"""

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .text import query_terms

SCHEMA = """
create table if not exists documents (
    id text primary key,
    title text not null,
    scheme text,
    language text,
    source_url text,
    path text,
    sha256 text,
    pages integer,
    added_at text
);
create table if not exists chunks (
    id integer primary key autoincrement,
    doc_id text not null references documents(id) on delete cascade,
    page integer not null,
    ordinal integer not null,
    text text not null,
    embedding blob
);
create index if not exists chunks_doc on chunks(doc_id);
create virtual table if not exists chunks_fts using fts5(
    text, content='chunks', content_rowid='id', tokenize='trigram'
);
create trigger if not exists chunks_ai after insert on chunks begin
    insert into chunks_fts(rowid, text) values (new.id, new.text);
end;
create trigger if not exists chunks_ad after delete on chunks begin
    insert into chunks_fts(chunks_fts, rowid, text) values ('delete', old.id, old.text);
end;
create table if not exists meta (key text primary key, value text);
"""



@dataclass(frozen=True)
class Document:
    id: str
    title: str
    scheme: str | None = None
    language: str | None = None
    source_url: str | None = None
    path: str | None = None
    sha256: str | None = None
    pages: int | None = None


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    doc_id: str
    title: str
    source_url: str | None
    page: int
    text: str
    score: float


class Store:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("pragma foreign_keys = on")
        self.db.execute("pragma journal_mode = wal")
        self.db.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._vec_ids: np.ndarray | None = None
        self._vec_mat: np.ndarray | None = None

    # ---------- writes ----------

    def document_sha(self, doc_id: str) -> str | None:
        row = self.db.execute("select sha256 from documents where id = ?", (doc_id,)).fetchone()
        return row["sha256"] if row else None

    def replace_document(
        self,
        doc: Document,
        chunks: list[tuple[int, int, str]],
        embeddings: list[list[float]] | None,
        embed_model: str | None,
    ) -> None:
        """Insert or replace a document and all of its chunks: (page, ordinal, text)."""
        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError("one embedding per chunk is required")
        with self._lock, self.db:
            if embeddings:
                self._check_embedding_space(embed_model, len(embeddings[0]))
            self.db.execute("delete from chunks where doc_id = ?", (doc.id,))
            self.db.execute("delete from documents where id = ?", (doc.id,))
            self.db.execute(
                "insert into documents values (?,?,?,?,?,?,?,?,?)",
                (doc.id, doc.title, doc.scheme, doc.language, doc.source_url, doc.path,
                 doc.sha256, doc.pages, datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
            rows = []
            for i, (page, ordinal, text) in enumerate(chunks):
                blob = np.asarray(embeddings[i], dtype=np.float32).tobytes() if embeddings else None
                rows.append((doc.id, page, ordinal, text, blob))
            self.db.executemany(
                "insert into chunks(doc_id, page, ordinal, text, embedding) values (?,?,?,?,?)", rows
            )
        self._vec_ids = self._vec_mat = None

    def delete_document(self, doc_id: str) -> None:
        with self._lock, self.db:
            self.db.execute("delete from chunks where doc_id = ?", (doc_id,))
            self.db.execute("delete from documents where id = ?", (doc_id,))
        self._vec_ids = self._vec_mat = None

    def _check_embedding_space(self, model: str | None, dim: int) -> None:
        meta = dict(self.db.execute("select key, value from meta").fetchall())
        if "embed_dim" in meta:
            if int(meta["embed_dim"]) != dim or meta.get("embed_model") != model:
                raise ValueError(
                    f"index was built with {meta.get('embed_model')} ({meta['embed_dim']}d); "
                    f"got {model} ({dim}d). Rebuild the index to switch embedding models."
                )
        else:
            self.db.executemany(
                "insert into meta values (?, ?)", [("embed_model", model or ""), ("embed_dim", str(dim))]
            )

    # ---------- reads ----------

    def keyword_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """BM25 over trigrams. Terms shorter than 3 characters cannot be matched by a trigram index."""
        terms = query_terms(query)
        if not terms:
            return []
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        rows = self.db.execute(
            "select rowid, bm25(chunks_fts) as s from chunks_fts where chunks_fts match ? order by s limit ?",
            (match, k),
        ).fetchall()
        return [(r["rowid"], -r["s"]) for r in rows]  # bm25() is lower-is-better

    def _load_vectors(self) -> None:
        rows = self.db.execute("select id, embedding from chunks where embedding is not null").fetchall()
        if not rows:
            self._vec_ids, self._vec_mat = np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32)
            return
        mat = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        self._vec_ids = np.array([r["id"] for r in rows], dtype=np.int64)
        self._vec_mat = mat

    def has_vectors(self) -> bool:
        if self._vec_ids is None:
            self._load_vectors()
        return bool(len(self._vec_ids))

    def vector_search(self, query_vec: list[float], k: int) -> list[tuple[int, float]]:
        if not self.has_vectors():
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        if q.shape[0] != self._vec_mat.shape[1]:
            raise ValueError("query embedding dimension does not match the index")
        q /= np.linalg.norm(q) + 1e-12
        scores = self._vec_mat @ q
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(self._vec_ids[i]), float(scores[i])) for i in top]

    def get_hits(self, scored: list[tuple[int, float]]) -> list[Hit]:
        if not scored:
            return []
        ids = [cid for cid, _ in scored]
        rows = self.db.execute(
            f"select c.id, c.doc_id, c.page, c.text, d.title, d.source_url from chunks c "
            f"join documents d on d.id = c.doc_id where c.id in ({','.join('?' * len(ids))})",
            ids,
        ).fetchall()
        by_id = {r["id"]: r for r in rows}
        return [
            Hit(cid, r["doc_id"], r["title"], r["source_url"], r["page"], r["text"], score)
            for cid, score in scored
            if (r := by_id.get(cid)) is not None
        ]

    def documents(self) -> list[dict]:
        rows = self.db.execute(
            "select d.*, count(c.id) as chunks from documents d left join chunks c on c.doc_id = d.id "
            "group by d.id order by d.title"
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        docs = self.db.execute("select count(*) from documents").fetchone()[0]
        chunks = self.db.execute("select count(*) from chunks").fetchone()[0]
        vecs = self.db.execute("select count(*) from chunks where embedding is not null").fetchone()[0]
        meta = dict(self.db.execute("select key, value from meta").fetchall())
        return {"documents": docs, "chunks": chunks, "embedded_chunks": vecs,
                "embed_model": meta.get("embed_model")}

    def close(self) -> None:
        self.db.close()
