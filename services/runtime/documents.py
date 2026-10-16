"""Private user documents (docs/SECURITY.md: private data, tenant isolation).

Upload -> original kept in Supabase Storage at user-documents/<user id>/<document id> (Storage policies, called with
the user's own token) -> text extracted -> chunked -> embedded with the corpus model at the serving dimension ->
rx.user_chunks rows owned by the user (RLS). Private documents are never added to the shared corpus. Per-user limits
keep one account from exhausting the Free-plan database (500 MB shared by everyone).
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import uuid

import httpx
import numpy as np

BUCKET = "user-documents"
MAX_BYTES = 2_000_000
MAX_DOCS = 10
MAX_CHUNKS = 400          # per user, across documents
CHUNK_CHARS = 1200
DIMS = 256                # must equal config/runtime.yaml retrieval.dense_dims (same space as the corpus vectors)
TYPES = ("application/pdf", "text/plain", "text/markdown")


class DocumentError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def extract_text(data: bytes, media_type: str) -> str:
    if media_type == "application/pdf":
        from pypdf import PdfReader  # already a dependency (PDF fallback of the dataset pipeline)
        try:
            return "\n\n".join((page.extract_text() or "") for page in PdfReader(io.BytesIO(data)).pages)
        except Exception as e:
            raise DocumentError(422, "The PDF could not be read.") from e
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise DocumentError(422, "Text files must be UTF-8.") from e


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Paragraph packing up to `size` characters; an over-long paragraph is cut at the last space before the limit."""
    chunks, cur = [], ""
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        while len(para) > size:
            cut = para.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            chunks.append(para[:cut].strip())
            para = para[cut:].strip()
        if not para:
            continue
        if cur and len(cur) + 1 + len(para) > size:
            chunks.append(cur)
            cur = para
        else:
            cur = f"{cur}\n{para}" if cur else para
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c]


class Storage:
    """Supabase Storage with the USER's access token: the bucket policies (owner folder) authorise every call."""

    def __init__(self, user, transport: httpx.BaseTransport | None = None):
        self.base = os.environ["SUPABASE_URL"].rstrip("/") + "/storage/v1/object"
        self.headers = {"Authorization": f"Bearer {user.token}", "apikey": os.environ["SUPABASE_PUBLISHABLE_KEY"]}
        self.transport = transport

    def _client(self) -> httpx.Client:
        return httpx.Client(headers=self.headers, timeout=60, transport=self.transport)

    def upload(self, path: str, data: bytes, media_type: str) -> None:
        with self._client() as c:
            r = c.post(f"{self.base}/{BUCKET}/{path}", content=data, headers={"Content-Type": media_type, "x-upsert": "false"})
        if r.status_code >= 300:
            raise DocumentError(502, f"Storage upload failed ({r.status_code}).")

    def remove(self, paths: list[str]) -> None:
        if not paths:
            return
        with self._client() as c:
            r = c.request("DELETE", f"{self.base}/{BUCKET}", json={"prefixes": paths})
        if r.status_code >= 300:
            raise DocumentError(502, f"Storage delete failed ({r.status_code}).")

    def list(self, prefix: str) -> list[str]:
        with self._client() as c:
            r = c.post(f"{self.base}/list/{BUCKET}", json={"prefix": prefix, "limit": 1000, "offset": 0})
        if r.status_code >= 300:
            raise DocumentError(502, f"Storage listing failed ({r.status_code}).")
        return [f"{prefix.rstrip('/')}/{o['name']}" for o in r.json() if o.get("id")]


def ingest(uq, user, gw, storage: Storage, title: str, media_type: str, data: bytes) -> dict:
    """uq(sql, params) runs one statement as `user` (RLS). Returns the document row as a dict."""
    if media_type not in TYPES:
        raise DocumentError(415, "Upload a PDF, plain-text or Markdown file.")
    if not data or len(data) > MAX_BYTES:
        raise DocumentError(413, f"Files must be between 1 byte and {MAX_BYTES // 1_000_000} MB.")
    title = (title or "Untitled document").strip()[:200] or "Untitled document"
    docs, used = uq("select count(*), coalesce(sum(chunks), 0) from rx.user_documents where user_id = %s", (user.id,))[0]
    if docs >= MAX_DOCS:
        raise DocumentError(409, f"You can keep at most {MAX_DOCS} private documents. Delete one first.")
    chunks = chunk_text(extract_text(data, media_type))
    if not chunks:
        raise DocumentError(422, "No text could be extracted from this file.")
    if used + len(chunks) > MAX_CHUNKS:
        raise DocumentError(413, f"This document needs {len(chunks)} passages but only {MAX_CHUNKS - used} remain in your quota.")
    sha = hashlib.sha256(data).hexdigest()
    if uq("select 1 from rx.user_documents where user_id = %s and content_sha256 = %s", (user.id, sha)):
        raise DocumentError(409, "This document is already in your library.")
    doc_id = str(uuid.uuid4())
    path = f"{user.id}/{doc_id}"
    uq("insert into rx.user_documents (id, user_id, title, media_type, bytes, content_sha256, storage_path) values (%s,%s,%s,%s,%s,%s,%s)",
       (doc_id, user.id, title, media_type, len(data), sha, path))
    try:
        storage.upload(path, data, media_type)
        rows = []
        for i in range(0, len(chunks), 64):
            vecs, call = gw.embed("embedding", chunks[i:i + 64])
            if vecs is None:
                raise DocumentError(502, f"Embedding failed ({call.status}).")
            v = np.asarray(vecs, dtype=np.float32)[:, :DIMS]
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            rows += [(doc_id, user.id, i + j, chunks[i + j], "[" + ",".join(f"{x:.5f}" for x in row) + "]") for j, row in enumerate(v)]
        uq("insert into rx.user_chunks (document_id, user_id, ordinal, text, embedding) values "
           + ",".join(["(%s,%s,%s,%s,%s::extensions.halfvec)"] * len(rows)), tuple(x for r in rows for x in r))
        uq("update rx.user_documents set status = 'READY', chunks = %s where id = %s", (len(rows), doc_id))
    except Exception as e:
        uq("update rx.user_documents set status = 'FAILED', error = %s where id = %s", (str(getattr(e, "message", type(e).__name__))[:300], doc_id))
        raise
    return {"id": doc_id, "title": title, "media_type": media_type, "bytes": len(data), "chunks": len(chunks), "status": "READY"}


def delete_document(uq, user, storage: Storage, doc_id: str) -> bool:
    row = uq("select storage_path from rx.user_documents where id = %s and user_id = %s", (doc_id, user.id))
    if not row:
        return False
    if row[0][0]:
        storage.remove([row[0][0]])
    uq("delete from rx.user_documents where id = %s and user_id = %s", (doc_id, user.id))  # chunks cascade
    return True


def delete_all_objects(user, storage: Storage) -> int:
    """Account deletion: remove every stored original of this user (Storage rows cannot be deleted via SQL safely)."""
    paths = storage.list(f"{user.id}/")
    storage.remove(paths)
    return len(paths)
