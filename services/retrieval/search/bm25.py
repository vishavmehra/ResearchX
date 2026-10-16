"""Okapi BM25 over retrieval chunks (lexical path of hybrid retrieval, ARCHITECTURE §51).

PostgreSQL full-text ranking (ts_rank) has no IDF and is not BM25, so the lexical index is built from rx.chunks and
saved as a compact artifact: CSR postings in numpy (~100K units, milliseconds per query, no extra service and no
index build at serverless cold start). The tokenizer keeps technical terms whole AND split: "llama-2-7b" ->
llama-2-7b, llama, 2, 7b; "3e-5" stays one token; so exact model / dataset / hyperparameter strings match exactly
while partial queries still match. Scores are candidate signals, never evidence.
"""

from __future__ import annotations

import json
import re
from array import array
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

VERSION = "bm25-0.2.0"
_TECH = re.compile(r"[a-z0-9]+(?:[.\-_/+][a-z0-9]+)*")
_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:e-?\d+)?")
STOP = frozenset("a an and are as at be by for from has have in is it its of on or that the this to was were with we our".split())


def tokenize(text: str) -> list[str]:
    out = []
    for t in _TECH.findall(text.lower()):
        if t in STOP:
            continue
        out.append(t)
        parts = re.split(r"[.\-_/+]", t)
        if len(parts) > 1 and not _NUMBER.fullmatch(t):  # numbers like 0.001 / 3e-5 stay whole only
            out.extend(p for p in parts if p and p not in STOP)
    return out


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.vocab: dict[str, int] = {}

    def build(self, docs: Iterable[tuple[str, str]]) -> BM25:
        terms, postings, freqs, lens = array("i"), array("i"), array("i"), array("i")
        for doc_id, text in docs:
            i = len(self.ids)
            self.ids.append(doc_id)
            tf = Counter(tokenize(text))
            lens.append(sum(tf.values()))
            for term, n in tf.items():
                terms.append(self.vocab.setdefault(term, len(self.vocab)))
                postings.append(i)
                freqs.append(n)
        t = np.frombuffer(terms, dtype=np.int32)
        order = np.argsort(t, kind="stable")  # group postings by term, docs stay ascending
        counts = np.bincount(t, minlength=len(self.vocab))
        self._set(np.concatenate([[0], np.cumsum(counts)]).astype(np.int64), np.frombuffer(postings, dtype=np.int32)[order],
                  np.frombuffer(freqs, dtype=np.int32)[order], np.frombuffer(lens, dtype=np.int32))
        return self

    def _set(self, indptr, indices, tf, doc_len) -> None:
        self.indptr, self.indices = indptr, indices
        self.tf = tf.astype(np.float32)
        self.doc_len = doc_len.astype(np.float32)
        n, df = len(self.ids), np.diff(indptr)
        self.avgdl = float(self.doc_len.mean()) if n else 0.0
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self._norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))

    def search(self, query: str, k: int = 50, keep: Callable[[str], bool] | None = None) -> list[tuple[str, float]]:
        scores = np.zeros(len(self.ids), dtype=np.float32)
        for term in set(tokenize(query)):
            j = self.vocab.get(term)
            if j is None:
                continue
            s, e = self.indptr[j], self.indptr[j + 1]
            d, tf = self.indices[s:e], self.tf[s:e]
            scores[d] += self.idf[j] * tf * (self.k1 + 1) / (tf + self._norm[d])
        hit = np.flatnonzero(scores)
        n = min(len(hit), k * 4 if keep else k)
        if n == 0:
            return []
        top = hit[np.argpartition(-scores[hit], n - 1)[:n]] if n < len(hit) else hit
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.ids[i], float(scores[i])) for i in top if keep is None or keep(self.ids[i])][:k]

    def save(self, path: Path, meta: dict | None = None) -> None:
        """<path>.npz holds the postings; <path>.json the ids, vocabulary, parameters and build metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(_with(path, ".npz"), indptr=self.indptr, indices=self.indices, tf=self.tf.astype(np.uint16),
                            doc_len=self.doc_len.astype(np.int32))
        vocab = sorted(self.vocab, key=self.vocab.__getitem__)
        _with(path, ".json").write_text(json.dumps({"version": VERSION, "k1": self.k1, "b": self.b, **(meta or {}),
                                                    "ids": self.ids, "vocab": vocab}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BM25:
        meta = json.loads(_with(path, ".json").read_text(encoding="utf-8"))
        if meta["version"] != VERSION:
            raise ValueError(f"index {path} is {meta['version']}, code is {VERSION}: rebuild it")
        m = cls(meta["k1"], meta["b"])
        m.ids, m.vocab = meta["ids"], {t: i for i, t in enumerate(meta["vocab"])}
        z = np.load(_with(path, ".npz"))
        m._set(z["indptr"], z["indices"], z["tf"], z["doc_len"])
        return m


def _with(path: Path, suffix: str) -> Path:
    return path.parent / (path.name + suffix)  # not with_suffix: release names like v0.2.0 contain dots
