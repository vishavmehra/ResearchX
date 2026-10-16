"""In-process caches of the research runtime (docs/ARCHITECTURE.md §68): query embeddings and reranker grades.

Keys are versioned (model id, embedding or prompt version, dataset release) and scoped to one user ("anon" for runs
without one), so nothing derived from one user's question or private documents is served to another. Entries expire
after ttl_s; beyond max_items the least recently used go first; a restart empties both. Durable caches (answers,
external search results) live in rx.cache_entries, per user under RLS.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any


def key(*parts) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


class TTLCache:
    def __init__(self, max_items: int, ttl_s: float):
        self.max_items, self.ttl_s = max_items, ttl_s
        self._d: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, k: str) -> Any | None:
        with self._lock:
            hit = self._d.get(k)
            if hit is None or hit[0] < time.monotonic():
                self._d.pop(k, None)
                self.misses += 1
                return None
            self._d.move_to_end(k)
            self.hits += 1
            return hit[1]

    def put(self, k: str, value: Any) -> None:
        with self._lock:
            self._d[k] = (time.monotonic() + self.ttl_s, value)
            self._d.move_to_end(k)
            while len(self._d) > self.max_items:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
