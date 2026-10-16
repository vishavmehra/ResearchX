"""Authorization scope of every retrieval call (docs/SECURITY.md: tenant isolation).

Retrieval always takes a scope and has no default, so no caller can forget authorization. The shared ResearchX corpus
(curated papers, public experiments and evidence) is readable by every signed-in user. Private user documents are
searched only for `scope.user`, inside a transaction running as that user (RLS) AND with an explicit owner filter in
SQL, so another tenant's private text can never reach ranking, reranking, caching or the model context.
"""

from __future__ import annotations

from dataclasses import dataclass

from services.auth.supabase import AuthUser


@dataclass(frozen=True)
class RetrievalScope:
    user: AuthUser | None  # None = shared corpus only (offline scripts and evaluation): no private data, no answer cache

    @property
    def user_id(self) -> str | None:
        return self.user.id if self.user else None
