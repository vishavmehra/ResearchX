"""PostgreSQL (Supabase) connection for backend services. Credentials come only from DATABASE_URL."""

from __future__ import annotations

import psycopg

from services.env import require


def connect(**kwargs) -> psycopg.Connection:
    return psycopg.connect(require("DATABASE_URL"), connect_timeout=20, **kwargs)
