"""Create a confirmed e-mail/password account for demos (no confirmation e-mail, no MFA) and check that it signs in.

Writes one row to auth.users and one to auth.identities through DATABASE_URL (the same statements the integration
tests use). Idempotent: an existing account with that e-mail is left unchanged. The generated password is printed once.
Prefer an address on a domain that receives no mail (for example example.com, RFC 2606), so nobody can take the account
over through a password-reset inbox. A shared account must never be used for private documents.

Usage:
    python scripts/dev/create_demo_account.py --email researchx.judge@example.com
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.env import load_env  # noqa: E402
from services.storage.db import connect  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--email", required=True)
    a = ap.parse_args()
    load_env()
    email = a.email.strip().lower()
    with connect(autocommit=True) as conn:
        row = conn.execute("select id from auth.users where email = %s", (email,)).fetchone()
        if row:
            print(f"exists: {email} ({row[0]}); not changed")
            return
        uid, password = str(uuid.uuid4()), "Demo-" + secrets.token_urlsafe(12)
        conn.execute("insert into auth.users (instance_id, id, aud, role, email, encrypted_password, email_confirmed_at, raw_app_meta_data, "
                     "raw_user_meta_data, created_at, updated_at, confirmation_token, recovery_token, email_change_token_new, email_change) "
                     "values ('00000000-0000-0000-0000-000000000000', %s, 'authenticated', 'authenticated', %s, "
                     "extensions.crypt(%s, extensions.gen_salt('bf')), now(), '{\"provider\":\"email\",\"providers\":[\"email\"]}', '{}', "
                     "now(), now(), '', '', '', '')", (uid, email, password))
        conn.execute("insert into auth.identities (user_id, provider_id, identity_data, provider, last_sign_in_at, created_at, updated_at) "
                     "values (%s, %s, %s, 'email', now(), now(), now())",
                     (uid, uid, json.dumps({"sub": uid, "email": email, "email_verified": True})))
    r = httpx.post(f"{os.environ['SUPABASE_URL'].rstrip('/')}/auth/v1/token?grant_type=password",
                   headers={"apikey": os.environ["SUPABASE_PUBLISHABLE_KEY"]}, json={"email": email, "password": password}, timeout=30)
    print(f"created {email} ({uid}); sign-in HTTP {r.status_code}")
    print(f"password (shown once): {password}")


if __name__ == "__main__":
    main()
