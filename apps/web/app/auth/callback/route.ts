import type { EmailOtpType } from "@supabase/supabase-js";
import { NextResponse } from "next/server";
import { safeNext, serverSupabase } from "@/lib/supabase/server";

// Provider-side failures worth their own message; anything else is a generic "did not complete".
const OAUTH_ERRORS: Record<string, string> = {
  access_denied: "oauth_cancelled",
  provider_email_needs_verification: "oauth_unverified",
  identity_already_exists: "identity_exists",
  manual_linking_disabled: "linking_disabled",
};

// Landing point of Supabase redirects: email links (sign-up confirmation, password recovery) and OAuth: a provider
// sign-in (flow=oauth), connecting a provider in Settings (flow=link) or signing in again before deleting an account
// that has no password (flow=reauth). PKCE redirects carry ?code=, token-hash email templates carry ?token_hash=&type=,
// and a failed attempt carries ?error=&error_code=. Only same-site paths are accepted as the destination.
export async function GET(request: Request) {
  const url = new URL(request.url);
  const p = url.searchParams;
  const next = safeNext(p.get("next"));
  const flow = p.get("flow");
  const failed = (reason: string) =>
    NextResponse.redirect(
      new URL(
        flow === "link" || flow === "reauth" ? `/settings/security?error=${reason}` : `/login?error=${flow === "oauth" ? reason : "link"}`,
        url.origin,
      ),
    );

  const providerError = p.get("error_code") ?? p.get("error");
  if (providerError) return failed(OAUTH_ERRORS[providerError] ?? "oauth");
  const code = p.get("code");
  const tokenHash = p.get("token_hash");
  const type = p.get("type") as EmailOtpType | null;
  const supabase = await serverSupabase();
  const { error } = code
    ? await supabase.auth.exchangeCodeForSession(code)
    : tokenHash && type
      ? await supabase.auth.verifyOtp({ token_hash: tokenHash, type })
      : { error: new Error("missing code") };
  return error ? failed("oauth") : NextResponse.redirect(new URL(next, url.origin));
}
