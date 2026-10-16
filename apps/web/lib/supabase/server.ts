import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";

export async function serverSupabase() {
  const store = await cookies();
  return createServerClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY!, {
    cookies: {
      getAll: () => store.getAll(),
      setAll: (list) => {
        try {
          list.forEach(({ name, value, options }) => store.set(name, value, options));
        } catch {
          // called from a Server Component: the proxy refreshes the session cookies instead
        }
      },
    },
  });
}

/**
 * Sign-in providers switched on in Supabase Auth (Authentication → Sign In / Providers), read from the public
 * /auth/v1/settings endpoint, so a provider button appears only once the provider works. Empty when Auth is unreachable:
 * email sign-in still renders.
 */
export async function authProviders(): Promise<string[]> {
  try {
    const r = await fetch(`${process.env.NEXT_PUBLIC_SUPABASE_URL}/auth/v1/settings`, {
      headers: { apikey: process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY! },
      next: { revalidate: 60 },
      signal: AbortSignal.timeout(3000),
    });
    const external: Record<string, boolean> = r.ok ? ((await r.json()).external ?? {}) : {};
    return Object.keys(external).filter((p) => external[p] === true);
  } catch {
    return [];
  }
}

/** Only same-site relative paths are accepted as post-login destinations (no open redirects). */
export function safeNext(next: string | null | undefined, fallback = "/"): string {
  return next && next.startsWith("/") && !next.startsWith("//") && !next.startsWith("/\\") ? next : fallback;
}
