import { createBrowserClient } from "@supabase/ssr";

// Browser client: publishable key only (safe to expose). Sessions live in cookies managed by @supabase/ssr, so the
// server-side proxy can read and refresh them; nothing is put in localStorage by this app.
export function supabase() {
  return createBrowserClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY!);
}
