import { LoginForm } from "@/components/auth";
import { authProviders, safeNext } from "@/lib/supabase/server";

export const metadata = { title: "Sign in" };

export default async function Page({ searchParams }: { searchParams: Promise<{ next?: string; notice?: string; error?: string }> }) {
  const sp = await searchParams;
  return <LoginForm next={safeNext(sp.next)} notice={sp.notice ?? sp.error} providers={await authProviders()} />;
}
