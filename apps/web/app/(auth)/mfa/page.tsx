import { MfaChallenge } from "@/components/auth";
import { safeNext } from "@/lib/supabase/server";

export const metadata = { title: "Two-factor authentication" };

export default async function Page({ searchParams }: { searchParams: Promise<{ next?: string }> }) {
  const sp = await searchParams;
  return <MfaChallenge next={safeNext(sp.next)} />;
}
