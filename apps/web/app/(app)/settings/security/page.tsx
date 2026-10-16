import { SecurityPage } from "@/components/settings";
import { authProviders } from "@/lib/supabase/server";

export const metadata = { title: "Security" };

export default async function Page({ searchParams }: { searchParams: Promise<{ notice?: string; error?: string }> }) {
  const sp = await searchParams;
  return <SecurityPage providers={await authProviders()} notice={sp.error ?? sp.notice} />;
}
