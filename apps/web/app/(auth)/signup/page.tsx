import { SignupForm } from "@/components/auth";
import { authProviders } from "@/lib/supabase/server";

export const metadata = { title: "Create account" };

export default async function Page() {
  return <SignupForm providers={await authProviders()} />;
}
