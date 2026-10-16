import { ResetForm } from "@/components/auth";

export const metadata = { title: "Choose a new password" };

// Reached through /auth/callback with a recovery link, which signs the user in for this step (the proxy requires it).
export default function Page() {
  return <ResetForm />;
}
