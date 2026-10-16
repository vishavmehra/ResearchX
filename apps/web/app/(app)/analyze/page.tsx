import { Suspense } from "react";
import { Analyze } from "@/components/analyze";

export const metadata = { title: "Analyze" };

export default function Page() {
  return (
    <Suspense fallback={null}>
      <Analyze />
    </Suspense>
  );
}
