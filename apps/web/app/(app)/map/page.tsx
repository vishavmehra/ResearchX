import { Suspense } from "react";
import { KnowledgeMap } from "@/components/knowledge-map";

export const metadata = { title: "Knowledge Map" };

export default function Page() {
  return (
    <Suspense fallback={null}>
      <KnowledgeMap />
    </Suspense>
  );
}
