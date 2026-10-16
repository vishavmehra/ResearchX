import { notFound } from "next/navigation";
import { Chat } from "@/components/research";

export const metadata = { title: "Research" };

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export default async function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  if (!UUID.test(id)) notFound();
  return <Chat key={id} conversationId={id} />;
}
