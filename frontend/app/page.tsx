"use client";

import { useState } from "react";
import { UploadPanel } from "@/components/UploadPanel";
import { DocumentList } from "@/components/DocumentList";
import { ChatPanel } from "@/components/ChatPanel";
import { Separator } from "@/components/ui/separator";
import { useSession } from "@/lib/useSession";

export default function Home() {
  const sessionId = useSession();
  const [refreshKey, setRefreshKey] = useState(0);

  function handleUploaded() {
    setRefreshKey((k) => k + 1);
  }

  if (!sessionId) return null;

  return (
    <div className="flex h-full w-full overflow-hidden">
      {/* Left sidebar */}
      <aside className="flex w-72 shrink-0 flex-col gap-5 border-r border-[#2a2d3a] bg-[#1a1d27] p-4 overflow-y-auto">
        <UploadPanel sessionId={sessionId} onUploaded={handleUploaded} />
        <Separator className="bg-[#2a2d3a]" />
        <DocumentList sessionId={sessionId} refreshKey={refreshKey} />
      </aside>

      {/* Right chat panel */}
      <div className="flex flex-1 flex-col overflow-hidden">
        <ChatPanel sessionId={sessionId} />
      </div>
    </div>
  );
}
