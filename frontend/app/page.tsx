"use client";

import { useState } from "react";
import { UploadPanel } from "@/components/UploadPanel";
import { DocumentList } from "@/components/DocumentList";
import { ChatPanel } from "@/components/ChatPanel";
import { Navbar } from "@/components/Navbar";
import { Separator } from "@/components/ui/separator";
import { useSession } from "@/lib/useSession";

export default function Home() {
  const sessionId = useSession();
  const [refreshKey, setRefreshKey] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  function handleUploaded() {
    setRefreshKey((k) => k + 1);
  }

  if (!sessionId) return null;

  return (
    <div className="flex h-full w-full flex-col">
      <Navbar
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen((v) => !v)}
      />

      <div className="relative flex flex-1 overflow-hidden">
        {/* Backdrop (mobile only) */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 z-30 bg-black/60 md:hidden"
            onClick={() => setSidebarOpen(false)}
            aria-hidden="true"
          />
        )}

        {/* Left sidebar */}
        <aside
          className={[
            "fixed left-0 top-14 bottom-0 z-40 flex w-72 shrink-0 flex-col gap-5",
            "border-r border-[#2a2d3a] bg-[#1a1d27] p-4 overflow-y-auto",
            "transition-transform duration-300 ease-in-out",
            "md:relative md:top-0 md:z-auto md:translate-x-0 md:transition-none",
            sidebarOpen ? "translate-x-0" : "-translate-x-full",
          ].join(" ")}
        >
          <UploadPanel sessionId={sessionId} onUploaded={handleUploaded} />
          <Separator className="bg-[#2a2d3a]" />
          <DocumentList sessionId={sessionId} refreshKey={refreshKey} />
        </aside>

        {/* Right chat panel */}
        <div className="flex flex-1 flex-col overflow-hidden">
          <ChatPanel sessionId={sessionId} />
        </div>
      </div>
    </div>
  );
}
