"use client";

import { useEffect, useState } from "react";
import { FileText, File, Trash2, Upload } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { listDocuments, deleteDocument, type Doc } from "@/lib/api";

interface DocumentListProps {
  sessionId: string;
  refreshKey: number;
}

function fileIcon(filename: string) {
  const ext = filename.split(".").pop()?.toLowerCase();
  if (ext === "pdf") return <File className="h-4 w-4 shrink-0 text-[#f43f5e]" />;
  if (ext === "docx" || ext === "doc")
    return <FileText className="h-4 w-4 shrink-0 text-[#06b6d4]" />;
  return <FileText className="h-4 w-4 shrink-0 text-[#64748b]" />;
}

export function DocumentList({ sessionId, refreshKey }: DocumentListProps) {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listDocuments(sessionId)
      .then(setDocs)
      .catch((e) => setError(e.message));
  }, [sessionId, refreshKey]);

  async function handleDelete(filename: string) {
    setDeletingId(filename);
    // Optimistic removal
    setDocs((prev) => prev.filter((d) => d.filename !== filename));
    try {
      await deleteDocument(sessionId, filename);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed");
      // Re-fetch to restore if delete failed
      listDocuments(sessionId).then(setDocs).catch(() => null);
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <h2 className="px-1 text-xs font-semibold uppercase tracking-widest text-[#64748b]">
        Indexed Documents
      </h2>

      {error && (
        <p className="rounded-md bg-[#f43f5e]/10 px-3 py-2 text-xs text-[#f43f5e]">
          {error}
        </p>
      )}

      {docs.length === 0 ? (
        <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-[#2a2d3a] py-8 text-center">
          <Upload className="h-6 w-6 text-[#64748b]" />
          <p className="text-xs text-[#64748b]">No documents indexed</p>
        </div>
      ) : (
        <ScrollArea className="h-[280px]">
          <ul className="flex flex-col gap-1 pr-3">
            {docs.map((doc) => (
              <li
                key={doc.filename}
                className="group flex items-center gap-2 rounded-lg border border-[#2a2d3a] bg-[#1a1d27] px-3 py-2 transition-colors hover:border-[#3a3d4a]"
              >
                {fileIcon(doc.filename)}
                <span
                  className="min-w-0 flex-1 truncate text-xs text-[#f1f5f9]"
                  title={doc.filename}
                >
                  {doc.filename}
                </span>
                <Badge
                  variant="secondary"
                  className="shrink-0 bg-[#2a2d3a] text-[#64748b] text-[10px]"
                >
                  {doc.chunk_count}
                </Badge>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6 shrink-0 opacity-0 transition-opacity group-hover:opacity-100 hover:bg-[#f43f5e]/10 hover:text-[#f43f5e]"
                  onClick={() => handleDelete(doc.filename)}
                  disabled={deletingId === doc.filename}
                  aria-label={`Delete ${doc.filename}`}
                >
                  <Trash2 className="h-3 w-3" />
                </Button>
              </li>
            ))}
          </ul>
        </ScrollArea>
      )}
    </div>
  );
}
