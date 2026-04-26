"use client";

import { useCallback, useRef, useState } from "react";
import { CloudUpload, CheckCircle2, AlertCircle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { uploadDocument } from "@/lib/api";

interface UploadPanelProps {
  sessionId: string;
  onUploaded: () => void;
}

type Status = "idle" | "uploading" | "success" | "error";

export function UploadPanel({ sessionId, onUploaded }: UploadPanelProps) {
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState<string>("");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFile(file: File) {
    setStatus("uploading");
    setMessage("");
    try {
      const result = await uploadDocument(sessionId, file);
      setStatus("success");
      setMessage(`"${result.filename}" indexed — ${result.chunk_count} chunks`);
      onUploaded();
      setTimeout(() => setStatus("idle"), 4000);
    } catch (e) {
      setStatus("error");
      setMessage(e instanceof Error ? e.message : "Upload failed");
    }
  }

  function handleInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => setDragging(false), []);

  return (
    <div className="flex flex-col gap-3">
      <h2 className="px-1 text-xs font-semibold uppercase tracking-widest text-[#64748b]">
        Upload Document
      </h2>

      {/* Drop zone */}
      <div
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`flex cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed px-4 py-6 text-center transition-colors
          ${dragging
            ? "border-[#06b6d4] bg-[#06b6d4]/5"
            : "border-[#2a2d3a] hover:border-[#06b6d4]/50 hover:bg-[#1a1d27]"
          }`}
      >
        {status === "uploading" ? (
          <Loader2 className="h-6 w-6 animate-spin text-[#06b6d4]" />
        ) : (
          <CloudUpload
            className={`h-6 w-6 ${dragging ? "text-[#06b6d4]" : "text-[#64748b]"}`}
          />
        )}
        <p className="text-xs text-[#64748b]">
          {status === "uploading"
            ? "Uploading…"
            : "Drag & drop or click to browse"}
        </p>
        <p className="text-[10px] text-[#64748b]/60">.pdf · .docx · .txt</p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.docx,.txt"
        className="hidden"
        onChange={handleInputChange}
      />

      {/* Status message */}
      {status === "success" && (
        <div className="flex items-center gap-2 rounded-md bg-[#10b981]/10 px-3 py-2 text-xs text-[#10b981]">
          <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
          {message}
        </div>
      )}
      {status === "error" && (
        <div className="flex items-center gap-2 rounded-md bg-[#f43f5e]/10 px-3 py-2 text-xs text-[#f43f5e]">
          <AlertCircle className="h-3.5 w-3.5 shrink-0" />
          {message}
        </div>
      )}

      {/* Index button (alternative for accessibility) */}
      <Button
        onClick={() => inputRef.current?.click()}
        disabled={status === "uploading"}
        className="w-full bg-[#06b6d4] text-[#0f1117] text-xs font-semibold hover:bg-[#0e7490] disabled:opacity-50"
        size="sm"
      >
        {status === "uploading" ? (
          <>
            <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
            Indexing…
          </>
        ) : (
          "Select & Index"
        )}
      </Button>
    </div>
  );
}
