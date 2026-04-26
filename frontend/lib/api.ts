const BASE = "/api";

export interface Doc {
  filename: string;
  chunk_count: number;
  indexed_at?: string;
}

export interface UploadResult {
  status: string;
  filename: string;
  chunk_count: number;
}

export async function uploadDocument(sessionId: string, file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/upload`, {
    method: "POST",
    headers: { "X-Session-ID": sessionId },
    body: form,
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({ message: res.statusText }));
    throw new Error(data.message || `Upload failed: ${res.status}`);
  }
  return res.json();
}

export async function listDocuments(sessionId: string): Promise<Doc[]> {
  const res = await fetch(`${BASE}/documents`, {
    headers: { "X-Session-ID": sessionId },
  });
  if (!res.ok) throw new Error(`Failed to list documents: ${res.status}`);
  return res.json();
}

export async function deleteDocument(sessionId: string, filename: string): Promise<void> {
  const res = await fetch(`${BASE}/documents/${encodeURIComponent(filename)}`, {
    method: "DELETE",
    headers: { "X-Session-ID": sessionId },
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({ message: res.statusText }));
    throw new Error(data.message || `Delete failed: ${res.status}`);
  }
}

/**
 * Stream tokens from /query?q=... via SSE.
 * Calls onToken for each token, onDone when stream ends, onError on failure.
 */
export async function streamQuery(
  sessionId: string,
  query: string,
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (err: Error) => void
): Promise<void> {
  try {
    const res = await fetch(`${BASE}/query?q=${encodeURIComponent(query)}`, {
      headers: { "X-Session-ID": sessionId },
    });
    if (!res.ok || !res.body) {
      const data = await res.json().catch(() => ({ message: res.statusText }));
      throw new Error(data.message || `Query failed: ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const token = line.slice(6);
          if (token === "[DONE]") {
            onDone();
            return;
          }
          onToken(token);
        }
      }
    }

    onDone();
  } catch (err) {
    onError(err instanceof Error ? err : new Error(String(err)));
  }
}
