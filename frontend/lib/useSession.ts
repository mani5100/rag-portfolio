"use client";

import { useEffect, useState } from "react";

export function useSession(): string {
  const [sessionId, setSessionId] = useState("");

  useEffect(() => {
    let id = sessionStorage.getItem("rag-session-id");
    if (!id) {
      id = crypto.randomUUID();
      sessionStorage.setItem("rag-session-id", id);
    }
    setSessionId(id);
  }, []);

  return sessionId;
}
