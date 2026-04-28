"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Bot, User, Send, Loader2, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { streamQuery } from "@/lib/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
}

export function ChatPanel({ sessionId }: { sessionId: string }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when new content arrives
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 96)}px`;
  }, [input]);

  const sendMessage = useCallback(async () => {
    const query = input.trim();
    if (!query || isStreaming) return;

    setInput("");
    setIsStreaming(true);

    const userMsg: Message = {
      id: `u-${Date.now()}`,
      role: "user",
      content: query,
    };

    const assistantId = `a-${Date.now()}`;
    const assistantMsg: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      streaming: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    await streamQuery(
      sessionId,
      query,
      (token) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: m.content + token } : m
          )
        );
      },
      () => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, streaming: false } : m
          )
        );
        setIsStreaming(false);
      },
      (err) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: `Error: ${err.message}`, streaming: false }
              : m
          )
        );
        setIsStreaming(false);
      }
    );
  }, [input, isStreaming]);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  const inputBar = (
    <div className="w-full px-4 py-3">
      <div className="mx-auto flex max-w-2xl items-end gap-2 rounded-xl border border-[#2a2d3a] bg-[#1a1d27] px-3 py-2 focus-within:border-[#06b6d4]/50 transition-colors">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your documents…"
          rows={1}
          className={`flex-1 resize-none bg-transparent text-sm text-[#f1f5f9] placeholder-[#64748b] outline-none transition-all ${
            input ? "text-left" : "text-center"
          }`}
          disabled={isStreaming}
        />
        <Button
          onClick={sendMessage}
          disabled={!input.trim() || isStreaming}
          size="icon"
          className="h-10 w-10 sm:h-8 sm:w-8 shrink-0 rounded-lg bg-[#06b6d4] text-[#0f1117] hover:bg-[#0e7490] disabled:opacity-40"
        >
          {isStreaming ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </Button>
      </div>
      <p className="hidden sm:block mt-1.5 text-center text-[10px] text-[#64748b]">
        Press Enter to send · Shift+Enter for new line
      </p>
    </div>
  );

  if (messages.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-6">
        <WelcomeScreen />
        {inputBar}
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      {/* Message area */}
      <ScrollArea className="flex-1 px-4 py-4">
        <div className="mx-auto flex max-w-2xl flex-col gap-4">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
        </div>
        <div ref={bottomRef} />
      </ScrollArea>

      {/* Input bar */}
      <div className="border-t border-[#2a2d3a] bg-[#0f1117]">
        {inputBar}
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="flex max-w-[80%] items-start gap-2">
          <div className="rounded-2xl rounded-tr-sm bg-[#06b6d4] px-4 py-2.5 text-sm text-[#0f1117]">
            {message.content}
          </div>
          <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[#2a2d3a]">
            <User className="h-3.5 w-3.5 text-[#f1f5f9]" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="flex max-w-[80%] items-start gap-2">
        <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[#06b6d4]/15 border border-[#06b6d4]/30">
          <Bot className="h-3.5 w-3.5 text-[#06b6d4]" />
        </div>
        <div
          className={`rounded-2xl rounded-tl-sm border border-[#2a2d3a] bg-[#1a1d27] px-4 py-2.5 text-sm text-[#f1f5f9] ${
            message.streaming ? "typing-cursor" : ""
          }`}
        >
          {message.content || (
            <span className="text-[#64748b] italic">Thinking…</span>
          )}
        </div>
      </div>
    </div>
  );
}

function WelcomeScreen() {
  return (
    <div className="flex flex-col items-center gap-4 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-[#06b6d4]/10 border border-[#06b6d4]/20">
        <Sparkles className="h-7 w-7 text-[#06b6d4]" />
      </div>
      <div className="flex flex-col gap-1">
        <h2 className="text-lg font-semibold text-[#f1f5f9]">
          Ask your documents
        </h2>
        <p className="max-w-xs text-sm text-[#64748b]">
          Upload documents on the left, then ask questions to get AI-powered
          answers grounded in your content.
        </p>
      </div>
      <div className="flex flex-wrap justify-center gap-2">
        {[
          "Summarize the main topics",
          "What are the key findings?",
          "Explain the methodology",
        ].map((q) => (
          <span
            key={q}
            className="rounded-full border border-[#2a2d3a] bg-[#1a1d27] px-3 py-1 text-xs text-[#64748b]"
          >
            {q}
          </span>
        ))}
      </div>
    </div>
  );
}
