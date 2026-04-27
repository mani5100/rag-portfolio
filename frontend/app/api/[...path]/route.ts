import type { NextRequest } from "next/server";

// Never cache — these are dynamic API calls (including SSE streams).
export const dynamic = "force-dynamic";

const BACKEND = process.env.BACKEND_URL || "http://localhost:8000";

async function proxy(req: NextRequest, ctx: RouteContext<"/api/[...path]">) {
  const { path } = await ctx.params;
  const url = new URL(req.url);
  const backendUrl = `${BACKEND}/${path.join("/")}${url.search}`;

  const headers = new Headers(req.headers);
  headers.delete("host");
  headers.delete("connection");
  headers.delete("upgrade");

  const upstream = await fetch(backendUrl, {
    method: req.method,
    headers,
    body: req.body,
    // @ts-expect-error: duplex is required when forwarding a streaming request body
    duplex: "half",
  });

  // Pass upstream.body (ReadableStream) straight through — no buffering.
  return new Response(upstream.body, {
    status: upstream.status,
    headers: upstream.headers,
  });
}

export { proxy as GET, proxy as POST, proxy as DELETE };
