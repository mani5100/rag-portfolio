import type { NextConfig } from "next";

// API requests are proxied via app/api/[...path]/route.ts to preserve
// streaming (SSE) — rewrites would buffer the response before forwarding.
const nextConfig: NextConfig = {};

export default nextConfig;
