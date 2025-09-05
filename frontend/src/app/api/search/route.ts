import { NextRequest } from "next/server";

export async function GET(req: NextRequest) {
  const q = req.nextUrl.searchParams.get("q") ?? "";
  const topk = req.nextUrl.searchParams.get("topk") ?? "10";

  if (q.length < 1) {
    return Response.json({ error: "q is required" }, { status: 400 });
  }

  const upstream = await fetch(
    `${process.env.API_BASE_INTERNAL}/search/?q=${encodeURIComponent(q)}&topk=${encodeURIComponent(topk)}`
  );

  const text = await upstream.text();
  return new Response(text, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("Content-Type") ?? "application/json" },
  });
}
