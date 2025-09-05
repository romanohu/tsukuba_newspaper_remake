"use client";
import { it } from "node:test";
import { useState, useTransition } from "react";

type SearchItem = { title?: string; path?: string; score?: number; hit_terms?: string[]};
type SearchResponse = { results: SearchItem[] };

export default function SearchPage() {
  const [q, setQ] = useState("");
  const [topk, setTopk] = useState(10);
  const [data, setData] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null); setData(null);

    startTransition(async () => {
      try {
        const r = await fetch(`/api/search?q=${encodeURIComponent(q)}&topk=${topk}`);
        if (!r.ok) throw new Error(await r.text());
        setData((await r.json()) as SearchResponse);
      } catch (err: any) {
        setError(err.message ?? "検索に失敗しました");
      }
    });
  };

  

  return (
    <main>
      <h1>新聞検索</h1>
      <form onSubmit={onSubmit}>
        <input
               value={q} onChange={(e) => setQ(e.target.value)}
               placeholder="キーワード" />
        <input
               type="number" min={1} max={100}
               value={topk} onChange={(e) => setTopk(Number(e.target.value))}/>
        <button disabled={isPending}>
          {isPending ? "検索中…" : "検索"}
        </button>
      </form>

      {error && <p>Error: {error}</p>}
      {data && (
        <ul>
          {data.results.map((it, i) => (
            <li key={i}>
              {it.title && (() => {
                    const [num, page] = it.title.split("_");
                    const numInt = parseInt(num, 10);
                    const label = numInt >= 360 ? `${numInt}号p${page}` : `${numInt}号p${page}`;
                    const href = numInt >= 360
                        ? `https://www.tsukuba.ac.jp/about/public-newspaper/pdf/${numInt}.pdf`
                        : `https://www.tsukuba.ac.jp/about/public-newspaper/${numInt}.pdf`;
                    return (
                        <a href={href} target="_blank" rel="noopener noreferrer">
                        {label}
                        </a>
                    );
                    })()}
              {typeof it.score === "number" && (
                <div>score: {it.score.toFixed(4)}</div>
              )}
              <div>{it.hit_terms?.join(", ")}</div>
            </li>
          ))}
          {data.results.length === 0 && <li>結果はありませんでした。</li>}
        </ul>
      )}
    </main>
  );
}
