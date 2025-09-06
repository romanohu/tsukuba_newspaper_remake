"use client";
import { useState } from "react";

type Mode = "bm25" | "hybrid";

export default function SearchBox() {
  const [q, setQ] = useState("");
  const [mode, setMode] = useState<Mode>("bm25");
  const [topk, setTopk] = useState(10);
  const [results, setResults] = useState<any[]>([]);
  const API_BASE = "http://localhost:8000";

  const handleSearch = async () => {
    const endpoint =
      mode === "bm25" ? "/search/" : "/hybrid_search/";
    const url = new URL(API_BASE + endpoint);
    url.searchParams.set("q", q);
    url.searchParams.set("topk", String(topk));

    if (mode === "hybrid") {
      url.searchParams.set("fusion", "rrf");      
      url.searchParams.set("bm25_k", "50");
      url.searchParams.set("vec_k", "200");
      url.searchParams.set("w_bm25", "0.6");
      url.searchParams.set("w_vec", "0.4");
    }

    const res = await fetch(url.toString());
    const json = await res.json();
    setResults(json.results ?? []);
  };

  return (
    <div style={{ display: "grid", gap: 8 }}>
      <div>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="検索語"
        />
        <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
          <option value="bm25">BM25</option>
          <option value="hybrid">Hybrid (BM25+Embedding)</option>
        </select>
        <input
          type="number"
          value={topk}
          min={1}
          max={100}
          onChange={(e) => setTopk(Number(e.target.value))}
          style={{ width: 80, marginLeft: 8 }}
        />
        <button onClick={handleSearch} style={{ marginLeft: 8 }}>
          検索
        </button>
      </div>

      {results && (
        <ul>
          {results.map((it, i) => (
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
          {results.length === 0 && <li>結果はありませんでした。</li>}
        </ul>
      )}
    </div>
  );
}
