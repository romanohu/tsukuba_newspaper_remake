# hybrid_search.py
import sqlite3
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from .bm25_search import search_bm25, fetch_docs_meta
from .vec_search import search_vec


def rrf_fuse(
    bm25_scores: Dict[int, float], vec_scores: Dict[int, float], k: int = 60
) -> Dict[int, float]:
    b_rank = {
        d: r
        for r, (d, _) in enumerate(
            sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True), start=1
        )
    }
    v_rank = {
        d: r
        for r, (d, _) in enumerate(
            sorted(vec_scores.items(), key=lambda x: x[1], reverse=True), start=1
        )
    }
    docs = set(b_rank) | set(v_rank)
    return {
        d: 1.0 / (k + b_rank.get(d, 10**9)) + 1.0 / (k + v_rank.get(d, 10**9))
        for d in docs
    }


def _minmax_norm(d: Dict[int, float]) -> Dict[int, float]:
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def weighted_sum_fuse(
    bm25_scores: Dict[int, float], vec_scores: Dict[int, float], w_bm25=0.6, w_vec=0.4
) -> Dict[int, float]:
    B = _minmax_norm(bm25_scores)
    V = _minmax_norm(vec_scores)
    docs = set(B) | set(V)
    return {d: w_bm25 * B.get(d, 0.0) + w_vec * V.get(d, 0.0) for d in docs}


def search_hybrid(
    db_path: str,
    faiss_index_path: str,
    meta_json_path: str,
    query: str,
    topk: int = 10,
    bm25_k: int = 50,
    vec_k: int = 200,
    fusion: str = "rrf",
    w_bm25: float = 0.6,
    w_vec: float = 0.4,
    include_path: bool = False,
) -> List[Dict]:
    bm25_rows = search_bm25(
        db_path=db_path,
        query=query,
        topk=bm25_k,
        include_path=False,
        return_terms=False,
    )
    bm25_scores: Dict[int, float] = {
        row["doc_id"]: float(row["score"]) for row in bm25_rows
    }

    vec_hits = search_vec(
        faiss_index_path=faiss_index_path,
        meta_json_path=meta_json_path,
        query=query,
        topk=vec_k,
    )
    vec_scores: Dict[int, float] = defaultdict(float)
    for doc_id, score in vec_hits:
        if score > vec_scores[doc_id]:
            vec_scores[doc_id] = score

    if fusion == "rrf":
        fused = rrf_fuse(bm25_scores, vec_scores)
    else:
        fused = weighted_sum_fuse(bm25_scores, vec_scores, w_bm25=w_bm25, w_vec=w_vec)

    if not fused:
        return []

    all_doc_ids = sorted({d for d in fused.keys()})
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    meta_docs = fetch_docs_meta(cur, all_doc_ids)
    conn.close()

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:topk]
    results = []
    for r, (doc_id, score) in enumerate(ranked, start=1):
        ext_id, title, path, dl = meta_docs.get(doc_id, ("?", "?", "?", 0))
        row = {
            "rank": r,
            "doc_id": doc_id,
            "ext_id": ext_id,
            "title": title,
            "score": float(score),
            "length": dl,
        }
        if include_path:
            row["path"] = path
        results.append(row)
    return results
