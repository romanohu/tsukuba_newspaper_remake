# vec_search.py
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .bm25_search import fetch_docs_meta
import sqlite3


def _load_faiss(index_path: str):
    return faiss.read_index(index_path)


def _load_vec_meta(meta_path: str) -> List[dict]:
    return json.loads(Path(meta_path).read_text(encoding="utf-8"))


def _encode_query(model: SentenceTransformer, query: str) -> np.ndarray:
    return model.encode("query: " + query, normalize_embeddings=True).astype("float32")

def add_meta_info(doc_ids, db_path="../data/bm25_index.sqlite"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    meta_docs = fetch_docs_meta(cur, doc_ids)
    conn.close()
    return meta_docs


def search_vec_forhybrid(
    faiss_index_path: str,
    meta_json_path: str,
    query: str,
    model_name: str = "intfloat/multilingual-e5-small",
    topk: int = 200,
) -> List[Tuple[int, float]]:
    index = _load_faiss(faiss_index_path)
    meta = _load_vec_meta(meta_json_path)
    model = SentenceTransformer(model_name)
    q = _encode_query(model, query)
    D, I = index.search(np.expand_dims(q, 0), topk)
    hits: List[Tuple[int, float]] = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0:
            continue
        doc_id = int(meta[idx]["doc_id"])
        hits.append((doc_id, float(score)))
    return hits

def search_vec(
    faiss_index_path: str,
    meta_json_path: str,
    query: str,
    model_name: str = "intfloat/multilingual-e5-small",
    topk: int = 200,
) -> List[dict]:
    index = _load_faiss(faiss_index_path)
    meta = _load_vec_meta(meta_json_path)
    model = SentenceTransformer(model_name)
    q = _encode_query(model, query)
    D, I = index.search(np.expand_dims(q, 0), topk)
    hits = []
    doc_ids = []
    scores = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0:
            continue
        doc_id = int(meta[idx]["doc_id"])
        doc_ids.append(doc_id)
        scores.append(float(score))
    meta_docs = add_meta_info(doc_ids)
    for doc_id, score in zip(doc_ids, scores):
        if doc_id in meta_docs:
            ext_id, title, path, length = meta_docs[doc_id]
            hits.append({
                "doc_id": doc_id,
                "score": score,
                "title": title,
                "path": path,
                "ext_id": ext_id,
                "length": length,
            })
    return hits
