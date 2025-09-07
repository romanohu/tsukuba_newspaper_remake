# vec_search.py
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def _load_faiss(index_path: str):
    return faiss.read_index(index_path)


def _load_vec_meta(meta_path: str) -> List[dict]:
    return json.loads(Path(meta_path).read_text(encoding="utf-8"))


def _encode_query(model: SentenceTransformer, query: str) -> np.ndarray:
    return model.encode("query: " + query, normalize_embeddings=True).astype("float32")


def search_vec(
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
