from fastapi import FastAPI
from fastapi import Query
from search.search import search_bm25
from search.hybrid_search import search_hybrid, search_vec, _fetch_docs_meta
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/search/")
def search(q: str = Query(..., min_length=1), topk: int = Query(10, ge=1, le=100)):
    results = search_bm25(
        db_path="../data/bm25_index.sqlite",
        query=q,
        topk=topk,
        include_path=False,
        return_terms=True,
        word_ngrams=[1, 2, 3],
        char_ngrams=[3, 4],
    )
    return {"results": results}


@app.get("/vec_search/")
def vec_search(
    q: str = Query(..., min_length=1),
    topk: int = Query(10, ge=1, le=100),
    faiss_index: str = Query("../data/faiss.index"),
    vec_meta: str = Query("../data/vec_meta.json"),
    model: str = Query("intfloat/multilingual-e5-small"),
):
    results = search_vec(
        faiss_index_path=faiss_index,
        meta_json_path=vec_meta,
        query=q,
        topk=topk,
        model_name=model,
    )
    import sqlite3

    conn = sqlite3.connect("../data/bm25_index.sqlite")
    cur = conn.cursor()
    meta_docs = {}
    doc_ids = [doc_id for doc_id, _ in results]
    for chunk in [doc_ids[i : i + 999] for i in range(0, len(doc_ids), 999)]:
        meta_docs.update(_fetch_docs_meta(cur, chunk))
    conn.close()
    out = []
    for doc_id, score in results:
        if doc_id in meta_docs:
            ext_id, title, path, length = meta_docs[doc_id]
            out.append(
                {
                    "doc_id": doc_id,
                    "score": score,
                    "title": title,
                    "path": path,
                    "ext_id": ext_id,
                    "length": length,
                }
            )
    return {"results": out}


@app.get("/hybrid_search/")
def hybrid_search(
    q: str = Query(..., min_length=1),
    topk: int = Query(10, ge=1, le=100),
    bm25_k: int = Query(50, ge=1, le=1000),
    vec_k: int = Query(200, ge=1, le=5000),
    fusion: str = Query("rrf", pattern="^(rrf|wsum)$"),
    w_bm25: float = Query(0.6, ge=0.0, le=1.0),
    w_vec: float = Query(0.4, ge=0.0, le=1.0),
    db_path: str = Query("../data/bm25_index.sqlite"),
    faiss_index: str = Query("../data/faiss.index"),
    vec_meta: str = Query("../data/vec_meta.json"),
    model: str = Query("intfloat/multilingual-e5-small"),
    include_path: bool = Query(True),
):
    results = search_hybrid(
        db_path=db_path,
        faiss_index_path=faiss_index,
        meta_json_path=vec_meta,
        query=q,
        topk=topk,
        bm25_k=bm25_k,
        vec_k=vec_k,
        fusion=fusion,
        w_bm25=w_bm25,
        w_vec=w_vec,
        include_path=include_path,
        model_name=model,
        word_ngrams=[1, 2, 3],
        char_ngrams=[3],
    )
    return {"results": results}
