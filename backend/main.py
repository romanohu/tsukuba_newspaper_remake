from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# 検索モジュール
from search.bm25_search import search_bm25
from search.hybrid_search import search_hybrid
from search.vec_search import search_vec
from search.bm25_search import fetch_docs_meta

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.get("/bm25_search/")
def bm25_search(
    q: str = Query(..., min_length=1),
    topk: int = Query(10, ge=1, le=100),
):
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
    faiss_index: str = "../data/faiss.index",
    vec_meta: str = "../data/vec_meta.json",
):
    hits = search_vec(
        faiss_index_path=faiss_index,
        meta_json_path=vec_meta,
        query=q,
        topk=topk,
    )
    return {"results": hits}


@app.get("/hybrid_search/")
def hybrid_search_endpoint(
    q: str = Query(..., min_length=1),
    topk: int = 10,
    bm25_k: int = 50,
    vec_k: int = 200,
    fusion: str = Query("rrf", pattern="^(rrf|wsum)$"),
    w_bm25: float = 0.6,
    w_vec: float = 0.4,
    include_path: bool = True,
):
    results = search_hybrid(
        db_path="../data/bm25_index.sqlite",
        faiss_index_path="../data/faiss.index",
        meta_json_path="../data/vec_meta.json",
        query=q,
        topk=topk,
        bm25_k=bm25_k,
        vec_k=vec_k,
        fusion=fusion,
        w_bm25=w_bm25,
        w_vec=w_vec,
        include_path=include_path,
    )
    return {"results": results}
