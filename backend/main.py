from fastapi import FastAPI
from fastapi import Query
from typing import Optional, List
from search import search_bm25
from hybrid_search import search_hybrid
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
        include_path=True,
        word_ngrams=[1],
        char_ngrams=[],
    )
    return {"results": results}

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
        )
        return {"results": results}
