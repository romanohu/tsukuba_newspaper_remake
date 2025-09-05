from fastapi import FastAPI
from fastapi import Query
from typing import Optional
from search import search_bm25

app = FastAPI()

@app.get("/search/")
def search(q: str = Query(..., min_length=1), topk: int = Query(10, ge=1, le=100)):
    results = search_bm25(
        db_path="../data/bm25_index.sqlite",
        query=q,
        topk=topk,
        include_path=True,
    )
    return {"results": results}
