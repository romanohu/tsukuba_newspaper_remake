from fastapi import FastAPI
from fastapi import Query
from typing import Optional
from search import search_bm25
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
        word_ngrams=[1,2,3],
        char_ngrams=[2,3],
    )
    return {"results": results}
