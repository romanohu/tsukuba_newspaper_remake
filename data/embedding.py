import argparse, json, sqlite3, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss


def read_text_file(path: str) -> str:
    p = Path(path)
    for enc in ("utf-8", "cp932"):
        try: return p.read_text(encoding=enc, errors="ignore")
        except Exception: pass
    return p.read_bytes().decode("utf-8", errors="ignore")


def chunk_text(s: str, max_chars=1000, stride=800):
    s = s.strip()
    if not s: return []
    chunks = []
    i = 0
    n = len(s)
    while i < n:
        j = min(i + max_chars, n)
        chunks.append(s[i:j])
        if j == n: break
        i += stride
    return chunks


def load_model(name: str = "intfloat/multilingual-e5-small") -> SentenceTransformer:
    return SentenceTransformer(name)

def main():
    ap = argparse.ArgumentParser(description="Build FAISS embeddings from SQLite docs")
    ap.add_argument("db")
    ap.add_argument("--out-index", default="faiss.index")
    ap.add_argument("--out-meta", default="vec_meta.json")
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--stride", type=int, default=800)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    rows = cur.execute("SELECT id, ext_id, path FROM docs").fetchall()

    model = load_model(args.model)


    def to_corpus_input(text): return "passage: " + text

    vecs = []
    meta = [] 
    for doc_id, ext_id, path in rows:
        text = read_text_file(path)
        chunks = chunk_text(text, max_chars=args.max_chars, stride=args.stride)
        for k, chunk in enumerate(chunks):
            meta.append({"doc_id": int(doc_id), "ext_id": ext_id, "chunk_id": k})
            vecs.append(model.encode(to_corpus_input(chunk), normalize_embeddings=True))

    if not vecs:
        print("No text to index.")
        return

    X = np.vstack(vecs).astype("float32") 
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)         
    index.add(X)

    faiss.write_index(index, args.out_index)
    Path(args.out_meta).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] vectors={len(meta)}  dim={dim}")
    print(f"Saved -> {args.out_index}, {args.out_meta}")

if __name__ == "__main__":
    main()