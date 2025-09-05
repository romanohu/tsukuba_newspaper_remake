from janome.tokenizer import Tokenizer
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import re
import sqlite3
import unicodedata

# 形態素/前処理の設定
_t = Tokenizer(wakati=False)
_USE_POS = {"名詞", "動詞", "形容詞"}
_STOPWORDS = {"する","ある","いる","なる","できる","れる","こと","これ","それ","あれ","ため","よう","また","そして","ので","から","に","へ","で","を","が","は","です","ます","だ","な","の","と","や","も","ね","よ","その","この","あの"}

# テキスト正規化（空白圧縮・Unicode記号/制御除去）
def normalize(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text).strip()
    out = []
    for ch in text:
        c = unicodedata.category(ch)
        if c.startswith("S") or c.startswith("C") or c in ("Zl","Zp"):
            continue
        out.append(ch)
    return "".join(out)

# 単語トークン化（品詞/ストップワード適用）
def tokenize_words(text: str):
    text = normalize(text)
    for tok in _t.tokenize(text):
        base = tok.base_form if tok.base_form != "*" else tok.surface
        pos = tok.part_of_speech.split(",")[0]
        if pos in _USE_POS and base and base not in _STOPWORDS:
            yield base

# 単語n-gram生成
def word_ngrams(tokens, ns):
    toks = list(tokens)
    for n in ns:
        if n <= 0: continue
        if n == 1:
            for t in toks: yield t
        else:
            for i in range(0, max(0, len(toks)-n+1)):
                yield "␟".join(toks[i:i+n])

# 文字n-gram生成
def char_ngrams(text: str, ns):
    s = re.sub(r"\s+", "", normalize(text))
    for n in ns:
        if n <= 0: continue
        for i in range(0, max(0, len(s)-n+1)):
            yield s[i:i+n]

# テキストファイル読込（UTF-8→CP932フォールバック）
def read_text_file(p: Path) -> str:
    for enc in ("utf-8","cp932"):
        try:
            return p.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    return p.read_bytes().decode("utf-8", errors="ignore")

# ルート群からファイル一覧を収集
def collect_files(roots: list[Path], pattern: str) -> list[Path]:
    files = []
    for r in roots:
        files += list(r.rglob(pattern))
    files = sorted(set(files), key=lambda p: (str(p.parent), p.name))
    return files

# 号数と外部ID（issue_stem）を生成
def make_issue_and_extid(path: Path, roots: list[Path]) -> tuple[str|None, str]:
    issue = None
    rel = None
    for root in roots:
        try:
            rel = path.relative_to(root)
            break
        except ValueError:
            continue
    if rel is not None:
        parts = rel.parts
        issue = parts[0] if len(parts) >= 2 else None
    stem = path.stem
    ext_id = f"{issue}_{stem}" if issue else stem
    return issue, ext_id

# SQLiteスキーマ作成
def init_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript("""
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS params(
      key TEXT PRIMARY KEY,
      value TEXT
    );
    CREATE TABLE IF NOT EXISTS docs(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ext_id TEXT UNIQUE,
      title TEXT,
      issue TEXT,
      path TEXT,
      length INTEGER
    );
    CREATE TABLE IF NOT EXISTS terms(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      term TEXT UNIQUE,
      df INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS postings(
      term_id INTEGER,
      doc_id INTEGER,
      tf INTEGER,
      PRIMARY KEY(term_id, doc_id),
      FOREIGN KEY(term_id) REFERENCES terms(id),
      FOREIGN KEY(doc_id) REFERENCES docs(id)
    );
    CREATE INDEX IF NOT EXISTS idx_terms_term ON terms(term);
    CREATE INDEX IF NOT EXISTS idx_posts_term ON postings(term_id);
    CREATE INDEX IF NOT EXISTS idx_posts_doc  ON postings(doc_id);
    """)
    conn.commit()

# termをIDに解決（新規は挿入）
def get_term_id(cur: sqlite3.Cursor, cache: dict, term: str) -> int:
    tid = cache.get(term)
    if tid: return tid
    cur.execute("INSERT OR IGNORE INTO terms(term, df) VALUES(?, 0)", (term,))
    cur.execute("SELECT id FROM terms WHERE term = ?", (term,))
    tid = cur.fetchone()[0]
    cache[term] = tid
    return tid

# 文書1件の登録（docs/postings/df計測）
def insert_document(cur: sqlite3.Cursor, roots: list[Path], path: Path, word_ngs, char_ngs,
                    term_cache: dict, df_counter: defaultdict[int, int]) -> tuple[int,int,str]:
    text = read_text_file(path)
    if not text.strip():
        return (0, 0, "")

    issue, ext_id = make_issue_and_extid(path, roots)
    title = ext_id

    cur.execute("SELECT id FROM docs WHERE ext_id = ?", (ext_id,))
    row = cur.fetchone()

    if row:
        doc_id = int(row[0])
        cur.execute("DELETE FROM postings WHERE doc_id = ?", (doc_id,))
        cur.execute("UPDATE docs SET path=?, issue=?, title=? WHERE id=?",
                    (str(path), issue, title, doc_id))
    else:
        cur.execute(
            "INSERT INTO docs(ext_id, title, issue, path, length) VALUES(?,?,?,?,0)",
            (ext_id, title, issue, str(path))
        )
        doc_id = cur.lastrowid

    tokens = []
    if word_ngs:
        base = list(tokenize_words(text))
        tokens.extend(word_ngrams(base, word_ngs))
    if char_ngs:
        tokens.extend(char_ngrams(text, char_ngs))
    tokens = list(tokens)
    length = len(tokens)

    if length > 0:
        tf = Counter(tokens)
        seen = set()
        for term, f in tf.items():
            tid = get_term_id(cur, term_cache, term)
            cur.execute(
                "INSERT OR REPLACE INTO postings(term_id, doc_id, tf) VALUES(?,?,?)",
                (tid, doc_id, int(f))
            )
            if tid not in seen:
                df_counter[tid] += 1
                seen.add(tid)

    cur.execute("UPDATE docs SET length=? WHERE id=?", (length, doc_id))

    return (doc_id, length, ext_id)


def finalize_stats(conn: sqlite3.Connection, df_counter: defaultdict[int,int], k1: float, b: float,
                   settings: dict):
    cur = conn.cursor()
    rows = [(int(df), int(tid)) for tid, df in df_counter.items()]
    if rows:
        cur.executemany("UPDATE terms SET df = COALESCE(df,0) + ? WHERE id = ?", rows)
    cur.execute("SELECT COUNT(*), COALESCE(AVG(length),0) FROM docs")
    N, avgdl = cur.fetchone()
    cur.execute("INSERT OR REPLACE INTO params(key, value) VALUES(?,?)", ("k1", str(k1)))
    cur.execute("INSERT OR REPLACE INTO params(key, value) VALUES(?,?)", ("b", str(b)))
    cur.execute("INSERT OR REPLACE INTO params(key, value) VALUES(?,?)", ("N", str(int(N))))
    cur.execute("INSERT OR REPLACE INTO params(key, value) VALUES(?,?)", ("avgdl", str(float(avgdl))))
    cur.execute("INSERT OR REPLACE INTO params(key, value) VALUES(?,?)", ("settings", json.dumps(settings, ensure_ascii=False)))
    conn.commit()

def parse_int_list(s: str | None):
    if not s: return []
    return [int(x) for x in re.split(r"[,\s]+", s.strip()) if x]

def main():
    ap = argparse.ArgumentParser(description="Build BM25 inverted index into SQLite (Unicode-clean, n-grams).")
    ap.add_argument("roots", nargs="+", help="One or more root directories (e.g., txts)")
    ap.add_argument("--pattern", default="*.txt")
    ap.add_argument("--db", default="bm25_index.sqlite")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--word-ngrams", default="1")
    ap.add_argument("--char-ngrams", default="")
    args = ap.parse_args()

    roots = [Path(r).resolve() for r in args.roots]
    files = collect_files(roots, args.pattern)
    word_ngs = parse_int_list(args.word_ngrams)
    char_ngs = parse_int_list(args.char_ngrams)

    settings = {
        "tokenize": {"use_pos": sorted(_USE_POS), "stopwords": sorted(_STOPWORDS)},
        "ngrams": {"word": word_ngs, "char": char_ngs},
        "pattern": args.pattern,
        "roots": [str(r) for r in roots],
    }

    conn = sqlite3.connect(args.db)
    init_db(conn)

    cur = conn.cursor()
    term_cache: dict[str, int] = {}
    df_counter: defaultdict[int,int] = defaultdict(int)

    inserted = 0
    total_len = 0

    cur.execute("PRAGMA synchronous=OFF")
    cur.execute("BEGIN")
    try:
        for p in files:
            doc_id, length, ext_id = insert_document(cur, roots, p, word_ngs, char_ngs, term_cache, df_counter)
            if doc_id:
                inserted += 1
                total_len += length
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    finalize_stats(conn, df_counter, args.k1, args.b, settings)

    avgdl = (total_len / inserted) if inserted else 0.0
    print(f"[OK] Indexed docs: {inserted}, avgdl: {avgdl:.2f}")
    print(f"Word n-grams: {word_ngs or 'disabled'}, Char n-grams: {char_ngs or 'disabled'}")
    print(f"SQLite DB -> {args.db}")

if __name__ == "__main__":
    main()
