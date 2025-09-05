import json
import math
import re
import sqlite3
import unicodedata
from collections import defaultdict
from typing import Iterable, Dict, List, Tuple, Optional

from janome.tokenizer import Tokenizer

_t = Tokenizer(wakati=False)
_USE_POS = {"名詞", "動詞", "形容詞"}
_STOPWORDS = {
    "する","ある","いる","なる","できる","れる","こと","これ","それ","あれ",
    "ため","よう","また","そして","ので","から","に","へ","で","を","が","は",
    "です","ます","だ","な","の","と","や","も","ね","よ","その","この","あの"
}
_WORD_NGS: List[int] = [1]
_CHAR_NGS: List[int] = []


def normalize(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text).strip()
    out = []
    for ch in text:
        c = unicodedata.category(ch)
        if c.startswith("S") or c.startswith("C") or c in ("Zl", "Zp"):
            continue
        out.append(ch)
    return "".join(out)

def tokenize_words(text: str) -> Iterable[str]:
    text = normalize(text)
    for tok in _t.tokenize(text):
        base = tok.base_form if tok.base_form != "*" else tok.surface
        pos = tok.part_of_speech.split(",")[0]
        if pos in _USE_POS and base and base not in _STOPWORDS:
            yield base

def word_ngrams(tokens: Iterable[str], ns: List[int]) -> Iterable[str]:
    toks = list(tokens)
    for n in ns:
        if n <= 0:
            continue
        if n == 1:
            for t in toks:
                yield t
        else:
            for i in range(0, max(0, len(toks) - n + 1)):
                yield "␟".join(toks[i:i+n])

def char_ngrams(text: str, ns: List[int]) -> Iterable[str]:
    s = re.sub(r"\s+", "", normalize(text))
    for n in ns:
        if n <= 0:
            continue
        for i in range(0, max(0, len(s) - n + 1)):
            yield s[i:i+n]

def _query_terms(query: str) -> List[str]:
    terms = []
    if _WORD_NGS:
        base = list(tokenize_words(query))
        terms.extend(word_ngrams(base, _WORD_NGS))
    if _CHAR_NGS:
        terms.extend(char_ngrams(query, _CHAR_NGS))
    return terms


def _load_settings(cur: sqlite3.Cursor) -> Tuple[float, float, int, float, dict]:
    kv = dict(cur.execute("SELECT key, value FROM params").fetchall())
    k1 = float(kv.get("k1", "1.5"))
    b = float(kv.get("b", "0.75"))
    N = int(float(kv.get("N", "0")))
    avgdl = float(kv.get("avgdl", "0.0"))
    settings = json.loads(kv.get("settings", "{}")) if "settings" in kv else {}
    return k1, b, N, avgdl, settings

def _apply_tokenizer_settings(settings: dict):
    global _USE_POS, _STOPWORDS, _WORD_NGS, _CHAR_NGS
    tokset = settings.get("tokenize", {})
    if "use_pos" in tokset: _USE_POS = set(tokset["use_pos"])
    if "stopwords" in tokset: _STOPWORDS = set(tokset["stopwords"])
    ngs = settings.get("ngrams", {})
    _WORD_NGS = list(ngs.get("word", [1]))
    _CHAR_NGS = list(ngs.get("char", []))

def _override_ngrams(word_ngrams_override: Optional[List[int]], char_ngrams_override: Optional[List[int]]):
    """DB設定のあとに、検索時指定で n-gram 範囲を上書き（None は上書きしない / [] は無効化）。"""
    global _WORD_NGS, _CHAR_NGS
    if word_ngrams_override is not None:
        _WORD_NGS = list(word_ngrams_override)
    if char_ngrams_override is not None:
        _CHAR_NGS = list(char_ngrams_override)

def _fetch_term_rows(cur: sqlite3.Cursor, terms: List[str]) -> Dict[str, Tuple[int, int]]:
    if not terms:
        return {}
    uniq = list(set(terms))
    qmarks = ",".join(["?"] * len(uniq))
    rows = cur.execute(f"SELECT id, term, df FROM terms WHERE term IN ({qmarks})", uniq).fetchall()
    out = {}
    for tid, term, df in rows:
        out[term] = (int(tid), int(df))
    return out

def _fetch_postings(cur: sqlite3.Cursor, term_ids: List[int]) -> List[Tuple[int,int,int]]:
    if not term_ids:
        return []
    qmarks = ",".join(["?"] * len(term_ids))
    return cur.execute(
        f"SELECT term_id, doc_id, tf FROM postings WHERE term_id IN ({qmarks})", term_ids
    ).fetchall()

def _fetch_docs_meta(cur: sqlite3.Cursor, doc_ids: List[int]) -> Dict[int, Tuple[str,str,str,int]]:
    if not doc_ids:
        return {}
    qmarks = ",".join(["?"] * len(doc_ids))
    rows = cur.execute(
        f"SELECT id, ext_id, title, path, length FROM docs WHERE id IN ({qmarks})", doc_ids
    ).fetchall()
    return {int(i): (ext_id, title, path, int(length)) for (i, ext_id, title, path, length) in rows}

def _bm25_rank(
    postings: List[Tuple[int,int,int]],
    termid_to_idf: Dict[int, float],
    docid_to_len: Dict[int, int],
    k1: float, b: float, avgdl: float
) -> Dict[int, float]:
    scores: Dict[int, float] = defaultdict(float)
    for term_id, doc_id, tf in postings:
        dl = docid_to_len.get(int(doc_id), 0)
        if dl == 0:
            continue
        K = k1 * (1 - b + b * (dl / avgdl if avgdl > 0 else 0.0))
        idf = termid_to_idf.get(int(term_id), 0.0)
        scores[int(doc_id)] += idf * ((int(tf) * (k1 + 1)) / (int(tf) + K))
    return scores


def search_bm25(
    db_path: str,
    query: str,
    topk: int = 10,
    include_path: bool = False,
    return_fields: Optional[List[str]] = None,
    return_terms: bool = True,
    *,
    word_ngrams: Optional[List[int]] = None,
    char_ngrams: Optional[List[int]] = None,
) -> List[Dict]:
    if return_fields is None:
        return_fields = ["rank", "doc_id", "ext_id", "title", "score", "length"] + (["path"] if include_path else [])

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    k1, b, N, avgdl, settings = _load_settings(cur)
    _apply_tokenizer_settings(settings)
    _override_ngrams(word_ngrams, char_ngrams)

    q_terms = _query_terms(query)
    if not q_terms:
        return []

    term_rows = _fetch_term_rows(cur, q_terms)
    if not term_rows:
        return []

    termid_to_term = {tid: term for term, (tid, _) in term_rows.items()}

    term_ids = [tid for (tid, _) in term_rows.values()]
    df_map = {tid: df for (_, (tid, df)) in term_rows.items()}
    termid_to_idf = {tid: (math.log((N - df + 0.5) / (df + 0.5) + 1.0) if N > 0 else 0.0)
                     for tid, df in df_map.items()}

    posts = _fetch_postings(cur, term_ids)
    if not posts:
        return []

    doc_ids = sorted({doc_id for (_, doc_id, _) in posts})
    meta = _fetch_docs_meta(cur, doc_ids)
    docid_to_len = {doc_id: length for doc_id, (_, _, _, length) in meta.items()}

    doc_hit_terms: Dict[int, set] = defaultdict(set)
    for term_id, doc_id, tf in posts:
        if return_terms and term_id in termid_to_term:
            doc_hit_terms[doc_id].add(termid_to_term[term_id])

    scores = _bm25_rank(posts, termid_to_idf, docid_to_len, k1, b, avgdl)
    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
    results = []
    for r, (doc_id, score) in enumerate(ranked, start=1):
        ext_id, title, path, dl = meta.get(doc_id, ("?", "?", "?", 0))
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
        if return_terms:
            row["hit_terms"] = sorted(doc_hit_terms.get(doc_id, []))
        results.append(row)
    return results
