"""Local store for ingested SEC filings: chunks, their vectors and tags.

ingest_filings.py writes here; the app and run_eval.py read from here. Each filing is kept as:
  <TICKER>_<FORM>_<period>.chunks.jsonl   one line per chunk: text + ticker, form, period, filed, accession
  <TICKER>_<FORM>_<period>.npy            the chunk vectors, same order as the .jsonl lines
  <TICKER>_<FORM>_<period>.md             the filing text the chunks came from
and manifest.json lists every stored filing. Every chunk carries its company, form and period,
so a search can be limited to one company (or one form or period) before ranking.

The folder is filing_store/ next to this file (set NITROSTOX_STORE_DIR to move it). Reading
needs no Gemini key; search_store() embeds the query, so it does.
"""
import json
import os

import numpy as np
import pandas as pd

STORE_DIR = os.getenv("NITROSTOX_STORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "filing_store"))
MANIFEST = "manifest.json"


def _path(name: str) -> str:
    return os.path.join(STORE_DIR, name)


def manifest() -> list[dict]:
    """Every stored filing, oldest first. [] when nothing was ingested yet."""
    try:
        with open(_path(MANIFEST), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def _write_manifest(entries: list[dict]) -> None:
    tmp = _path(MANIFEST + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    os.replace(tmp, _path(MANIFEST))  # never leaves a half-written manifest behind


def find(accession: str) -> dict | None:
    return next((m for m in manifest() if m["accession"] == accession), None)


def latest(ticker: str, form: str | None = None) -> dict | None:
    """The most recently filed stored filing for a ticker (optionally of one form)."""
    t = ticker.upper().strip()
    rows = [m for m in manifest() if m["ticker"] == t and (form is None or m["form"] == form)]
    return max(rows, key=lambda m: m["filed"]) if rows else None


def save(meta: dict, chunks: list[str], embeddings: np.ndarray, text: str) -> dict:
    """Stores one filing. meta needs ticker, form, period, filed and accession; the rest is kept as is.

    A filing with the same accession number replaces the old copy.
    """
    os.makedirs(STORE_DIR, exist_ok=True)
    base = f"{meta['ticker']}_{meta['form'].replace('/', '-')}_{meta['period']}"
    tags = {k: meta[k] for k in ("ticker", "form", "period", "filed", "accession")}
    with open(_path(base + ".chunks.jsonl"), "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            f.write(json.dumps({"chunk_id": i, **tags, "text": c}, ensure_ascii=False) + "\n")
    np.save(_path(base + ".npy"), np.asarray(embeddings, dtype=np.float32))
    with open(_path(base + ".md"), "w", encoding="utf-8") as f:
        f.write(text)

    entry = {**meta, "files": base, "n_chunks": len(chunks)}
    entries = [m for m in manifest() if m["accession"] != meta["accession"]]
    _write_manifest(entries + [entry])
    return entry


def load(entry: dict) -> tuple[list[dict], np.ndarray]:
    """(chunk records, vector matrix) for one manifest entry, in the same order."""
    with open(_path(entry["files"] + ".chunks.jsonl"), encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    matrix = np.load(_path(entry["files"] + ".npy"))
    if len(records) != len(matrix):
        raise ValueError(f"{entry['files']}: {len(records)} chunks but {len(matrix)} vectors; ingest it again")
    return records, matrix


def load_text(entry: dict) -> str:
    with open(_path(entry["files"] + ".md"), encoding="utf-8") as f:
        return f.read()


def load_chunks(ticker: str | None = None, form: str | None = None,
                period: str | None = None) -> tuple[list[dict], np.ndarray]:
    """All stored chunks that match the tags given (None = any), with their vectors."""
    records, mats = [], []
    for m in manifest():
        if ticker and m["ticker"] != ticker.upper().strip():
            continue
        if form and m["form"] != form:
            continue
        if period and m["period"] != period:
            continue
        r, mat = load(m)
        records.extend(r)
        mats.append(mat)
    return records, (np.vstack(mats) if mats else np.zeros((0, 0), dtype=np.float32))


def search_store(query: str, ticker: str | None = None, form: str | None = None,
                 period: str | None = None, top_k: int = 5, mode: str | None = None) -> pd.DataFrame:
    """Hybrid search over the stored chunks of one company (or form, or period).

    Same ranking the app uses (sec_rag.search_in_memory), run only on the chunks that match the
    filter, so another company's filing can never show up. Adds ticker, form and period columns.
    """
    from keyword_search import BM25Index
    from sec_rag import SECVectorRAG  # needs GEMINI_API_KEY to embed the query

    records, matrix = load_chunks(ticker, form, period)
    if not records:
        return pd.DataFrame()
    texts = [r["text"] for r in records]
    df = SECVectorRAG(ticker or "ALL").search_in_memory(query, texts, matrix, BM25Index(texts), top_k, mode)
    if df.empty:
        return df
    for col in ("ticker", "form", "period"):
        df[col] = [records[i][col] for i in df["chunk_index"]]
    return df
