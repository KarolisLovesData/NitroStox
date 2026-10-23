"""Ingest the latest SEC filing of several companies into the local filing store.

  python ingest_filings.py                          # AMZN, MSFT and GOOGL
  python ingest_filings.py --tickers AAPL NVDA      # other companies
  python ingest_filings.py --dry-run                # download and chunk only: no Gemini calls, nothing saved
  python ingest_filings.py --force                  # ingest again even if the filing is already stored

For each ticker it finds the newest 10-Q or 10-K on SEC EDGAR, skips it if that exact filing
(same accession number) is already in the store, otherwise downloads it as Markdown, splits it
with the same table-aware chunker the app uses, embeds the chunks with Gemini and saves them in
filing_store/ tagged with ticker, form and period. Empty and duplicate chunks are dropped before
embedding; odd chunks (very short, oversized, no words) are kept but listed in the report.

The app then reads a stored filing instead of downloading and embedding it again. Run this
again when a company files a new 10-Q or 10-K: only the new filing is embedded.

Needs SEC_IDENTITY (your name and email, which the SEC asks for) and GEMINI_API_KEY in .env.
--dry-run needs no Gemini key. One filing is about 15 to 25 embedding calls (20 chunks each).
"""
import argparse
import datetime
import os
import re
import traceback

import numpy as np
from dotenv import load_dotenv

import filing_store
from chunking import chunk_filing_text

DEFAULT_TICKERS = ["AMZN", "MSFT", "GOOGL"]
FORMS = ("10-Q", "10-K")
FOREIGN_FORMS = ("20-F",)  # only tried when a company files no 10-Q or 10-K
CHUNK_SIZE = 2000          # same defaults as SECVectorRAG.chunk_text
OVERLAP = 300

SHORT_CHARS = 100          # a chunk body shorter than this is flagged
MIN_LETTERS = 20           # fewer letters than this: probably a stray table fragment
MIN_FILING_CHARS = 30_000  # a 10-Q is usually 150,000+ characters
MIN_CHUNKS = 40


def latest_filing(company):
    """Newest 10-Q or 10-K (amendments excluded); a 20-F if the company files neither."""
    for forms in (FORMS, FOREIGN_FORMS):
        best = None
        for form in forms:
            try:
                filings = company.get_filings(form=form, amendments=False)
            except TypeError:  # older edgartools without the amendments filter
                filings = company.get_filings(form=form)
            if len(filings) == 0:
                continue
            f = filings.head(1)[0]
            if best is None or str(f.filing_date) > str(best.filing_date):
                best = f
        if best is not None:
            return best
    return None


def filing_text(filing) -> str:
    """Markdown keeps each table with its header row; plain text is the fallback."""
    text = ""
    if hasattr(filing, "markdown"):
        try:
            text = filing.markdown() or ""
        except Exception as e:
            print(f"   markdown() failed ({e}); trying plain text")
    return text or filing.text() or ""


def _body(chunk: str) -> str:
    """The chunk without its "[Section: ...]" first line."""
    first, _, rest = chunk.partition("\n")
    return rest if first.startswith("[Section:") else chunk


def check_chunks(chunks: list[str]) -> tuple[list[str], dict]:
    """Drops empty and duplicate chunks; flags odd ones. Returns (kept chunks, issues)."""
    issues = {"empty": 0, "duplicate": 0, "short": [], "oversized": [], "no_words": []}
    kept, seen = [], set()
    for c in chunks:
        body = _body(c).strip()
        if not body:
            issues["empty"] += 1
            continue
        if c in seen:
            issues["duplicate"] += 1
            continue
        seen.add(c)
        i = len(kept)
        kept.append(c)
        if len(body) < SHORT_CHARS:
            issues["short"].append(i)
        if len(c) > CHUNK_SIZE * 1.5:
            issues["oversized"].append(i)
        if len(re.findall(r"[A-Za-z]", body)) < MIN_LETTERS:
            issues["no_words"].append(i)
    return kept, issues


def describe(issues: dict, n_chunks: int, text_len: int) -> list[str]:
    """Plain-language lines for the report."""
    lines = []
    if text_len < MIN_FILING_CHARS:
        lines.append(f"filing text is only {text_len:,} characters; it may be incomplete")
    if n_chunks < MIN_CHUNKS:
        lines.append(f"only {n_chunks} chunks; the filing may not have downloaded fully")
    if issues["empty"] or issues["duplicate"]:
        lines.append(f"dropped {issues['empty']} empty and {issues['duplicate']} duplicate chunks")
    for key, what in (("short", f"under {SHORT_CHARS} characters"),
                      ("oversized", f"over {int(CHUNK_SIZE * 1.5):,} characters"),
                      ("no_words", "almost no words (table fragments)")):
        if issues[key]:
            ids = issues[key]
            shown = ", ".join(map(str, ids[:8])) + (" ..." if len(ids) > 8 else "")
            lines.append(f"{len(ids)} chunk{'s' if len(ids) > 1 else ''} {what}: {shown}")
    return lines


def ingest(ticker: str, dry_run: bool, force: bool) -> dict:
    from edgar import Company

    company = Company(ticker)
    filing = latest_filing(company)
    if filing is None:
        return {"status": "failed", "note": "no 10-Q, 10-K or 20-F found"}

    # report_date is in the company's filing list; period_of_report reads the filing header
    period = str(getattr(filing, "report_date", "") or getattr(filing, "period_of_report", "")
                 or filing.filing_date)
    meta = {
        "ticker": ticker,
        "company": getattr(company, "name", ticker),
        "form": filing.form,
        "period": period,
        "filed": str(filing.filing_date),
        "accession": filing.accession_no,
    }
    print(f"   latest: {meta['form']} for period {period}, filed {meta['filed']} ({meta['accession']})")

    if not force and filing_store.find(meta["accession"]):
        return {**meta, "status": "skipped", "note": "already in the store"}

    text = filing_text(filing)
    if not text.strip():
        return {**meta, "status": "failed", "note": "SEC returned an empty document"}

    raw = chunk_filing_text(text, CHUNK_SIZE, OVERLAP)
    chunks, issues = check_chunks(raw)
    notes = describe(issues, len(chunks), len(text))
    print(f"   {len(text):,} characters -> {len(chunks)} chunks")
    if dry_run:
        return {**meta, "status": "dry run", "chunks": len(chunks), "notes": notes}

    from sec_rag import EMBED_MODEL, SECVectorRAG  # needs GEMINI_API_KEY

    vectors = SECVectorRAG(ticker).embed_chunks(chunks)
    pairs = [(c, v) for c, v in zip(chunks, vectors) if len(v) > 0]
    failed = len(chunks) - len(pairs)
    if not pairs:
        return {**meta, "status": "failed", "note": "Gemini returned no embeddings (key or quota)"}
    if failed:
        # Saving a filing with holes would make search quietly miss parts of it
        return {**meta, "status": "failed",
                "note": f"{failed} of {len(chunks)} chunks got no embedding (quota?); nothing saved, run again later"}

    meta.update({
        "embed_model": EMBED_MODEL,
        "chunker": "table",
        "chunk_size": CHUNK_SIZE,
        "ingested_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "notes": notes,
    })
    filing_store.save(meta, [p[0] for p in pairs], np.vstack([p[1] for p in pairs]), text)
    return {**meta, "status": "new", "chunks": len(pairs)}


def main():
    ap = argparse.ArgumentParser(description="Store the latest 10-Q/10-K of several companies for search.")
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--dry-run", action="store_true", help="download and chunk only; no Gemini calls, nothing saved")
    ap.add_argument("--force", action="store_true", help="ingest again even if the filing is already stored")
    args = ap.parse_args()

    load_dotenv()
    from edgar import set_identity
    set_identity(os.getenv("SEC_IDENTITY", "DataAnalyst user@example.com"))

    results = {}
    for t in [x.upper().strip() for x in args.tickers]:
        print(f"\n{t}")
        try:
            results[t] = ingest(t, args.dry_run, args.force)
        except Exception as e:
            traceback.print_exc()
            results[t] = {"status": "failed", "note": f"{type(e).__name__}: {e}"}

    print("\nREPORT")
    for t, r in results.items():
        what = f"{r.get('form', '')} {r.get('period', '')}".strip()
        size = f", {r['chunks']} chunks" if r.get("chunks") else ""
        note = f" ({r['note']})" if r.get("note") and what else r.get("note", "")
        print(f"  {t:6} {r['status']:8} {what}{size}{note}")
        for line in r.get("notes", []):
            print(f"         - {line}")
    print(f"\nStore: {filing_store.STORE_DIR} ({len(filing_store.manifest())} filings)")
    if any(r["status"] == "failed" for r in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
