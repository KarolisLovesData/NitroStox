"""Look at how a filing gets chunked, without Gemini (nothing is embedded).

  python preview_chunks.py                       # AMZN: download the latest 10-Q as Markdown
  python preview_chunks.py --ticker MSFT
  python preview_chunks.py --file filing_AMZN.txt  # chunk a file you already saved
  python preview_chunks.py --find "Net income"     # print the chunks containing this text

Saves the Markdown as filing_<TICKER>.md, then prints how many chunks the new chunker makes,
how many table chunks carry their header, and the chunks that contain --find, next to the old
2,000-character chunks for comparison. Needs SEC_IDENTITY in .env (your name and email, which
the SEC asks for); no Gemini key is used.
"""
import argparse
import os

from dotenv import load_dotenv

from chunking import chunk_filing_text, looks_like_markdown


def fixed_chunks(text: str, size: int = 2000, overlap: int = 300) -> list[str]:
    """The old chunker, copied here so this script does not need the Gemini key."""
    return [text[i:i + size].strip() for i in range(0, len(text), size - overlap) if text[i:i + size].strip()]


def download_markdown(ticker: str) -> str:
    from edgar import Company, set_identity
    set_identity(os.getenv("SEC_IDENTITY", "DataAnalyst user@example.com"))
    company = Company(ticker)
    for form in ("10-Q", "10-K", "20-F", "6-K"):
        filings = company.get_filings(form=form)
        if len(filings):
            filing = filings.head(1)[0]
            print(f"Latest {form}: {filing.filing_date} (accession {filing.accession_no})")
            if not hasattr(filing, "markdown"):
                raise SystemExit("Your edgartools has no filing.markdown(). Run: pip install -U edgartools")
            return filing.markdown()
    raise SystemExit(f"No filings found for {ticker}")


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AMZN")
    ap.add_argument("--file", default="", help="chunk this saved file instead of downloading")
    ap.add_argument("--find", default="Net income", help="show the chunks that contain this text")
    ap.add_argument("--show", type=int, default=2, help="how many matching chunks to print")
    args = ap.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = download_markdown(args.ticker.upper())
        out = f"filing_{args.ticker.upper()}.md"
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved {out} ({len(text):,} characters)")

    new = chunk_filing_text(text)
    old = fixed_chunks(text)
    tables = [c for c in new if "\n|" in c or c.split("\n", 1)[-1].startswith("|")]
    with_period = [c for c in tables if any(w in c.lower() for w in ("months ended", "year ended", "as of"))]
    print(f"\nFormat: {'Markdown' if looks_like_markdown(text) else 'plain text'}")
    print(f"Old chunker: {len(old)} chunks | New chunker: {len(new)} chunks, longest {max(map(len, new)):,} chars")
    print(f"Table chunks: {len(tables)}, of which {len(with_period)} show a period header (e.g. 'Months Ended')")

    for name, chunks in (("NEW", new), ("OLD", old)):
        hits = [c for c in chunks if args.find.lower() in c.lower()]
        print(f"\n===== {name}: {len(hits)} chunks contain '{args.find}' (showing {min(args.show, len(hits))}) =====")
        for c in hits[:args.show]:
            print(c[:2000])
            print("-" * 60)


if __name__ == "__main__":
    main()
