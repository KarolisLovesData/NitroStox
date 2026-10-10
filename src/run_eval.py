"""NitroStox SEC-RAG evaluation runner.

Put this file in your NitroStox project folder, next to sec_rag.py,
upgraded_nitrostox.py, filing_AMZN.txt and eval_questions_AMZN.json.

  python run_eval.py --limit 3           # quick smoke test, retrieval only, first 3 questions
  python run_eval.py                     # retrieval test on all questions
  python run_eval.py --agent             # also ask the agent every question
  python run_eval.py --agent --runs 2    # ask each question twice (answers vary between runs)

What it measures
  A. Retrieval: for each answerable question, does the chunk search put a chunk that
     contains your evidence_snippet in the top 3 / top 5? Reported strict (primary snippet
     only) and lenient (primary or alt_snippets). The CSV also lists the top-5 chunk ids.
  B. Agent (with --agent): the question goes through run_gemini_agent, the same path a
     user takes. The script records the answer, whether the agent called the SEC tool,
     and whether the chunks it received contained your snippet. You grade the answer.

The agent test is pinned to your saved filing: StockAnalyzer is patched in this process
only, so it reads filing_AMZN.txt instead of fetching the latest filing from EDGAR.
Your source files are not modified.

Needs GEMINI_API_KEY in .env. --agent also needs the GCP credentials the app already uses.
"""
import argparse
import csv
import datetime
import functools
import inspect
import json
import re
import time

import numpy as np

from sec_rag import SECVectorRAG


def norm(s: str) -> str:
    """Collapse all whitespace runs. The filing has long runs of spaces inside tables."""
    return re.sub(r"\s+", " ", s or "").strip()


def load_inputs(questions_path: str, filing_path: str):
    with open(questions_path, encoding="utf-8") as f:
        spec = json.load(f)
    # Default newline handling matches what the app sees (text with \n line breaks).
    with open(filing_path, encoding="utf-8") as f:
        text = f.read()
    return spec, text


def build_index(ticker: str, text: str):
    rag = SECVectorRAG(ticker)
    chunks = rag.chunk_text(text)  # same defaults the app uses
    embeddings = rag.embed_chunks(chunks)
    pairs = [(c, e) for c, e in zip(chunks, embeddings) if e is not None and len(e) > 0]
    dropped = len(chunks) - len(pairs)
    if dropped:
        print(f"WARNING: {dropped} of {len(chunks)} chunks got no embedding (quota?). "
              "Retrieval numbers are unreliable. Wait a minute and rerun.")
    if not pairs:
        raise SystemExit("No embeddings were generated. Check GEMINI_API_KEY and quota.")
    matrix = np.vstack([p[1] for p in pairs]).astype(np.float32)
    return rag, [p[0] for p in pairs], matrix, len(chunks)


def all_snippets(q):
    """Primary evidence snippet first, then any alternates (all whitespace-normalized)."""
    if not q.get("evidence_snippet"):
        return []
    return [norm(q["evidence_snippet"])] + [norm(s) for s in q.get("alt_snippets", [])]


def retrieval_eval(rag, chunks, embeddings, questions):
    """Run each question text as the search query; find where the evidence chunk ranks.

    strict  = primary snippet only
    lenient = primary snippet or any alternate snippet
    """
    results = {}
    for q in questions:
        df = rag.vector_search_in_memory(q["question"], chunks, embeddings, top_k=5)
        texts = [norm(t) for t in df["chunk_text"].tolist()] if not df.empty else []
        ids = [int(i) for i in df["chunk_index"].tolist()] if not df.empty else []
        snips = all_snippets(q)
        rank_strict = rank_lenient = None
        if snips:
            for i, t in enumerate(texts, start=1):
                if rank_strict is None and snips[0] in t:
                    rank_strict = i
                if rank_lenient is None and any(s in t for s in snips):
                    rank_lenient = i
        if df.empty:
            print(f"WARNING: no search results for {q['id']} (query embedding failed?). Counted as a miss.")
        results[q["id"]] = {
            "empty": df.empty,
            "rank_strict": rank_strict,
            "rank_lenient": rank_lenient,
            "has_snippet": bool(snips),
            "top_ids": ids,
        }
    return results


def setup_agent(ticker: str, text: str):
    """Import the app module and pin it to the saved filing. Returns (module, call_log).

    The app caches the knowledge base per ticker (_KB_CACHE), so only the part that
    downloads the filing is replaced; the filing is embedded once and reused.
    """
    import upgraded_nitrostox as nx

    def pinned_build_uncached(self):
        rag = SECVectorRAG(self.ticker)
        raw_chunks = rag.chunk_text(text)
        raw_embeddings = rag.embed_chunks(raw_chunks)
        pairs = [(c, e) for c, e in zip(raw_chunks, raw_embeddings) if e is not None and len(e) > 0]
        if not pairs:
            print("Pinned build failed: no embeddings (quota?).")
            return 0
        self.sec_chunks = [p[0] for p in pairs]
        self.sec_embeddings = np.vstack([p[1] for p in pairs]).astype(np.float32)
        nx._KB_CACHE[self.ticker] = {
            "chunks": self.sec_chunks,
            "embeddings": self.sec_embeddings,
            "built_at": time.time(),
        }
        return len(self.sec_chunks)

    if not hasattr(nx.StockAnalyzer, "_build_sec_knowledge_base_uncached"):
        raise SystemExit("upgraded_nitrostox.py has no _build_sec_knowledge_base_uncached; "
                         "send me the current file so I can adapt run_eval.py.")
    nx._KB_CACHE.pop(ticker.upper().strip(), None)  # make sure no live-filing cache is reused
    nx.StockAnalyzer._build_sec_knowledge_base_uncached = pinned_build_uncached

    call_log = []
    original_tool = nx.tool_query_sec_filings

    @functools.wraps(original_tool)  # keeps name, signature and docstring the SDK reads
    def logged_tool(ticker: str, query: str) -> str:
        out = original_tool(ticker, query)
        call_log.append({"query": query, "output": out})
        return out

    nx.tool_query_sec_filings = logged_tool
    return nx, call_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval_questions_AMZN.json")
    ap.add_argument("--filing", default="filing_AMZN.txt")
    ap.add_argument("--ticker", default="AMZN")
    ap.add_argument("--agent", action="store_true", help="also run every question through the agent")
    ap.add_argument("--runs", type=int, default=1, help="agent runs per question")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke test)")
    args = ap.parse_args()

    mode = f"retrieval + agent ({args.runs} run(s) per question)" if args.agent else "retrieval ONLY (add --agent to also ask the agent)"
    print(f"MODE: {mode}")
    spec, text = load_inputs(args.questions, args.filing)
    questions = spec["questions"][: args.limit] if args.limit else spec["questions"]

    sig = inspect.signature(SECVectorRAG.chunk_text).parameters
    chunk_size, overlap = sig["chunk_size"].default, sig["overlap"].default
    print(f"Filing: {args.filing} ({len(text):,} chars) | chunk_size={chunk_size} overlap={overlap}")

    rag, chunks, embeddings, n_total = build_index(args.ticker, text)
    print(f"Index: {len(chunks)} of {n_total} chunks embedded")

    retr = retrieval_eval(rag, chunks, embeddings, questions)

    # Summary over answerable questions that have a snippet
    scored = [q for q in questions if q.get("answerable") is True and q.get("evidence_snippet")]

    def count(key, limit):
        return sum(1 for q in scored if retr[q["id"]][key] is not None and retr[q["id"]][key] <= limit)

    n = len(scored)
    print(f"\nRETRIEVAL (question text used as the query, answerable questions only, n={n})")
    print(f"  strict  (primary snippet only):      hit@3 {count('rank_strict', 3)}/{n}   hit@5 {count('rank_strict', 5)}/{n}")
    print(f"  lenient (primary or alternate):      hit@3 {count('rank_lenient', 3)}/{n}   hit@5 {count('rank_lenient', 5)}/{n}")
    empties = [qid for qid, r in retr.items() if r["empty"]]
    if empties:
        print(f"  WARNING: {len(empties)} queries returned nothing: {empties}. Rerun before trusting these numbers.")
    miss_s = [q["id"] for q in scored if retr[q["id"]]["rank_strict"] is None]
    miss_l = [q["id"] for q in scored if retr[q["id"]]["rank_lenient"] is None]
    print(f"  not in top 5 (strict):  {miss_s if miss_s else 'none'}")
    print(f"  not in top 5 (lenient): {miss_l if miss_l else 'none'}")

    agent_rows = {}
    if args.agent:
        nx, call_log = setup_agent(args.ticker, text)
        for q in questions:
            snips = all_snippets(q)
            for run in range(1, args.runs + 1):
                call_log.clear()
                try:
                    answer = nx.run_gemini_agent(q["question"], active_ticker=args.ticker)
                except Exception as e:  # keep going; record the failure
                    answer = f"[ERROR] {e}"
                retrieved = norm(" ".join(c["output"] for c in call_log))
                agent_rows[(q["id"], run)] = {
                    "called": bool(call_log),
                    "queries": " | ".join(c["query"] for c in call_log),
                    "snippet_in_chunks": (any(sn in retrieved for sn in snips) if call_log else False) if snips else None,
                    "answer": answer,
                }
                print(f"  {q['id']} run {run}: sec_tool={'yes' if call_log else 'NO'}")
                time.sleep(1.5)

        asked = list(agent_rows.values())
        used = sum(1 for r in asked if r["called"])
        print(f"\nAGENT: SEC tool called on {used}/{len(asked)} answers")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"eval_results_{stamp}.csv"
    fields = ["run_at", "id", "category", "answerable", "question", "expected_answer",
              "rank_strict", "rank_lenient", "hit_at_3_lenient", "hit_at_5_lenient", "top5_chunk_ids",
              "run", "agent_called_sec_tool", "agent_search_queries", "agent_chunks_contain_snippet",
              "agent_answer", "grade", "notes"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for q in questions:
            r = retr[q["id"]]
            base = {
                "run_at": stamp, "id": q["id"], "category": q["category"], "answerable": q["answerable"],
                "question": q["question"], "expected_answer": q.get("expected_answer") or "",
                "rank_strict": r["rank_strict"] if r["rank_strict"] is not None else "",
                "rank_lenient": r["rank_lenient"] if r["rank_lenient"] is not None else "",
                "hit_at_3_lenient": (r["rank_lenient"] is not None and r["rank_lenient"] <= 3) if r["has_snippet"] else "",
                "hit_at_5_lenient": (r["rank_lenient"] is not None) if r["has_snippet"] else "",
                "top5_chunk_ids": " ".join(str(i) for i in r["top_ids"]),
            }
            if args.agent:
                for run in range(1, args.runs + 1):
                    a = agent_rows[(q["id"], run)]
                    w.writerow({**base, "run": run, "agent_called_sec_tool": a["called"],
                                "agent_search_queries": a["queries"],
                                "agent_chunks_contain_snippet": "" if a["snippet_in_chunks"] is None else a["snippet_in_chunks"],
                                "agent_answer": a["answer"], "grade": "", "notes": ""})
            else:
                w.writerow({**base, "run": "", "agent_called_sec_tool": "", "agent_search_queries": "",
                            "agent_chunks_contain_snippet": "", "agent_answer": "", "grade": "", "notes": ""})
    print(f"\nSaved: {out_path}")
    if not args.agent:
        print("NOTE: this was retrieval only, so the agent_answer columns are empty. "
              "Run again with:  python run_eval.py --agent --runs 2")


if __name__ == "__main__":
    main()
