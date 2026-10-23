"""NitroStox SEC-RAG evaluation runner.

Put this file in your NitroStox project folder, next to sec_rag.py,
upgraded_nitrostox.py, filing_AMZN.md and the eval question files.

The default question set is eval_questions_AMZN_tools.json: 20 questions mixing filing questions,
price/indicator questions, watchlist requests and questions no tool can answer. The original 20
filing-only questions are still in eval_questions_AMZN.json:

  python run_eval.py --agent --filing filing_AMZN.md                 # 20 answers: the usual run
  python run_eval.py --agent --skip-retrieval --filing filing_AMZN.md   # same, without the chunk search test
  python run_eval.py --agent --questions eval_questions_AMZN.json    # the original filing-only questions
  python run_eval.py --agent --runs 2 --skip-retrieval --questions eval_questions_AMZN_tools50.json
                                                                     # 50 questions for comparing changes (30 judge-free)
  python run_eval.py --agent --agent-model <gemini model> ...         # same run with another agent model

  python run_eval.py --limit 3           # quick smoke test, retrieval only, first 3 questions
  python run_eval.py                     # retrieval test on all questions
  python run_eval.py --agent             # also ask the agent every question
  python run_eval.py --agent --runs 2    # ask each question twice (answers vary between runs)
  python run_eval.py --agent --no-judge  # skip the LLM judge (number check still runs)
  python run_eval.py --search all        # retrieval for vector, hybrid and keyword search side by side
  python run_eval.py --agent --runs 2 --search vector   # agent with vector-only search
  python run_eval.py --agent --runs 2 --search hybrid   # agent with hybrid (keyword + vector) search
  python run_eval.py --chunker fixed     # old 2,000-character chunks, to compare with the default "table"
  python run_eval.py --agent --runs 2 --resume eval_trace_<stamp>_<mode>.jsonl
                                         # continue a run that stopped early (quota, Ctrl+C, crash)

If a run stops early, the CSV and summary are still written for the answers finished so far,
and the trace file has every finished answer. Re-run with --resume to do only the rest.

What it measures
  A. Retrieval: for each answerable question, does the chunk search put a chunk that
     contains your evidence_snippet in the top 3 / top 5? Reported strict (primary snippet
     only) and lenient (primary or alt_snippets). The CSV also lists the top-5 chunk ids.
  B. Agent (with --agent): the question goes through run_gemini_agent, the same path a
     user takes. The script records the answer, which tools the agent called,
     and whether the chunks it received contained your snippet. Each answer is then graded:
       - tool choice: did the agent call exactly the tools in the question's expected_tools
         (filing search, signals table, watchlist, or none)? Reported as % of correct choices.
       - number check: numbers in the answer that do not appear in the tool output
       - citation check: every [chunk N] the answer cites was really retrieved, every sentence
         with a number has a citation, and those numbers are in the chunk it cites. Also: does a
         cited chunk contain your evidence_snippet (the citation points at the right place)?
       - LLM judge: correctness vs expected_answer, grounded in the tool output or not,
         wrong-period figures, whether the citations support their claims, and refusal on
         unanswerable questions. The judge is Claude
         (default claude-sonnet-5-5, needs ANTHROPIC_API_KEY and `pip install anthropic`), a
         different model family from the Gemini agent it grades.
     The judge is a model too: spot-check some of its grades by hand before trusting them.

Outputs (same timestamp; <mode> is the search mode: vector, hybrid or keyword):
  eval_results_<stamp>_<mode>.csv     one row per question (and run), graded
  eval_trace_<stamp>_<mode>.jsonl     with --agent: every tool call, the chunk ids, scores and text it
                                      returned, the answer and both grades
  eval_summary_<stamp>_<mode>.json    with --agent: metrics overall and per category
With --search all (the default) retrieval runs once per mode on the same embeddings, and the
CSV and the agent use hybrid.

The agent test is pinned to your saved filing: StockAnalyzer is patched in this process
only, so it reads the saved filing instead of fetching the latest filing from EDGAR.
Your source files are not modified.

Needs GEMINI_API_KEY in .env. --agent also needs the GCP credentials the app already uses.
"""
import argparse
import csv
import datetime
import functools
import inspect
import json
import os
import re
import time

import numpy as np

import sec_rag
from eval_grading import (DEFAULT_JUDGE_MODEL, check_citations, judge_answer, make_judge_client,
                          number_check, summarize)
from keyword_search import BM25Index
from sec_rag import SECVectorRAG

SEARCH_MODES = ["vector", "hybrid", "keyword"]


AGENT_ERROR_PREFIXES = ("Gemini Agent execution failed", "[ERROR]")
MAX_CONSECUTIVE_FAILURES = 3  # stop early rather than burn through every question while Gemini is down


def is_agent_error(answer: str) -> bool:
    """run_gemini_agent returns its exception text as the answer, so errors look like answers."""
    return (answer or "").startswith(AGENT_ERROR_PREFIXES)


def ask_agent(nx, question: str, ticker: str, attempts: int = 4) -> tuple[str, bool]:
    """One agent turn, retried with a growing pause when Gemini is overloaded (503) or rate limited (429).

    Returns (answer, rewritten): rewritten is True when the app sent the agent back to fix its citations.
    """
    answer, rewritten = "", False
    for attempt in range(attempts):
        try:
            if hasattr(nx, "run_gemini_agent_detailed"):
                res = nx.run_gemini_agent_detailed(question, active_ticker=ticker)
                answer, rewritten = res["answer"], bool(res.get("rewritten"))
            else:
                answer = nx.run_gemini_agent(question, active_ticker=ticker)
        except Exception as e:
            answer = f"[ERROR] {e}"
        if not is_agent_error(answer):
            return answer, rewritten
        if attempt < attempts - 1:
            wait = 15 * 2 ** attempt
            print(f"    agent call failed ({answer[:90]}...), retrying in {wait}s")
            time.sleep(wait)
    return answer, rewritten


def norm(s: str) -> str:
    """Collapse all whitespace runs. The filing has long runs of spaces inside tables.

    Markdown table pipes, bold markers and $ signs are removed too, so snippets copied from
    the plain-text filing still match Markdown chunks (the Markdown tables repeat "$" in
    every column; the plain text does not).
    """
    s = re.sub(r"[|*$]", " ", s or "")
    return re.sub(r"\s+", " ", s).strip()


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
    kept = [p[0] for p in pairs]
    return rag, kept, matrix, BM25Index(kept), len(chunks)


def all_snippets(q):
    """Primary evidence snippet first, then any alternates (all whitespace-normalized)."""
    if not q.get("evidence_snippet"):
        return []
    return [norm(q["evidence_snippet"])] + [norm(s) for s in q.get("alt_snippets", [])]


def retrieval_eval(rag, chunks, embeddings, bm25, questions, mode="hybrid"):
    """Run each question text as the search query; find where the evidence chunk ranks.

    mode = vector | hybrid | keyword (same search the app runs with NITROSTOX_SEARCH_MODE)

    strict  = primary snippet only
    lenient = primary snippet or any alternate snippet
    """
    results = {}
    for q in questions:
        df = rag.search_in_memory(q["question"], chunks, embeddings, bm25, top_k=5, mode=mode)
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
            why = "no keyword matched" if mode == "keyword" else "query embedding failed?"
            print(f"WARNING: no {mode} search results for {q['id']} ({why}). Counted as a miss.")
        results[q["id"]] = {
            "empty": df.empty,
            "rank_strict": rank_strict,
            "rank_lenient": rank_lenient,
            "has_snippet": bool(snips),
            "top_ids": ids,
        }
    return results


def tool_choice(expected, calls):
    """True when the agent called exactly the expected tools (order and repeats ignored).

    expected is the question's expected_tools list ([] = no tool); None when the question has none.
    """
    if expected is None:
        return None
    return {c["tool"] for c in calls} == set(expected)


def setup_agent(ticker: str, text: str):
    """Import the app module and pin it to the saved filing. Returns (module, call_log).

    call_log gets one entry per tool call the agent makes. SEC entries include the chunk
    ids and similarity scores, recorded from the same search the tool ran.

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
        self.sec_bm25 = BM25Index(self.sec_chunks)
        nx._KB_CACHE[self.ticker] = {
            "chunks": self.sec_chunks,
            "embeddings": self.sec_embeddings,
            "bm25": self.sec_bm25,
            "built_at": time.time(),
        }
        return len(self.sec_chunks)

    if not hasattr(nx.StockAnalyzer, "_build_sec_knowledge_base_uncached"):
        raise SystemExit("upgraded_nitrostox.py has no _build_sec_knowledge_base_uncached; "
                         "send me the current file so I can adapt run_eval.py.")
    nx._KB_CACHE.pop(ticker.upper().strip(), None)  # make sure no live-filing cache is reused
    nx.StockAnalyzer._build_sec_knowledge_base_uncached = pinned_build_uncached

    call_log = []
    last_search = []  # search_sec_filings results seen during the current SEC tool call

    original_search = nx.StockAnalyzer.search_sec_filings

    @functools.wraps(original_search)
    def recorded_search(self, query, top_k=3):
        df = original_search(self, query, top_k)
        last_search.append(df)
        return df

    nx.StockAnalyzer.search_sec_filings = recorded_search

    original_sec = nx.tool_query_sec_filings

    @functools.wraps(original_sec)  # keeps name, signature and docstring the SDK reads
    def logged_sec(ticker: str, query: str) -> str:
        last_search.clear()
        out = original_sec(ticker, query)
        chunks = []
        for df in last_search:
            if df is not None and not df.empty:
                chunks.extend({"chunk_index": int(r.chunk_index),
                               "score": float(r.similarity_score) if r.similarity_score is not None else float("nan"),
                               "text": r.chunk_text} for r in df.itertuples())
        call_log.append({"tool": "sec", "ticker": ticker, "query": query, "output": out, "chunks": chunks})
        return out

    original_tech = nx.tool_get_technical_signals

    @functools.wraps(original_tech)
    def logged_tech(ticker: str) -> str:
        out = original_tech(ticker)
        call_log.append({"tool": "technical", "ticker": ticker, "output": out})
        return out

    original_watch = nx.tool_manage_watchlist

    @functools.wraps(original_watch)
    def logged_watch(ticker: str, action: str) -> str:
        # Never touch the real watchlist from an eval run. Reply like the real tool, so the
        # answer and the judge read normally, and mark the call as not executed in the trace.
        done = {"ADD": "added {} to", "REMOVE": "removed {} from"}.get(str(action).upper().strip())
        out = (f"Successfully {done.format(str(ticker).upper().strip())} BigQuery watchlist." if done
               else f"Invalid action '{action}'. Action must be 'ADD' or 'REMOVE'.")
        call_log.append({"tool": "watchlist", "ticker": ticker, "action": action, "output": out,
                         "executed": False})
        return out

    nx.tool_query_sec_filings = logged_sec
    nx.tool_get_technical_signals = logged_tech
    nx.tool_manage_watchlist = logged_watch
    return nx, call_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval_questions_AMZN_tools.json",
                    help="question file. Default: the 20-question tool-choice set. "
                         "eval_questions_AMZN.json is the original 20 filing-only questions.")
    ap.add_argument("--filing", default="filing_AMZN.md")
    ap.add_argument("--ticker", default="AMZN")
    ap.add_argument("--agent", action="store_true", help="also run every question through the agent")
    ap.add_argument("--runs", type=int, default=1, help="agent runs per question")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke test)")
    ap.add_argument("--resume", metavar="TRACE_JSONL", default="",
                    help="continue an earlier --agent run: reuse its finished answers and ask only the rest")
    ap.add_argument("--chunker", choices=["table", "fixed"], default="table",
                    help="table = header-keeping chunks the app now uses; fixed = the old 2,000-character windows "
                         "(run both to compare)")
    ap.add_argument("--agent-model", default="",
                    help="Gemini model the agent uses (default: the app's own, gemini-3.7-flash). "
                         "Run the same questions with two models to compare their tool choice.")
    ap.add_argument("--skip-retrieval", action="store_true",
                    help="with --agent: skip the chunk search test and only run the agent "
                         "(saves one round of embeddings)")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge (number check still runs)")
    ap.add_argument("--search", choices=SEARCH_MODES + ["all"], default="all",
                    help="search mode: vector, hybrid (keyword + vector, RRF) or keyword. "
                         "'all' compares the three in the retrieval test; the CSV and the agent then use hybrid.")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                    help="model for the LLM judge. claude-* uses ANTHROPIC_API_KEY; gemini-* uses GEMINI_API_KEY. "
                         "Default is a Claude model so the judge is not the same model family as the agent.")
    args = ap.parse_args()
    if args.agent_model:
        os.environ["NITROSTOX_AGENT_MODEL"] = args.agent_model  # read when upgraded_nitrostox is imported

    mode = f"retrieval + agent ({args.runs} run(s) per question)" if args.agent else "retrieval ONLY (add --agent to also ask the agent)"
    print(f"MODE: {mode}")
    spec, text = load_inputs(args.questions, args.filing)
    questions = spec["questions"][: args.limit] if args.limit else spec["questions"]

    SECVectorRAG.DEFAULT_STRATEGY = args.chunker  # the app and the agent test both read this
    sig = inspect.signature(SECVectorRAG.chunk_text).parameters
    chunk_size, overlap = sig["chunk_size"].default, sig["overlap"].default
    print(f"Filing: {args.filing} ({len(text):,} chars) | chunker={args.chunker} "
          f"chunk_size={chunk_size} overlap={overlap}")

    search_mode = "hybrid" if args.search == "all" else args.search  # used for the CSV and the agent
    modes = SEARCH_MODES if args.search == "all" else [args.search]
    filing_qs = [q for q in questions if "sec" in q.get("expected_tools", ["sec"])]
    if args.skip_retrieval:
        if not args.agent:
            raise SystemExit("--skip-retrieval only makes sense together with --agent")
        print("Retrieval test skipped: the agent builds its own search index.")
        retr_by_mode, modes = {}, [search_mode]
        retr = {}
    else:
        rag, chunks, embeddings, bm25, n_total = build_index(args.ticker, text)
        print(f"Index: {len(chunks)} of {n_total} chunks embedded (keyword index on the same chunks)")
        retr_by_mode = {m: retrieval_eval(rag, chunks, embeddings, bm25, filing_qs, mode=m) for m in modes}
        retr = retr_by_mode[search_mode]

    # Summary over answerable questions that have a snippet
    scored = [q for q in filing_qs if q.get("answerable") is True and q.get("evidence_snippet")]
    n = len(scored)

    for m, rm in retr_by_mode.items():
        def count(key, limit):
            return sum(1 for q in scored if rm[q["id"]][key] is not None and rm[q["id"]][key] <= limit)

        print(f"\nRETRIEVAL [{m}] (question text used as the query, answerable questions only, n={n})")
        print(f"  strict  (primary snippet only):      hit@3 {count('rank_strict', 3)}/{n}   hit@5 {count('rank_strict', 5)}/{n}")
        print(f"  lenient (primary or alternate):      hit@3 {count('rank_lenient', 3)}/{n}   hit@5 {count('rank_lenient', 5)}/{n}")
        empties = [qid for qid, r in rm.items() if r["empty"]]
        if empties and m != "keyword":  # keyword search can legitimately find nothing
            print(f"  WARNING: {len(empties)} queries returned nothing: {empties}. Rerun before trusting these numbers.")
        miss_s = [q["id"] for q in scored if rm[q["id"]]["rank_strict"] is None]
        miss_l = [q["id"] for q in scored if rm[q["id"]]["rank_lenient"] is None]
        print(f"  not in top 5 (strict):  {miss_s if miss_s else 'none'}")
        print(f"  not in top 5 (lenient): {miss_l if miss_l else 'none'}")

    if len(modes) > 1:
        # Questions whose evidence chunk (lenient) ranks differently between modes; '-' = not in top 5
        changed = [q for q in scored if len({retr_by_mode[m][q["id"]]["rank_lenient"] for m in modes}) > 1]
        print(f"\nRANK CHANGES (lenient rank of the evidence chunk, '-' = not in top 5): {len(changed)} of {n}")
        if changed:
            print("  " + "id".ljust(10) + "".join(m.ljust(10) for m in modes))
            for q in changed:
                cells = [retr_by_mode[m][q["id"]]["rank_lenient"] for m in modes]
                print("  " + q["id"].ljust(10) + "".join(("-" if c is None else str(c)).ljust(10) for c in cells))

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    agent_rows = {}
    stopped_early = False
    trace_path = ""
    if args.resume and not args.agent:
        raise SystemExit("--resume only makes sense together with --agent")
    if args.agent:
        sec_rag.SEARCH_MODE = search_mode  # search_sec_filings reads this when the agent searches
        print(f"Agent search mode: {search_mode}")
        nx, call_log = setup_agent(args.ticker, text)
        judge_client = None
        if not args.no_judge:
            judge_client = make_judge_client(args.judge_model)  # fails fast if the key or package is missing

        trace_path = f"eval_trace_{stamp}_{search_mode}.jsonl"
        trace_mode = "w"
        if args.resume:
            trace_path, trace_mode = args.resume, "a"
            m = re.search(r"eval_trace_(\d{8}_\d{4})", os.path.basename(args.resume))
            stamp = m.group(1) if m else stamp  # keep the original stamp so outputs line up
            with open(args.resume, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        if "tool_ok" not in rec:  # trace from before tool choice was recorded
                            exp = next((q.get("expected_tools") for q in questions if q["id"] == rec["id"]), None)
                            rec["expected_tools"], rec["tool_ok"] = exp, tool_choice(exp, rec.get("tool_calls", []))
                        agent_rows[(rec["id"], rec["run"])] = rec
            # Answers that were API errors, or whose judge call failed, are not finished: redo them
            bad = [k for k, r in agent_rows.items()
                   if is_agent_error(r["answer"]) or (not args.no_judge and (r.get("judge") or {}).get("judge_error"))]
            for k in bad:
                del agent_rows[k]
            if judge_client:
                # Keep one judge for the whole baseline: re-grade (no new agent call) any answer a different judge graded
                by_id = {q["id"]: q for q in questions}
                stale = [k for k, r in agent_rows.items() if r.get("judge_model") != args.judge_model and k[0] in by_id]
                for k in stale:
                    r = agent_rows[k]
                    tool_text = "\n\n".join(c["output"] for c in r.get("tool_calls", []))
                    j = judge_answer(judge_client, args.judge_model, by_id[k[0]], r["answer"], tool_text)
                    if "judge_error" in j:
                        del agent_rows[k]  # ask this one again from scratch
                    else:
                        r["judge"], r["judge_model"] = j, args.judge_model
                if stale:
                    print(f"Re-graded {len(stale)} earlier answers with {args.judge_model} so one judge grades everything")
            with open(args.resume, "w", encoding="utf-8") as f:  # drop error rows, keep re-graded ones
                for r in agent_rows.values():
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Resuming: {len(agent_rows)} finished answers loaded from {args.resume}"
                  + (f"; {len(bad)} earlier answers had API errors and will be asked again" if bad else ""))

        def run_agent_loop(trace):
            failures = 0
            for q in questions:
                snips = all_snippets(q)
                for run in range(1, args.runs + 1):
                    if (q["id"], run) in agent_rows:
                        continue
                    call_log.clear()
                    answer, rewritten = ask_agent(nx, q["question"], args.ticker)
                    if is_agent_error(answer):
                        # Not a real answer: do not save it, so --resume asks this question again
                        failures += 1
                        print(f"  {q['id']} run {run}: API error, not saved ({answer[:100]})")
                        if failures >= MAX_CONSECUTIVE_FAILURES:
                            raise RuntimeError(f"Gemini failed {failures} answers in a row; try again later with --resume")
                        continue
                    calls = list(call_log)
                    tool_ok = tool_choice(q.get("expected_tools"), calls)
                    sec_calls = [c for c in calls if c["tool"] == "sec"]
                    tool_text = "\n\n".join(c["output"] for c in calls)
                    retrieved = norm(" ".join(c["output"] for c in sec_calls))
                    chunk_hits = [h for c in sec_calls for h in c["chunks"]]

                    numbers = number_check(answer, tool_text)
                    sources = {str(h["chunk_index"]): h["text"] for h in chunk_hits}
                    if any(c["tool"] == "technical" for c in calls):
                        sources["signals"] = "\n".join(c["output"] for c in calls if c["tool"] == "technical")
                    cites = check_citations(answer, sources)
                    cited_text = norm(" ".join(sources[c] for c in cites["cited"] if c in sources))
                    cited_evidence = (any(sn in cited_text for sn in snips) if snips else None)
                    judge = None
                    if judge_client and not q.get("tool_only"):  # tool_only questions have no expected answer to grade
                        judge = judge_answer(judge_client, args.judge_model, q, answer, tool_text)
                        time.sleep(1.0)
                        if "judge_error" in judge:
                            failures += 1
                            print(f"  {q['id']} run {run}: judge API error, not saved ({judge['judge_error'][:100]})")
                            if failures >= MAX_CONSECUTIVE_FAILURES:
                                raise RuntimeError(f"Gemini failed {failures} calls in a row; try again later with --resume")
                            continue
                    failures = 0

                    row = {
                        "id": q["id"], "category": q["category"], "answerable": q["answerable"],
                        "called": bool(sec_calls),
                        "tools": " ".join(c["tool"] for c in calls),
                        "expected_tools": q.get("expected_tools"),
                        "tool_ok": tool_ok,
                        "queries": " | ".join(c["query"] for c in sec_calls),
                        "chunk_ids": " ".join(str(h["chunk_index"]) for h in chunk_hits),
                        "chunk_scores": " ".join(f"{h['score']:.3f}" for h in chunk_hits),
                        "snippet_in_chunks": (any(sn in retrieved for sn in snips) if sec_calls else False) if snips else None,
                        "answer": answer,
                        "number_check": numbers,
                        "citations": cites,
                        "cited_evidence": cited_evidence,
                        "rewritten": rewritten,
                        "judge": judge,
                        "judge_model": args.judge_model if judge else None,
                    }
                    agent_rows[(q["id"], run)] = row
                    trace.write(json.dumps({"run_at": stamp, "run": run, "question": q["question"],
                                            "expected_answer": q.get("expected_answer"),
                                            "tool_calls": calls, **row}, ensure_ascii=False) + "\n")
                    trace.flush()  # a finished answer survives a crash or Ctrl+C

                    grade = judge.get("correctness", "judge error") if judge else "-"
                    flag = f" unsupported numbers: {numbers['unsupported_numbers']}" if numbers["unsupported_numbers"] else ""
                    flag += "" if cites["ok"] else (f" citation problems: {len(cites['invalid'])} invalid, "
                                                    f"{len(cites['uncited_claims'])} uncited, "
                                                    f"{len(cites['unsupported'])} not in cited chunk")
                    used = " ".join(dict.fromkeys(c["tool"] for c in calls)) or "none"
                    want = (" ".join(q["expected_tools"]) or "none") if q.get("expected_tools") is not None else "?"
                    pick = "" if tool_ok is None else (" ok" if tool_ok else " WRONG")
                    print(f"  {q['id']} run {run}: tools={used} (expected {want}){pick} "
                          f"chunks=[{row['chunk_ids']}] judge={grade}{flag}")
                    time.sleep(1.5)

        with open(trace_path, trace_mode, encoding="utf-8") as trace:
            try:
                run_agent_loop(trace)
            except (KeyboardInterrupt, Exception) as e:
                stopped_early = True
                print(f"\nSTOPPED EARLY: {type(e).__name__}: {e}")

        summary = summarize(list(agent_rows.values()))
        summary_path = f"eval_summary_{stamp}_{search_mode}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"run_at": stamp, "filing": args.filing, "runs": args.runs, "search_mode": search_mode,
                       "chunker": args.chunker,
                       "agent_model": args.agent_model or getattr(nx, "AGENT_MODEL", "default"),
                       "judge_model": None if args.no_judge else args.judge_model,
                       "complete": not stopped_early, "answers_finished": len(agent_rows),
                       "answers_expected": len(questions) * args.runs, **summary}, f, indent=2)
        print("\nAGENT SUMMARY")
        print(json.dumps(summary, indent=2))
        print(f"\nSaved: {trace_path}\nSaved: {summary_path}")
    out_path = f"eval_results_{stamp}_{search_mode}.csv"
    fields = ["run_at", "search_mode", "id", "category", "answerable", "question", "expected_answer",
              "rank_strict", "rank_lenient", "hit_at_3_lenient", "hit_at_5_lenient", "top5_chunk_ids",
              "run", "expected_tools", "agent_tools_called", "tool_choice_ok", "agent_called_sec_tool", "agent_search_queries",
              "agent_chunk_ids", "agent_chunk_scores", "agent_chunks_contain_snippet",
              "agent_answer", "unsupported_numbers", "citations", "citations_ok", "invalid_citations",
              "uncited_claims", "numbers_not_in_cited_chunk", "cited_chunks_contain_evidence", "rewritten",
              "judge_correctness", "judge_grounded", "judge_period_error", "judge_citations_supported",
              "judge_unsupported_claims", "judge_reason", "grade", "notes"]
    agent_cols = fields[fields.index("run"):]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for q in questions:
            r = retr.get(q["id"]) or {"rank_strict": None, "rank_lenient": None, "has_snippet": False, "top_ids": []}
            base = {
                "run_at": stamp, "search_mode": search_mode, "id": q["id"], "category": q["category"], "answerable": q["answerable"],
                "question": q["question"], "expected_answer": q.get("expected_answer") or "",
                "rank_strict": r["rank_strict"] if r["rank_strict"] is not None else "",
                "rank_lenient": r["rank_lenient"] if r["rank_lenient"] is not None else "",
                "hit_at_3_lenient": (r["rank_lenient"] is not None and r["rank_lenient"] <= 3) if r["has_snippet"] else "",
                "hit_at_5_lenient": (r["rank_lenient"] is not None) if r["has_snippet"] else "",
                "top5_chunk_ids": " ".join(str(i) for i in r["top_ids"]),
            }
            if args.agent:
                for run in range(1, args.runs + 1):
                    a = agent_rows.get((q["id"], run))
                    if a is None:  # run stopped before this answer; resume will fill it in
                        continue
                    j = a["judge"] or {}
                    ct = a.get("citations") or {}  # absent in traces from before citations were added
                    exp = a.get("expected_tools", q.get("expected_tools"))
                    w.writerow({**base, "run": run, "agent_tools_called": a["tools"],
                                "expected_tools": "" if exp is None else (" ".join(exp) or "none"),
                                "tool_choice_ok": "" if a.get("tool_ok") is None else a["tool_ok"],
                                "agent_called_sec_tool": a["called"],
                                "agent_search_queries": a["queries"],
                                "agent_chunk_ids": a["chunk_ids"], "agent_chunk_scores": a["chunk_scores"],
                                "agent_chunks_contain_snippet": "" if a["snippet_in_chunks"] is None else a["snippet_in_chunks"],
                                "agent_answer": a["answer"],
                                "unsupported_numbers": " ; ".join(a["number_check"]["unsupported_numbers"]),
                                "citations": " ".join(ct.get("cited", [])),
                                "citations_ok": ct.get("ok", ""),
                                "invalid_citations": " ".join(ct.get("invalid", [])),
                                "uncited_claims": " ; ".join(ct.get("uncited_claims", [])),
                                "numbers_not_in_cited_chunk": " ; ".join(
                                    f"{', '.join(u['numbers'])} ({u['sentence'][:80]})" for u in ct.get("unsupported", [])),
                                "cited_chunks_contain_evidence": "" if a.get("cited_evidence") is None else a["cited_evidence"],
                                "rewritten": a.get("rewritten", ""),
                                "judge_citations_supported": j.get("citations_supported", ""),
                                "judge_correctness": j.get("correctness", j.get("judge_error", "")),
                                "judge_grounded": j.get("grounded", ""),
                                "judge_period_error": j.get("period_error", ""),
                                "judge_unsupported_claims": " ; ".join(j.get("unsupported_claims", [])),
                                "judge_reason": j.get("reason", ""), "grade": "", "notes": ""})
            else:
                w.writerow({**base, **{c: "" for c in agent_cols}})
    print(f"\nSaved: {out_path}")
    if stopped_early:
        print(f"\nThe run did not finish: {len(agent_rows)} of {len(questions) * args.runs} answers are saved.\n"
              f"To do only the rest, run:\n  python run_eval.py --agent --runs {args.runs} --search {search_mode} --resume {trace_path}")
    if not args.agent:
        print("NOTE: this was retrieval only, so the agent_answer columns are empty. "
              "Run again with:  python run_eval.py --agent --runs 2")


if __name__ == "__main__":
    main()
