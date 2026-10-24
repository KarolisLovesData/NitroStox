"""Run the eval on several stored companies and print one table.

  python eval_companies.py                          # the 40 questions in eval_questions_companies40.json
  python eval_companies.py --no-judge               # tool choice and citations only (no Claude calls)
  python eval_companies.py --questions eval_questions_standard.json --no-judge
                                                    # 20 neutral questions asked of every company
  python eval_companies.py --tickers AMZN MSFT      # only these stored companies
  python eval_companies.py --retrieval              # search only: does vector, hybrid or keyword search find the
                                                    # right passage? (no agent, no judge, no Claude credit)

Runs run_eval.py once per company on the filing ingest_filings.py stored for it (agent only,
saved vectors, so the only Gemini calls are the agent's answers and its search queries; the
judge is Claude and needs ANTHROPIC_API_KEY). A question file whose questions carry a ticker
asks each company only its own questions. Each company gets its own CSV, trace and summary;
this script prints one table. Extra options (e.g. --runs 2, --agent-model ...) are passed to
run_eval.py as they are.
"""
import argparse
import json
import os
import re
import subprocess
import sys

DEFAULT_TICKERS = ["AMZN", "MSFT", "GOOGL"]


def retrieval_table(tickers, questions, extra):
    """Runs run_eval's retrieval test per company and prints how often each search finds the right passage."""
    found = {}  # ticker -> {mode: (top3, top5, n)}
    for t in tickers:
        print(f"\n===== {t} =====", flush=True)
        cmd = [sys.executable, "run_eval.py", "--from-store", "--ticker", t, "--questions", questions, *extra]
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        mode = None
        found[t] = {}
        for line in proc.stdout:
            print(line, end="", flush=True)
            m = re.match(r"RETRIEVAL \[(\w+)\]", line.strip())
            if m:
                mode = m.group(1)
            m = re.match(r"lenient .*hit@3 (\d+)/(\d+)\s+hit@5 (\d+)/(\d+)", line.strip())
            if m and mode:
                found[t][mode] = (int(m.group(1)), int(m.group(3)), int(m.group(2)))
        proc.wait()

    modes = [m for m in ("vector", "hybrid", "keyword") if any(m in v for v in found.values())]
    print("\nRIGHT PASSAGE FOUND (question used as the search; top 3 / top 5 results)")
    print(f"  {'ticker':8}" + "".join(f"{m:16}" for m in modes))
    total = {m: [0, 0, 0] for m in modes}
    for t, v in found.items():
        cells = []
        for m in modes:
            if m in v:
                a, b, n = v[m]
                cells.append(f"{a}/{n} | {b}/{n}")
                total[m] = [total[m][0] + a, total[m][1] + b, total[m][2] + n]
            else:
                cells.append("did not run")
        print(f"  {t:8}" + "".join(f"{c:16}" for c in cells))
    print(f"  {'TOTAL':8}" + "".join(f"{str(total[m][0]) + '/' + str(total[m][2]) + ' | ' + str(total[m][1]) + '/' + str(total[m][2]):16}"
                                      for m in modes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--questions", default="eval_questions_companies40.json")
    ap.add_argument("--retrieval", action="store_true",
                    help="only test the search (vector vs hybrid vs keyword); no agent answers, no judge")
    args, extra = ap.parse_known_args()
    if args.retrieval:
        return retrieval_table([x.upper().strip() for x in args.tickers], args.questions, extra)

    summaries = {}
    for t in [x.upper().strip() for x in args.tickers]:
        print(f"\n===== {t} =====", flush=True)
        cmd = [sys.executable, "run_eval.py", "--agent", "--skip-retrieval", "--from-store",
               "--ticker", t, "--questions", args.questions, *extra]
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}  # emoji in the app's messages
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        path = None
        for line in proc.stdout:  # show run_eval's progress live and catch the summary file name
            print(line, end="", flush=True)
            m = re.match(r"Saved: (eval_summary_\S+\.json)", line.strip())
            path = m.group(1) if m else path
        proc.wait()
        if path:
            with open(path, encoding="utf-8") as f:
                summaries[t] = json.load(f)
        else:
            summaries[t] = None

    print("\nRESULTS BY COMPANY")
    print(f"  {'ticker':8}{'tool choice':16}{'correct':10}{'partial':9}{'wrong':7}{'declined ok':13}"
          f"{'citations ok':14}finished")
    for t, s in summaries.items():
        if not s:
            print(f"  {t:8}did not run (see the error above)")
            continue
        o = s["overall"]
        a = o.get("answerable", {})
        graded = lambda k: f"{a[k]}/{a['n']}" if a else "-"
        done = f"{s['answers_finished']}/{s['answers_expected']}"
        print(f"  {t:8}{o.get('tool_choice_correct', '-'):16}{graded('correct'):10}{graded('partial'):9}"
              f"{graded('incorrect'):7}{o.get('unanswerable_declined', '-'):13}{o.get('citations_ok', '-'):14}{done}")

    done_runs = [s for s in summaries.values() if s]
    if len(done_runs) > 1:
        def total(key, sub=None):
            ok = n = 0
            for s in done_runs:
                v = s["overall"].get(key)
                if sub and v:
                    ok, n = ok + v[sub], n + v["n"]
                elif isinstance(v, str) and re.match(r"\d+/\d+", v):
                    a_, b_ = map(int, re.match(r"(\d+)/(\d+)", v).groups())
                    ok, n = ok + a_, n + b_
            return f"{ok}/{n}" if n else "-"
        finished = f"{sum(s['answers_finished'] for s in done_runs)}/{sum(s['answers_expected'] for s in done_runs)}"
        print(f"  {'TOTAL':8}{total('tool_choice_correct'):16}{total('answerable', 'correct'):10}"
              f"{total('answerable', 'partial'):9}{total('answerable', 'incorrect'):7}"
              f"{total('unanswerable_declined'):13}{total('citations_ok'):14}{finished}")

    for t, s in summaries.items():
        if not s:
            continue
        cats = []
        for cat, b in s["by_category"].items():
            m = re.match(r"(\d+)/(\d+)", b.get("tool_choice_correct", ""))
            if m and m.group(1) != m.group(2):
                cats.append(cat)
        if cats:
            print(f"  {t}: wrong tool choices in {', '.join(cats)} (see its CSV, column tool_choice_ok)")


if __name__ == "__main__":
    main()
