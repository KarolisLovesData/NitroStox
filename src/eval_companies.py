"""Run the eval on several stored companies and print one table.

  python eval_companies.py                          # the 40 questions in eval_questions_companies40.json
  python eval_companies.py --no-judge               # tool choice and citations only (no Claude calls)
  python eval_companies.py --questions eval_questions_standard.json --no-judge
                                                    # 20 neutral questions asked of every company
  python eval_companies.py --tickers AMZN MSFT      # only these stored companies

Runs run_eval.py once per company on the filing ingest_filings.py stored for it (agent only,
saved vectors, so the only Gemini calls are the agent's answers and its search queries; the
judge is Claude and needs ANTHROPIC_API_KEY). A question file whose questions carry a ticker
asks each company only its own questions. Each company gets its own CSV, trace and summary;
this script prints one table. Extra options (e.g. --runs 2, --agent-model ...) are passed to
run_eval.py as they are.
"""
import argparse
import json
import re
import subprocess
import sys

DEFAULT_TICKERS = ["AMZN", "MSFT", "GOOGL"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--questions", default="eval_questions_companies40.json")
    args, extra = ap.parse_known_args()

    summaries = {}
    for t in [x.upper().strip() for x in args.tickers]:
        print(f"\n===== {t} =====", flush=True)
        cmd = [sys.executable, "run_eval.py", "--agent", "--skip-retrieval", "--from-store",
               "--ticker", t, "--questions", args.questions, *extra]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
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
