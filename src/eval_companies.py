"""Run the same company-neutral questions on several stored companies and compare them.

  python eval_companies.py                          # AMZN, MSFT, GOOGL with eval_questions_standard.json
  python eval_companies.py --tickers AMZN MSFT      # other stored companies
  python eval_companies.py --limit 3                # quick check: first 3 questions per company

Runs run_eval.py once per company on the filing ingest_filings.py stored for it (agent only,
no judge, saved vectors, so the only Gemini calls are the agent's answers and its search
queries). Each company gets its own CSV, trace and summary; this script prints one table.
Extra options (e.g. --runs 2, --agent-model ...) are passed to run_eval.py as they are.
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
    ap.add_argument("--questions", default="eval_questions_standard.json")
    args, extra = ap.parse_known_args()

    summaries = {}
    for t in [x.upper().strip() for x in args.tickers]:
        print(f"\n===== {t} =====", flush=True)
        cmd = [sys.executable, "run_eval.py", "--agent", "--skip-retrieval", "--no-judge", "--from-store",
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
    print(f"  {'ticker':8}{'tool choice':18}{'citations ok':15}{'unsupported numbers':22}finished")
    for t, s in summaries.items():
        if not s:
            print(f"  {t:8}did not run (see the error above)")
            continue
        o = s["overall"]
        done = f"{s['answers_finished']}/{s['answers_expected']}"
        print(f"  {t:8}{o.get('tool_choice_correct', '-'):18}{o.get('citations_ok', '-'):15}"
              f"{str(o.get('answers_with_unsupported_numbers', '-')):22}{done}")

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
