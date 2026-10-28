"""Second judge: another model (default Llama 3.3 70B on Groq) re-grades answers Claude already graded, and the two grades are compared.

  python second_judge.py                       # newest hybrid run of each company (the 40-question eval)
  python second_judge.py --mode vector         # newest vector run instead
  python second_judge.py --traces eval_trace_20261024_0914_AMZN_hybrid.jsonl ...   # these runs only

No new agent answers: Gemini is not called. The second judge reads the same question, expected answer,
tool output and answer that Claude read (same prompt, judge_answer in eval_grading.py) and
grades them again. Needs GROQ_API_KEY in .env
(or XAI_API_KEY with --judge-model grok-4.7 for Grok).

The second judge's grades are saved next to the trace (second_judge_<trace name>_<model>.jsonl), so a run
that stops early continues where it left off. The original trace files are not changed.
"""
import argparse
import glob
import json
import os
import re
import time

from dotenv import load_dotenv

from eval_grading import SECOND_JUDGE_MODEL, judge_answer, make_judge_client

GRADES = ("correct", "partial", "incorrect", "declined", "answered")


def newest_traces(mode: str) -> list[str]:
    """Newest trace per company for this search mode, e.g. eval_trace_20261024_0914_AMZN_hybrid.jsonl."""
    newest = {}
    for path in sorted(glob.glob(f"eval_trace_*_{mode}.jsonl")):
        m = re.match(r"eval_trace_\d{8}_\d{4}_([A-Z.]+)_", os.path.basename(path))
        if m:
            newest[m.group(1)] = path  # sorted by stamp, so the last one wins
    return list(newest.values())


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", default=None, help="trace files to re-grade (default: newest per company)")
    ap.add_argument("--mode", default="hybrid", help="search mode of the default traces: hybrid or vector")
    ap.add_argument("--questions", default="eval_questions_companies40.json")
    ap.add_argument("--judge-model", default=SECOND_JUDGE_MODEL, help="second judge model: llama-* on Groq or grok-* on xAI")
    args = ap.parse_args()

    traces = args.traces or newest_traces(args.mode)
    if not traces:
        raise SystemExit(f"No eval_trace_*_{args.mode}.jsonl files here. Run eval_companies.py first.")
    with open(args.questions, encoding="utf-8") as f:
        by_id = {q["id"]: q for q in json.load(f)["questions"]}
    client = make_judge_client(args.judge_model)
    name = "Grok" if args.judge_model.startswith("grok") else "Llama" if args.judge_model.startswith("llama") else "Judge 2"

    pairs = []  # (id, first judge grade, second judge grade, first reason, second reason)
    for trace in traces:
        out_path = f"second_judge_{os.path.splitext(os.path.basename(trace))[0]}_{args.judge_model}.jsonl"
        done = {(r["id"], r["run"]): r for r in load_jsonl(out_path)}
        rows = [r for r in load_jsonl(trace) if r.get("judge") and "judge_error" not in r["judge"]]
        print(f"\n{trace}: {len(rows)} graded answers ({len(done)} already re-graded)", flush=True)
        with open(out_path, "a", encoding="utf-8") as out:
            for r in rows:
                key = (r["id"], r["run"])
                if key not in done:
                    q = by_id.get(r["id"])
                    if q is None:
                        print(f"  {r['id']}: not in {args.questions}, skipped")
                        continue
                    tool_text = "\n\n".join(c["output"] for c in r.get("tool_calls", []))
                    j = judge_answer(client, args.judge_model, q, r["answer"], tool_text)
                    if "judge_error" in j:
                        print(f"  {r['id']}: {name} error, not saved ({j['judge_error'][:100]})")
                        continue
                    done[key] = {"id": r["id"], "run": r["run"], "judge_model": args.judge_model, "judge": j}
                    out.write(json.dumps(done[key], ensure_ascii=False) + "\n")
                    out.flush()
                    time.sleep(1.0)
                first, second = r["judge"]["correctness"], done[key]["judge"]["correctness"]
                print(f"  {r['id']}: Claude {first}, {name} {second}{'' if first == second else '  <- differ'}", flush=True)
                pairs.append((r["id"], first, second, r["judge"].get("reason", ""), done[key]["judge"].get("reason", "")))

    if not pairs:
        return
    print(f"\nCLAUDE vs {name.upper()} ({args.judge_model}) on the same {len(pairs)} answers")
    print(f"  {'grade':12}{'Claude':>8}{name:>8}")
    for g in GRADES:
        a, b = sum(p[1] == g for p in pairs), sum(p[2] == g for p in pairs)
        if a or b:
            print(f"  {g:12}{a:>8}{b:>8}")
    same = sum(p[1] == p[2] for p in pairs)
    print(f"  Same grade: {same}/{len(pairs)} ({100 * same / len(pairs):.0f}%)")
    differ = [p for p in pairs if p[1] != p[2]]
    if differ:
        print("\nWHERE THEY DIFFER")
        for qid, a, b, ra, rb in differ:
            print(f"  {qid}: Claude {a}, {name} {b}\n    Claude: {ra}\n    {name}: {rb}")


if __name__ == "__main__":
    main()
