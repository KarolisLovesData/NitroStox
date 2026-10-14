"""Answer grading for run_eval.py.

Two checks per agent answer:
  1. number_check: every number in the answer must appear in what the agent's tools returned.
     Cheap and deterministic. It catches invented or calculated figures (a gross margin the
     filing never reports, a derived prior-year revenue). It does NOT catch a real number
     attributed to the wrong period; the judge covers that.
  2. judge_answer: an LLM grades the answer against expected_answer and the tool output
     (correctness, grounding, period errors, refusal on unanswerable questions).
"""
import json
import re
import time

from google.genai import types

# A number, optionally followed by a unit. The lookbehind skips ids like Q2, 10-Q, GB200, Note 4.
_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9.\-])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?!,?\d|\.\d|-\w)"
    r"(?:\)?\s*(%|percent\b|billion\b|bn\b|B\b|million\b|mn\b|M\b|thousand\b|K\b))?"
)
_SCALE_TO_MILLIONS = {"billion": 1000.0, "bn": 1000.0, "b": 1000.0,
                      "million": 1.0, "mn": 1.0, "m": 1.0,
                      "thousand": 0.001, "k": 0.001}

REL_TOL = 0.005      # 18.16 billion matches 18,164 (million)
PCT_ABS_TOL = 0.05   # percentages must match as reported; a computed 36.8% vs reported 37% is flagged


def extract_numbers(text: str) -> list[dict]:
    """Numbers in text as {raw, value, unit} where unit is 'pct', 'mn' (scaled to millions) or ''."""
    found = []
    for m in _NUM_RE.finditer(text or ""):
        whole, frac, unit = m.group(1), m.group(2) or "", (m.group(3) or "").lower()
        value = float(whole.replace(",", "") + frac)
        if unit in ("%", "percent"):
            found.append({"raw": m.group(0).strip(), "value": value, "unit": "pct"})
        elif unit:
            found.append({"raw": m.group(0).strip(), "value": value * _SCALE_TO_MILLIONS[unit], "unit": "mn"})
        else:
            found.append({"raw": m.group(0).strip(), "value": value, "unit": ""})
    return found


def _is_noise(n: dict) -> bool:
    """Years, list markers and other small bare integers carry no checkable claim."""
    if n["unit"]:
        return False
    v = n["value"]
    if v.is_integer() and (v < 10 or 1990 <= v <= 2100):
        return True
    return False


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= REL_TOL * max(abs(a), abs(b), 1e-9)


def number_check(answer: str, source_text: str) -> dict:
    """Flags numbers in the answer that cannot be found in source_text (the tool output)."""
    source = extract_numbers(source_text)
    source_any = [s["value"] for s in source]  # bare table cells are usually in millions
    source_pct = [s["value"] for s in source if s["unit"] == "pct"]

    checked, unsupported = 0, []
    for n in extract_numbers(answer):
        if _is_noise(n):
            continue
        checked += 1
        if n["unit"] == "pct":
            ok = any(abs(n["value"] - s) <= PCT_ABS_TOL for s in source_pct)
        else:
            ok = any(_close(n["value"], s) for s in source_any)
        if not ok and n["raw"] not in unsupported:
            unsupported.append(n["raw"])
    return {"numbers_checked": checked, "unsupported_numbers": unsupported}


JUDGE_PROMPT = """You are grading an answer from a financial research agent that answers questions about one SEC filing.
Use ONLY the TOOL OUTPUT below as the source of truth for what the agent saw. EXPECTED ANSWER and GRADER NOTE come from a human who read the filing.

QUESTION: {question}
ANSWERABLE FROM THE FILING: {answerable}
EXPECTED ANSWER: {expected}
GRADER NOTE: {note}

TOOL OUTPUT THE AGENT RECEIVED:
<<<
{tool_output}
>>>

AGENT ANSWER:
<<<
{answer}
>>>

Grade it and return JSON with exactly these keys:
- "correctness": for an answerable question one of "correct", "partial", "incorrect", "declined"; for a question that is not answerable one of "declined" (it says the information is not in the filing) or "answered" (it gives an answer anyway).
- "grounded": true if every factual claim and number in the answer is stated in the tool output; false if anything comes from outside knowledge or is calculated and presented as reported.
- "period_error": true if any figure is attributed to the wrong period (for example last year's value given as this quarter's).
- "unsupported_claims": list of short strings, each a claim or number not supported by the tool output (empty list if none).
- "reason": one sentence explaining the grade.
"""


def judge_answer(client, model: str, q: dict, answer: str, tool_output: str,
                 max_tool_chars: int = 30000) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=q["question"],
        answerable="yes" if q.get("answerable") else "no",
        expected=q.get("expected_answer") or "(none: the filing does not contain this)",
        note=q.get("note") or "(none)",
        tool_output=(tool_output or "(the agent called no tools)")[:max_tool_chars],
        answer=answer,
    )
    config = types.GenerateContentConfig(temperature=0, response_mime_type="application/json")
    last_err = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(model=model, contents=[prompt], config=config)
            data = json.loads(resp.text)
            if isinstance(data, list) and data:
                data = data[0]
            return {
                "correctness": str(data.get("correctness", "")).lower(),
                "grounded": bool(data.get("grounded")),
                "period_error": bool(data.get("period_error")),
                "unsupported_claims": list(data.get("unsupported_claims") or []),
                "reason": str(data.get("reason", "")),
            }
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    return {"judge_error": str(last_err)}


def summarize(rows: list[dict]) -> dict:
    """Per-category and overall metrics from the per-answer rows built in run_eval.main()."""
    def block(rs):
        out = {"answers": len(rs)}
        if not rs:
            return out
        answerable = [r for r in rs if r["answerable"] is True]
        out["sec_tool_called"] = sum(r["called"] for r in rs)
        with_snip = [r for r in answerable if r["snippet_in_chunks"] is not None]
        if with_snip:
            out["chunks_contain_evidence"] = f"{sum(bool(r['snippet_in_chunks']) for r in with_snip)}/{len(with_snip)}"
        out["answers_with_unsupported_numbers"] = sum(1 for r in rs if r["number_check"]["unsupported_numbers"])
        judged = [r for r in rs if r.get("judge") and "judge_error" not in r["judge"]]
        if judged:
            ja = [r for r in judged if r["answerable"] is True]
            ju = [r for r in judged if r["answerable"] is not True]
            if ja:
                out["answerable"] = {k: sum(r["judge"]["correctness"] == k for r in ja)
                                     for k in ("correct", "partial", "incorrect", "declined")}
                out["answerable"]["n"] = len(ja)
            if ju:
                out["unanswerable_declined"] = f"{sum(r['judge']['correctness'] == 'declined' for r in ju)}/{len(ju)}"
            out["grounded"] = f"{sum(r['judge']['grounded'] for r in judged)}/{len(judged)}"
            out["period_errors"] = sum(r["judge"]["period_error"] for r in judged)
        out["judge_errors"] = sum(1 for r in rs if r.get("judge") and "judge_error" in r["judge"])
        return out

    cats = sorted({r["category"] for r in rows})
    return {"overall": block(rows), "by_category": {c: block([r for r in rows if r["category"] == c]) for c in cats}}
