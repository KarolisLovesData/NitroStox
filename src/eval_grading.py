"""Answer grading for run_eval.py.

Three checks per agent answer:
  1. number_check: every number in the answer must appear in what the agent's tools returned.
     Cheap and deterministic. It catches invented or calculated figures (a gross margin the
     filing never reports, a derived prior-year revenue). It does NOT catch a real number
     attributed to the wrong period; the judge covers that.
  2. judge_answer: an LLM grades the answer against expected_answer and the tool output
     (correctness, grounding, period errors, whether the [chunk N] citations support their claims,
     refusal on unanswerable questions).
  3. check_citations (from grounding.py, the same check the app runs): cited chunks were really
     retrieved, every sentence with a number has a citation, and its numbers are in the cited chunk.
     The judge is a Claude model by default (ANTHROPIC_API_KEY), so the model that grades is
     not the Gemini model that wrote the answer. A Gemini judge is still possible by passing a
     gemini-* model name, and a Grok judge (XAI_API_KEY) by passing a grok-* name.
"""
import json
import re
import time

from google.genai import types

from grounding import check_citations, extract_numbers, unsupported_numbers  # noqa: F401 (extract_numbers re-exported)

DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"
DEFAULT_GROK_MODEL = "grok-4.7"  # second judge (second_judge.py), a third model family
XAI_URL = "https://api.x.ai/v1/chat/completions"


def make_judge_client(model: str):
    """Client for the judge model: Anthropic for claude-* names, xAI for grok-*, Gemini otherwise."""
    if model.startswith("grok"):
        import os
        import requests
        key = os.getenv("XAI_API_KEY")
        if not key:
            raise SystemExit("The judge model is Grok but XAI_API_KEY is not set. Add it to .env "
                             "(create a key at console.x.ai).")
        session = requests.Session()  # xAI speaks the OpenAI chat format, so plain requests is enough
        session.headers.update({"Authorization": f"Bearer {key}"})
        return session
    if model.startswith("claude"):
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise SystemExit("The judge model is Claude but ANTHROPIC_API_KEY is not set. Add it to .env, "
                             "or use --no-judge, or pass --judge-model gemini-3.7-flash.")
        try:
            import anthropic
        except ImportError:
            raise SystemExit("The judge needs the anthropic package: pip install anthropic")
        return anthropic.Anthropic(max_retries=4)
    import os
    from google import genai
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def _parse_json(text: str):
    """Claude returns the JSON as plain text, sometimes inside a code fence: take the outermost braces."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"judge reply had no JSON object: {text[:120]!r}")
    return json.loads(text[start:end + 1])


def _ask_judge(client, model: str, prompt: str) -> dict:
    if model.startswith("claude"):
        # No temperature: Sonnet 5.5 rejects non-default sampling values. Thinking stays at its default;
        # effort "medium" is enough for a rubric grade.
        resp = client.messages.create(
            model=model, max_tokens=8000,
            output_config={"effort": "medium"},
            messages=[{"role": "user", "content": prompt + "\nReply with the JSON object only."}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"judge declined to answer ({getattr(resp.stop_details, 'category', None)})")
        return _parse_json("".join(b.text for b in resp.content if b.type == "text"))
    if model.startswith("grok"):
        resp = client.post(XAI_URL, timeout=180, json={
            "model": model, "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt + "\nReply with the JSON object only."}],
        })
        resp.raise_for_status()
        return _parse_json(resp.json()["choices"][0]["message"]["content"])
    config = types.GenerateContentConfig(temperature=0, response_mime_type="application/json")
    resp = client.models.generate_content(model=model, contents=[prompt], config=config)
    return json.loads(resp.text)

def number_check(answer: str, source_text: str) -> dict:
    """Flags numbers in the answer that cannot be found in source_text (the tool output)."""
    checked, unsupported = unsupported_numbers(answer, source_text)
    return {"numbers_checked": checked, "unsupported_numbers": unsupported}


JUDGE_PROMPT = """You are grading an answer from a financial research agent. It answers from one SEC filing, a technical signals table (price, moving averages, RSI, volume) and a watchlist tool.
Use ONLY the TOOL OUTPUT below as the source of truth for what the agent saw. EXPECTED ANSWER and GRADER NOTE come from a human who read the filing.

QUESTION: {question}
ANSWERABLE WITH THE AGENT'S TOOLS: {answerable}
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
- "correctness": for an answerable question one of "correct", "partial", "incorrect", "declined"; for a question that is not answerable one of "declined" (it says it does not have the information or cannot do it) or "answered" (it gives an answer anyway).
- "grounded": true if every factual claim and number in the answer is stated in the tool output; false if anything comes from outside knowledge or is calculated and presented as reported.
- "period_error": true if any figure is attributed to the wrong period (for example last year's value given as this quarter's).
- "citations_supported": the tool output labels each filing chunk "[chunk N]" and the signals table "[signals]". true if every citation tag in the answer points at a chunk that states the claim it is attached to and every factual sentence has one; false otherwise (also false for an answer with factual claims and no citations; true for a plain refusal or a watchlist confirmation).
- "unsupported_claims": list of short strings, each a claim or number not supported by the tool output (empty list if none).
- "reason": one sentence explaining the grade.
"""


def judge_answer(client, model: str, q: dict, answer: str, tool_output: str,
                 max_tool_chars: int = 30000) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=q["question"],
        answerable="yes" if q.get("answerable") else "no",
        expected=q.get("expected_answer") or "(none: the agent should decline)",
        note=q.get("note") or "(none)",
        tool_output=(tool_output or "(the agent called no tools)")[:max_tool_chars],
        answer=answer,
    )
    last_err = None
    for attempt in range(4):
        try:
            data = _ask_judge(client, model, prompt)
            if isinstance(data, list) and data:
                data = data[0]
            return {
                "correctness": str(data.get("correctness", "")).lower(),
                "grounded": bool(data.get("grounded")),
                "period_error": bool(data.get("period_error")),
                "citations_supported": bool(data.get("citations_supported")),
                "unsupported_claims": list(data.get("unsupported_claims") or []),
                "reason": str(data.get("reason", "")),
            }
        except Exception as e:
            last_err = e
            if attempt < 3:
                time.sleep(5 * 2 ** attempt)  # 5s, 10s, 20s: 503 "high demand" spikes usually pass
    return {"judge_error": str(last_err)}


def summarize(rows: list[dict]) -> dict:
    """Per-category and overall metrics from the per-answer rows built in run_eval.main()."""
    def block(rs):
        out = {"answers": len(rs)}
        if not rs:
            return out
        answerable = [r for r in rs if r["answerable"] is True]
        picked = [r for r in rs if r.get("tool_ok") is not None]
        if picked:
            ok = sum(bool(r["tool_ok"]) for r in picked)
            out["tool_choice_correct"] = f"{ok}/{len(picked)} ({100 * ok / len(picked):.0f}%)"
        out["sec_tool_called"] = sum(r["called"] for r in rs)
        with_snip = [r for r in answerable if r["snippet_in_chunks"] is not None]
        if with_snip:
            out["chunks_contain_evidence"] = f"{sum(bool(r['snippet_in_chunks']) for r in with_snip)}/{len(with_snip)}"
        out["answers_with_unsupported_numbers"] = sum(1 for r in rs if r["number_check"]["unsupported_numbers"])
        cited = [r for r in rs if r.get("citations")]  # rows from older traces have no citation check
        if cited:
            out["citations_ok"] = f"{sum(r['citations']['ok'] for r in cited)}/{len(cited)}"
            out["answers_with_invalid_citations"] = sum(1 for r in cited if r["citations"]["invalid"])
            out["answers_with_uncited_numbers"] = sum(1 for r in cited if r["citations"]["uncited_claims"])
            out["answers_rewritten_for_citations"] = sum(1 for r in cited if r.get("rewritten"))
            ev = [r for r in cited if r["answerable"] is True and r.get("cited_evidence") is not None]
            if ev:
                out["cited_chunks_contain_evidence"] = f"{sum(bool(r['cited_evidence']) for r in ev)}/{len(ev)}"
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
            jc = [r for r in judged if "citations_supported" in r["judge"]]
            if jc:
                out["judge_citations_supported"] = f"{sum(r['judge']['citations_supported'] for r in jc)}/{len(jc)}"
        out["judge_errors"] = sum(1 for r in rs if r.get("judge") and "judge_error" in r["judge"])
        return out

    cats = sorted({r["category"] for r in rows})
    return {"overall": block(rows), "by_category": {c: block([r for r in rows if r["category"] == c]) for c in cats}}
