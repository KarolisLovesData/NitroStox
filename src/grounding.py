"""Citations for agent answers, and the checks that keep them honest.

The SEC tool labels every chunk it returns, e.g. "[chunk 12]", and the technical-signals
tool labels its table "[signals]". The agent must put those tags after every sentence that
states a fact or a number. check_citations() then verifies, without any model call:
  - every cited chunk was actually returned by a tool in this turn,
  - every sentence with a number carries a citation,
  - every number in a cited sentence appears in the chunks that sentence cites.
The app uses it to send the agent back for one rewrite, and the eval reports it.

extract_numbers() also backs the eval's number_check (eval_grading.py imports it from here).
"""
import re

SIGNALS_TAG = "signals"
_CITE_RE = re.compile(r"\[(?:chunks?\s+(\d+(?:\s*(?:,|and|&)\s*(?:chunk\s+)?\d+)*)|(signals))\]", re.I)
_CHUNK_HEAD_RE = re.compile(r"^\[chunk (\d+)\](?: \(similarity [\d.]+\))?\n", re.M)

NOT_IN_FILING = "The filing doesn't say."


# ---------- numbers ----------

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


def is_noise(n: dict) -> bool:
    """Years, list markers and other small bare integers carry no checkable claim."""
    if n["unit"]:
        return False
    v = n["value"]
    return v.is_integer() and (v < 10 or 1990 <= v <= 2100)


_DATE_RE = re.compile(r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+\d{1,2}\b", re.I)


def strip_dates(text: str) -> str:
    """'June 30, 2026' -> ' , 2026' so the day is not checked as a figure."""
    return _DATE_RE.sub(" ", text or "")


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= REL_TOL * max(abs(a), abs(b), 1e-9)


def unsupported_numbers(text: str, source_text: str) -> tuple[int, list[str]]:
    """(numbers checked, numbers in text that cannot be found in source_text)."""
    source = extract_numbers(source_text)
    source_any = [s["value"] for s in source]  # bare table cells are usually in millions
    source_pct = [s["value"] for s in source if s["unit"] == "pct"]
    checked, missing = 0, []
    for n in extract_numbers(strip_dates(text)):
        if is_noise(n):
            continue
        checked += 1
        if n["unit"] == "pct":
            ok = any(abs(n["value"] - s) <= PCT_ABS_TOL for s in source_pct)
        else:
            ok = any(_close(n["value"], s) for s in source_any)
        if not ok and n["raw"] not in missing:
            missing.append(n["raw"])
    return checked, missing


# ---------- tool output labels ----------

def format_chunks(rows) -> str:
    """SEC tool output: one labeled block per chunk. rows: iterable of (chunk_index, score, text)."""
    def head(i, s):
        # similarity is missing only when the query embedding failed and the keyword list answered alone
        return f"[chunk {int(i)}]" + (f" (similarity {float(s):.3f})" if s is not None and s == s else "")

    return "\n\n---\n\n".join(f"{head(i, s)}\n{t}" for i, s, t in rows)


def parse_tool_sources(tool_output: str) -> dict[str, str]:
    """{'12': chunk text, ...} from an SEC tool output, or {'signals': table} from the signals tool."""
    if tool_output.startswith(f"[{SIGNALS_TAG}]"):
        return {SIGNALS_TAG: tool_output}
    heads = list(_CHUNK_HEAD_RE.finditer(tool_output))
    out = {}
    for k, m in enumerate(heads):
        end = heads[k + 1].start() if k + 1 < len(heads) else len(tool_output)
        out[m.group(1)] = tool_output[m.end():end].removesuffix("\n\n---\n\n")
    return out


# ---------- checking an answer ----------

def cited_ids(text: str) -> list[str]:
    ids = []
    for m in _CITE_RE.finditer(text or ""):
        if m.group(2):
            ids.append(SIGNALS_TAG)
        else:
            ids.extend(re.findall(r"\d+", m.group(1)))
    return list(dict.fromkeys(ids))


def _sentences(answer: str) -> list[str]:
    """Sentences and list items. A citation right after the full stop belongs to that sentence."""
    out = []
    for line in (answer or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?!\[(?:chunk|signals))", line)
        out.extend(p for p in parts if p.strip())
    return out


def _has_claim_number(sentence: str) -> bool:
    return any(not is_noise(n) for n in extract_numbers(strip_dates(_CITE_RE.sub(" ", sentence))))


def check_citations(answer: str, sources: dict[str, str]) -> dict:
    """Checks the citations in answer against sources ({id: text} the tools returned this turn).

    Returns:
      cited             ids the answer cites
      invalid           cited ids no tool returned (made-up or from an earlier turn)
      uncited_claims    sentences with a number but no citation
      unsupported       [{"sentence", "numbers"}] numbers that are not in the chunks the sentence cites
      claims            sentences with a number (the denominator for the rates above)
      ok                True when nothing above was found
    """
    cited = cited_ids(answer)
    invalid = [c for c in cited if c not in sources]
    uncited, unsupported, claims = [], [], 0
    for s in _sentences(answer):
        if not _has_claim_number(s):
            continue
        claims += 1
        ids = cited_ids(s)
        if not ids:
            uncited.append(s)
            continue
        text = "\n".join(sources[i] for i in ids if i in sources)
        _, missing = unsupported_numbers(_CITE_RE.sub(" ", s), text)
        if missing:
            unsupported.append({"sentence": s, "numbers": missing})
    return {
        "cited": cited,
        "invalid": invalid,
        "uncited_claims": uncited,
        "unsupported": unsupported,
        "claims": claims,
        "ok": not (invalid or uncited or unsupported),
    }


def repair_request(check: dict) -> str:
    """The follow-up message that sends the agent back to fix its citations."""
    problems = []
    if check["invalid"]:
        problems.append("You cited sources that no tool returned in this turn: "
                        + ", ".join(f"[chunk {c}]" if c != SIGNALS_TAG else "[signals]" for c in check["invalid"]))
    for s in check["uncited_claims"][:6]:
        problems.append(f'No citation: "{s}"')
    for u in check["unsupported"][:6]:
        problems.append(f'{", ".join(u["numbers"])} is not in the source cited for: "{u["sentence"]}"')
    return (
        "Your answer breaks the citation rules:\n- " + "\n- ".join(problems) + "\n\n"
        "Rewrite the whole answer. Keep only statements the tool output supports, each followed by the "
        "tag of the chunk that states it. Copy numbers exactly as the cited chunk shows them, with their "
        "period. Remove anything you calculated or know from elsewhere. If nothing is left, reply: "
        f'"{NOT_IN_FILING}" You may search the filing again first.'
    )


def warning_note(check: dict) -> str:
    """Shown under an answer that still fails after the rewrite."""
    n = len(check["uncited_claims"]) + len(check["unsupported"]) + len(check["invalid"])
    return (f"\n\n> ⚠️ {n} statement(s) above could not be matched to the cited filing text. "
            "Check the sources before relying on them.")
