"""Table-aware chunking for SEC filing text.

The old chunker cut the filing every 2,000 characters. A financial table usually spans more
than one cut, so the rows ended up in a chunk without the header that says which column is
which period ("Three Months Ended June 30, 2025 | 2026"). The 2026-10-04 eval caught the
agent reading last year's net income as this quarter's because of that.

This chunker:
  - splits the text into paragraphs and tables instead of fixed windows,
  - repeats a table's caption and column header lines at the top of every chunk that holds
    its rows (marked "(table continued)" after the first),
  - starts every chunk with the section it comes from, e.g.
    "[Section: Item 2. Management's Discussion and Analysis > Net Sales]".

It works on the plain text edgar's filing.text() and the BeautifulSoup fallback produce:
table cells are separated by runs of spaces (or box-drawing characters), prose is not.
Line spacing inside rows is kept as is, so column positions still line up.
"""
import re

_BOX_CHARS = "│┃┆┊|─━┄┈┌┐└┘├┤┬┴┼╭╮╯╰═║╔╗╚╝╠╣╦╩╬"
_BOX_RE = re.compile(f"[{re.escape(_BOX_CHARS)}]")
_GAP_RE = re.compile(r"\S(?: {2,}|\t)\S")  # two cells separated by 2+ spaces or a tab
_NUM_CELL_RE = re.compile(r"^\(?\$?\(?-?\d[\d,]*(?:\.\d+)?\)?%?\)?$")
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_PERIOD_WORDS_RE = re.compile(
    r"\b(months ended|weeks ended|year ended|quarter ended|as of|december 31|june 30|march 31|"
    r"september 30|fiscal|q[1-4])\b", re.I)
_SECTION_RE = re.compile(r"^(PART\s+I{1,3}V?\b|Item\s+\d+[A-Z]?\.)", re.I)


def _clean(line: str) -> str:
    return _BOX_RE.sub(" ", line).rstrip()


def _cells(line: str) -> list[str]:
    return [c for c in re.split(r" {2,}|\t", _clean(line).strip()) if c]


def _numeric_tokens(line: str) -> list[str]:
    toks = _clean(line).replace("$", " ").split()
    return [t for t in toks if _NUM_CELL_RE.match(t) or t in ("—", "–", "-")]


def is_table_line(line: str) -> bool:
    if not line.strip():
        return False
    if _BOX_RE.search(line) and len(_BOX_RE.sub("", line).strip()) > 0:
        return True
    c = _clean(line).strip()
    cells = _cells(line)
    if _GAP_RE.search(c) and len(cells) >= 2:
        # Space-aligned columns. Prose with a double space after a period is not a table: a row
        # needs a number cell, a year/period header, or several short cells.
        if any(_NUM_CELL_RE.match(x.replace("$", "").replace(" ", "")) or x in ("—", "–", "-") for x in cells):
            return True
        if is_header_line(c):
            return True
        if len(cells) >= 3 and sum(map(len, cells)) / len(cells) < 40:
            return True
    # "Net income $ 18,164 62,647 $ 35,291 $ 92,902" after the spacing was collapsed:
    # a short label followed by numbers, ending in a number (wrapped prose ends in words).
    toks = c.replace("$", " ").split()
    nums = _numeric_tokens(c)
    tail = toks[-3:]
    return (len(nums) >= 2 and bool(toks) and toks[-1] in nums
            and sum(t in nums for t in tail) >= 2 and len(c) < 160)


def is_header_line(line: str) -> bool:
    """Column header rows: period words, or only years ('2025   2026   2025   2026')."""
    c = _clean(line).strip()
    if not c:
        return False
    nums = _numeric_tokens(c)
    if nums and all(_YEAR_RE.match(n.strip("()$")) for n in nums):
        words = [t for t in c.replace(",", " ").split() if not _YEAR_RE.match(t)]
        return len(" ".join(words)) < 60
    return bool(_PERIOD_WORDS_RE.search(c)) and not _has_data(c)


_DATE_RE = re.compile(r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+\d{1,2}\b", re.I)


def _has_data(line: str) -> bool:
    """A data row has at least one number that is not a year or part of a date ('June 30,')."""
    return any(not _YEAR_RE.match(n.strip("()$,")) for n in _numeric_tokens(_DATE_RE.sub(" ", line)))


def _caption_like(par: str) -> bool:
    lines = [l.strip() for l in par.splitlines() if l.strip()]
    if not lines or len(lines) > 5 or _is_heading(par):
        return False
    return all(len(l) <= 110 and not _SECTION_RE.match(l) and (not l.endswith(".") or len(l) < 30)
               for l in lines)


def _is_heading(par: str) -> bool:
    lines = [l for l in par.splitlines() if l.strip()]
    if len(lines) != 1:
        return False
    t = lines[0].strip()
    if _SECTION_RE.match(t):
        return True
    if len(t) > 90 or t.endswith((".", ",", ";", ":")) or _has_data(t):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'’\-]*", t)
    if not words:
        return False
    caps = sum(w[0].isupper() for w in words if len(w) > 3 or w.isupper())
    long_words = sum(1 for w in words if len(w) > 3 or w.isupper())
    return long_words > 0 and caps / long_words >= 0.8


def _blocks(text: str) -> list[dict]:
    """Splits text into {'kind': 'prose'|'table', ...} blocks in document order."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks, i, n = [], 0, len(lines)
    para: list[str] = []

    def flush_para():
        if any(l.strip() for l in para):
            blocks.append({"kind": "prose", "text": "\n".join(para).strip("\n")})
        para.clear()

    while i < n:
        if is_table_line(lines[i]) or (is_header_line(lines[i]) and _table_follows(lines, i)):
            # A table: table lines, header lines, sub-labels ("Operating expenses:") and single blank
            # lines between them, until two blank lines or a prose line.
            start = i
            j = i
            last_table = i
            while j < n:
                l = lines[j]
                if not l.strip():
                    if j + 1 < n and not lines[j + 1].strip():
                        break
                    j += 1
                    continue
                # A short label row ("Operating expenses:") sits right against the rows; a heading
                # after a blank line ("Net Sales") ends the table instead.
                label_row = len(l.strip()) < 70 and (l.rstrip().endswith(":") or (
                    lines[j - 1].strip() and j + 1 < n and is_table_line(lines[j + 1])))
                if is_table_line(l) or is_header_line(l) or label_row:
                    last_table = j
                    j += 1
                    continue
                break
            rows = lines[start:last_table + 1]
            # Caption: short lines just above the table that are not prose ("CONSOLIDATED STATEMENTS
            # OF OPERATIONS", "(in millions, except per share data)", "(unaudited)").
            flush_para()
            caption = []
            if blocks and blocks[-1]["kind"] == "prose" and _caption_like(blocks[-1]["text"]):
                caption = [l.strip() for l in blocks.pop()["text"].splitlines() if l.strip()]
            if sum(1 for r in rows if _has_data(r)) >= 1:
                blocks.append({"kind": "table", "caption": caption, "rows": rows})
            else:  # header-looking lines with no data: keep as text
                blocks.append({"kind": "prose", "text": "\n".join(caption + rows)})
            i = last_table + 1
            continue
        if not lines[i].strip():
            flush_para()
        else:
            para.append(lines[i])
        i += 1
    flush_para()
    return blocks


def _table_follows(lines: list[str], i: int, look: int = 4) -> bool:
    seen = 0
    for k in range(i + 1, len(lines)):
        l = lines[k]
        if not l.strip():
            continue
        if is_table_line(l):
            return True
        seen += 1
        if seen >= look:
            return False
    return False


def _split_table(rows: list[str]) -> list[tuple[list[str], list[str]]]:
    """Groups rows under the header lines above them. A header after data starts a new group
    (segment tables often stack 'Three Months Ended' and 'Six Months Ended' blocks)."""
    groups, header, body = [], [], []
    for r in rows:
        if is_header_line(r) and not _has_data(r):
            if body:
                groups.append((header, body))
                header, body = [], []
            header.append(r)
        elif not body and not _has_data(r) and r.strip() and not header:
            header.append(r)  # column titles that carry no period words, above the first number
        else:
            body.append(r)
    if body or header:
        groups.append((header, body))
    return groups


def _split_prose(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        while len(sent) > limit:  # one giant "sentence" (a list without periods)
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(sent[:limit])
            sent = sent[limit:]
        if len(cur) + len(sent) + 1 > limit and cur:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip() if cur else sent
    if cur:
        parts.append(cur)
    return parts


def chunk_filing_text(text: str, chunk_size: int = 2000, overlap: int = 300) -> list[str]:
    """Paragraph- and table-aware chunks, each prefixed with its section."""
    section, subsection = "", ""
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    cur_label = None
    last_was_table = False

    def label():
        if section and subsection and subsection != section:
            return f"[Section: {section} > {subsection}]"
        if section or subsection:
            return f"[Section: {section or subsection}]"
        return ""

    def flush(carry: bool = True):
        nonlocal cur, cur_len, cur_label, last_was_table
        body = "\n\n".join(cur).strip()
        if body:
            chunks.append(f"{cur_label}\n{body}" if cur_label else body)
        tail = cur[-1] if carry and cur and len(cur[-1]) <= overlap and not last_was_table else None
        cur, cur_len, cur_label, last_was_table = [], 0, None, False
        if tail:
            cur.append(tail)
            cur_len = len(tail)

    def add(piece: str, is_table: bool = False):
        nonlocal cur_len, cur_label, last_was_table
        lab = label()
        budget = chunk_size - len(lab) - 1
        if cur and (cur_len + len(piece) + 2 > budget or (cur_label is not None and lab != cur_label)):
            flush(carry=not is_table and lab == cur_label)
            if cur and cur_len + len(piece) + 2 > budget:  # no room for the overlap paragraph
                cur.clear()
                cur_len = 0
        if cur_label is None:
            cur_label = lab
        cur.append(piece)
        cur_len += len(piece) + 2
        last_was_table = is_table

    for b in _blocks(text):
        if b["kind"] == "prose":
            par = b["text"].strip()
            if _is_heading(par):
                head = par.strip()
                if _SECTION_RE.match(head):
                    section, subsection = head[:100], ""
                else:
                    subsection = head[:100]
                continue
            budget = chunk_size - len(label()) - 1
            for piece in _split_prose(par, budget):
                add(piece)
            continue

        caption = "\n".join(b["caption"])
        for header, body in _split_table(b["rows"]):
            head_text = "\n".join(x for x in [caption, "\n".join(header)] if x).strip("\n")
            budget = chunk_size - len(label()) - 1 - len(head_text) - 30
            budget = max(budget, chunk_size // 3)  # a huge header still leaves room for rows
            pieces, cur_rows, size = [], [], 0
            for r in body:
                if cur_rows and size + len(r) + 1 > budget:
                    pieces.append(cur_rows)
                    cur_rows, size = [], 0
                cur_rows.append(r)
                size += len(r) + 1
            if cur_rows:
                pieces.append(cur_rows)
            for k, rows in enumerate(pieces):
                parts = [head_text] if head_text else []
                if k:
                    parts.append("(table continued)")
                parts.append("\n".join(rows).strip("\n"))
                add("\n".join(parts), is_table=True)
    flush(carry=False)
    return chunks
