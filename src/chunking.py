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


_RULE_RE = re.compile(r"^[\s" + re.escape(_BOX_CHARS) + r"\-_=]+$")


def _is_rule(line: str) -> bool:
    """A separator line made only of dashes or box characters, e.g. the line under a table header."""
    return bool(line.strip()) and bool(_RULE_RE.match(line))


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
            seen_data = False
            while j < n:
                l = lines[j]
                if is_table_line(l) and _has_data(l):
                    seen_data = True
                if not l.strip():
                    if j + 1 < n and not lines[j + 1].strip() and seen_data:
                        break
                    j += 1
                    continue
                if _is_rule(l):  # the line under the column headers, or between sections of a table
                    j += 1
                    continue
                # A label row ("Operating activities:", "Adjustments to reconcile ...:") sits right
                # against the rows and has no figures; a heading after a blank line ("Net Sales")
                # ends the table instead.
                # After a blank line it must not end in a full stop (a sentence) and must not be followed
                # by new column headers (then it titles the next table, e.g. "Net Sales").
                nxt = next((x for x in lines[j + 1:j + 6] if x.strip()), "")
                label_row = (not _has_data(l) and len(l.strip()) <= 200
                             and (lines[j - 1].strip() or l.rstrip().endswith(":")
                                  or (not l.rstrip().endswith(".") and not is_header_line(nxt)))
                             and _table_follows(lines, j, look=2))
                # Before the first row of figures, "ASSETS" or "Current assets:" may sit after a blank
                # line under the column headers; they still belong to the table.
                if not label_row and not seen_data:
                    label_row = (not _has_data(l) and len(l.strip()) <= 80 and _table_follows(lines, j, look=3))
                if is_table_line(l) or is_header_line(l) or label_row:
                    last_table = j
                    j += 1
                    continue
                break
            rows = [r.rstrip() for r in lines[start:last_table + 1] if not _is_rule(r)]
            # Caption: the short lines just above the table that are not headings or prose
            # ("AMAZON.COM, INC.", "(in millions, except per share data)", "(unaudited)"). They are
            # often separated by blank lines, so take up to 4 of the preceding blocks.
            flush_para()
            caption = []
            while (blocks and len(caption) < 4 and blocks[-1]["kind"] == "prose"
                   and _caption_like(blocks[-1]["text"]) and not _SECTION_RE.match(blocks[-1]["text"].strip())):
                caption = [l.strip() for l in blocks.pop()["text"].splitlines() if l.strip()] + caption
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
    """Paragraph- and table-aware chunks, each prefixed with its section.

    Markdown input (filing.markdown()) goes to the simpler chunk_markdown(); plain text falls
    back to the layout-guessing chunker below.
    """
    if looks_like_markdown(text):
        return chunk_markdown(text, chunk_size, overlap)
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
        budget = (table_limit if is_table else chunk_size) - len(lab) - 1
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

    table_limit = int(chunk_size * 1.75)  # wide statements: a header can take a third of a chunk
    prev_head, prose_gap = "", 99  # header of the last table, and how many prose blocks came since
    for b in _blocks(text):
        if b["kind"] == "prose":
            par = b["text"].strip()
            if _is_heading(par):
                prev_head = ""
                head = par.strip()
                if _SECTION_RE.match(head):
                    section, subsection = head[:160], ""
                else:
                    subsection = head[:160]
                continue
            prose_gap += 1
            budget = chunk_size - len(label()) - 1
            for piece in _split_prose(par, budget):
                add(piece)
            continue

        caption = "\n".join(b["caption"])
        for g, (header, body) in enumerate(_split_table(b["rows"])):
            head_text = "\n".join(x for x in [caption, "\n".join(header)] if x).strip("\n")
            if not header and not caption and prev_head and prose_gap <= 2 and g == 0:
                head_text = prev_head  # same table, interrupted by a line of text or a page break
            elif header:
                prev_head = head_text
            prose_gap = 0
            budget = table_limit - len(label()) - 1 - len(head_text) - 30
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


# ---------- Markdown filings (the simpler path) ----------
# edgartools' filing.markdown() writes tables as "| a | b |" rows and headings as "#" lines, so
# nothing has to be guessed: a table is a run of lines starting with "|", and its header is the
# rows above the first row with a real number in it.

_MD_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_MD_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_MD_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.*?)\s*#*\s*$")
_MD_TAG_RE = re.compile(r"^\s*<u>.*</u>\s*$", re.I)  # page-navigation lines such as "Table of Contents"


def _md_plain(text: str) -> str:
    """Drops Markdown noise from prose and captions: **bold**, backslash escapes, extra spaces."""
    return re.sub(r"\s+", " ", re.sub(r"\\(.)", r"\1", text.replace("**", ""))).strip()


def looks_like_markdown(text: str) -> bool:
    """True when the text has Markdown tables (a header separator row like |---|---|)."""
    return any(_MD_SEP_RE.match(l) and "|" in l for l in text.splitlines())


def chunk_markdown(text: str, chunk_size: int = 2000, overlap: int = 300) -> list[str]:
    """Chunks a Markdown filing: whole tables with their header rows, paragraphs, section prefix."""
    lines = text.replace("\r\n", "\n").split("\n")
    heads: list[str] = []        # current heading path, e.g. ["Item 2. MD&A", "Net Sales"]
    chunks: list[str] = []
    cur: list[str] = []          # pieces of the chunk being built
    cur_label = ""

    def label() -> str:
        return f"[Section: {' > '.join(heads[-2:])}]" if heads else ""

    def flush():
        nonlocal cur
        if cur:
            body = "\n\n".join(cur)
            chunks.append(f"{cur_label}\n{body}" if cur_label else body)
        cur = []

    def add(piece: str, keep_tail: bool = True):
        nonlocal cur_label
        lab = label()
        size = len(lab) + sum(len(p) + 2 for p in cur) + len(piece)
        if cur and (size > chunk_size or lab != cur_label):
            tail = cur[-1] if keep_tail and lab == cur_label and len(cur[-1]) <= overlap \
                and not cur[-1].startswith("|") else None
            flush()
            if tail and len(lab) + len(tail) + len(piece) + 4 <= chunk_size:
                cur.append(tail)
        cur_label = lab
        cur.append(piece)

    i, n = 0, len(lines)
    para: list[str] = []
    pending: list[str] = []  # the last short paragraphs: they may be the title of a table below them

    def add_pending():
        for p_ in pending:
            for piece in _split_prose(p_, chunk_size - len(label()) - 2):
                add(piece)
        pending.clear()

    def flush_para():
        if para:
            text_ = _md_plain(" ".join(l.strip() for l in para))
            para.clear()
            if not text_:
                return
            if len(text_) > 250:  # a real paragraph, not a title
                add_pending()
            elif len(pending) >= 4:
                first = pending.pop(0)
                for piece in _split_prose(first, chunk_size - len(label()) - 2):
                    add(piece)
            pending.append(text_)

    while i < n:
        line = lines[i]
        h = _MD_HEADING_RE.match(line)
        if h:
            flush_para()
            add_pending()
            level, title = len(h.group(1)), _md_plain(h.group(2))
            if re.match(r"PART\s+[IVX]+\b", title, re.I):
                level = 1  # a filing's PART starts a new top level, whatever "#" depth the converter used
            heads[:] = heads[:level - 1] + [title[:160]]
            i += 1
            continue
        if _MD_ROW_RE.match(line):
            flush_para()
            rows = []
            while i < n and _MD_ROW_RE.match(lines[i]):
                rows.append(lines[i].rstrip())
                i += 1
            # Header = rows before the first one with a real number (years and dates don't count)
            k = 0
            while k < len(rows) and (_MD_SEP_RE.match(rows[k]) or not _has_data(rows[k])):
                k += 1
            header, body = rows[:k], rows[k:]
            caption_lines = pending[-4:]
            del pending[-4:]
            add_pending()
            head_text = "\n".join(caption_lines + header)
            budget = max(chunk_size - len(label()) - len(head_text) - 30, chunk_size // 3)
            flush()  # every table starts its own chunk, so two statements are never mixed
            part: list[str] = []
            first = True
            for r in body + [None]:
                if r is None or (part and sum(len(x) + 1 for x in part) + len(r) > budget):
                    if part or first:
                        add("\n".join(([head_text] if head_text else [])
                                      + ([] if first else ["(table continued)"]) + part), keep_tail=False)
                    part, first = [], False
                if r is not None:
                    part.append(r)
            continue
        if not line.strip():
            flush_para()
        elif _MD_TAG_RE.match(line):
            flush_para()
        else:
            para.append(line)
        i += 1
    flush_para()
    add_pending()
    flush()
    return chunks
