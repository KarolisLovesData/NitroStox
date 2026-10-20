"""Small BM25 keyword index for SEC filing chunks.

Sits next to the vector index (sec_rag.py) so exact terms such as line items,
years, dollar figures, tickers and form names ("10-Q") are not lost by embeddings.
No Gemini import here, so it can be tested offline.
"""
import math
import re
from collections import Counter

# Text-to-SQL STOP_WORDS plus common question words. Finance words (net, income,
# sales, quarter, months, ended) are deliberately NOT stop words.
STOP_WORDS = {
    "what", "is", "the", "are", "by", "for", "across", "compare", "show", "return",
    "get", "rank", "list", "and", "or", "in", "of", "to", "with", "whole", "dataset",
    "total", "average", "each", "all",
    "a", "an", "was", "were", "did", "does", "do", "how", "much", "many", "which",
    "during", "from", "on", "at", "as", "its", "it", "this", "that", "be", "been",
    "has", "have", "had", "company", "filing", "report",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.&-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercased keyword tokens with Markdown noise removed and numbers normalized.

    "$1,234.5", "(1,234.5)" and "1234.5" all become "1234.5"; "12.5%" becomes "12.5".
    Compound tokens ("10-q", "s&p") are kept and their parts are added too.
    """
    s = (text or "").lower()
    # Markdown noise: table pipes, bold/italic markers, headings, table separator runs
    s = s.replace("|", " ").replace("**", " ").replace("__", " ").replace("#", " ")
    s = re.sub(r"[-:]{2,}", " ", s)
    # Numbers: 1,234 -> 1234, drop $ and %, (1234) -> 1234
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
    s = s.replace("$", " ").replace("%", " ")
    s = re.sub(r"\((\d[\d.]*)\)", r" \1 ", s)

    tokens = []
    for tok in _TOKEN_RE.findall(s):
        tok = tok.rstrip(".")
        if not tok:
            continue
        if tok not in STOP_WORDS and len(tok) > 1:
            tokens.append(tok)
        parts = re.split(r"[.&-]", tok)
        # A decimal like 12.5 is one number, not two parts
        if len(parts) > 1 and not re.fullmatch(r"\d+\.\d+", tok):
            tokens.extend(p for p in parts if p and p not in STOP_WORDS)
    return tokens


class BM25Index:
    """Okapi BM25 over a fixed list of chunks. chunk_index = position in the list."""

    def __init__(self, chunks: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(tokenize(c)) for c in chunks]
        self.doc_lengths = [sum(tc.values()) for tc in self.term_counts]
        self.n_docs = len(chunks)
        self.avgdl = (sum(self.doc_lengths) / self.n_docs) if self.n_docs else 0.0
        df = Counter()
        for tc in self.term_counts:
            df.update(tc.keys())
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, top_n: int = 20) -> list[tuple[int, float]]:
        """Returns (chunk_index, score) for chunks with score > 0, best first."""
        terms = list(dict.fromkeys(tokenize(query)))
        terms = [t for t in terms if t in self.idf]
        if not terms or not self.avgdl:
            return []

        scored = []
        for i, tc in enumerate(self.term_counts):
            dl_norm = self.k1 * (1 - self.b + self.b * self.doc_lengths[i] / self.avgdl)
            score = 0.0
            for t in terms:
                tf = tc.get(t, 0)
                if tf:
                    score += self.idf[t] * tf * (self.k1 + 1) / (tf + dl_norm)
            if score > 0:
                scored.append((i, score))

        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_n]
