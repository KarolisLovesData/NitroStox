"""Offline checks for the keyword index and hybrid search. No Gemini calls, no pytest.

  python check_keyword_search.py                        # tokenizer, BM25 and fusion checks
  python check_keyword_search.py --filing filing_AMZN.md  # also show the top 3 keyword-search chunks
                                                        # for 5 eval questions (keyword only)
"""
import argparse
import hashlib
import json
import os

import numpy as np

from keyword_search import BM25Index, tokenize


def check_tokenize():
    toks = tokenize("Net sales were $143,313 million (12.5%) in Q3 2024; see Form 10-Q")
    for t in ["143313", "12.5", "2024", "10-q", "10", "q", "net", "sales"]:
        assert t in toks, f"missing {t!r} in {toks}"
    assert "were" not in toks and "in" not in toks, toks
    # "$1,234.5", "(1,234.5)" and "1234.5" are the same token
    assert tokenize("$1,234.5") == tokenize("(1,234.5)") == tokenize("1234.5") == ["1234.5"]
    # Markdown table noise is removed
    assert tokenize("| **Net income** | --- | 2,345 |") == ["net", "income", "2345"]
    print("✅ tokenize")


TOY_CHUNKS = [
    "[Section: Results] Net sales increased 11% to $158.9 billion in 2024.",
    "[Section: Income] Net income was $15.3 billion in Q3 2024, compared with $9.9 billion.",
    "[Section: Risk] Supply chain risks and foreign exchange could affect income in 2023.",
    "[Section: Legal] We are subject to legal proceedings and claims.",
]


def check_bm25():
    idx = BM25Index(TOY_CHUNKS)
    hits = idx.search("net income 2024", 10)
    assert hits and hits[0][0] == 1, hits
    assert idx.search("what is the total of all", 10) == []
    assert idx.search("blockchain", 10) == []
    print("✅ BM25Index")


def fake_vector(text: str, dim: int = 64) -> np.ndarray:
    """Hashed bag-of-words vector, a stand-in for a Gemini embedding."""
    v = np.zeros(dim, dtype=np.float32)
    for t in tokenize(text):
        v[int(hashlib.md5(t.encode()).hexdigest(), 16) % dim] += 1.0
    v[0] += 0.01  # never an all-zero vector
    return v


def check_fusion():
    os.environ.setdefault("GEMINI_API_KEY", "dummy")  # sec_rag needs one at import; no call is made
    import sec_rag

    state = {"fail": False}

    def fake_query_embedding(text):
        if state["fail"]:
            raise RuntimeError("fake embedding failure")
        return tuple(fake_vector(text))

    sec_rag._cached_query_embedding = fake_query_embedding
    rag = sec_rag.SECVectorRAG("TEST")
    chunks = TOY_CHUNKS
    emb = np.vstack([fake_vector(c) for c in chunks])
    bm25 = BM25Index(chunks)
    q = "net income 2024"

    vec = rag.vector_search_in_memory(q, chunks, emb, top_k=3)
    via_mode = rag.search_in_memory(q, chunks, emb, bm25, top_k=3, mode="vector")
    assert vec.equals(via_mode), (vec, via_mode)

    hyb = rag.search_in_memory(q, chunks, emb, bm25, top_k=3, mode="hybrid")
    assert {"chunk_index", "chunk_text", "similarity_score"} <= set(hyb.columns), hyb.columns
    assert int(vec.iloc[0]["chunk_index"]) == 1 and int(hyb.iloc[0]["chunk_index"]) == 1, hyb
    assert hyb.iloc[0]["vector_rank"] == 1 and hyb.iloc[0]["keyword_rank"] == 1, hyb
    assert abs(hyb.iloc[0]["rrf_score"] - 2 / 61) < 1e-6, hyb
    assert abs(hyb.iloc[0]["similarity_score"] - vec.iloc[0]["similarity_score"]) < 1e-6

    kw = rag.search_in_memory(q, chunks, emb, bm25, top_k=3, mode="keyword")
    assert [int(i) for i in kw["chunk_index"]] == [i for i, _ in bm25.search(q, 3)], kw

    assert rag.search_in_memory(q, chunks, emb, None, top_k=3).equals(vec)  # no keyword index -> vector

    state["fail"] = True
    fallback = rag.search_in_memory(q, chunks, emb, bm25, top_k=3, mode="hybrid")
    assert [int(i) for i in fallback["chunk_index"]] == [i for i, _ in bm25.search(q, 3)], fallback
    print("✅ hybrid fusion (vector / hybrid / keyword / embedding failure)")


def preview_filing(filing_path: str, questions_path: str, n: int = 5):
    """Top 3 keyword-search chunks for the first n eval questions on a real filing (no Gemini)."""
    from chunking import chunk_filing_text

    with open(filing_path, encoding="utf-8") as f:
        chunks = chunk_filing_text(f.read(), 2000, 300)
    with open(questions_path, encoding="utf-8") as f:
        questions = [q for q in json.load(f)["questions"] if q.get("answerable")][:n]
    idx = BM25Index(chunks)
    print(f"\n{len(chunks)} chunks from {filing_path}")
    for q in questions:
        print(f"\n🔎 {q['id']}: {q['question']}")
        for rank, (i, score) in enumerate(idx.search(q["question"], 3), start=1):
            preview = " ".join(chunks[i].split())[:160]
            print(f"  {rank}. chunk {i}  bm25={score:.2f}  {preview}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--filing", default="", help="optional saved filing, e.g. filing_AMZN.md")
    ap.add_argument("--questions", default="eval_questions_AMZN.json")
    args = ap.parse_args()

    check_tokenize()
    check_bm25()
    check_fusion()
    print("All checks passed.")
    if args.filing:
        preview_filing(args.filing, args.questions)
