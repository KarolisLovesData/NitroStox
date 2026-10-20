import os
import functools
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor
from chunking import chunk_filing_text

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("⚠️ GEMINI_API_KEY not found in environment or .env file.")

client = genai.Client(
    api_key=gemini_api_key,
    http_options=types.HttpOptions(api_version="v1")
)

EMBED_MODEL = "models/gemini-embedding-001"
EMBED_BATCH_SIZE = 20  # chunks per embed_content call (~300 chunks -> ~15 calls instead of ~300)

# Search mode: "hybrid" (BM25 keyword + vector, fused with RRF), "vector" or "keyword"
SEARCH_MODE = os.getenv("NITROSTOX_SEARCH_MODE", "hybrid")
RRF_K = 60            # same k constant as the Text-to-SQL engine
RRF_CANDIDATES = 20   # each list contributes its top 20 to the fusion
VECTOR_WEIGHT = 1.0
KEYWORD_WEIGHT = 1.0


def _extract_vectors(response) -> list[tuple]:
    """Pulls vectors out of an embed_content response (handles both response shapes)."""
    if getattr(response, "embeddings", None):
        return [tuple(e.values) for e in response.embeddings]
    if getattr(response, "embedding", None):
        return [tuple(response.embedding.values)]
    return []


def _embed_with_retry(texts: list[str]) -> list[tuple]:
    """One embed_content call with exponential backoff on quota errors.

    Raises on final failure instead of returning an empty result, so a transient
    error is never stored by lru_cache and the next call gets a fresh attempt.
    """
    last_err = None
    for attempt in range(3):
        try:
            return _extract_vectors(
                client.models.embed_content(model=EMBED_MODEL, contents=texts)
            )
        except Exception as e:
            last_err = e
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                break
    raise RuntimeError(f"Gemini embedding failed: {last_err}")


@functools.lru_cache(maxsize=256)
def _cached_query_embedding(input_text: str) -> tuple:
    """Cache for search queries only. Chunk vectors live in the knowledge-base cache instead."""
    vectors = _embed_with_retry([input_text])
    if not vectors:
        raise RuntimeError("Gemini returned no embedding")
    return vectors[0]


class SECVectorRAG:

    def __init__(self, ticker: str):
        self.ticker = ticker.upper().strip()

    def chunk_text(
            self, text_content: str, chunk_size: int = 2000, overlap: int = 300
    ) -> list[str]:
        return chunk_filing_text(text_content, chunk_size, overlap)

    def generate_embedding(self, input_text: str) -> list[float]:
        """Embeds one string (used for search queries). Returns [] on failure."""
        try:
            return list(_cached_query_embedding(input_text))
        except Exception as e:
            print(f"⚠️ Gemini Embedding Error: {e}")
            return []

    def embed_chunks(self, chunks: list[str]) -> list[np.ndarray]:
        """Embeds chunks in batches. Returns one vector per chunk, in order.

        A chunk that failed to embed comes back as an empty list so callers can
        filter it out while keeping chunk/vector indexes aligned.
        """
        print(f"⏳ Generating vectors for {len(chunks)} chunks in RAM...")
        batches = [
            chunks[i:i + EMBED_BATCH_SIZE]
            for i in range(0, len(chunks), EMBED_BATCH_SIZE)
        ]

        def embed_batch(batch: list[str]) -> list:
            try:
                vectors = _embed_with_retry(batch)
            except Exception as e:
                print(f"⚠️ Batch embedding failed: {e}")
                return [[] for _ in batch]

            if len(vectors) != len(batch):
                # Unexpected response shape: fall back to one call per chunk
                single = [self.generate_embedding(c) for c in batch]
                return [np.asarray(v, dtype=np.float32) if v else [] for v in single]

            return [np.asarray(v, dtype=np.float32) for v in vectors]

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(embed_batch, batches))

        return [vec for batch_vectors in results for vec in batch_vectors]

    def _vector_ranking(self, user_query: str, chunk_embeddings) -> list[tuple[int, float]]:
        """All chunks as (chunk_index, cosine similarity), best first. [] if the query embedding failed."""
        query_vector = self.generate_embedding(user_query)

        if not query_vector or len(chunk_embeddings) == 0:
            return []

        # Convert to NumPy arrays
        query_vec = np.asarray(query_vector, dtype=np.float32)
        matrix = np.asarray(chunk_embeddings, dtype=np.float32)

        # L2 Normalize the vectors
        query_norm = query_vec / np.linalg.norm(query_vec)
        matrix_norms = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

        # Compute cosine similarity via dot product
        similarities = np.dot(matrix_norms, query_norm)

        order = np.argsort(similarities)[::-1]
        return [(int(idx), float(similarities[idx])) for idx in order]

    def vector_search_in_memory(self, user_query: str, chunks: list[str], chunk_embeddings,
                                top_k: int = 5) -> pd.DataFrame:
        """Executes high-speed cosine similarity search using NumPy in Python RAM."""
        ranking = self._vector_ranking(user_query, chunk_embeddings)
        if not ranking:
            return pd.DataFrame()

        results = []
        for idx, score in ranking[:top_k]:
            results.append({
                "chunk_index": idx,
                "chunk_text": chunks[idx],
                "similarity_score": round(score, 4)
            })

        return pd.DataFrame(results)

    def search_in_memory(self, user_query: str, chunks: list[str], chunk_embeddings, bm25,
                         top_k: int = 5, mode: str = None) -> pd.DataFrame:
        """Hybrid search: BM25 keyword list + vector list, fused with Reciprocal Rank Fusion.

        mode: "hybrid" (default from NITROSTOX_SEARCH_MODE), "vector" or "keyword".
        Returns the same columns as vector_search_in_memory (chunk_index, chunk_text,
        similarity_score = cosine) plus rrf_score, vector_rank and keyword_rank.
        """
        mode = (mode or SEARCH_MODE).lower()
        if mode == "vector" or bm25 is None:
            return self.vector_search_in_memory(user_query, chunks, chunk_embeddings, top_k)

        # The vector list is computed in keyword mode too, so similarity_score keeps its meaning
        vec_full = self._vector_ranking(user_query, chunk_embeddings)
        kw = bm25.search(user_query, RRF_CANDIDATES)
        if not vec_full and mode != "keyword":
            print("⚠️ Query embedding failed; keyword-only results.")

        cosine = {idx: score for idx, score in vec_full}
        vec_rank = {idx: r for r, (idx, _) in enumerate(vec_full[:RRF_CANDIDATES], start=1)}
        kw_rank = {idx: r for r, (idx, _) in enumerate(kw, start=1)}

        # RRF: score += weight / (k + rank), rank 1-based (Text-to-SQL uses 0-based rank + 1)
        rrf = {}
        if mode != "keyword":
            for idx, r in vec_rank.items():
                rrf[idx] = rrf.get(idx, 0.0) + VECTOR_WEIGHT / (RRF_K + r)
        for idx, r in kw_rank.items():
            rrf[idx] = rrf.get(idx, 0.0) + KEYWORD_WEIGHT / (RRF_K + r)

        if not rrf:
            return pd.DataFrame()

        # Hybrid ties go to the better vector rank, then the better keyword rank
        big = len(chunks) + 1
        if mode == "keyword":
            ordered = sorted(rrf, key=lambda i: (kw_rank[i], i))
        else:
            ordered = sorted(rrf, key=lambda i: (-rrf[i], vec_rank.get(i, big), kw_rank.get(i, big)))

        results = []
        for idx in ordered[:top_k]:
            results.append({
                "chunk_index": idx,
                "chunk_text": chunks[idx],
                "similarity_score": round(cosine[idx], 4) if idx in cosine else None,
                "rrf_score": round(rrf[idx], 6),
                "vector_rank": vec_rank.get(idx),
                "keyword_rank": kw_rank.get(idx),
            })

        return pd.DataFrame(results)
