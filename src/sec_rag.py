import os
import functools
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor

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
        chunks = []
        start = 0
        text_length = len(text_content)

        while start < text_length:
            end = start + chunk_size
            chunk = text_content[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += chunk_size - overlap

        return chunks

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

    def vector_search_in_memory(self, user_query: str, chunks: list[str], chunk_embeddings,
                                top_k: int = 5) -> pd.DataFrame:
        """Executes high-speed cosine similarity search using NumPy in Python RAM."""
        query_vector = self.generate_embedding(user_query)

        if not query_vector or len(chunk_embeddings) == 0:
            return pd.DataFrame()

        # Convert to NumPy arrays
        query_vec = np.asarray(query_vector, dtype=np.float32)
        matrix = np.asarray(chunk_embeddings, dtype=np.float32)

        # L2 Normalize the vectors
        query_norm = query_vec / np.linalg.norm(query_vec)
        matrix_norms = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

        # Compute cosine similarity via dot product
        similarities = np.dot(matrix_norms, query_norm)

        # Sort and get top_k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "chunk_index": idx,
                "chunk_text": chunks[idx],
                "similarity_score": round(float(similarities[idx]), 4)
            })

        return pd.DataFrame(results)
