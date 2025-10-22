"""
Continuous reranking worker service.
Processes chunks that have embeddings but no rerank scores, improving search quality.
"""

import math
import os
import time
from typing import Iterable, List, Sequence, Tuple

import httpx
import psycopg
from psycopg_pool import ConnectionPool

# Configuration from environment variables
DB_URL = os.getenv("DATABASE_URL")
OLLAMA = os.getenv("OLLAMA_HOST", "http://192.168.7.215:11434")
MODEL = os.getenv("RERANK_MODEL", "all-minilm")
BATCH = int(os.getenv("BATCH_SIZE", "50"))
SLEEP = int(os.getenv("SLEEP_SEC", "30"))
REQUEST_TIMEOUT = int(os.getenv("RERANK_TIMEOUT", "300"))

def fetch_candidates(conn: psycopg.Connection, limit: int = BATCH) -> Sequence[Tuple[int, str, str]]:
    """
    Fetch chunks that need reranking (have embeddings but no rank scores).
    
    Args:
        conn: Database connection
        limit: Maximum number of chunks to process in one batch
        
    Returns:
        List of (chunk_id, chunk_text, query_text) tuples
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                c.id,
                c.text,
                COALESCE(NULLIF(d.title, ''), d.source_uri) AS query_text
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            JOIN embeddings e ON e.chunk_id = c.id
            WHERE e.rank_score IS NULL
              AND e.embedding IS NOT NULL
            ORDER BY c.id
            FOR UPDATE SKIP LOCKED
            LIMIT %s;
        """, (limit,))
        return cur.fetchall()

def embed_batch(client: httpx.Client, inputs: Iterable[str]) -> List[Sequence[float]]:
    """
    Batch embed the provided texts using the rerank model.
    Optimized to handle multiple inputs efficiently.
    """
    input_list = list(inputs)
    if not input_list:
        return []

    payload = {"model": MODEL, "input": input_list}
    resp = client.post(f"{OLLAMA}/api/embed", json=payload)
    resp.raise_for_status()
    data = resp.json()
    embeddings = data.get("embeddings")
    if embeddings is not None:
        return embeddings
    # Fallback for single embedding format (though Ollama should always return array)
    return [data["embedding"]]

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)

def update_scores(conn: psycopg.Connection, ids: Sequence[int], scores: Sequence[float]) -> None:
    """
    Update the database with rerank scores for processed chunks.
    
    Args:
        conn: Database connection
        ids: List of chunk IDs
        scores: List of relevance scores
    """
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE embeddings SET rank_score = %s WHERE chunk_id = %s",
            [(float(score), cid) for cid, score in zip(ids, scores)],
        )
    conn.commit()

def main():
    """
    Main worker loop that continuously processes chunks for reranking.
    Uses document titles or filenames as synthetic queries for scoring.
    Optimized with connection pooling and single-batch embedding.
    """
    print(f"🎯 Reranker worker started using {MODEL} (timeout={REQUEST_TIMEOUT}s)")

    with ConnectionPool(DB_URL, min_size=1, max_size=4) as pool:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            while True:
                try:
                    with pool.connection() as conn:
                        rows = fetch_candidates(conn)
                        if not rows:
                            time.sleep(SLEEP)
                            continue

                        ids, chunk_texts, query_texts = zip(*rows)
                        # Use query metadata when available, otherwise fall back to the chunk text.
                        query_inputs = [q if q else t for q, t in zip(query_texts, chunk_texts)]

                        # Keep requests bounded by truncating very long chunks that make poor rerank candidates.
                        truncated_chunks = [text[:4000] if text else "" for text in chunk_texts]

                        # OPTIMIZATION: Combine both query and doc embeddings in a single API call
                        # Interleave query and doc inputs: [q1, d1, q2, d2, ...]
                        combined_inputs = []
                        for q, d in zip(query_inputs, truncated_chunks):
                            combined_inputs.append(q)
                            combined_inputs.append(d)

                        # Get all embeddings in one batch
                        all_vectors = embed_batch(client, combined_inputs)

                        # De-interleave: split back into query and doc vectors
                        query_vectors = all_vectors[0::2]  # Even indices
                        doc_vectors = all_vectors[1::2]    # Odd indices

                        scores = [
                            cosine_similarity(q_vec, d_vec)
                            if q_vec and d_vec
                            else 0.0
                            for q_vec, d_vec in zip(query_vectors, doc_vectors)
                        ]
                        update_scores(conn, ids, scores)
                        print(f"✅ Reranked {len(ids)} chunks.")
                except httpx.TimeoutException as e:
                    print(f"⏱️ Reranking timeout (model may be slow): {e}")
                    time.sleep(SLEEP)
                except httpx.HTTPError as e:
                    print(f"🌐 HTTP error communicating with Ollama: {e}")
                    time.sleep(30)
                except psycopg.OperationalError as e:
                    print(f"💾 Database connection error: {e}")
                    time.sleep(30)
                except Exception as e:
                    print(f"💥 Unexpected error: {e}")
                    time.sleep(10)

if __name__ == "__main__":
    main()
