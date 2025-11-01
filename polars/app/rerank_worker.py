"""
Continuous reranking worker service.

This worker improves search quality by calculating relevance scores for chunks.
It runs after the embedding worker, processing chunks that have been embedded
but don't yet have rank_score values.

The reranking approach:
1. Use document title/filename as a synthetic "query" representing the document's topic
2. Embed both the query (title) and the chunk text
3. Calculate cosine similarity between them
4. Store the score for use in blended ranking during search

This score indicates how representative a chunk is of its parent document's main topic.
Chunks with high scores are more likely to be relevant when the document is retrieved.

Performance optimization: The worker interleaves query and document embeddings in a
single batch API call, reducing latency by ~50% compared to separate calls.
"""

import logging
import math
import os
import time
from typing import Iterable, List, Sequence, Tuple

import httpx
import psycopg
from psycopg_pool import ConnectionPool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables with sensible defaults
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("DATABASE_URL environment variable is required")
OLLAMA = os.getenv("OLLAMA_HOST", "http://192.168.7.215:11434")  # Ollama API endpoint
MODEL = os.getenv("RERANK_MODEL", "all-minilm")  # Can be different from main embedding model
BATCH = int(os.getenv("BATCH_SIZE", "50"))  # Chunks to rerank per iteration
SLEEP = int(os.getenv("SLEEP_SEC", "30"))  # Seconds to wait when no work available
REQUEST_TIMEOUT = int(os.getenv("RERANK_TIMEOUT", "300"))  # 5-minute timeout for large batches

def fetch_candidates(conn: psycopg.Connection, limit: int = BATCH) -> Sequence[Tuple[int, str, str]]:
    """
    Fetch chunks that need reranking using row-level locking.

    Selects chunks that:
    - Have been embedded (embedding IS NOT NULL)
    - Haven't been reranked yet (rank_score IS NULL)

    Uses "FOR UPDATE SKIP LOCKED" to enable safe concurrent processing by
    multiple rerank_worker instances.

    Args:
        conn: Database connection (should be from a transaction)
        limit: Maximum number of chunks to process in one batch

    Returns:
        List of (chunk_id, chunk_text, query_text) tuples where:
        - chunk_id: ID of the chunk to rerank
        - chunk_text: The actual text content
        - query_text: Document title (or source_uri fallback) used as synthetic query
    """
    with conn.cursor() as cur:
        # Find chunks ready for reranking
        # COALESCE prioritizes document title, falls back to source_uri if title is empty
        # ORDER BY c.id ensures deterministic processing order
        # FOR UPDATE SKIP LOCKED enables concurrent workers without conflicts
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
    Batch embed multiple texts using Ollama's embedding API.

    This function is used to embed both queries and documents in a single call,
    which is significantly faster than making separate API calls.

    Args:
        client: HTTP client with timeout configured
        inputs: List of text strings to embed (can be mixed query/doc texts)

    Returns:
        List of embedding vectors in the same order as inputs

    Raises:
        httpx.HTTPError: If the API request fails
        httpx.TimeoutException: If the request exceeds the configured timeout
    """
    input_list = list(inputs)
    if not input_list:
        return []

    payload = {"model": MODEL, "input": input_list}
    resp = client.post(f"{OLLAMA}/api/embed", json=payload)
    resp.raise_for_status()
    data = resp.json()

    # Handle batch response (typical case)
    embeddings = data.get("embeddings")
    if embeddings is not None:
        return embeddings

    # Fallback for single embedding format (rare, but handle gracefully)
    return [data["embedding"]]

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Cosine similarity measures the cosine of the angle between two vectors,
    ranging from -1 (opposite) to 1 (identical). It's commonly used for
    semantic similarity because it's independent of vector magnitude.

    Args:
        a: First embedding vector
        b: Second embedding vector

    Returns:
        Similarity score in range [-1, 1], or 0.0 if calculation fails
    """
    # Validate vectors have same dimensionality
    if len(a) != len(b):
        return 0.0

    # Calculate dot product: sum of element-wise products
    dot_product = sum(x * y for x, y in zip(a, b))

    # Calculate magnitudes (L2 norms) of both vectors
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))

    # Avoid division by zero for null vectors with epsilon tolerance
    EPSILON = 1e-10
    if magnitude_a < EPSILON or magnitude_b < EPSILON:
        logger.warning("Near-zero vector magnitude detected in cosine similarity calculation")
        return 0.0

    # Cosine similarity = dot product / (magnitude_a * magnitude_b)
    return dot_product / (magnitude_a * magnitude_b)

def update_scores(conn: psycopg.Connection, ids: Sequence[int], scores: Sequence[float]) -> None:
    """
    Update the database with calculated rerank scores.

    These scores represent how well each chunk represents its parent document's
    main topic. They can be used during search to blend vector similarity with
    document-level relevance.

    Args:
        conn: Database connection (will commit the transaction)
        ids: List of chunk IDs (must match length of scores)
        scores: List of relevance scores (typically cosine similarities in range [0, 1])
    """
    with conn.cursor() as cur:
        # Batch update all scores in a single transaction
        cur.executemany(
            "UPDATE embeddings SET rank_score = %s WHERE chunk_id = %s",
            [(float(score), cid) for cid, score in zip(ids, scores)],
        )
    # Commit to release row locks acquired by FOR UPDATE
    conn.commit()

def main():
    """
    Main worker loop that continuously processes chunks for reranking.

    This is an infinite loop designed to run as a background service. The workflow:
    1. Fetch chunks that have embeddings but no rank scores
    2. Use document title as a synthetic query representing the document's topic
    3. Embed both queries and chunks in a SINGLE optimized API call (interleaved)
    4. Calculate cosine similarity between each query-chunk pair
    5. Store scores in database
    6. Repeat

    The interleaving optimization (step 3) is the key innovation here:
    - Instead of: embed_batch(queries) + embed_batch(chunks) = 2 API calls
    - We do: embed_batch([q1, c1, q2, c2, ...]) = 1 API call
    - This reduces latency by ~50% and Ollama server load

    The interleaving works because:
    - Ollama processes [q1, c1, q2, c2, ...] and returns embeddings in same order
    - We split results using even/odd indices: [0::2] for queries, [1::2] for chunks
    """
    logger.info(f"🎯 Reranker worker started using {MODEL} (timeout={REQUEST_TIMEOUT}s)")

    # Use connection pooling for efficient database access with 30s acquisition timeout
    # The timeout prevents deadlocks under high load
    with ConnectionPool(DB_URL, min_size=1, max_size=4, timeout=30) as pool:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            while True:
                try:
                    with pool.connection() as conn:
                        # Step 1: Fetch chunks that need reranking
                        rows = fetch_candidates(conn)
                        if not rows:
                            # No work available - sleep and check again later
                            time.sleep(SLEEP)
                            continue

                        # Step 2: Extract chunk IDs, texts, and synthetic queries
                        ids, chunk_texts, query_texts = zip(*rows)

                        # Use document title as query, fallback to chunk text if no title
                        query_inputs = [q if q else t for q, t in zip(query_texts, chunk_texts)]

                        # Truncate very long chunks to keep API payloads reasonable
                        # 4000 chars is enough for context while keeping latency low
                        truncated_chunks = [text[:4000] if text else "" for text in chunk_texts]

                        # Step 3: OPTIMIZATION - Interleave queries and docs for single API call
                        # This is the most complex part of the code, so let's break it down:
                        #
                        # Without optimization (2 API calls):
                        #   query_vectors = embed_batch([q1, q2, q3, ...])
                        #   doc_vectors = embed_batch([d1, d2, d3, ...])
                        #
                        # With optimization (1 API call):
                        #   combined = [q1, d1, q2, d2, q3, d3, ...]
                        #   all_vectors = embed_batch(combined)
                        #   query_vectors = all_vectors[0::2]  # Extract even indices
                        #   doc_vectors = all_vectors[1::2]    # Extract odd indices
                        combined_inputs = []
                        for q, d in zip(query_inputs, truncated_chunks):
                            combined_inputs.append(q)  # Even index (0, 2, 4, ...)
                            combined_inputs.append(d)  # Odd index (1, 3, 5, ...)

                        # Get all embeddings in one batch (huge performance win!)
                        all_vectors = embed_batch(client, combined_inputs)

                        # Step 4: De-interleave results back into separate lists
                        query_vectors = all_vectors[0::2]  # Every 2nd element starting at 0
                        doc_vectors = all_vectors[1::2]    # Every 2nd element starting at 1

                        # Step 5: Calculate cosine similarity for each query-doc pair
                        scores = [
                            cosine_similarity(q_vec, d_vec)
                            if q_vec and d_vec  # Ensure both vectors exist
                            else 0.0
                            for q_vec, d_vec in zip(query_vectors, doc_vectors)
                        ]

                        # Step 6: Write scores to database and commit
                        update_scores(conn, ids, scores)
                        logger.info(f"✅ Reranked {len(ids)} chunks.")

                # Error handling: Different strategies for different error types
                except httpx.TimeoutException as e:
                    # Ollama is slow but likely still working - short retry delay
                    logger.warning(f"⏱️ Reranking timeout (model may be slow): {e}")
                    time.sleep(SLEEP)
                except httpx.HTTPError as e:
                    # Network or Ollama API error - longer delay before retry
                    logger.error(f"🌐 HTTP error communicating with Ollama: {e}")
                    time.sleep(30)
                except psycopg.OperationalError as e:
                    # Database connection issue - longer delay before retry
                    logger.error(f"💾 Database connection error: {e}")
                    time.sleep(30)
                except Exception as e:
                    # Unexpected error - log with traceback and continue with moderate delay
                    chunk_preview = ids[:5] if 'ids' in locals() else 'unknown'
                    logger.exception(f"💥 Unexpected error processing chunks {chunk_preview}: {e}")
                    time.sleep(10)

if __name__ == "__main__":
    main()
