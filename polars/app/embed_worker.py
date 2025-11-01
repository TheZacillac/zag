"""
Continuous embedding worker service.

This worker is a critical part of the RAG pipeline. It runs continuously,
polling the database for text chunks that need to be embedded, calling Ollama's
embedding API in batches, and writing the resulting vectors back to PostgreSQL.

Key features:
- Batch processing for efficiency (default: 64 chunks at once)
- Row-level locking (FOR UPDATE SKIP LOCKED) to prevent duplicate work
- Connection pooling for database performance
- Comprehensive error handling with different strategies per error type
- Graceful degradation when Ollama is slow or unavailable

The worker is designed to be run as a long-lived service (e.g., in Docker).
Multiple instances can run concurrently thanks to the row-level locking strategy.
"""

import logging
import os, time, httpx, psycopg
from psycopg_pool import ConnectionPool
from typing import Iterable, List, Sequence, Tuple

from vector_utils import to_pgvector

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
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")  # Model must match query embedding
BATCH_SIZE = int(os.getenv("EMBED_BATCH", "64"))  # Chunks to process per iteration
SLEEP_SEC = int(os.getenv("EMBED_SLEEP", "15"))  # Seconds to wait when no work available
REQUEST_TIMEOUT = int(os.getenv("EMBED_TIMEOUT", "300"))  # 5-minute timeout for large batches

def fetch_pending(conn: psycopg.Connection, limit: int) -> Sequence[Tuple[int, str]]:
    """
    Fetch chunks that need embeddings from the database using row-level locking.

    This function uses "FOR UPDATE SKIP LOCKED" to enable concurrent processing:
    - FOR UPDATE: Locks the selected rows so other workers can't modify them
    - SKIP LOCKED: Skips rows already locked by other workers, preventing contention

    This pattern allows multiple embed_worker instances to run in parallel safely.
    Each worker will grab different chunks and process them independently.

    Args:
        conn: Database connection (should be from a transaction)
        limit: Maximum number of chunks to process in one batch

    Returns:
        List of (chunk_id, text) tuples for chunks that need embedding
    """
    with conn.cursor() as cur:
        # Find chunks where embedding column is NULL (not yet embedded)
        # The JOIN ensures we only get chunks that have an embeddings row
        # FOR UPDATE SKIP LOCKED prevents multiple workers from grabbing the same rows
        cur.execute("""
          SELECT c.id, c.text
          FROM chunks c
          JOIN embeddings e ON e.chunk_id = c.id
          WHERE e.embedding IS NULL
          FOR UPDATE SKIP LOCKED
          LIMIT %s;
        """, (limit,))
        return cur.fetchall()

def embed_texts(client: httpx.Client, texts: Iterable[str]) -> List[Sequence[float]]:
    """
    Generate embeddings for a list of texts using Ollama's batch API.

    This function calls Ollama's /api/embed endpoint which accepts multiple
    input texts and returns their embeddings in a single request. This is much
    more efficient than making individual requests per text.

    Args:
        client: HTTP client with timeout configured
        texts: List of text strings to embed (typically 64 chunks)

    Returns:
        List of embedding vectors (each vector is a list of floats)
        The order of embeddings matches the order of input texts

    Raises:
        httpx.HTTPError: If the API request fails
        httpx.TimeoutException: If the request exceeds the configured timeout
    """
    payload = {"model": EMBED_MODEL, "input": list(texts)}
    r = client.post(f"{OLLAMA}/api/embed", json=payload)
    r.raise_for_status()
    data = r.json()

    # Handle batch response (typical case)
    embeddings = data.get("embeddings")
    if embeddings is not None:
        return embeddings

    # Handle single embedding response (fallback for singleton batches)
    # Some Ollama versions return {"embedding": [...]} for single inputs
    return [data["embedding"]]

def write_back(conn: psycopg.Connection, ids: Sequence[int], vecs: Sequence[Sequence[float]]) -> None:
    """
    Update the database with generated embeddings using batch update.

    This function writes all embeddings in a single transaction for atomicity.
    If any update fails, the entire batch is rolled back, ensuring we can retry.

    Args:
        conn: Database connection (will commit the transaction)
        ids: List of chunk IDs (must match length of vecs)
        vecs: List of embedding vectors (each vector is a list of floats)

    Note: Vectors are converted to the pgvector textual format before being
    stored, ensuring compatibility without requiring custom adapters.
    """
    with conn.cursor() as cur:
        # Use executemany for efficient batch updates
        # Each embedding vector is paired with its corresponding chunk_id
        cur.executemany(
            "UPDATE embeddings SET embedding = %s::vector WHERE chunk_id = %s",
            [(to_pgvector(v), cid) for cid, v in zip(ids, vecs)],
        )
    # Commit the transaction to release the row locks acquired by FOR UPDATE
    conn.commit()

def main():
    """
    Main worker loop that continuously processes chunks for embedding.

    This is an infinite loop designed to run as a background service. The workflow:
    1. Fetch a batch of chunks that need embeddings (with row-level locking)
    2. Call Ollama API to generate embeddings
    3. Write embeddings back to database
    4. Repeat

    Connection pooling ensures efficient database access across iterations.
    Different error types receive different handling strategies:
    - Timeouts: Short sleep, retry (model might be processing large batch)
    - HTTP errors: Longer sleep (Ollama might be down or restarting)
    - DB errors: Longer sleep (database might be under maintenance)
    """
    logger.info(f"⚙️ Embed worker started (batch={BATCH_SIZE}, sleep={SLEEP_SEC}s, timeout={REQUEST_TIMEOUT}s)")

    # Create connection pool with 1-4 connections and 30s acquisition timeout
    # This is more efficient than opening/closing connections each iteration
    # The timeout prevents deadlocks under high load
    with ConnectionPool(DB_URL, min_size=1, max_size=4, timeout=30) as pool:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            while True:
                try:
                    # Get a connection from the pool (context manager returns it automatically)
                    with pool.connection() as conn:
                        # Step 1: Fetch batch of chunks to embed
                        rows = fetch_pending(conn, BATCH_SIZE)
                        if not rows:
                            # No work available - sleep and check again later
                            time.sleep(SLEEP_SEC)
                            continue

                        # Step 2: Separate chunk IDs from text content
                        ids, texts = zip(*rows)

                        # Step 3: Generate embeddings from Ollama
                        vectors = embed_texts(client, texts)

                        # Step 4: Write embeddings to database and commit
                        write_back(conn, ids, vectors)
                        logger.info(f"✅ Embedded {len(ids)} chunks")

                # Error handling: Different strategies for different error types
                except httpx.TimeoutException as e:
                    # Ollama is slow but likely still working - short retry delay
                    logger.warning(f"⏱️ Embedding timeout (model may be slow): {e}")
                    time.sleep(SLEEP_SEC)
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
