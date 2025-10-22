"""
Continuous embedding worker service.
Processes chunks that need embeddings by calling Ollama's embedding API.
"""

import os, time, httpx, psycopg
from psycopg_pool import ConnectionPool
from typing import Iterable, List, Sequence, Tuple

# Configuration from environment variables
DB_URL = os.getenv("DATABASE_URL")
OLLAMA = os.getenv("OLLAMA_HOST", "http://192.168.7.215:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embedding-gemma")
BATCH_SIZE = int(os.getenv("EMBED_BATCH", "64"))
SLEEP_SEC = int(os.getenv("EMBED_SLEEP", "15"))
REQUEST_TIMEOUT = int(os.getenv("EMBED_TIMEOUT", "300"))

def fetch_pending(conn: psycopg.Connection, limit: int) -> Sequence[Tuple[int, str]]:
    """
    Fetch chunks that need embeddings from the database.
    
    Args:
        conn: Database connection
        limit: Maximum number of chunks to process in one batch
        
    Returns:
        List of (chunk_id, text) tuples
    """
    with conn.cursor() as cur:
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
    Generate embeddings for a list of texts using Ollama.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors
    """
    payload = {"model": EMBED_MODEL, "input": list(texts)}
    r = client.post(f"{OLLAMA}/api/embed", json=payload)
    r.raise_for_status()
    data = r.json()
    embeddings = data.get("embeddings")
    if embeddings is not None:
        return embeddings
    # Allow embeds API to return a single embedding object for singleton batches
    return [data["embedding"]]

def write_back(conn: psycopg.Connection, ids: Sequence[int], vecs: Sequence[Sequence[float]]) -> None:
    """
    Update the database with generated embeddings.
    
    Args:
        conn: Database connection
        ids: List of chunk IDs
        vecs: List of embedding vectors
    """
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE embeddings SET embedding = %s WHERE chunk_id = %s",
            [(v, cid) for cid, v in zip(ids, vecs)],
        )
    conn.commit()

def main():
    """
    Main worker loop that continuously processes chunks for embedding.
    Fetches pending chunks, generates embeddings, and updates the database.
    Uses connection pooling for better database performance.
    """
    print(f"⚙️ Embed worker started (batch={BATCH_SIZE}, sleep={SLEEP_SEC}s, timeout={REQUEST_TIMEOUT}s)")

    # Create connection pool for better performance
    with ConnectionPool(DB_URL, min_size=1, max_size=4) as pool:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            while True:
                try:
                    with pool.connection() as conn:
                        rows = fetch_pending(conn, BATCH_SIZE)
                        if not rows:
                            time.sleep(SLEEP_SEC)
                            continue
                        ids, texts = zip(*rows)
                        vectors = embed_texts(client, texts)
                        write_back(conn, ids, vectors)
                        print(f"✅ Embedded {len(ids)} chunks")
                except httpx.TimeoutException as e:
                    print(f"⏱️ Embedding timeout (model may be slow): {e}")
                    time.sleep(SLEEP_SEC)
                except httpx.HTTPError as e:
                    print(f"🌐 HTTP error communicating with Ollama: {e}")
                    time.sleep(30)  # Longer sleep for network issues
                except psycopg.OperationalError as e:
                    print(f"💾 Database connection error: {e}")
                    time.sleep(30)  # Longer sleep for DB issues
                except Exception as e:
                    print(f"💥 Unexpected error: {e}")
                    time.sleep(10)

if __name__ == "__main__":
    main()
