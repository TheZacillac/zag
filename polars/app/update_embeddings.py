"""
One-time script to update pending embeddings for existing chunks.
This is a utility script for batch processing chunks that don't have embeddings yet.
"""

import os, json, httpx, polars as pl, psycopg

from vector_utils import to_pgvector

# Configuration from environment variables
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:ragpass@db:5432/ragdb")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")  # point to your running Ollama box
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")

def fetch_pending(conn, limit=256):
    """
    Fetch chunks that need embeddings from the database.
    
    Args:
        conn: Database connection
        limit: Maximum number of chunks to fetch
        
    Returns:
        List of (chunk_id, text) tuples
    """
    with conn.cursor() as cur:
        cur.execute("""
          SELECT c.id, c.text
          FROM chunks c
          JOIN embeddings e ON e.chunk_id = c.id
          WHERE e.embedding IS NULL
          LIMIT %s
        """, (limit,))
        return cur.fetchall()

def embed(texts):
    """
    Generate embeddings for a list of texts using Ollama.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (list[list[float]])
    """
    payload = {"model": EMBED_MODEL, "input": texts}
    with httpx.Client(timeout=60) as client:
        r = client.post(f"{OLLAMA_HOST}/api/embed", json=payload)
        r.raise_for_status()
        data = r.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise ValueError("No embeddings returned from API")
        return embeddings  # list[list[float]]

def write_back(conn, ids, vecs):
    """
    Update the database with generated embeddings.
    
    Args:
        conn: Database connection
        ids: List of chunk IDs
        vecs: List of embedding vectors
    """
    with conn.cursor() as cur:
        for cid, v in zip(ids, vecs):
            cur.execute(
                "UPDATE embeddings SET embedding = %s::vector WHERE chunk_id = %s",
                (to_pgvector(v), cid),
            )
    conn.commit()

if __name__ == "__main__":
    # Main execution: fetch pending chunks, generate embeddings, and update database
    with psycopg.connect(DATABASE_URL) as conn:
        rows = fetch_pending(conn)
        if not rows:
            print("No pending embeddings.")
            raise SystemExit(0)
        ids, texts = zip(*rows)
        vectors = embed(list(texts))
        write_back(conn, ids, vectors)
        print(f"Updated {len(ids)} embeddings.")
