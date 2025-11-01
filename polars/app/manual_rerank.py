#!/usr/bin/env python3
"""Manual reranking script to work around Ollama 500 errors"""
import psycopg
import httpx
import math
import os
import sys

def cosine_similarity(a, b):
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a < 1e-10 or mag_b < 1e-10:
        return 0.0
    return dot / (mag_a * mag_b)

DB_URL = os.getenv("DATABASE_URL")
OLLAMA = os.getenv("OLLAMA_HOST", "http://192.168.7.215:11434")
MODEL = "all-minilm:latest"

print("Starting manual reranking...", flush=True)
print(f"DB: {DB_URL}", flush=True)
print(f"Ollama: {OLLAMA}", flush=True)

try:
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.id, c.text, COALESCE(NULLIF(d.title, ''), d.source_uri) AS query
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                JOIN embeddings e ON e.chunk_id = c.id
                WHERE e.rank_score IS NULL
                ORDER BY c.id
            """)
            rows = cur.fetchall()
            print(f"Found {len(rows)} unranked chunks", flush=True)

            if not rows:
                print("No unranked chunks found!", flush=True)
                sys.exit(0)

            with httpx.Client(timeout=60) as client:
                for i, (chunk_id, text, query) in enumerate(rows):
                    try:
                        # Embed query and chunk
                        resp = client.post(f"{OLLAMA}/api/embed", json={"model": MODEL, "input": [query, text[:4000]]})
                        resp.raise_for_status()
                        embeddings = resp.json()["embeddings"]

                        # Calculate similarity
                        score = cosine_similarity(embeddings[0], embeddings[1])

                        # Update database
                        cur.execute("UPDATE embeddings SET rank_score = %s WHERE chunk_id = %s", (float(score), chunk_id))
                        conn.commit()

                        print(f"✓ Ranked chunk {chunk_id} ({i+1}/{len(rows)}): {score:.4f}", flush=True)

                    except Exception as e:
                        print(f"✗ Failed chunk {chunk_id}: {e}", flush=True)
                        continue

    print("Done!", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
