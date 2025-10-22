"""
FastAPI service for document ingestion and processing.
Main entry point for the RAG system's file upload and processing pipeline.
"""

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from pathlib import Path
import os, psycopg, httpx

from utils_ingest import ingest_file

from fastapi.middleware.cors import CORSMiddleware

# FastAPI application for document ingestion service
# Supports integration with open-webui and other frontend applications
app = FastAPI(title="polars-worker")

# Configure CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class ChunkResult(BaseModel):
    chunk_id: int
    text: str
    title: str
    source_uri: str
    distance: float

@app.get("/healthz")
def health():
    """Health check endpoint for service monitoring."""
    return {"ok": True, "service": "polars-worker"}

@app.post("/ingest/file")
def ingest_file_endpoint(file: UploadFile):
    """
    File ingestion endpoint that processes uploaded documents.

    Workflow:
    1. Save uploaded file to temporary location
    2. Extract text and chunk the document
    3. Store document metadata and chunks in database
    4. Clean up temporary file
    """
    # Save uploaded file to temporary directory
    path = Path("/tmp") / file.filename
    with open(path, "wb") as f:
        f.write(file.file.read())

    # Process file through ingestion pipeline
    with psycopg.connect(os.getenv("DATABASE_URL")) as conn:
        ingest_file(conn, path)

    # Clean up temporary file
    path.unlink(missing_ok=True)
    return {"ingested": file.filename}

@app.post("/query")
async def query_endpoint(req: QueryRequest):
    """
    RAG query endpoint that finds relevant document chunks.

    Workflow:
    1. Embed the query using Ollama
    2. Search for similar chunks using pgvector
    3. Return top-k most relevant chunks with metadata
    """
    # Get embedding for the query
    ollama_host = os.getenv("OLLAMA_HOST", "http://192.168.7.215:11434")
    embed_model = os.getenv("EMBED_MODEL", "embeddinggemma")

    async with httpx.AsyncClient(timeout=30.0) as client:
        embed_resp = await client.post(
            f"{ollama_host}/api/embed",
            json={"model": embed_model, "input": [req.query]}
        )
        embed_resp.raise_for_status()
        embeddings = embed_resp.json().get("embeddings", [])
        if not embeddings:
            return {"error": "Failed to generate query embedding", "chunks": []}
        query_embedding = embeddings[0]

    # Query database for similar chunks
    with psycopg.connect(os.getenv("DATABASE_URL")) as conn:
        with conn.cursor() as cur:
            # Use pgvector cosine similarity search
            cur.execute("""
                SELECT
                    e.chunk_id,
                    c.text,
                    d.title,
                    d.source_uri,
                    e.embedding <=> %s::vector AS distance
                FROM embeddings e
                JOIN chunks c ON c.id = e.chunk_id
                JOIN documents d ON d.id = c.document_id
                WHERE e.embedding IS NOT NULL
                ORDER BY distance
                LIMIT %s
            """, (query_embedding, req.top_k))

            results = []
            for row in cur.fetchall():
                results.append({
                    "chunk_id": row[0],
                    "text": row[1],
                    "title": row[2],
                    "source_uri": row[3],
                    "distance": float(row[4])
                })

    return {"query": req.query, "chunks": results}
