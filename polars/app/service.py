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

# Request/Response models for API endpoints
class QueryRequest(BaseModel):
    """Request model for semantic search queries."""
    query: str  # The user's question or search query
    top_k: int = 5  # Number of most relevant chunks to return (default: 5)

class ChunkResult(BaseModel):
    """Response model for a single document chunk result."""
    chunk_id: int  # Unique identifier for the chunk
    text: str  # The actual text content of the chunk
    title: str  # Document title for attribution
    source_uri: str  # Original file path or URI
    distance: float  # Cosine distance (0 = identical, 2 = opposite)

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
    RAG query endpoint that finds relevant document chunks using semantic search.

    This is the core retrieval endpoint for the RAG system. It converts user queries
    into embeddings and uses pgvector's approximate nearest neighbor (ANN) search
    to find semantically similar document chunks.

    Workflow:
    1. Embed the query using Ollama's embedding model
    2. Search for similar chunks using pgvector cosine similarity (<=> operator)
    3. Return top-k most relevant chunks with metadata

    Args:
        req: QueryRequest containing the query string and optional top_k parameter

    Returns:
        Dictionary with the original query and a list of matching chunks with:
        - chunk_id: Unique identifier for the chunk
        - text: The actual text content of the chunk
        - title: Document title (for attribution)
        - source_uri: Original file path or URI
        - distance: Cosine distance (0 = identical, 2 = opposite)
    """
    # Step 1: Get embedding vector for the user's query
    # These settings should match the worker that generates document embeddings
    ollama_host = os.getenv("OLLAMA_HOST", "http://192.168.7.215:11434")
    embed_model = os.getenv("EMBED_MODEL", "embeddinggemma")

    # Call Ollama's embedding API with a 30-second timeout
    # The API accepts a batch of inputs, but we only need to embed one query
    async with httpx.AsyncClient(timeout=30.0) as client:
        embed_resp = await client.post(
            f"{ollama_host}/api/embed",
            json={"model": embed_model, "input": [req.query]}
        )
        embed_resp.raise_for_status()
        embeddings = embed_resp.json().get("embeddings", [])
        if not embeddings:
            return {"error": "Failed to generate query embedding", "chunks": []}
        query_embedding = embeddings[0]  # Extract the first (and only) embedding

    # Step 2: Search for similar chunks using pgvector
    with psycopg.connect(os.getenv("DATABASE_URL")) as conn:
        with conn.cursor() as cur:
            # Use pgvector's <=> operator for cosine distance calculation
            # This is an approximate nearest neighbor (ANN) search powered by
            # the HNSW index on embeddings.embedding
            #
            # JOIN order matters for performance:
            # - Start with embeddings table (has vector index)
            # - JOIN to chunks for text content
            # - JOIN to documents for metadata
            #
            # WHERE clause ensures we only search chunks that have been embedded
            # ORDER BY distance ensures closest matches come first (lower is better)
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

            # Step 3: Format results into JSON-serializable dictionaries
            results = []
            for row in cur.fetchall():
                results.append({
                    "chunk_id": row[0],
                    "text": row[1],
                    "title": row[2],
                    "source_uri": row[3],
                    "distance": float(row[4])  # Convert Decimal to float for JSON
                })

    return {"query": req.query, "chunks": results}
