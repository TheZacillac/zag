"""
FastAPI service for document ingestion and processing.
Main entry point for the RAG system's file upload and processing pipeline.
"""

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from pathlib import Path
import os
import tempfile
import uuid
import psycopg
import httpx

from utils_ingest import ingest_file
from vector_utils import to_pgvector

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

EXPECTED_EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# Request/Response models for API endpoints
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

@app.get("/models")
async def list_models():
    """
    List available Ollama models for chat completion.

    Returns:
        Dictionary with available models and the default model name.
    """
    ollama_host = os.getenv("OLLAMA_HOST", "http://192.168.7.215:11434")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{ollama_host}/api/tags")
            resp.raise_for_status()
            data = resp.json()

            return {
                "models": [m["name"] for m in data.get("models", [])],
                "default": os.getenv("CHAT_MODEL", "llama3.2")
            }
    except (httpx.HTTPError, ValueError) as e:
        return {"error": f"Failed to fetch models: {str(e)}", "models": [], "default": ""}

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
    # Validate filename
    if not file.filename:
        return {"error": "Filename is required"}

    # Validate file type
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md', '.html', '.pptx', '.doc', '.rtf'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return {"error": f"File type {file_ext} not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}

    # Sanitize filename to prevent path traversal
    safe_filename = os.path.basename(file.filename)
    temp_dir = Path(tempfile.gettempdir())
    unique_name = f"{uuid.uuid4().hex}_{safe_filename}"
    path = temp_dir / unique_name

    try:
        # Save uploaded file to temporary directory with size validation
        MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
        file_content = file.file.read(MAX_FILE_SIZE + 1)

        if len(file_content) == 0:
            return {"error": "Empty files cannot be ingested"}

        if len(file_content) > MAX_FILE_SIZE:
            return {"error": f"File too large (max {MAX_FILE_SIZE // (1024 * 1024)}MB)"}

        with open(path, "wb") as f:
            f.write(file_content)

        # Process file through ingestion pipeline with timeout
        with psycopg.connect(
            os.getenv("DATABASE_URL"),
            connect_timeout=10,
            options="-c statement_timeout=30000"
        ) as conn:
            ingest_file(conn, path)

        return {"ingested": safe_filename}
    except psycopg.OperationalError as e:
        return {"error": f"Database connection error: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}
    finally:
        # Ensure cleanup even if an exception occurs
        if path.exists():
            path.unlink(missing_ok=True)

@app.get("/search")
async def search_endpoint(q: str, k: int = 5):
    """
    RAG search endpoint that finds relevant document chunks using semantic search.

    This is the core retrieval endpoint for the RAG system. It converts user queries
    into embeddings and uses pgvector's approximate nearest neighbor (ANN) search
    to find semantically similar document chunks.

    Workflow:
    1. Embed the query using Ollama's embedding model
    2. Search for similar chunks using pgvector cosine similarity (<=> operator)
    3. Return top-k most relevant chunks with metadata

    Args:
        q: The search query string
        k: Number of most relevant chunks to return (default: 5)

    Returns:
        Dictionary with the original query and a list of matching chunks with:
        - chunk_id: Unique identifier for the chunk
        - text: The actual text content of the chunk
        - title: Document title (for attribution)
        - source_uri: Original file path or URI
        - distance: Cosine distance (0 = identical, 2 = opposite)
    """
    # Validate inputs
    if not q or not q.strip():
        return {"error": "Query parameter q is required and cannot be empty"}

    try:
        k = int(k)
        if k < 1 or k > 100:
            return {"error": "Parameter k must be between 1 and 100"}
    except (TypeError, ValueError):
        return {"error": "Parameter k must be an integer"}
    # Step 1: Get embedding vector for the user's query
    # These settings should match the worker that generates document embeddings
    ollama_host = os.getenv("OLLAMA_HOST", "http://192.168.7.215:11434")
    embed_model = os.getenv("EMBED_MODEL", "embeddinggemma")

    # Call Ollama's embedding API with a 30-second timeout
    # The API accepts a batch of inputs, but we only need to embed one query
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            embed_resp = await client.post(
                f"{ollama_host}/api/embed",
                json={"model": embed_model, "input": [q]}
            )
            embed_resp.raise_for_status()
            data = embed_resp.json()
            embeddings = data.get("embeddings", [])
            if not embeddings:
                return {"error": "Failed to generate query embedding", "chunks": []}
            query_embedding = embeddings[0]  # Extract the first (and only) embedding

            # Validate vector dimension
            if len(query_embedding) != EXPECTED_EMBED_DIM:
                return {
                    "error": (
                        f"Embedding dimension mismatch: expected {EXPECTED_EMBED_DIM}, "
                        f"got {len(query_embedding)}"
                    ),
                    "chunks": [],
                }

            query_vector = to_pgvector(query_embedding)
    except (httpx.HTTPError, ValueError) as e:
        return {"error": f"Failed to generate embeddings: {str(e)}", "chunks": []}

    # Step 2: Search for similar chunks using pgvector
    try:
        with psycopg.connect(
            os.getenv("DATABASE_URL"),
            connect_timeout=10,
            options="-c statement_timeout=30000"
        ) as conn:
            with conn.cursor() as cur:
                # Use pgvector's <=> operator for cosine distance calculation
                # This is an approximate nearest neighbor (ANN) search powered by
                # the IVFFLAT index on embeddings.embedding
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
                """, (query_vector, k))

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

        return {"query": q, "chunks": results}
    except psycopg.OperationalError as e:
        return {"error": f"Database error: {str(e)}", "chunks": []}

@app.post("/answer")
async def answer_endpoint(request: dict):
    """
    RAG-powered answer generation endpoint.

    Performs semantic search to find relevant document chunks, then uses an LLM
    to generate a natural language answer based on the retrieved context.

    Args:
        request: Dictionary containing:
            - query: The user's question (required)
            - k: Number of chunks to retrieve (default: 5)
            - model: Ollama model name to use (optional, uses CHAT_MODEL env var if not specified)

    Returns:
        Dictionary with:
            - query: Original query
            - chunks: Retrieved document chunks
            - answer: LLM-generated answer based on chunks
            - model: Model name used for generation
            - error: Error message if generation failed
    """
    query = request.get("query", "").strip()
    k = request.get("k", 5)
    model = request.get("model", "").strip() or os.getenv("CHAT_MODEL", "llama3.2")

    # Validate inputs
    if not query:
        return {"error": "Query is required and cannot be empty"}

    try:
        k = int(k)
        if k < 1 or k > 100:
            return {"error": "Parameter k must be between 1 and 100"}
    except (TypeError, ValueError):
        return {"error": "Parameter k must be an integer"}

    # Step 1: Retrieve relevant chunks using the search endpoint logic
    ollama_host = os.getenv("OLLAMA_HOST", "http://192.168.7.215:11434")
    embed_model = os.getenv("EMBED_MODEL", "embeddinggemma")

    try:
        # Get embedding for the query
        async with httpx.AsyncClient(timeout=30.0) as client:
            embed_resp = await client.post(
                f"{ollama_host}/api/embed",
                json={"model": embed_model, "input": [query]}
            )
            embed_resp.raise_for_status()
            data = embed_resp.json()
            embeddings = data.get("embeddings", [])
            if not embeddings:
                return {"error": "Failed to generate query embedding", "chunks": []}
            query_embedding = embeddings[0]
            if len(query_embedding) != EXPECTED_EMBED_DIM:
                return {
                    "error": (
                        f"Embedding dimension mismatch: expected {EXPECTED_EMBED_DIM}, "
                        f"got {len(query_embedding)}"
                    ),
                    "chunks": [],
                }
            query_vector = to_pgvector(query_embedding)
    except (httpx.HTTPError, ValueError) as e:
        return {"error": f"Failed to generate embeddings: {str(e)}", "chunks": []}

    # Step 2: Search for similar chunks
    try:
        with psycopg.connect(
            os.getenv("DATABASE_URL"),
            connect_timeout=10,
            options="-c statement_timeout=30000"
        ) as conn:
            with conn.cursor() as cur:
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
                """, (query_vector, k))

                chunks = []
                for row in cur.fetchall():
                    chunks.append({
                        "chunk_id": row[0],
                        "text": row[1],
                        "title": row[2],
                        "source_uri": row[3],
                        "distance": float(row[4])
                    })
    except psycopg.Error as e:
        return {"error": f"Database error: {str(e)}", "chunks": []}

    if not chunks:
        return {"query": query, "chunks": [], "answer": "No relevant information found in the knowledge base."}

    # Step 3: Generate answer using LLM with retrieved context
    context = "\n\n".join([f"[{i+1}] {chunk['title']}: {chunk['text']}" for i, chunk in enumerate(chunks)])
    prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context from knowledge base:
{context}

Question: {query}

Please provide a clear, accurate answer based on the context above. If the context doesn't contain enough information to fully answer the question, say so."""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            gen_resp = await client.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            gen_resp.raise_for_status()
            gen_data = gen_resp.json()
            answer = gen_data.get("response", "")

            return {
                "query": query,
                "chunks": chunks,
                "answer": answer,
                "model": model
            }
    except (httpx.HTTPError, ValueError) as e:
        return {
            "query": query,
            "chunks": chunks,
            "error": f"Failed to generate answer: {str(e)}"
        }
