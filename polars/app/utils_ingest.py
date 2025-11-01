"""
Document processing utilities for the RAG system.

This module handles the first stage of the RAG pipeline: converting documents
into text chunks ready for embedding. It uses the Unstructured library to extract
text from various formats (PDF, DOCX, TXT, etc.) and intelligently chunks the
content for optimal retrieval.

Key design decisions:
- Chunk size: 800 characters (balances context vs. specificity)
- Overlap: 100 characters (ensures continuity across chunk boundaries)
- Word boundary preservation: Prevents splitting words mid-character
"""

import logging
import os, re
import psycopg
from pathlib import Path
from unstructured.partition.auto import partition

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_text(path: Path) -> str:
    """
    Extract text content from various document formats using the Unstructured library.

    The Unstructured library automatically detects the file type and applies the
    appropriate parser (PDF, DOCX, HTML, Markdown, plain text, etc.). It handles
    complex document structures like tables, headers, and multi-column layouts.

    Args:
        path: Path to the document file (can be any format Unstructured supports)

    Returns:
        Cleaned text content with normalized whitespace and excessive newlines removed

    Note: This function filters elements to only those with text content, which
    excludes images, charts, and other non-textual elements.
    """
    # Partition the document into semantic elements (paragraphs, titles, etc.)
    # The auto partitioner detects file type and chooses the appropriate strategy
    elements = partition(filename=str(path))

    # Extract text from each element and join with newlines
    # Filter to only elements that have text (excludes images, etc.)
    raw_text = "\n".join([e.text for e in elements if hasattr(e, "text")]).strip()

    # Normalize excessive newlines (3+ newlines become 2 newlines)
    # This preserves paragraph breaks while removing large empty spaces
    return re.sub(r"\n{3,}", "\n\n", raw_text)

def chunk_text(text: str, size: int = 800, overlap: int = 100):
    """
    Split text into overlapping chunks optimized for semantic embedding.

    Chunking strategy:
    - Target size of 800 chars (roughly 150-200 words) balances:
      * Large enough: Provides sufficient context for meaningful embeddings
      * Small enough: Keeps embeddings focused on specific topics
    - 100 char overlap ensures important information near boundaries isn't lost
    - Word boundary preservation prevents mid-word splits that harm readability

    The overlap is crucial for RAG systems because:
    - A query might match text near a chunk boundary
    - Overlap ensures adjacent chunks share context, improving retrieval recall

    Args:
        text: Input text to chunk (typically from extract_text())
        size: Target maximum chunk size in characters (default: 800)
        overlap: Number of characters to overlap between consecutive chunks (default: 100)

    Returns:
        List of text chunks, each ≤ size characters (may be smaller at word boundaries)

    Example:
        >>> chunk_text("This is a test document with multiple words.", size=20, overlap=5)
        ['This is a test', 'test document with', 'with multiple', 'multiple words.']
        # Note the 'test', 'with', and 'multiple' appear in consecutive chunks
    """
    # Validate parameters to prevent infinite loops
    if overlap >= size:
        raise ValueError(f"Overlap ({overlap}) must be less than chunk size ({size})")
    if size <= 0:
        raise ValueError(f"Chunk size must be positive, got {size}")

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        # Calculate the end position for this chunk
        end = min(start + size, text_len)

        # If we're not at the end of the text, try to break at a word boundary
        # This prevents splitting words like "understand" into "under" and "stand"
        if end < text_len:
            # Search backwards from 'end' to 'start' for the last space character
            space_pos = text.rfind(' ', start, end)

            # Only use the space if it's not too far back (at least halfway through chunk)
            # This prevents creating very small chunks if there are no nearby spaces
            if space_pos > start + (size // 2):
                end = space_pos + 1  # Include the space in the chunk

        # Extract the chunk and remove leading/trailing whitespace
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks (skip whitespace-only chunks)
            chunks.append(chunk)

        # If we've reached the end of the text, we're done
        if end >= text_len:
            break

        # Move start position for next chunk, accounting for overlap
        next_start = end - overlap

        # Ensure we always advance by at least 1 character to prevent infinite loops
        if next_start <= start:
            next_start = start + 1

        start = next_start

    return chunks

def ingest_file(conn: psycopg.Connection, path: Path):
    """
    Main ingestion pipeline: process a document and store it in the database.

    This function is the entry point for document ingestion, called by the
    /ingest/file API endpoint. It orchestrates the entire ingestion workflow:

    1. Extract text from the document (using Unstructured library)
    2. Split text into overlapping chunks (800 char with 100 char overlap)
    3. Create a document record in the database
    4. Create chunk records linked to the document
    5. Create embedding placeholder records (NULL until embed_worker processes)

    After this function completes:
    - embed_worker will pick up chunks with NULL embeddings and populate them
    - rerank_worker will then calculate rank_score values
    - The chunks become searchable via the /search endpoint

    Args:
        conn: Database connection (will be committed at the end)
        path: Path to the document file to ingest

    Database Schema:
        documents: (id, source_uri, title)
        chunks: (id, document_id, chunk_index, text)
        embeddings: (chunk_id, embedding, rank_score)

    Note: This function creates embeddings rows with NULL values, which signals
    to the embed_worker that these chunks need processing.
    """
    # Step 1: Extract text from the document (handles PDF, DOCX, etc.)
    text = extract_text(path)

    # Step 2: Split into overlapping chunks for embedding
    chunks = chunk_text(text)

    with conn.cursor() as cur:
        # Step 3: Create document record and get its ID
        # source_uri stores the full path, title stores just the filename
        cur.execute(
            "INSERT INTO documents (source_uri, title) VALUES (%s, %s) RETURNING id",
            (str(path), path.name)
        )
        doc_id = cur.fetchone()[0]

        # Step 4 & 5: Create chunk and embedding records
        for i, chunk in enumerate(chunks):
            # Skip empty chunks to avoid wasting storage and processing
            if not chunk or not chunk.strip():
                continue

            # Insert chunk record with its position in the document
            cur.execute(
                "INSERT INTO chunks (document_id, chunk_index, text) VALUES (%s, %s, %s) RETURNING id",
                (doc_id, i, chunk),
            )
            chunk_id = cur.fetchone()[0]

            # Create embedding placeholder (NULL signals embed_worker to process it)
            # The embed_worker will UPDATE this row with the actual embedding vector
            cur.execute(
                "INSERT INTO embeddings (chunk_id, embedding) VALUES (%s, NULL)",
                (chunk_id,)
            )

    # Commit the transaction to make chunks visible to workers
    conn.commit()
    logger.info(f"📥 {path.name}: {len(chunks)} chunks staged (awaiting embedding).")
