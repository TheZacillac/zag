"""
Document processing utilities for the RAG system.
Handles text extraction, chunking, and database storage of documents.
"""

import os, re
import psycopg
from pathlib import Path
from unstructured.partition.auto import partition

def extract_text(path: Path) -> str:
    """
    Extract text content from various document formats using Unstructured.
    
    Args:
        path: Path to the document file
        
    Returns:
        Cleaned text content with normalized whitespace
    """
    elements = partition(filename=str(path))
    # Normalize excessive newlines and extract text from elements
    return re.sub(r"\n{3,}", "\n\n", "\n".join([e.text for e in elements if hasattr(e, "text")]).strip())

def chunk_text(text: str, size: int = 800, overlap: int = 100):
    """
    Split text into overlapping chunks for embedding processing.
    Respects word boundaries to avoid splitting words mid-character.

    Args:
        text: Input text to chunk
        size: Maximum chunk size in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + size, text_len)

        # If not at the end of text, try to break at word boundary
        if end < text_len:
            # Look backwards for a space to break at
            space_pos = text.rfind(' ', start, end)
            if space_pos > start + (size // 2):  # Only use space if it's not too far back
                end = space_pos + 1  # Include the space

        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)

        # Move start position with overlap for continuity
        # Ensure we make progress even with small chunks
        start = max(start + 1, end - overlap)

    return chunks

def ingest_file(conn: psycopg.Connection, path: Path):
    """
    Main ingestion function that processes a document file.
    
    Workflow:
    1. Extract text from document
    2. Split into chunks
    3. Store document metadata in database
    4. Store chunks with placeholder embeddings
    
    Args:
        conn: Database connection
        path: Path to the document file
    """
    # Extract and chunk the document text
    text = extract_text(path)
    chunks = chunk_text(text)

    with conn.cursor() as cur:
        # Insert document record and get ID
        cur.execute("INSERT INTO documents (source_uri, title) VALUES (%s,%s) RETURNING id", (str(path), path.name))
        doc_id = cur.fetchone()[0]
        
        # Insert each chunk with placeholder embedding
        for i, chunk in enumerate(chunks):
            cur.execute(
                "INSERT INTO chunks (document_id, chunk_index, text) VALUES (%s,%s,%s) RETURNING id",
                (doc_id, i, chunk),
            )
            cid = cur.fetchone()[0]
            # Create embedding placeholder (NULL until embed worker processes it)
            cur.execute(
                "INSERT INTO embeddings (chunk_id, embedding) VALUES (%s,NULL)", (cid,)
            )
    conn.commit()
    print(f"📥 {path.name}: {len(chunks)} chunks staged (awaiting embedding).")
