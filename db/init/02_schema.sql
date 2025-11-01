-- ============================================
-- RAG System Database Schema
-- ============================================
-- This schema defines the core tables for a RAG (Retrieval-Augmented Generation) system
-- using PostgreSQL with pgvector for efficient vector similarity search.
--
-- Table Structure:
--   documents:  Metadata for each source document/file
--   chunks:     Text chunks extracted from documents for processing
--   embeddings: Vector embeddings for each chunk + optional rerank scores
--
-- Key Design Notes:
-- * VECTOR dimension is set to 768 (adjust to match your embedding model)
-- * Cosine distance is the default similarity metric (vector_cosine_ops + <=>)
-- * All statements are idempotent (safe to run multiple times)
-- * Proper indexing for efficient vector similarity search

-- ============================================
-- Utility Functions and Triggers
-- ============================================

-- Function to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$;

-- ============================================
-- Documents Table
-- ============================================
-- Stores metadata for each source document/file in the RAG system
CREATE TABLE IF NOT EXISTS documents (
  id          BIGSERIAL PRIMARY KEY,        -- Unique document identifier
  source_uri  TEXT,                         -- File path or URL of the source document
  title       TEXT,                         -- Human-readable title/name
  created_at  TIMESTAMPTZ DEFAULT NOW(),    -- When document was first ingested
  updated_at  TIMESTAMPTZ DEFAULT NOW()     -- When document was last modified
);

-- Prevent duplicate ingestion of the same file/title combination
CREATE UNIQUE INDEX IF NOT EXISTS ux_documents_source_title
  ON documents (source_uri, title);

-- Automatically update updated_at timestamp on document changes
DROP TRIGGER IF EXISTS trg_documents_updated_at ON documents;
CREATE TRIGGER trg_documents_updated_at
BEFORE UPDATE ON documents
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ============================================
-- Chunks Table
-- ============================================
-- Stores text chunks extracted from documents for embedding and retrieval
CREATE TABLE IF NOT EXISTS chunks (
  id           BIGSERIAL PRIMARY KEY,        -- Unique chunk identifier
  document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,  -- Parent document
  chunk_index  INT NOT NULL,                 -- Sequential index within the document (0..N)
  text         TEXT NOT NULL,                -- The actual text content of the chunk
  created_at   TIMESTAMPTZ DEFAULT NOW()     -- When chunk was created
);

-- Ensure unique chunk indices per document (prevent duplicates)
CREATE UNIQUE INDEX IF NOT EXISTS ux_chunks_doc_idx
  ON chunks (document_id, chunk_index);

-- Index for efficient lookups by document
CREATE INDEX IF NOT EXISTS idx_chunks_doc
  ON chunks (document_id);

-- ============================================
-- Embeddings Table
-- ============================================
-- Stores vector embeddings for each chunk, enabling semantic similarity search
-- Note: Adjust VECTOR(768) dimension to match your embedding model's output
CREATE TABLE IF NOT EXISTS embeddings (
  chunk_id    BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,  -- One-to-one with chunks
  embedding   VECTOR(768),       -- Vector embedding (NULL until embedder processes it)
  rank_score  REAL,              -- Optional reranker quality score for improved relevance
  created_at  TIMESTAMPTZ DEFAULT NOW()     -- When embedding was created
);

-- ============================================
-- Vector Similarity Search Indexes
-- ============================================

-- IVFFLAT index for fast approximate nearest neighbor search using cosine distance
-- This is the primary index for semantic similarity queries
-- lists parameter: ~sqrt(num_rows) is a good starting point; tune based on performance
CREATE INDEX IF NOT EXISTS idx_embeddings_embedding_cosine
  ON embeddings USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- Alternative: L2 distance index (uncomment if you prefer L2 over cosine similarity)
-- CREATE INDEX idx_embeddings_embedding_l2
--   ON embeddings USING ivfflat (embedding vector_l2_ops)
--   WITH (lists = 100);

-- Index for efficient sorting by rerank scores (improves query performance)
CREATE INDEX IF NOT EXISTS idx_embeddings_rank_score
  ON embeddings (rank_score DESC NULLS LAST);

-- ============================================
-- Query Feedback Table
-- ============================================
-- Stores user feedback on search results for progressive learning
-- This enables the system to learn which results are helpful for specific queries
CREATE TABLE IF NOT EXISTS query_feedback (
  id            BIGSERIAL PRIMARY KEY,
  query_text    TEXT NOT NULL,                -- The original user query
  chunk_id      BIGINT REFERENCES chunks(id) ON DELETE CASCADE,  -- Chunk that was rated
  was_helpful   BOOLEAN,                      -- User thumbs up/down (NULL = not rated yet)
  clicked       BOOLEAN DEFAULT FALSE,        -- Whether user clicked/viewed this result
  created_at    TIMESTAMPTZ DEFAULT NOW()     -- When feedback was recorded
);

-- Index for efficient query feedback lookups
CREATE INDEX IF NOT EXISTS idx_feedback_query
  ON query_feedback(query_text);

-- Index for efficient chunk feedback aggregation
CREATE INDEX IF NOT EXISTS idx_feedback_chunk
  ON query_feedback(chunk_id);

-- Index for finding helpful results
CREATE INDEX IF NOT EXISTS idx_feedback_helpful
  ON query_feedback(chunk_id, was_helpful)
  WHERE was_helpful IS NOT NULL;

-- ============================================
-- Convenience Views
-- ============================================

-- View to easily identify chunks that still need embeddings
-- Useful for monitoring the embedding pipeline progress
CREATE OR REPLACE VIEW pending_embeddings AS
SELECT c.id AS chunk_id, d.title, d.source_uri, c.chunk_index
FROM chunks c
JOIN documents d ON d.id = c.document_id
LEFT JOIN embeddings e ON e.chunk_id = c.id
WHERE e.embedding IS NULL;

-- ============================================
-- Data Validation and Safety Checks
-- ============================================

-- Placeholder for future data validation checks
-- pgvector automatically enforces vector dimension constraints at storage time
DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_name = 'embeddings' AND column_name = 'embedding'
  ) THEN
    -- Future custom validation logic can be added here
    NULL;
  END IF;
END$$;
