-- ============================================
-- pgvector Extension Setup
-- ============================================
-- This file sets up the pgvector extension for vector similarity search.
-- pgvector enables PostgreSQL to store and query high-dimensional vectors
-- efficiently, which is essential for RAG (Retrieval-Augmented Generation) systems.

-- Enable pgvector extension (idempotent - safe to run multiple times)
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================
-- Optional Performance Tuning Settings
-- ============================================
-- Uncomment and adjust these settings based on your system resources and performance needs:

-- Increase memory for index maintenance operations
-- SET maintenance_work_mem = '1GB';

-- Allow more parallel workers for index creation
-- SET max_parallel_maintenance_workers = 2;

-- IVFFLAT index tuning (query-time setting)
-- Higher values = better recall but slower queries
-- SET ivfflat.probes = 10;

-- HNSW index tuning (if you switch to HNSW indexes in the future)
-- Higher values = better recall but slower queries
-- SET hnsw.ef_search = 40;
