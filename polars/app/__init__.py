"""
RAG system shared configuration module.

Provides centralized configuration management for all RAG components.
All workers (FastAPI service, embedder, reranker) use these environment-based settings.
"""

import os

# Database connection configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:ragpass@db:5432/ragdb")

# Ollama service configuration
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://ollama:11434")

# Model configuration for different tasks
EMBED_MODEL  = os.getenv("EMBED_MODEL", "embedding-gemma")    # Text embedding model
RERANK_MODEL = os.getenv("RERANK_MODEL", "all-minilm")       # Reranking model
