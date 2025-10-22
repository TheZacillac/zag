#!/bin/bash

# RAG System Startup Script
# Single command to start the entire RAG system

echo "🚀 Starting RAG System..."
echo "========================="
echo ""
echo "This will start:"
echo "  • Database (Postgres + pgvector)"
echo "  • RAG API (polars-worker)"
echo "  • Document digester"
echo "  • Embedding worker"
echo "  • Reranking worker"
echo "  • Chat TUI interface"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Start the full system
docker compose up --build
