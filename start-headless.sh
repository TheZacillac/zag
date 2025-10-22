#!/bin/bash

# RAG System Headless Startup Script
# Single command to start the RAG system without chat-tui (API only)

echo "🔧 Starting RAG System (Headless Mode)..."
echo "=========================================="
echo ""
echo "This will start:"
echo "  • Database (Postgres + pgvector)"
echo "  • RAG API (polars-worker)"
echo "  • Document digester"
echo "  • Embedding worker"
echo "  • Reranking worker"
echo ""
echo "Chat TUI will NOT be started"
echo "Use the API at: http://localhost:8080"
echo ""
echo "To stop: docker compose -f docker-compose.headless.yml down"
echo ""

# Start the headless system
docker compose -f docker-compose.headless.yml up --build -d

echo ""
echo "✅ RAG System started in headless mode!"
echo "📊 Check status: docker compose -f docker-compose.headless.yml ps"
echo "📋 View logs: docker compose -f docker-compose.headless.yml logs -f"
