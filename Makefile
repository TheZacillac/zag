# RAG System Makefile
# Convenient commands for managing the RAG system

.PHONY: help build up headless down logs clean dev-chat status monitor

# Default target
help:
	@echo "RAG System Commands"
	@echo "=================="
	@echo ""
	@echo "Main Commands:"
	@echo "  make up            - Start full RAG system with chat-tui"
	@echo "  make headless      - Start RAG system without chat-tui (API only)"
	@echo "  make down          - Stop all services"
	@echo "  make build         - Build all services"
	@echo "  make logs          - Show logs for all services"
	@echo "  make clean         - Clean up containers and volumes"
	@echo ""
	@echo "Development Commands:"
	@echo "  make dev-chat      - Run chat-tui locally (requires Python deps)"
	@echo "  make status        - Check service status"
	@echo "  make monitor       - Monitor digester service health"
	@echo ""

# Main system commands
up:
	@echo "🚀 Starting full RAG system with chat-tui..."
	docker compose up --build

headless:
	@echo "🔧 Starting RAG system without chat-tui (API only)..."
	docker compose -f docker-compose.headless.yml up --build -d

down:
	@echo "🛑 Stopping RAG services..."
	docker compose down

build:
	@echo "🔨 Building RAG services..."
	docker compose build

logs:
	@echo "📋 Showing RAG service logs..."
	docker compose logs -f

clean:
	@echo "🧹 Cleaning up RAG system..."
	docker compose down -v --remove-orphans
	docker system prune -f

# Development
dev-chat:
	@echo "🐍 Starting chat-tui in development mode..."
	cd chat-tui && ./run-local.sh

status:
	@echo "📊 RAG System Status:"
	@echo "===================="
	@docker compose ps
	@echo ""
	@echo "Health Checks:"
	@echo "RAG API: $$(curl -s http://localhost:8080/healthz 2>/dev/null && echo "✅ Healthy" || echo "❌ Unhealthy")"
	@echo "Ollama:  $$(curl -s http://192.168.7.215:11434/api/tags 2>/dev/null && echo "✅ Healthy" || echo "❌ Unhealthy")"

monitor:
	@echo "🔍 Monitoring digester service..."
	@./monitor-digester.sh
