#!/bin/bash

# RAG Chat TUI Launcher
# Quick script to launch the chat-tui application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 RAG Chat TUI Launcher${NC}"
echo "================================"

# Check if docker compose is available
if ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ docker compose not found. Please install Docker Compose.${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

echo -e "${YELLOW}📋 Available launch options:${NC}"
echo "1. Docker Compose (Recommended) - Full stack with RAG services"
echo "2. Docker Compose (Standalone) - Just the TUI, assumes services are running"
echo "3. Local Python - Run directly with Python (requires local setup)"
echo ""

read -p "Choose option (1-3): " choice

case $choice in
    1)
        echo -e "${GREEN}🐳 Starting full RAG stack with chat-tui...${NC}"
        echo "This will start: database, polars-worker, embedder, reranker, and chat-tui"
        echo ""
        docker compose --profile chat up --build
        ;;
    2)
        echo -e "${GREEN}🐳 Starting chat-tui only...${NC}"
        echo "Make sure your RAG services are already running!"
        echo ""
        docker compose --profile chat run --rm chat-tui
        ;;
    3)
        echo -e "${GREEN}🐍 Starting with local Python...${NC}"
        echo "Make sure you have the required dependencies installed:"
        echo "  pip install textual httpx rich pyperclip"
        echo ""
        
        # Check if Python dependencies are available
        if ! python3 -c "import textual, httpx, rich, pyperclip" 2>/dev/null; then
            echo -e "${RED}❌ Missing Python dependencies. Please install them first:${NC}"
            echo "  pip install textual httpx rich pyperclip"
            exit 1
        fi
        
        # Set environment variables for local development
        export POLARS_API=${POLARS_API:-"http://localhost:8080"}
        export OLLAMA_HOST=${OLLAMA_HOST:-"http://192.168.7.215:11434"}
        export CHAT_MODEL=${CHAT_MODEL:-"llama3.2"}
        
        echo -e "${YELLOW}🔧 Using environment:${NC}"
        echo "  POLARS_API: $POLARS_API"
        echo "  OLLAMA_HOST: $OLLAMA_HOST"
        echo "  CHAT_MODEL: $CHAT_MODEL"
        echo ""
        
        cd chat-tui
        python3 app.py
        ;;
    *)
        echo -e "${RED}❌ Invalid option. Please choose 1, 2, or 3.${NC}"
        exit 1
        ;;
esac
