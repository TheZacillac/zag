#!/bin/bash

# Local development launcher for RAG Chat TUI
# Run the TUI directly with Python (no Docker)

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🐍 RAG Chat TUI - Local Development${NC}"
echo "====================================="

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo -e "${RED}❌ app.py not found. Please run this script from the chat-tui directory.${NC}"
    exit 1
fi

# Check Python dependencies
echo -e "${YELLOW}🔍 Checking Python dependencies...${NC}"
missing_deps=()

if ! python3 -c "import textual" 2>/dev/null; then
    missing_deps+=("textual")
fi

if ! python3 -c "import httpx" 2>/dev/null; then
    missing_deps+=("httpx")
fi

if ! python3 -c "import rich" 2>/dev/null; then
    missing_deps+=("rich")
fi

if ! python3 -c "import pyperclip" 2>/dev/null; then
    missing_deps+=("pyperclip")
fi

if [ ${#missing_deps[@]} -ne 0 ]; then
    echo -e "${RED}❌ Missing dependencies: ${missing_deps[*]}${NC}"
    echo "Install them with:"
    echo "  pip install ${missing_deps[*]}"
    exit 1
fi

echo -e "${GREEN}✅ All dependencies found!${NC}"

# Set environment variables
export POLARS_API=${POLARS_API:-"http://localhost:8080"}
export OLLAMA_HOST=${OLLAMA_HOST:-"http://192.168.7.215:11434"}
export CHAT_MODEL=${CHAT_MODEL:-"llama3.2"}

echo -e "${YELLOW}🔧 Configuration:${NC}"
echo "  POLARS_API: $POLARS_API"
echo "  OLLAMA_HOST: $OLLAMA_HOST"
echo "  CHAT_MODEL: $CHAT_MODEL"
echo ""

# Check if services are running
echo -e "${YELLOW}🔍 Checking if RAG services are running...${NC}"
if curl -s "$POLARS_API/healthz" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ RAG API is running${NC}"
else
    echo -e "${RED}⚠️  RAG API not responding at $POLARS_API${NC}"
    echo "Make sure your RAG services are running first!"
    echo ""
fi

if curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Ollama is running${NC}"
else
    echo -e "${RED}⚠️  Ollama not responding at $OLLAMA_HOST${NC}"
    echo "Make sure Ollama is running!"
    echo ""
fi

echo -e "${GREEN}🚀 Starting RAG Chat TUI...${NC}"
echo "Press Ctrl+C to quit"
echo ""

python3 app.py

