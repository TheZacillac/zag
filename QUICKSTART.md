# Quick Start Guide - RAG System
**Get up and running in 5 minutes!**

---

## Prerequisites

✅ **Docker & Docker Compose** installed  
✅ **Ollama** running (locally or on network)  
✅ **Ollama models** pulled:
```bash
ollama pull embeddinggemma
ollama pull all-minilm
ollama pull llama3.2  # or your preferred chat model
```

---

## Option 1: Quick Start (1 command!)

```bash
./start.sh
```

That's it! This starts:
- PostgreSQL database with pgvector
- RAG API server
- Document ingester
- Embedding worker
- Reranking worker
- Interactive chat TUI

---

## Option 2: Headless (API only)

```bash
./start-headless.sh
```

Starts everything except the chat TUI. Access API at http://localhost:8080

---

## Option 3: Using Make

```bash
# Start full system with chat
make up

# Start headless
make headless

# View all commands
make help
```

---

## First Use - Add Documents

### Step 1: Drop files into digestion/

```bash
# Copy any document (PDF, TXT, MD, DOCX, etc.)
cp ~/Documents/myfile.pdf digestion/

# Or create a test file
echo "RAG systems enhance LLMs with document retrieval" > digestion/test.txt
```

### Step 2: Watch it process

```bash
# The digester uploads it automatically
docker compose logs -f digester

# The embedder creates vectors
docker compose logs -f embedder

# The reranker calculates scores
docker compose logs -f reranker
```

Files are **automatically deleted** from `digestion/` after successful upload.

---

## Query Your Documents

### Via Chat TUI (Interactive)

If you ran `./start.sh`, the TUI is already running!

**Keyboard Shortcuts:**
- `Enter` - Send message
- `Ctrl+Y` - Copy last response
- `Ctrl+L` - Clear chat
- `Ctrl+C` - Quit

**Commands:**
- `/model <name>` - Change the chat model
- `/models` - List available models

### Via API (Programmatic)

```bash
# Search for relevant chunks
curl "http://localhost:8080/search?q=your%20query&k=5"

# Health check
curl http://localhost:8080/healthz

# Upload file programmatically
curl -X POST http://localhost:8080/ingest/file \
  -F "file=@myfile.pdf"
```

---

## Configuration (Optional)

### If Ollama is on a different machine:

1. Copy environment template:
```bash
cp env.example .env
```

2. Edit `.env` and set your Ollama location:
```bash
OLLAMA_HOST=http://192.168.1.100:11434
```

3. Restart:
```bash
docker compose down
docker compose up --build
```

### Change models:

Edit `.env`:
```bash
EMBED_MODEL=nomic-embed-text
RERANK_MODEL=bge-reranker-base
CHAT_MODEL=mixtral
```

---

## Verify Everything Works

### Check Service Status

```bash
docker compose ps
```

All services should show "healthy" or "running".

### Check Logs

```bash
# All services
docker compose logs

# Specific service
docker compose logs -f embedder

# Follow all logs
docker compose logs -f
```

### Test Search

```bash
# After uploading a document, search for it
curl "http://localhost:8080/search?q=test&k=5" | jq
```

Expected response:
```json
{
  "query": "test",
  "chunks": [
    {
      "chunk_id": 1,
      "text": "RAG systems enhance...",
      "title": "test.txt",
      "source_uri": "/tmp/test.txt",
      "distance": 0.234
    }
  ]
}
```

---

## Common Issues

### "Connection refused" to Ollama

**Problem:** Ollama not reachable from Docker

**Solutions:**

1. **If Ollama is on your host machine:**
   ```bash
   # Use host.docker.internal (default in docker-compose.yml)
   # No changes needed!
   ```

2. **If Ollama is on different machine:**
   ```bash
   # Set OLLAMA_HOST in .env file
   echo "OLLAMA_HOST=http://192.168.1.100:11434" > .env
   ```

3. **Test connectivity:**
   ```bash
   # From your machine
   curl http://localhost:11434/api/tags
   
   # From inside Docker
   docker compose exec polars-worker curl http://host.docker.internal:11434/api/tags
   ```

### Embeddings not being generated

**Check:**
1. Is embedder running? `docker compose ps embedder`
2. Can it reach Ollama? `docker compose logs embedder`
3. Is the model available? `ollama list`

**Fix:**
```bash
# Pull the embedding model
ollama pull embeddinggemma

# Restart embedder
docker compose restart embedder
```

### Search returns no results

**Possible causes:**
1. **No documents uploaded yet** - Drop files in `digestion/`
2. **Embeddings still processing** - Check `docker compose logs embedder`
3. **Wrong vector dimension** - Must match model (768 for embeddinggemma)

**Check:**
```bash
# See if embeddings exist
docker compose exec db psql -U rag -d ragdb -c "SELECT COUNT(*) FROM embeddings WHERE embedding IS NOT NULL;"
```

### Digester not processing files

**Check:**
1. Is digester running? `docker compose ps digester`
2. Is directory mounted? `docker compose exec digester ls /digestion`
3. Are files being detected? `docker compose logs digester`

**Fix:**
```bash
# Ensure directory exists
mkdir -p digestion

# Restart digester
docker compose restart digester
```

---

## Stop the System

```bash
# Stop all services (data persists)
docker compose down

# Stop and remove volumes (fresh start)
docker compose down -v
```

---

## Useful Commands

### Monitor in Real-Time

```bash
# Watch all logs
docker compose logs -f

# Watch specific service
docker compose logs -f embedder

# Check system status
make status
```

### Database Access

```bash
# Connect to database
docker compose exec db psql -U rag -d ragdb

# Useful queries:
# SELECT COUNT(*) FROM documents;
# SELECT COUNT(*) FROM chunks;
# SELECT COUNT(*) FROM embeddings WHERE embedding IS NOT NULL;
```

### Clean Up

```bash
# Remove all data and start fresh
make clean

# Or manually:
docker compose down -v
docker system prune -f
```

---

## Performance Tips

### For faster embeddings:

1. **Increase batch size:**
   ```bash
   # In .env
   EMBED_BATCH=128  # Default is 64
   ```

2. **Run multiple embedder instances:**
   ```bash
   docker compose up -d --scale embedder=3
   ```

### For better search quality:

1. **Adjust chunk size** (in `polars/app/utils_ingest.py`):
   ```python
   chunk_text(text, size=1000, overlap=150)  # Default: 800, 100
   ```

2. **Increase k results:**
   ```bash
   curl "http://localhost:8080/search?q=query&k=10"  # Default k=5
   ```

---

## Next Steps

### Learn More
- Read `README.md` for architecture details
- See `SYSTEM_REVIEW.md` for technical deep-dive
- Check `CONFIG_FIX_GUIDE.md` for advanced configuration

### Customize
- Change models in `.env`
- Adjust chunk sizes in `utils_ingest.py`
- Modify UI colors in `chat-tui/app.py`

### Integrate
- Use `/search` API endpoint in your app
- Build custom frontend
- Connect to other services

### Deploy to Production
- See `SYSTEM_REVIEW.md` deployment section
- Set up monitoring (Prometheus + Grafana)
- Configure backups for PostgreSQL
- Add authentication and rate limiting

---

## Getting Help

### Check Documentation
- `README.md` - Full documentation
- `SYSTEM_REVIEW.md` - Technical analysis
- `CONFIG_FIX_GUIDE.md` - Configuration help
- `dataflow.md` - Data pipeline explanation

### Check Logs
```bash
# All services
docker compose logs

# Specific issue
docker compose logs embedder | grep ERROR
```

### Health Checks
```bash
# API health
curl http://localhost:8080/healthz

# Database health
docker compose exec db pg_isready -U rag -d ragdb

# Ollama health
curl http://localhost:11434/api/tags
```

---

## Summary

```bash
# Complete workflow in 4 commands:
./start.sh                           # 1. Start system
cp myfile.pdf digestion/             # 2. Add document
# Wait ~30 seconds for processing    # 3. Wait
curl "http://localhost:8080/search?q=test&k=5"  # 4. Search!
```

**That's it! You now have a fully functional RAG system running.**

Happy retrieving! 🚀

