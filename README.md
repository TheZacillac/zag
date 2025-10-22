# RAG Minimal — Postgres + pgvector + Polars + Ollama (embed + rerank)

A lean Retrieval-Augmented Generation ingestion/search stack:

- **Postgres 16 + pgvector** — stores documents, chunks, vectors, rerank scores  
- **Polars worker (FastAPI)** — APIs for ingest & search  
- **Digester** — watches a folder; auto-uploads files then deletes them  
- **Embedder** — async worker that batches embeddings via your **Ollama** embed model  
- **Reranker** — async worker that scores chunks via your **Ollama** reranker model  

> Your generator LLM(s) live elsewhere (already running in Ollama).

---

## How it works (at 10,000 ft)

1. **Drop files** into `./digestion/`.  
2. **Digester** posts them to the Polars API and **deletes** source files.  
3. **Polars API** extracts text (Unstructured), chunks it, and stores rows (embedding=NULL).  
4. **Embedder** polls for pending chunks, batches → **Ollama `/api/embed`**, writes vectors.  
5. **Reranker** polls for embedded chunks, calls **Ollama reranker** (e.g., `all-minilm`) to set `rank_score`.  
6. **Search**: FastAPI `/search` embeds the query, runs pgvector ANN, (optionally) blends similarity + `rank_score`, returns top-k chunks.

---

## Folder structure

```
rag-minimal/
├─ docker-compose.yml
│
├─ db/
│  └─ init/
│     ├─ 01_pgvector.sql      # CREATE EXTENSION vector;
│     └─ 02_schema.sql        # documents/chunks/embeddings (+ rank_score), indexes
│
├─ polars/
│  ├─ Dockerfile
│  └─ app/
│     ├─ __init__.py          # env config (DB URL, OLLAMA host, models)
│     ├─ service.py           # FastAPI: /healthz, /ingest/file, /search
│     ├─ utils_ingest.py      # Unstructured text extraction + chunking
│     ├─ embed_worker.py      # async embedder (polls, batches, writes vectors)
│     └─ rerank_worker.py     # async reranker (writes rank_score)
│
├─ digester/
│  ├─ Dockerfile
│  └─ watcher.py              # watches ./digestion; uploads & auto-deletes
│
├─ digestion/                 # drop files here; removed after ingest
└─ data/                      # optional: extra CSVs / ETL assets
```

---

## Data flow

```mermaid
flowchart LR
A[Drop file in ./digestion] --> B[digester]
B -->|POST /ingest/file| C[polars-worker (FastAPI)]
C -->|extract + chunk| D[(Postgres)]
D <-->|poll pending| E[embedder]
E -->|/api/embed (Ollama)| D
D <-->|poll embedded| F[reranker]
F -->|/api/generate (rerank)| D
G[search client] -->|GET /search?q=...| C
C -->|embed query + ANN + (rank_score blend)| D
C -->|JSON results| G
```

---

## Endpoints

- `GET /healthz` → `{"ok": true, "embed_model": "..."}`  
- `POST /ingest/file` → multipart file upload (PDF/MD/TXT/CSV, etc.)  
- `GET /search?q=...&k=5&metric=cosine` → top-k chunks (uses pgvector ANN)  
  - `metric`: `cosine` (default) or `l2` — must match your index opclass  
  - Optional blending with `rank_score` can be enabled in SQL (see tips)

---

## Quick start

### Option 1: Full System with Chat TUI (Recommended)
```bash
# Single command to start everything
./start.sh                    # Interactive startup script
# OR
make up                       # Using Makefile
# OR
docker compose up --build     # Direct docker compose
```

### Option 2: Headless API-Only System
```bash
# Single command to start API-only system
./start-headless.sh           # Headless startup script
# OR
make headless                 # Using Makefile
# OR
docker compose -f docker-compose.headless.yml up --build -d
```

### Option 3: Traditional API Setup
```bash
# 1) Bring up the stack
docker compose up -d --build

# 2) Confirm services
docker compose ps

# 3) Health check
curl http://localhost:8080/healthz

# 4) Ingest: drop a file
cp some.pdf digestion/         # digester uploads then deletes it

# 5) Watch logs
docker compose logs -f embedder
docker compose logs -f reranker

# 6) Search
curl "http://localhost:8080/search?q=project%20roadmap&k=5"
```

---

## Launch Scripts

The project includes several convenient launch scripts:

### 🚀 Quick Launch Options

| Script | Purpose | Command |
|--------|---------|---------|
| `./chat.sh` | Interactive launcher with options | `./chat.sh` |
| `./quick-chat.sh` | One-liner to start everything | `./quick-chat.sh` |
| `make chat` | Makefile command | `make chat` |
| `make help` | Show all available commands | `make help` |

### 🐍 Local Development

For local development without Docker:
```bash
cd chat-tui
./run-local.sh              # Local Python launcher
# OR
make dev-chat               # Using Makefile
```

### 📋 Available Make Commands

```bash
make help          # Show all commands
make chat          # Start full stack with chat-tui
make quick-chat    # Quick start chat-tui only
make up            # Start all RAG services
make down          # Stop all services
make logs          # Show service logs
make status        # Check system status
make clean         # Clean up everything
```

---

## Configuration (env vars)

Set in `docker-compose.yml` (or move to a `.env` file and reference):

- **DB**: `POSTGRES_DB=ragdb`, `POSTGRES_USER=rag`, `POSTGRES_PASSWORD=ragpass`  
- **Polars/Workers**:  
  - `DATABASE_URL=postgresql://rag:ragpass@db:5432/ragdb`  
  - `OLLAMA_HOST=http://ollama:11434`  
  - `EMBED_MODEL=embedding-gemma` *(must match your Ollama embed model)*  
  - `RERANK_MODEL=all-minilm` *(or `bge-reranker-base`, etc.)*  
  - `EMBED_BATCH=64`, `EMBED_SLEEP=15` (seconds)  
  - `BATCH_SIZE=50`, `SLEEP_SEC=30` for reranker

**Vector dimension**: set in `db/init/02_schema.sql` (`VECTOR(768)`). Match your embed model’s output.

**Distance metric**: index uses `vector_cosine_ops` by default. If you switch to L2, change the index and pass `metric=l2` in `/search`.

---

## Retrieval ranking (blend example)

To blend similarity and precomputed rerank score in `/search`, adjust the ORDER BY:

```sql
-- cosine similarity (1 - distance) * 0.6 + rank_score * 0.4
ORDER BY (0.6 * (1.0 - (e.embedding <=> q.qv)) + 0.4 * COALESCE(e.rank_score, 0)) DESC
```

---

## Operational notes

- **Idempotent init**: SQL files can run repeatedly (safe for redeploys).  
- **Scaling**: Increase `lists` in IVFFLAT index as data grows (≈√N heuristic).  
- **Backpressure**: Tune `EMBED_BATCH`/`SLEEP` to match GPU/CPU capacity.  
- **Security**: Keep `prefilter` SQL out of user control; parameterize if you add filters.  
- **Garbage in/out**: Only *source text* is embedded; model outputs are **not** stored as facts.

---

## Troubleshooting

- Embeddings never finish → check `embedder` logs and `EMBED_MODEL` name matches Ollama.  
- Rerank scores stay NULL → confirm reranker model name and Ollama supports rerank format.  
- Search empty → ensure vectors exist:  
  ```sql
  SELECT COUNT(*) FROM embeddings WHERE embedding IS NOT NULL;
  ```  
- Wrong dimension error → update `VECTOR(<dims>)` in schema to your embed size and recreate.

---

## What this repo **is not**

- A generation server. Your LLM(s) for answering are separate.  
- A doc QA UI. This is the backend ingestion + retrieval core you can wire into any app.

---

**That’s it.** Minimal moving parts, clean separation of concerns, and easy to iterate.
