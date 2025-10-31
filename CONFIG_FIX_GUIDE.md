# Configuration Fix Guide
**Priority:** HIGH  
**Time Required:** 30 minutes  
**Impact:** Ensures system works across different environments

---

## Issue: OLLAMA_HOST Configuration Inconsistency

### Current State

The system currently has **three different OLLAMA_HOST configurations** across files:

1. **docker-compose.yml** (main file):
   ```yaml
   OLLAMA_HOST: http://host.docker.internal:11434
   ```
   - Used by: polars-worker, embedder, reranker, chat-tui
   - Works when: Ollama is running on the Docker host machine

2. **docker-compose.headless.yml**:
   ```yaml
   OLLAMA_HOST: http://192.168.7.215:11434
   ```
   - Hardcoded IP address
   - Not portable across environments

3. **Python file defaults** (service.py, embed_worker.py, rerank_worker.py, app.py):
   ```python
   OLLAMA = os.getenv("OLLAMA_HOST", "http://192.168.7.215:11434")
   ```
   - Hardcoded fallback IP

---

## Recommended Solution

### Option A: Use Environment Variables (Recommended)

**Pros:**
- Most flexible
- Easy to override per environment
- No code changes needed

**Implementation:**

1. Create a `.env` file in the project root:
   ```bash
   cp env.example .env
   ```

2. Edit `.env` with your Ollama location:
   ```bash
   # For Ollama on host machine
   OLLAMA_HOST=http://host.docker.internal:11434
   
   # OR for Ollama on specific IP
   # OLLAMA_HOST=http://192.168.7.215:11434
   ```

3. Update `docker-compose.yml` to use environment variables:
   ```yaml
   environment:
     OLLAMA_HOST: ${OLLAMA_HOST:-http://host.docker.internal:11434}
   ```

4. Update `docker-compose.headless.yml` similarly:
   ```yaml
   environment:
     OLLAMA_HOST: ${OLLAMA_HOST:-http://host.docker.internal:11434}
   ```

### Option B: Standardize on host.docker.internal (Quick Fix)

**Pros:**
- Works out of the box on Docker Desktop (Mac/Windows/Linux)
- No environment variables needed
- One-time fix

**Cons:**
- Requires Ollama to be on the same machine as Docker

**Implementation:**

Update `docker-compose.headless.yml` to match `docker-compose.yml`:
```yaml
environment:
  OLLAMA_HOST: http://host.docker.internal:11434
```

---

## Step-by-Step Fix (Option A - Recommended)

### 1. Update docker-compose.yml

Find all instances of `OLLAMA_HOST` and update them:

```yaml
services:
  polars-worker:
    environment:
      OLLAMA_HOST: ${OLLAMA_HOST:-http://host.docker.internal:11434}
      
  embedder:
    environment:
      OLLAMA_HOST: ${OLLAMA_HOST:-http://host.docker.internal:11434}
      
  reranker:
    environment:
      OLLAMA_HOST: ${OLLAMA_HOST:-http://host.docker.internal:11434}
      
  chat-tui:
    environment:
      OLLAMA_HOST: ${OLLAMA_HOST:-http://host.docker.internal:11434}
```

### 2. Update docker-compose.headless.yml

Same changes:

```yaml
services:
  polars-worker:
    environment:
      OLLAMA_HOST: ${OLLAMA_HOST:-http://host.docker.internal:11434}
      
  embedder:
    environment:
      OLLAMA_HOST: ${OLLAMA_HOST:-http://host.docker.internal:11434}
      
  reranker:
    environment:
      OLLAMA_HOST: ${OLLAMA_HOST:-http://host.docker.internal:11434}
```

### 3. Update Python Defaults (Optional but Recommended)

Update hardcoded defaults in Python files to match:

**polars/app/service.py** (line 94):
```python
ollama_host = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
```

**polars/app/embed_worker.py** (line 25):
```python
OLLAMA = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
```

**polars/app/rerank_worker.py** (line 32):
```python
OLLAMA = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
```

**chat-tui/app.py** (line 39):
```python
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
```

### 4. Create .env File

```bash
# Copy the example
cp env.example .env

# Edit if your Ollama is on a different host
# nano .env  # or vim, code, etc.
```

### 5. Test the Configuration

```bash
# Stop any running services
docker compose down

# Start with the new configuration
docker compose up --build

# In another terminal, test the connection
curl http://localhost:8080/healthz

# Check if Ollama is reachable from inside a container
docker compose exec polars-worker curl http://host.docker.internal:11434/api/tags
```

---

## Verification Steps

### 1. Check Environment Variables are Loaded

```bash
# View the effective configuration
docker compose config | grep OLLAMA_HOST
```

Expected output:
```
OLLAMA_HOST: http://host.docker.internal:11434
```

### 2. Test Embedding Endpoint

```bash
# Test that embedder can reach Ollama
docker compose logs embedder | tail -20
```

Should see:
```
⚙️ Embed worker started (batch=64, sleep=15s, timeout=300s)
```

NOT errors like:
```
🌐 HTTP error communicating with Ollama: ...
```

### 3. Test Search Endpoint

```bash
# Upload a test document
echo "Test document about RAG systems" > test.txt
cp test.txt digestion/

# Wait a few seconds for processing
sleep 10

# Search for it
curl "http://localhost:8080/search?q=RAG&k=5"
```

---

## Deployment-Specific Configurations

### Local Development (Ollama on Host)
```env
OLLAMA_HOST=http://host.docker.internal:11434
```

### Server Deployment (Ollama on Same Machine)
```env
OLLAMA_HOST=http://host.docker.internal:11434
```

### Server Deployment (Ollama on Different Machine)
```env
OLLAMA_HOST=http://192.168.7.215:11434
```

### Kubernetes Deployment
```env
OLLAMA_HOST=http://ollama-service.default.svc.cluster.local:11434
```

---

## Troubleshooting

### Error: "Connection refused" or "Unable to reach Ollama"

**Check 1:** Is Ollama running?
```bash
# On the machine where Ollama is installed
curl http://localhost:11434/api/tags
```

**Check 2:** Is the hostname/IP correct?
```bash
# From your local machine
curl http://192.168.7.215:11434/api/tags
```

**Check 3:** Can Docker containers reach it?
```bash
# From inside a container
docker compose exec polars-worker curl $OLLAMA_HOST/api/tags
```

**Check 4:** Firewall blocking?
- Ensure port 11434 is open on the Ollama host
- Check Docker network can reach external IPs

### Error: "host.docker.internal: Name or service not known"

**Solution:** You're on Linux, and `host.docker.internal` might not be configured.

**Fix:** Use explicit IP address:
```env
# In .env file
OLLAMA_HOST=http://192.168.7.215:11434
```

Or add to docker-compose.yml:
```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

### Models Not Found

**Error:** `Model "embeddinggemma" not found`

**Solution:**
```bash
# On the Ollama machine
ollama pull embeddinggemma
ollama pull all-minilm
ollama pull llama3.2
```

---

## Summary

After applying these fixes:

✅ **Configuration is consistent** across all files  
✅ **Environment-based** (easy to customize per deployment)  
✅ **Portable** (works in dev, staging, production)  
✅ **Documented** (env.example explains all options)  
✅ **Maintainable** (single source of truth: .env file)

**Time to implement:** 20-30 minutes  
**Difficulty:** Easy  
**Impact:** High (enables cross-environment deployment)

