
1️⃣ Drop file → /digestion/
   ↓
2️⃣ digester → POST /ingest/file → polars-worker
   ↓
3️⃣ polars-worker:
       - Extract text (Unstructured)
       - Chunk text
       - Store in Postgres (embedding=NULL)
   ↓
4️⃣ embedder (async):
       - Pull unembedded chunks
       - Call Ollama embed model
       - Store vectors
   ↓
5️⃣ reranker (async):
       - Pull embedded, unranked chunks
       - Call Ollama rerank model
       - Update rank_score
   ↓
6️⃣ search endpoint (user query):
       - Embed query → ANN search + rank_score weighting
       - Return top-k chunks
