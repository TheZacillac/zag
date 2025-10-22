FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for:
# - Postgres (libpq-dev)
# - PDF/image processing (poppler-utils, libmagic1)
# - OpenCV (libgl1)
# - Misc unstructured extras (tesseract optional)
# - curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    poppler-utils \
    libmagic1 \
    libgl1 \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.114 uvicorn[standard]==0.30 \
    psycopg[binary]==3.2.1 psycopg-pool==3.2.1 \
    httpx==0.27.2 \
    polars==1.9.0 pyarrow==17.0.0 \
    "unstructured[local-inference,pdf,docx,md,pptx]==0.14.4" \
    pdfminer.six==20231228 pillow==10.4.0 \
    python-multipart==0.0.9 \
    nltk

# Download required NLTK data
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"

COPY app /app

EXPOSE 8080
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8080"]
