FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for clipboard support
RUN apt-get update && apt-get install -y --no-install-recommends \
    xclip \
    xsel \
    wl-clipboard \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set up environment for clipboard access
ENV DISPLAY=:99

# Install Python dependencies
RUN pip install --no-cache-dir \
    textual==0.47.1 \
    httpx==0.27.2 \
    rich==13.7.0 \
    pyperclip==1.8.2

COPY app.py /app/

CMD ["python", "-u", "app.py"]
