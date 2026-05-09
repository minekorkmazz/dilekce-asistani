FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Embedding modeli build sırasında indir
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('trmteb/turkish-embedding-model')"

COPY app.py indexer.py ./