FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    fastapi==0.128.8 \
    uvicorn==0.39.0 \
    python-multipart==0.0.20 \
    jinja2==3.1.6 \
    aiofiles==25.1.0 \
    sentence-transformers==5.1.2 \
    pandas==2.3.3 \
    pillow==11.3.0

COPY class_create.py training.py inference.py app.py ./
COPY templates/ templates/
COPY static/ static/
COPY trained_models/ trained_models/

RUN mkdir -p uploads

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

ENV OLLAMA_HOST=http://ollama:11434
ENV OLLAMA_MODEL=mistral

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
