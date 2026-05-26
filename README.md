# Product Description Generator

A two-stage pipeline that generates e-commerce product descriptions from photos.

1. A ResNet-50 vision encoder maps the image into a shared embedding space with Sentence-BERT
2. A visual vocabulary (~100 terms, 6 categories) decodes the embedding into structured attributes
3. Mistral 7B (via Ollama) writes a catalog-style description from those attributes

Trained on the Amazon Berkeley Objects (ABO) dataset.

## Quick start (Docker)

```bash
docker compose up -d --build
```

This builds the web app, starts Ollama, and pulls the Mistral model automatically.
Open `http://localhost:8000` once all services are up.

## Quick start (local)

```bash
pip install -r requirements.txt
ollama pull mistral
ollama serve &
python app.py
```

## Project structure

| File | Description |
|------|-------------|
| `app.py` | FastAPI web application |
| `training.py` | Vision encoder model + training loop |
| `inference.py` | Visual vocabulary + feature decoding |
| `class_create.py` | Dataset class + image transforms |
| `dataset_clean.py` | ABO dataset cleaning script |
| `test.py` | Step-by-step inference demo |
| `Dockerfile` | Container build for the web app |
| `docker-compose.yml` | Web + Ollama services |
| `deploy/` | Self-contained deployment folder with setup script |

## Tests

```bash
python -m pytest tests/
```
