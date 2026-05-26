import os
import io
import csv
import uuid
import time
import json
import urllib.request
from typing import List

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer

from class_create import val_transform
from training import ProductVisionEncoder
from inference import (
    build_vocabulary_index,
    decode_features,
    VISUAL_VOCABULARY,
)

UPLOAD_DIR = 'uploads'
MODEL_PATH = 'trained_models/trained_product_cnn.pth'
EMBED_SIZE = 384
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_URL = f'{OLLAMA_HOST}/api/generate'
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'mistral')
CACHE_FILE = 'cache/generation_cache.json'

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)


def load_cache() -> list:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return []


def save_to_cache(entry: dict):
    cache = load_cache()
    cache.insert(0, entry)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

app = FastAPI(title='Product Description Generator')
app.mount('/static', StaticFiles(directory='static'), name='static')
app.mount('/uploads', StaticFiles(directory='uploads'), name='uploads')
templates = Jinja2Templates(directory='templates')

print('[*] Loading CNN model...')
device = torch.device('cpu')
cnn_model = ProductVisionEncoder(embed_size=EMBED_SIZE)
cnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
cnn_model.eval()
print('[*] CNN model loaded')

print('[*] Loading text encoder & vocabulary index...')
text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
vocab_entries, vocab_embeddings = build_vocabulary_index()
print(f'[*] Vocabulary: {len(vocab_entries)} terms ready')


def encode_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert('RGB')
    image_tensor = val_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = cnn_model(image_tensor)
    return embedding.squeeze(0).numpy()


def generate_description_ollama(decoded_features: dict) -> str:
    category_labels = {
        'product_type': 'Product type',
        'color': 'Color',
        'material': 'Material',
        'style': 'Style',
        'texture_finish': 'Texture/Finish',
        'shape': 'Shape/Size',
    }
    feature_parts = []
    for category, label in category_labels.items():
        if category in decoded_features and decoded_features[category]:
            terms = [term for term, _ in decoded_features[category]]
            feature_parts.append(f"- {label}: {', '.join(terms)}")

    features_text = '\n'.join(feature_parts)

    prompt = (
        'You are an e-commerce copywriter. Based on the visual features detected '
        'in a product image by a computer vision model, write a short, appealing '
        "product description (3-4 sentences). Focus on the product's appearance, "
        'materials, and style. Do not invent brand names or prices. '
        'Do not mention that features were detected by a model.\n\n'
        f'Detected visual features:\n{features_text}\n\n'
        'Product description:'
    )

    request_data = json.dumps({
        'model': OLLAMA_MODEL,
        'prompt': prompt,
        'stream': False,
        'options': {
            'temperature': 0.7,
            'top_p': 0.9,
            'num_predict': 200,
        },
    }).encode('utf-8')

    req = urllib.request.Request(
        OLLAMA_URL,
        data=request_data,
        headers={'Content-Type': 'application/json'},
    )

    with urllib.request.urlopen(req, timeout=60) as response:
        result = json.loads(response.read().decode('utf-8'))

    return result['response'].strip()


def check_ollama() -> bool:
    try:
        req = urllib.request.Request(f'{OLLAMA_HOST}/api/tags')
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {
        'request': request,
        'ollama_available': check_ollama(),
    })


@app.post('/analyze')
async def analyze_image(file: UploadFile = File(...)):
    start = time.time()

    ext = os.path.splitext(file.filename or 'img.jpg')[1].lower()
    if ext not in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'):
        return JSONResponse({'error': 'Unsupported image format. Use JPG, PNG, or WebP.'}, status_code=400)

    filename = f'{uuid.uuid4().hex}{ext}'
    filepath = os.path.join(UPLOAD_DIR, filename)

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        return JSONResponse({'error': 'Image too large (max 10 MB).'}, status_code=400)

    with open(filepath, 'wb') as f:
        f.write(contents)

    try:
        image_emb = encode_image(filepath)
    except Exception as e:
        os.remove(filepath)
        return JSONResponse({'error': f'Failed to process image: {e}'}, status_code=500)

    decoded = decode_features(image_emb, vocab_entries, vocab_embeddings)

    features = {}
    category_labels = {
        'product_type': 'Product Type',
        'color': 'Color',
        'material': 'Material',
        'style': 'Style',
        'texture_finish': 'Texture / Finish',
        'shape': 'Shape / Size',
    }
    for cat, label in category_labels.items():
        if cat in decoded and decoded[cat]:
            features[label] = [
                {'term': term, 'score': round(score, 3)}
                for term, score in decoded[cat]
            ]

    description = None
    ollama_error = None
    if check_ollama():
        try:
            description = generate_description_ollama(decoded)
        except Exception as e:
            ollama_error = str(e)
    else:
        ollama_error = 'Ollama is not running. Start it with: ollama serve'

    elapsed = round(time.time() - start, 2)

    result = {
        'image_url': f'/uploads/{filename}',
        'features': features,
        'description': description,
        'ollama_error': ollama_error,
        'embedding_dim': int(image_emb.shape[0]),
        'elapsed_seconds': elapsed,
    }

    save_to_cache({
        'id': filename,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'source': 'upload',
        'original_filename': file.filename,
        **result,
    })

    return JSONResponse(result)


@app.get('/history')
async def get_history():
    return JSONResponse({'items': load_cache()})


@app.delete('/history/{item_id}')
async def delete_history_item(item_id: str):
    cache = load_cache()
    cache = [e for e in cache if e.get('id') != item_id]
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)
    return JSONResponse({'ok': True})


@app.delete('/history')
async def clear_history():
    with open(CACHE_FILE, 'w') as f:
        json.dump([], f)
    return JSONResponse({'ok': True})


CATEGORY_LABELS = {
    'product_type': 'Product Type',
    'color': 'Color',
    'material': 'Material',
    'style': 'Style',
    'texture_finish': 'Texture / Finish',
    'shape': 'Shape / Size',
}

MAX_BATCH = 25


def build_features_dict(decoded: dict) -> dict:
    features = {}
    for cat, label in CATEGORY_LABELS.items():
        if cat in decoded and decoded[cat]:
            features[label] = [
                {'term': term, 'score': round(score, 3)}
                for term, score in decoded[cat]
            ]
    return features


def process_single_image(filepath: str) -> dict:
    image_emb = encode_image(filepath)
    decoded = decode_features(image_emb, vocab_entries, vocab_embeddings)
    features = build_features_dict(decoded)

    description = None
    ollama_error = None
    if check_ollama():
        try:
            description = generate_description_ollama(decoded)
        except Exception as e:
            ollama_error = str(e)
    else:
        ollama_error = 'Ollama is not running'

    return {
        'features': features,
        'description': description,
        'ollama_error': ollama_error,
        'embedding_dim': int(image_emb.shape[0]),
    }


@app.post('/batch-analyze')
async def batch_analyze(files: List[UploadFile] = File(...)):
    if len(files) > MAX_BATCH:
        return JSONResponse(
            {'error': f'Too many files. Maximum is {MAX_BATCH}.'},
            status_code=400,
        )

    allowed_ext = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    results = []
    batch_id = uuid.uuid4().hex[:8]

    for idx, file in enumerate(files):
        ext = os.path.splitext(file.filename or 'img.jpg')[1].lower()
        if ext not in allowed_ext:
            results.append({
                'index': idx,
                'filename': file.filename,
                'error': f'Unsupported format: {ext}',
            })
            continue

        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            results.append({
                'index': idx,
                'filename': file.filename,
                'error': 'File too large (max 10 MB)',
            })
            continue

        saved_name = f'{uuid.uuid4().hex}{ext}'
        filepath = os.path.join(UPLOAD_DIR, saved_name)
        with open(filepath, 'wb') as f:
            f.write(contents)

        start = time.time()
        try:
            result = process_single_image(filepath)
            elapsed = round(time.time() - start, 2)
            entry = {
                'index': idx,
                'filename': file.filename,
                'image_url': f'/uploads/{saved_name}',
                'elapsed_seconds': elapsed,
                **result,
            }
            results.append(entry)

            save_to_cache({
                'id': saved_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source': f'batch-{batch_id}',
                'original_filename': file.filename,
                'image_url': f'/uploads/{saved_name}',
                'features': result['features'],
                'description': result['description'],
                'ollama_error': result['ollama_error'],
                'embedding_dim': result['embedding_dim'],
                'elapsed_seconds': elapsed,
            })
        except Exception as e:
            os.remove(filepath)
            results.append({
                'index': idx,
                'filename': file.filename,
                'error': str(e),
            })

    return JSONResponse({'batch_id': batch_id, 'total': len(files), 'results': results})


@app.post('/export-csv')
async def export_csv(request: Request):
    body = await request.json()
    rows = body.get('results', [])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['filename', 'description'])

    for row in rows:
        writer.writerow([
            row.get('filename', ''),
            row.get('description') or row.get('error') or row.get('ollama_error') or '',
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type='text/csv',
        headers={"Content-Disposition": f"attachment; filename=batch_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"},
    )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=True)
