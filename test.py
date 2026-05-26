import torch
import numpy as np
from PIL import Image
from class_create import val_transform
from training import ProductVisionEncoder
from inference import (
    build_vocabulary_index,
    decode_features,
    format_features,
    generate_description,
)

TEST_IMAGE = 'test.jpg'

print("=" * 60)
print("STAGE 1: Loading the trained CNN model")
print("=" * 60)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = ProductVisionEncoder(embed_size=384).to(device)
model.load_state_dict(torch.load('trained_models/trained_product_cnn.pth', map_location=device))
model.eval()

print(f"  Device: {device}")
print(f"  Model: ProductVisionEncoder (ResNet-50 backbone -> 384-dim projection)")
print(f"  Weights: trained_product_cnn.pth")
print(f"  Status: loaded successfully")

print()
print("=" * 60)
print(f"STAGE 2: Encoding image '{TEST_IMAGE}' with CNN")
print("=" * 60)

image = Image.open(TEST_IMAGE).convert("RGB")
print(f"  Original image size: {image.size}")

image_tensor = val_transform(image).unsqueeze(0).to(device)
print(f"  After transforms: tensor shape {image_tensor.shape}")
print(f"    (1 image, 3 color channels, 224x224 pixels)")

with torch.no_grad():
    embedding = model(image_tensor)

embedding_np = embedding.squeeze(0).cpu().numpy()
print(f"  CNN output: {embedding_np.shape[0]}-dimensional embedding vector")
print(f"  First 10 values: {np.array2string(embedding_np[:10], precision=4)}")
print(f"  Min: {embedding_np.min():.4f}  Max: {embedding_np.max():.4f}  "
      f"Mean: {embedding_np.mean():.4f}")

print()
print("=" * 60)
print("STAGE 3: Decoding CNN embedding into visual features")
print("=" * 60)
print("  Building visual vocabulary (colors, materials, shapes, etc.)...")

vocab_entries, vocab_embeddings = build_vocabulary_index()
print(f"  Vocabulary size: {len(vocab_entries)} terms across "
      f"{len(set(cat for cat, _ in vocab_entries))} categories")

decoded = decode_features(embedding_np, vocab_entries, vocab_embeddings)

print()
print("  Decoded features (term + cosine similarity score):")
print("  " + "-" * 50)
print(format_features(decoded))
print("  " + "-" * 50)
print()
print("  Higher scores = CNN is more confident about that attribute.")
print("  These features are what the CNN 'sees' in the image.")

print()
print("=" * 60)
print("STAGE 4: Generating product description with Mistral")
print("=" * 60)
print("  Sending decoded features to Mistral via Ollama...")
print("  (Mistral writes a description grounded in the CNN's visual features)")
print()

try:
    description = generate_description(decoded)
    print("  === GENERATED DESCRIPTION ===")
    print()
    print(f"  {description}")
    print()
    print("  ==============================")
except Exception as e:
    print(f"  ERROR: Could not reach Ollama. Is it running? ({e})")
    print("  Start it with: ollama serve")

print()
print("=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)
print(f"  Image:      {TEST_IMAGE}")
print(f"  CNN output: {embedding_np.shape[0]}-dim vector")

top_type = (decoded.get('product_type') or [('unknown', 0)])[0]
top_color = (decoded.get('color') or [('unknown', 0)])[0]
top_material = (decoded.get('material') or [('unknown', 0)])[0]
print(f"  Top type:   {top_type[0]} (confidence: {top_type[1]:.2f})")
print(f"  Top color:  {top_color[0]} (confidence: {top_color[1]:.2f})")
print(f"  Top material: {top_material[0]} (confidence: {top_material[1]:.2f})")
print(f"  LLM:        Mistral 7B via Ollama")
print("=" * 60)
