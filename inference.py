import torch
import json
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import urllib.request
from class_create import val_transform
from training import ProductVisionEncoder


VISUAL_VOCABULARY = {
    'product_type': [
        'sofa', 'couch', 'chair', 'table', 'desk', 'bed', 'lamp', 'shelf',
        'cabinet', 'dresser', 'bench', 'stool', 'ottoman', 'rug', 'curtains',
        'earrings', 'necklace', 'bracelet', 'ring', 'watch', 'jewelry',
        'shoes', 'sneakers', 'boots', 'sandals', 'heels', 'flats',
        'shirt', 'dress', 'jacket', 'pants', 'sweater', 'coat', 'hat',
        'bottle', 'jar', 'can', 'box', 'bag', 'container', 'cup', 'mug',
        'phone case', 'headphones', 'speaker', 'keyboard', 'mouse',
        'toy', 'doll', 'puzzle', 'game', 'ball',
        'candle', 'vase', 'frame', 'mirror', 'clock', 'pillow', 'blanket',
        'pan', 'pot', 'knife', 'cutting board', 'plate', 'bowl',
        'shampoo', 'cream', 'lotion', 'perfume', 'soap', 'oil',
        'supplement', 'vitamins', 'protein powder',
        'tool', 'wrench', 'drill', 'screwdriver', 'tape',
    ],
    'color': [
        'red', 'blue', 'green', 'black', 'white', 'grey', 'silver',
        'gold', 'brown', 'beige', 'cream', 'pink', 'purple', 'orange',
        'yellow', 'navy', 'teal', 'burgundy', 'ivory', 'tan',
        'multicolor', 'transparent', 'dark', 'light', 'bright', 'pastel',
    ],
    'material': [
        'leather', 'fabric', 'cotton', 'silk', 'wool', 'linen', 'velvet',
        'wood', 'bamboo', 'metal', 'steel', 'aluminum', 'iron', 'brass',
        'glass', 'crystal', 'ceramic', 'porcelain', 'stone', 'marble',
        'plastic', 'rubber', 'silicone', 'acrylic',
        'sterling silver', 'gold plated', 'stainless steel',
    ],
    'style': [
        'modern', 'contemporary', 'classic', 'vintage', 'retro',
        'minimalist', 'rustic', 'industrial', 'bohemian', 'elegant',
        'casual', 'formal', 'sporty', 'luxury', 'mid-century',
        'traditional', 'farmhouse', 'scandinavian', 'art deco',
    ],
    'texture_finish': [
        'smooth', 'rough', 'matte', 'glossy', 'polished', 'brushed',
        'textured', 'woven', 'knitted', 'embroidered', 'quilted',
        'distressed', 'satin', 'frosted', 'hammered',
    ],
    'shape': [
        'round', 'rectangular', 'square', 'oval', 'cylindrical',
        'slim', 'compact', 'large', 'small', 'tall', 'wide', 'narrow',
        'curved', 'angular', 'flat', 'tapered',
    ],
}


def build_vocabulary_index():
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2')

    vocab_entries = []
    phrases = []

    for category, terms in VISUAL_VOCABULARY.items():
        for term in terms:
            if category == 'product_type':
                phrase = f"a {term}"
            elif category == 'color':
                phrase = f"a product that is {term} colored"
            elif category == 'material':
                phrase = f"a product made of {term}"
            elif category == 'style':
                phrase = f"a {term} style product"
            elif category == 'texture_finish':
                phrase = f"a product with {term} finish"
            elif category == 'shape':
                phrase = f"a {term} shaped product"
            else:
                phrase = term

            vocab_entries.append((category, term))
            phrases.append(phrase)

    embeddings = text_encoder.encode(phrases, convert_to_numpy=True)

    return vocab_entries, embeddings


def decode_features(image_embedding, vocab_entries, vocab_embeddings, top_per_category=3):
    img_norm = image_embedding / (np.linalg.norm(image_embedding) + 1e-8)
    vocab_norms = vocab_embeddings / (
        np.linalg.norm(vocab_embeddings, axis=1, keepdims=True) + 1e-8
    )

    similarities = vocab_norms @ img_norm

    from collections import defaultdict
    category_scores = defaultdict(list)

    for i, (category, term) in enumerate(vocab_entries):
        category_scores[category].append((term, float(similarities[i])))

    decoded = {}
    for category, scored_terms in category_scores.items():
        scored_terms.sort(key=lambda x: x[1], reverse=True)
        top = [(term, score) for term, score in scored_terms[:top_per_category]
               if score > 0.1]
        decoded[category] = top

    return decoded


def format_features(decoded_features):
    lines = []
    category_labels = {
        'product_type': 'Product type',
        'color': 'Color',
        'material': 'Material',
        'style': 'Style',
        'texture_finish': 'Texture/Finish',
        'shape': 'Shape/Size',
    }

    for category, label in category_labels.items():
        if category in decoded_features and decoded_features[category]:
            terms = [f"{term} ({score:.2f})" for term, score in decoded_features[category]]
            lines.append(f"  {label}: {', '.join(terms)}")

    return '\n'.join(lines)


def generate_description(decoded_features):
    feature_parts = []
    category_labels = {
        'product_type': 'Product type',
        'color': 'Color',
        'material': 'Material',
        'style': 'Style',
        'texture_finish': 'Texture/Finish',
        'shape': 'Shape/Size',
    }
    for category, label in category_labels.items():
        if category in decoded_features and decoded_features[category]:
            terms = [term for term, _ in decoded_features[category]]
            feature_parts.append(f"- {label}: {', '.join(terms)}")

    features_text = '\n'.join(feature_parts)

    prompt = (
        "You are an e-commerce copywriter. Based on the visual features detected "
        "in a product image by a computer vision model, write a short, appealing "
        "product description (3-4 sentences). Focus on the product's appearance, "
        "materials, and style. Do not invent brand names or prices. "
        "Do not mention that features were detected by a model.\n\n"
        f"Detected visual features:\n{features_text}\n\n"
        "Product description:"
    )

    request_data = json.dumps({
        'model': 'mistral',
        'prompt': prompt,
        'stream': False,
        'options': {
            'temperature': 0.7,
            'top_p': 0.9,
            'num_predict': 200,
        }
    }).encode('utf-8')

    req = urllib.request.Request(
        'http://localhost:11434/api/generate',
        data=request_data,
        headers={'Content-Type': 'application/json'},
    )

    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))

    return result['response'].strip()


def load_cnn(weights_path='trained_models/trained_product_cnn.pth', embed_size=384):
    device = torch.device('cpu')
    model = ProductVisionEncoder(embed_size=embed_size)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device


def encode_image(image_path, model, device):
    image = Image.open(image_path).convert('RGB')
    image_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(image_tensor)

    return embedding.squeeze(0).numpy()


def run_inference(image_path):
    print(f'[*] processing: {image_path}')

    cnn_model, device = load_cnn()
    image_emb = encode_image(image_path, cnn_model, device)
    print(f'[*] image embedding: {image_emb.shape}')

    print('[*] decoding visual features...')
    vocab_entries, vocab_embeddings = build_vocabulary_index()
    decoded = decode_features(image_emb, vocab_entries, vocab_embeddings)

    print(f'\n[*] decoded features from CNN:')
    print(format_features(decoded))

    print(f'\n[*] generating description with Mistral...')
    description = generate_description(decoded)
    print(f'\n=== GENERATED DESCRIPTION ===')
    print(description)
    print(f'=============================\n')
    return description


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('usage: python inference.py <image_path>')
        print('example: python inference.py abo_dataset/images/small/83/83222ad0.jpg')
        sys.exit(1)

    run_inference(sys.argv[1])
