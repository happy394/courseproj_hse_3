import torch
import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from class_create import val_transform
from training import ProductVisionEncoder


def build_description_index(csv_path='abo_dataset/cleaned_abo_dataset.csv'):
    print('[*] building description index...')
    df = pd.read_csv(csv_path)
    descriptions = df['description'].tolist()

    text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    text_embeddings = text_encoder.encode(descriptions, show_progress_bar=True, convert_to_numpy=True)

    print(f'[*] indexed {len(descriptions)} descriptions')
    return descriptions, text_embeddings


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


def retrieve_similar(image_embedding, text_embeddings, descriptions, top_k=5):
    img_norm = image_embedding / (np.linalg.norm(image_embedding) + 1e-8)
    txt_norms = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)

    similarities = txt_norms @ img_norm
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for i in top_indices:
        results.append((descriptions[i], float(similarities[i])))
    return results


def generate_description(similar_products, max_new_tokens=80):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    llm = GPT2LMHeadModel.from_pretrained('gpt2')
    llm.eval()

    examples = '\n'.join(f'- {desc}' for desc, _ in similar_products[:3])
    prompt = (
        f"The following are similar products:\n"
        f"{examples}\n\n"
        f"Write a short, appealing product description for a similar item:\n"
    )

    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        output = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated = full_text[len(prompt):].strip()

    for stop in ['\nProduct:', '\n\n']:
        if stop in generated:
            generated = generated[:generated.index(stop)]

    return generated.strip()


def run_inference(image_path):
    print(f'[*] processing: {image_path}')

    cnn_model, device = load_cnn()
    image_emb = encode_image(image_path, cnn_model, device)
    print(f'[*] image embedding shape: {image_emb.shape}')

    descriptions, text_embeddings = build_description_index()
    similar = retrieve_similar(image_emb, text_embeddings, descriptions)

    print(f'\n[*] top-5 similar products:')
    for desc, score in similar:
        print(f'  {score:.3f}  {desc}')

    print(f'\n[*] generating description...')
    description = generate_description(similar)
    print(f'\n=== GENERATED DESCRIPTION ===')
    print(description)
    print(f'=============================\n')
    return description


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('usage: python inference.py <image_path>')
        sys.exit(1)

    run_inference(sys.argv[1])
