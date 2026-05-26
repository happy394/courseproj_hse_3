import pandas as pd
import glob
import os

print('[*] loading and cleaning dataset')

all_listings = []
listing_files = glob.glob(f'abo_dataset/listings/metadata/listings_*.json.gz')

for file in listing_files:
    df = pd.read_json(file, lines=True)
    all_listings.append(df)

meta = pd.concat(all_listings, ignore_index=True)

def get_en_us_text(name_list):
    if isinstance(name_list, list):
        for item in name_list:
            if item.get('language_tag') == 'en_US':
                return item.get('value')
    return None

meta['description'] = meta['item_name'].apply(get_en_us_text)
meta = meta.dropna(subset=['description', 'main_image_id'])
meta = meta[['item_id', 'description', 'main_image_id']]
image_meta = pd.read_csv(f'abo_dataset/images/metadata/images.csv.gz')

dataset = meta.merge(image_meta, left_on='main_image_id', right_on='image_id')
dataset['local_image_path'] = 'abo_dataset/images/small/' + dataset['path']
dataset = dataset[dataset['local_image_path'].apply(os.path.exists)]

cleaned_data = dataset[['item_id', 'description', 'local_image_path']]
print(f'[*] cleaning complete. total image-text pairs: {len(cleaned_data)}')

cleaned_data.to_csv('abo_dataset/cleaned_abo_dataset.csv', index=False)
