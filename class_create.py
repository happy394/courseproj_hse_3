import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ECommerceProductDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

    def __len__(self): return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['local_image_path']
        description = row['description']

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'description': description
        }


cleaned_data = 'abo_dataset/cleaned_abo_dataset.csv'
cleaned_data_df = pd.read_csv(cleaned_data)
train_dataset = ECommerceProductDataset(dataframe=cleaned_data_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for batch in train_loader:
    images = batch['image']
    descriptions = batch['description']
    print(f'batch image tensor shape: {images.shape}') # should be [32, 3, 224, 224]
    print(f'sample description: {descriptions[0]}')
    break
