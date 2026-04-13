import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ECommerceProductDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row['local_image_path']).convert('RGB')
        image = self.transform(image)
        return {
            'image': image,
            'description': row['description']
        }


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def get_dataloaders(csv_path=None, batch_size=32,
                    val_ratio=0.15, seed=42):
    df = pd.read_csv(csv_path)

    total = len(df)
    val_size = int(total * val_ratio)
    train_size = total - val_size

    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:].copy()

    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    train_dataset = ECommerceProductDataset(train_df, transform=train_transform)
    val_dataset = ECommerceProductDataset(val_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f'[*] dataset loaded: {train_size} train / {val_size} val samples')
    return train_loader, val_loader


if __name__ == '__main__':
    train_loader, val_loader = get_dataloaders(csv_path='abo_dataset/cleaned_abo_dataset.csv')
    for batch in train_loader:
        print(f'batch image tensor shape: {batch["image"].shape}')
        print(f'sample description: {batch["description"][0]}')
        break
