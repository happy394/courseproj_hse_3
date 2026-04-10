import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sentence_transformers import SentenceTransformer
from class_create import get_dataloaders

class ProductVisionEncoder(nn.Module):
    def __init__(self, embed_size):
        super(ProductVisionEncoder, self).__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.projection = nn.Sequential(
            nn.Linear(resnet.fc.in_features, embed_size),
            nn.BatchNorm1d(embed_size),
        )

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        return self.projection(features)


def train():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'[*] training on device: {device}')

    train_loader, val_loader = get_dataloaders(batch_size=32)

    text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    embed_size = text_encoder.get_sentence_embedding_dimension()  # 384
    print(f'[*] text embedding size: {embed_size}')

    model = ProductVisionEncoder(embed_size=embed_size).to(device)

    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    num_epochs = 10
    best_val_loss = float('inf')

    print('[*] starting training...')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            descriptions = batch['description']

            text_emb = text_encoder.encode(
                descriptions, convert_to_tensor=True
            ).to(device).clone()

            visual_emb = model(images)

            target = torch.ones(images.size(0), device=device)
            loss = criterion(visual_emb, text_emb, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f'  epoch {epoch+1}, batch {batch_idx+1}/{len(train_loader)}, loss: {loss.item():.4f}')

        avg_train = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                descriptions = batch['description']

                text_emb = text_encoder.encode(
                    descriptions, convert_to_tensor=True
                ).to(device).clone()

                visual_emb = model(images)
                target = torch.ones(images.size(0), device=device)
                loss = criterion(visual_emb, text_emb, target)
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'--- epoch {epoch+1}/{num_epochs}  train_loss: {avg_train:.4f}  val_loss: {avg_val:.4f}  lr: {current_lr:.6f} ---')

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'trained_models/trained_product_cnn.pth')
            print(f'[*] saved best model (val_loss: {avg_val:.4f})')

    print(f'[*] training complete. best val_loss: {best_val_loss:.4f}')


if __name__ == '__main__':
    train()
