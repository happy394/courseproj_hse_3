import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sentence_transformers import SentenceTransformer
from class_create import get_dataloaders


class ProductVisionEncoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(resnet.fc.in_features, embed_size),
            nn.BatchNorm1d(embed_size),
        )

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        return self.projection(features)


def freeze_early_layers(model):
    frozen_count = 0
    trainable_count = 0
    for name, param in model.resnet.named_parameters():
        if 'layer3' not in name and 'layer4' not in name:
            param.requires_grad = False
            frozen_count += 1
        else:
            trainable_count += 1

    for param in model.projection.parameters():
        trainable_count += 1

    print(f'[*] frozen {frozen_count} params (early layers), {trainable_count} params trainable (layer3 + layer4 + projection)')


def train():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'[*] device: {device}')

    train_loader, val_loader = get_dataloaders(csv_path='abo_dataset/cleaned_abo_dataset.csv', batch_size=32)

    text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    embed_size = text_encoder.get_sentence_embedding_dimension()  # 384
    print(f'[*] text embedding size: {embed_size}')

    model = ProductVisionEncoder(embed_size=embed_size).to(device)
    freeze_early_layers(model)

    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    num_epochs = 30
    best_val_loss = float('inf')
    patience = 7
    epochs_no_improve = 0

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

            pos_target = torch.ones(images.size(0), device=device)
            pos_loss = criterion(visual_emb, text_emb, pos_target)

            shuffled_idx = torch.randperm(text_emb.size(0))
            neg_text_emb = text_emb[shuffled_idx]
            neg_target = -torch.ones(images.size(0), device=device)
            neg_loss = criterion(visual_emb, neg_text_emb, neg_target)

            loss = (pos_loss + neg_loss) / 2

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

                pos_target = torch.ones(images.size(0), device=device)
                pos_loss = criterion(visual_emb, text_emb, pos_target)

                shuffled_idx = torch.randperm(text_emb.size(0))
                neg_text_emb = text_emb[shuffled_idx]
                neg_target = -torch.ones(images.size(0), device=device)
                neg_loss = criterion(visual_emb, neg_text_emb, neg_target)

                loss = (pos_loss + neg_loss) / 2
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'--- epoch {epoch+1}/{num_epochs}  train_loss: {avg_train:.4f}  val_loss: {avg_val:.4f}  lr: {current_lr:.6f} ---')

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'trained_models/trained_product_cnn.pth')
            print(f'  -> saved best model (val_loss: {avg_val:.4f})')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'[*] early stopping: no improvement for {patience} epochs')
                break

    print(f'[*] training complete. best val_loss: {best_val_loss:.4f}')


if __name__ == '__main__':
    train()
