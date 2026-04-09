import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel
from class_create import train_loader

class ProductVisionEncoder(nn.Module):
    def __init__(self, embed_size):
        super(ProductVisionEncoder, self).__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.batch_norm = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        visual_embeddings = self.batch_norm(self.linear(features))

        return visual_embeddings

llm_embedding_size = 768
cnn_model = ProductVisionEncoder(embed_size=llm_embedding_size)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f'[*] training on device: {device}')

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
text_encoder = AutoModel.from_pretrained('distilbert-base-uncased').to(device)

for param in text_encoder.parameters():
    param.requires_grad = False

cnn_model = ProductVisionEncoder(embed_size=768).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.0001)

num_epochs = 5

print('[*] starting training...')
for epoch in range(num_epochs):
    cnn_model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        descriptions = batch['description']

        text_inputs = tokenizer(descriptions, padding=True, truncation=True, return_tensors='pt').to(device)

        with torch.no_grad():
            text_outputs = text_encoder(**text_inputs)
            target_embeddings = text_outputs.last_hidden_state[:, 0, :]

        optimizer.zero_grad()
        visual_embeddings = cnn_model(images)

        loss = criterion(visual_embeddings, target_embeddings)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_loader)
    print(f'--- Epoch {epoch+1} completed. Average Loss: {epoch_loss:.4f} ---')

print('[*] training finished')

torch.save(cnn_model.state_dict(), 'trained_models/trained_product_cnn.pth')
print('model saved to trained_product_cnn.pth')
