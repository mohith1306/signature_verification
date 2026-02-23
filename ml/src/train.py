# src/train.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from src.model_training import SiameseNetwork
from data_loader import SignaturePairDataset
from config import *
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

dataset = SignaturePairDataset(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

def contrastive_loss(output1, output2, label, margin=1.0):
    distance = F.pairwise_distance(output1, output2)
    return torch.mean((1 - label) * torch.pow(distance, 2) +
                      label * torch.pow(torch.clamp(margin - distance, min=0.0), 2))

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for img1, img2, label in tqdm(loader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        out1, out2 = model(img1, img2)
        loss = contrastive_loss(out1, out2, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss/len(loader):.4f}")

torch.save(model.state_dict(), "models/siamese_model.pth")
print("✅ Model trained and saved.")
