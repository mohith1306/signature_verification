# src/dataset_loader.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class SignaturePairDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.users = os.listdir(data_dir)

    def __getitem__(self, idx):
        user = random.choice(self.users)
        user_path = os.path.join(self.data_dir, user)
        genuine = [f for f in os.listdir(user_path) if "g" in f]
        forged = [f for f in os.listdir(user_path) if "f" in f]

        # 50% genuine-genuine, 50% genuine-forged
        if random.random() < 0.5:
            img1, img2 = random.sample(genuine, 2)
            label = 1.0
        else:
            img1 = random.choice(genuine)
            img2 = random.choice(forged)
            label = 0.0

        img1 = Image.open(os.path.join(user_path, img1)).convert('L')
        img2 = Image.open(os.path.join(user_path, img2)).convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

    def __len__(self):
        return 10000  # arbitrary large number for sampling
