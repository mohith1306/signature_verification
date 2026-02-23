# src/data_preprocessing.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = "data/raw/BHSig"
PROCESSED_DATA_PATH = "data/processed"

def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label)
    return images, labels

def preprocess():
    genuine_imgs, genuine_labels = load_images(os.path.join(RAW_DATA_PATH, "genuine"), 0)
    forged_imgs, forged_labels = load_images(os.path.join(RAW_DATA_PATH, "forged"), 1)

    X = np.array(genuine_imgs + forged_imgs)
    y = np.array(genuine_labels + forged_labels)

    X = X / 255.0  # normalize
    X = np.expand_dims(X, axis=-1)  # (128,128,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    np.savez(os.path.join(PROCESSED_DATA_PATH, "train_data_1.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(PROCESSED_DATA_PATH, "test_data_1.npz"), X=X_test, y=y_test)

    print("✅ Data preprocessing complete! Saved processed files in data/processed/")

if __name__ == "__main__":
    preprocess()
