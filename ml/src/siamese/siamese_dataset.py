import os
import cv2
import numpy as np
from tensorflow.keras.utils import img_to_array

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                img = img_to_array(img) / 255.0
                images.append(img)
    return np.array(images)

def create_pairs(genuine_path, forged_path):
    print("📂 Loading dataset...")
    genuine_images = load_images_from_folder(genuine_path)
    forged_images = load_images_from_folder(forged_path)

    print(f"✅ Loaded {len(genuine_images)} genuine and {len(forged_images)} forged images")

    if len(genuine_images) == 0 or len(forged_images) == 0:
        print("❌ ERROR: No images found! Check your folder paths.")
        return np.array([]), np.array([])

    # Positive pairs (same person)
    pos_pairs = []
    for i in range(len(genuine_images) - 1):
        pos_pairs.append([genuine_images[i], genuine_images[i + 1]])

    # Negative pairs (different person)
    neg_pairs = []
    min_len = min(len(genuine_images), len(forged_images))
    for i in range(min_len):
        neg_pairs.append([genuine_images[i], forged_images[i]])

    pairs = np.array(pos_pairs + neg_pairs)
    labels = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))

    print(f"✅ Created {len(pairs)} pairs (pos: {len(pos_pairs)}, neg: {len(neg_pairs)})")
    return pairs, labels
