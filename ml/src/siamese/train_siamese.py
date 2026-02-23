from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from siamese_dataset import create_pairs
from siamese_model import build_siamese_model
import os
import numpy as np
import tensorflow as tf

# ==== PATHS === =
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
GENUINE_PATH = os.path.join(PROJECT_ROOT, "data/raw/CEDAR/genuine")
FORGED_PATH = os.path.join(PROJECT_ROOT, "data/raw/CEDAR/forged")
MODEL_OUTPUT = os.path.join(PROJECT_ROOT, "models/siamese_signature_model.h5")

# ==== CREATE PAIRS === =
pairs, labels = create_pairs(GENUINE_PATH, FORGED_PATH)

if len(pairs) == 0:
    print("❌ No data to train.")
    exit()

# ==== SPLIT DATA === =
X1 = np.array([p[0] for p in pairs])
X2 = np.array([p[1] for p in pairs])
y = np.array(labels)

X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
    X1, X2, y, test_size=0.2, random_state=42
)

# ==== DATA AUGMENTATION === =
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='constant',
    cval=0
)

# ==== BUILD MODEL === =
model = build_siamese_model(input_shape=(128, 128, 1))
print("🚀 Starting training...")

# Apply augmentation to training data
X1_train_aug = next(datagen.flow(X1_train, batch_size=len(X1_train), shuffle=False))
X2_train_aug = next(datagen.flow(X2_train, batch_size=len(X2_train), shuffle=False))

# Combine original and augmented data for better training
X1_train_combined = np.concatenate([X1_train, X1_train_aug], axis=0)
X2_train_combined = np.concatenate([X2_train, X2_train_aug], axis=0)
y_train_combined = np.concatenate([y_train, y_train], axis=0)

history = model.fit(
    [X1_train_combined, X2_train_combined], y_train_combined,
    validation_data=([X1_test, X2_test], y_test),
    batch_size=16,
    epochs=30,
    verbose=1
)

# ==== SAVE MODEL === =
os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
model.save(MODEL_OUTPUT)
print(f"✅ Model saved to {MODEL_OUTPUT}")
