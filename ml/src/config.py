# src/config.py
import os

DATA_DIR = os.path.join("data", "processed")
PAIR_DIR = os.path.join("data", "pairs")

IMAGE_SIZE = (155, 220)  # resize all images
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
THRESHOLD = 0.85  # similarity threshold for match
