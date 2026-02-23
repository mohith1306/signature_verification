# siamese_model/siamese_preprocess.py
import cv2
import numpy as np
import os

def preprocess_image(img_path, target_size=(128, 128)):
    """Reads and preprocesses a signature image."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    return img
