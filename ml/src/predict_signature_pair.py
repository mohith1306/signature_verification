import os
import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from siamese.siamese_model import AbsoluteDifference

MODEL_PATH = "models/siamese_signature_model.h5"

def preprocess_image(image_path):
    """Load and preprocess a signature image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Resize to model input size (128x128)
    img = cv2.resize(img, (128, 128))
    
    # Normalize pixel values to [0, 1]
    img = img.astype("float32") / 255.0
    
    # Add channel dimension (128, 128, 1)
    img = np.expand_dims(img, axis=-1)
    
    # Add batch dimension (1, 128, 128, 1)
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_similarity(sig1_path, sig2_path):
    """Compare two signature images"""
    print(f"\n🔍 Comparing signatures:")
    print(f"  Signature 1: {sig1_path}")
    print(f"  Signature 2: {sig2_path}")
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    model = load_model(MODEL_PATH, custom_objects={"AbsoluteDifference": AbsoluteDifference})
    
    # Preprocess both images
    img1 = preprocess_image(sig1_path)
    img2 = preprocess_image(sig2_path)
    
    # Predict similarity
    similarity_score = model.predict([img1, img2], verbose=0)[0][0]
    
    # Interpret result
    print(f"\n📊 Results:")
    print(f"  Similarity Score: {similarity_score:.4f}")
    print(f"  Threshold: 0.5000")
    
    if similarity_score > 0.5:
        confidence = similarity_score * 100
        print(f"  ✅ GENUINE MATCH (Confidence: {confidence:.2f}%)")
        print(f"  → These signatures likely belong to the same person")
    else:
        confidence = (1 - similarity_score) * 100
        print(f"  ❌ FORGED/DIFFERENT (Confidence: {confidence:.2f}%)")
        print(f"  → These signatures likely belong to different people")
    
    return similarity_score

def main():
    if len(sys.argv) != 3:
        print("Usage: python src/predict_signature_pair.py <signature1.png> <signature2.png>")
        print("\nExample:")
        print('  python src/predict_signature_pair.py "path/to/sig1.png" "path/to/sig2.png"')
        sys.exit(1)
    
    sig1_path = sys.argv[1]
    sig2_path = sys.argv[2]
    
    try:
        predict_similarity(sig1_path, sig2_path)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

