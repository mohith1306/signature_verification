import os
import numpy as np
from siamese.siamese_dataset import create_pairs

# Project root is one level up from src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GENUINE_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "CEDAR", "genuine")
FORGED_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "CEDAR", "forged")
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test_data_1.npz")

def main():
    print(f"Genuine: {GENUINE_PATH}")
    print(f"Forged: {FORGED_PATH}")
    if not os.path.exists(GENUINE_PATH):
        raise FileNotFoundError(f"Genuine path not found: {GENUINE_PATH}")
    if not os.path.exists(FORGED_PATH):
        raise FileNotFoundError(f"Forged path not found: {FORGED_PATH}")
    
    pairs, labels = create_pairs(GENUINE_PATH, FORGED_PATH)
    if len(pairs) == 0:
        print("No pairs generated.")
        return
    X1 = np.array([p[0] for p in pairs], dtype="float32")
    X2 = np.array([p[1] for p in pairs], dtype="float32")
    y  = np.array(labels, dtype="int32")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(OUT_PATH, X1=X1, X2=X2, y=y)
    print(f"Saved test pairs to {OUT_PATH} with shapes: X1={X1.shape}, X2={X2.shape}, y={y.shape}")

if __name__ == "__main__":
    main()