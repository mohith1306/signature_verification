# src/model_evaluation.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from siamese.siamese_model import AbsoluteDifference

PROCESSED_PATH = "data/processed/"
MODEL_PATH = "models/siamese_signature_model.h5"

def load_data():
    data = np.load(os.path.join(PROCESSED_PATH, "test_data_1.npz"))
    if not {"X1","X2","y"}.issubset(data.files):
        raise ValueError("test_data_1.npz must contain X1, X2, y for Siamese evaluation.")
    return data["X1"], data["X2"], data["y"]

def evaluate_model():
    X1, X2, y_test = load_data()
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    model = load_model(MODEL_PATH, custom_objects={"AbsoluteDifference": AbsoluteDifference})

    print("\nEvaluating model on pair test data...")
    loss, acc = model.evaluate([X1, X2], y_test, verbose=0)
    print(f"Accuracy: {acc*100:.2f}%  Loss: {loss:.4f}")
    
    y_proba = model.predict([X1, X2], verbose=0).ravel()
    y_pred = (y_proba > 0.5).astype("int32")
    
    # Print metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Forged","Genuine"]))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Forged","Genuine"], yticklabels=["Forged","Genuine"])
    plt.title("Confusion Matrix - Signature Verification")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    print("Saved confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    evaluate_model()