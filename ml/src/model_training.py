# src/model_training.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

PROCESSED_PATH = "data/processed/"
MODEL_SAVE_PATH = "models/siamese_signature_model.h5"

def load_data():
    train_data = np.load(os.path.join(PROCESSED_PATH, "train_data_1.npz"))
    test_data = np.load(os.path.join(PROCESSED_PATH, "test_data_1.npz"))

    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    print("✅ Data loaded successfully!")
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    return X_train, y_train, X_test, y_test


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model():
    X_train, y_train, X_test, y_test = load_data()
    model = build_model()

    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stop]
    )

    print("✅ Model training complete!")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")

    return model, history


if __name__ == "__main__":
    train_model()
