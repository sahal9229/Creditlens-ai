"""
train_model.py
--------------
Standalone training script for Credit Card Default Prediction ANN.
Replicates the exact pipeline from the learning notebook with the typo fix.

Artifacts produced:
  - credit_ann_model.keras   (trained Keras model)
  - scaler.pkl               (fitted StandardScaler)
"""

import os
import numpy as np
import pandas as pd
import joblib

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


def train_and_save(output_dir: str = ".") -> dict:
    """
    Fetch the UCI dataset, train the ANN, and save model + scaler.
    Returns a dict with test_loss and test_accuracy.
    """

    # ── 1. Fetch dataset ──────────────────────────────────────────────
    print("[1/6] Fetching UCI dataset (id=350) ...")
    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features
    y = dataset.data.targets

    print(f"      Dataset shape: X={X.shape}, y={y.shape}")
    print(f"      Class distribution:\n{y.value_counts()}\n")

    # ── 2. Train / Test split ─────────────────────────────────────────
    print("[2/6] Splitting data (80/20, random_state=44, stratified) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=0.8,
        random_state=44,
        stratify=y,
    )
    print(f"      Train: {X_train.shape[0]} samples")
    print(f"      Test:  {X_test.shape[0]} samples\n")

    # ── 3. Feature scaling ────────────────────────────────────────────
    print("[3/6] Fitting StandardScaler on training data ...")
    scaler = StandardScaler()
    X_train_scale = scaler.fit_transform(X_train)
    X_test_scale = scaler.transform(X_test)

    # ── 4. Build model ────────────────────────────────────────────────
    print("[4/6] Building Keras Sequential model ...")
    model = Sequential([
        Input(shape=(23,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()
    print()

    # ── 5. Train ──────────────────────────────────────────────────────
    print("[5/6] Training (epochs=50, batch_size=32, validation_split=0.2) ...")
    history = model.fit(
        X_train_scale,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
    )

    # ── 6. Evaluate & Save ────────────────────────────────────────────
    print("\n[6/6] Evaluating on test set ...")
    test_loss, test_accuracy = model.evaluate(X_test_scale, y_test)
    print(f"      Test Loss:     {test_loss:.4f}")
    print(f"      Test Accuracy: {test_accuracy:.4f}")

    model_path = os.path.join(output_dir, "credit_ann_model.keras")
    scaler_path = os.path.join(output_dir, "scaler.pkl")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n[OK] Model saved  -> {model_path}")
    print(f"[OK] Scaler saved -> {scaler_path}")

    return {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results = train_and_save(output_dir=script_dir)
    print(f"\nDone. Test accuracy: {results['test_accuracy']:.2%}")
