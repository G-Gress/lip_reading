"""
Train the lip-reading model on a sample dataset.
Saves the trained model to models/lip_model.keras
"""

from src.ml_logic import data, preprocessor, model
from src.ml_logic.alphabet import char_to_num
import numpy as np
import tensorflow as tf

def train():
    print("🔍 Loading data...")
    X_raw, y_raw = data.load_data()  # ✅ KEEP: Works if data format is standardized

    # ❗ TO MODIFY: Limit to 10 samples for demo purposes only
    N = 10
    X_raw = X_raw[:N]
    y_raw = y_raw[:N]
    print(f"🖼️ Loaded {len(X_raw)} samples.")

    print("⚙️ Preprocessing...")
    X = []
    y = []

    for i in range(N):  # ❗ TO MODIFY: Replace with full loop or train/test split logic
        try:
            frames = preprocessor.preprocess(X_raw[i])  # ✅ KEEP: Resizing and normalizing
            if frames.shape[0] < 1:
                raise ValueError("No valid frames to preprocess.")
            X.append(frames)

            # ⚠️ REVIEW: Only the first character is used → adapt if classifying full words
            label_char = y_raw[i][0].lower()
            y_idx = char_to_num(tf.constant(label_char)).numpy().item()
            y.append(y_idx)

        except Exception as e:
            print(f"⚠️ Skipped sample {i} due to error: {e}")

    if len(X) == 0:
        print("❌ No valid data to train.")
        return

    print("📐 Padding sequences to uniform length...")
    max_len = max([x.shape[0] for x in X])
    X_padded = np.array([
        np.pad(x, ((0, max_len - x.shape[0]), (0, 0), (0, 0), (0, 0)), mode='constant')
        for x in X
    ])
    X = X_padded
    y = np.array(y)

    print(f"✅ Final training set: {X.shape}, Labels: {y.shape}")

    print("📦 Building model...")
    lip_model = model.build_model()  # ✅ KEEP: Model is built using external clean logic

    print("🏋️ Training model...")
    lip_model.fit(X, y, epochs=3, batch_size=2)  # ❗ TO MODIFY: Tune hyperparameters later

    print("💾 Saving model...")
    model.save_model(lip_model)  # ✅ KEEP: Reusable save function
    print("✅ Model saved!")

if __name__ == "__main__":
    train()
