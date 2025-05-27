"""
Train the lip-reading model on a small sample.
Saves the trained model to models/lip_model.keras
"""

from src.ml_logic import data, preprocessor, model
from src.ml_logic.alphabet import char_to_num
import numpy as np
import tensorflow as tf

def train():
    print("🔍 Loading data...")
    X_raw, y_raw = data.load_data()

    # Use only first N samples for quick testing
    N = 10
    X_raw = X_raw[:N]
    y_raw = y_raw[:N]
    print(f"🖼️ Loaded {len(X_raw)} samples.")

    print("⚙️ Preprocessing...")
    X = []
    y = []

    for i in range(N):
        try:
            frames = preprocessor.preprocess(X_raw[i])  # shape: (T, 46, 140, 1)
            if frames.shape[0] < 1:
                raise ValueError("No valid frames to preprocess.")
            X.append(frames)

            label = y_raw[i]
            label_char = label[0].lower()  # Convert to lowercase
            y_idx = char_to_num(tf.constant(label_char)).numpy().item()
            y.append(y_idx)
        except Exception as e:
            print(f"⚠️ Skipped sample {i} due to error: {e}")

    if len(X) == 0:
        print("❌ No valid data to train.")
        return

    # Pad sequences to uniform length
    print("📐 Padding sequences to uniform length...")
    max_len = max([x.shape[0] for x in X])
    X_padded = np.array([
        np.pad(x, ((0, max_len - x.shape[0]), (0, 0), (0, 0), (0, 0)), mode='constant')
        for x in X
    ])
    X = X_padded  # shape: (N, max_time, 46, 140, 1)
    y = np.array(y)

    print(f"✅ Final training set: {X.shape}, Labels: {y.shape}")

    print("📦 Building model...")
    lip_model = model.build_model()

    print("🏋️ Training model...")
    lip_model.fit(X, y, epochs=3, batch_size=2)

    print("💾 Saving model...")
    model.save_model(lip_model)
    print("✅ Model saved!")

if __name__ == "__main__":
    train()
