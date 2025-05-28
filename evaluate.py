# evaluate.py

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from src.ml_logic.model import load_model
from src.ml_logic.data import load_test_data
from src.ml_logic.preprocessor import preprocess
from src.ml_logic.alphabet import char_to_num


def evaluate():
    print("🔍 Loading model...")
    model = load_model()
    if model is None:
        print("❌ Model not found.")
        return

    print("📦 Loading test data...")
    X_raw, y_raw = load_test_data(limit=10)  # small test set

    print("⚙️ Preprocessing...")
    X = []
    y = []

    for i in range(len(X_raw)):
        try:
            # Preprocess input data (normalize and resize)
            frames = preprocess(X_raw[i])  # shape: (T, 46, 140, 1)
            if frames.shape[0] < 1:
                continue
            X.append(frames)

            # Convert the first character of the label to a numeric index
            label_char = y_raw[i][0].lower()
            label_tensor = tf.constant(label_char)
            label_index = char_to_num(label_tensor).numpy().item()
            y.append(label_index)

        except Exception as e:
            print(f"⚠️ Skipped sample {i}: {e}")

    if len(X) == 0:
        print("❌ No valid test data.")
        return

    print("📐 Padding sequences to uniform length...")
    max_len = max([x.shape[0] for x in X])
    X_padded = np.array([
        np.pad(x, ((0, max_len - x.shape[0]), (0, 0), (0, 0), (0, 0)), mode='constant')
        for x in X
    ])
    y = np.array(y)

    print("🧪 Evaluating model...")
    y_pred = model.predict(X_padded)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("📊 Classification Report:")
    print(classification_report(y, y_pred_classes))


if __name__ == "__main__":
    evaluate()
