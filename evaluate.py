# evaluate.py

from src.ml_logic.model import load_model
from src.ml_logic.data import load_test_data
from sklearn.metrics import classification_report
import numpy as np

def evaluate():
    print("🔍 Loading model...")
    model = load_model()
    if model is None:
        print("❌ Model not found.")
        return

    print("📦 Loading test data...")
    X_test, y_test = load_test_data()

    print("🧪 Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print("📊 Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes))

if __name__ == "__main__":
    evaluate()
