import os
from tensorflow import keras

def load_model(model_path="models/ctc_model.keras"):
    """
    Load a trained CTC model from a specified path.

    Args:
        model_path (str): Path to the saved .keras model

    Returns:
        keras.Model or None: Loaded model or None if not found
    """
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return None

    print(f"✅ Loading model from: {model_path}")
    return keras.models.load_model(model_path, compile=False)  # compile=False に注意！
