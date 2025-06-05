import os
import tensorflow as tf
from tensorflow import keras
from src.ml_logic.alphabet import num_to_char

def load_model(path="models",
               model_name="ctc_model.keras",
               weights_name="checkpoint_epoch26_loss0.80.weights.h5"):
    """
    Load a trained CTC model structure and weights.

    Args:
        path (str): Directory where model and weights are saved
        model_name (str): Name of the .keras model file
        weights_name (str): Name of the .h5 weights file

    Returns:
        keras.Model or None: Loaded model or None if not found
    """
    model_path = os.path.join(path, model_name)
    weights_path = os.path.join(path, weights_name)

    if not os.path.exists(model_path):
        print(f"❌ Model structure file not found at: {model_path}")
        return None

    if not os.path.exists(weights_path):
        print(f"❌ Weights file not found at: {weights_path}")
        return None

    print(f"✅ Loading model structure from: {model_path}")
    model = keras.models.load_model(model_path, compile=False)

    print(f"✅ Loading weights from: {weights_path}")
    model.load_weights(weights_path)

    print("✅ Model fully loaded and ready!")
    return model


def load_delib_model(
    path="models",
    model_name="lipr_v2.keras",
    weights_name="v2_loss48.66.weights.h5"
):
    """
    Load the delib model. If weights_name is provided, weights will be loaded separately.
    Otherwise, assume the .keras model includes weights.
    """
    model_path = os.path.join(path, model_name)
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        return None

    print(f"✅ Loading model from: {model_path}")
    model = keras.models.load_model(model_path, compile=False)

    # Optional: separate weights
    if weights_name:
        weights_path = os.path.join(path, weights_name)
        if os.path.exists(weights_path):
            print(f"✅ Loading weights from: {weights_path}")
            model.load_weights(weights_path)
        else:
            print(f"⚠️ Weights file not found at: {weights_path} (skipping)")

    print("✅ delib model fully loaded and ready!")
    return model
