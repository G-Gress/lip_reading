"""
This script performs inference on a single lip-reading video.

Steps:
1. Load and preprocess the input video.
2. Load the trained model.
3. Predict using the model.
4. Decode and display the predicted output.
"""

import sys
import numpy as np
import tensorflow as tf
from src.ml_logic.model import load_model
from src.ml_logic.preprocessor import preprocess_video
from src.ml_logic.alphabet import num_to_char


def predict_on_video(video_path: str):
    """
    Run inference on a single video file (.mpg).
    """
    print(f"🎥 Preprocessing video: {video_path}")
    video_tensor = preprocess_video(video_path)

    if video_tensor is None:
        print("❌ Failed to preprocess video.")
        return

    print("🤖 Loading trained model...")
    model = load_model()
    if model is None:
        print("❌ No trained model found. Please run train.py first.")
        return

    print("🔮 Predicting...")
    yhat = model.predict(video_tensor)

    print("📖 Decoding prediction (argmax)...")
    predicted_class = np.argmax(yhat[0])
    predicted_char = num_to_char(tf.constant([predicted_class])).numpy()[0].decode("utf-8")

    print("📝 Predicted character:")
    print(f"👉 {predicted_char}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py path_to_video.mpg")
    else:
        predict_on_video(sys.argv[1])
