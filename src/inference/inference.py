"""
This script performs inference on all videos in the inference_videos folder
using a trained CTC model.

Steps:
1. Load and preprocess each video.
2. Load the trained CTC model.
3. Predict using the model.
4. Decode the predicted output using CTC decoding.
"""

import os
import glob
import numpy as np
import tensorflow as tf
from src.ml_logic.model import load_model
from src.ml_logic.preprocessor import preprocess_video
from src.ml_logic.alphabet import num_to_char


def decode_prediction(y_pred: tf.Tensor) -> str:
    """
    Decode the output tensor from the model using CTC decoding.

    Args:
        y_pred (tf.Tensor): Model prediction of shape (1, time, vocab_size)

    Returns:
        str: Decoded text from the model output
    """
    decoded, _ = tf.keras.backend.ctc_decode(
        y_pred,
        input_length=tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1]),
        greedy=True
    )
    prediction = decoded[0][0].numpy()


    # ✅ Remove invalid indices (-1)
    prediction = prediction[prediction != -1]

    text = tf.strings.reduce_join(num_to_char(prediction)).numpy().decode("utf-8")
    return text


def predict_on_video(video_path: str, model: tf.keras.Model):
    """
    Run inference on a single video file and decode the output.
    """
    print(f"\n🎥 Preprocessing video: {video_path}")
    video_tensor = preprocess_video(video_path)

    if video_tensor is None:
        print("❌ Failed to preprocess video.")
        return

    video_tensor = tf.expand_dims(video_tensor, axis=0)

    print("🔮 Predicting...")
    y_pred = model.predict(video_tensor)

    print("📖 Decoding prediction using CTC...")
    decoded_text = decode_prediction(y_pred)

    print("📝 Predicted transcription:")
    print(f"👉 {decoded_text}")


def predict_all_videos():
    """
    Perform inference on all videos in the inference_videos folder.
    """
    print("🤖 Loading trained model...")
    model = load_model()
    if model is None:
        print("❌ No trained model found.")
        return

    video_paths = glob.glob("inference_videos/*.mpg") + glob.glob("inference_videos/*.mp4")

    if not video_paths:
        print("⚠️ No video files found in 'inference_videos/'")
        return

    for video_path in video_paths:
        predict_on_video(video_path, model)

if __name__ == "__main__":
    predict_all_videos()
