import os
import glob
import numpy as np
import tensorflow as tf
from src.ml_logic.model import load_model
from src.ml_logic.preprocessor import preprocess_video

from src.ml_logic.preprocess_for_streamlit import preprocess_video_streamlit
from src.ml_logic.alphabet import num_to_char, decode_streamlit


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
    prediction = prediction[prediction != -1]
    text = tf.strings.reduce_join(num_to_char(prediction)).numpy().decode("utf-8")
    return text


def run_inference(video_path: str) -> str:
    """
    Perform inference on a single video and return predicted text.

    Args:
        video_path (str): Path to the video file.

    Returns:
        str: Decoded prediction text or error message.
    """
    model = load_model()
    if model is None:
        return "❌ No trained model found."

    if not os.path.exists(video_path):
        return f"❌ File not found: {video_path}"

    video_tensor = preprocess_video(video_path)
    if video_tensor is None:
        return "❌ Failed to preprocess video."

    video_tensor = tf.expand_dims(video_tensor, axis=0)
    y_pred = model.predict(video_tensor)

    decoded_text = decode_prediction(y_pred)
    return decoded_text




def run_inference_streamlit(video_path: str, model):
    """
    FOR STREAMLIT USE:
    Perform inference on a single video and return decoded predicted text
    and the preprocessed video file.

    Args:
        video_path (str): Path to the video file.
        model: instantiated model

    Returns:
        str: Decoded prediction text or error message.
        frames = preprocessed video file
    """
    frames = preprocess_video_streamlit(video_path)
    if frames is None:
        return "❌ Failed to preprocess video."

    video_tensor = tf.expand_dims(frames, axis=0)
    y_pred = model.predict(video_tensor)

    decoded_text = decode_streamlit(y_pred)
    return decoded_text, frames

