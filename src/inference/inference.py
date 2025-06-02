import os
import glob
import numpy as np
import tensorflow as tf
from src.ml_logic.model import load_model
from src.ml_logic.preprocessor import preprocess_video
from src.ml_logic.alphabet import num_to_char
from ml_logic.preprocessor import preprocess_video, normalize_frames


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



def decode_prediction(prediction: tf.Tensor):
    decoded_tensor, _ = tf.keras.backend.ctc_decode(prediction, [75], greedy=True)
    decoded_sequence = decoded_tensor[0][0].numpy()
    decoded_text = tf.strings.reduce_join([num_to_char(i) for i in decoded_sequence])
    return decoded_text

from pathlib import Path
def run_prediction(model, input):
    '''
    Takes a model and either:
    - a path to a video (str or Path)
    - a list or array of frames (np.ndarray or list of np.ndarray)
    Returns the prediction as a string.
    '''
    # Preprocessing
    if isinstance(input, (str, Path)):
        processed_video = preprocess_video(input)
    elif isinstance(input, (np.ndarray, list)):
        processed_video = normalize_frames(input)
    else:
        raise ValueError("Unsupported input type")

    # Prediction
    prediction = model(processed_video)
    decoded_text = decode_prediction(prediction)

    return decoded_text.numpy().decode()
