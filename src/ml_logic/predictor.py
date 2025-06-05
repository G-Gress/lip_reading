from src.ml_logic.preprocessor import preprocess_video_dlib
from src.ml_logic.preprocessor import preprocess_video
import tensorflow as tf
from src.ml_logic.alphabet import num_to_char
from pathlib import Path
import torch

def decode_prediction(prediction: tf.Tensor):
    '''
    Takes the prediction and decode it with TensorFlow's ctc_decode.
    Handles empty or invalid predictions gracefully.
    '''
    input_length = tf.shape(prediction)[1]
    decoded_tensor, _ = tf.keras.backend.ctc_decode(prediction, [input_length], greedy=True)
    decoded_sequence = decoded_tensor[0][0].numpy()

    # 空だった場合の対策
    if decoded_sequence.size == 0:
        return tf.constant("")

    # 安全に文字に変換
    try:
        chars = [num_to_char(i).numpy().decode('utf-8') for i in decoded_sequence if i >= 0]
        decoded_text = "".join(chars)
    except Exception as e:
        print(f"[Decode Error] {e}")
        decoded_text = "[decode error]"

    return tf.constant(decoded_text)



def run_prediction_no_align(model, input_video):
    # パスかテンソルかを判別
    if isinstance(input_video, (str, Path)):
        video_tensor = preprocess_video_dlib(str(input_video))
    else:
        video_tensor = input_video

    input_tensor = tf.expand_dims(video_tensor, axis=0)  # shape: (1, T, 50, 100, 1)

    if video_tensor.shape[0] == 0:
        print("[Error] No valid frames extracted.")
        return "[invalid video]", ""

    prediction = model.predict(input_tensor)
    decoded_text = decode_prediction(prediction)
    return decoded_text.numpy().decode(), ""

def run_prediction(model, path: str):
    '''
    Takes a model and a path to a video and return the prediction as a string.
    '''
    # Preprocessing
    processed_video = preprocess_video(path)
    # Prediction
    prediction = model(processed_video)
    # Decoding
    decoded_text = decode_prediction(prediction)

    return decoded_text.numpy().decode()
