from src.ml_logic.preprocessor import preprocess_video_no_align
from src.ml_logic.preprocessor import preprocess_video
import tensorflow as tf
from src.ml_logic.alphabet import num_to_char
from pathlib import Path

def decode_prediction(prediction: tf.Tensor):
    input_length = tf.shape(prediction)[1]
    decoded_tensor, _ = tf.keras.backend.ctc_decode(prediction, [input_length], greedy=True)
    decoded_sequence = decoded_tensor[0][0].numpy()
    decoded_text = tf.strings.reduce_join([num_to_char(i) for i in decoded_sequence])
    return decoded_text

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


def run_prediction_no_align(model, input_video) -> str:
    """
    Run inference on a lip-reading video (no alignment required).
    Accepts either a preprocessed video tensor or a video file path (mp4).

    Args:
        model: A trained CTC-based lip-reading model.
        input_video: Either a tf.Tensor of shape (T, H, W, 1) or a str/Path to the mp4 file.

    Returns:
        A decoded text string predicted from the video.
    """

    # 📦 If input is a path, load and preprocess the video
    if isinstance(input_video, (str, Path)):
        video_tensor = preprocess_video_no_align(str(input_video))
    else:
        video_tensor = input_video  # already preprocessed

    # 📏 Add batch dimension: (1, T, H, W, 1)
    input_tensor = tf.expand_dims(video_tensor, axis=0)

    # 🤖 Model prediction
    prediction = model.predict(input_tensor)

    # 🧮 Compute input length for CTC decoding (batch size = 1)
    input_len = tf.shape(prediction)[1:2]

    # 🔤 CTC decoding
    decoded_tensor, _ = tf.keras.backend.ctc_decode(prediction, input_length=input_len, greedy=False,
    beam_width=10)

    # 🪄 Convert indices to characters and join
    decoded_sequence = decoded_tensor[0][0].numpy()
    decoded_chars = [num_to_char(i).numpy().decode("utf-8") for i in decoded_sequence if i != -1]
    decoded_text = "".join(decoded_chars)

    return decoded_text
