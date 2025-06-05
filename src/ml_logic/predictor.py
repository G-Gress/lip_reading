from src.ml_logic.preprocessor import preprocess_video_dlib
from src.ml_logic.data_fine_tune import pad_or_trim_video
from src.ml_logic.preprocessor import preprocess_video
import tensorflow as tf
from src.ml_logic.alphabet import num_to_char
from pathlib import Path
import torch

def decode_prediction(prediction: tf.Tensor):
    '''
    Takes the prediction and decode it with TensorFlow's ctc_decode
    '''
    decoded_tensor, _ = tf.keras.backend.ctc_decode(prediction, [75], greedy=True)
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
    if isinstance(input_video, (str, Path)):
        video_tensor = preprocess_video_dlib(str(input_video))
        video_tensor = pad_or_trim_video(video_tensor, target_length=75)
        video_tensor = tf.reshape(video_tensor, (75, 46, 140, 1))
        print("📐 video_tensor final shape:", video_tensor.shape)
    else:
        video_tensor = input_video

    input_tensor = tf.expand_dims(video_tensor, axis=0)
    prediction = model.predict(input_tensor)
    print("🔢 Prediction shape:", prediction.shape)
    print("📊 First logit vector (at t=0):", prediction[0, 0])

    decoded_text = decode_prediction(prediction)
    return decoded_text
