from src.ml_logic.preprocessor import preprocess_video
import tensorflow as tf
from src.ml_logic.alphabet import num_to_char

def decode_prediction(prediction: tf.Tensor):
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

import tensorflow as tf
from src.ml_logic.alphabet import num_to_char

def run_prediction_no_label(model, video_tensor: tf.Tensor) -> str:
    """
    Predict the character sequence from a preprocessed video tensor.
    This version does not require labels or alignments.

    Args:
        model: Trained CTC model
        video_tensor: A Tensor of shape (T, H, W, C), e.g., (75, 46, 140, 1)

    Returns:
        Decoded prediction string
    """
    # Expand dims to add batch dimension: (1, T, H, W, C)
    input_tensor = tf.expand_dims(video_tensor, axis=0)

    # Model prediction
    y_pred = model.predict(input_tensor)

    # Take argmax to get predicted indices
    y_pred_argmax = tf.argmax(y_pred, axis=-1)[0]  # shape: (T,)

    # Remove blank tokens (index 0) and repeated characters
    decoded_indices = []
    prev_index = -1
    for index in y_pred_argmax.numpy():
        if index != 0 and index != prev_index:
            decoded_indices.append(index)
        prev_index = index

    # Convert indices to characters
    decoded_chars = [num_to_char(index).numpy().decode("utf-8") for index in decoded_indices]

    return "".join(decoded_chars)
