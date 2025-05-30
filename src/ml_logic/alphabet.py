"""
This module provides mappings between characters and numerical indices.
Used for encoding text (char to num) and decoding model output (num to char).
"""

import tensorflow as tf

# Define the full vocabulary
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# Char to num converter
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Num to char converter
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)


def encode(text: str) -> tf.Tensor:
    """
    Encode a string into a tensor of integer indices.
    Example: "hello" → [8, 5, 12, 12, 15]
    """
    chars = tf.strings.unicode_split(text.lower(), input_encoding="UTF-8")
    return char_to_num(chars)

def decode(indices: tf.Tensor) -> str:
    """
    Decode a tensor of integer indices into a string.
    Example: [8, 5, 12, 12, 15] → "hello"
    """
    chars = num_to_char(indices)
    return tf.strings.reduce_join(chars).numpy().decode("utf-8")

def decode_streamlit(model_pred, sequence_length = [75]) -> str:
    """
    Decode a prediction from model.predict (tensor of integer) into a string.
    Used for streamlit
    """
    decoded = tf.keras.backend.ctc_decode(model_pred, sequence_length, greedy=False)[0][0].numpy()

    for x in range(len(model_pred)):
        prediction = tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8')

    return prediction
