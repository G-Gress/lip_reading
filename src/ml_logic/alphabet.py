"""
This module provides mappings between characters and numerical indices.
Used for encoding text (char to num) and decoding model output (num to char).
"""

import tensorflow as tf

# Define the full vocabulary
vocab = list("abcdefghijklmnopqrstuvwxyz'?1234567890 ")

# Conversion: character → integer
char_to_num = tf.keras.layers.StringLookup(
    vocabulary=vocab,
    oov_token=""
)

# Conversion: integer → character
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    invert=True
)
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
