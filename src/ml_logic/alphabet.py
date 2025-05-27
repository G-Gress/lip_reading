"""
This module provides mappings between characters and numerical indices.
Used for encoding text (char to num) and decoding model output (num to char).
"""

import tensorflow as tf
from tensorflow.keras.layers import StringLookup

# Define the character set: a-z, apostrophe, and space
characters = [char for char in "abcdefghijklmnopqrstuvwxyz' "]

# Convert characters to numeric indices
char_to_num = StringLookup(vocabulary=characters, oov_token="", name="char_to_num")

# Convert numeric indices back to characters
num_to_char = StringLookup(vocabulary=characters, oov_token="", invert=True, name="num_to_char")
