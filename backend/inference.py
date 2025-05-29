#!/usr/bin/env python3
import sys, os

# 1) Find the lip_reading/ folder (parent of backend/)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 2) Insert it onto Python’s search path
sys.path.insert(0, ROOT)

import argparse
import json
import tensorflow as tf
import numpy as np

# your imports from src/ml_logic
from src.ml_logic.model import load_model
from src.ml_logic.preprocess_for_streamlit import preprocess_video
from src.ml_logic.alphabet import decode  # or however you get num_to_char

def build_decoders():
    vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
    char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
    num_to_char = tf.keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(),
        oov_token="",
        invert=True
    )
    return num_to_char

def main():
    p = argparse.ArgumentParser(description="LipNet inference")
    p.add_argument("--video",   required=True, help="Path to the MP4 file")
    p.add_argument("--weights", required=True, help="Path to .h5 weights")
    args = p.parse_args()

    # 1) load & prepare model
    model = load_model()
    model.load_weights(args.weights)

    # 2) preprocess video into the (75,46,140,1) numpy array
    frames = preprocess_video(args.video)
    frames = np.expand_dims(frames, axis=0)  # shape (1,75,46,140,1)

    # 3) predict
    yhat = model.predict(frames)  # (1,75,41)

    # 4) ctc decode
    seq_len = [frames.shape[1]]
    decoded, _ = tf.keras.backend.ctc_decode(yhat, seq_len, greedy=False)
    decoded = decoded[0].numpy()  # shape (1,75)

    # 5) turn tokens into string
    num_to_char = build_decoders()
    transcription = ""
    for token in decoded[0]:
        ch = num_to_char(tf.constant([token])).numpy()[0].decode("utf-8")
        transcription += ch

    # 6) output JSON
    print(json.dumps({"transcription": transcription}))

if __name__ == "__main__":
    main()
