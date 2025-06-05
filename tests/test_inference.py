import pytest
import numpy as np
import tensorflow as tf
from src.ml_logic import model, preprocessor

def test_preprocess_video_shape():
    sample_path = "raw_data/videos/s1/bbaf2n.mpg"
    video_tensor = preprocessor.preprocess_video(sample_path)
    assert isinstance(video_tensor, tf.Tensor)
    assert video_tensor.shape[0] == 1  # batch dim
    assert video_tensor.shape[-1] == 1  # channel

def test_model_loading():
    m = model.load_model()
    assert m is not None
    assert hasattr(m, "predict")

def test_model_predict_output_shape():
    m = model.load_model()
    dummy_input = tf.random.normal((1, 24, 46, 140, 1))
    output = m.predict(dummy_input)
    assert len(output.shape) == 2  # (batch, num_classes)

def test_ctc_decode():
    yhat = tf.random.uniform((1, 24, 28))  # dummy output
    decoded, _ = tf.keras.backend.ctc_decode(yhat, input_length=tf.constant([24]))
    assert isinstance(decoded[0], tf.Tensor)
