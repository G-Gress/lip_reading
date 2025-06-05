import numpy as np
import tensorflow as tf
import pytest
from src.ml_logic import preprocessor

def test_crop_face_region_returns_image():
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = preprocessor.crop_face_region(dummy_frame)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 3  # HWC

def test_normalize_frames_output_shape():
    dummy_frames = [np.random.randint(0, 255, (46, 140, 1), dtype=np.uint8) for _ in range(5)]
    result = preprocessor.normalize_frames(dummy_frames)
    assert isinstance(result, tf.Tensor)
    assert result.shape == (5, 46, 140, 1)

def test_preprocess_returns_tensor():
    dummy_frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(10)]
    result = preprocessor.preprocess(dummy_frames)
    assert isinstance(result, tf.Tensor)
    assert result.shape[-1] == 1  # grayscale

# Optional: Test actual video preprocessing if the file exists
def test_preprocess_video_shape():
    video_path = "raw_data/videos/s1/bbaf2n.mpg"  
    result = preprocessor.preprocess_video(video_path)
    assert result is not None
    assert result.shape.ndims == 5  # (1, T, H, W, C)
