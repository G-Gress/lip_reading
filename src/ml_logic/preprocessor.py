import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

def preprocess_video(video_path: str, max_time: int = None) -> tf.Tensor:
    print("🔥 preprocess_video called!")
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (140, 46))
        resized = np.expand_dims(resized, axis=-1)
        frames.append(resized)

    cap.release()

    if not frames:
        print("❌ No frames extracted from video.")
        return None

    if max_time is not None:
        frames = frames[:max_time]
        while len(frames) < max_time:
            frames.append(np.zeros((46, 140, 1), dtype=np.float32))

    video_array = np.array(frames, dtype=np.float32) / 255.0
    video_tensor = tf.convert_to_tensor(video_array)
    print("✅ Shape of preprocessed input:", video_tensor.shape)
    return video_tensor


def normalize_frames(frames, max_time: int = None) -> tf.Tensor:
    gray_frames = []

    for frame in frames:
        if frame.ndim == 3 and frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(frame, (140, 46))
        resized = np.expand_dims(resized, axis=-1)
        gray_frames.append(resized)

    if max_time is not None:
        gray_frames = gray_frames[:max_time]
        while len(gray_frames) < max_time:
            gray_frames.append(np.zeros((46, 140, 1), dtype=np.float32))

    video_array = np.array(gray_frames, dtype=np.float32) / 255.0
    return tf.convert_to_tensor(video_array)


def preprocess(input, max_time: int = None) -> tf.Tensor:
    if isinstance(input, (str, Path)):
        return preprocess_video(input, max_time=max_time)
    elif isinstance(input, (list, np.ndarray)):
        return normalize_frames(input, max_time=max_time)
    else:
        raise ValueError(f"Unsupported input type: {type(input)}")
