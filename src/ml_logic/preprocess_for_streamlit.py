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

    # GAB's CODE
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
      # Get one frame as a numpy array
      ret, frame = cap.read()
      if not ret:
          break
      # Grayscale conversion
      gray = tf.image.rgb_to_grayscale(frame) # => Returns 3D tensor
      # Add the frame to the list
      frames.append(gray[190:236, 80:220, :])
    # Release the video
    cap.release()

    # Normalize the data with z-score normalization
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    frames = tf.cast((frames - mean), tf.float32) / std

    return frames
