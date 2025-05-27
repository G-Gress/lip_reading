"""
This module provides preprocessing utilities for lip-reading.
It includes frame normalization and cropping facial regions.
"""

import numpy as np
import tensorflow as tf

def normalize_frames(frames: list) -> tf.Tensor:
    """
    Normalize a list of frames by subtracting mean and dividing by std.

    Args:
        frames(list):List of image frames (as numpy arrays)

    Returns:
        tf.Tensor: Normalized tensor od shape(n_frames,H,W,C)
    """

    #Convert to Tensor
    frames_tensor = tf.convert_to_tensor(frames,dtype=tf.float32)

    #Compute mean and std
    mean = tf.math.reduce_mean(frames_tensor)
    std = tf.math.reduce_std(frames_tensor)

    # Normalize
    normalized = (frames_tensor - mean) / std

    return normalized

import dlib
import cv2

# Initialize dlib's face detector (HOG-based)
face_detector = dlib.get_frontal_detector()

def crop_face_region(frame:np.ndarray) -> np.ndarray:
    """
    Detect and crop the face region from a single frame using dlib.

    Args:
        frame (np.ndarray): An image frame in BGR format (as read by cv2)

    Returns:
        np.ndarray: Cropped face image. If no face is detected, return the original frame.
    """
    # Convert from BGR to RGB (because dlib expects RGB input)
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # Detect faces (there might be multiple, but we use the first one)
    faces = face_detector(rgb_frame)

    if len(faces) == 0:
        return frame

    # Get the first detected face
    face = faces[0]
    x1,y1,x2,y2 = face.left(),face.top,face.right(),face.bottom()

    # Crop the face region (make sure it doesn't exceed the image boundaries)
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    return frame[y1:y2, x1:x2]
