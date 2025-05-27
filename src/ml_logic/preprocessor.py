import tensorflow as tf
import numpy as np
import cv2
import dlib

# Initialize face detector
face_detector = dlib.get_frontal_face_detector()

def crop_face_region(frame: np.ndarray) -> np.ndarray:
    """
    Detect and crop the face region from a single frame using dlib.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb_frame)

    if len(faces) == 0:
        return frame

    face = faces[0]
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    return frame[y1:y2, x1:x2]

def normalize_frames(frames: list) -> tf.Tensor:
    """
    Normalize a list of frames by subtracting mean and dividing by std.
    """
    frames_tensor = tf.convert_to_tensor(frames, dtype=tf.float32)
    mean = tf.math.reduce_mean(frames_tensor)
    std = tf.math.reduce_std(frames_tensor)
    return (frames_tensor - mean) / std

def preprocess(frames: list) -> tf.Tensor:
    """
    Preprocess a list of frames for training.
    """
    processed_frames = []

    for frame in frames:
        cropped = crop_face_region(frame)
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            resized = cv2.resize(cropped, (140, 46))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            expanded = np.expand_dims(gray, axis=-1)
            processed_frames.append(expanded)

    if len(processed_frames) == 0:
        raise ValueError("No valid frames to preprocess.")

    return normalize_frames(processed_frames)

def preprocess_video(video_path: str) -> tf.Tensor:
    """
    Preprocess a single video file for inference.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        return None

    processed_frames = []
    for frame in frames:
        cropped = crop_face_region(frame)
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            resized = cv2.resize(cropped, (140, 46))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            expanded = np.expand_dims(gray, axis=-1)
            processed_frames.append(expanded)

    if len(processed_frames) == 0:
        return None

    normalized = normalize_frames(processed_frames)
    return tf.expand_dims(normalized, axis=0)
