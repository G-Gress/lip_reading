import tensorflow as tf
from .data import load_data
import cv2
import tensorflow as tf
import numpy as np
from src.ml_logic.data import load_data_mp4
from src.ml_logic.data import load_video

def map_function(path):
    '''
    Wrapper function for load_data()
    Converts a Tensor path into two tensors (video, alignment)
    '''
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result

# Video preprocessing
def preprocess_video(path: str):
    '''
    Convert a video from a path into a tensor ready for prediction.
    '''
    video_tensor = tf.convert_to_tensor(path)
    processed_video, _ = map_function(video_tensor)
    processed_video = tf.expand_dims(processed_video, axis=0)
    return processed_video


def preprocess_video_auto_crop(path: str, target_size=(46, 140)) -> tf.Tensor:
    """
    Automatically detect the mouth region, crop it, convert to grayscale,
    normalize the pixel values, and return it as a Tensor.
    """
    cap = cv2.VideoCapture(path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            mouth_roi = gray[y + h//2 : y + h, x : x + w]
        else:
            mouth_roi = gray[230:230+target_size[0], 250:250+target_size[1]]

        resized = cv2.resize(mouth_roi, (target_size[1], target_size[0]))  # (W, H)
        resized = np.expand_dims(resized, axis=-1)  # (H, W, 1)
        frames.append(resized)

    cap.release()

    video = np.array(frames, dtype=np.float32)
    mean = np.mean(video)
    std = np.std(video) + 1e-8
    video = (video - mean) / std

    return tf.convert_to_tensor(video)

def preprocess_video_no_align(path: str) -> tf.Tensor:
    """
    Preprocess video by fixed cropping (190:236, 80:220), grayscale conversion,
    normalization, and return as a Tensor of shape (T, 46, 140, 1)
    """
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Grayscale with OpenCV (NumPy)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # shape: (H, W)

        # 2. Crop to fixed mouth region
        cropped = gray[190:236, 80:220]  # shape: (46, 140)

        # 3. Add channel dimension
        cropped = np.expand_dims(cropped, axis=-1)  # shape: (46, 140, 1)

        frames.append(cropped)

    cap.release()

    # Stack all frames into a video tensor
    video = np.stack(frames).astype(np.float32)  # shape: (T, 46, 140, 1)

    # Z-score normalization
    mean = np.mean(video)
    std = np.std(video) + 1e-6
    video = (video - mean) / std

    return tf.convert_to_tensor(video)

import cv2
import numpy as np
import tensorflow as tf

def preprocess_video_dynamic_crop(path: str, target_size=(46, 140)) -> tf.Tensor:
    """
    Detect face, crop mouth region dynamically, convert to grayscale,
    normalize with z-score, and return a Tensor of shape (T, 46, 140, 1).
    This matches the training format.
    """
    cap = cv2.VideoCapture(path)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 2: Detect face
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = faces[0]  # use first detected face
            # Step 3: Define mouth region relative to face box
            mouth_y1 = y + int(0.65 * h)
            mouth_y2 = mouth_y1 + target_size[0]
            mouth_x1 = x + int((w - target_size[1]) / 2)
            mouth_x2 = mouth_x1 + target_size[1]
        else:
            # fallback if face not detected
            mouth_y1, mouth_y2 = 190, 236
            mouth_x1, mouth_x2 = 80, 220

        # Step 4: Crop and expand dims
        mouth_roi = gray[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
        mouth_roi = cv2.resize(mouth_roi, (target_size[1], target_size[0]))
        mouth_roi = np.expand_dims(mouth_roi, axis=-1)
        frames.append(mouth_roi)

    cap.release()

    # Step 5: Stack and normalize
    video = np.array(frames, dtype=np.float32)
    mean = np.mean(video)
    std = np.std(video) + 1e-6
    video = (video - mean) / std

    return tf.convert_to_tensor(video)
