import tensorflow as tf
from .data import load_data
import cv2
import tensorflow as tf
import numpy as np

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

def preprocess_video_for_inference(path: str) -> tf.Tensor:
    '''
    1. Load video
    2. Crop mouth region
    3. Convert to grayscale
    4. Normalize (z-score)
    5. Return as Tensor (shape: [frames, height, width, 1])
    '''

    cap = cv2.VideoCapture(path)
    frames = []

    # 口元座標（例）
    MOUTH_X, MOUTH_Y, MOUTH_W, MOUTH_H = 250, 230 , 140, 46

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 口元をクロップ
        mouth_frame = frame[MOUTH_Y:MOUTH_Y+MOUTH_H, MOUTH_X:MOUTH_X+MOUTH_W]

        # 2. グレースケール変換
        gray = cv2.cvtColor(mouth_frame, cv2.COLOR_BGR2GRAY)  # shape: (H, W)

        # 3. チャンネル次元を追加（H, W, 1）に
        gray = np.expand_dims(gray, axis=-1)

        if len(frames) < 5:
            cv2.imshow(f"Frame {len(frames)}", gray)
            cv2.waitKey(300)

        frames.append(gray)

    cap.release()

    video = np.array(frames, dtype=np.float32)  # shape: (T, H, W, 1)

    # 4. Zスコア正規化
    mean = np.mean(video)
    std = np.std(video) + 1e-8  # ゼロ除算防止
    video = (video - mean) / std

    # Tensor化
    return tf.convert_to_tensor(video)

import cv2
import numpy as np
import tensorflow as tf

def preprocess_video_auto_crop(path: str, target_size=(46, 140)) -> tf.Tensor:
    """
    口元を自動で検出し、クロップ → グレースケール → 正規化して Tensor を返す
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
