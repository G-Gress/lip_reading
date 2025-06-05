import os
from pathlib import Path
import cv2
from src.params import RAW_DATA_DIR
import tensorflow as tf
from src.ml_logic.alphabet import char_to_num
import numpy
from src.ml_logic.utils import extract_lip_region
from imutils import face_utils
import dlib
import numpy as np


def load_alignments(path: str) -> tf.Tensor:
    with open(path, "r") as f:
        lines = f.readlines()

    tokens = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3 and parts[2] != "sil":
            tokens.append(parts[2])  # ["bin", "blue", ...]

    joined = " ".join(tokens)  # "binblueattwonow"
    chars = tf.strings.unicode_split(joined, input_encoding='UTF-8')

    return char_to_num(chars)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

def load_video(path: str) -> tf.Tensor:
    '''
    Load a video from a path, extract the lips with dlib, or use the previous successful frame.
    If no previous successful frame is available and detection fails, the frame is skipped.
    Convert to grayscale, normalize with z-score normalization, and return a numpy array of frames.
    '''
    cap = cv2.VideoCapture(path)
    frames = []
    last_successful_frame = None  # Keep track of the last successfully extracted grayscale lip frame

    # Check if video capture is successful
    if not cap.isOpened():
        print(f"Error: Could not open video file {path}")
        # Return an empty tensor or handle the error appropriately
        return tf.zeros([0, 50, 100, 1], dtype=tf.float32)

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        # Read the next frame
        ret, frame = cap.read()
        if not ret: # Check if frame was read successfully
            continue # Skip if frame reading failed
        # Extract the face with dlib
        face = detector(frame)

        if len(face) > 0: # If a face has been detected
            shape = predictor(frame, face[0])
            shape = face_utils.shape_to_np(shape)
            lip = extract_lip_region(frame, shape)

            # Ensure the extracted lip region is not empty before processing
            if lip.size > 0:
                # Grayscale conversion
                gray = tf.image.rgb_to_grayscale(lip)
                # Save the frame
                frames.append(gray)
                last_successful_frame = gray # Update the last successful frame

                debug_frame = (gray.numpy().squeeze() * 255).astype(np.uint8)
                cv2.imshow("Mouth Region", debug_frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):  # 'q'で中断可
                    break
            else:
                # If extraction failed even with a detected face, try using the previous successful frame
                if last_successful_frame is not None:
                    frames.append(last_successful_frame)
                # Else: If no previous successful frame, the frame is skipped implicitly (not appended)

        else: # No face detected
            # If no previous successful frame is available, the frame is skipped implicitly (not appended)
            if last_successful_frame is not None:
                 # Use the last successful frame as a placeholder if available
                frames.append(last_successful_frame)
            # Else: If no previous successful frame, the frame is skipped implicitly (not appended)


    cap.release()

    cv2.destroyAllWindows()

    # Handle the case where no frames were processed
    if not frames:
        # Return an empty tensor
        empty_frame_shape = (50, 100, 1)
        return tf.zeros([0] + list(empty_frame_shape), dtype=tf.float32)

    # Normalize the data with z-score normalization
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))

    # Add a small epsilon to std to avoid division by zero
    std = tf.maximum(std, tf.keras.backend.epsilon())


    return tf.cast((frames - mean), tf.float32) / std

def load_video_dlib(path: str) -> tf.Tensor:
    '''
    Load a video from a path, convert it to grayscale, crop it to the face,
    normalize it with z-score normalization, and return a numpy array of the frames.
    '''
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
      # Get one frame as a numpy array
      ret, frame = cap.read()
      # Grayscale conversion
      #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # => Returns 2D image

      faces = detector(frame)
      if len(faces) > 0:
          shape = predictor(frame, faces[0])
          shape = face_utils.shape_to_np(shape)
          lip = extract_lip_region(frame, shape)

      gray = tf.image.rgb_to_grayscale(lip) # => Returns 3D tensor
      # Add the frame to the list
      frames.append(gray)
    # Release the video
    cap.release()

    # Normalize the data with z-score normalization
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))

    return tf.cast((frames - mean), tf.float32) / std
