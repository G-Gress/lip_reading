import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List
from imutils import face_utils
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def extract_lip_region(frame, landmarks):
    '''
    Get a frame and exctract the lips area.
    '''
    # Use a margin of 10
    margin = 10
    # Use the landmarks for the outer lip
    left = landmarks[48][0] - margin
    right = landmarks[54][0] + margin
    top = landmarks[50][1] - margin
    bottom = landmarks[58][1] + margin
    # Ensure the coordinates are within image bounds
    left = max(left, 0)
    right = min(right, frame.shape[1])
    top = max(top, 0)
    bottom = min(bottom, frame.shape[0])
    mouth = frame[top:bottom, left:right]
    if mouth.size == 0:
        # Return a small placeholder or handle as an error
        # Returning a small dummy array might prevent errors later, but might not be ideal for training
        print("Warning: Extracted lip region is empty.")
        return np.zeros((50, 100, frame.shape[-1]), dtype=frame.dtype) # Assuming original frame might be color

    mouth_resized = cv2.resize(mouth, (100, 50))  # Resize to a fixed size
    return mouth_resized

def preprocess_video_streamlit(video_path: str):
    print("🔥 preprocess_video called!")
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    frames = []

    last_successful_frame = None  # Keep track of the last successfully extracted grayscale lip frame

    # Check if video capture is successful
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
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






    # # GAB's CODE
    # for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    #   # Get one frame as a numpy array
    #   ret, frame = cap.read()
    #   if not ret:
    #       break
    #   # Grayscale conversion
    #   gray = tf.image.rgb_to_grayscale(frame) # => Returns 3D tensor
    #   # Add the frame to the list
    #   frames.append(gray[190:236, 80:220, :])
    # # Release the video
    # cap.release()

    # # Normalize the data with z-score normalization
    # mean = tf.math.reduce_mean(frames)
    # std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    # frames = tf.cast((frames - mean), tf.float32) / std

    # return frames
