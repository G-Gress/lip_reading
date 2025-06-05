import numpy as np
import cv2

def extract_lip_region(frame, landmarks):
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
    mouth_resized = cv2.resize(mouth, (100, 50))  # Resize to a fixed size
    return mouth_resized
