"""
This file defines and centralizes all shared parameters and file paths.
"""

from pathlib import Path

# Root of the project
BASE_DIR = Path(__file__).resolve().parents[1]

# Paths to key directories
RAW_DATA_DIR = BASE_DIR / "raw_data"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Path to the dlib facial landmark model
DLIB_MODEL_PATH = BASE_DIR / "shape_predictor_68_face_landmarks.dat"

# Model training hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
IMG_HEIGHT = 64
IMG_WIDTH = 64
