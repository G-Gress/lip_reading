# Load the trained model from file
from src.ml_logic.model import load_model

# Preprocess the video for model input
# (This function should convert a video file into the required input format, e.g., a NumPy array)
from src.ml_logic.preprocessor import preprocess_video

import numpy as np

def predict_on_video(video_path):
    """
    Run inference (prediction) on a single video file.
    This function loads the model, preprocesses the input video,
    and prints out the predicted class.
    """

    print(f"🎥 Preprocessing video: {video_path}")
    video_data = preprocess_video(video_path)  # Expected shape: (1, time, height, width, channels)

    if video_data is None:
        print("❌ Failed to process video.")
        return

    print("🤖 Loading model...")
    model = load_model()  # Loads the trained model from disk

    if model is None:
        print("❌ No model found.")
        return

    print("🔮 Making prediction...")
    prediction = model.predict(video_data)  # Predict probabilities for each class
    predicted_class = np.argmax(prediction)  # Choose the class with the highest probability

    print(f"🎯 Predicted class index: {predicted_class}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python inference.py path_to_video.mp4")
    else:
        predict_on_video(sys.argv[1])
