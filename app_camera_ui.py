import cv2
from pathlib import Path
import time
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from src.ml_logic.model import load_delib_model
from src.ml_logic.preprocessor import preprocess_video_dlib
from src.ml_logic.predictor import run_prediction_no_align
import numpy as np

# === Settings ===
SAVE_PATH = "test_videos/test1.mp4"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
RECORD_SECONDS = 4  # Duration of recording (seconds)
MARGIN = 20

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("\n📸 Please align your mouth in the red box and press 'r' to record")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw detected face(s)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.putText(frame, "Align face & press 'r' to record", (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Face Alignment UI", frame)

        key = cv2.waitKey(1)

        if key == ord('r'):
            print("\n🎬 Recording starts in 3...")
            time.sleep(1)
            print("2...")
            time.sleep(1)
            print("1...")
            time.sleep(1)
            print("🎥 Recording...")

            os.makedirs(Path(SAVE_PATH).parent, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(SAVE_PATH, fourcc, 25.0, (FRAME_WIDTH, FRAME_HEIGHT))

            start_time = time.time()
            while time.time() - start_time < RECORD_SECONDS:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)


                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                cv2.imshow("Face Alignment UI", frame)

            out.release()
            print(f"✅ Saved video to: {SAVE_PATH}")
            break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # === Predict ===
    print("\n🤖 Running inference...")
    model = load_delib_model()
    if model is None:
        print("❌ Failed to load model.")
        return

    video_tensor = preprocess_video_dlib(str(SAVE_PATH))
    for i in range(video_tensor.shape[0]):
        frame = (video_tensor[i].numpy().squeeze() * 255).astype(np.uint8)
        cv2.imshow("Preview: Cropped Mouth Video", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    print(f"📐 Shape of processed video: {video_tensor.shape}")

    # ✅ Show first cropped frame
    sample = video_tensor[0].numpy().squeeze()
    plt.imshow(sample, cmap='gray')
    plt.title("First Cropped Mouth Frame")
    plt.axis("off")
    plt.show()

    prediction = run_prediction_no_align(model, video_tensor)

    print(f"✅ Prediction result: {prediction}")

if __name__ == "__main__":
    main()
