import cv2
import time
import os
from pathlib import Path
import tensorflow as tf
from src.ml_logic.model import load_model
from src.ml_logic.preprocessor import preprocess_video_auto_crop  # ← 自動クロップ版
from src.ml_logic.predictor import run_prediction_no_label

# === 設定 ===
SAVE_PATH = "test_videos/test1.mp4"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MOUTH_X, MOUTH_Y, MOUTH_W, MOUTH_H = 250, 270, 140, 46  # 枠の位置とサイズ（参考用）
RECORD_SECONDS = 5  # 録画時間（秒）

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("\n📸 Please align your mouth in the red box and press 'r' to record")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 赤枠（ガイド用）
        cv2.rectangle(frame, (MOUTH_X, MOUTH_Y), (MOUTH_X + MOUTH_W, MOUTH_Y + MOUTH_H), (0, 0, 255), 2)
        cv2.putText(frame, "Align mouth here & press 'r' to record", (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Mouth Alignment UI", frame)
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
                cv2.rectangle(frame, (MOUTH_X, MOUTH_Y), (MOUTH_X + MOUTH_W, MOUTH_Y + MOUTH_H), (0, 0, 255), 2)
                cv2.imshow("Mouth Alignment UI", frame)

            out.release()
            print(f"✅ Saved video to: {SAVE_PATH}")
            break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # === 推論 ===
    print("\n🤖 Running inference...")
    model = load_model()
    if model is None:
        print("❌ Failed to load model.")
        return

    video_tensor = preprocess_video_auto_crop(SAVE_PATH)
    prediction = run_prediction_no_label(model, video_tensor)
    print(f"✅ Prediction result: {prediction}")

if __name__ == "__main__":
    main()
