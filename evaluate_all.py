import tensorflow as tf
from pathlib import Path
from src.ml_logic.model import load_model
from src.ml_logic.predictor import run_prediction
from src.ml_logic.data import load_alignments
from src.ml_logic.alphabet import num_to_char
from src.ml_logic.eval import wer

# --- Parameters ---
VIDEO_DIR = Path("raw_data/videos/s1")
ALIGN_DIR = Path("raw_data/alignments/s1")
MAX_SAMPLES = 10  # select number

# --- Load the trained model ---
print("✅ Loading the trained model...")
model = load_model()
print("✅ Model fully loaded and ready!\n")

# --- Load video files ---
video_paths = sorted(VIDEO_DIR.glob("*.mpg"))[:MAX_SAMPLES]
print(f"📦 Starting evaluation for {len(video_paths)} video(s)...\n")

# --- Init counters ---
total_wer = 0.0
total_samples = 0

# --- Evaluation loop ---
for video_path in video_paths:
    file_name = video_path.stem
    align_path = ALIGN_DIR / f"{file_name}.align"

    if not align_path.exists():
        print(f"⚠️ Skipping {file_name}: alignment file not found.")
        continue

    try:
        # Load ground truth
        y_true_tensor = load_alignments(str(align_path))
        chars = num_to_char(y_true_tensor)
        y_true = tf.strings.reduce_join(chars).numpy().decode()

        # Predict
        y_pred = run_prediction(model, str(video_path))

        # Compute WER
        error = wer(y_true, y_pred)
        total_wer += error
        total_samples += 1

        # Display
        print(f"📁 Video: {file_name}.mpg")
        print(f"✅ Ground Truth: {y_true}")
        print(f"🔡 Predicted :   {y_pred}")
        print(f"📊 WER: {error:.2%}\n")

    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}\n")

# --- Final summary ---
if total_samples > 0:
    average_wer = total_wer / total_samples
    print(f"📊 Average WER over {total_samples} sample(s): {average_wer:.2%}")
else:
    print("⚠️ No valid samples evaluated.")
