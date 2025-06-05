import tensorflow as tf
from src.ml_logic.model import load_model
from src.ml_logic.predictor import run_prediction
from src.ml_logic.data import load_alignments
from src.ml_logic.alphabet import num_to_char
from src.ml_logic.eval import wer
from pathlib import Path

# --- Load the trained model ---
model = load_model()

# --- Define the sample to evaluate ---
file_name = "bbaf2n"
video_path = f"raw_data/videos/s1/{file_name}.mpg"
alignment_path = f"raw_data/alignments/s1/{file_name}.align"

# --- Load and convert the ground truth alignment ---
y_true_tensor = load_alignments(alignment_path)
#print("🔢 Raw tensor:", y_true_tensor.numpy())

chars = num_to_char(y_true_tensor)
#print("🔡 Decoded chars:", [c.numpy().decode() for c in chars])
y_true_text = tf.strings.reduce_join(chars).numpy().decode()

# --- Run prediction on the video ---
predicted_str = run_prediction(model, video_path)

# --- Compute WER ---
error_rate = wer(y_true_text, predicted_str)

# --- Show summary only ---
print(f"📁 Video: {file_name}.mpg")
print(f"✅ Ground Truth: {y_true_text}")
print(f"🔡 Predicted :   {predicted_str}")
print(f"📊 WER: {error_rate:.2%}")
