import numpy as np
import tensorflow as tf

from src.ml_logic.model import load_model
from src.ml_logic.data import load_test_data
from src.ml_logic.preprocessor import normalize_frames
from src.ml_logic.alphabet import num_to_char
from src.ml_logic.eval import wer  # ✅ WER = Word Error Rate

def decode_prediction(y_pred: tf.Tensor) -> str:
    """
    Decode the CTC model output into readable text.

    Args:
        y_pred (tf.Tensor): Raw model output with shape (1, time, vocab_size)

    Returns:
        str: Decoded transcription string
    """
    # Use TensorFlow's built-in CTC decoder
    decoded, _ = tf.keras.backend.ctc_decode(
        y_pred,
        input_length=tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1]),
        greedy=True
    )

    prediction = decoded[0][0].numpy()
    valid_indices = prediction[prediction != -1]  # Filter out invalid tokens
    print("🧩 Cleaned prediction indices:", valid_indices)

    # Convert integer indices to characters
    chars = num_to_char(valid_indices)
    print("🔡 Decoded characters:", chars.numpy())

    # Join the characters into a final string
    text = tf.strings.reduce_join(chars).numpy().decode("utf-8")
    return text

def evaluate():
    """
    Evaluate the trained CTC model on a test dataset.
    """
    print("🔍 Loading trained model...")
    model = load_model()
    if model is None:
        print("❌ Model not found.")
        return
    print("✅ Model input shape:", model.input_shape)

    print("📦 Loading test dataset...")
    X_raw, y_raw = load_test_data(limit=10)

    print("⚙️ Running predictions and decoding...")
    y_true, y_pred = [], []

    for i in range(len(X_raw)):
        try:
            # Normalize video frames and pad to fixed length
            video_tensor = normalize_frames(X_raw[i], max_time=75)
            if video_tensor is None or video_tensor.shape[0] == 0:
                continue

            print("✅ Shape of preprocessed input:", video_tensor.shape)

            # Expand to match batch dimension
            X_input = tf.expand_dims(video_tensor, axis=0)

            # Run prediction
            y_hat = model.predict(X_input)

            # Decode output to string
            decoded = decode_prediction(y_hat)

            y_true.append(y_raw[i])
            y_pred.append(decoded)

            print(f"✅ Sample {i} - True: {y_raw[i]} | Predicted: {decoded}")

        except Exception as e:
            print(f"⚠️ Skipped sample {i} due to error: {e}")

    if not y_pred:
        print("❌ No valid predictions.")
        return

    # Calculate Word Error Rate (WER) across all samples
    print("\n📊 Calculating Word Error Rate (WER)...")
    wer_total = np.mean([wer(t, p) for t, p in zip(y_true, y_pred)])
    print(f"🧮 Average WER: {wer_total:.2%}")

if __name__ == "__main__":
    evaluate()
