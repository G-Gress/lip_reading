import numpy as np
import tensorflow as tf
from src.ml_logic.model import load_model
from src.ml_logic.data import load_test_data
from src.ml_logic.preprocessor import normalize_frames
from src.ml_logic.alphabet import num_to_char
from src.ml_logic.eval import wer  # ✅ WERのみインポート

def decode_prediction(y_pred: tf.Tensor) -> str:
    decoded, _ = tf.keras.backend.ctc_decode(
        y_pred,
        input_length=tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1]),
        greedy=True
    )
    prediction = decoded[0][0].numpy()
    valid_indices = prediction[prediction != -1]
    print("🧩 Cleaned prediction indices:", valid_indices)

    chars = num_to_char(valid_indices)
    print("🔡 Decoded characters:", chars.numpy())

    text = tf.strings.reduce_join(chars).numpy().decode("utf-8")
    return text

def evaluate():
    print("🔍 Loading model...")
    model = load_model()
    if model is None:
        print("❌ Model not found.")
        return
    print("✅ Model input shape:", model.input_shape)

    print("📦 Loading test data...")
    X_raw, y_raw = load_test_data(limit=10)

    print("⚙️ Preprocessing and evaluating...")
    y_true, y_pred = [], []

    for i in range(len(X_raw)):
        try:
            video_tensor = normalize_frames(X_raw[i], max_time=75)
            if video_tensor is None or video_tensor.shape[0] == 0:
                continue

            print("✅ Shape of preprocessed input:", video_tensor.shape)

            X_input = tf.expand_dims(video_tensor, axis=0)
            y_hat = model.predict(X_input)

            decoded = decode_prediction(y_hat)

            y_true.append(y_raw[i])
            y_pred.append(decoded)

            print(f"✅ Sample {i} - True: {y_raw[i]} | Pred: {decoded}")

        except Exception as e:
            print(f"⚠️ Skipped sample {i}: {e}")

    if not y_pred:
        print("❌ No valid predictions.")
        return

    # ✅ WERのみを出力
    print("\n📊 Word Error Rate (WER):")
    wer_total = np.mean([wer(t, p) for t, p in zip(y_true, y_pred)])
    print(f"🧮 Average WER: {wer_total:.2%}")

if __name__ == "__main__":
    evaluate()
