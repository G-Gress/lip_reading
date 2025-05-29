import numpy as np
import tensorflow as tf
import cv2

from tensorflow.keras.callbacks import ModelCheckpoint

# Load custom modules
from src.ml_logic import data, model, alphabet, preprocessor

def train():
    print("🔍 Loading training data...")
    X_raw, y_raw = data.load_data()

    # ✅ Use a small subset of data for testing
    N = 50
    X_raw = X_raw[:N]
    y_raw = y_raw[:N]
    print(f"🖼️ Loaded {len(X_raw)} video samples.")

    print("⚙️ Preprocessing video frames and labels...")
    X = []
    label_tensors = []
    input_lengths = []
    label_lengths = []

    for i in range(N):
        try:
            frames = X_raw[i]  # List of video frames (already extracted)

            # ✅ Normalize and resize frames
            video_tensor = preprocessor.normalize_frames(frames)  # shape: (time, 46, 140, 1)

            if video_tensor is None or video_tensor.shape[0] == 0:
                raise ValueError("Invalid video input")

            # Store inputs and lengths
            X.append(video_tensor)
            input_lengths.append(video_tensor.shape[0])  # number of time steps

            # Convert label to integer tensor
            label = y_raw[i]
            label_tensor = alphabet.encode(label)
            label_tensors.append(label_tensor)
            label_lengths.append(len(label_tensor))

        except Exception as e:
            print(f"⚠️ Skipping sample {i} due to error: {e}")

    if not X:
        print("❌ No valid data to train on.")
        return

    # ✅ Pad input sequences (videos) to uniform length
    max_time = 75
    X_padded = []
    for x in X:
        pad_len = max_time - x.shape[0]
        if pad_len < 0:
            x = x[:max_time]
            pad_len = 0
        padding = tf.zeros((pad_len, 46, 140, 1), dtype=tf.float32)
        padded = tf.concat([x, padding], axis=0)
        X_padded.append(padded)
    X = tf.stack(X_padded)  # shape: (N, 75, 46, 140, 1)

    # ✅ Pad label sequences (integers)
    label_tensors = tf.keras.preprocessing.sequence.pad_sequences(
        label_tensors, padding="post"
    )

    print("✅ Final input shape:", X.shape)

    # ✅ Load trained base model
    base_model = model.load_model("models/ctc_model.keras")
    if base_model is None:
        print("❌ Could not load base model.")
        return

    # ✅ Build the CTC training model
    video_input = tf.keras.Input(shape=X.shape[1:], name="video")
    labels = tf.keras.Input(shape=(None,), dtype="int32", name="labels")
    input_len = tf.keras.Input(shape=(1,), dtype="int32", name="input_length")
    label_len = tf.keras.Input(shape=(1,), dtype="int32", name="label_length")

    y_pred = base_model(video_input)

    # Add CTC loss layer
    ctc_loss = tf.keras.layers.Lambda(
        lambda args: tf.keras.backend.ctc_batch_cost(*args),
        name="ctc_loss"
    )([labels, y_pred, input_len, label_len])

    ctc_model = tf.keras.Model(
        inputs=[video_input, labels, input_len, label_len],
        outputs=ctc_loss
    )

    ctc_model.compile(optimizer="adam", loss=lambda y_true, y_pred: y_pred)

    checkpoint = ModelCheckpoint("model_checkpoint.keras", save_best_only=True)

    # ✅ Train the model
    ctc_model.fit(
        x={
            "video": X,
            "labels": label_tensors,
            "input_length": np.array(input_lengths)[:, None],
            "label_length": np.array(label_lengths)[:, None]
        },
        y=np.zeros(len(X)),  # dummy target (CTC loss is used directly)
        batch_size=2,
        epochs=10,
        validation_split=0.1,
        callbacks=[checkpoint]
    )

    print("✅ Training complete!")

if __name__ == "__main__":
    train()
