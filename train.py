import numpy as np
import tensorflow as tf
import cv2

from tensorflow.keras.callbacks import ModelCheckpoint

from src.ml_logic import data, model, alphabet, preprocessor  # ← preprocessor を忘れずに import！

def train():
    print("🔍 Loading data...")
    X_raw, y_raw = data.load_data()

    # ✅ 少量のデータでテスト
    N = 50
    X_raw = X_raw[:N]
    y_raw = y_raw[:N]
    print(f"🖼️ Loaded {len(X_raw)} samples.")

    print("⚙️ Preprocessing videos and labels...")
    X = []
    label_tensors = []
    input_lengths = []
    label_lengths = []

    for i in range(N):
        try:
            frames = X_raw[i]  # すでに frame 群（リスト of ndarray）

            # ✅ 修正：frames を preprocessor で正規化
            video_tensor = preprocessor.normalize_frames(frames)  # shape: (time, 46, 140, 1)

            if video_tensor is None or video_tensor.shape[0] == 0:
                raise ValueError("Invalid video tensor")

            X.append(video_tensor)
            input_lengths.append(video_tensor.shape[0])  # time steps

            label = y_raw[i]
            label_tensor = alphabet.encode(label)
            label_tensors.append(label_tensor)
            label_lengths.append(len(label_tensor))

        except Exception as e:
            print(f"⚠️ Skipped sample {i} due to error: {e}")

    if not X:
        print("❌ No valid training data.")
        return

    # ✅ Padding for X (4D) manually
    max_time = 75
    X_padded = []
    for x in X:
        pad_len = max_time - x.shape[0]
        if pad_len < 0:
            x = x[:max_time]
            pad_len = 0
        pad = tf.zeros((pad_len, 46, 140, 1), dtype=tf.float32)
        x_padded = tf.concat([x, pad], axis=0)
        X_padded.append(x_padded)
    X = tf.stack(X_padded)

    # ✅ Padding for label tensors
    label_tensors = tf.keras.preprocessing.sequence.pad_sequences(
        label_tensors, padding="post"
    )

    print("✅ Input shape for model:", X.shape)

    # ✅ Load pre-trained model
    base_model = model.load_model("models/ctc_model.keras")
    if base_model is None:
        print("❌ モデルが読み込めませんでした")
        return

    # ✅ Define CTC loss model
    video_input = tf.keras.Input(shape=X.shape[1:], name="video")
    labels = tf.keras.Input(shape=(None,), dtype="int32", name="labels")
    input_len = tf.keras.Input(shape=(1,), dtype="int32", name="input_length")
    label_len = tf.keras.Input(shape=(1,), dtype="int32", name="label_length")

    y_pred = base_model(video_input)

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

    ctc_model.fit(
        x={
            "video": X,
            "labels": label_tensors,
            "input_length": np.array(input_lengths)[:, None],
            "label_length": np.array(label_lengths)[:, None]
        },
        y=np.zeros(len(X)),  # dummy output
        batch_size=2,
        epochs=10,
        validation_split=0.1,
        callbacks=[checkpoint]
    )

    print("✅ Training completed!")

if __name__ == "__main__":
    train()
