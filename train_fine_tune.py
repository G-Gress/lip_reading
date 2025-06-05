import tensorflow as tf
from src.ml_logic.model import load_delib_model
from tensorflow.keras.callbacks import Callback
from src.ml_logic.predictor import decode_prediction
from src.ml_logic.alphabet import num_to_char
from src.ml_logic.data_fine_tune import load_fine_tune_dataset
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
import os
import datetime

from tensorflow.keras.callbacks import Callback

class DisplayPredictionCallback(Callback):
    def __init__(self, dataset, num_samples=1):
        super().__init__()
        self.dataset = dataset
        self.num_samples = num_samples

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n🧪 Epoch {epoch+1} - Sample predictions:")
        for (inputs, y_true) in self.dataset.take(self.num_samples):
            (video_batch, label_len), label_tensor = inputs, y_true

            # 1バッチ（最初の1つ）だけ取り出す
            video = video_batch[0:1]        # shape: (1, time, 50, 100, 1)
            label = label_tensor[0]         # shape: (label_len,)
            pred = self.model.predict(video)

            # ✅ 予測のデコード
            decoded_pred = decode_prediction(pred)

            # ✅ 正解ラベルのデコード
            label_str = "".join([num_to_char(i).numpy().decode("utf-8") for i in label])

            print(f"🟡 Ground truth: {label_str}")
            print(f"🔵 Prediction   : {decoded_pred}")

# CTC損失関数（音声認識向け）
# 修正後の CTCLoss
def CTCLoss(y_true, y_pred):
    batch_size = tf.shape(y_pred)[0]
    time_steps = tf.shape(y_pred)[1]

    input_length = tf.ones(shape=(batch_size, 1), dtype=tf.int32) * time_steps
    label_length = tf.ones(shape=(tf.shape(y_true)[0], 1), dtype=tf.int32) * tf.shape(y_true)[1]

    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

# 学習率スケジューラー（10エポック以降に徐々に減衰）
def scheduler(epoch, lr):
    return float(lr if epoch < 10 else lr * tf.math.exp(-0.1))

# データセット準備（ジェネレータ → padded_batch）
# 修正後の prepare_dataset
def prepare_dataset(X, y, batch_size=2):
    def gen():
        for xi, yi in zip(X, y):
            yield xi, yi  # ⛳️ ここを戻す

    output_signature = (
        tf.TensorSpec(shape=(None, 50, 100, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )

    dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    return dataset.padded_batch(batch_size)

def main():
    print("🔄 Loading fine-tune dataset...")
    X, y = load_fine_tune_dataset(
        csv_path="data/fine_tune/labels.csv",
        videos_dir="data/fine_tune/videos"
    )

    if not X:
        print("❌ No training data found.")
        return

    # Train/Test split（80/20）
    split = int(len(X) * 0.8)
    train_X, test_X = X[:split], X[split:]
    train_y, test_y = y[:split], y[split:]

    train_ds = prepare_dataset(train_X, train_y)
    test_ds = prepare_dataset(test_X, test_y)

    # モデル読み込み
    print("🤖 Loading model...")
    model = load_delib_model()
    if model is None:
        print("❌ Failed to load model.")
        return

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=CTCLoss)

    # ✅ TensorBoard用ログディレクトリ作成
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # ✅ Checkpoint & Scheduler
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join("models", "checkpoint_epoch{epoch:02d}_loss{loss:.2f}.weights.h5"),
        monitor="loss",
        save_weights_only=True
    )
    schedule_callback = LearningRateScheduler(scheduler)

    # トレーニング開始
    print("🚀 Start training...")
    display_cb = DisplayPredictionCallback(test_ds)

    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=100,
        callbacks=[checkpoint_callback, schedule_callback, tensorboard_callback, display_cb],
        verbose=1
    )

    # 最終モデル保存
    print("💾 Saving final model...")
    model.save("models/finetuned_model.keras")

    # TensorBoard URL表示
    print(f"📊 View training at: http://localhost:6006/")

if __name__ == "__main__":
    main()
