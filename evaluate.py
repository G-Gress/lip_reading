import sys
from pathlib import Path
import numpy as np

# 🔧 src ディレクトリをインポートパスに追加
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.append(str(SRC_DIR))

# ✅ 正しく各モジュールをインポート
from ml_logic.model import load_model
from ml_logic.data import load_test_data
from ml_logic.eval import wer
from inference.inference import run_prediction  # ← 重要：フォルダ名も inference

def evaluate(limit=10):
    print("🔍 Loading model...")
    model = load_model()

    print(f"📦 Loading test data... (limit={limit})")
    X_raw, y_raw = load_test_data(limit=limit)

    y_true, y_pred = [], []

    for i, (sample, true_text) in enumerate(zip(X_raw, y_raw)):
        try:
            video_path = sample[0]  # 例: "raw_data/videos/...mpg"
            pred_text = run_prediction(model, video_path)

            y_true.append(true_text)
            y_pred.append(pred_text)

            print(f"✅ Sample {i} - True: {true_text} | Pred: {pred_text}")
        except Exception as e:
            print(f"❌ Sample {i} - Error: {e}")
            y_true.append(true_text)
            y_pred.append("")

    # WER (Word Error Rate) の計算
    try:
        wer_scores = [wer(t, p) for t, p in zip(y_true, y_pred)]
        wer_avg = np.mean(wer_scores)
        print(f"\n📊 Average WER: {wer_avg:.2%}")
    except Exception as e:
        print(f"⚠️ Failed to calculate WER: {e}")

if __name__ == "__main__":
    evaluate()
