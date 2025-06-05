import tensorflow as tf
import pandas as pd
from pathlib import Path
from src.ml_logic.alphabet import char_to_num

def pad_or_trim_video(video_tensor, target_length=140):
    current_length = video_tensor.shape[0]
    if current_length < target_length:
        padding = tf.zeros((target_length - current_length, *video_tensor.shape[1:]), dtype=video_tensor.dtype)
        return tf.concat([video_tensor, padding], axis=0)
    elif current_length > target_length:
        return video_tensor[:target_length]
    else:
        return video_tensor

def load_fine_tune_dataset(csv_path: str, videos_dir: str):
    """
    Load fine-tuning dataset from CSV and video directory.

    Args:
        csv_path: Path to the labels.csv file.
        videos_dir: Path to the directory containing videos.

    Returns:
        A tuple (X, y):
            X: list of preprocessed and padded video tensors
            y: list of label tensors (as int sequences)
    """
    import src.ml_logic.preprocessor as preprocessor

    df = pd.read_csv(csv_path)

    X = []
    y = []

    for _, row in df.iterrows():
        video_path = Path(videos_dir) / row["filename"]
        label_text = row["label"]

        try:
            video_tensor = preprocessor.preprocess_video_dlib(str(video_path))
            video_tensor = pad_or_trim_video(video_tensor, target_length=140)

            label_tensor = tf.convert_to_tensor([char_to_num(c) for c in label_text], dtype=tf.int32)

            X.append(video_tensor)
            y.append(label_tensor)
        except Exception as e:
            print(f"⚠️ Skipped {video_path} due to error: {e}")

    return X, y
