from pathlib import Path
import cv2
from src.params import RAW_DATA_DIR

def load_alignment_paths():
    """
    Returns a sorted list of all .txt alignment file paths
    under raw_data/alignments
    アライメントファイル（.txt）のパスをソートしてリストで返す
    """
    alignments_dir = RAW_DATA_DIR / "alignments"
    txt_files = sorted(alignments_dir.glob("**/*.txt"))
    return txt_files

def load_video_paths():
    """
    Returns a sorted list of all .mpg video file paths
    under raw_data/videos
    動画ファイル（.mpg）のパスをソートしてリストで返す
    """
    videos_dir = RAW_DATA_DIR / "videos"
    mpg_files = sorted(videos_dir.glob("**/*.mpg"))
    return mpg_files

def read_alignment_file(path):
    """
    Reads an alignment (.txt) file and returns a list of (word, start_frame, end_frame) tuples.

    Example:
    [("PLACE", 0, 15), ("RED", 16, 30), ...]

    アライメントファイルを読み込んで、(単語, 開始フレーム, 終了フレーム)のタプルのリストを返す
    """
    words = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start_frame = int(parts[0])
                end_frame = int(parts[1])
                word = parts[2]
                words.append((word, start_frame, end_frame))
    return words

def load_video_frames(path):
    """
    Loads a video (.mpg) file and returns a list of frames (as numpy arrays).
    動画ファイルをフレーム単位で読み込み、numpy配列のリストとして返す
    """
    cap = cv2.VideoCapture(str(path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames
