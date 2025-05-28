from pathlib import Path
import cv2
from src.params import RAW_DATA_DIR

SCALE = 1 / 1000  # Convert alignment timestamp (ms) to seconds
                 # .align files give start/end in milliseconds
                 # Frame index ≈ time_in_seconds * fps (≈25)

def load_alignment_paths():
    """
    Return a sorted list of .align alignment file paths
    under raw_data/alignments.
    """
    alignments_dir = RAW_DATA_DIR / "alignments"
    return sorted(alignments_dir.glob("**/*.align"))

def load_video_paths():
    """
    Return a sorted list of .mpg video file paths
    under raw_data/videos.
    """
    videos_dir = RAW_DATA_DIR / "videos"
    return sorted(videos_dir.glob("**/*.mpg"))

def read_alignment_file(path):
    """
    Read a .align file and return a list of (word, start_time, end_time) tuples.
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
    Load video frames as a list of numpy arrays.
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

def load_data():
    """
    Load all video and alignment files, and extract per-word frame sequences.

    Process:
    - For each .mpg video, find its corresponding .align file
    - For each word in the alignment, convert its start/end time to frame indices
    - Extract the corresponding sequence of frames (as numpy arrays)

    Returns:
        X (list): List of sequences of frames (1 word = [frame1, frame2, ...])
        y (list): List of word labels (e.g., ["hello", "world", ...])
    """
    video_paths = load_video_paths()
    X, y = [], []

    for video_path in video_paths:
        speaker = video_path.parts[-2]
        video_name = video_path.stem
        alignment_path = RAW_DATA_DIR / "alignments" / speaker / f"{video_name}.align"

        if not alignment_path.exists():
            continue

        frames = load_video_frames(video_path)
        n_frames = len(frames)

        word_infos = read_alignment_file(alignment_path)

        for word, start, end in word_infos:
            start_idx = max(0, int(start * SCALE))
            end_idx = min(n_frames - 1, int(end * SCALE))

            if start_idx >= end_idx:
                continue

            word_frames = frames[start_idx:end_idx + 1]
            X.append(word_frames)
            y.append(word)

    return X, y

def load_test_data(limit=100):
    """
    Load a small portion of data for evaluation (test mode).
    """
    X, y = load_data()
    X = X[:limit]
    y = y[:limit]
    return X, y
