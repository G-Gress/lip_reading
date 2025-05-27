from pathlib import Path
import cv2
from src.params import RAW_DATA_DIR

def load_alignment_paths():
    """
    Returns a sorted list of all .align alignment file paths
    under raw_data/alignments
    """
    alignments_dir = RAW_DATA_DIR / "alignments"
    align_files = sorted(alignments_dir.glob("**/*.align"))
    return align_files

def load_video_paths():
    """
    Returns a sorted list of all .mpg video file paths
    under raw_data/videos
    """
    videos_dir = RAW_DATA_DIR / "videos"
    mpg_files = sorted(videos_dir.glob("**/*.mpg"))
    return mpg_files

def read_alignment_file(path):
    """
    Reads an alignment  file and returns a list of (word, start_frame, end_frame) tuples.

    Example:
    [("PLACE", 0, 15), ("RED", 16, 30), ...]
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
    Load and prepare video frame sequences and their corresponding words.

    Returns:
        X (list): List of frame sequences (each item = list of frames for one word)
        y (list): List of word labels
    """
    video_paths = load_video_paths()

    X = []
    y = []

    for video_path in video_paths:
        # Guess the corresponding alignment file path
        speaker = video_path.parts[-2]
        video_name = video_path.stem  # e.g., "vid1" without extension
        alignment_path = Path(RAW_DATA_DIR) / "alignments" / speaker / (video_name + ".align")

        if not alignment_path.exists():
            continue  # Skip if the alignment file does not exist

        # Load video frames
        frames = load_video_frames(video_path)

        # Load alignment content
        word_infos = read_alignment_file(alignment_path)

        for word, start, end in word_infos:
            word_frames = frames[start:end + 1]  # Extract relevant frame segment
            X.append(word_frames)
            y.append(word)

    return X, y
