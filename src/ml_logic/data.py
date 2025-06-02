import os
from pathlib import Path
import cv2
from src.params import RAW_DATA_DIR
import tensorflow as tf
from src.ml_logic.alphabet import char_to_num



# Data loader
def load_video(path: str) -> tf.Tensor:
    '''
    Load a video from a path, convert it to grayscale, crop it to the face,
    normalize it with z-score normalization, and return a numpy array of the frames.
    '''
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
      # Get one frame as a numpy array
      ret, frame = cap.read()
      # Grayscale conversion
      #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # => Returns 2D image
      gray = tf.image.rgb_to_grayscale(frame) # => Returns 3D tensor
      # Add the frame to the list
      frames.append(gray[190:236, 80:220, :])
    # Release the video
    cap.release()

    # Normalize the data with z-score normalization
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))

    return tf.cast((frames - mean), tf.float32) / std

def load_alignments(path: str) -> tf.Tensor:
    with open(path, "r") as f:
        lines = f.readlines()

    tokens = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3 and parts[2] != "sil":
            tokens.append(parts[2])  # ["bin", "blue", ...]

    joined = " ".join(tokens)  # "binblueattwonow"
    chars = tf.strings.unicode_split(joined, input_encoding='UTF-8')

    return char_to_num(chars)

def load_data(path: tf.Tensor):
  '''
  Take a path as a tensor, load the video and corresponding alignments,
  and return two tensors, one for the processed frames,
  one for the encoded tokens.
  '''
  # Convert the path back into a string
  path = bytes.decode(path.numpy())

  # Get file name from path
  file_name = path.split('/')[-1].split('.')[0]

  # Get path from file name
  video_path = os.path.join('raw_data/videos/s1',f'{file_name}.mpg')
  alignment_path = os.path.join('raw_data/alignments/s1',f'{file_name}.align')

   # Get path from file name
#   video_path = os.path.join('./drive/MyDrive/Project/Lip_reading/raw_data//videos/s1',f'{file_name}.mpg')
#   alignment_path = os.path.join('./drive/MyDrive/Project/Lip_reading/raw_data//alignments/s1',f'{file_name}.align')

  # Load data
  frames = load_video(video_path)
  alignments = load_alignments(alignment_path)

  return frames, alignments

def load_data_mp4(path: tf.Tensor):
    '''
    Load a .mp4 video for inference (no alignment needed).
    Returns: (video_frames_tensor, dummy_label_tensor)
    '''
    path_str = path.numpy().decode("utf-8")
    frames = load_video(path_str)
    dummy_align = tf.constant([0], dtype=tf.int64)  # Dummy label since not used
    return frames, dummy_align
