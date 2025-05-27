from src.params import RAW_DATA_DIR
from src.ml_logic.data import load_video_paths, load_video_frames, read_alignment_file
from src.ml_logic.preprocessor import crop_face_region
from pathlib import Path
import cv2

# 1本だけ取得
video_path = load_video_paths()[0]
video_name = video_path.stem
speaker = video_path.parts[-2]
alignment_path = RAW_DATA_DIR / "alignments" / speaker / f"{video_name}.align"

print(f"🎥 Video file: {video_path}")
print(f"📄 Alignment file: {alignment_path}")

# 動画フレーム読み込み
frames = load_video_frames(video_path)
print(f"🖼️ Total video frames: {len(frames)}")

# アライメント読み込み
word_infos = read_alignment_file(alignment_path)
print(f"📝 Alignment entries: {len(word_infos)}")
for word, start, end in word_infos:
    print(f"   🔤 Word: {word}, frames: {start}-{end}")

# 顔検出の確認
print("\n🧠 Face detection check:")
for i, frame in enumerate(frames[:10]):  # 最初の10フレームだけチェック
    cropped = crop_face_region(frame)
    h, w = cropped.shape[:2]
    if h > 0 and w > 0:
        print(f"✅ Frame {i}: face detected ({h}x{w})")
    else:
        print(f"❌ Frame {i}: no face detected")
