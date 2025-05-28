from src.ml_logic import data

def test_load_video_paths():
    video_paths = data.load_video_paths()
    assert isinstance(video_paths, list)
    assert all(str(p).endswith(".mpg") for p in video_paths)

def test_load_alignment_paths():
    alignment_paths = data.load_alignment_paths()
    assert isinstance(alignment_paths, list)
    assert all(str(p).endswith(".align") for p in alignment_paths)

def test_load_data_output_format():
    X, y = data.load_data()
    assert isinstance(X, list)
    assert isinstance(y, list)
    assert len(X) == len(y)
