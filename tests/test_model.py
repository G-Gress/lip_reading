import pytest
from src.ml_logic.model import build_model, save_model, load_model
import os

def test_build_model():
    model = build_model()
    assert model is not None
    assert hasattr(model, "predict")

def test_save_and_load_model(tmp_path):
    model = build_model()
    model_path = tmp_path / "test_model.keras"
    save_model(model, path=str(model_path))

    assert model_path.exists(), "Model file was not saved."

    loaded_model = load_model(path=str(model_path))
    assert loaded_model is not None
    assert hasattr(loaded_model, "predict")
