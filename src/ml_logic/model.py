import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Input

def build_model():
    """
    Build a simple 3D CNN model for lip-reading.

    Input shape:
    - (time, height, width, channels) = (24, 46, 140, 1)

    Output:
    - softmax over 28 classes (a-z + apostrophe + space)

    ⚠️ NOTE:
    This model is currently designed to predict only **one character per video**.
    It works for classification tasks where each input corresponds to a **single letter**.

    💡 If you want to extend this model to predict **words or full sentences** in the future:
    - You will need to change the model architecture to handle sequences (e.g., RNN, CTC, Transformer)
    - You may need to modify:
        - The output layer (e.g., sequence of characters)
        - The loss function (e.g., CTC loss instead of sparse categorical crossentropy)
        - The labels (e.g., use full sequences instead of single characters)
    """

    model = Sequential([
        Input(shape=(24, 46, 140, 1)),  # Time, Height, Width, Channel

        Conv3D(8, (3, 3, 3), activation='relu', padding='same'),
        MaxPooling3D((1, 2, 2)),

        Conv3D(16, (3, 3, 3), activation='relu', padding='same'),
        MaxPooling3D((2, 2, 2)),

        Flatten(),
        Dense(64, activation='relu'),
        Dense(28, activation='softmax')  # 26 letters + apostrophe + space
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def save_model(model, path="models/lip_model.keras"):
    """
    Save the trained model to the specified path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"✅ Model saved to {path}")


def load_model(path="models/lip_model.keras"):
    """
    Load a saved model from the given path.
    Returns the model object if found, else None.
    """
    if os.path.exists(path):
        print(f"📦 Loading model from {path}")
        return keras_load_model(path)
    else:
        print(f"❌ No model found at {path}")
        return None
