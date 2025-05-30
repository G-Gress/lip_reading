import streamlit as st
import os

from src.ml_logic.model import load_model
from src.ml_logic.preprocess_for_streamlit import preprocess_video
from src.ml_logic.alphabet import decode
from src.ml_logic.data import return_words
import tensorflow as tf
import numpy as np

# Encoder
# Vocabulary to encode
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# Char to num converter
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Num to char converter
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)


# LOAD MODEL WITH UPDATED WEIGHTS
@st.cache_resource
def load_lipnet_model(weights = "model_weights/checkpoint_epoch25_loss0.79.weights.h5"):

    model = load_model() #used with Kazus ml_logic.model load_model.py
    model.load_weights(weights)
    return model

model = load_lipnet_model()


# CUSTOM STYLE

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }

    </style>
""", unsafe_allow_html=True)


# Title
st.title("Lip Reading MVP")

#Video Option Selection
st.markdown("Select a video to simulate lip reading transcription.")

#Generate two columns
col1, col2 = st.columns([0.6, 0.4], gap='medium')

with col1:
    st.header("🎥 Video Preview")
    # Set correct path to videos
    VIDEO_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'raw_data', 'videos', 's1_mp4'))

    # Get video files
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.mov', '.avi', '.mpg'))]

    # #Debug
    # st.write("Video folder path:", VIDEO_FOLDER)
    # # st.write("Available files:", video_files)

    if not video_files:
        st.warning("No video files found in the folder.")
    else:
        # Streamlit UI to select video
        selected_video = st.selectbox("Choose a video:", video_files)

        # Load and show selected video
        video_path = os.path.join(VIDEO_FOLDER, selected_video)
        video_name, ext = os.path.splitext(selected_video)
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
        st.video(video_bytes)



with col2:
    st.header("🗣️ Transcribed Text (Simulated)")

    st.markdown("These are the original words")

    #Select correct alignment file and read
    alignment_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'raw_data', 'alignments', 's1', f'{video_name}.align'))
    words_in_alignment = return_words(alignment_path)
    st.success(words_in_alignment)

    if st.button("Transcribe"):

        #preprocess and predict
        frames = preprocess_video(video_path)

        # Testing model and frame shapes
        # st.write("Model input shape:", model.input_shape)
        # st.write("Preprocessed input shape:", frames.shape)
        frames = np.expand_dims(frames, axis=0)  # shape becomes (1, 75, 46, 140, 1)

        yhat = model.predict(frames) #shape (1, 75, 41)
        # Testing prediction output shape
        # st.write(yhat.shape)
        # st.write(yhat.dtype)

        sequence_length = [75]

        decoded = tf.keras.backend.ctc_decode(yhat, sequence_length, greedy=False)[0][0].numpy()

        for x in range(len(yhat)):
            prediction = tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8')

        # prediction = decode(decoded)

        st.success(f"Transcription: {prediction}")
