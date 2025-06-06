import streamlit as st
import streamlit.components.v1 as components

import io
import os
import base64

from src.ml_logic.model import load_model
from src.inference.inference import run_inference_streamlit
import numpy as np
import imageio
import time

from PIL import Image

def show_ndarray_frames_as_gif(frames_np: list[np.ndarray]):
  """
  Takes a list of ndarrays as input, representing frames of an image,
  and embeds them as a GIF on the streamlit page.  The GIF data is
  embedded directly on the page using a blob URL, bypassing the filesystem.
  """

  # step 1: convert each ndarray frame to a PIL image
  frames_pil = [Image.fromarray(frame) for frame in frames_np]

  # step 2: use pillow to save the .gif image to a byte stream
  gif_bytes = io.BytesIO(b"")
  frames_pil[0].save(gif_bytes, format="gif", save_all=True, append_images=frames_pil[1:], loop=0)

  # step 3: convert the byte stream to a base64 string representation
  base64_image_string = base64.b64encode(gif_bytes.getvalue()).decode()

  # step 4: create a blob URL
  img_data_url = 'data:image/gif;base64,' + base64_image_string

  # step 5: embed the blob URL directly inside of an HTML image tag
  st.write(
    f'<img src="{img_data_url}" width="600"/>',
    unsafe_allow_html=True
  )

# Load model with updated weights
@st.cache_resource
def load_lipnet_model(weights = "model_weights/checkpoint_epoch20_loss0.92.weights.h5"):

    model = load_model() #used with Kazus ml_logic.model load_model.py
    model.load_weights(weights)
    return model

model = load_lipnet_model()


# Custom style
# st.markdown("""
#     <style>
#     .main {
#         background-color: #f8f9fa;
#     }
#     </style>
# """, unsafe_allow_html=True)

# Title
st.title("Lip Reading MVP")

#Video Option Selection
st.markdown("Select a video to simulate lip reading transcription.")
st.header("🎥 Video Preview")

# Set correct path to videos
VIDEO_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'raw_data', 'videos', 's1_mp4'))

# Get video files
video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.mov', '.avi', '.mpg'))]

if not video_files:
    st.warning("No video files found in the folder.")
else:
    # Streamlit UI to select video
    selected_video = st.selectbox("Choose a video:", video_files)

    # Load and show selected video
    video_path = os.path.join(VIDEO_FOLDER, selected_video)

    with open(video_path, 'rb') as f:
        video_bytes = f.read()

    video_base64 = base64.b64encode(video_bytes).decode()

    components.html(f"""
    <video width="100%" muted controls>
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """,
    height=550
    )

    # Run predictions and get preprocessed videos
    prediction, frames  = run_inference_streamlit(video_path=video_path, model=model)


if st.button("Transcribe"):

    st.header("Computer Vision")
    st.info('This is all the machine learning model sees when making a prediction')


    # Make Animation for computer vision
    frames_for_gif = [(frame.numpy().squeeze() * 255).astype(np.uint8) for frame in frames]

    show_ndarray_frames_as_gif(frames_for_gif)

    # imageio.mimsave('animation.gif', frames_for_gif, fps=10)
    # st.image('animation.gif', width=600)

    #Stall Transcription to give time to talk about computer vision
    time.sleep(2)

    #Transcription prediction
    st.header("🗣️ Transcribed Text")
    st.write("Transcription:")
    st.success(f"{prediction}")

    # Code to Implement showing the original text
    # time.sleep(0.5)

    # try:
    #     #Select correct alignment file and read if available
    #     video_name, ext = os.path.splitext(selected_video)
    #     alignment_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'raw_data', 'alignments', 's1', f'{video_name}.align'))
    #     words_in_alignment = return_words(alignment_path)
    #     st.markdown("Versus the original words")
    #     st.success(words_in_alignment)

    # except Exception as e:
    #     # Return nothing if not available
    #     pass
