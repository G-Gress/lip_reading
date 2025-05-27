import streamlit as st
import os

# CUSTOM STYLE

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }

    .transcription-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
        font-size: 18px;
        color: #2c3e50;
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

    #Debug
    st.write("Video folder path:", VIDEO_FOLDER)
    # st.write("Available files:", video_files)

    if not video_files:
        st.warning("No video files found in the folder.")
    else:
        # Streamlit UI to select video
        selected_video = st.selectbox("Choose a video:", video_files)

        # Load and show selected video
        video_path = os.path.join(VIDEO_FOLDER, selected_video)
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
        st.video(video_bytes)

with col2:
    st.header("🗣️ Transcribed Text (Simulated)")
    # Dummy transcription (placeholder for model output)
    st.markdown("This is where the transcription from the lip reading model will appear.")

    transcription = ""
    if selected_video == "bbaf2n.mp4":
        st.success("Simulated: 'Bin blue at f two now'")
    elif selected_video == "sample2.mp4":
        st.success("Simulated: 'Can you read my lips?'")
    else:
        st.info("Simulated: 'Lip reading in progress...'")

    st.markdown(f"<div class='transcription-box'>{transcription}</div>", unsafe_allow_html=True)
