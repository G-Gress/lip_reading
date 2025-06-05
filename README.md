# 🧠 Lip Reading

A Streamlit-based demo app for visual speech recognition (lip reading), using a deep learning model inspired by [LipNet](https://arxiv.org/abs/1611.01599).

## 🚀 Overview

This project takes a short video clip of a person speaking, processes the visual frames, and predicts the transcribed text **without using audio**. It's a prototype aimed at exploring silent communication tools for accessibility and speech-impaired settings.

Key features:
- Upload and preview pre-processed video clips.
- Real-time frame preprocessing and model inference.
- Generates a visual animation of the video used by the model.
- Returns a transcription decoded from the model's CTC output.
- Optional: Display ground-truth alignment if available.

---
