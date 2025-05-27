// src/App.jsx
import React, { useRef, useEffect, useState } from 'react';
import { sendFrameToAPI } from './api';

function App() {
  const videoRef = useRef(null);
  const [transcription, setTranscription] = useState('');

  useEffect(() => {
    // Start webcam
    navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user' }
    })
      .then(stream => {
        videoRef.current.srcObject = stream;
      })
      .catch(err => {
        console.error("Error accessing camera: ", err);
      });

    // Start interval to capture frames
    const interval = setInterval(() => {
      captureAndSendFrame();
    }, 2000); // every 2 seconds

    return () => clearInterval(interval);
  }, []);

  // Old Version of the capture and send to frame
  // const captureAndSendFrame = async () => {
  //   const canvas = document.createElement('canvas');
  //   const video = videoRef.current;
  //   if (!video) return;

  //   canvas.width = video.videoWidth;
  //   canvas.height = video.videoHeight;
  //   const ctx = canvas.getContext('2d');
  //   ctx.drawImage(video, 0, 0);
  //   const imageBase64 = canvas.toDataURL('image/jpeg');

  //   const result = await sendFrameToAPI(imageBase64);
  //   if (result) setTranscription(result);
  // };

  // New Version that sends smaller blobs than Data-URLs.
  const captureAndSendFrame = async () => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

  // toBlob is asynchronous
    canvas.toBlob(async blob => {
      try {
      const form = new FormData();
      form.append('frame', blob, 'frame.jpg');
      const result = await sendFrameToAPI(form);
      setTranscription(result || '');
      } catch (e) {
      console.error('Failed to send frame:', e);
      }
    }, 'image/jpeg');
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>🎤 Lip Reading App</h1>
      <video ref={videoRef} autoPlay style={{ width: '100%', maxWidth: '600px', transform: 'scaleX(-1)' }} />
      <h2>📝 Transcription:</h2>
      <p style={{ fontSize: '1.5rem' }}>{transcription}</p>
    </div>
  );
}

export default App;
