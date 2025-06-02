// src/App.jsx
import React, { useRef, useState } from "react";

export default function App() {
  const videoRef = useRef(null);
  const recorderRef = useRef(null);
  const [recording, setRecording] = useState(false);
  const [transcription, setTranscription] = useState("");

  // Start webcam on mount
  React.useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false })
      .then(stream => {
        videoRef.current.srcObject = stream;
      })
      .catch(console.error);
  }, []);

  const startRecording = () => {
    const stream = videoRef.current.srcObject;
    const recorder = new MediaRecorder(stream, { mimeType: "video/webm" });
    const chunks = [];
    recorder.ondataavailable = e => chunks.push(e.data);
    recorder.onstop = async () => {
      const blob = new Blob(chunks, { type: "video/webm" });
      console.log("⏹️ Recording stopped, blob size:", blob.size);
      await sendToServer(blob);
    };
    recorder.start();
    recorderRef.current = recorder;
    setRecording(true);

    // Stop after 3 seconds
    setTimeout(() => {
      recorder.stop();
      setRecording(false);
    }, 3000);
  };

  const sendToServer = async (blob) => {
    const form = new FormData();
    form.append("video", blob, "clip.webm");

    try {
      console.log("🛰️ Sending blob to server…", blob);
      const res = await fetch("/api/predict", {
        method: "POST",
        body: form,
      });
      console.log("⬅️ Response status:", res.status);

      if (!res.ok) {
        const text = await res.text();
        console.error("🚨 Server error response:", text);
        setTranscription(`Error: ${text}`);
        return;
      }

      const data = await res.json();
      console.log("📄 Response JSON:", data);
      setTranscription(data.transcription || "[no transcription]");
    } catch (err) {
      console.error("❌ Fetch failed:", err);
      setTranscription(`Fetch error: ${err.message}`);
    }
  };


  return (
    <div className="p-4">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ width: 320, height: 240, border: "1px solid #ccc" }}
      />
      <div className="mt-2">
        <button
          onClick={startRecording}
          disabled={recording}
          className="px-4 py-2 bg-blue-500 text-white rounded"
        >
          {recording ? "Recording…" : "Record 3 s"}
        </button>
      </div>
      {transcription && (
        <p className="mt-4"><strong>Transcription:</strong> {transcription}</p>
      )}
    </div>
  );
}



// FOR LIVESTREAM APP (IF POSSIBLE)
// // src/App.jsx
// import React, { useRef, useEffect, useState } from 'react';
// import { sendFrameToAPI } from './api';

// function App() {
//   const videoRef = useRef(null);
//   const [transcription, setTranscription] = useState('');

//   useEffect(() => {
//     // Start webcam
//     navigator.mediaDevices.getUserMedia({
//       video: { facingMode: 'user' }
//     })
//       .then(stream => {
//         videoRef.current.srcObject = stream;
//       })
//       .catch(err => {
//         console.error("Error accessing camera: ", err);
//       });

//     // Start interval to capture frames
//     const interval = setInterval(() => {
//       captureAndSendFrame();
//     }, 2000); // every 2 seconds

//     return () => clearInterval(interval);
//   }, []);

//   // New Version that sends smaller blobs than Data-URLs.
//   const captureAndSendFrame = async () => {
//     const video = videoRef.current;
//     if (!video || video.videoWidth === 0) return;

//     const canvas = document.createElement('canvas');
//     canvas.width = video.videoWidth;
//     canvas.height = video.videoHeight;
//     canvas.getContext('2d').drawImage(video, 0, 0);

//   // toBlob is asynchronous
//     canvas.toBlob(async blob => {
//       try {
//       const form = new FormData();
//       form.append('frame', blob, 'frame.jpg');
//       const result = await sendFrameToAPI(form);
//       setTranscription(result || '');
//       } catch (e) {
//       console.error('Failed to send frame:', e);
//       }
//     }, 'image/jpeg');
//   };

//   return (
//     <div style={{ padding: '20px' }}>
//       <h1>🎤 Lip Reading App</h1>
//       <video ref={videoRef} autoPlay style={{ width: '100%', maxWidth: '600px', transform: 'scaleX(-1)' }} />
//       <h2>📝 Transcription:</h2>
//       <p style={{ fontSize: '1.5rem' }}>{transcription}</p>
//     </div>
//   );
// }

// export default App;
