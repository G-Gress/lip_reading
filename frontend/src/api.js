// src/api.js
export async function sendFrameToAPI(imageBase64) {
  try {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageBase64 }),
    });

    const data = await response.json();
    return data.transcription || '[No transcription returned]';
  } catch (error) {
    console.error("API call failed:", error);
    return '[API error]';
  }
}
