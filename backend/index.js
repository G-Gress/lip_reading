// server/index.js
import express from "express";
import multer from "multer";
import path from "path";
import fs from "fs";
import { spawn } from "child_process";
import ffmpeg from "fluent-ffmpeg";

const app = express();
const upload = multer({ dest: "uploads/" });

app.post("/api/predict", upload.single("video"), async (req, res) => {
  try {
    const inputPath = req.file.path;                  // e.g. uploads/abc123
    const webmPath = inputPath + ".webm";
    const mp4Path  = inputPath + ".mp4";

    // Rename to .webm
    await fs.promises.rename(inputPath, webmPath);

    // Convert WebM → MP4 at 25 fps (or whatever your model needs)
    await new Promise((resolve, reject) => {
      ffmpeg(webmPath)
        .duration(3)
        .videoFilters('scale=360:288')
        .fps(25)
        .output(mp4Path)
        .on("end", resolve)
        .on("error", reject)
        .run();
    });

    // Call your Python inference script:
    const py = spawn("python3", [
      "inference.py",
      "--video", mp4Path,
      "--weights", "model_weights/checkpoint_epoch25_loss0.79.weights.h5"
    ]);

    let output = "";
    py.stdout.on("data", data => { output += data.toString(); });
    py.stderr.on("data", data => console.error("PY ERR:", data.toString()));

    py.on("close", async (code) => {
      if (code !== 0) {
        return res.status(500).json({ error: "Model inference failed" });
      }
      // Assume your script prints JSON: { transcription: "..." }
      const result = JSON.parse(output);
      res.json(result);
      // (Optionally clean up files here)
      try {
        await fs.promises.unlink(webmPath);
        await fs.promises.unlink(mp4Path);
      } catch (cleanupErr) {
        console.error("Cleanup failed:", cleanupErr);
      }
    });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.listen(8000, () => console.log("Listening on http://localhost:8000"));
