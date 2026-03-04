# Video Annotator - Sign Language Segmentation Tool

A fast, keyboard-driven desktop tool to annotate sign language videos by marking sentence boundaries, now enhanced with **AI-powered split suggestions**.

---

## 🌟 Features

- **Keyboard-first design**: High-speed annotation using shortcuts.
- **AI Suggestion**: Automatically detects resting poses (hands down) to suggest potential split points.
- **Real-time Progress**: Visual feedback during AI analysis.
- **Timeline Visualization**: Interactive canvas showing split points and current playback.
- **Auto-save**: Never lose progress when switching between videos.
- **Python Integration**: Includes scripts for both AI analysis and high-speed video cutting.

---

## 🛠 Prerequisites

### 1. For the Desktop App (Required)
- [Node.js](https://nodejs.org/) (v18 or higher)

### 2. For AI Suggestions (Required for "Suggest by AI")
- Python 3.8 - 3.11 (MediaPipe supports up to 3.11)
- Install dependencies:
  ```bash
  pip install opencv-python mediapipe numpy
  ```

### 3. For Video Cutting (Optional)
- Python 3.8+
- Install dependencies:
  ```bash
  pip install imageio imageio-ffmpeg
  ```

---

## 🚀 Installation & Setup

1. **Clone and Install JS dependencies:**
   ```bash
   npm install
   ```

2. **Configure Python Path:**
   Open `main.js` and update the `pythonPath` variable to point to your Python executable or Virtual Environment:
   ```javascript
   // In main.js (around line 118)
   const pythonPath = 'path/to/your/venv/Scripts/python.exe'; // Windows
   // or 'path/to/your/venv/bin/python'; // Linux/macOS
   ```

3. **Start the app:**
   ```bash
   npm start
   ```

---

## 💡 Usage Guide

### 1. Open a Folder
Click **"Open Folder"** and select a directory. The sidebar will list all compatible videos.
- 🟢 Green dot: Already annotated.
- ⚪ Gray dot: New video.

### 2. AI-Assisted Annotation (New!)
If a video is new, a **"Suggest by AI"** button will appear in the transport bar.
- Click it to start the MediaPipe analysis.
- The button will show real-time progress (e.g., `Analyzing... 45%`).
- The AI detects "Resting Poses" (when the signer lowers their hands) and marks the middle of those moments as splits.
- You can then manually add, move, or delete these suggestions.

### 3. Manual Annotation
- Press `Space` to mark a split at the current time.
- Use `Arrow Keys` for navigation (see shortcuts below).
- Click on any timestamp in the **Splits Panel** to jump to that moment.

### 4. Save & Export
- Press `S` to save. Annotations are stored as `<video_name>.json` in the same folder.
- Press `N` to move to the next video.

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| **Space** | Mark split at current timestamp |
| **Left / Right** | Seek +/- 1 second |
| **Shift + Left / Right** | Seek +/- 0.2 second (Fine tuning) |
| **Up / Down** | Adjust playback speed (0.25x to 2x) |
| **Ctrl + Z** | Undo last split |
| **S** | Save current annotation |
| **N** | Next video (Auto-saves current) |
| **P** or **.** | Toggle Play / Pause |

---

## 🤖 How the AI works (`suggest_splits.py`)

The AI uses **MediaPipe Pose Landmarking** to analyze the video at 6 FPS (for speed). It triggers a split suggestion when:
1. **Wrists** are below the elbows.
2. **Wrists** are near the hip level.
3. **Low Motion**: The distance between hand coordinates in consecutive frames is minimal.
4. **Duration**: The resting pose lasts longer than 0.6 seconds.

---

## ✂️ Cutting Videos into Clips

Once you have the `.json` files, use the provided Python script to generate the actual video segments. This process is extremely fast as it uses `stream copy` (no re-encoding).

```bash
# Cut a specific video
python cut_video.py path/to/video_001.json

# Cut all videos in a folder and save to a "clips" directory
python cut_video.py path/to/video_001.json --out_dir ./my_clips/
```

---

## 📂 Project Structure

- `main.js`: Electron main process & Python spawning logic.
- `renderer.js`: UI logic, Canvas timeline, and event handling.
- `suggest_splits.py`: MediaPipe logic for resting pose detection.
- `cut_video.py`: FFmpeg-based script for segmenting videos.
- `index.html` / `style.css`: The "Cyber-Dark" themed interface.

---

## 📝 License

This tool is designed for research and data labeling in the field of Sign Language Recognition (SLR). Feel free to modify the `is_resting_pose` logic in `suggest_splits.py` to better fit your specific dataset.