# Video Annotator - Sign Language Segmentation Tool

A fast, keyboard-driven desktop tool to annotate sign language videos by marking sentence boundaries.

---

## Prerequisites

### For Annotation (Required)
| Requirement | Version |
|---|---|
| [Node.js](https://nodejs.org/) | 18 or higher |

### For Video Cutting (Optional)
| Requirement | Version |
|---|---|
| Python | 3.8 or higher |
| imageio, imageio-ffmpeg | Latest (pip install) |

Note: `imageio-ffmpeg` includes its own ffmpeg binary, no system installation needed.

---

## Installation & Launch

```bash
npm install
npm start
```

---

## Usage

### 1. Open a Folder
Click "Open Folder" in the sidebar and select the folder containing your videos (.mp4, .avi, .mov, .mkv, .webm).

The sidebar will list all videos. Already-annotated videos are highlighted with a green indicator.
The app auto-loads the first unannotated video.

### 2. Annotate
Watch the video and press Space at each sentence boundary.
A green marker appears on the timeline and the timestamp is added to the Splits panel.

### 3. Save & Continue
- Press `S` to save current annotation
- Press `N` to auto-save and load the next video

The annotation is saved as a JSON file next to the video:
```
videos/
  video_001.mp4
  video_001.json   (saved here)
```

### 4. Resume
Reload the app and open the same folder. Previously marked splits reload automatically.

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| Space | Mark split at current timestamp |
| Left/Right Arrow | Seek +/- 1 second |
| Shift + Left/Right Arrow | Seek +/- 0.2 second |
| Up/Down Arrow | Speed up / slow down (0.25x to 2x) |
| Ctrl + Z | Undo last split |
| S | Save annotation |
| N | Next video (auto-saves) |
| P or . | Toggle play/pause |

---

## Output Format

```json
{
  "video": "video_001.mp4",
  "fps": 30,
  "duration": 876.3,
  "splits": [12.34, 19.80, 25.10, 31.92]
}
```

Segments are inferred from splits. Example:
- Segment 1: 0 to 12.34 seconds
- Segment 2: 12.34 to 19.80 seconds
- Segment 3: 19.80 to 25.10 seconds

---

## Cut Video into Clips

After annotating, run the Python cutter to split videos into segments:

First, install dependencies:
```bash
pip install imageio imageio-ffmpeg
```

Then cut videos:
```bash
# Cut one video
python cut_video.py videos/video_001.json

# Custom output folder
python cut_video.py videos/video_001.json --out_dir clips/video_001/

# Batch - cut all annotated videos in a folder
for %f in (videos\*.json) do python cut_video.py "%f" --out_dir clips\
```

Output clips: `video_001_001.mp4`, `video_001_002.mp4`, etc.

Note: Uses ffmpeg stream-copy via imageio-ffmpeg (no re-encoding, very fast). Already-existing clips are skipped.

---

## Project Structure

```
video-annotator/
├── main.js         Electron main process
├── preload.js      IPC bridge for secure communication
├── index.html      UI layout
├── style.css       Dark theme styles
├── renderer.js     UI logic and keyboard shortcuts
├── cut_video.py    Video segment cutter (Python + ffmpeg)
├── package.json    Electron app configuration
├── .gitignore      Git ignore rules
└── README.md       This file
```

---

## Features

- Keyboard-first design for fast annotation
- Real-time timeline visualization
- Auto-save on video switch
- Persistent annotations (JSON format)
- Batch video processing support
- Python script for automated video segmentation

---

## License

This project is provided as-is for sign language annotation tasks.

---

## Support

For issues or improvements, please create an issue in the repository.
