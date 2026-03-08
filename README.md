# 🏭 AI-Based Industrial Safety Monitoring System Using Computer Vision

> **B.Tech Final Year Project** | Python • OpenCV • YOLOv8 • PyTorch

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [What is Used in This Project](#-what-is-used-in-this-project)
- [Project Folder Structure](#-project-folder-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [How Each File Works](#-how-each-file-works)
- [Alert System](#-alert-system)
- [Dataset](#-dataset)
- [Bug Fixes Applied](#-bug-fixes-applied)
- [Future Improvements](#-future-improvements)
- [Viva Tips](#-viva-tips)

---

## 📖 Project Overview

This project is an **AI-powered real-time industrial safety monitoring system** that uses
**CCTV cameras** and **computer vision** to protect workers in dangerous industrial
environments such as:

- 🪨 Stone crushing plants
- ⚙️ Press machine workshops
- 🔄 Conveyor belt facilities
- 🏗️ Heavy manufacturing units

The system continuously watches video footage, detects when a **worker enters a
predefined danger zone** near a hazardous machine, and immediately triggers
multiple safety alerts — including a simulated machine stop command.

---

## ❗ Problem Statement

Every year, thousands of industrial workers are injured or killed due to accidental
contact with dangerous machinery. Traditional safety methods fail because:

- Human supervisors **cannot watch every zone** at the same time
- Workers **ignore warning signs** or bypass physical barriers
- Manual reaction time is **too slow** to prevent accidents
- Surveillance footage is only reviewed **after** an incident occurs

This project solves all these problems with a **24/7 automated AI system**.

---

## 🛠️ What is Used in This Project

This section explains every technology, library, and tool used in this project and
**why** it was chosen.

---

### 🐍 Python
**What it is:**
Python is the programming language used to write the entire project.

**Why used:**
- Most popular language for AI and Machine Learning
- Simple and easy to read syntax
- Huge library support (OpenCV, PyTorch, NumPy, etc.)
- Runs on Windows, Linux, and Mac

---

### 👁️ OpenCV (cv2)
**What it is:**
OpenCV (Open Source Computer Vision Library) is a library for real-time image
and video processing.

**Why used:**
- Reads video files and live camera feeds using `VideoCapture`
- Resizes, crops, and processes each video frame
- Draws bounding boxes, danger zone rectangles, and text overlays on frames
- Displays the live output window using `imshow`
- Saves screenshots and alert snapshots using `imwrite`
- Has a built-in **HOG Person Detector** used when YOLOv8 is not available

**Where used in code:**
```python
import cv2
cap = cv2.VideoCapture("video.mp4")   # open video
ret, frame = cap.read()               # read each frame
cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)  # draw box
cv2.imshow("Safety Monitor", frame)   # show window
```

---

### 🔢 NumPy
**What it is:**
NumPy is a Python library for numerical computing and array operations.

**Why used:**
- Video frames from OpenCV are stored as **NumPy arrays** with shape
  `(height, width, 3)` for BGR color values
- Used to create blank frames for paused/error states
- Used for mathematical operations on coordinates
- YOLOv8 and HOG detectors internally use NumPy arrays

**Where used in code:**
```python
import numpy as np
blank = np.zeros((540, 960, 3), np.uint8)  # create black frame
```

---

### 🐼 Pandas
**What it is:**
Pandas is a Python library for data analysis and working with tabular data
(like Excel spreadsheets).

**Why used:**
- Saves the **alert log** to a CSV file with timestamps, frame numbers,
  and number of persons detected in the danger zone
- Creates annotation template CSV files for the dataset
- Allows easy reading and analysis of alert history

**Where used in code:**
```python
import pandas as pd
df = pd.DataFrame(all_alerts)
df.to_csv("alert_log.csv", index=False)
```

---

### 📊 Matplotlib
**What it is:**
Matplotlib is a Python library for creating charts and data visualizations.

**Why used:**
- Generates the **safety analysis report chart** after processing a video
- Creates a **timeline chart** showing when alerts were triggered
- Creates a **bar chart** summarizing total detections and alerts per video
- Saves charts as PNG images for the project report

**Where used in code:**
```python
import matplotlib.pyplot as plt
plt.plot(frames, alert_counts)
plt.savefig("safety_analysis_report.png")
```

---

### 🔦 PyTorch
**What it is:**
PyTorch is a deep learning framework used to build and run neural networks.

**Why used:**
- YOLOv8 is built on top of PyTorch
- Handles all the deep learning computations internally
- Provides GPU (CUDA) support for faster detection
- The pretrained YOLOv8 model weights are loaded and run through PyTorch

**Where used in code:**
```python
import torch
# PyTorch is used internally by YOLOv8
model = YOLO("yolov8n.pt")  # loads PyTorch model
```

---

### 🎯 YOLOv8 (Ultralytics)
**What it is:**
YOLOv8 (You Only Look Once version 8) is the latest and most advanced
real-time object detection model by Ultralytics. It detects objects in
images and video in a single forward pass of the neural network.

**Why used:**
- Detects **persons** (workers) in video frames with high accuracy
- Runs in **real-time** at 30-160 FPS depending on hardware
- Pretrained on **COCO dataset** which includes 'person' class (ID 0)
- No custom training needed — works immediately after download
- Simple Python API: `model(frame, classes=[0])` to detect only persons

**Where used in code:**
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")           # load model (auto-downloads)
results = model(frame, conf=0.40, classes=[0])  # detect persons only
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])     # get coordinates
```

**YOLOv8 Models Available:**

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolov8n.pt | 6 MB | Fastest | Good |
| yolov8s.pt | 22 MB | Fast | Better |
| yolov8m.pt | 50 MB | Medium | Best for this project |

---

### 🔍 HOG Person Detector (OpenCV Built-in)
**What it is:**
HOG stands for **Histogram of Oriented Gradients**. It is a classical
computer vision algorithm built into OpenCV that detects persons without
needing deep learning or internet.

**Why used:**
- Used as a **fallback** when YOLOv8 or internet is not available
- Works 100% offline with no extra downloads
- Already included in OpenCV installation
- Good enough for demo and testing purposes

**Where used in code:**
```python
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
boxes, weights = hog.detectMultiScale(frame, winStride=(4,4))
```

---

### 📥 yt-dlp
**What it is:**
yt-dlp is an open-source command-line tool for downloading videos from
YouTube and other websites.

**Why used:**
- Downloads the industrial safety videos from YouTube to create the dataset
- Supports downloading in specific quality (480p MP4)
- Used by `extract_frames.py` to automatically get the dataset videos

**Where used in code:**
```python
import subprocess
subprocess.run(["yt-dlp", "--format", "mp4", url, "--output", output_path])
```

---

### 💻 VS Code (Visual Studio Code)
**What it is:**
VS Code is a free, powerful code editor by Microsoft.

**Why used:**
- Writing and editing all Python files
- Running the Python scripts directly from terminal
- Debugging and fixing errors with built-in debugger
- Syntax highlighting and auto-complete for Python

---

### 📓 Jupyter Notebook
**What it is:**
Jupyter Notebook is an interactive Python environment that runs in the browser.

**Why used:**
- Exploring and visualizing dataset frames
- Running code step-by-step for testing
- Creating visual analysis reports
- Good for presenting project results to teachers/examiners

---

## 📁 Project Folder Structure

```
D:\project\ML-Accident Prevention\
│
├── src\                          ← All Python source code files
│   ├── live_camera_monitor.py    ← MAIN FILE — Live camera + video detection
│   ├── safety_monitor.py         ← Core safety monitoring system
│   ├── extract_frames.py         ← Download YouTube videos & extract frames
│   ├── configure_zone.py         ← Interactive danger zone configurator
│   └── analyze_alerts.py         ← Alert log analyzer & chart generator
│
├── dataset\                      ← Dataset folder
│   ├── raw_videos\               ← Downloaded MP4 videos
│   ├── frames_video1\            ← Extracted frames from Video 1
│   └── frames_video2\            ← Extracted frames from Video 2
│
├── live_outputs\                 ← Auto-created when you run the system
│   ├── screenshots\              ← Screenshots saved with S key
│   ├── alert_snapshots\          ← Auto-saved when alert triggers
│   └── alert_log.txt             ← All alerts with timestamps
│
├── outputs\                      ← Processed video outputs
│   ├── processed_videos\         ← Annotated output videos
│   ├── alert_log.csv             ← CSV log of all alerts
│   └── safety_analysis_report.png ← Charts and analysis
│
├── requirements.txt              ← All Python libraries to install
└── README.md                     ← This file
```

---

## ⚙️ Installation

### Step 1 — Install Python
Download Python 3.10 or higher from: https://www.python.org/downloads/
Make sure to check **"Add Python to PATH"** during installation.

### Step 2 — Install Libraries
Open Command Prompt and run:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install opencv-python numpy pandas matplotlib torch torchvision ultralytics yt-dlp
```

### Step 3 — Verify Installation
```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "from ultralytics import YOLO; print('YOLOv8 OK')"
```

---

## ▶️ How to Run

### Open Command Prompt
```bash
cd "D:\project\ML-Accident Prevention"
```

### Run the Main File
```bash
python src/live_camera_monitor.py
```

### Choose Your Mode
```
  ┌─────────────────────────────────────────────────┐
  │  1  →  LIVE CAMERA                              │
  │         Point your webcam at another screen     │
  │         playing the industrial video            │
  ├─────────────────────────────────────────────────┤
  │  2  →  VIDEO FILE                               │
  │         Provide path to any .mp4 video          │
  └─────────────────────────────────────────────────┘
  Enter 1 or 2 :
```

### Keyboard Controls
| Key | Action |
|-----|--------|
| `Z` | Draw danger zone — click and drag on machine area |
| `S` | Save screenshot |
| `SPACE` | Pause / Resume |
| `R` | Reset danger zone to default |
| `Q` | Quit |

---

## 📂 How Each File Works

### `live_camera_monitor.py` — MAIN FILE
The most important file. Contains two modes:
- **Mode 1**: Opens your webcam → you point it at another screen → AI detects persons
- **Mode 2**: You provide a video file path → AI processes it and detects persons

Key functions:
- `crop_black_borders()` — removes black borders from video automatically
- `detect()` — runs HOG person detection on each frame
- `draw()` — draws bounding boxes, danger zone, HUD, and alert banner
- `fire_alert()` — triggers all 5 alert actions
- `run_monitor()` — main loop shared by both modes

---

### `safety_monitor.py` — Core Detection System
The original safety monitoring script using YOLOv8.
- Loads YOLOv8 model
- Runs real-time person detection
- Checks danger zone intrusions
- Manages alert cooldown and logging
- Run with: `python src/safety_monitor.py --source 0`

---

### `extract_frames.py` — Dataset Creator
Downloads industrial videos from YouTube and extracts frames.
- Uses yt-dlp to download videos
- Uses OpenCV to extract 1 frame every 15 frames
- Saves frames as JPEG images in dataset folder
- Creates annotation template CSV
- Run with: `python src/extract_frames.py`

---

### `configure_zone.py` — Zone Configurator
Interactive tool to visually define the danger zone.
- Opens a camera or video frame
- You click and drag to draw a rectangle
- Saves the coordinates to JSON config file
- Run with: `python src/configure_zone.py --source 0`

---

### `analyze_alerts.py` — Alert Analyzer
Reads the alert log CSV and generates charts.
- Timeline chart of alerts over time
- Hourly distribution bar chart
- Summary statistics
- Run with: `python src/analyze_alerts.py`

---

## 🚨 Alert System

When a worker enters the danger zone, **5 alerts fire simultaneously**:

| # | Alert Type | Description |
|---|-----------|-------------|
| 1 | 🖥️ Visual Warning | Red banner `⚠ DANGER ZONE ENTERED` on screen |
| 2 | 📡 Operator Alert | Console message sent to CCTV monitoring operator |
| 3 | 🔔 Alarm | Audio beep / terminal bell |
| 4 | 📋 Log Entry | Timestamped entry saved to `alert_log.txt` |
| 5 | 🛑 Machine Stop | Simulated stop command printed to console |

---

## 📹 Dataset

| Source | URL | Content | Frames |
|--------|-----|---------|--------|
| Video 1 | https://youtu.be/bVHod9L73Pc | Stone crusher / press machine with workers | ~300 |
| Video 2 | https://youtube.com/shorts/hXrkYIzjDg4 | Conveyor belt with workers | ~300 |

**How dataset was created:**
1. Videos downloaded using `yt-dlp`
2. Frames extracted every 15 frames using OpenCV
3. Each frame resized to 720×480 pixels and saved as JPEG
4. YOLOv8 pretrained on COCO is used — no manual labeling needed

---

## 🔧 Bug Fixes Applied

Three important fixes were made after testing on the real industrial video:

### Fix 1 — Black Border Auto-Crop
**Problem:** The uploaded video had large black borders around the actual content.
The danger zone was being drawn over the black empty area instead of the machine.

**Solution:** Added `crop_black_borders()` function that automatically detects the
actual content area using thresholding and crops to it before processing.

```python
def crop_black_borders(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    return frame[y:y+h, x:x+w]
```

---

### Fix 2 — Lower Detection Sensitivity
**Problem:** Workers standing close to the press machine are partially hidden
(occluded) by the machine body. With threshold 0.25, these partial persons
were being missed completely.

**Solution:** Lowered sensitivity from `0.25` → `0.15`

```python
SENSITIVITY = 0.15   # was 0.25 — lowered to catch partially occluded workers
```

---

### Fix 3 — Finer Detection Stride
**Problem:** With `winStride=(8,8)` and detection every 3rd frame, fast-moving
workers near machines were being missed between detection cycles.

**Solution:** Changed to `winStride=(4,4)` and detect every **2nd frame** instead of 3rd.

```python
boxes, weights = hog.detectMultiScale(
    small,
    winStride=(4, 4),   # was (8,8)
    scale=1.03           # was 1.05
)
# detect every 2nd frame instead of every 3rd
if frame_num % 2 == 0:
    detections = detect(frame)
```

---

## 🔮 Future Improvements

| Feature | Technology | Description |
|---------|-----------|-------------|
| PPE Detection | Custom YOLOv8 | Detect helmet, safety vest, gloves |
| Mobile Alerts | Twilio SMS API | Send SMS to supervisor's phone |
| IoT Sensors | MQTT + Raspberry Pi | Fuse camera data with sensors |
| Real Machine Stop | Modbus TCP / PLC | Actually cut machine power |
| Multi-Camera | Multi-threading | Monitor multiple zones at once |
| Web Dashboard | Flask + HTML | View live feeds from browser |

---

## 🎓 Viva Tips

**Q: What does YOLO stand for?**
A: You Only Look Once — it detects objects in a single pass through the neural network.

**Q: Why YOLOv8 over older versions?**
A: Anchor-free detection, better accuracy (37.3 mAP vs 28.0), faster speed, simple API.

**Q: What is HOG?**
A: Histogram of Oriented Gradients — a classical CV algorithm that detects person
shapes by analyzing edge directions in image regions.

**Q: What is the COCO dataset?**
A: Microsoft Common Objects in Context — 330K images, 80 classes including 'person'.
YOLOv8 pretrained on this is used directly for worker detection.

**Q: How does the danger zone work?**
A: A rectangle (x1,y1,x2,y2) is defined around the machine. The center point of
each detected person's bounding box is checked against this rectangle.

**Q: What is the confidence threshold?**
A: Minimum probability score (0.15) for a detection to be accepted. Lower = more
sensitive, catches partial persons but may have some false positives.

---

## 📄 License

MIT License — Free for educational and research purposes.

---

## 👨‍💻 Author

B.Tech Final Year Project
Department of Computer Science & Engineering
Academic Year: 2024-2025
"# Accident-Prevention" 
