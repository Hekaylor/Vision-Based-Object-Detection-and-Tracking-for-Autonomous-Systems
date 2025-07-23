# Vision-Based Object Detection and Tracking for Autonomous Systems

This repository contains a vision-based perception pipeline for object detection and tracking in video feeds. The system uses YOLOv8 for real-time object detection and DeepSORT for multi-object tracking. It is designed to support autonomous decision-making systems by providing structured information about dynamic objects in the environment.

## Contact Information

For any comments, questions, or concerns, please contact Hailey Kaylor at HKaylor@Mines.edu.

---

## Repository Structure

```
/Vision-Based-Object-Detection-and-Tracking-for-Autonomous-Systems
├── Code/                  # Python scripts for detection, tracking, and video processing
├── Results/               # Output videos, plots, and tracking logs
└── References/            # Technical notes, model details, and research documentation
```

---

## Overview

The project enables object-level situational awareness from video streams using a modular pipeline consisting of:

- **YOLOv8** (You Only Look Once) for real-time object detection
- **DeepSORT** for identity-aware object tracking
- **OpenCV** for video input/output and frame processing
- Output plots and logs for performance evaluation and system analysis

It is designed to support applications in autonomous navigation, robotics, and intelligent surveillance.

---

## Getting Started

### Requirements

- Python 3.8+
- Recommended packages:
  - `ultralytics` (for YOLOv8)
  - `opencv-python`
  - `deep_sort_realtime`
  - `matplotlib`
  - `pandas`

Install dependencies with:

```bash
pip install ultralytics opencv-python deep_sort_realtime matplotlib pandas
```

---

### Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/Hekaylor/Vision-Based-Object-Detection-and-Tracking-for-Autonomous-Systems.git
   cd Vision-Based-Object-Detection-and-Tracking-for-Autonomous-Systems
   ```

2. Prepare your input video or camera stream.

3. Run the main detection and tracking pipeline from the `Code/` directory:

   ```bash
   python Code/detect_and_track.py
   ```

   *(Make sure to update the video path and model path in the script as needed)*

4. Check `Results/` for output video clips, object tracking logs, and performance plots.

---

## Highlights

- **Real-time object detection** using YOLOv8 pretrained models  
- **Multi-object identity tracking** using DeepSORT  
- **Bounding box overlay and object trail visualization**  
- **Distance and decision metrics** for downstream analysis  
- **Exported tracking logs and filtered plots** for system diagnostics  

---

## Folder Details

### `Code/`
Contains:
- `cv.py`: Main pipeline combining YOLOv8 + DeepSORT
- Video capture and processing logic
- Frame annotation, tracking, and data logging
- Optional plotting scripts for performance metrics

### `Results/`
Contains:
- Processed videos with annotated bounding boxes and object IDs
- CSV logs of object positions and classifications
- Plots of distance vs. frame/time and decision triggers

### `References/`
Contains:
- Notes on YOLO model selection and tuning
- DeepSORT configuration and parameter explanations
- Documentation used for implementation and evaluation
