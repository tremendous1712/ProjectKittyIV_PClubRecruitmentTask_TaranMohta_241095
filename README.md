# Badminton Shot Analysis Pipeline

This repository contains a pipeline to analyze badminton shot videos, extract pose keypoints, compute shot-wise features, predict shot type, and generate feedback based on predefined rules.

## Folder Structure

```
ProjectKittyIV_PClubRecruitmentTask_TaranMohta_241095/
│
├── data/                      # Sample input files (videos, CSVs)
│
├── models/                    # Pretrained model .pkl files
│
├── notebooks/                 # Reference notebooks, experiments
│
├── output/                    # Generated outputs
│   ├── frames/                # Extracted raw frames from videos
│   ├── cropped_frames/        # Cropped frames focusing on court/players
│   ├── keypoints_and_features/# JSON or CSV with keypoints and computed features
│   ├── all_frames_keypoints.csv
│   ├── H_test.json            # Homography matrix JSON file
│   ├── crop_rect_coordinates.csv
│   ├── shotwise_features.csv
│   └── shotwise_feedback.csv
│
├── src/                       # Source code modules
│   ├── config.py              # Configuration settings and constants
│   ├── utils/                 # Utility scripts for modular functionality
│   │   ├── video_utils.py     # Video reading, frame extraction
│   │   ├── court_detection.py # Court detection and cropping logic
│   │   ├── homography.py      # Homography matrix calculation and transformation
│   │   ├── csv_utils.py       # CSV reading/writing helpers
│   │   └── feature_utils.py   # Feature computation (angles, pose-based metrics)
│   │
│   ├── pipeline/              # Stepwise pipeline scripts
│   │   ├── step1_extract_crop.py       # Extract frames and crop court area
│   │   ├── step2_keypoints.py           # Pose keypoint extraction (MediaPipe or similar)
│   │   ├── step3_homography_transform.py # Apply homography transforms to keypoints
│   │   ├── step4_shotwise_features.py    # Aggregate shotwise features
│   │   └── step5_prediction_feedback.py  # Model inference & feedback generation
│   │
│   └── main.py                # Orchestrates the entire pipeline end-to-end
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview, setup, usage instructions

```

## Installation

```bash
git clone https://github.com/tremendous1712/ProjectKittyIV_PClubRecruitmentTask_TaranMohta_241095.git
cd ProjectKittyIV_PClubRecruitmentTask_TaranMohta_241095
python3 -m venv venv
source venv/bin/activate     # For Linux/macOS
# OR for Windows PowerShell:
# python -m venv venv
# .\venv\Scripts\Activate.ps1
# OR for Windows Command Prompt:
# python -m venv venv
# venv\Scripts\activate.bat

pip install --upgrade pip
pip install -r requirements.txt
```

## To Run

```bash
python src/main.py
```
