# Badminton Shot Analysis Pipeline

This repository contains a pipeline to analyze badminton shot videos, extract pose keypoints, compute shot-wise features, predict shot type, and generate feedback based on predefined rules.

## Installation

```bash
git clone https://github.com/yourusername/badminton_pipeline.git
cd badminton_pipeline
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Folder Structure

```
badminton_pipeline/
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

## To Run

```bash
python src/main.py
```

## Data & Models

- **data/**: Place your `example_video.mp4`, `shot_frames.csv`, and feedback rules CSV here.
- **models/**: Place pretrained `xgb_model_label_0.pkl` … `xgb_model_label_3.pkl` here.
