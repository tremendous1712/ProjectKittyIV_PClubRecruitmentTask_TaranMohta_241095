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
├── data/                      # Sample input files (video, CSVs)
│
├── models/                    # Pretrained model .pkl files
│
├── notebooks/                 # Reference notebooks
│
├── output/                    # Generated outputs
│   ├── frames/
│   ├── cropped_frames/
│   ├── keypoints_and_features/
│   ├── all_frames_keypoints.csv
│   ├── H_test.json
│   ├── crop_rect_coordinates.csv
│   ├── shotwise_features.csv
│   └── shotwise_feedback.csv
│
├── src/                       # Source code
│   ├── config.py
│   ├── utils/
│   │   ├── video_utils.py
│   │   ├── court_detection.py
│   │   ├── homography.py
│   │   ├── csv_utils.py
│   │   └── feature_utils.py
│   │
│   ├── pipeline/
│   │   ├── step1_extract_crop.py
│   │   ├── step2_keypoints.py
│   │   ├── step3_homography_transform.py
│   │   ├── step4_shotwise_features.py
│   │   └── step5_prediction_feedback.py
│   │
│   ├── top_features_nearby.py
│   └── main.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

```bash
python src/main.py   --video data/example_video.mp4   --shots_csv data/shot_frames.csv   --models_dir models   --rules_csv data/badminton_pose_feedback_filled_f.csv   --output_dir output
```

## Data & Models

- **data/**: Place your `example_video.mp4`, `shot_frames.csv`, and feedback rules CSV here.
- **models/**: Place pretrained `xgb_model_label_0.pkl` … `xgb_model_label_3.pkl` here.
