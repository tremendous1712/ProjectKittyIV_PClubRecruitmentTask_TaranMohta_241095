import os

# ─── BASE & DATA PATHS ────────────────────────────────────────────────────────────
BASE_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
OUTPUT_DIR      = os.path.join(BASE_DIR, "output")

# ─── INPUT FILE DEFAULTS ─────────────────────────────────────────────────────────
VIDEO_PATH          = os.path.join(DATA_DIR, "your_video.mp4")
CSV_WITH_SHOTS      = os.path.join(DATA_DIR, "shot_frames.csv")
RULES_CSV_PATH      = os.path.join(DATA_DIR, "badminton_pose_feedback_filled_f.csv")

# ─── OUTPUT / INTERMEDIATE PATHS ─────────────────────────────────────────────────
OUTPUT_FRAME_DIR        = os.path.join(OUTPUT_DIR, "frames")
OUTPUT_CROPPED_DIR      = os.path.join(OUTPUT_DIR, "cropped_frames")
OUTPUT_KEYPOINTS_DIR    = os.path.join(OUTPUT_DIR, "keypoints_and_features")
INTERMEDIATE_CSV        = os.path.join(OUTPUT_DIR, "all_frames_keypoints.csv")
H_TEST_JSON             = os.path.join(OUTPUT_DIR, "H_test.json")
CROP_RECT_CSV           = os.path.join(OUTPUT_DIR, "crop_rect_coordinates.csv")
FINAL_SHOT_FEATURES_CSV = os.path.join(OUTPUT_DIR, "shotwise_features.csv")
FINAL_FEEDBACK_CSV      = os.path.join(OUTPUT_DIR, "shotwise_feedback.csv")

# ─── PIPELINE PARAMETERS ──────────────────────────────────────────────────────────
FPS           = 30
FRAME_RANGE   = 30  # ±30 frames around impact
