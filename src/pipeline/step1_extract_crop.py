import os
from config import (
    VIDEO_PATH,
    CSV_WITH_SHOTS,
    OUTPUT_FRAME_DIR,
    OUTPUT_CROPPED_DIR,
    H_TEST_JSON,
    CROP_RECT_CSV,
    FRAME_RANGE,
    OUTPUT_DIR
)
from utils.video_utils import save_first_frame_from_csv, save_and_crop_frames
from utils.homography import compute_and_save_homography, compute_and_save_crop_rect
from utils.csv_utils import read_shots_csv

def step1_extract_and_crop(
    video_path=VIDEO_PATH,
    shots_csv=CSV_WITH_SHOTS,
    output_frame_dir=OUTPUT_FRAME_DIR,
    output_crop_dir=OUTPUT_CROPPED_DIR
):
    """
    1) save_first_frame_from_csv → OUTPUT_DIR/first_frame.jpg
    2) compute_and_save_homography → H_TEST_JSON
    3) compute_and_save_crop_rect → CROP_RECT_CSV
    4) read_shots_csv → df_shots
    5) save_and_crop_frames(df_shots)
    """
    # 1) Save first frame
    first_frame_path = save_first_frame_from_csv(video_path, shots_csv, OUTPUT_DIR)

    # 2) Compute homography
    corners = compute_and_save_homography(first_frame_path, H_TEST_JSON)

    # 3) Compute crop rectangle
    compute_and_save_crop_rect(corners, CROP_RECT_CSV)

    # 4) Read shot CSV
    df_shots = read_shots_csv(shots_csv)

    # 5) Extract & crop frames
    save_and_crop_frames(df_shots)
