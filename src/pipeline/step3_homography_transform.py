import pandas as pd
import numpy as np
from tqdm import tqdm
from config import INTERMEDIATE_CSV, H_TEST_JSON, CROP_RECT_CSV
from utils.homography import load_homography, apply_homography
from utils.video_utils import get_crop_size

def apply_homography_on_csv(
    intermediate_csv=INTERMEDIATE_CSV,
    h_test_json=H_TEST_JSON
):
    """
    1) Load H_test from JSON
    2) H_train fixed → compute inv
    3) For each row in intermediate_csv, apply homography using get_crop_size() for w,h
    4) Overwrite intermediate_csv
    """
    H_test = load_homography(h_test_json)
    H_train = np.array([
        [1.0606627138451856, 0.3439667138673875, -497.8978610611467],
        [0.011919581725440269, 4.586662980083068, -1176.8442597836183],
        [4.8478780487874024e-06, 0.002012572611469505, 1.0]
    ])
    H_TRAIN_INV = np.linalg.inv(H_train)

    df = pd.read_csv(intermediate_csv)
    w, h = get_crop_size()   # EXACTLY your original call

    for i in tqdm(range(len(df)), desc="Applying homography"):
        xs = df.loc[i, [f'kp_{k}_x' for k in range(33)]].values * w
        ys = df.loc[i, [f'kp_{k}_y' for k in range(33)]].values * h
        x1, y1 = apply_homography(xs, ys, H_test)
        x2, y2 = apply_homography(x1, y1, H_TRAIN_INV)
        df.loc[i, [f'kp_{k}_x' for k in range(33)]] = x2 / 700
        df.loc[i, [f'kp_{k}_y' for k in range(33)]] = y2 / 310

    df.to_csv(intermediate_csv, index=False)
    print("✅ Updated CSV with homography-transformed keypoints")
