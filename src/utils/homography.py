import os
import json
import numpy as np
import cv2
import pandas as pd
from .court_detection import get_court_corners

def compute_and_save_homography(img_path, json_path):
    """
    EXACTLY your original:
    1) get_court_corners → 4 points
    2) findHomography to real coordinates
    3) save JSON
    """
    corners = np.array(get_court_corners(img_path), dtype=np.float32)
    real = np.array([[25,150], [325,150], [325,810], [25,810]], dtype=np.float32)
    H, _ = cv2.findHomography(corners, real)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(H.tolist(), f, indent=2)
    return corners

def compute_and_save_crop_rect(corners, csv_path):
    """
    EXACTLY your original:
    1) bottom-left → corners[3]
    2) mid of TR & BR → (tr[0]+br[0])/2, (tr[1]+br[1])/2
    3) save to CSV with columns x1,y1,x2,y2
    """
    x1, y1 = corners[3]
    tr, br = corners[1], corners[2]
    x2 = int((tr[0] + br[0]) / 2)
    y2 = int((tr[1] + br[1]) / 2)
    df = pd.DataFrame([{'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    return x1, y1, x2, y2

def load_homography(path):
    with open(path, 'r') as f:
        return np.array(json.load(f))

def apply_homography(xs, ys, H):
    pts = np.vstack([xs, ys, np.ones_like(xs)])
    p2 = H @ pts
    w = np.where(p2[2] == 0, 1e-6, p2[2])
    return p2[0] / w, p2[1] / w
