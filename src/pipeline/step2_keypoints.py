import os
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
from config import OUTPUT_CROPPED_DIR, INTERMEDIATE_CSV
from utils.csv_utils import save_csv

def extract_and_save_keypoints():
    """
    EXACTLY your original logic:
    For each shot folder under OUTPUT_CROPPED_DIR/shot_*, read 61 frames + 'impact.jpg',
    run MediaPipe Pose, collect all kp_{k}_x/y plus elbow_angle, torso_lean_angle, wrist_above_head at idx==30,
    then compute time-based features for each shot and save all to INTERMEDIATE_CSV.
    """
    rows = []
    shot_dirs = sorted(glob.glob(os.path.join(OUTPUT_CROPPED_DIR, "shot_*")))

    mp_pose = mp.solutions.pose

    for shot_folder in tqdm(shot_dirs, desc="Extract keypoints"):
        shot_no = os.path.basename(shot_folder).split('_')[1]
        frames = [None] * 61
        for i in range(61):
            p = os.path.join(shot_folder, f"frame_{i:03d}.jpg")
            if os.path.exists(p):
                frames[i] = cv2.imread(p)
        impact_p = os.path.join(shot_folder, 'impact.jpg')
        if os.path.exists(impact_p):
            frames[30] = cv2.imread(impact_p)

        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.3
        )
        for idx, frame in enumerate(frames):
            row = {'shot_no': shot_no, 'frame_idx': idx}
            if frame is not None:
                res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if res.pose_landmarks:
                    for k, lm in enumerate(res.pose_landmarks.landmark):
                        row[f'kp_{k}_x'] = lm.x
                        row[f'kp_{k}_y'] = lm.y
                    if idx == 30:
                        lmks = res.pose_landmarks.landmark
                        def pt(k): return np.array([lmks[k].x, lmks[k].y])
                        a, b, c = pt(12), pt(14), pt(16)
                        ba, bc = a - b, c - b
                        elbow_angle = np.degrees(
                            np.arccos(
                                np.clip(
                                    np.dot(ba, bc) /
                                    (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6),
                                    -1.0, 1.0
                                )
                            )
                        )
                        shoulder_mid = (pt(11) + pt(12)) / 2
                        hip_mid = (pt(23) + pt(24)) / 2
                        vec = shoulder_mid - hip_mid
                        torso_lean_angle = np.degrees(np.arctan2(vec[0], vec[1]))
                        wrist_above_head = int(pt(16)[1] < shoulder_mid[1])
                        row['elbow_angle'] = elbow_angle
                        row['torso_lean_angle'] = torso_lean_angle
                        row['wrist_above_head'] = wrist_above_head
            for f in [
                'elbow_angle',
                'torso_lean_angle',
                'wrist_above_head',
                'max_wrist_velocity',
                'wrist_velocity_increase',
                'shoulder_rotation_change'
            ]:
                if f not in row:
                    row[f] = None
            rows.append(row)
        pose.close()

    df_all = pd.DataFrame(rows)

    # Compute time-based features exactly as before
    for shot_no, group in df_all.groupby('shot_no'):
        if len(group) != 61:
            continue
        group = group.sort_values('frame_idx')
        xs = group[f'kp_16_x'].values
        ys = group[f'kp_16_y'].values
        vels = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        max_vel = np.max(vels)
        vel_inc = vels[30] - np.min(vels[:30])
        left_sh = np.stack([group[f'kp_11_x'].values, group[f'kp_11_y'].values], axis=1)
        right_sh = np.stack([group[f'kp_12_x'].values, group[f'kp_12_y'].values], axis=1)
        shoulder_vec = right_sh - left_sh
        angles = np.arctan2(shoulder_vec[:, 1], shoulder_vec[:, 0])
        angle_change = np.degrees(np.max(angles) - np.min(angles))
        idx_mid = group[group['frame_idx'] == 30].index[0]
        df_all.at[idx_mid, 'max_wrist_velocity'] = max_vel
        df_all.at[idx_mid, 'wrist_velocity_increase'] = vel_inc
        df_all.at[idx_mid, 'shoulder_rotation_change'] = angle_change

    save_csv(df_all, INTERMEDIATE_CSV, index=False)
    print(f"✅ Saved intermediate keypoints CSV: {INTERMEDIATE_CSV}")
