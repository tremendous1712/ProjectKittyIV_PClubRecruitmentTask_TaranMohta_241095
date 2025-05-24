import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from config import INTERMEDIATE_CSV, FINAL_SHOT_FEATURES_CSV

def extract_features(shot_df):
    """
    EXACTLY your original: interpolate, compute mean/std/min/max, velocity features, pairwise distances, extra mid-frame features.
    """
    shot_df = shot_df.interpolate().ffill().bfill()
    n = 61
    data = shot_df[[f'kp_{i}_{axis}' for i in range(33) for axis in ['x','y']]].values.reshape(n,33,2)
    mean_xy = data.mean(axis=0).flatten()
    std_xy  = data.std(axis=0).flatten()
    min_xy  = data.min(axis=0).flatten()
    max_xy  = data.max(axis=0).flatten()
    vel = np.linalg.norm(np.diff(data,axis=0),axis=2)
    mean_vel = vel.mean(axis=0)
    std_vel  = vel.std(axis=0)
    dists = np.array([pdist(frame) for frame in data])
    mean_dist = dists.mean(axis=0)
    std_dist  = dists.std(axis=0)
    extra = shot_df[shot_df['frame_idx']==30][[
        'elbow_angle',
        'torso_lean_angle',
        'wrist_above_head',
        'max_wrist_velocity',
        'wrist_velocity_increase',
        'shoulder_rotation_change'
    ]].iloc[0].values
    return np.concatenate([extra, mean_xy, std_xy, min_xy, max_xy, mean_vel, std_vel, mean_dist, std_dist])

def build_shotwise_features(
    intermediate_csv=INTERMEDIATE_CSV,
    output_csv=FINAL_SHOT_FEATURES_CSV
):
    """
    EXACTLY your original:
    Read INTERMEDIATE_CSV, group by shot_no, extract_features, assemble columns, save to output_csv.
    """
    df = pd.read_csv(intermediate_csv)
    results = []
    for shot_no, grp in df.groupby('shot_no'):
        if len(grp) != 61:
            continue
        feat = extract_features(grp.sort_values('frame_idx'))
        results.append(np.concatenate([[shot_no], feat]))
    cols = ['shot_no','elbow_angle','torso_lean_angle','wrist_above_head','max_wrist_velocity','wrist_velocity_increase','shoulder_rotation_change']
    for stat in ['mean','std','min','max']:
        for i in range(33):
            cols += [f'kp_{i}_x_{stat}', f'kp_{i}_y_{stat}']
    for axis in ['mean_vel','std_vel']:
        cols += [f'kp_{i}_{axis}' for i in range(33)]
    num_pairs = (33*32)//2
    for stat in ['mean_dist','std_dist']:
        cols += [f'pair_{j}_{stat}' for j in range(num_pairs)]
    out_df = pd.DataFrame(results, columns=cols)
    out_df.to_csv(output_csv, index=False)
    print(f"✅ Shot-wise features saved to {output_csv}")
    return out_df
