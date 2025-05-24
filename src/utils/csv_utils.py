import pandas as pd
import os

def read_shots_csv(csv_path):
    """
    Read CSV with columns frame_num, shot_no; drop NaNs; cast to int.
    """
    df = pd.read_csv(csv_path).dropna(subset=["frame_num"])
    df["frame_num"] = df["frame_num"].astype(int)
    df["shot_no"]   = df["shot_no"].astype(int)
    return df

def save_csv(df, output_path, index=False):
    """
    Ensure directory exists, then write df.to_csv.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=index)
