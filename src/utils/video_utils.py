import cv2
import os
import pandas as pd
from tqdm import tqdm
from config import VIDEO_PATH, CSV_WITH_SHOTS, OUTPUT_FRAME_DIR, OUTPUT_CROPPED_DIR, FRAME_RANGE, CROP_RECT_CSV
import csv

def capture_shot_frames(video_path, output_csv_path="data/shot_frames.csv"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    print("Press 's' to save current frame as shot, 'q' to quit.")
    print("Current frame number will be shown on screen.")

    shot_frames = []
    shot_no = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        display_frame = frame.copy()
        cv2.putText(display_frame, f"Frame: {frame_num}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Video - Press 's' to Save Frame, 'q' to Quit", display_frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('s'):
            print(f"Saved Shot {shot_no} at Frame {frame_num}")
            shot_frames.append([shot_no, frame_num])
            shot_no += 1

        elif key == ord('q'):
            print("Finished capturing shots.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save shot frames to CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["shot_no", "frame_num"])
        writer.writerows(shot_frames)

    print(f"Shot frames saved to {output_csv_path}")

def save_first_frame_from_csv(
    video_path=VIDEO_PATH,
    csv_path=CSV_WITH_SHOTS,
    output_dir=os.path.join(os.path.dirname(VIDEO_PATH), "..")  # fallback to OUTPUT_DIR
):
    """
    EXACTLY as in your original:
    Read shot_frames.csv, get first 'frame_num', open video, save that frame as first_frame.jpg.
    """
    df = pd.read_csv(csv_path)
    if "frame_num" not in df.columns:
        raise ValueError("CSV does not contain 'frame_num' column")

    first_frame_num = int(df.iloc[0]["frame_num"])
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame {first_frame_num} from video")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "first_frame.jpg")
    cv2.imwrite(output_path, frame)
    print(f"Saved frame {first_frame_num} as: {output_path}")
    return output_path

def save_and_crop_frames(df_shots):
    """
    EXACTLY as in your original blob:
    1) Read crop rectangle from CROP_RECT_CSV.
    2) Cast to int, swap if needed.
    3) For each shot, extract ±FRAME_RANGE frames, save raw & cropped.
    """
    # ─── Read crop-rectangle CSV ─────────────────────────────────────────────────
    crop_rect = pd.read_csv(CROP_RECT_CSV)
    x1 = crop_rect.iloc[0]['x1']
    y1 = crop_rect.iloc[0]['y1']
    x2 = crop_rect.iloc[0]['x2']
    y2 = crop_rect.iloc[0]['y2']
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    y1=y1-30
    y2=y2+30
    CROP_COORDS = (x1, y1, x2, y2)

    for _, row in tqdm(df_shots.iterrows(), total=len(df_shots), desc="Extract & crop frames"):
        impact   = int(row['frame_num'])
        shot_idx = int(row['shot_no'])
        start    = max(0, impact - FRAME_RANGE)
        end      = impact + FRAME_RANGE

        cap   = cv2.VideoCapture(VIDEO_PATH)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if impact >= total:
            cap.release()
            continue
        if end >= total:
            end = total - 1

        shot_frame_dir = os.path.join(OUTPUT_FRAME_DIR, f"shot_{shot_idx:05d}")
        shot_crop_dir  = os.path.join(OUTPUT_CROPPED_DIR, f"shot_{shot_idx:05d}")
        os.makedirs(shot_frame_dir, exist_ok=True)
        os.makedirs(shot_crop_dir, exist_ok=True)

        for i, fno in enumerate(range(start, end + 1)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ret, frame = cap.read()
            if not ret:
                continue
            filename  = 'impact.jpg' if fno == impact else f'frame_{i:03d}.jpg'
            raw_path  = os.path.join(shot_frame_dir, filename)
            crop_path = os.path.join(shot_crop_dir, filename)
            cv2.imwrite(raw_path, frame)

            # ─── EXACTLY your original crop slice ───────────────────────────────────
            crop = frame[CROP_COORDS[1]:CROP_COORDS[3], CROP_COORDS[0]:CROP_COORDS[2]]
            cv2.imwrite(crop_path, crop)
        cap.release()

def get_crop_size():
    """
    EXACTLY as in your original:
    Read CROP_RECT_CSV, cast to int, swap if needed, return (width, height).
    """
    crop_rect = pd.read_csv(CROP_RECT_CSV)
    x1 = crop_rect.iloc[0]['x1']
    y1 = crop_rect.iloc[0]['y1']
    x2 = crop_rect.iloc[0]['x2']
    y2 = crop_rect.iloc[0]['y2']
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return x2 - x1, y2 - y1
