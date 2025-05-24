import os
import argparse
import shutil
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import cv2

from config import (
    VIDEO_PATH,
    CSV_WITH_SHOTS,
    OUTPUT_DIR,
    OUTPUT_FRAME_DIR,
    OUTPUT_CROPPED_DIR,
    OUTPUT_KEYPOINTS_DIR,
    MODELS_DIR,
    RULES_CSV_PATH,
    FINAL_SHOT_FEATURES_CSV,
    FINAL_FEEDBACK_CSV
)

from pipeline.step1_extract_crop import step1_extract_and_crop
from pipeline.step2_keypoints import extract_and_save_keypoints
from pipeline.step3_homography_transform import apply_homography_on_csv
from pipeline.step4_shotwise_features import build_shotwise_features
from pipeline.step5_prediction_feedback import step5_predict_and_feedback

from utils.video_utils import capture_shot_frames

def parse_args():
    parser = argparse.ArgumentParser(description="Badminton Shot Analysis Pipeline")
    parser.add_argument(
        "--video", default=VIDEO_PATH,
        help="Path to input video"
    )
    parser.add_argument(
        "--shots_csv", default=CSV_WITH_SHOTS,
        help="CSV with 'frame_num' and 'shot_no' columns"
    )
    parser.add_argument(
        "--models_dir", default=MODELS_DIR,
        help="Directory containing XGBoost model .pkl files"
    )
    parser.add_argument(
        "--rules_csv", default=RULES_CSV_PATH,
        help="CSV containing feedback rules & thresholds"
    )
    parser.add_argument(
        "--output_dir", default=OUTPUT_DIR,
        help="Base directory for all outputs"
    )
    return parser.parse_args()

def clear_output_subfolders(output_dir):
    """
    Delete all contents inside each subfolder of output_dir,
    keeping the subfolders themselves.
    """
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist. Nothing to clear.")
        return

    for foldername in os.listdir(output_dir):
        folderpath = os.path.join(output_dir, foldername)
        if os.path.isdir(folderpath):
            for item in os.listdir(folderpath):
                item_path = os.path.join(folderpath, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

def show_shot_carousel(feedback_csv_path, frames_dir):
    """
    Display one impact frame per shot. Each image has a semi-transparent
    overlay box from (40,20) to (800,350). Inside that box, we:
      - Show "Shot: <type>" at (50,40)
      - Show each line of feedback (quotes removed, hyphens→"and"),
        prefixed with a bullet • and wrapped within x=50..780,
        starting at y=80 and stopping before y=350.
    Press any key to advance; press 'q' to quit.
    """
    df = pd.read_csv(feedback_csv_path)
    print("\n📺 Showing impact frames with shot type and feedback.")
    print("    Press any key for next shot; press 'q' to quit.")

    for _, row in df.iterrows():
        shot_no = int(row['shot_no'])
        shot_type = row.get('predicted_shot_type', 'Unknown')
        feedback_raw = row.get('feedback', '')

        # Build path: OUTPUT_FRAME_DIR/shot_<shot_no:05d>/impact.jpg
        shot_folder = f"shot_{shot_no:05d}"
        frame_path = os.path.join(frames_dir, shot_folder, "impact.jpg")

        if not os.path.exists(frame_path):
            print(f"⚠️ Impact frame not found for shot {shot_no} at {frame_path}")
            continue

        img = cv2.imread(frame_path)
        if img is None:
            print(f"⚠️ Could not read image at {frame_path}")
            continue

        h, w = img.shape[:2]  # typically 720×1280

        # 1) Draw a semi‐transparent black rectangle from (40,20) to (800,350)
        overlay = img.copy()
        top_left = (40, 20)
        bottom_right = (800, 350)
        cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), thickness=-1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # 2) Clean and split feedback into lines, prefix each with a bullet:
        #    - Remove all double quotes: " , “ , ”
        #    - Replace hyphens (‐, –, —, -) with "and"
        raw_lines = feedback_raw.split("\n")
        cleaned_lines = []
        for ln in raw_lines:
            cl = ln.replace('"', '').replace('“', '').replace('”', '')
            cl = cl.replace('—', ' and ').replace('–', ' and ').replace('-', ' and ')
            cl = cl.strip()
            if cl:
                cleaned_lines.append(" - " + cl)

        # 3) Overlay "Shot: <shot_type>" at (50,40)
        cv2.putText(
            img,
            f"Shot: {shot_type}",
            (50, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # 4) Prepare to wrap text within [50,780] horizontally,
        #    starting at y=80, with line_height based on font metrics
        x_start = 50
        x_end = 780
        y_start = 80
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1

        # Compute line height
        (tx, ty), baseline = cv2.getTextSize("Ay", font, font_scale, thickness)
        line_height = ty + baseline + 5

        # Helper to wrap a single string into multiple lines
        def wrap_text(text, max_width):
            words = text.split(" ")
            wrapped = []
            current_line = ""
            for word in words:
                tentative = current_line + (" " if current_line else "") + word
                (w_text, _), _ = cv2.getTextSize(tentative, font, font_scale, thickness)
                if w_text <= max_width:
                    current_line = tentative
                else:
                    if current_line:
                        wrapped.append(current_line)
                    current_line = word
            if current_line:
                wrapped.append(current_line)
            return wrapped

        # Apply wrapping to each cleaned line
        wrapped_all = []
        max_width = x_end - x_start
        for cl in cleaned_lines:
            wrapped_all.extend(wrap_text(cl, max_width))

        # 5) Overlay wrapped lines until y exceeds 350
        y = y_start + ty
        for line in wrapped_all:
            if y > 350 - 10:  # leave ~10px bottom margin
                break
            cv2.putText(
                img,
                line,
                (x_start, y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )
            y += line_height

        # 6) Show the annotated frame
        cv2.imshow("Shot Impact Frame", img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()




if __name__ == "__main__":
    args = parse_args()

    # 🔹 Prompt user to upload video
    print("📁 Please select a video file to analyze...")
    root = tk.Tk()
    root.withdraw()
    selected_video = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
    )
    if not selected_video:
        print("❌ No video selected. Exiting.")
        exit()

    # 🔹 Copy selected video to VIDEO_PATH (configured in config.py)
    shutil.copy(selected_video, VIDEO_PATH)
    print(f"✅ Video copied to: {VIDEO_PATH}")
    args.video = VIDEO_PATH

    # 🔹 Clean output/ folders
    clear_output_subfolders(args.output_dir)

    # 🔹 Shot frame capture (OpenCV GUI)
    capture_shot_frames(video_path=args.video, output_csv_path=args.shots_csv)

    # 🔹 Ensure required output folders exist
    os.makedirs(args.output_dir,          exist_ok=True)
    os.makedirs(OUTPUT_FRAME_DIR,         exist_ok=True)
    os.makedirs(OUTPUT_CROPPED_DIR,       exist_ok=True)
    os.makedirs(OUTPUT_KEYPOINTS_DIR,     exist_ok=True)

    🔹 Step 1: Extract & Crop frames
    step1_extract_and_crop(
        video_path=args.video,
        shots_csv=args.shots_csv,
        output_frame_dir=OUTPUT_FRAME_DIR,
        output_crop_dir=OUTPUT_CROPPED_DIR
    )

    # 🔹 Step 2: Extract keypoints
    extract_and_save_keypoints()

    # 🔹 Step 3: Apply homography
    apply_homography_on_csv()

    # 🔹 Step 4: Build shotwise features
    build_shotwise_features()

    # 🔹 Step 5: Predict + Feedback
    step5_predict_and_feedback(
        models_dir=args.models_dir,
        shotwise_csv=FINAL_SHOT_FEATURES_CSV,
        rules_csv=args.rules_csv,
        out_csv=FINAL_FEEDBACK_CSV
    )

    # 🔹 Show one impact frame per shot, using shot_no to find 'impact.jpg'
    show_shot_carousel(FINAL_FEEDBACK_CSV, OUTPUT_FRAME_DIR)

    print("✅ Pipeline complete. Check the 'output/' folder for results.")
