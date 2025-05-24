import os
import numpy as np
import pandas as pd
import joblib
import ast
from config import FINAL_SHOT_FEATURES_CSV, MODELS_DIR, RULES_CSV_PATH, FINAL_FEEDBACK_CSV

def load_model_and_features(models_dir=MODELS_DIR):
    """
    Load the multiclass XGB model and its training features list.
    Assumes you saved both model and features as a dict during training.
    """
    model_path = os.path.join(models_dir, "xgb_multiclass_model_all_features.pkl")
    data = joblib.load(model_path)
    model = data["model"] if isinstance(data, dict) else data
    features = data["features"] if isinstance(data, dict) else None

    if features is None:
        raise ValueError("Feature list not found in model file! Save both model and features during training.")
    return model, features


def predict_shot_types(shotwise_df, model, feature_cols):
    """
    Predict shot type using a single multiclass model and correct feature columns.
    """
    # Ensure feature columns exist in the DataFrame
    missing_cols = [f for f in feature_cols if f not in shotwise_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns for prediction: {missing_cols}")

    X = shotwise_df[feature_cols]
    probs = model.predict_proba(X)  # shape: [n_samples, 4]

    shotwise_df["predicted_shot_type_encoded"] = np.argmax(probs, axis=1)
    label_map = {0: 'clear', 1: 'drop', 2: 'net shot', 3: 'smash'}
    shotwise_df["predicted_shot_type"] = shotwise_df["predicted_shot_type_encoded"].map(label_map)

    return shotwise_df


def parse_thresholds(threshold_str):
    thresholds = {}
    for part in threshold_str.split(';'):
        if ':' in part:
            key, val = part.split(':')
            thresholds[key.strip()] = ast.literal_eval(val.strip())
    return thresholds


def step5_predict_and_feedback(
    models_dir=MODELS_DIR,
    shotwise_csv=FINAL_SHOT_FEATURES_CSV,
    rules_csv=RULES_CSV_PATH,
    out_csv=FINAL_FEEDBACK_CSV
):
    """
    1) Load shotwise_features CSV
    2) Load model + features, predict shot types
    3) Compute derived features
    4) Load rules, parse thresholds, apply feedback rules
    5) Save results
    """
    shotwise_df = pd.read_csv(shotwise_csv)
    
    model, feature_cols = load_model_and_features(models_dir)
    shotwise_df = predict_shot_types(shotwise_df, model, feature_cols)

    # Derived features
    if {'kp_12_x_max', 'kp_11_x_max'}.issubset(shotwise_df.columns):
        shotwise_df['kp_12_x_max - kp_11_x_max'] = (
            shotwise_df['kp_12_x_max'] - shotwise_df['kp_11_x_max']
        )
    if all(col in shotwise_df.columns for col in ['kp_23_x_mean','kp_23_y_mean','kp_24_x_mean','kp_24_y_mean']):
        dx = shotwise_df['kp_23_x_mean'] - shotwise_df['kp_24_x_mean']
        dy = shotwise_df['kp_23_y_mean'] - shotwise_df['kp_24_y_mean']
        shotwise_df['pair_23_24_mean_dist'] = np.sqrt(dx**2 + dy**2)

    rules_df = pd.read_csv(rules_csv)
    rules_df['parsed_thresholds'] = rules_df['Threshold'].apply(parse_thresholds)

    shotwise_df['shot_type_lower'] = shotwise_df['predicted_shot_type'].str.lower()
    rules_df['shot_type_lower'] = (
        rules_df['Shot Type']
        .str.extract(r'([a-zA-Z ]+)')[0]
        .str.lower()
        .str.strip()
    )

    feedback_results = []
    for _, row in shotwise_df.iterrows():
        shot_type = row['shot_type_lower']
        matching_rules = rules_df[rules_df['shot_type_lower'] == shot_type]
        triggered_feedbacks = []
        for _, rule in matching_rules.iterrows():
            thresholds = rule['parsed_thresholds']
            feedback_text = rule['Feedback Inference']
            violation = False
            for feature, bounds in thresholds.items():
                if feature not in shotwise_df.columns:
                    continue
                val = row[feature]
                if not (bounds[0] <= val <= bounds[1]):
                    violation = True
                    break
            if violation:
                triggered_feedbacks.append(feedback_text)

        feedback_results.append({
            'shot_no': row['shot_no'],
            'predicted_shot_type': row['predicted_shot_type'],
            'feedback_count': len(triggered_feedbacks),
            'feedback': "\n".join(triggered_feedbacks) if triggered_feedbacks else ""
        })

    out_df = pd.DataFrame(feedback_results)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved shotwise feedback to: {out_csv}")
    return out_df
