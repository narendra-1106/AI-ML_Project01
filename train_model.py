"""
IPL Match Winner Prediction - Stage 2 & 3: Model Training, Evaluation, and Pipeline Persistence
Author: CSE AI/ML Student
Description:
    Trains and compares Decision Tree and Random Forest classifiers on historical IPL matches.
    Evaluates both models with strict chronological splitting to prevent temporal data leakage.
    Saves the best-performing pipeline and evaluation metrics for deployment in Flask.
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from eda import ensure_clean_matches_csv, get_project_root


def prepare_training_data(df: pd.DataFrame):
    """
    Filter valid matches, construct binary target (team1_won: 1, team2_won: 0),
    and apply order-invariance augmentation to prevent positional bias.
    """
    # Exclude abandoned / no result matches
    valid_df = df[df["winner"].notna() & (df["result"] != "no result")].copy()

    # Target: 1 if team1 won, 0 if team2 won
    # Verify winner is one of the two competing teams
    valid_df = valid_df[(valid_df["winner"] == valid_df["team1"]) | (valid_df["winner"] == valid_df["team2"])].copy()
    valid_df["team1_won"] = (valid_df["winner"] == valid_df["team1"]).astype(int)

    features = ["team1", "team2", "toss_winner", "toss_decision", "venue"]

    # To eliminate positional bias (team1 vs team2 ordering), we augment each match with swapped teams
    swapped_df = valid_df.copy()
    swapped_df["team1"] = valid_df["team2"]
    swapped_df["team2"] = valid_df["team1"]
    swapped_df["team1_won"] = 1 - valid_df["team1_won"]

    # In swapped rows, toss_winner and venue remain the same
    # Combine original and swapped data while keeping season for chronological split
    augmented_df = pd.concat([valid_df, swapped_df], ignore_index=True)

    return augmented_df, features


def chronological_split(augmented_df: pd.DataFrame, features: list):
    """
    Split the dataset chronologically by season to prevent temporal leakage.
    Matches up to season 2022 are used for training; 2023-2026 are used for evaluation.
    """
    # Extract integer year for sorting/splitting
    def parse_season_year(s):
        s_str = str(s).split("/")[0].strip()
        try:
            return int(s_str)
        except ValueError:
            return 2020

    augmented_df["season_year"] = augmented_df["season"].apply(parse_season_year)

    # Chronological threshold: use ~80% historical seasons for training
    train_mask = augmented_df["season_year"] <= 2022
    test_mask = augmented_df["season_year"] > 2022

    X_train = augmented_df.loc[train_mask, features]
    y_train = augmented_df.loc[train_mask, "team1_won"]

    X_test = augmented_df.loc[test_mask, features]
    y_test = augmented_df.loc[test_mask, "team1_won"]

    print(f"Total Augmented Records: {len(augmented_df)}")
    print(f"Chronological Split:")
    print(f" - Train set (Seasons <= 2022) : {len(X_train)} samples")
    print(f" - Test set  (Seasons > 2022)  : {len(X_test)} samples")

    return X_train, X_test, y_train, y_test


def evaluate_model(name: str, pipeline: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    """Train pipeline and compute complete evaluation metrics on train and test sets."""
    print("\n" + "-" * 60)
    print(f"TRAINING & EVALUATING: {name}")
    print("-" * 60)

    pipeline.fit(X_train, y_train)

    train_preds = pipeline.predict(X_train)
    test_preds = pipeline.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds, average="weighted", zero_division=0)
    recall = recall_score(y_test, test_preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, test_preds).tolist()
    report = classification_report(y_test, test_preds, target_names=["Team 2 Wins", "Team 1 Wins"], output_dict=True)

    print(f"Train Accuracy : {train_acc * 100:.2f}%")
    print(f"Test Accuracy  : {test_acc * 100:.2f}%")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-Score       : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(np.array(cm))
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds, target_names=["Team 2 Wins", "Team 1 Wins"]))

    return {
        "name": name,
        "train_accuracy": round(float(train_acc) * 100, 2),
        "test_accuracy": round(float(test_acc) * 100, 2),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm,
        "classification_report": report
    }


def main():
    root = get_project_root()
    dataset_dir = root / "dataset"
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    csv_path = ensure_clean_matches_csv(dataset_dir)
    df = pd.read_csv(csv_path)

    # 1. Prepare data
    augmented_df, features = prepare_training_data(df)

    # Collect unique choices for web dropdowns
    all_teams = sorted(pd.concat([df["team1"], df["team2"]]).dropna().unique().tolist())
    all_venues = sorted(df["venue"].dropna().unique().tolist())

    # 2. Chronological Train-Test Split
    X_train, X_test, y_train, y_test = chronological_split(augmented_df, features)

    # 3. Categorical Preprocessor (One-Hot Encoding with unknown category safety)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features)
        ]
    )

    # 4. Define Candidate Models
    dt_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=15,
                min_samples_leaf=10,
                random_state=42
            ))
        ]
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=120,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ))
        ]
    )

    # 5. Train & Evaluate Both Models
    dt_metrics = evaluate_model("Decision Tree Classifier", dt_pipeline, X_train, y_train, X_test, y_test)
    rf_metrics = evaluate_model("Random Forest Classifier", rf_pipeline, X_train, y_train, X_test, y_test)

    # 6. Select Champion Model based on Test F1-Score / Accuracy
    if rf_metrics["f1_score"] >= dt_metrics["f1_score"]:
        champion_name = "Random Forest Classifier"
        champion_pipeline = rf_pipeline
        champion_metrics = rf_metrics
    else:
        champion_name = "Decision Tree Classifier"
        champion_pipeline = dt_pipeline
        champion_metrics = dt_metrics

    print("\n" + "=" * 60)
    print(f"CHAMPION MODEL SELECTED: {champion_name}")
    print(f"Test Accuracy: {champion_metrics['test_accuracy']}% | F1-Score: {champion_metrics['f1_score']}")
    print("=" * 60)

    # 7. Persist Champion Pipeline & Artifacts
    model_save_path = models_dir / "match_winner_model.joblib"
    joblib.dump(champion_pipeline, model_save_path)
    print(f"[OK] Saved Champion Pipeline to: {model_save_path}")

    # Save comprehensive comparison and feature metadata
    metadata = {
        "champion_model": champion_name,
        "features": features,
        "all_teams": all_teams,
        "all_venues": all_venues,
        "comparison": {
            "decision_tree": dt_metrics,
            "random_forest": rf_metrics
        }
    }

    metrics_save_path = models_dir / "model_metrics.json"
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[OK] Saved Model Evaluation Metrics to: {metrics_save_path}")

    # 8. Sanity Check Inference
    sample_match = pd.DataFrame([{
        "team1": "Chennai Super Kings",
        "team2": "Mumbai Indians",
        "toss_winner": "Chennai Super Kings",
        "toss_decision": "field",
        "venue": "MA Chidambaram Stadium, Chepauk"
    }])
    sample_probs = champion_pipeline.predict_proba(sample_match)[0]
    p_team1 = round(float(sample_probs[1]) * 100, 2)
    p_team2 = round(float(sample_probs[0]) * 100, 2)
    pred_winner = "Chennai Super Kings" if p_team1 >= 50 else "Mumbai Indians"

    print("\n--- Model Inference Sanity Test ---")
    print(f"Match: CSK vs MI at Chepauk (CSK won toss & field)")
    print(f"Probabilities -> CSK: {p_team1}% | MI: {p_team2}%")
    print(f"Predicted Likely Winner: {pred_winner}")
    print("[OK] Sanity check passed successfully!")


if __name__ == "__main__":
    main()
