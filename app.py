"""
IPL Data Analytics and Match Winner Prediction - Flask Web Application
Author: CSE AI/ML Student
Description:
    Full-stack web application providing interactive IPL data analytics,
    team performance dashboards, and machine learning match winner predictions.
"""

import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import joblib
import pandas as pd
import numpy as np

from database import init_db, save_prediction, get_prediction_history, clear_history

app = Flask(__name__)
app.secret_key = "ipl_prediction_secret_key_2026"

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "match_winner_model.joblib"
METRICS_PATH = ROOT_DIR / "models" / "model_metrics.json"
EDA_PATH = ROOT_DIR / "dataset" / "eda_summary.json"
MATCHES_CSV = ROOT_DIR / "dataset" / "matches.csv"

# Global references loaded once on startup
model_pipeline = None
model_metadata = {}
eda_data = {}
matches_df = None


def load_application_resources():
    """Load model, metadata, and dataset into memory on startup."""
    global model_pipeline, model_metadata, eda_data, matches_df

    init_db()

    # Load EDA summary
    if EDA_PATH.exists():
        with open(EDA_PATH, "r", encoding="utf-8") as f:
            eda_data = json.load(f)

    # Load Model Metadata
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            model_metadata = json.load(f)

    # Load Pipeline
    if MODEL_PATH.exists():
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"[OK] Loaded ML Pipeline from: {MODEL_PATH}")
    else:
        print("[WARNING] Model file not found. Please run train_model.py first.")

    # Load matches dataset for dynamic filters and calculations
    if MATCHES_CSV.exists():
        matches_df = pd.read_csv(MATCHES_CSV)
        print(f"[OK] Loaded {len(matches_df)} matches from: {MATCHES_CSV}")


load_application_resources()


@app.route("/")
def index():
    """Home page with dataset summary cards and project introduction."""
    stats = {
        "total_matches": eda_data.get("total_matches", 1212),
        "total_seasons": eda_data.get("total_seasons", 19),
        "total_venues": eda_data.get("total_venues", 59),
        "total_teams": eda_data.get("total_teams", 14),
        "top_team": "Mumbai Indians",
        "top_wins": 155
    }
    return render_template("index.html", stats=stats)


@app.route("/prediction")
def prediction_form():
    """Render the match prediction input form with valid teams and venues."""
    teams = model_metadata.get("all_teams", [])
    venues = model_metadata.get("all_venues", [])
    if not teams and eda_data:
        teams = eda_data.get("teams_list", [])

    return render_template("prediction.html", teams=teams, venues=venues)


@app.route("/predict", methods=["POST"])
def predict():
    """Handle match prediction form submission, run inference, and log record."""
    if model_pipeline is None:
        flash("Machine learning model is currently not loaded. Please train the model first.", "danger")
        return redirect(url_for("prediction_form"))

    team1 = request.form.get("team1", "").strip()
    team2 = request.form.get("team2", "").strip()
    toss_winner = request.form.get("toss_winner", "").strip()
    toss_decision = request.form.get("toss_decision", "").strip().lower()
    venue = request.form.get("venue", "").strip()

    # Server-side validation
    if not team1 or not team2 or not toss_winner or not toss_decision or not venue:
        flash("All match parameters are required. Please fill out the complete form.", "warning")
        return redirect(url_for("prediction_form"))

    if team1 == team2:
        flash("Team 1 and Team 2 cannot be the same franchise. Please choose two distinct teams.", "danger")
        return redirect(url_for("prediction_form"))

    if toss_winner not in [team1, team2]:
        flash("Toss winner must be either Team 1 or Team 2.", "danger")
        return redirect(url_for("prediction_form"))

    if toss_decision not in ["bat", "field"]:
        flash("Toss decision must be either 'bat' or 'field'.", "danger")
        return redirect(url_for("prediction_form"))

    # Prepare DataFrame for pipeline
    input_data = pd.DataFrame([{
        "team1": team1,
        "team2": team2,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "venue": venue
    }])

    try:
        probabilities = model_pipeline.predict_proba(input_data)[0]
        # Class 1 = team1 wins, Class 0 = team2 wins
        prob_team1 = round(float(probabilities[1]) * 100, 2)
        prob_team2 = round(float(probabilities[0]) * 100, 2)

        if prob_team1 >= 50.0:
            predicted_winner = team1
            confidence = prob_team1
        else:
            predicted_winner = team2
            confidence = prob_team2

        # Save prediction into SQLite (safe execution - will not disrupt inference on read-only environments)
        try:
            save_prediction(
                team1=team1,
                team2=team2,
                toss_winner=toss_winner,
                toss_decision=toss_decision,
                venue=venue,
                predicted_winner=predicted_winner,
                prob_team1=prob_team1,
                prob_team2=prob_team2
            )
        except Exception as db_err:
            print(f"[INFO] Prediction save skipped or memory-only: {db_err}")

        # Historical Head-to-Head between these two teams
        h2h_played = 0
        h2h_t1_wins = 0
        h2h_t2_wins = 0
        if matches_df is not None:
            sub = matches_df[
                ((matches_df["team1"] == team1) & (matches_df["team2"] == team2)) |
                ((matches_df["team1"] == team2) & (matches_df["team2"] == team1))
            ]
            h2h_played = len(sub)
            h2h_t1_wins = int((sub["winner"] == team1).sum())
            h2h_t2_wins = int((sub["winner"] == team2).sum())

        result_context = {
            "team1": team1,
            "team2": team2,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision.capitalize(),
            "venue": venue,
            "predicted_winner": predicted_winner,
            "confidence": confidence,
            "prob_team1": prob_team1,
            "prob_team2": prob_team2,
            "model_name": model_metadata.get("champion_model", "Random Forest Classifier"),
            "h2h_played": h2h_played,
            "h2h_t1_wins": h2h_t1_wins,
            "h2h_t2_wins": h2h_t2_wins
        }

        return render_template("result.html", result=result_context)

    except Exception as e:
        flash(f"Inference error during match prediction: {str(e)}", "danger")
        return redirect(url_for("prediction_form"))


@app.route("/dashboard")
def dashboard():
    """Render interactive data analytics dashboard powered by Chart.js."""
    return render_template("dashboard.html", eda=eda_data)


@app.route("/api/dashboard-data")
def api_dashboard_data():
    """Return actual historical metrics for dynamic Chart.js rendering."""
    if not eda_data and EDA_PATH.exists():
        with open(EDA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = eda_data

    # Matches per season
    season_distribution = {}
    if matches_df is not None:
        counts = matches_df["season"].astype(str).value_counts().sort_index()
        season_distribution = {k: int(v) for k, v in counts.items()}

    # Team wins and win %
    team_stats = data.get("team_statistics", {})
    sorted_teams = sorted(team_stats.items(), key=lambda item: item[1]["wins"], reverse=True)

    team_names = [t[0] for t in sorted_teams]
    team_wins = [t[1]["wins"] for t in sorted_teams]
    team_matches = [t[1]["matches_played"] for t in sorted_teams]
    team_win_pcts = [t[1]["win_percentage"] for t in sorted_teams]

    toss_data = data.get("toss_impact", {})
    top_venues = data.get("top_venues", {})

    return jsonify({
        "seasons": list(season_distribution.keys()),
        "matches_per_season": list(season_distribution.values()),
        "team_names": team_names,
        "team_wins": team_wins,
        "team_matches": team_matches,
        "team_win_pcts": team_win_pcts,
        "toss_impact": toss_data,
        "top_venues_labels": list(top_venues.keys()),
        "top_venues_counts": list(top_venues.values())
    })


@app.route("/team-stats")
def team_stats():
    """Team statistics page with interactive selector and comprehensive metrics."""
    teams = eda_data.get("teams_list", [])
    selected_team = request.args.get("team", "Mumbai Indians")
    if selected_team not in teams and teams:
        selected_team = teams[0]

    team_info = {}
    h2h_records = []
    season_trend = {}

    if matches_df is not None and selected_team:
        team_matches = matches_df[(matches_df["team1"] == selected_team) | (matches_df["team2"] == selected_team)]
        total_played = len(team_matches)
        total_wins = int((team_matches["winner"] == selected_team).sum())
        win_pct = round((total_wins / total_played * 100), 2) if total_played > 0 else 0.0

        # Toss stats
        toss_won = int((team_matches["toss_winner"] == selected_team).sum())
        toss_won_match_won = int(((team_matches["toss_winner"] == selected_team) & (team_matches["winner"] == selected_team)).sum())
        toss_win_rate = round((toss_won_match_won / toss_won * 100), 2) if toss_won > 0 else 0.0

        # Season-wise breakdown
        for season, group in team_matches.groupby("season"):
            s_played = len(group)
            s_won = int((group["winner"] == selected_team).sum())
            season_trend[str(season)] = {"played": s_played, "won": s_won}

        # Head-to-head against other teams
        other_teams = [t for t in teams if t != selected_team]
        for opp in other_teams:
            opp_matches = team_matches[(team_matches["team1"] == opp) | (team_matches["team2"] == opp)]
            if len(opp_matches) > 0:
                opp_played = len(opp_matches)
                opp_won = int((opp_matches["winner"] == selected_team).sum())
                opp_lost = int((opp_matches["winner"] == opp).sum())
                h2h_records.append({
                    "opponent": opp,
                    "played": opp_played,
                    "won": opp_won,
                    "lost": opp_lost,
                    "win_pct": round((opp_won / opp_played * 100), 1)
                })

        h2h_records = sorted(h2h_records, key=lambda x: x["played"], reverse=True)

        team_info = {
            "name": selected_team,
            "total_played": total_played,
            "total_wins": total_wins,
            "win_percentage": win_pct,
            "toss_won": toss_won,
            "toss_win_rate": toss_win_rate,
            "season_trend": season_trend
        }

    return render_template("team_stats.html", teams=teams, selected_team=selected_team, team_info=team_info, h2h=h2h_records)


@app.route("/history")
def history():
    """Display persistent log of match predictions from SQLite."""
    records = get_prediction_history(limit=50)
    return render_template("history.html", records=records)


@app.route("/clear-history", methods=["POST"])
def clear_prediction_history():
    """Clear recorded prediction history."""
    clear_history()
    flash("Prediction history cleared successfully.", "info")
    return redirect(url_for("history"))


@app.route("/about")
def about():
    """Project description, machine learning architecture, viva questions, and limitations."""
    metrics = model_metadata.get("comparison", {})
    return render_template("about.html", metrics=metrics)


if __name__ == "__main__":
    print("\nStarting IPL Match Analytics & Prediction Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)
