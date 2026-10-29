"""
IPL Data Analytics and Match Winner Prediction - Stage 1: Dataset Loading & EDA
Author: CSE AI/ML Student
Description:
    This script loads, validates, and performs in-depth Exploratory Data Analysis (EDA)
    on historical IPL match data. It also exports key statistical summaries and charts.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def get_project_root() -> Path:
    """Return the absolute path of the project root directory."""
    return Path(__file__).resolve().parent


def ensure_clean_matches_csv(dataset_dir: Path) -> Path:
    """
    Ensure dataset/matches.csv exists with human-readable team names.
    If only raw ipl_matches_data.csv and teams_data.csv exist, automatically
    join and export dataset/matches.csv.
    """
    matches_csv = dataset_dir / "matches.csv"
    raw_matches_csv = dataset_dir / "ipl_matches_data.csv"
    teams_csv = dataset_dir / "teams_data.csv"

    if matches_csv.exists():
        print(f"[OK] Found standard matches dataset: {matches_csv}")
        return matches_csv

    if raw_matches_csv.exists() and teams_csv.exists():
        print("[INFO] Constructing canonical matches.csv from raw IPL data and team mappings...")
        raw_matches = pd.read_csv(raw_matches_csv)
        teams = pd.read_csv(teams_csv)
        team_map = dict(zip(teams["team_id"], teams["team_name"]))

        df = raw_matches.copy()
        # Map IDs to actual franchise names and standardize naming variations
        for col in ["team1", "team2", "toss_winner", "match_winner"]:
            df[f"{col}_name"] = df[col].map(team_map).replace({
                "Rising Pune Supergiants": "Rising Pune Supergiant"
            })

        # When a match was abandoned or had no result, winner is NaN
        df.loc[df["result"] == "no result", "match_winner_name"] = None

        clean_df = pd.DataFrame({
            "match_id": df["match_id"],
            "season": df["season"],
            "city": df["city"],
            "match_date": df["match_date"],
            "venue": df["venue"],
            "team1": df["team1_name"],
            "team2": df["team2_name"],
            "toss_winner": df["toss_winner_name"],
            "toss_decision": df["toss_decision"],
            "winner": df["match_winner_name"],
            "result": df["result"],
            "win_by_runs": df["win_by_runs"].fillna(0).astype(int),
            "win_by_wickets": df["win_by_wickets"].fillna(0).astype(int),
            "player_of_match": df.get("player_of_match", np.nan)
        })

        clean_df.to_csv(matches_csv, index=False)
        print(f"[OK] Created {matches_csv} with {len(clean_df)} records.")
        return matches_csv

    raise FileNotFoundError(
        f"Could not find dataset in {dataset_dir}.\n"
        f"Please ensure either 'matches.csv' or 'ipl_matches_data.csv' is present in the dataset folder."
    )


def load_and_validate_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load dataset and validate core columns, data types, and check for missing values.
    """
    print("\n" + "=" * 70)
    print("STEP 1: DATASET INGESTION & SCHEMA VALIDATION")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    print(f"Dataset successfully loaded. Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")

    expected_cols = ["season", "team1", "team2", "toss_winner", "toss_decision", "venue", "winner", "result"]
    missing_cols = [c for c in expected_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    print("[PASS] All required features for match analytics and ML modeling are present.")
    
    # Missing value audit
    null_counts = df.isnull().sum()
    print("\nMissing Values Audit:")
    for col, count in null_counts.items():
        if count > 0:
            pct = (count / len(df)) * 100
            print(f" - {col:18s}: {count:4d} missing ({pct:5.2f}%)")

    # Duplicates check
    if "match_id" in df.columns:
        dups = df.duplicated(subset=["match_id"]).sum()
        print(f"\nDuplicate match IDs: {dups}")
    else:
        dups = df.duplicated().sum()
        print(f"\nDuplicate entire rows: {dups}")

    return df


def perform_eda(df: pd.DataFrame, output_dir: Path, images_dir: Path) -> dict:
    """
    Perform deep Exploratory Data Analysis and generate statistical summaries and plots.
    """
    print("\n" + "=" * 70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA) & CRICKET INSIGHTS")
    print("=" * 70)

    total_matches = len(df)
    seasons = sorted(df["season"].dropna().unique().astype(str))
    venues = sorted(df["venue"].dropna().unique())
    
    # All participating teams
    all_teams = sorted(pd.concat([df["team1"], df["team2"]]).dropna().unique())

    print(f"Total IPL Matches Analyzed : {total_matches}")
    print(f"Seasons Covered            : {len(seasons)} ({seasons[0]} to {seasons[-1]})")
    print(f"Total Unique Venues        : {len(venues)}")
    print(f"Total Unique Franchises    : {len(all_teams)}")

    # 1. Match Results Breakdown
    result_counts = df["result"].value_counts().to_dict()
    print("\n--- 1. Match Outcome Distribution ---")
    for res, count in result_counts.items():
        print(f"  {res:12s}: {count:4d} ({count / total_matches * 100:.2f}%)")

    # 2. Team Performance: Matches, Wins, Win Percentage
    team_stats = {}
    for team in all_teams:
        matches_played = len(df[(df["team1"] == team) | (df["team2"] == team)])
        wins = len(df[df["winner"] == team])
        win_pct = round((wins / matches_played * 100), 2) if matches_played > 0 else 0.0
        team_stats[team] = {
            "matches_played": matches_played,
            "wins": wins,
            "win_percentage": win_pct
        }

    team_stats_df = pd.DataFrame.from_dict(team_stats, orient="index")
    team_stats_df = team_stats_df.sort_values(by="wins", ascending=False)

    print("\n--- 2. Franchise Leaderboard (Ranked by Total Wins) ---")
    print(team_stats_df.to_string())

    # 3. Toss Impact Analysis
    # Does winning toss equate to winning the match?
    valid_winner_df = df[df["winner"].notna() & (df["result"] != "no result")].copy()
    valid_winner_df["toss_match_winner"] = valid_winner_df["toss_winner"] == valid_winner_df["winner"]
    toss_win_match_win_count = valid_winner_df["toss_match_winner"].sum()
    toss_win_match_win_pct = (toss_win_match_win_count / len(valid_winner_df)) * 100

    print("\n--- 3. Toss Impact on Match Outcome ---")
    print(f"Matches where Toss Winner won Match : {toss_win_match_win_count} / {len(valid_winner_df)} ({toss_win_match_win_pct:.2f}%)")
    print(f"Matches where Toss Loser won Match  : {len(valid_winner_df) - toss_win_match_win_count} / {len(valid_winner_df)} ({100 - toss_win_match_win_pct:.2f}%)")

    # Toss decision breakdown
    toss_decision_counts = df["toss_decision"].str.lower().value_counts()
    print("\nToss Decision Breakdown:")
    for decision, count in toss_decision_counts.items():
        print(f"  Chose to {decision.upper():5s}: {count:4d} times ({count / total_matches * 100:.2f}%)")

    # Win rate by toss decision
    field_df = valid_winner_df[valid_winner_df["toss_decision"].str.lower() == "field"]
    bat_df = valid_winner_df[valid_winner_df["toss_decision"].str.lower() == "bat"]
    field_win_pct = (field_df["toss_match_winner"].sum() / len(field_df) * 100) if len(field_df) > 0 else 0
    bat_win_pct = (bat_df["toss_match_winner"].sum() / len(bat_df) * 100) if len(bat_df) > 0 else 0

    print(f"Success rate when choosing to FIELD : {field_win_pct:.2f}%")
    print(f"Success rate when choosing to BAT   : {bat_win_pct:.2f}%")

    # 4. Top 10 Venues by Matches Hosted
    venue_counts = df["venue"].value_counts().head(10)
    print("\n--- 4. Top 10 Most Hosted Venues ---")
    for venue, cnt in venue_counts.items():
        print(f"  {venue:45s}: {cnt:3d} matches")

    # 5. Head-to-Head Top Rivalries
    print("\n--- 5. Iconic Rivalries Head-to-Head ---")
    rivalries = [
        ("Mumbai Indians", "Chennai Super Kings"),
        ("Royal Challengers Bangalore", "Kolkata Knight Riders"),
        ("Chennai Super Kings", "Royal Challengers Bangalore"),
        ("Mumbai Indians", "Kolkata Knight Riders"),
        ("Delhi Capitals", "Punjab Kings"),
    ]
    h2h_data = {}
    for t1, t2 in rivalries:
        sub = valid_winner_df[
            ((valid_winner_df["team1"] == t1) & (valid_winner_df["team2"] == t2)) |
            ((valid_winner_df["team1"] == t2) & (valid_winner_df["team2"] == t1))
        ]
        t1_wins = (sub["winner"] == t1).sum()
        t2_wins = (sub["winner"] == t2).sum()
        h2h_data[f"{t1} vs {t2}"] = {
            "total_matches": len(sub),
            f"{t1}_wins": int(t1_wins),
            f"{t2}_wins": int(t2_wins)
        }
        print(f"  {t1} vs {t2}: Played {len(sub)} | {t1}: {t1_wins} wins | {t2}: {t2_wins} wins")

    # Save summary dictionary for web app & report
    summary = {
        "total_matches": int(total_matches),
        "total_seasons": len(seasons),
        "seasons_list": seasons,
        "total_venues": len(venues),
        "total_teams": len(all_teams),
        "teams_list": all_teams,
        "team_statistics": team_stats,
        "toss_impact": {
            "toss_winner_won_match_pct": round(toss_win_match_win_pct, 2),
            "field_decision_win_pct": round(field_win_pct, 2),
            "bat_decision_win_pct": round(bat_win_pct, 2),
            "decision_breakdown": {k: int(v) for k, v in toss_decision_counts.items()}
        },
        "top_venues": {k: int(v) for k, v in venue_counts.items()},
        "head_to_head": h2h_data
    }

    summary_file = output_dir / "eda_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"\n[OK] Saved EDA statistical summary to: {summary_file}")

    # Generate visual charts for college viva & documentation
    generate_eda_plots(df, team_stats_df, images_dir)

    return summary


def generate_eda_plots(df: pd.DataFrame, team_stats_df: pd.DataFrame, images_dir: Path):
    """Generate and save clean, high-resolution EDA figures for project reports."""
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING EDA VISUALIZATION CHARTS")
    print("=" * 70)
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Matches Won by Each Franchise
    plt.figure(figsize=(12, 6))
    top_teams = team_stats_df.head(10)
    colors = sns.color_palette("viridis", len(top_teams))
    bars = plt.barh(top_teams.index[::-1], top_teams["wins"][::-1], color=colors)
    plt.title("Top IPL Franchises by Total Wins (2008 - 2026)", fontsize=14, weight="bold", pad=15)
    plt.xlabel("Total Wins", fontsize=12)
    plt.ylabel("Team", fontsize=12)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1.5, bar.get_y() + bar.get_height() / 2, f"{int(width)}",
                 va="center", ha="left", fontsize=10, weight="bold")
    plt.tight_layout()
    plot1_path = images_dir / "eda_top_teams_wins.png"
    plt.savefig(plot1_path, dpi=200)
    plt.close()
    print(f"[OK] Generated: {plot1_path}")

    # 2. Matches Played Per Season Trend
    plt.figure(figsize=(12, 5))
    season_counts = df["season"].astype(str).value_counts().sort_index()
    plt.plot(season_counts.index, season_counts.values, marker="o", color="#1e3c72", linewidth=2.5, markersize=6)
    plt.fill_between(season_counts.index, season_counts.values, color="#2a5298", alpha=0.15)
    plt.title("IPL Matches Played Per Season", fontsize=14, weight="bold", pad=15)
    plt.xlabel("Season", fontsize=12)
    plt.ylabel("Number of Matches", fontsize=12)
    plt.xticks(rotation=45)
    for i, v in enumerate(season_counts.values):
        plt.text(i, v + 1, str(v), ha="center", fontsize=9, weight="bold")
    plt.tight_layout()
    plot2_path = images_dir / "eda_matches_per_season.png"
    plt.savefig(plot2_path, dpi=200)
    plt.close()
    print(f"[OK] Generated: {plot2_path}")

    # 3. Toss Decision Distribution and Win Impact
    plt.figure(figsize=(10, 5))
    toss_counts = df["toss_decision"].str.capitalize().value_counts()
    plt.subplot(1, 2, 1)
    plt.pie(toss_counts.values, labels=toss_counts.index, autopct="%1.1f%%",
            startangle=140, colors=["#3498db", "#f39c12"], explode=(0.05, 0))
    plt.title("Toss Decision Preference", fontsize=12, weight="bold")

    plt.subplot(1, 2, 2)
    valid_winner_df = df[df["winner"].notna() & (df["result"] != "no result")].copy()
    valid_winner_df["toss_match_winner"] = (valid_winner_df["toss_winner"] == valid_winner_df["winner"]).map(
        {True: "Toss Winner Won Match", False: "Toss Winner Lost Match"}
    )
    outcome_counts = valid_winner_df["toss_match_winner"].value_counts()
    plt.pie(outcome_counts.values, labels=outcome_counts.index, autopct="%1.1f%%",
            startangle=140, colors=["#2ecc71", "#e74c3c"], explode=(0.05, 0))
    plt.title("Toss Win vs Match Win Impact", fontsize=12, weight="bold")

    plt.tight_layout()
    plot3_path = images_dir / "eda_toss_impact.png"
    plt.savefig(plot3_path, dpi=200)
    plt.close()
    print(f"[OK] Generated: {plot3_path}")


def main():
    """Main execution flow for Stage 1."""
    root = get_project_root()
    dataset_dir = root / "dataset"
    images_dir = root / "static" / "images"

    try:
        csv_path = ensure_clean_matches_csv(dataset_dir)
        df = load_and_validate_dataset(csv_path)
        summary = perform_eda(df, dataset_dir, images_dir)

        print("\n" + "=" * 70)
        print("STAGE 1 COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("Summary of Achievements:")
        print(" 1. Dataset ingested and canonical 'dataset/matches.csv' verified.")
        print(f" 2. Processed {summary['total_matches']} historical matches across {summary['total_seasons']} seasons.")
        print(f" 3. Generated and exported statistical insights into: dataset/eda_summary.json")
        print(f" 4. Saved visualization plots into: static/images/")
        print(" 5. Ready for Stage 2 (Feature Engineering & Model Training).\n")

    except Exception as e:
        print(f"\n[ERROR] Stage 1 encountered an issue: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
