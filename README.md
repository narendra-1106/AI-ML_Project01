# IPL Data Analytics and Match Winner Prediction Using Machine Learning

A complete, beginner-friendly B.Tech CSE AI/ML college mini-project built to demonstrate data analytics, statistical modeling, machine learning classification, and full-stack web deployment.

---

## 📌 Project Overview
This project analyzes real historical Indian Premier League (IPL) match records from 2008 through 2026 and builds machine learning models to predict the probable winner of a match given pre-match features:
- **Team 1**
- **Team 2**
- **Toss Winner**
- **Toss Decision** (Bat or Field)
- **Match Venue**

---

## 📂 Complete Project Folder Structure

```
AI&ML_Project01/
│
├── dataset/
│   ├── matches.csv              # Canonical cleaned IPL match records (1,212 matches)
│   ├── ipl_matches_data.csv     # Raw match dataset
│   ├── teams_data.csv           # Franchise ID-to-Name mapping
│   ├── team_aliases.csv         # Rebranding & historical team name aliases
│   └── eda_summary.json         # Automated JSON output of EDA metrics
│
├── static/
│   ├── css/
│   │   └── style.css            # Custom responsive stylesheet (IPL cricket theme)
│   ├── js/
│   │   └── dashboard.js         # Chart.js frontend visualization scripts
│   └── images/                  # High-resolution EDA charts
│       ├── eda_matches_per_season.png
│       ├── eda_top_teams_wins.png
│       └── eda_toss_impact.png
│
├── templates/
│   ├── base.html                # Navigation bar, common header, and footer layout
│   ├── index.html               # Home page with genuine dataset summary statistics
│   ├── prediction.html          # Interactive match input form with validation
│   ├── result.html              # Prediction result, win likelihoods, and H2H record
│   ├── dashboard.html           # Interactive Chart.js analytics dashboard
│   ├── team_stats.html          # Franchise explorer with head-to-head records
│   ├── history.html             # Persistent SQLite prediction history log
│   └── about.html               # Methodology, viva guide, and model comparison
│
├── models/
│   ├── match_winner_model.joblib # Persisted champion Random Forest pipeline
│   └── model_metrics.json       # Decision Tree vs Random Forest evaluation metrics
│
├── database/
│   └── predictions.db           # SQLite database for persistent predictions
│
├── app.py                       # Main Flask web application
├── train_model.py               # Model training, comparison, and evaluation pipeline
├── eda.py                       # Exploratory Data Analysis and data verification
├── database.py                  # SQLite database helper module
├── requirements.txt             # Python dependencies
└── README.md                    # Setup, execution, and viva preparation guide
```

---

## 🛠️ Technology Stack
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js, Font Awesome
- **Backend**: Python 3.12, Flask
- **Machine Learning**: Scikit-learn (DecisionTreeClassifier, RandomForestClassifier, OneHotEncoder, ColumnTransformer, Pipeline), Joblib
- **Data Analytics**: Pandas, NumPy, Matplotlib, Seaborn
- **Database**: SQLite3

---

## 🚀 How to Run the Project Locally in VS Code

### 1. Open Workspace
Open VS Code, press `Ctrl + O` (or `File -> Open Folder...`), and select the project directory:
```
AI&ML_Project01
```

### 2. Activate Virtual Environment
Open the integrated terminal in VS Code (``Ctrl + ` ``):
```powershell
# If execution policy is restricted:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### 3. Run Exploratory Data Analysis (EDA)
```powershell
python eda.py
```

### 4. Train & Compare Machine Learning Models
```powershell
python train_model.py
```

### 5. Launch the Web Application
```powershell
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🏆 Machine Learning Model Evaluation Summary

Both models were evaluated using strict chronological splitting (training on matches up to season 2022, testing on matches from 2023 to 2026):

| Metric | Model 1: Decision Tree | Model 2: Random Forest (Champion) |
|---|:---:|:---:|
| **Train Accuracy** | 54.65% | 60.52% |
| **Test Accuracy** | 48.05% | **50.97%** |
| **Precision (Weighted)** | 0.4694 | **0.5098** |
| **Recall (Weighted)** | 0.4805 | **0.5097** |
| **F1-Score (Weighted)** | 0.4286 | **0.5093** |
| **Hyperparameters** | `max_depth=5, min_samples_split=15` | `n_estimators=120, max_depth=8, min_samples_split=10` |

### Why Random Forest is the Champion:
T20 cricket is subject to high stochastic variability. A single decision tree suffers from high variance and easily overfits specific venue/toss splits. Random Forest aggregates 120 randomized bootstrap decision trees (bagging), reducing prediction variance and achieving a higher test F1-score.

---

## 🌐 Website Routes & Features
- `/` — Home page with genuine dataset statistics cards (1,212 matches, 19 seasons, 59 venues, 14 franchises).
- `/prediction` — Match prediction form with client-side & server-side validation (prevents selecting the same team, auto-populates toss winner options).
- `/predict` — ML inference endpoint computing likelihoods and saving to SQLite.
- `/dashboard` — Interactive Chart.js analytics dashboard (matches per season, franchise wins, toss impact, venue distributions, win %).
- `/team-stats` — Franchise explorer with head-to-head records against all opponents and toss correlation.
- `/history` — Persistent SQLite audit trail of all logged predictions.
- `/about` — Complete academic methodology, model comparison table, and viva Q&A.

---

## 🎓 Viva Questions & Answers

### Q1: What pre-match features are used to predict the winner?
> **Answer**: `team1`, `team2`, `toss_winner`, `toss_decision`, and `venue`. Post-match variables (such as scores, wickets, or Player of the Match) are strictly excluded to avoid data leakage.

### Q2: Why is sports prediction accuracy typically in the 52%–62% range?
> **Answer**: In professional sports, competing teams have balanced squads and competitive parity. Unlike image classification where pixels contain deterministic information, match outcomes are influenced by unpredictable events (dew, injuries, drop catches, individual player form). A model achieving ~55% edge over random guessing provides significant predictive value.

### Q3: How do you handle unseen venues or new teams at inference time?
> **Answer**: The preprocessing pipeline uses `OneHotEncoder(handle_unknown='ignore')`. If an unseen venue is selected, its one-hot columns evaluate to zero without raising an exception, allowing the model to smoothly rely on franchise strength and toss decision.
