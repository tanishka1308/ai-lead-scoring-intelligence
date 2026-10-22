"""Train and compare lead scoring models with leakage-safe preprocessing."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.scoring import FEATURES

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "leads.csv"
ARTIFACTS = ROOT / "artifacts"
TARGET = "converted"
CATEGORICAL = ["company_size", "industry", "source", "country", "meeting_attended"]
NUMERIC = ["days_since_contact", "website_visits", "emails_opened"]


def build_preprocessor() -> ColumnTransformer:
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([("categorical", categorical, CATEGORICAL), ("numeric", numeric, NUMERIC)])


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Dataset missing. Run `python generate_data.py` first.")
    data = pd.read_csv(DATA).drop_duplicates()
    X, y = data[FEATURES], data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }
    reports = {}
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        probability = pipeline.predict_proba(X_test)[:, 1]
        predicted = (probability >= 0.5).astype(int)
        reports[name] = {
            "precision": round(float(precision_score(y_test, predicted, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predicted, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predicted, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
            "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        }
        fitted[name] = pipeline
    winner = max(reports, key=lambda name: reports[name]["roc_auc"])
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(fitted[winner], ARTIFACTS / "model.joblib")
    result = {
        "dataset": "synthetic lead sample",
        "records_after_deduplication": int(len(data)),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "positive_rate": round(float(y.mean()), 4),
        "selection_metric": "roc_auc",
        "selected_model": winner,
        "calibration": "not evaluated; scores are ranking scores, not calibrated probabilities",
        "metrics_at_threshold_0_5": reports,
    }
    (ARTIFACTS / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
