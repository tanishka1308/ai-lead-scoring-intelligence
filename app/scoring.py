import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

FEATURES = [
    "company_size", "industry", "source", "country", "days_since_contact",
    "website_visits", "emails_opened", "meeting_attended",
]
MODEL_PATH = Path(os.getenv("LEAD_MODEL_PATH", Path(__file__).resolve().parents[1] / "artifacts" / "model.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "lead-scorer-v1")


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run `python train.py` first.")
    return joblib.load(MODEL_PATH)


def priority_for(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def score_leads(leads: list[dict]) -> list[dict]:
    frame = pd.DataFrame(leads, columns=FEATURES)
    probabilities = load_model().predict_proba(frame)[:, 1]
    results = []
    for lead, probability in zip(leads, probabilities):
        score = round(float(probability) * 100, 2)
        results.append({"score": score, "priority": priority_for(score), "model_version": MODEL_VERSION})
    return results
