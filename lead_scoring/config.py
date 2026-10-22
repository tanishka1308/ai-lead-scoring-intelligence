"""Configuration and schema definitions for the AI Lead Scoring system."""
import os
from pathlib import Path
from typing import List, Set

# Base Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
ARCHIVE_DIR: Path = MODELS_DIR / "archive"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
RAW_DATA_PATH: Path = DATA_DIR / "leads_raw.csv"
SAMPLE_UPLOAD_PATH: Path = DATA_DIR / "sample_upload.csv"
DEFAULT_DB_PATH: Path = DATA_DIR / "leads.db"
MODEL_PATH: Path = MODELS_DIR / "model.joblib"
METADATA_PATH: Path = MODELS_DIR / "metadata.json"
METRICS_PATH: Path = REPORTS_DIR / "metrics.json"
EVALUATION_PATH: Path = REPORTS_DIR / "evaluation.md"


def get_db_path() -> Path:
    """Return the database path from environment variable LEAD_DB_PATH or default.
    Ensures parent directories exist.
    """
    env_path = os.getenv("LEAD_DB_PATH")
    if env_path:
        path = Path(env_path).resolve()
    else:
        path = DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# Feature Definitions
TARGET_COLUMN: str = "converted"
ID_COLUMN: str = "lead_id"

NUMERIC_FEATURES: List[str] = [
    "company_size",
    "website_visits",
    "pages_per_visit",
    "emails_opened",
    "email_clicks",
    "days_since_last_contact",
    "budget_usd",
    "demo_requested",
]

CATEGORICAL_FEATURES: List[str] = [
    "industry",
    "lead_source",
    "job_level",
    "country",
]

ALL_FEATURE_COLUMNS: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Validation Rules & Enums
REQUIRED_FIELDS: List[str] = [
    "lead_source",
    "industry",
    "job_level",
    "company_size",
    "website_visits",
    "demo_requested",
]

OPTIONAL_FIELDS: List[str] = [
    "pages_per_visit",
    "emails_opened",
    "email_clicks",
    "days_since_last_contact",
    "budget_usd",
    "country",
]

ALLOWED_INDUSTRIES: Set[str] = {
    "technology",
    "finance",
    "healthcare",
    "retail",
    "education",
    "manufacturing",
    "other",
}

ALLOWED_LEAD_SOURCES: Set[str] = {
    "organic_search",
    "paid_ad",
    "referral",
    "email_campaign",
    "social_media",
    "event",
    "other",
}

ALLOWED_JOB_LEVELS: Set[str] = {
    "c_level",
    "vp",
    "director",
    "manager",
    "individual_contributor",
    "other",
}

# Priority Tier Thresholds
# Score = round(100 * probability, 1)
# High: >= 60.0, Medium: >= 30.0 and < 60.0, Low: < 30.0
PRIORITY_HIGH_THRESHOLD: float = 60.0
PRIORITY_MEDIUM_THRESHOLD: float = 30.0

# Expected Calibration Error threshold for marking deployed model as calibrated
ECE_CALIBRATION_THRESHOLD: float = 0.05
