"""Synthetic lead dataset generator with realistic messy enterprise data."""
import argparse
import random
from typing import Tuple
import numpy as np
import pandas as pd
from lead_scoring.config import (
    ALLOWED_INDUSTRIES,
    ALLOWED_JOB_LEVELS,
    ALLOWED_LEAD_SOURCES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    SAMPLE_UPLOAD_PATH,
)

COMMON_COUNTRIES = [
    "United States",
    "United Kingdom",
    "Germany",
    "Canada",
    "France",
    "Australia",
    "India",
    "Singapore",
    "Netherlands",
    "Brazil",
]


def generate_synthetic_data(
    n_samples: int = 3200, random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate realistic messy synthetic leads for training and sample_upload.csv.

    Returns:
        Tuple of (messy_train_df, sample_upload_df)
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    # 1. Base clean features
    company_sizes = np.random.choice(
        [5, 15, 45, 120, 350, 800, 2500, 8000],
        size=n_samples,
        p=[0.20, 0.25, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02],
    )
    website_visits = np.random.negative_binomial(2, 0.25, size=n_samples)  # mean ~6
    pages_per_visit = np.round(np.random.gamma(3.0, 1.2, size=n_samples), 1)  # mean ~3.6
    emails_opened = np.random.poisson(4, size=n_samples)

    # Email clicks must be <= emails_opened
    email_clicks = np.array([
        np.random.binomial(opened, 0.35) if opened > 0 else 0
        for opened in emails_opened
    ])

    days_since_last_contact = np.random.exponential(45, size=n_samples).astype(int)
    days_since_last_contact = np.clip(days_since_last_contact, 0, 365)

    budget_usd = np.round(
        company_sizes * np.random.uniform(50, 400, size=n_samples)
        + np.random.exponential(5000, size=n_samples),
        -2,
    )

    demo_requested = np.random.choice([0, 1], size=n_samples, p=[0.78, 0.22])

    industries = np.random.choice(list(ALLOWED_INDUSTRIES), size=n_samples)
    lead_sources = np.random.choice(list(ALLOWED_LEAD_SOURCES), size=n_samples)
    job_levels = np.random.choice(list(ALLOWED_JOB_LEVELS), size=n_samples)
    countries = np.random.choice(COMMON_COUNTRIES, size=n_samples)

    # Weights for latent log-odds
    job_weights = {
        "c_level": 0.85,
        "vp": 0.65,
        "director": 0.40,
        "manager": 0.10,
        "individual_contributor": -0.35,
        "other": -0.20,
    }
    industry_weights = {
        "technology": 0.50,
        "finance": 0.40,
        "healthcare": 0.25,
        "manufacturing": 0.10,
        "retail": -0.15,
        "education": -0.30,
        "other": 0.0,
    }
    source_weights = {
        "referral": 0.90,
        "event": 0.50,
        "organic_search": 0.35,
        "email_campaign": 0.25,
        "paid_ad": 0.10,
        "social_media": -0.20,
        "other": 0.0,
    }

    job_eff = np.array([job_weights[j] for j in job_levels])
    ind_eff = np.array([industry_weights[i] for i in industries])
    src_eff = np.array([source_weights[s] for s in lead_sources])

    # Latent logit
    logit = (
        -3.5
        + 1.75 * demo_requested
        + 0.35 * np.log1p(company_sizes / 30.0)
        + 0.03 * website_visits
        + 0.05 * pages_per_visit
        + 1.10 * (email_clicks / (emails_opened + 1.0))
        - 0.007 * days_since_last_contact
        + 0.25 * np.log1p(budget_usd / 8000.0)
        + job_eff
        + ind_eff
        + src_eff
        + np.random.normal(0, 0.55, size=n_samples)
    )

    probs = 1.0 / (1.0 + np.exp(-logit))
    converted = np.random.binomial(1, probs)

    lead_ids = [f"LEAD-{10000 + i}" for i in range(n_samples)]

    df = pd.DataFrame({
        "lead_id": lead_ids,
        "lead_source": lead_sources,
        "industry": industries,
        "job_level": job_levels,
        "company_size": company_sizes,
        "website_visits": website_visits,
        "pages_per_visit": pages_per_visit,
        "emails_opened": emails_opened,
        "email_clicks": email_clicks,
        "days_since_last_contact": days_since_last_contact,
        "budget_usd": budget_usd,
        "demo_requested": demo_requested,
        "country": countries,
        "converted": converted,
    })

    # 2. Inject deliberate messiness
    df_messy = df.copy()

    # Mixed casing and whitespace in ~15% of categorical rows
    for cat in ["industry", "lead_source", "job_level"]:
        mask = np.random.rand(n_samples) < 0.15
        df_messy.loc[mask, cat] = df_messy.loc[mask, cat].apply(
            lambda x: f"  {x.upper()}  " if random.random() < 0.5 else f" {x.title()} "
        )

    # 2-5% missing values in selective columns
    cols_to_null = ["budget_usd", "days_since_last_contact", "pages_per_visit", "emails_opened", "country"]
    for col in cols_to_null:
        null_mask = np.random.rand(n_samples) < 0.035
        df_messy.loc[null_mask, col] = np.nan

    # A few impossible negative values (~10 rows)
    neg_indices = np.random.choice(n_samples, size=10, replace=False)
    for i, idx in enumerate(neg_indices):
        if i % 3 == 0:
            df_messy.loc[idx, "website_visits"] = - random.randint(1, 10)
        elif i % 3 == 1:
            df_messy.loc[idx, "budget_usd"] = - random.randint(100, 5000)
        else:
            df_messy.loc[idx, "company_size"] = - random.randint(1, 50)

    # A few missing targets (~15 rows)
    missing_target_idx = np.random.choice(n_samples, size=15, replace=False)
    df_messy.loc[missing_target_idx, "converted"] = np.nan

    # ~3% exact duplicate rows
    n_dupes = int(n_samples * 0.03)
    dupe_rows = df_messy.iloc[:n_dupes].copy()
    df_messy = pd.concat([df_messy, dupe_rows], ignore_index=True)

    # 3. Create sample_upload.csv: 60 rows, NO converted column, exactly 2 invalid rows
    sample_rows = df.iloc[100:158].copy().drop(columns=["converted"])
    # Reset IDs
    sample_rows["lead_id"] = [f"SAMPLE-{2000 + i}" for i in range(len(sample_rows))]

    # Deliberate invalid row 1: unknown category
    invalid_row_1 = pd.DataFrame([{
        "lead_id": "SAMPLE-INVALID-01",
        "lead_source": "referral",
        "industry": "astronomy_and_alchemy",  # UNKNOWN CATEGORY
        "job_level": "director",
        "company_size": 250,
        "website_visits": 8,
        "pages_per_visit": 4.5,
        "emails_opened": 6,
        "email_clicks": 3,
        "days_since_last_contact": 14,
        "budget_usd": 45000.0,
        "demo_requested": 1,
        "country": "Germany",
    }])

    # Deliberate invalid row 2: negative website visits
    invalid_row_2 = pd.DataFrame([{
        "lead_id": "SAMPLE-INVALID-02",
        "lead_source": "organic_search",
        "industry": "technology",
        "job_level": "manager",
        "company_size": 150,
        "website_visits": -7,  # NEGATIVE VISITS
        "pages_per_visit": 3.0,
        "emails_opened": 4,
        "email_clicks": 1,
        "days_since_last_contact": 30,
        "budget_usd": 20000.0,
        "demo_requested": 0,
        "country": "United States",
    }])

    # Concatenate so invalid rows are at positions 15 and 40
    sample_upload_df = pd.concat(
        [sample_rows.iloc[:15], invalid_row_1, sample_rows.iloc[15:39], invalid_row_2, sample_rows.iloc[39:]],
        ignore_index=True,
    )

    return df_messy, sample_upload_df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic lead scoring dataset.")
    parser.add_argument("--samples", type=int, default=3200, help="Number of samples to generate")
    args = parser.parse_args()

    train_df, sample_upload_df = generate_synthetic_data(n_samples=args.samples)

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(RAW_DATA_PATH, index=False)
    sample_upload_df.to_csv(SAMPLE_UPLOAD_PATH, index=False)

    valid_targets = train_df["converted"].dropna()
    pos_rate = valid_targets.mean() * 100

    print(f"Generated {len(train_df)} raw leads saved to {RAW_DATA_PATH}")
    print(f"Positive target conversion rate: {pos_rate:.1f}% ({int(valid_targets.sum())}/{len(valid_targets)})")
    print(f"Generated {len(sample_upload_df)} sample upload rows saved to {SAMPLE_UPLOAD_PATH}")


if __name__ == "__main__":
    main()
