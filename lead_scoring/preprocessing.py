"""Stateless cleaning and leakage-free sklearn preprocessing pipeline."""
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lead_scoring.config import (
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def clean_raw_data(
    df: pd.DataFrame, is_training: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Perform stateless data cleaning before splitting.

    Rules:
    - Normalise categories (strip whitespace, lowercase)
    - Coerce numerics to float/int
    - Replace impossible/out-of-range values with NaN
    - Cross-field: if email_clicks > emails_opened, set email_clicks to NaN
    - Drop exact duplicate rows and duplicate lead_ids
    - Drop rows with missing target (when target present and is_training=True)

    Returns:
        Tuple of (cleaned_df, report_dict)
    """
    initial_rows = len(df)
    report: Dict[str, Any] = {
        "initial_rows": initial_rows,
        "exact_duplicates_dropped": 0,
        "duplicate_lead_ids_dropped": 0,
        "missing_targets_dropped": 0,
        "out_of_range_values_nulled": 0,
        "final_rows": 0,
    }

    clean_df = df.copy()

    # 1. Drop exact duplicates
    before_exact = len(clean_df)
    clean_df = clean_df.drop_duplicates()
    report["exact_duplicates_dropped"] = before_exact - len(clean_df)

    # 2. Drop duplicate lead_id if column exists
    if ID_COLUMN in clean_df.columns:
        before_id = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=[ID_COLUMN], keep="first")
        report["duplicate_lead_ids_dropped"] = before_id - len(clean_df)

    # 3. Drop missing target rows
    if TARGET_COLUMN in clean_df.columns and is_training:
        before_target = len(clean_df)
        clean_df = clean_df.dropna(subset=[TARGET_COLUMN])
        # Also ensure target is binary 0 or 1
        clean_df = clean_df[clean_df[TARGET_COLUMN].isin([0, 1, 0.0, 1.0])]
        clean_df[TARGET_COLUMN] = clean_df[TARGET_COLUMN].astype(int)
        report["missing_targets_dropped"] = before_target - len(clean_df)

    # 4. Normalise categorical features
    for cat_col in CATEGORICAL_FEATURES:
        if cat_col in clean_df.columns:
            clean_df[cat_col] = (
                clean_df[cat_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": np.nan, "none": np.nan, "<na>": np.nan, "": np.nan})
            )

    # 5. Coerce numerics and replace out-of-range values with NaN
    nulled_count = 0
    for num_col in NUMERIC_FEATURES:
        if num_col in clean_df.columns:
            clean_df[num_col] = pd.to_numeric(clean_df[num_col], errors="coerce")

            # Out of range checks
            if num_col == "company_size":
                invalid_mask = clean_df[num_col] < 1
            elif num_col == "demo_requested":
                invalid_mask = ~clean_df[num_col].isin([0, 1])
            else:
                invalid_mask = clean_df[num_col] < 0

            nulled_count += int(invalid_mask.sum())
            clean_df.loc[invalid_mask, num_col] = np.nan

    # Cross-field rule: email_clicks <= emails_opened
    if "email_clicks" in clean_df.columns and "emails_opened" in clean_df.columns:
        cross_mask = clean_df["email_clicks"] > clean_df["emails_opened"]
        nulled_count += int(cross_mask.sum())
        clean_df.loc[cross_mask, "email_clicks"] = np.nan

    report["out_of_range_values_nulled"] = nulled_count
    report["final_rows"] = len(clean_df)

    return clean_df.reset_index(drop=True), report


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform stratified train/test split. Target and lead_id are excluded from features.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

    feature_cols = [c for c in df.columns if c in NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy().astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def build_preprocessor() -> ColumnTransformer:
    """Construct sklearn ColumnTransformer for numeric and categorical pipelines.

    Zero leakage: Imputation, missing indicators, scaling, and one-hot encoding
    are encapsulated inside the pipeline to be fitted exclusively on training data.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor
