"""Framework-independent LeadScorer component for single, batch, and dataframe scoring."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid
import joblib
import numpy as np
import pandas as pd

from lead_scoring.config import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PRIORITY_HIGH_THRESHOLD,
    PRIORITY_MEDIUM_THRESHOLD,
)
from lead_scoring.validation import validate_record


class LeadScorer:
    """Production lead scorer wrapping scikit-learn pipeline and decision thresholds."""

    def __init__(self, model_path: Optional[Path | str] = None):
        path = Path(model_path) if model_path else MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {path}. Please train the model first using "
                f"'python -m lead_scoring.train'."
            )
        self.bundle: Dict[str, Any] = joblib.load(path)
        self.pipeline = self.bundle["pipeline"]
        self.threshold: float = float(self.bundle.get("threshold", 0.5))
        self.priority_thresholds = self.bundle.get(
            "priority_thresholds",
            {"high": PRIORITY_HIGH_THRESHOLD, "medium": PRIORITY_MEDIUM_THRESHOLD},
        )
        self.is_calibrated: bool = bool(self.bundle.get("is_calibrated", False))
        self.ece: float = float(self.bundle.get("ece", 1.0))
        self.model_version: str = str(self.bundle.get("model_version", "unknown"))
        self.model_name: str = str(self.bundle.get("model_name", "unknown"))

    def _determine_priority(self, score: float) -> str:
        high = self.priority_thresholds.get("high", PRIORITY_HIGH_THRESHOLD)
        med = self.priority_thresholds.get("medium", PRIORITY_MEDIUM_THRESHOLD)
        if score >= high:
            return "High"
        if score >= med:
            return "Medium"
        return "Low"

    def score_records(
        self, records: List[Dict[str, Any]], strict: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Score a list of raw record dictionaries.

        Args:
            records: List of dictionaries representing lead inputs.
            strict: If True, raises ValueError if any record is invalid.

        Returns:
            Tuple of (results, errors)
        """
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        valid_items: List[Tuple[int, Dict[str, Any], str]] = []

        for idx, rec in enumerate(records):
            clean_rec, rec_errors = validate_record(rec)
            lead_id = clean_rec.get("lead_id") or f"LEAD-AUTO-{uuid.uuid4().hex[:8].upper()}"
            clean_rec["lead_id"] = lead_id

            if rec_errors:
                err_payload = {
                    "index": idx,
                    "lead_id": lead_id,
                    "errors": rec_errors,
                    "record": rec,
                }
                errors.append(err_payload)
            else:
                valid_items.append((idx, clean_rec, lead_id))

        if strict and errors:
            raise ValueError(
                f"Strict scoring failed: {len(errors)} record(s) failed validation. "
                f"Details: {errors}"
            )

        if not valid_items:
            return results, errors

        # Convert valid records to DataFrame with exact feature columns
        df_valid = pd.DataFrame([item[1] for item in valid_items])
        
        # Ensure all expected columns exist
        for col in ALL_FEATURE_COLUMNS:
            if col not in df_valid.columns:
                df_valid[col] = np.nan

        X = df_valid[ALL_FEATURE_COLUMNS]
        probs = self.pipeline.predict_proba(X)[:, 1]

        for (orig_idx, clean_rec, lead_id), prob in zip(valid_items, probs):
            score = round(float(prob * 100.0), 1)
            priority = self._determine_priority(score)
            calibrated_prob = round(float(prob), 4) if self.is_calibrated else None

            results.append({
                "index": orig_idx,
                "lead_id": lead_id,
                "score": score,
                "priority": priority,
                "calibrated_probability": calibrated_prob,
                "model_version": self.model_version,
                "details": clean_rec,
            })

        # Keep results sorted by original index
        results.sort(key=lambda r: r["index"])
        return results, errors

    def score_dataframe(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Score an entire pandas DataFrame.
        Keeps all rows (including invalid ones) and adds a 'validation_error' column.

        Returns:
            Tuple of (output_df, errors_list)
        """
        records = df.to_dict(orient="records")
        results, errors = self.score_records(records, strict=False)

        output_df = df.copy()

        # Add or populate target columns
        if "lead_id" not in output_df.columns:
            output_df["lead_id"] = [
                f"LEAD-AUTO-{uuid.uuid4().hex[:8].upper()}" for _ in range(len(output_df))
            ]
        else:
            # Fill missing lead_ids
            null_id_mask = output_df["lead_id"].isna() | (output_df["lead_id"].astype(str).str.strip() == "")
            output_df.loc[null_id_mask, "lead_id"] = [
                f"LEAD-AUTO-{uuid.uuid4().hex[:8].upper()}" for _ in range(int(null_id_mask.sum()))
            ]

        output_df["score"] = np.nan
        output_df["priority"] = None
        output_df["calibrated_probability"] = None
        output_df["model_version"] = self.model_version
        output_df["validation_error"] = None

        # Populate valid results
        for r in results:
            idx = r["index"]
            output_df.at[idx, "lead_id"] = r["lead_id"]
            output_df.at[idx, "score"] = r["score"]
            output_df.at[idx, "priority"] = r["priority"]
            output_df.at[idx, "calibrated_probability"] = r["calibrated_probability"]

        # Populate invalid rows
        for err in errors:
            idx = err["index"]
            output_df.at[idx, "validation_error"] = "; ".join(err["errors"])

        return output_df, errors
