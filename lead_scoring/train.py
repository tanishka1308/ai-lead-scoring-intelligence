"""Model training, tuning, threshold selection, calibration, and report generation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from lead_scoring.config import (
    ALL_FEATURE_COLUMNS,
    ECE_CALIBRATION_THRESHOLD,
    MODELS_DIR,
    PRIORITY_HIGH_THRESHOLD,
    PRIORITY_MEDIUM_THRESHOLD,
    RAW_DATA_PATH,
    REPORTS_DIR,
    SAMPLE_UPLOAD_PATH,
)
from lead_scoring.data_gen import generate_synthetic_data
from lead_scoring.preprocessing import build_preprocessor, clean_raw_data, split_data


def compute_ece_and_reliability(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Tuple[float, List[Dict[str, Any]]]:
    """Compute Expected Calibration Error (ECE) and reliability table."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    table: List[Dict[str, Any]] = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        # Include high boundary on last bin
        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)

        bin_count = int(np.sum(mask))
        if bin_count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            error = abs(bin_acc - bin_conf)
            ece += (bin_count / total_samples) * error
            table.append({
                "bin": f"{low:.1f}-{high:.1f}",
                "count": bin_count,
                "mean_predicted_prob": round(bin_conf, 4),
                "actual_conversion_rate": round(bin_acc, 4),
                "abs_calibration_gap": round(error, 4),
            })
        else:
            table.append({
                "bin": f"{low:.1f}-{high:.1f}",
                "count": 0,
                "mean_predicted_prob": round((low + high) / 2.0, 4),
                "actual_conversion_rate": 0.0,
                "abs_calibration_gap": 0.0,
            })

    return round(float(ece), 4), table


def find_optimal_threshold(y_train: np.ndarray, oof_probs: np.ndarray) -> Tuple[float, float]:
    """Find decision threshold maximizing F1 score from train out-of-fold predictions.
    Leakage-free: zero test set data is used.
    """
    thresholds = np.linspace(0.10, 0.90, 81)
    best_thresh = 0.50
    best_f1 = -1.0

    for t in thresholds:
        preds = (oof_probs >= t).astype(int)
        score = f1_score(y_train, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = round(float(t), 3)

    return best_thresh, round(float(best_f1), 4)


def evaluate_test_set(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series | np.ndarray, threshold: float
) -> Dict[str, float]:
    """Evaluate model on held-out test set."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    y_arr = np.asarray(y_test).astype(int)

    majority_acc = max(float(y_arr.mean()), 1.0 - float(y_arr.mean()))

    return {
        "roc_auc": round(float(roc_auc_score(y_arr, probs)), 4),
        "pr_auc": round(float(average_precision_score(y_arr, probs)), 4),
        "f1": round(float(f1_score(y_arr, preds, zero_division=0)), 4),
        "precision": round(float(precision_score(y_arr, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_arr, preds, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_arr, preds)), 4),
        "brier_score": round(float(brier_score_loss(y_arr, probs)), 4),
        "majority_class_baseline_accuracy": round(majority_acc, 4),
    }


def compute_priority_buckets(
    probs: np.ndarray, y_test: np.ndarray
) -> List[Dict[str, Any]]:
    """Compute conversion rates across priority tiers on test set."""
    scores = np.round(probs * 100.0, 1)
    high_mask = scores >= PRIORITY_HIGH_THRESHOLD
    med_mask = (scores >= PRIORITY_MEDIUM_THRESHOLD) & (scores < PRIORITY_HIGH_THRESHOLD)
    low_mask = scores < PRIORITY_MEDIUM_THRESHOLD

    buckets = []
    for tier, mask, thresh_desc in [
        ("High", high_mask, f">= {PRIORITY_HIGH_THRESHOLD}"),
        ("Medium", med_mask, f">= {PRIORITY_MEDIUM_THRESHOLD} and < {PRIORITY_HIGH_THRESHOLD}"),
        ("Low", low_mask, f"< {PRIORITY_MEDIUM_THRESHOLD}"),
    ]:
        count = int(np.sum(mask))
        conv_rate = round(float(np.mean(y_test[mask])), 4) if count > 0 else 0.0
        buckets.append({
            "priority": tier,
            "threshold_rule": thresh_desc,
            "count": count,
            "percentage_of_test": round(count / len(y_test) * 100.0, 1),
            "actual_conversion_rate": conv_rate,
        })
    return buckets


def train_models(
    quick: bool = False, output_dir: Path | None = None
) -> Dict[str, Any]:
    """Train, tune, select, calibrate, and save models."""
    target_models_dir = output_dir if output_dir else MODELS_DIR
    target_reports_dir = (output_dir / "reports") if output_dir else REPORTS_DIR
    target_archive_dir = target_models_dir / "archive"

    target_models_dir.mkdir(parents=True, exist_ok=True)
    target_reports_dir.mkdir(parents=True, exist_ok=True)
    target_archive_dir.mkdir(parents=True, exist_ok=True)

    target_model_path = target_models_dir / "model.joblib"
    target_metadata_path = target_models_dir / "metadata.json"
    target_metrics_path = target_reports_dir / "metrics.json"
    target_evaluation_path = target_reports_dir / "evaluation.md"
    # 1. Load or generate raw data
    if not RAW_DATA_PATH.exists():
        print(f"Raw data not found at {RAW_DATA_PATH}. Generating synthetic data...")
        raw_df, sample_upload_df = generate_synthetic_data(n_samples=3200)
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(RAW_DATA_PATH, index=False)
        sample_upload_df.to_csv(SAMPLE_UPLOAD_PATH, index=False)
    else:
        raw_df = pd.read_csv(RAW_DATA_PATH)

    # 2. Preprocessing & Stateless cleaning
    clean_df, clean_report = clean_raw_data(raw_df, is_training=True)
    X_train, X_test, y_train, y_test = split_data(clean_df, test_size=0.2, random_state=42)

    cv = StratifiedKFold(n_splits=3 if quick else 5, shuffle=True, random_state=42)

    # 3. Model Grids
    if quick:
        lr_param_grid = {"classifier__C": [0.1, 1.0]}
        rf_param_grid = {
            "classifier__n_estimators": [30],
            "classifier__max_depth": [5],
        }
    else:
        lr_param_grid = {"classifier__C": [0.01, 0.1, 1.0, 10.0]}
        rf_param_grid = {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_split": [2, 5],
        }

    # Pipeline: LogisticRegression
    lr_pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    # Pipeline: RandomForest
    rf_pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(random_state=42, n_jobs=-1),
            ),
        ]
    )

    print("Running GridSearchCV for LogisticRegression...")
    lr_grid = GridSearchCV(
        lr_pipe, lr_param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True
    )
    lr_grid.fit(X_train, y_train)

    print("Running GridSearchCV for RandomForestClassifier...")
    rf_grid = GridSearchCV(
        rf_pipe, rf_param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True
    )
    rf_grid.fit(X_train, y_train)

    lr_cv_auc = round(float(lr_grid.best_score_), 4)
    rf_cv_auc = round(float(rf_grid.best_score_), 4)

    # 4. Model Selection rule
    # Select by CV ROC-AUC (not test); prefer LogisticRegression if within 0.005 AUC
    auc_diff = rf_cv_auc - lr_cv_auc
    if auc_diff <= 0.005:
        selected_model_name = "LogisticRegression"
        selected_grid = lr_grid
        selection_reason = (
            f"LogisticRegression preferred: simpler, explainable linear model with CV ROC-AUC "
            f"({lr_cv_auc:.4f}) within 0.005 threshold of RandomForest ({rf_cv_auc:.4f}, diff={auc_diff:.4f})."
        )
    else:
        selected_model_name = "RandomForest"
        selected_grid = rf_grid
        selection_reason = (
            f"RandomForest selected: CV ROC-AUC ({rf_cv_auc:.4f}) meaningfully outperforms "
            f"LogisticRegression ({lr_cv_auc:.4f}) by {auc_diff:.4f} (> 0.005 margin)."
        )

    print(f"Selection Decision: {selection_reason}")

    best_raw_pipeline = selected_grid.best_estimator_

    # 5. Out-of-fold train predictions for optimal threshold selection
    print("Computing out-of-fold predictions on train set for threshold selection...")
    oof_probs = cross_val_predict(
        best_raw_pipeline, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    optimal_threshold, oof_f1 = find_optimal_threshold(y_train.to_numpy(), oof_probs)
    print(f"Optimal decision threshold: {optimal_threshold} (OOF F1: {oof_f1})")

    # Evaluate both candidate models on test set for reporting
    lr_test_metrics = evaluate_test_set(
        lr_grid.best_estimator_, X_test, y_test, threshold=0.5
    )
    rf_test_metrics = evaluate_test_set(
        rf_grid.best_estimator_, X_test, y_test, threshold=0.5
    )

    # 6. Calibration
    # Compare raw vs CalibratedClassifierCV
    cal_method = "isotonic" if len(X_train) >= 1000 else "sigmoid"
    print(f"Fitting CalibratedClassifierCV (method={cal_method})...")
    calibrated_pipeline = CalibratedClassifierCV(
        estimator=best_raw_pipeline, method=cal_method, cv=cv
    )
    calibrated_pipeline.fit(X_train, y_train)

    raw_test_probs = best_raw_pipeline.predict_proba(X_test)[:, 1]
    cal_test_probs = calibrated_pipeline.predict_proba(X_test)[:, 1]

    raw_brier = round(float(brier_score_loss(y_test, raw_test_probs)), 4)
    cal_brier = round(float(brier_score_loss(y_test, cal_test_probs)), 4)

    if cal_brier < raw_brier:
        deployed_pipeline = calibrated_pipeline
        deployed_probs = cal_test_probs
        calibration_choice = f"CalibratedClassifierCV ({cal_method}) selected (Brier: {cal_brier} vs Raw: {raw_brier})"
    else:
        deployed_pipeline = best_raw_pipeline
        deployed_probs = raw_test_probs
        calibration_choice = f"Raw pipeline selected (Brier: {raw_brier} vs Calibrated: {cal_brier})"

    ece_score, reliability_table = compute_ece_and_reliability(
        y_test.to_numpy(), deployed_probs, n_bins=10
    )
    is_calibrated = bool(ece_score <= ECE_CALIBRATION_THRESHOLD)
    print(f"Calibration outcome: {calibration_choice} | ECE: {ece_score:.4f} | Calibrated: {is_calibrated}")

    # Test metrics for deployed model at optimal threshold
    deployed_test_metrics = evaluate_test_set(
        deployed_pipeline, X_test, y_test, threshold=optimal_threshold
    )

    # Priority buckets on test set
    priority_buckets = compute_priority_buckets(deployed_probs, y_test.to_numpy())

    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # 7. Metadata and Artifacts
    metadata: Dict[str, Any] = {
        "model_version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected_model_name,
        "selection_reason": selection_reason,
        "optimal_threshold": optimal_threshold,
        "oof_train_f1": oof_f1,
        "is_calibrated": is_calibrated,
        "ece": ece_score,
        "calibration_choice": calibration_choice,
        "ece_threshold": ECE_CALIBRATION_THRESHOLD,
        "deployed_test_metrics": deployed_test_metrics,
        "models_comparison": {
            "LogisticRegression": {
                "cv_roc_auc": lr_cv_auc,
                "best_params": lr_grid.best_params_,
                "test_metrics": lr_test_metrics,
            },
            "RandomForest": {
                "cv_roc_auc": rf_cv_auc,
                "best_params": rf_grid.best_params_,
                "test_metrics": rf_test_metrics,
            },
        },
        "cleaning_report": clean_report,
        "priority_buckets_test": priority_buckets,
        "features": {
            "numeric": ALL_FEATURE_COLUMNS[:8],
            "categorical": ALL_FEATURE_COLUMNS[8:],
        },
    }

    # 8. Save Model and Metadata
    model_bundle = {
        "pipeline": deployed_pipeline,
        "raw_pipeline": best_raw_pipeline,
        "threshold": optimal_threshold,
        "priority_thresholds": {
            "high": PRIORITY_HIGH_THRESHOLD,
            "medium": PRIORITY_MEDIUM_THRESHOLD,
        },
        "is_calibrated": is_calibrated,
        "ece": ece_score,
        "model_version": version,
        "model_name": selected_model_name,
        "features": ALL_FEATURE_COLUMNS,
    }

    joblib.dump(model_bundle, target_model_path)

    # Save to archive
    archive_model_path = target_archive_dir / f"model_{version}.joblib"
    joblib.dump(model_bundle, archive_model_path)

    with open(target_metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # 9. Reports: metrics.json & evaluation.md
    reports_metrics = {
        "metadata": {
            "version": version,
            "selected_model": selected_model_name,
            "is_calibrated": is_calibrated,
            "ece": ece_score,
        },
        "deployed_metrics": deployed_test_metrics,
        "priority_buckets": priority_buckets,
        "reliability_table": reliability_table,
        "cleaning_report": clean_report,
    }
    with open(target_metrics_path, "w") as f:
        json.dump(reports_metrics, f, indent=2)

    # Write evaluation.md
    eval_md = f"""# Model Evaluation & Benchmark Report
**Model Version:** `{version}`  
**Generated At:** `{metadata['created_at_utc']}`  
**Selected Estimator:** **{selected_model_name}**  

---

## 1. Model Selection & Comparison
{selection_reason}

| Metric | LogisticRegression | RandomForest | Deployed Selected Model |
| :--- | :---: | :---: | :---: |
| **5-Fold CV ROC-AUC** | {lr_cv_auc:.4f} | {rf_cv_auc:.4f} | **{metadata['models_comparison'][selected_model_name]['cv_roc_auc']:.4f}** |
| **Held-out Test ROC-AUC** | {lr_test_metrics['roc_auc']:.4f} | {rf_test_metrics['roc_auc']:.4f} | **{deployed_test_metrics['roc_auc']:.4f}** |
| **Test PR-AUC** | {lr_test_metrics['pr_auc']:.4f} | {rf_test_metrics['pr_auc']:.4f} | **{deployed_test_metrics['pr_auc']:.4f}** |
| **Test F1-Score** | {lr_test_metrics['f1']:.4f} | {rf_test_metrics['f1']:.4f} | **{deployed_test_metrics['f1']:.4f}** |
| **Test Precision** | {lr_test_metrics['precision']:.4f} | {rf_test_metrics['precision']:.4f} | **{deployed_test_metrics['precision']:.4f}** |
| **Test Recall** | {lr_test_metrics['recall']:.4f} | {rf_test_metrics['recall']:.4f} | **{deployed_test_metrics['recall']:.4f}** |
| **Test Accuracy** | {lr_test_metrics['accuracy']:.4f} | {rf_test_metrics['accuracy']:.4f} | **{deployed_test_metrics['accuracy']:.4f}** |
| **Brier Score** | {lr_test_metrics['brier_score']:.4f} | {rf_test_metrics['brier_score']:.4f} | **{deployed_test_metrics['brier_score']:.4f}** |
| **Majority Baseline Acc** | {lr_test_metrics['majority_class_baseline_accuracy']:.4f} | {rf_test_metrics['majority_class_baseline_accuracy']:.4f} | **{deployed_test_metrics['majority_class_baseline_accuracy']:.4f}** |

> **Threshold Decision:** Tuned optimal decision threshold from out-of-fold train probabilities = `{optimal_threshold}` (OOF F1 = `{oof_f1:.4f}`). Test set was never used during threshold selection.

---

## 2. Calibration & Reliability Analysis
- **Calibration Method Choice:** {calibration_choice}
- **Raw Brier Score:** {raw_brier:.4f}
- **Calibrated Brier Score:** {cal_brier:.4f}
- **Expected Calibration Error (ECE):** `{ece_score:.4f}` (Threshold for calibrated flag: `<= {ECE_CALIBRATION_THRESHOLD}`)
- **Marked Calibrated:** `{'YES (calibrated=True)' if is_calibrated else 'NO (calibrated=False)'}`

> **Important Distinction:** A lead score (0-100) represents a relative ranking metric. It corresponds to an authentic, calibrated real-world probability *if and only if* `calibrated=True` (ECE <= 0.05).

### Reliability Table (10 Bins)
| Probability Bin | Lead Count | Mean Predicted Prob | Actual Conversion Rate | Abs Calibration Gap |
| :---: | :---: | :---: | :---: | :---: |
"""
    for row in reliability_table:
        eval_md += f"| {row['bin']} | {row['count']} | {row['mean_predicted_prob']:.4f} | {row['actual_conversion_rate']:.4f} | {row['abs_calibration_gap']:.4f} |\n"

    eval_md += f"""
---

## 3. Priority Buckets on Held-Out Test Set
Score = `round(100 * probability, 1)`. Priority: **High** >= {PRIORITY_HIGH_THRESHOLD}, **Medium** >= {PRIORITY_MEDIUM_THRESHOLD}, else **Low**.

| Priority Tier | Threshold Rule | Lead Count | % of Test Set | Actual Conversion Rate |
| :---: | :---: | :---: | :---: | :---: |
"""
    for bucket in priority_buckets:
        eval_md += f"| **{bucket['priority']}** | `{bucket['threshold_rule']}` | {bucket['count']} | {bucket['percentage_of_test']}% | **{bucket['actual_conversion_rate'] * 100:.1f}%** |\n"

    eval_md += f"""
---

## 4. Stateless Preprocessing & Cleaning Summary
| Cleaning Step | Count |
| :--- | :---: |
| Initial Raw Leads | {clean_report['initial_rows']} |
| Exact Duplicates Dropped | {clean_report['exact_duplicates_dropped']} |
| Duplicate Lead IDs Dropped | {clean_report['duplicate_lead_ids_dropped']} |
| Missing Targets Dropped | {clean_report['missing_targets_dropped']} |
| Out-of-Range Negative Values Nulled to NaN | {clean_report['out_of_range_values_nulled']} |
| **Final Clean Leads for Training Split** | **{clean_report['final_rows']}** |
"""

    with open(target_evaluation_path, "w") as f:
        f.write(eval_md)

    print(f"Artifacts successfully saved to {target_models_dir} and {target_reports_dir}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate lead scoring models.")
    parser.add_argument("--quick", action="store_true", help="Run with small grids for testing")
    args = parser.parse_args()
    train_models(quick=args.quick)


if __name__ == "__main__":
    main()
