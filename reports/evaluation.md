# Model Evaluation & Benchmark Report
**Model Version:** `20261022_060609`  
**Generated At:** `2026-10-22T06:06:09.260866+00:00`  
**Selected Estimator:** **LogisticRegression**  

---

## 1. Model Selection & Comparison
LogisticRegression preferred: simpler, explainable linear model with CV ROC-AUC (0.7658) within 0.005 threshold of RandomForest (0.7667, diff=0.0009).

| Metric | LogisticRegression | RandomForest | Deployed Selected Model |
| :--- | :---: | :---: | :---: |
| **5-Fold CV ROC-AUC** | 0.7658 | 0.7667 | **0.7658** |
| **Held-out Test ROC-AUC** | 0.7353 | 0.7336 | **0.7353** |
| **Test PR-AUC** | 0.5024 | 0.4941 | **0.5024** |
| **Test F1-Score** | 0.3241 | 0.2524 | **0.4836** |
| **Test Precision** | 0.6481 | 0.5909 | **0.4682** |
| **Test Recall** | 0.2160 | 0.1605 | **0.5000** |
| **Test Accuracy** | 0.7708 | 0.7582 | **0.7284** |
| **Brier Score** | 0.1642 | 0.1630 | **0.1642** |
| **Majority Baseline Acc** | 0.7457 | 0.7457 | **0.7457** |

> **Threshold Decision:** Tuned optimal decision threshold from out-of-fold train probabilities = `0.26` (OOF F1 = `0.5487`). Test set was never used during threshold selection.

---

## 2. Calibration & Reliability Analysis
- **Calibration Method Choice:** Raw pipeline selected (Brier: 0.1642 vs Calibrated: 0.1644)
- **Raw Brier Score:** 0.1642
- **Calibrated Brier Score:** 0.1644
- **Expected Calibration Error (ECE):** `0.0304` (Threshold for calibrated flag: `<= 0.05`)
- **Marked Calibrated:** `YES (calibrated=True)`

> **Important Distinction:** A lead score (0-100) represents a relative ranking metric. It corresponds to an authentic, calibrated real-world probability *if and only if* `calibrated=True` (ECE <= 0.05).

### Reliability Table (10 Bins)
| Probability Bin | Lead Count | Mean Predicted Prob | Actual Conversion Rate | Abs Calibration Gap |
| :---: | :---: | :---: | :---: | :---: |
| 0.0-0.1 | 23 | 0.0803 | 0.0435 | 0.0369 |
| 0.1-0.2 | 324 | 0.1576 | 0.1451 | 0.0125 |
| 0.2-0.3 | 144 | 0.2350 | 0.2847 | 0.0497 |
| 0.3-0.4 | 37 | 0.3551 | 0.3784 | 0.0233 |
| 0.4-0.5 | 55 | 0.4547 | 0.4364 | 0.0184 |
| 0.5-0.6 | 31 | 0.5485 | 0.6452 | 0.0967 |
| 0.6-0.7 | 11 | 0.6233 | 0.6364 | 0.0131 |
| 0.7-0.8 | 4 | 0.7390 | 0.5000 | 0.2390 |
| 0.8-0.9 | 6 | 0.8666 | 0.6667 | 0.2000 |
| 0.9-1.0 | 2 | 0.9364 | 1.0000 | 0.0636 |

---

## 3. Priority Buckets on Held-Out Test Set
Score = `round(100 * probability, 1)`. Priority: **High** >= 60.0, **Medium** >= 30.0, else **Low**.

| Priority Tier | Threshold Rule | Lead Count | % of Test Set | Actual Conversion Rate |
| :---: | :---: | :---: | :---: | :---: |
| **High** | `>= 60.0` | 23 | 3.6% | **65.2%** |
| **Medium** | `>= 30.0 and < 60.0` | 123 | 19.3% | **47.1%** |
| **Low** | `< 30.0` | 491 | 77.1% | **18.1%** |

---

## 4. Stateless Preprocessing & Cleaning Summary
| Cleaning Step | Count |
| :--- | :---: |
| Initial Raw Leads | 3296 |
| Exact Duplicates Dropped | 96 |
| Duplicate Lead IDs Dropped | 0 |
| Missing Targets Dropped | 15 |
| Out-of-Range Negative Values Nulled to NaN | 10 |
| **Final Clean Leads for Training Split** | **3185** |
