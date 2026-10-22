# AFLIX Lead Scoring System

An end-to-end demo that ranks sales leads with a reproducible scikit-learn pipeline, exposes single and batch predictions through FastAPI, stores predictions in SQLite, and provides a Streamlit dashboard.

## Important score interpretation

The 0–100 score is the model's estimated conversion probability multiplied by 100. It is **not claimed to be calibrated**. The training script compares Logistic Regression and Random Forest using a held-out stratified test set and reports ROC-AUC, precision, recall, F1, and accuracy. The winning model is selected by ROC-AUC. Calibration can be added only after validation with a separate calibration procedure and appropriate reliability metrics.

## Project layout

```text
lead_scoring_system/
  app/                 API, database, shared scoring functions
  dashboard/           Streamlit dashboard
  data/                 Deterministic synthetic sample data (1,200 leads)
  artifacts/            Generated model and evaluation report
  tests/                Example API and scoring tests
  generate_data.py      Regenerate sample data
  train.py              Train, compare, and save model/evaluation
  requirements.txt
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python train.py
```

Training writes `artifacts/model.joblib` and `artifacts/evaluation.json`. The provided dataset is synthetic and contains no personal information.

## Run the API and dashboard

In one terminal:

```bash
uvicorn app.main:app --reload
```

Interactive API documentation: <http://127.0.0.1:8000/docs>

In another terminal:

```bash
streamlit run dashboard/app.py
```

Set `LEAD_DB_PATH` to change the SQLite file location. Set `MODEL_VERSION` to tag a trained model deployment (defaults to `lead-scorer-v1`).

## API examples

Single lead:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"company_size": "51-200", "industry": "Technology", "source": "Referral", "country": "US", "days_since_contact": 3, "website_visits": 7, "emails_opened": 4, "meeting_attended": true}'
```

Batch (up to 500 leads per request):

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"leads": [{"company_size":"1-50","industry":"Retail","source":"Webinar","country":"IN","days_since_contact":14,"website_visits":2,"emails_opened":1,"meeting_attended":false}]}'
```

Retrieve saved predictions with `GET /predictions?limit=100`. Invalid fields return a clear HTTP 422 response; unknown categories are accepted by the preprocessing pipeline. CSV upload in the dashboard expects the same feature columns as the sample CSV and downloads scored rows as `lead_scores.csv`.

## Data and leakage prevention

The deterministic generator creates 1,200 records with eight input features and a binary `converted` target. The target is generated from a noisy synthetic rule. The target column is excluded from the feature list. A stratified 80/20 split is created before fitting any imputation, scaling, or one-hot encoding. Those transformations are inside a scikit-learn Pipeline and are learned from training data only. Duplicate rows are removed during training, missing numeric values are median-imputed, and missing categoricals are most-frequent-imputed.

## Architecture and model choice

The training script compares Logistic Regression (simple baseline) and Random Forest (nonlinear model) under the same preprocessing and split. It selects the model with the higher test ROC-AUC, then persists the full pipeline. ROC-AUC evaluates ranking across thresholds; precision, recall, F1, and accuracy are also reported at the default 0.5 threshold. The priority labels use score cutoffs: High ≥ 70, Medium ≥ 40, otherwise Low. These thresholds are practical demo defaults, not business-optimized cutoffs.

FastAPI validates request shape with Pydantic, calls the shared scoring layer, and stores input features, score, priority, timestamp, and model version in SQLite. Streamlit reads stored predictions and evaluation metrics, displays summary charts, and supports bulk CSV scoring.

## Tests and edge cases

Example tests are in `tests/`. Run them with `pytest`. They cover a valid score, malformed input, batch validation, and persistence. Missing required features are rejected by request validation; missing values in the training data are handled by the pipeline. Duplicate training records are dropped. Scores are bounded to [0, 100].

## Review question notes

1. A 95% accuracy model that predicts almost all leads Low may be exploiting class imbalance. Inspect the confusion matrix, class distribution, recall/precision for converted leads, F1, PR-AUC, and compare with a majority-class baseline; choose a threshold based on sales capacity and the relative cost of missed conversions.
2. A 99% train score and 65% test score suggests overfitting or leakage. Verify split boundaries and preprocessing, check duplicates/entity overlap, feature leakage, distribution shift, model complexity, and data quality; use cross-validation and regularization.
3. A score can be `100 × predicted probability`. It is a calibrated probability only if calibration is evaluated and corrected on held-out data (for example, with Platt scaling or isotonic regression), with reliability plots/Brier score and no leakage from the final test set.
4. At one million leads, profile first; score in chunks, use vectorized inference and bulk database writes, add indexes and pagination, and move to a managed database/queue if throughput requires it. Avoid loading all CSV rows into memory.
5. Retrain on a schedule or triggered by drift/performance monitoring. Save model artifacts with immutable version identifiers and metadata, retain a champion/rollback path, log prediction distributions and delayed outcomes, and compare on a time-based holdout before promotion.

## Limitations

This is an assessment demo, not a production sales model. Synthetic patterns do not establish real-world performance. Before deployment, replace the data, define business costs and score thresholds with sales stakeholders, assess fairness and privacy obligations, calibrate if probabilities are needed, and monitor drift and outcomes.
