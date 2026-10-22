from fastapi import FastAPI, HTTPException, Query

from app.database import initialize, recent_predictions, save_prediction
from app.schema import BatchRequest, LeadFeatures
from app.scoring import score_leads

app = FastAPI(title="AFLIX Lead Scoring API", version="1.0.0")


@app.on_event("startup")
def startup() -> None:
    initialize()


def score_and_store(leads: list[LeadFeatures]) -> list[dict]:
    payloads = [lead.model_dump() for lead in leads]
    try:
        scored = score_leads(payloads)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    for payload, result in zip(payloads, scored):
        result["prediction_id"] = save_prediction(
            payload, result["score"], result["priority"], result["model_version"]
        )
    return scored


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(lead: LeadFeatures) -> dict:
    return score_and_store([lead])[0]


@app.post("/predict/batch")
def predict_batch(request: BatchRequest) -> dict:
    return {"count": len(request.leads), "results": score_and_store(request.leads)}


@app.get("/predictions")
def get_predictions(limit: int = Query(default=100, ge=1, le=1000)) -> dict:
    return {"results": recent_predictions(limit)}
