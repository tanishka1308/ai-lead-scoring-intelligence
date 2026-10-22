from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_missing_required_fields_are_rejected():
    response = client.post("/predict", json={"industry": "Technology"})
    assert response.status_code == 422


def test_batch_requires_at_least_one_lead():
    response = client.post("/predict/batch", json={"leads": []})
    assert response.status_code == 422


def test_invalid_numeric_range_is_rejected():
    response = client.post("/predict", json={
        "company_size": "1-50", "industry": "Technology", "source": "Referral", "country": "US",
        "days_since_contact": -1, "website_visits": 1, "emails_opened": 1, "meeting_attended": False,
    })
    assert response.status_code == 422
