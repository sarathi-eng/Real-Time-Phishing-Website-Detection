from fastapi.testclient import TestClient

import main


def test_health_endpoint_returns_standard_response():
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "data" in body
    assert body["data"]["service_status"] == "healthy"


def test_predict_endpoint_requires_valid_url():
    client = TestClient(main.app)

    response = client.post("/predict", json={"url": "   "})

    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "VALIDATION_ERROR"


def test_predict_endpoint_returns_standard_response_with_request_id(monkeypatch):
    client = TestClient(main.app)

    async def fake_predict(url, background_tasks):
        return {
            "url": url,
            "final_verdict": "SAFE (Green)",
            "confidence_score": 0.01,
            "analysis_depth": "lexical_fast_path",
            "decision_metadata": {},
            "decision_reason": "test",
            "raw_features": {},
            "degraded_mode": False,
            "cache_hit": False,
            "processing_time_ms": 1.0,
        }

    monkeypatch.setattr(main.hybrid_decision, "predict_with_fast_path", fake_predict)

    response = client.post(
        "/predict",
        headers={"x-request-id": "req-123"},
        json={"url": "example.com"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["meta"]["request_id"] == "req-123"
    assert body["data"]["url"] == "http://example.com"
