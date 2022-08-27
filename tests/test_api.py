"""FastAPI endpoint tests."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_nlp_minimal():
    r = client.post("/predict_nlp", json={"text": "I really enjoyed this movie"})
    assert r.status_code == 200
    body = r.json()
    assert body["sentiment"] in {"positive", "negative"}
    assert isinstance(body["sentiment_score"], float)
    assert "topic" in body
    assert isinstance(body["ner"], list)


def test_predict_nlp_rejects_empty():
    r = client.post("/predict_nlp", json={"text": ""})
    assert r.status_code == 422


def test_predict_vision_with_array():
    img = [[[0.0] * 8 for _ in range(8)] for _ in range(3)]   # 3x8x8
    r = client.post("/predict_vision", json={"image": img})
    assert r.status_code == 200
    body = r.json()
    assert "classification" in body
    assert body["seg_shape"] == [8, 8]


def test_predict_vision_rejects_bad_shape():
    img = [[[0.0] * 8 for _ in range(8)]]   # 1x8x8 -> reject
    r = client.post("/predict_vision", json={"image": img})
    assert r.status_code == 422


def test_predict_vision_rejects_no_payload():
    r = client.post("/predict_vision", json={})
    assert r.status_code == 422
