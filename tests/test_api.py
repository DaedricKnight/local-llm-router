"""Integration tests for FastAPI endpoints and OpenAI compatibility."""

from __future__ import annotations


def test_health_endpoint(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "mock" in data["providers"]


def test_models_endpoint(test_client):
    response = test_client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "slm" in ids
    assert "heavy" in ids


def test_chat_completions_caching_and_bypass(test_client):
    payload = {
        "model": "slm",
        "messages": [
            {"role": "user", "content": "Tell me a joke about caching."}
        ],
        "temperature": 0.5,
    }

    # 1. First call -> Cache MISS
    resp1 = test_client.post("/v1/chat/completions", json=payload)
    assert resp1.status_code == 200
    assert resp1.headers.get("x-cache") == "MISS"
    data1 = resp1.json()
    assert len(data1["choices"]) == 1

    # 2. Second identical call -> Cache HIT
    resp2 = test_client.post("/v1/chat/completions", json=payload)
    assert resp2.status_code == 200
    assert resp2.headers.get("x-cache") == "HIT"
    assert resp2.headers.get("x-router-provider") == "cache"
    assert int(resp2.headers.get("x-tokens-saved", 0)) > 0

    # 3. Third call with bypass header -> Cache MISS
    resp3 = test_client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-Cache-Bypass": "true"},
    )
    assert resp3.status_code == 200
    assert resp3.headers.get("x-cache") == "MISS"


def test_stats_and_cache_clear(test_client):
    # Perform a request to generate stats
    test_client.post(
        "/v1/chat/completions",
        json={"model": "slm", "messages": [{"role": "user", "content": "ping"}]},
    )

    stats_resp = test_client.get("/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["total_requests"] >= 1

    # Clear cache
    clear_resp = test_client.post("/v1/router/cache/clear")
    assert clear_resp.status_code == 200
    assert clear_resp.json()["status"] == "ok"
