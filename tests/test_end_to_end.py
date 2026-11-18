"""End-to-end scenario tests covering routing, caching, validation, and failover."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from local_llm_router.api.app import create_app
from local_llm_router.config import (
    AppConfig,
    CacheConfig,
    ProviderConfig,
    RoutingRuleConfig,
    ServerConfig,
    ValidatorConfig,
)
from local_llm_router.providers.mock import MockProvider


@pytest.fixture
def e2e_client():
    cfg = AppConfig(
        server=ServerConfig(),
        cache=CacheConfig(enabled=True, db_path=":memory:", ttl_seconds=3600),
        validator=ValidatorConfig(enabled=True, max_retries=2),
        routing=RoutingRuleConfig(
            default_route="slm",
            length_threshold_tokens=25,
            slm_target="mock:slm-small",
            heavy_target="mock:heavy-large",
            aliases={
                "slm": "mock:slm-small",
                "heavy": "mock:heavy-large",
                "default": "mock:slm-small",
            },
            fallback_chains={
                "slm-small": ["mock:slm-small", "mock:heavy-large"],
            },
        ),
        providers=[
            ProviderConfig(
                name="mock",
                type="mock",
                default_model="slm-small",
                models=["slm-small", "heavy-large"],
            )
        ],
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client, app


def test_end_to_end_full_flow(e2e_client):
    client, app = e2e_client

    # Configure mock responses
    mock_prov: MockProvider = app.state.registry.get("mock")
    mock_prov.set_response_for_model("slm-small", "SLM output")
    mock_prov.set_response_for_model("heavy-large", "Heavy output")

    # Step 1: Short request -> Length-based routing sends to slm-small
    short_resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert short_resp.status_code == 200
    assert short_resp.json()["choices"][0]["message"]["content"] == "SLM output"
    assert short_resp.headers["x-cache"] == "MISS"

    # Step 2: Repeat short request -> Cache HIT in 0.5ms!
    cached_resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert cached_resp.status_code == 200
    assert cached_resp.headers["x-cache"] == "HIT"
    assert cached_resp.json()["choices"][0]["message"]["content"] == "SLM output"

    # Step 3: Long request -> Length-based routing sends to heavy-large
    long_resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": "Please write a comprehensive architectural essay about routing " * 10,
                }
            ],
        },
    )
    assert long_resp.status_code == 200
    assert long_resp.json()["choices"][0]["message"]["content"] == "Heavy output"
    assert long_resp.headers["x-cache"] == "MISS"

    # Step 4: Check telemetry summary
    stats_resp = client.get("/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["total_requests"] == 3
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 2
    assert stats["total_tokens_saved"] > 0
