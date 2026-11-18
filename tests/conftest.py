"""Pytest fixtures for unit and integration testing."""

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
from local_llm_router.models import (
    ChatCompletionRequest,
    ChatMessage,
)
from local_llm_router.providers.mock import MockProvider


@pytest.fixture
def sample_messages():
    return [
        ChatMessage(role="system", content="You are a helpful coding assistant."),
        ChatMessage(role="user", content="Write a python function to add two numbers."),
    ]


@pytest.fixture
def sample_request(sample_messages):
    return ChatCompletionRequest(
        model="slm",
        messages=sample_messages,
        temperature=0.7,
    )


@pytest.fixture
def mock_provider():
    return MockProvider(
        name="mock",
        default_model="mock-model",
        default_response="def add(a, b): return a + b",
    )


@pytest.fixture
def test_config():
    return AppConfig(
        server=ServerConfig(),
        cache=CacheConfig(
            enabled=True,
            db_path=":memory:",
            ttl_seconds=3600,
            max_entries=100,
            normalize_whitespace=True,
        ),
        validator=ValidatorConfig(
            enabled=True,
            max_retries=2,
            strip_markdown=True,
        ),
        routing=RoutingRuleConfig(
            default_route="slm",
            length_threshold_tokens=50,
            slm_target="mock:mock-slm",
            heavy_target="mock:mock-heavy",
            tags={
                "code": "mock:mock-coder",
                "fast": "slm",
            },
            aliases={
                "slm": "mock:mock-slm",
                "heavy": "mock:mock-heavy",
                "default": "mock:mock-slm",
            },
            fallback_chains={
                "default": ["mock:primary", "mock:backup", "mock:final"]
            },
        ),
        providers=[
            ProviderConfig(
                name="mock",
                type="mock",
                default_model="mock-model",
                models=["mock-model", "mock-slm", "mock-heavy", "mock-coder", "primary", "backup", "final"],
            )
        ],
    )


@pytest.fixture
def test_client(test_config):
    app = create_app(test_config)
    with TestClient(app) as client:
        yield client
