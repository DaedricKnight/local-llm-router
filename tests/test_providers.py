"""Unit tests for provider interfaces and adapters."""

from __future__ import annotations

import httpx
import pytest

from local_llm_router.config import ProviderConfig
from local_llm_router.exceptions import (
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderTokenLimitExceededError,
)
from local_llm_router.models import ChatCompletionRequest, ChatMessage
from local_llm_router.providers.llama_cpp import LlamaCppProvider
from local_llm_router.providers.mock import MockProvider
from local_llm_router.providers.ollama import OllamaProvider
from local_llm_router.providers.openai_compatible import OpenAICompatibleProvider
from local_llm_router.providers.registry import (
    build_registry,
)


@pytest.mark.asyncio
async def test_mock_provider_success():
    provider = MockProvider(name="mock", default_response="Mock answer")
    req = ChatCompletionRequest(
        model="mock-model",
        messages=[ChatMessage(role="user", content="Hello")],
    )

    resp = await provider.complete(req)
    assert resp.choices[0].message.content == "Mock answer"
    assert resp.usage.prompt_tokens > 0
    assert resp.usage.completion_tokens > 0
    assert resp.router_metadata.provider == "mock"
    assert provider.call_count == 1
    assert await provider.health() is True


@pytest.mark.asyncio
async def test_mock_provider_failure_injection():
    # Test timeout
    p_timeout = MockProvider(fail_times=1, failure_type="timeout")
    req = ChatCompletionRequest(model="test", messages=[ChatMessage(role="user", content="Hi")])
    with pytest.raises(ProviderTimeoutError):
        await p_timeout.complete(req)
    # Next call should succeed
    succ = await p_timeout.complete(req)
    assert succ.choices[0].message.content == "Hello from MockProvider!"

    # Test rate limit
    p_rate = MockProvider(fail_times=1, failure_type="ratelimit")
    with pytest.raises(ProviderRateLimitError):
        await p_rate.complete(req)

    # Test token limit
    p_tokens = MockProvider(fail_times=1, failure_type="token_limit")
    with pytest.raises(ProviderTokenLimitExceededError):
        await p_tokens.complete(req)


@pytest.mark.asyncio
async def test_mock_provider_model_responses():
    provider = MockProvider()
    provider.set_response_for_model("model-a", "Response A")
    provider.set_response_for_model("model-b", "Response B")

    req_a = ChatCompletionRequest(model="model-a", messages=[ChatMessage(role="user", content="Test")])
    resp_a = await provider.complete(req_a)
    assert resp_a.choices[0].message.content == "Response A"

    req_b = ChatCompletionRequest(model="model-b", messages=[ChatMessage(role="user", content="Test")])
    resp_b = await provider.complete(req_b)
    assert resp_b.choices[0].message.content == "Response B"


@pytest.mark.asyncio
async def test_openai_compatible_provider_success():
    def mock_handler(request: httpx.Request):
        assert request.url.path == "/v1/chat/completions"
        content = {
            "id": "chatcmpl-123",
            "created": 1700000000,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "OpenAI Response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return httpx.Response(200, json=content)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    provider = OpenAICompatibleProvider(
        name="test-openai",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        client=client,
    )

    req = ChatCompletionRequest(model="gpt-4o", messages=[ChatMessage(role="user", content="Test")])
    resp = await provider.complete(req)

    assert resp.choices[0].message.content == "OpenAI Response"
    assert resp.usage.total_tokens == 15
    assert resp.router_metadata.provider == "test-openai"
    await provider.close()


@pytest.mark.asyncio
async def test_openai_compatible_provider_errors():
    def error_handler(request: httpx.Request):
        if "ratelimit" in str(request.url):
            return httpx.Response(429, text="Rate limit reached")
        elif "tokenlimit" in str(request.url):
            return httpx.Response(400, text="maximum context length exceeded")
        elif "servererror" in str(request.url):
            return httpx.Response(500, text="Internal server error")
        return httpx.Response(404, text="Not found")

    client = httpx.AsyncClient(transport=httpx.MockTransport(error_handler))

    # Rate limit test
    p1 = OpenAICompatibleProvider(name="rl", base_url="http://mock/ratelimit/v1", client=client)
    req = ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="hi")])
    with pytest.raises(ProviderRateLimitError):
        await p1.complete(req)

    # Token limit test
    p2 = OpenAICompatibleProvider(name="tl", base_url="http://mock/tokenlimit/v1", client=client)
    with pytest.raises(ProviderTokenLimitExceededError):
        await p2.complete(req)

    # 500 error test
    p3 = OpenAICompatibleProvider(name="se", base_url="http://mock/servererror/v1", client=client)
    with pytest.raises(ProviderConnectionError):
        await p3.complete(req)


@pytest.mark.asyncio
async def test_ollama_provider_health():
    def ollama_handler(request: httpx.Request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:1b"}]})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(ollama_handler))
    ollama = OllamaProvider(base_url="http://localhost:11434", client=client)

    assert await ollama.health() is True
    models = await ollama.list_local_models()
    assert "llama3.2:1b" in models
    await ollama.close()


@pytest.mark.asyncio
async def test_llama_cpp_provider_health():
    def llamacpp_handler(request: httpx.Request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(llamacpp_handler))
    llama_cpp = LlamaCppProvider(base_url="http://localhost:8080", client=client)

    assert await llama_cpp.health() is True
    await llama_cpp.close()


def test_provider_registry():
    configs = [
        ProviderConfig(name="mock1", type="mock", default_model="m1"),
        ProviderConfig(name="mock2", type="mock", default_model="m2"),
    ]
    registry = build_registry(configs)

    assert registry.get("mock1") is not None
    assert registry.get("mock2") is not None
    assert registry.get("nonexistent") is None
    assert len(registry.list_all()) == 2
