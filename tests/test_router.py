"""Unit tests for router engine, routing rules, and fallback policies."""

from __future__ import annotations

import pytest

from local_llm_router.config import RoutingRuleConfig
from local_llm_router.exceptions import AllProvidersFailedError
from local_llm_router.models import ChatCompletionRequest, ChatMessage
from local_llm_router.providers.mock import MockProvider
from local_llm_router.providers.registry import ProviderRegistry
from local_llm_router.router.engine import RouterEngine
from local_llm_router.router.rules import RoutingRulesEvaluator
from local_llm_router.telemetry.collector import TelemetryCollector


@pytest.fixture
def routing_config():
    return RoutingRuleConfig(
        default_route="slm",
        length_threshold_tokens=20,
        slm_target="mock:slm-model",
        heavy_target="mock:heavy-model",
        tags={"code": "mock:coder-model"},
        aliases={"slm": "mock:slm-model", "heavy": "mock:heavy-model"},
        fallback_chains={
            "default": ["mock:primary", "mock:backup"],
            "slm-model": ["mock:slm-model", "mock:heavy-model"],
        },
    )


def test_routing_rules_length(routing_config):
    evaluator = RoutingRulesEvaluator(routing_config)

    # Short prompt (<= 20 tokens)
    short_req = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="short text")],
    )
    dec_short = evaluator.evaluate(short_req, prompt_tokens=5)
    assert dec_short.rule_name == "length_slm"
    assert dec_short.target_model == "slm-model"

    # Long prompt (> 20 tokens)
    long_req = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="very long prompt " * 20)],
    )
    dec_long = evaluator.evaluate(long_req, prompt_tokens=40)
    assert dec_long.rule_name == "length_heavy"
    assert dec_long.target_model == "heavy-model"


def test_routing_rules_tag(routing_config):
    evaluator = RoutingRulesEvaluator(routing_config)
    req = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="fibonacci")],
        tags=["code"],
    )
    dec = evaluator.evaluate(req, prompt_tokens=10)
    assert dec.rule_name == "tag"
    assert dec.target_model == "coder-model"


def test_routing_rules_alias(routing_config):
    evaluator = RoutingRulesEvaluator(routing_config)
    req = ChatCompletionRequest(
        model="heavy",
        messages=[ChatMessage(role="user", content="solve logic puzzle")],
    )
    dec = evaluator.evaluate(req, prompt_tokens=10)
    assert dec.rule_name == "alias"
    assert dec.target_model == "heavy-model"


@pytest.mark.asyncio
async def test_router_engine_fallback_success(routing_config):
    registry = ProviderRegistry()
    mock_prov = MockProvider(
        name="mock",
        default_response="Backup succeeded",
    )
    # Fail first call (primary), then succeed on second call (backup)
    mock_prov.fail_times = 1
    mock_prov.failure_type = "timeout"
    registry.register(mock_prov)

    telemetry = TelemetryCollector()
    engine = RouterEngine(registry, routing_config, telemetry=telemetry)

    req = ChatCompletionRequest(
        model="slm",
        messages=[ChatMessage(role="user", content="test fallback")],
    )

    resp = await engine.execute(req)
    assert resp.choices[0].message.content == "Backup succeeded"
    assert len(resp.router_metadata.fallback_chain_used) == 2
    assert telemetry.get_stats().fallback_events_total == 1


@pytest.mark.asyncio
async def test_router_engine_all_fail(routing_config):
    registry = ProviderRegistry()
    mock_prov = MockProvider(name="mock")
    # Fail both calls
    mock_prov.fail_times = 5
    mock_prov.failure_type = "connection"
    registry.register(mock_prov)

    engine = RouterEngine(registry, routing_config)
    req = ChatCompletionRequest(
        model="slm",
        messages=[ChatMessage(role="user", content="failing test")],
    )

    with pytest.raises(AllProvidersFailedError) as exc_info:
        await engine.execute(req)

    assert len(exc_info.value.attempted_providers) >= 2
