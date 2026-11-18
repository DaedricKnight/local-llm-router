"""API routes: OpenAI-compatible /v1/chat/completions, /v1/models, and proxy telemetry."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, Response

from local_llm_router.api.dependencies import (
    get_cache,
    get_config,
    get_registry,
    get_router,
    get_telemetry,
    get_validator,
)
from local_llm_router.cache.key import compute_cache_key
from local_llm_router.cache.sqlite_cache import SQLiteCache
from local_llm_router.config import AppConfig
from local_llm_router.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelCard,
    ModelListResponse,
    TelemetryStats,
)
from local_llm_router.providers.registry import ProviderRegistry
from local_llm_router.router.engine import RouterEngine
from local_llm_router.telemetry.collector import TelemetryCollector
from local_llm_router.validator.loop import StructuredOutputLoop

router = APIRouter()


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    response_model_exclude_none=True,
    summary="OpenAI-compatible Chat Completions endpoint",
)
async def chat_completions(
    request: ChatCompletionRequest,
    response: Response,
    cache_control: Optional[str] = Header(None),
    x_cache_bypass: Optional[str] = Header(None),
    x_task_tag: Optional[str] = Header(None),
    x_max_retries: Optional[int] = Header(None),
    cfg: AppConfig = Depends(get_config),
    router_engine: RouterEngine = Depends(get_router),
    cache: SQLiteCache = Depends(get_cache),
    validator: StructuredOutputLoop = Depends(get_validator),
    telemetry: TelemetryCollector = Depends(get_telemetry),
) -> ChatCompletionResponse:
    """Route, validate, cache, and execute chat completions."""
    start_time = time.perf_counter()

    # Parse headers for bypass and overrides
    bypass = request.bypass_cache
    if x_cache_bypass and x_cache_bypass.lower() in ("true", "1", "yes"):
        bypass = True
    if cache_control and "no-cache" in cache_control.lower():
        bypass = True

    if x_task_tag:
        tags = [t.strip() for t in x_task_tag.split(",") if t.strip()]
        request.tags.extend(tags)

    if x_max_retries is not None:
        request.max_retries = x_max_retries

    cache_key = compute_cache_key(
        request,
        normalize_whitespace=cfg.cache.normalize_whitespace,
    )

    # 1. Cache lookup
    if not bypass and cfg.cache.enabled:
        cached_resp = await cache.get(cache_key)
        if cached_resp:
            saved_tokens = cached_resp.usage.prompt_tokens + cached_resp.usage.completion_tokens
            telemetry.record_cache_hit(saved_tokens)
            telemetry.record_request(
                success=True,
                latency_ms=0.5,
                provider="cache",
                prompt_tokens=cached_resp.usage.prompt_tokens,
                completion_tokens=cached_resp.usage.completion_tokens,
            )

            response.headers["X-Cache"] = "HIT"
            response.headers["X-Router-Provider"] = "cache"
            response.headers["X-Tokens-Saved"] = str(saved_tokens)
            response.headers["X-Latency-Ms"] = "0.5"
            return cached_resp

    if bypass:
        cache.record_bypass()
        telemetry.record_cache_bypass()
    else:
        telemetry.record_cache_miss()

    # 2. Execute via Validator loop and Router engine
    try:
        completion_resp = await validator.execute_with_validation(
            request,
            router_engine.execute,
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        telemetry.record_request(
            success=False,
            latency_ms=latency_ms,
            provider="unknown",
        )
        raise e

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    provider_used = (
        completion_resp.router_metadata.provider
        if completion_resp.router_metadata
        else "unknown"
    )

    # 3. Store in cache
    if cfg.cache.enabled and not (cache_control and "no-store" in cache_control.lower()):
        await cache.set(cache_key, request, completion_resp)

    # 4. Record telemetry
    telemetry.record_request(
        success=True,
        latency_ms=latency_ms,
        provider=provider_used,
        prompt_tokens=completion_resp.usage.prompt_tokens,
        completion_tokens=completion_resp.usage.completion_tokens,
    )

    # 5. Populate response headers
    response.headers["X-Cache"] = "MISS"
    response.headers["X-Router-Provider"] = provider_used
    response.headers["X-Latency-Ms"] = str(round(latency_ms, 2))
    if completion_resp.router_metadata:
        response.headers["X-Validation-Retries"] = str(
            completion_resp.router_metadata.validation_retries
        )

    return completion_resp


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(
    cfg: AppConfig = Depends(get_config),
    registry: ProviderRegistry = Depends(get_registry),
) -> ModelListResponse:
    """Return list of models, aliases, and available providers."""
    model_cards: List[ModelCard] = []

    # Configured aliases
    for alias in cfg.routing.aliases.keys():
        model_cards.append(ModelCard(id=alias, owned_by="router-alias"))

    # Provider models
    for p_cfg in cfg.providers:
        model_cards.append(ModelCard(id=f"{p_cfg.name}:{p_cfg.default_model}", owned_by=p_cfg.name))
        for m in p_cfg.models:
            model_cards.append(ModelCard(id=f"{p_cfg.name}:{m}", owned_by=p_cfg.name))

    # Add standard tags/targets
    for tag in ["slm", "heavy", "auto", "default"]:
        if not any(card.id == tag for card in model_cards):
            model_cards.append(ModelCard(id=tag, owned_by="router-virtual"))

    return ModelListResponse(data=model_cards)


@router.get("/health")
async def health(
    registry: ProviderRegistry = Depends(get_registry),
    cache: SQLiteCache = Depends(get_cache),
) -> Dict[str, Any]:
    """Health check endpoint for proxy and backends."""
    provider_health = {}
    for p in registry.list_all():
        provider_health[p.name] = await p.health()

    return {
        "status": "healthy",
        "providers": provider_health,
        "cache": "connected",
    }


@router.get("/v1/router/stats", response_model=TelemetryStats)
@router.get("/stats", response_model=TelemetryStats)
async def get_stats(
    telemetry: TelemetryCollector = Depends(get_telemetry),
) -> TelemetryStats:
    """Return runtime telemetry, cache hit rates, latency and token metrics."""
    return telemetry.get_stats()


@router.post("/v1/router/cache/clear")
async def clear_cache(cache: SQLiteCache = Depends(get_cache)) -> Dict[str, Any]:
    """Purge all cached responses."""
    deleted_count = await cache.clear()
    return {"status": "ok", "cleared_entries": deleted_count}
