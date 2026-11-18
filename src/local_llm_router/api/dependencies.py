"""FastAPI dependency injection utilities."""

from __future__ import annotations

from typing import cast

from fastapi import Request

from local_llm_router.cache.sqlite_cache import SQLiteCache
from local_llm_router.config import AppConfig
from local_llm_router.providers.registry import ProviderRegistry
from local_llm_router.router.engine import RouterEngine
from local_llm_router.telemetry.collector import TelemetryCollector
from local_llm_router.validator.loop import StructuredOutputLoop


def get_config(request: Request) -> AppConfig:
    return cast(AppConfig, request.app.state.config)


def get_registry(request: Request) -> ProviderRegistry:
    return cast(ProviderRegistry, request.app.state.registry)


def get_router(request: Request) -> RouterEngine:
    return cast(RouterEngine, request.app.state.router)


def get_cache(request: Request) -> SQLiteCache:
    return cast(SQLiteCache, request.app.state.cache)


def get_validator(request: Request) -> StructuredOutputLoop:
    return cast(StructuredOutputLoop, request.app.state.validator)


def get_telemetry(request: Request) -> TelemetryCollector:
    return cast(TelemetryCollector, request.app.state.telemetry)
