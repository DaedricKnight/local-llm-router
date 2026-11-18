"""FastAPI application factory with lifespan and OpenAI error handlers."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from local_llm_router.api.routes import router as api_router
from local_llm_router.cache.sqlite_cache import SQLiteCache
from local_llm_router.config import AppConfig, load_config
from local_llm_router.exceptions import (
    AllProvidersFailedError,
    MaxRetriesExceededError,
    OutputValidationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    RouterException,
)
from local_llm_router.providers.registry import build_registry
from local_llm_router.router.engine import RouterEngine
from local_llm_router.telemetry.collector import TelemetryCollector
from local_llm_router.validator.loop import StructuredOutputLoop

logger = logging.getLogger("local_llm_router")


def make_openai_error(message: str, error_type: str, code: int) -> JSONResponse:
    """Format error as standard OpenAI API error payload."""
    return JSONResponse(
        status_code=code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": code,
            }
        },
    )


def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    """Create and configure FastAPI proxy application."""
    cfg = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Startup
        logger.info("Initializing local-llm-router services...")
        registry = build_registry(cfg.providers)
        cache = SQLiteCache(cfg.cache)
        telemetry = TelemetryCollector()
        router = RouterEngine(registry, cfg.routing, telemetry=telemetry)
        validator = StructuredOutputLoop(cfg.validator)

        app.state.config = cfg
        app.state.registry = registry
        app.state.cache = cache
        app.state.telemetry = telemetry
        app.state.router = router
        app.state.validator = validator

        yield

        # Shutdown
        logger.info("Shutting down local-llm-router services...")
        await cache.close()
        await registry.close_all()

    app = FastAPI(
        title="Local LLM Router & Cache Proxy",
        description="Unified local/cloud LLM routing, SQLite caching, and JSON schema validation proxy.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Error handling
    @app.exception_handler(ProviderTimeoutError)
    async def timeout_handler(request: Request, exc: ProviderTimeoutError) -> JSONResponse:
        return make_openai_error(exc.message, "provider_timeout", 504)

    @app.exception_handler(ProviderRateLimitError)
    async def ratelimit_handler(request: Request, exc: ProviderRateLimitError) -> JSONResponse:
        return make_openai_error(exc.message, "rate_limit_exceeded", 429)

    @app.exception_handler(ProviderConnectionError)
    async def connection_handler(request: Request, exc: ProviderConnectionError) -> JSONResponse:
        return make_openai_error(exc.message, "provider_connection_error", 502)

    @app.exception_handler(AllProvidersFailedError)
    async def all_failed_handler(request: Request, exc: AllProvidersFailedError) -> JSONResponse:
        return make_openai_error(exc.message, "all_providers_failed", 502)

    @app.exception_handler(MaxRetriesExceededError)
    async def retries_exceeded_handler(
        request: Request, exc: MaxRetriesExceededError
    ) -> JSONResponse:
        return make_openai_error(exc.message, "validation_max_retries_exceeded", 422)

    @app.exception_handler(OutputValidationError)
    async def validation_handler(request: Request, exc: OutputValidationError) -> JSONResponse:
        return make_openai_error(exc.message, "json_schema_validation_failed", 422)

    @app.exception_handler(RouterException)
    async def generic_router_handler(request: Request, exc: RouterException) -> JSONResponse:
        return make_openai_error(exc.message, "router_error", 500)

    app.include_router(api_router)
    return app
