"""Mock provider for deterministic offline testing and Labelbox test suites."""

from __future__ import annotations

import asyncio
from typing import Callable, Dict, List, Optional

from local_llm_router.exceptions import (
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderTokenLimitExceededError,
)
from local_llm_router.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    RouterMetadata,
)
from local_llm_router.providers.base import BaseProvider


class MockProvider(BaseProvider):
    """Configurable mock provider that does not require any external server or API keys."""

    def __init__(
        self,
        name: str = "mock",
        default_model: str = "mock-model",
        default_response: str = "Hello from MockProvider!",
        model_responses: Optional[Dict[str, str]] = None,
        response_generator: Optional[Callable[[ChatCompletionRequest], str]] = None,
        simulated_latency_ms: float = 0.0,
        fail_times: int = 0,
        failure_type: str = "timeout",  # "timeout", "connection", "ratelimit", "token_limit"
        is_healthy: bool = True,
    ) -> None:
        super().__init__(name=name, default_model=default_model)
        self.default_response = default_response
        self.model_responses = model_responses or {}
        self.response_generator = response_generator
        self.simulated_latency_ms = simulated_latency_ms
        self.fail_times = fail_times
        self.failure_type = failure_type
        self.is_healthy = is_healthy
        self.call_count = 0
        self.recorded_requests: List[ChatCompletionRequest] = []

    def set_response_for_model(self, model: str, response: str) -> None:
        """Register specific response for a model name."""
        self.model_responses[model] = response

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Simulate LLM response generation with error injection and latency simulation."""
        self.call_count += 1
        self.recorded_requests.append(request)

        # Failure injection
        if self.call_count <= self.fail_times:
            if self.failure_type == "timeout":
                raise ProviderTimeoutError(
                    f"Mock simulated timeout on call #{self.call_count}",
                    provider_name=self.name,
                )
            elif self.failure_type == "ratelimit":
                raise ProviderRateLimitError(
                    f"Mock simulated rate limit 429 on call #{self.call_count}",
                    provider_name=self.name,
                    status_code=429,
                )
            elif self.failure_type == "token_limit":
                raise ProviderTokenLimitExceededError(
                    f"Mock context length exceeded on call #{self.call_count}",
                    provider_name=self.name,
                    status_code=400,
                )
            else:
                raise ProviderConnectionError(
                    f"Mock simulated connection error on call #{self.call_count}",
                    provider_name=self.name,
                )

        if self.simulated_latency_ms > 0:
            await asyncio.sleep(self.simulated_latency_ms / 1000.0)

        # Determine response content
        if self.response_generator:
            content = self.response_generator(request)
        elif request.model in self.model_responses:
            content = self.model_responses[request.model]
        else:
            content = self.default_response

        prompt_tokens = self.estimate_request_tokens(request)
        completion_tokens = self.count_tokens(content)

        return ChatCompletionResponse.create_simple(
            content=content,
            model=request.model or self.default_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            router_metadata=RouterMetadata(
                provider=self.name,
                model_used=request.model or self.default_model,
                cached=False,
                latency_ms=self.simulated_latency_ms,
            ),
        )

    async def health(self) -> bool:
        """Return configured simulated health state."""
        return self.is_healthy

    def reset(self) -> None:
        """Reset mock counters and history."""
        self.call_count = 0
        self.recorded_requests.clear()
