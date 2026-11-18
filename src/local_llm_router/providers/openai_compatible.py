"""Generic OpenAI-compatible provider adapter."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import httpx

from local_llm_router.exceptions import (
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderTokenLimitExceededError,
)
from local_llm_router.models import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    RouterMetadata,
    Usage,
)
from local_llm_router.providers.base import BaseProvider


class OpenAICompatibleProvider(BaseProvider):
    """Adapter for OpenAI and OpenAI-compatible services (vLLM, Groq, OpenRouter, etc.)."""

    def __init__(
        self,
        name: str = "openai",
        default_model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        timeout_seconds: float = 30.0,
        extra_headers: Optional[Dict[str, str]] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        super().__init__(name=name, default_model=default_model, timeout_seconds=timeout_seconds)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.extra_headers = extra_headers or {}

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)

        self._client = client or httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(self.timeout_seconds),
        )

    def _prepare_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Convert standard request to OpenAI-compatible dictionary."""
        model = request.model or self.default_model
        # Strip provider prefix if user passed "openai:gpt-4o"
        if ":" in model and model.startswith(f"{self.name}:"):
            model = model.split(":", 1)[1]

        messages_payload = []
        for msg in request.messages:
            item: Dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.name:
                item["name"] = msg.name
            messages_payload.append(item)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages_payload,
            "temperature": request.temperature,
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.response_format:
            if isinstance(request.response_format, dict):
                payload["response_format"] = request.response_format
            else:
                payload["response_format"] = request.response_format.model_dump(exclude_none=True)

        return payload

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Call standard /chat/completions endpoint."""
        url = f"{self.base_url}/chat/completions"
        payload = self._prepare_payload(request)
        start_time = time.perf_counter()

        try:
            resp = await self._client.post(url, json=payload)
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                f"Request to {url} timed out after {self.timeout_seconds}s: {e}",
                provider_name=self.name,
            ) from e
        except (httpx.ConnectError, httpx.NetworkError) as e:
            raise ProviderConnectionError(
                f"Failed to connect to provider at {url}: {e}",
                provider_name=self.name,
            ) from e
        except Exception as e:
            raise ProviderError(
                f"Unexpected client exception contacting {url}: {e}",
                provider_name=self.name,
            ) from e

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if resp.status_code == 429:
            raise ProviderRateLimitError(
                f"Rate limit exceeded (HTTP 429): {resp.text}",
                provider_name=self.name,
                status_code=429,
            )

        if resp.status_code == 400:
            error_text = resp.text.lower()
            if any(k in error_text for k in ["context_length", "maximum context", "tokens", "too long"]):
                raise ProviderTokenLimitExceededError(
                    f"Token context limit exceeded: {resp.text}",
                    provider_name=self.name,
                    status_code=400,
                )
            raise ProviderError(
                f"Bad request (HTTP 400): {resp.text}",
                provider_name=self.name,
                status_code=400,
            )

        if resp.status_code >= 500:
            raise ProviderConnectionError(
                f"Provider internal server error (HTTP {resp.status_code}): {resp.text}",
                provider_name=self.name,
                status_code=resp.status_code,
            )

        if resp.is_error:
            raise ProviderError(
                f"HTTP {resp.status_code}: {resp.text}",
                provider_name=self.name,
                status_code=resp.status_code,
            )

        try:
            data = resp.json()
        except Exception as e:
            raise ProviderError(
                f"Failed to parse provider response JSON: {resp.text}",
                provider_name=self.name,
            ) from e

        choices = []
        for choice_data in data.get("choices", []):
            msg_data = choice_data.get("message", {})
            choices.append(
                ChatChoice(
                    index=choice_data.get("index", 0),
                    message=ChatMessage(
                        role=msg_data.get("role", "assistant"),
                        content=msg_data.get("content", ""),
                    ),
                    finish_reason=choice_data.get("finish_reason", "stop"),
                )
            )

        usage_data = data.get("usage", {})
        prompt_tokens = usage_data.get("prompt_tokens", self.estimate_request_tokens(request))
        completion_tokens = usage_data.get(
            "completion_tokens",
            self.count_tokens(choices[0].message.content) if choices else 0,
        )

        model_name = payload["model"]

        return ChatCompletionResponse(
            id=data.get("id", f"chatcmpl-{int(time.time())}"),
            created=data.get("created", int(time.time())),
            model=model_name,
            choices=choices,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            router_metadata=RouterMetadata(
                provider=self.name,
                model_used=model_name,
                cached=False,
                latency_ms=round(latency_ms, 2),
            ),
        )

    async def health(self) -> bool:
        """Ping /models endpoint to verify provider connectivity."""
        url = f"{self.base_url}/models"
        try:
            resp = await self._client.get(url, timeout=5.0)
            return resp.status_code < 400
        except Exception:
            return False

    async def close(self) -> None:
        """Close HTTP client session."""
        await self._client.aclose()
