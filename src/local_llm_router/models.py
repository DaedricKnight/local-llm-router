"""Pydantic data models for OpenAI-compatible chat completions, routing, and metadata."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat completion message."""

    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str
    name: Optional[str] = None


class ResponseFormat(BaseModel):
    """Response format specification (e.g., json_object or json_schema)."""

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request with proxy extensions."""

    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    response_format: Optional[Union[Dict[str, Any], ResponseFormat]] = None

    # Extended routing and proxy parameters
    json_schema: Optional[Dict[str, Any]] = None
    tags: List[str] = Field(default_factory=list)
    bypass_cache: bool = False
    max_retries: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_system_prompt(self) -> Optional[str]:
        """Extract the system prompt if present in messages."""
        for msg in self.messages:
            if msg.role == "system":
                return msg.content
        return None

    def get_combined_prompt_text(self) -> str:
        """Combine all message texts for token counting and routing heuristics."""
        return "\n".join(f"{msg.role}: {msg.content}" for msg in self.messages)

    def get_effective_schema(self) -> Optional[Dict[str, Any]]:
        """Retrieve target JSON schema from either json_schema or response_format."""
        if self.json_schema:
            return self.json_schema
        if self.response_format:
            if isinstance(self.response_format, dict):
                if self.response_format.get("type") == "json_schema" and "json_schema" in self.response_format:
                    schema_data = self.response_format["json_schema"]
                    if isinstance(schema_data, dict) and "schema" in schema_data:
                        return schema_data["schema"]
                    return schema_data
                if "schema" in self.response_format:
                    return self.response_format["schema"]
            elif isinstance(self.response_format, ResponseFormat):
                if self.response_format.json_schema:
                    return self.response_format.json_schema
        return None


class ChatChoice(BaseModel):
    """Single completion choice."""

    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class Usage(BaseModel):
    """Token usage details."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RouterMetadata(BaseModel):
    """Telemetry and operational metadata attached to the response."""

    provider: str
    model_used: str
    cached: bool = False
    latency_ms: float = 0.0
    fallback_chain_used: List[str] = Field(default_factory=list)
    validation_retries: int = 0
    tokens_saved: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]
    usage: Usage = Field(default_factory=Usage)
    router_metadata: Optional[RouterMetadata] = None

    @classmethod
    def create_simple(
        cls,
        content: str,
        model: str,
        role: Literal["system", "user", "assistant", "tool", "function"] = "assistant",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        router_metadata: Optional[RouterMetadata] = None,
    ) -> ChatCompletionResponse:
        """Helper to create a standard single-choice response."""
        return cls(
            model=model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role=role, content=content),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            router_metadata=router_metadata,
        )


class ModelCard(BaseModel):
    """OpenAI-compatible model representation."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "local-llm-router"
    permission: List[Dict[str, Any]] = Field(default_factory=list)


class ModelListResponse(BaseModel):
    """OpenAI-compatible list of available models."""

    object: str = "list"
    data: List[ModelCard]


class CacheStatsResponse(BaseModel):
    """Cache telemetry details."""

    total_entries: int
    hit_count: int
    miss_count: int
    bypass_count: int
    hit_rate_percent: float
    tokens_saved: int
    size_bytes: int


class TelemetryStats(BaseModel):
    """Global proxy telemetry statistics."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    cache_hits: int
    cache_misses: int
    cache_bypasses: int
    cache_hit_rate_pct: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens_saved: int
    validation_retries_total: int
    fallback_events_total: int
    avg_latency_ms: float
    p95_latency_ms: float
    provider_calls: Dict[str, int]
