"""Routing rules: length-based, tag-based, and alias-based evaluation."""

from __future__ import annotations

from typing import Tuple

from pydantic import BaseModel

from local_llm_router.config import RoutingRuleConfig
from local_llm_router.models import ChatCompletionRequest


class RouteDecision(BaseModel):
    """The outcome of routing rule evaluation."""

    provider_name: str
    target_model: str
    rule_name: str
    reason: str

    @property
    def target_spec(self) -> str:
        """Formatted string representation e.g. provider:model."""
        return f"{self.provider_name}:{self.target_model}"


def parse_target_spec(spec: str, default_provider: str = "ollama") -> Tuple[str, str]:
    """Parse 'provider:model' or 'model' spec into (provider, model)."""
    if ":" in spec:
        parts = spec.split(":", 1)
        return parts[0], parts[1]
    return default_provider, spec


class RoutingRulesEvaluator:
    """Evaluates request attributes to choose target provider and model."""

    def __init__(self, config: RoutingRuleConfig) -> None:
        self.config = config

    def evaluate(self, request: ChatCompletionRequest, prompt_tokens: int) -> RouteDecision:
        """Determine the primary route for the completion request."""
        model_req = request.model.strip()

        # 1. Alias-based routing (e.g. model="slm", model="heavy", model="fast")
        if model_req in self.config.aliases:
            target_spec = self.config.aliases[model_req]
            provider, model = parse_target_spec(target_spec)
            return RouteDecision(
                provider_name=provider,
                target_model=model,
                rule_name="alias",
                reason=f"Matched alias '{model_req}' -> '{target_spec}'",
            )

        # 2. Tag-based routing (e.g. request.tags=["code"] or ["fast"])
        for tag in request.tags:
            tag_clean = tag.strip().lower()
            if tag_clean in self.config.tags:
                target_spec = self.config.tags[tag_clean]
                # If target_spec is an alias, resolve it
                if target_spec in self.config.aliases:
                    target_spec = self.config.aliases[target_spec]
                provider, model = parse_target_spec(target_spec)
                return RouteDecision(
                    provider_name=provider,
                    target_model=model,
                    rule_name="tag",
                    reason=f"Matched task tag '{tag_clean}' -> '{target_spec}'",
                )

        # 3. Explicit provider prefix (e.g. "ollama:llama3.2:1b" or "openai:gpt-4o")
        if ":" in model_req:
            provider, model = parse_target_spec(model_req)
            return RouteDecision(
                provider_name=provider,
                target_model=model,
                rule_name="explicit_prefix",
                reason=f"Parsed explicit provider spec '{model_req}'",
            )

        # 4. Length-based routing (for general models or "auto", "default")
        if model_req in ["auto", "default", "", "slm", "router"]:
            threshold = self.config.length_threshold_tokens
            if prompt_tokens <= threshold:
                target_spec = self.config.slm_target
                rule_name = "length_slm"
                reason = f"Prompt tokens ({prompt_tokens}) <= threshold ({threshold}) -> SLM"
            else:
                target_spec = self.config.heavy_target
                rule_name = "length_heavy"
                reason = f"Prompt tokens ({prompt_tokens}) > threshold ({threshold}) -> Heavy"

            if target_spec in self.config.aliases:
                target_spec = self.config.aliases[target_spec]
            provider, model = parse_target_spec(target_spec)
            return RouteDecision(
                provider_name=provider,
                target_model=model,
                rule_name=rule_name,
                reason=reason,
            )

        # 5. Default fallback: treat model as direct target with default provider
        provider, model = parse_target_spec(model_req, default_provider="openai")
        return RouteDecision(
            provider_name=provider,
            target_model=model,
            rule_name="direct_passthrough",
            reason=f"Direct model passthrough for '{model_req}'",
        )
