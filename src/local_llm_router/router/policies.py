"""Fallback policy and exception handling strategy for routing resilience."""

from __future__ import annotations

import logging
from typing import List, Tuple

from local_llm_router.config import RoutingRuleConfig
from local_llm_router.exceptions import (
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderTokenLimitExceededError,
)
from local_llm_router.router.rules import RouteDecision, parse_target_spec

logger = logging.getLogger(__name__)


class FallbackPolicy:
    """Computes fallback chains and evaluates failure recoverability."""

    def __init__(self, config: RoutingRuleConfig) -> None:
        self.config = config

    def build_chain(self, primary_decision: RouteDecision, requested_model: str) -> List[Tuple[str, str]]:
        """Construct ordered list of (provider, model) candidates."""
        candidates: List[Tuple[str, str]] = [
            (primary_decision.provider_name, primary_decision.target_model)
        ]

        # Look up specific chain for model, alias, or 'default'
        chain_specs = (
            self.config.fallback_chains.get(requested_model)
            or self.config.fallback_chains.get(primary_decision.rule_name)
            or self.config.fallback_chains.get("default")
            or []
        )

        for spec in chain_specs:
            resolved_spec = self.config.aliases.get(spec, spec)
            provider, model = parse_target_spec(resolved_spec)
            pair = (provider, model)
            if pair not in candidates:
                candidates.append(pair)

        return candidates

    def is_recoverable(self, exception: Exception) -> bool:
        """Determine if the exception warrants attempting fallback providers."""
        if isinstance(
            exception,
            (
                ProviderTimeoutError,
                ProviderConnectionError,
                ProviderRateLimitError,
                ProviderTokenLimitExceededError,
            ),
        ):
            return True

        if isinstance(exception, ProviderError) and exception.status_code is not None:
            # 5xx server errors or 429 rate limit
            return exception.status_code >= 500 or exception.status_code == 429

        return False
