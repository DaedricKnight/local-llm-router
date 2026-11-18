"""Router engine coordinating rules evaluation, provider execution, and fallback resilience."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from local_llm_router.config import RoutingRuleConfig
from local_llm_router.exceptions import (
    AllProvidersFailedError,
    ConfigurationError,
)
from local_llm_router.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    RouterMetadata,
)
from local_llm_router.providers.registry import ProviderRegistry
from local_llm_router.router.policies import FallbackPolicy
from local_llm_router.router.rules import RouteDecision, RoutingRulesEvaluator

logger = logging.getLogger(__name__)


class RouterEngine:
    """Core routing engine orchestrating models, SLMs, and cloud fallbacks."""

    def __init__(
        self,
        registry: ProviderRegistry,
        config: RoutingRuleConfig,
        telemetry: Optional[Any] = None,
    ) -> None:
        self.registry = registry
        self.config = config
        self.telemetry = telemetry
        self.evaluator = RoutingRulesEvaluator(config)
        self.policy = FallbackPolicy(config)

    def _estimate_tokens(self, request: ChatCompletionRequest) -> int:
        """Estimate tokens for the request using available providers or default heuristic."""
        providers = self.registry.list_all()
        if providers:
            return providers[0].estimate_request_tokens(request)
        # Fallback heuristic
        prompt_text = request.get_combined_prompt_text()
        return max(1, len(prompt_text.split()))

    def get_decision(self, request: ChatCompletionRequest) -> RouteDecision:
        """Evaluate and return routing decision without executing the call."""
        token_count = self._estimate_tokens(request)
        return self.evaluator.evaluate(request, token_count)

    async def execute(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Route request to the optimal model and handle automatic fallback on failure."""
        token_count = self._estimate_tokens(request)
        decision = self.evaluator.evaluate(request, token_count)
        chain = self.policy.build_chain(decision, request.model)

        logger.info(
            "Routing decision: %s:%s (rule=%s, reason=%s). Candidate chain: %s",
            decision.provider_name,
            decision.target_model,
            decision.rule_name,
            decision.reason,
            chain,
        )

        attempted_providers: List[str] = []
        errors: List[Exception] = []

        for idx, (provider_name, target_model) in enumerate(chain):
            provider = self.registry.get(provider_name)
            if not provider:
                err = ConfigurationError(f"Provider '{provider_name}' is not registered.")
                logger.warning("Provider '%s' missing from registry: %s", provider_name, err)
                attempted_providers.append(f"{provider_name}:{target_model}")
                errors.append(err)
                continue

            attempt_label = f"{provider_name}:{target_model}"
            attempted_providers.append(attempt_label)

            if idx > 0:
                logger.warning(
                    "Fallback triggered! Attempting secondary candidate #%d: %s",
                    idx + 1,
                    attempt_label,
                )
                if self.telemetry:
                    self.telemetry.record_fallback(provider_name, target_model)

            # Create candidate request
            candidate_req = request.model_copy(
                update={"model": target_model},
                deep=True,
            )

            try:
                response = await provider.complete(candidate_req)

                # Attach metadata about routing and fallback
                if response.router_metadata:
                    response.router_metadata.provider = provider_name
                    response.router_metadata.model_used = target_model
                    response.router_metadata.fallback_chain_used = attempted_providers
                else:
                    response.router_metadata = RouterMetadata(
                        provider=provider_name,
                        model_used=target_model,
                        cached=False,
                        fallback_chain_used=attempted_providers,
                    )

                logger.info(
                    "Successfully completed request using %s (tokens: prompt=%d, completion=%d)",
                    attempt_label,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )
                return response

            except Exception as e:
                logger.error("Execution failed on candidate %s: %s", attempt_label, e)
                errors.append(e)

                # Check if we should try next candidate
                if not self.policy.is_recoverable(e):
                    logger.warning("Error on %s is non-recoverable: %s", attempt_label, e)
                    # If this is the last candidate or non-recoverable, we fail immediately or continue
                    if idx == len(chain) - 1:
                        raise e

        # If all candidates exhausted
        raise AllProvidersFailedError(attempted_providers, errors)
