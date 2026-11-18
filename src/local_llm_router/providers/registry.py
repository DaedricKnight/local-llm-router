"""Provider registry and factory."""

from __future__ import annotations

from typing import Dict, List, Optional

from local_llm_router.config import ProviderConfig
from local_llm_router.exceptions import ConfigurationError
from local_llm_router.providers.base import BaseProvider
from local_llm_router.providers.llama_cpp import LlamaCppProvider
from local_llm_router.providers.mock import MockProvider
from local_llm_router.providers.ollama import OllamaProvider
from local_llm_router.providers.openai_compatible import OpenAICompatibleProvider


class ProviderRegistry:
    """Registry maintaining initialized provider instances."""

    def __init__(self) -> None:
        self._providers: Dict[str, BaseProvider] = {}

    def register(self, provider: BaseProvider) -> None:
        """Register a provider instance."""
        self._providers[provider.name] = provider

    def get(self, name: str) -> Optional[BaseProvider]:
        """Look up provider by name."""
        return self._providers.get(name)

    def list_all(self) -> List[BaseProvider]:
        """Return all registered providers."""
        return list(self._providers.values())

    async def close_all(self) -> None:
        """Close all provider resources."""
        for p in self._providers.values():
            await p.close()


def create_provider_from_config(cfg: ProviderConfig) -> BaseProvider:
    """Factory to instantiate a provider from its configuration."""
    if cfg.type == "mock":
        return MockProvider(
            name=cfg.name,
            default_model=cfg.default_model,
        )
    elif cfg.type == "ollama":
        return OllamaProvider(
            name=cfg.name,
            default_model=cfg.default_model,
            base_url=cfg.base_url or "http://localhost:11434",
            timeout_seconds=cfg.timeout_seconds,
        )
    elif cfg.type == "llama_cpp":
        return LlamaCppProvider(
            name=cfg.name,
            default_model=cfg.default_model,
            base_url=cfg.base_url or "http://localhost:8080",
            timeout_seconds=cfg.timeout_seconds,
        )
    elif cfg.type == "openai_compatible":
        return OpenAICompatibleProvider(
            name=cfg.name,
            default_model=cfg.default_model,
            base_url=cfg.base_url or "https://api.openai.com/v1",
            api_key=cfg.api_key,
            timeout_seconds=cfg.timeout_seconds,
        )
    else:
        raise ConfigurationError(f"Unknown provider type: {cfg.type}")


def build_registry(configs: List[ProviderConfig]) -> ProviderRegistry:
    """Build and populate a registry from a list of provider configs."""
    registry = ProviderRegistry()
    for cfg in configs:
        provider = create_provider_from_config(cfg)
        registry.register(provider)
    return registry
