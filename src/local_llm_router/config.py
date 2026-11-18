"""Configuration models and loader for local-llm-router."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from local_llm_router.exceptions import ConfigurationError


class ProviderConfig(BaseModel):
    """Configuration for an individual provider."""

    name: str
    type: Literal["ollama", "llama_cpp", "openai_compatible", "mock"]
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    default_model: str = "default-model"
    timeout_seconds: float = 30.0
    models: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)


class RoutingRuleConfig(BaseModel):
    """Routing rules configuration."""

    default_route: str = "slm"
    length_threshold_tokens: int = 250
    slm_target: str = "ollama:llama3.2:1b"
    heavy_target: str = "openai:gpt-4o-mini"
    tags: Dict[str, str] = Field(
        default_factory=lambda: {
            "fast": "slm",
            "code": "ollama:qwen2.5-coder",
            "heavy": "heavy",
        }
    )
    aliases: Dict[str, str] = Field(
        default_factory=lambda: {
            "slm": "ollama:llama3.2:1b",
            "heavy": "openai:gpt-4o-mini",
            "default": "ollama:llama3.2:1b",
            "fast": "ollama:llama3.2:1b",
        }
    )
    fallback_chains: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "default": [
                "ollama:llama3.2:1b",
                "openai:gpt-4o-mini",
                "mock:fallback",
            ]
        }
    )


class CacheConfig(BaseModel):
    """Caching layer configuration."""

    enabled: bool = True
    db_path: str = "data/cache.db"
    ttl_seconds: int = 86400  # 24 hours (0 = infinite)
    max_entries: int = 10000  # LRU threshold
    normalize_whitespace: bool = True


class ValidatorConfig(BaseModel):
    """Structured output validation configuration."""

    enabled: bool = True
    max_retries: int = 3
    strip_markdown: bool = True


class ServerConfig(BaseModel):
    """FastAPI & Uvicorn server configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"


class AppConfig(BaseSettings):
    """Top-level application configuration."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    routing: RoutingRuleConfig = Field(default_factory=RoutingRuleConfig)
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
    providers: List[ProviderConfig] = Field(
        default_factory=lambda: [
            ProviderConfig(
                name="ollama",
                type="ollama",
                base_url="http://localhost:11434",
                default_model="llama3.2:1b",
                models=["llama3.2:1b", "qwen2.5-coder"],
            ),
            ProviderConfig(
                name="llamacpp",
                type="llama_cpp",
                base_url="http://localhost:8080",
                default_model="local-model",
            ),
            ProviderConfig(
                name="openai",
                type="openai_compatible",
                base_url="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY", "mock-key"),
                default_model="gpt-4o-mini",
                models=["gpt-4o", "gpt-4o-mini"],
            ),
            ProviderConfig(
                name="mock",
                type="mock",
                default_model="mock-model",
                models=["mock-model", "fallback"],
            ),
        ]
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        """Load configuration from a YAML file."""
        file_path = Path(path)
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        except Exception as e:
            raise ConfigurationError(f"Failed to parse config file {path}: {e}") from e

    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """Find a provider config by its name."""
        for p in self.providers:
            if p.name == name:
                return p
        return None


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load config from path or default search locations."""
    if config_path and Path(config_path).exists():
        return AppConfig.from_yaml(config_path)

    default_paths = [
        Path("router.yaml"),
        Path("router.yml"),
        Path.home() / ".config" / "local-llm-router" / "router.yaml",
    ]
    for p in default_paths:
        if p.exists():
            return AppConfig.from_yaml(p)

    return AppConfig()
