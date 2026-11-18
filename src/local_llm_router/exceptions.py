"""Exception hierarchy for local-llm-router."""

from typing import Any, Dict, List, Optional


class RouterException(Exception):
    """Base exception for all router-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(RouterException):
    """Raised when there is an invalid configuration."""
    pass


class ProviderError(RouterException):
    """Base exception for provider communication errors."""

    def __init__(
        self,
        message: str,
        provider_name: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(f"[{provider_name}] {message}", details=details)
        self.provider_name = provider_name
        self.status_code = status_code


class ProviderConnectionError(ProviderError):
    """Raised when a provider cannot be reached (network/socket error)."""
    pass


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request times out."""
    pass


class ProviderRateLimitError(ProviderError):
    """Raised when provider returns HTTP 429 Too Many Requests."""
    pass


class ProviderTokenLimitExceededError(ProviderError):
    """Raised when the context window / token limit is exceeded."""
    pass


class AllProvidersFailedError(RouterException):
    """Raised when all candidate providers in the fallback chain have failed."""

    def __init__(self, attempted_providers: List[str], errors: List[Exception]) -> None:
        summary = "; ".join(f"{p}: {str(e)}" for p, e in zip(attempted_providers, errors, strict=False))
        super().__init__(
            f"All providers failed in fallback chain: {summary}",
            details={"attempted": attempted_providers, "errors": [str(e) for e in errors]},
        )
        self.attempted_providers = attempted_providers
        self.errors = errors


class OutputValidationError(RouterException):
    """Raised when model output fails structured JSON schema validation."""

    def __init__(self, message: str, raw_output: str, schema_errors: List[str]) -> None:
        super().__init__(message, details={"raw_output": raw_output, "schema_errors": schema_errors})
        self.raw_output = raw_output
        self.schema_errors = schema_errors


class MaxRetriesExceededError(OutputValidationError):
    """Raised when JSON schema validation retry loop reaches maximum attempts."""

    def __init__(self, attempts: int, last_raw_output: str, last_errors: List[str]) -> None:
        super().__init__(
            f"Structured output validation failed after {attempts} attempts: {', '.join(last_errors)}",
            raw_output=last_raw_output,
            schema_errors=last_errors,
        )
        self.attempts = attempts


class CacheError(RouterException):
    """Raised on cache storage/retrieval failures."""
    pass
