"""Base provider abstract class for LLM runtimes."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from local_llm_router.models import ChatCompletionRequest, ChatCompletionResponse


class BaseProvider(ABC):
    """Abstract base class for all LLM/SLM providers."""

    def __init__(
        self,
        name: str,
        default_model: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.name = name
        self.default_model = default_model
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Execute a chat completion request with the provider."""
        raise NotImplementedError

    @abstractmethod
    async def health(self) -> bool:
        """Check provider operational availability and responsiveness."""
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        """Estimate token count for a given text.

        Uses an accurate character heuristic (avg ~3.8 chars/token) suitable
        for both Latin and multilingual prompts in offline mode.
        """
        if not text:
            return 0
        # Basic word and char heuristic
        words = len(text.split())
        chars = len(text)
        # Average blend between word count and 3.8 char/token rule
        est = (chars / 3.8 + words * 1.3) / 2.0
        return max(1, math.ceil(est))

    def estimate_request_tokens(self, request: ChatCompletionRequest) -> int:
        """Estimate total input prompt tokens for the given request."""
        total = 0
        for msg in request.messages:
            total += 4  # overhead per message in chatml/openai format
            total += self.count_tokens(msg.content)
            if msg.name:
                total += self.count_tokens(msg.name)
        total += 2  # priming token
        return total

    async def close(self) -> None:  # noqa: B027
        """Clean up network sockets and client sessions."""
        pass
