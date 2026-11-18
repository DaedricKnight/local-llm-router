"""Ollama local runtime provider adapter."""

from __future__ import annotations

from typing import Dict, List, Optional

import httpx

from local_llm_router.providers.openai_compatible import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    """Adapter for local Ollama instances."""

    def __init__(
        self,
        name: str = "ollama",
        default_model: str = "llama3.2:1b",
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 60.0,
        extra_headers: Optional[Dict[str, str]] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        raw_base = base_url.rstrip("/")
        # Ollama supports OpenAI API at /v1
        v1_url = raw_base if raw_base.endswith("/v1") else f"{raw_base}/v1"
        super().__init__(
            name=name,
            default_model=default_model,
            base_url=v1_url,
            api_key=None,
            timeout_seconds=timeout_seconds,
            extra_headers=extra_headers,
            client=client,
        )
        self.root_url = raw_base.replace("/v1", "")

    async def list_local_models(self) -> List[str]:
        """Fetch list of models pulled into local Ollama storage."""
        url = f"{self.root_url}/api/tags"
        try:
            resp = await self._client.get(url, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    async def health(self) -> bool:
        """Check Ollama availability via /api/version or /api/tags."""
        for path in ["/api/version", "/api/tags"]:
            try:
                resp = await self._client.get(f"{self.root_url}{path}", timeout=3.0)
                if resp.status_code == 200:
                    return True
            except Exception:
                continue
        # Fall back to OpenAI /models check
        return await super().health()
