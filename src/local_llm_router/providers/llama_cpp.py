"""llama.cpp server provider adapter."""

from __future__ import annotations

from typing import Dict, Optional

import httpx

from local_llm_router.providers.openai_compatible import OpenAICompatibleProvider


class LlamaCppProvider(OpenAICompatibleProvider):
    """Adapter for local llama.cpp HTTP server (llama-server)."""

    def __init__(
        self,
        name: str = "llamacpp",
        default_model: str = "local-model",
        base_url: str = "http://localhost:8080",
        timeout_seconds: float = 60.0,
        extra_headers: Optional[Dict[str, str]] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        raw_base = base_url.rstrip("/")
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

    async def health(self) -> bool:
        """Check llama.cpp server health via /health endpoint."""
        try:
            resp = await self._client.get(f"{self.root_url}/health", timeout=3.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        return await super().health()
