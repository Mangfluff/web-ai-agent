"""
LLM interface module - communicates with OpenAI-compatible APIs (async).
"""

import os
import asyncio
from typing import Optional
from httpx import AsyncClient, Timeout


class LLMClient:
    """Async client for making LLM API calls (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o")
        # Default to OpenAI if no base URL
        if not self.base_url:
            self.base_url = "https://api.openai.com/v1"
        self._client: Optional[AsyncClient] = None

    async def _ensure_client(self):
        """Lazy-init the async client so we don't require API key at construction."""
        if self._client is not None:
            return
        if not self.api_key:
            raise ValueError(
                "LLM_API_KEY is not set. Provide it via constructor, --api-key, or .env file."
            )
        self._client = AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=Timeout(120.0),
        )

    async def chat(self, messages: list[dict], temperature: float = 0.1,
                   max_retries: int = 3) -> str:
        """Send a chat completion request with retry logic."""
        await self._ensure_client()

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                }
                resp = await self._client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"  LLM API error (attempt {attempt}/{max_retries}): {e}. Retrying in {wait}s...")
                    await asyncio.sleep(wait)
        raise RuntimeError(f"LLM API failed after {max_retries} attempts: {last_error}")

    async def close(self):
        if self._client:
            await self._client.aclose()