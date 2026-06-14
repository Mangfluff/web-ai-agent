"""
LLM interface module - communicates with OpenAI-compatible APIs.
"""

import os
from typing import Optional
from httpx import Client, Timeout


class LLMClient:
    """Client for making LLM API calls (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o")

        if not self.api_key:
            raise ValueError(
                "LLM_API_KEY is not set. Provide it via constructor or .env file."
            )

        # Default to OpenAI if no base URL
        if not self.base_url:
            self.base_url = "https://api.openai.com/v1"

        self._client = Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=Timeout(60.0),
        )

    def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        """Send a chat completion request and return the response text."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        resp = self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def close(self):
        self._client.close()