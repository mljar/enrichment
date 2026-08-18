"""OpenAI provider."""

from __future__ import annotations

import os
from typing import Optional

import httpx

from ..exceptions import ProviderConfigurationError
from .openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI provider using the modern Chat Completions HTTP API."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        timeout: float = 120.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ProviderConfigurationError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY."
            )
        super().__init__(
            base_url="https://api.openai.com/v1",
            api_key=key,
            model=model,
            timeout=timeout,
            client=client,
            name=self.name,
        )
