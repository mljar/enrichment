"""OpenAI-compatible Chat Completions provider."""

from __future__ import annotations

import json
from typing import Mapping, Optional

import httpx

from ..exceptions import (
    ProviderConfigurationError,
    ProviderResponseError,
    ProviderTemporaryError,
)
from ..models import CompletionRequest, CompletionResult
from .base import Provider


class OpenAICompatibleProvider(Provider):
    """Call an OpenAI-compatible ``/chat/completions`` endpoint."""

    name = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: float = 120.0,
        client: Optional[httpx.Client] = None,
        name: Optional[str] = None,
    ) -> None:
        if not base_url:
            raise ProviderConfigurationError("Provider base_url cannot be empty.")
        self.base_url = base_url.rstrip("/")
        self.default_model = model
        self.api_key = api_key
        self.timeout = timeout
        self.name = name or self.name
        self._headers = dict(headers or {})
        self._client = client or httpx.Client()
        self._owns_client = client is None

    def complete(self, request: CompletionRequest) -> CompletionResult:
        payload = self.build_chat_payload(request)
        headers = {"Content-Type": "application/json", **self.auth_headers()}
        model = payload["model"]
        response = self.send_http_request(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        return self.parse_chat_response(response.json(), model=str(model))

    def send_http_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Send a request and translate transport and HTTP failures."""
        kwargs.setdefault("timeout", self.timeout)
        try:
            response = self._client.request(method, url, **kwargs)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise ProviderTemporaryError(
                f"{self.name} request failed temporarily: {type(exc).__name__}."
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderResponseError(
                f"{self.name} request failed: {type(exc).__name__}."
            ) from exc

        self.raise_for_status(response)
        return response

    def auth_headers(self) -> dict[str, str]:
        """Return provider headers without forcing a content type."""
        headers = dict(self._headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def build_chat_payload(self, request: CompletionRequest) -> dict:
        """Build a Chat Completions request body."""
        model = request.model or self.default_model
        if not model:
            raise ProviderConfigurationError(
                f"A model must be configured for provider '{self.name}'."
            )
        encoded_input = json.dumps(
            request.input_data, ensure_ascii=False, default=str
        )
        return {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You enrich structured data. Follow the task exactly and "
                        "return only the requested value, without explanation or "
                        "reasoning."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Task:\n{request.instructions}\n\n"
                        "Input JSON:\n"
                        f"{encoded_input}"
                    ),
                },
            ],
        }

    def raise_for_status(self, response: httpx.Response) -> None:
        """Translate HTTP failures without exposing credentials or response bodies."""
        if response.status_code == 429 or response.status_code >= 500:
            error = ProviderTemporaryError(
                f"{self.name} returned HTTP {response.status_code}."
            )
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    error.retry_after = max(0.0, float(retry_after))
                except ValueError:
                    pass
            raise error
        if response.status_code >= 400:
            raise ProviderResponseError(
                f"{self.name} returned HTTP {response.status_code}."
            )

    def parse_chat_response(
        self, data: Mapping, *, model: str
    ) -> CompletionResult:
        """Parse a Chat Completions response body."""
        try:
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("response content is not text")
            usage = data.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderResponseError(
                f"{self.name} returned an invalid Chat Completions response."
            ) from exc

        return CompletionResult(
            content=content.strip(),
            model=data.get("model") or model,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
