"""MLJAR hosted enrichment provider."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Mapping, Optional

import httpx

from ..exceptions import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderResponseError,
    ProviderTemporaryError,
)
from ..models import (
    BatchItemResult,
    BatchStatus,
    CompletionRequest,
    CompletionResult,
)
from .base import BatchProvider


TOKEN_FILE_ENV = "MLJAR_RUNTIME_TOKEN_FILE"
BASE_URL_ENV = "MLJAR_RUNTIME_LICENSING_BASE_URL"
DEFAULT_BASE_URL = "https://platform.mljar.com"


class MLJARProvider(BatchProvider):
    """Use MLJAR-hosted enrichment with a Studio account token."""

    name = "mljar"
    default_model = "mljar-hosted"

    def __init__(
        self,
        *,
        token_file: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.token_file = token_file or os.getenv(TOKEN_FILE_ENV, "")
        self.base_url = (
            base_url or os.getenv(BASE_URL_ENV) or DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout = timeout
        self._client = client or httpx.Client()
        self._owns_client = client is None

    @classmethod
    def runtime_token_available(cls) -> bool:
        """Return whether the configured Studio token file contains a token."""
        token_file = (os.getenv(TOKEN_FILE_ENV) or "").strip()
        if not token_file:
            return False
        try:
            return bool(Path(token_file).read_text(encoding="utf-8").strip())
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise ProviderConfigurationError(
                "MLJAR Studio authentication file could not be read."
            ) from exc

    def _read_token(self) -> str:
        token_file = self.token_file.strip()
        if not token_file:
            raise ProviderConfigurationError(
                "MLJAR Studio authentication file is not configured."
            )
        try:
            token = Path(token_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ProviderConfigurationError(
                "MLJAR Studio authentication file could not be read."
            ) from exc
        if not token:
            raise ProviderAuthenticationError(
                "MLJAR authentication is missing. Please sign in again."
            )
        return token

    def _headers(self, *, request_id: Optional[str] = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Token {self._read_token()}",
            "Content-Type": "application/json",
        }
        if request_id:
            headers["X-Request-ID"] = request_id
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> httpx.Response:
        kwargs.setdefault("timeout", self.timeout)
        try:
            response = self._client.request(
                method,
                f"{self.base_url}{path}",
                headers=self._headers(request_id=request_id),
                **kwargs,
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise ProviderTemporaryError(
                f"MLJAR request failed temporarily: {type(exc).__name__}."
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderResponseError(
                f"MLJAR request failed: {type(exc).__name__}."
            ) from exc

        if response.status_code in {401, 403}:
            raise ProviderAuthenticationError(
                "MLJAR authentication failed. Please sign in again."
            )
        if response.status_code == 429 or response.status_code >= 500:
            error = ProviderTemporaryError(
                f"MLJAR returned HTTP {response.status_code}."
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
                f"MLJAR returned HTTP {response.status_code}."
            )
        return response

    def _json(self, response: httpx.Response) -> Mapping:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderResponseError(
                "MLJAR returned an invalid JSON response."
            ) from exc
        if not isinstance(payload, Mapping):
            raise ProviderResponseError(
                "MLJAR returned an invalid JSON response."
            )
        return payload

    def complete(self, request: CompletionRequest) -> CompletionResult:
        request_id = request.request_id or self._request_id(request)
        payload = {
            "request_id": request_id,
            "instructions": request.instructions,
            "input_data": request.input_data,
        }
        if request.response_schema is not None:
            payload["response_schema"] = request.response_schema
        response = self._request(
            "POST",
            "/api/ai/enrichment/completion/",
            request_id=request_id,
            json=payload,
        )
        return self._parse_completion(self._json(response))

    def submit_batch(
        self, requests: Mapping[str, CompletionRequest]
    ) -> BatchStatus:
        if not requests:
            raise ValueError("A batch must contain at least one request.")
        if len(requests) > 50_000:
            raise ValueError(
                "A single MLJAR batch can contain at most 50,000 requests."
            )
        items = []
        for custom_id, request in requests.items():
            item = {
                "custom_id": custom_id,
                "request_id": request.request_id or self._request_id(request),
                "instructions": request.instructions,
                "input_data": request.input_data,
            }
            if request.response_schema is not None:
                item["response_schema"] = request.response_schema
            items.append(item)
        request_id = self._batch_request_id(items)
        response = self._request(
            "POST",
            "/api/ai/enrichment/batches/",
            request_id=request_id,
            json={"request_id": request_id, "items": items},
        )
        return self._parse_batch(self._json(response))

    def get_batch(self, batch_id: str) -> BatchStatus:
        response = self._request(
            "GET", f"/api/ai/enrichment/batches/{batch_id}/"
        )
        return self._parse_batch(self._json(response))

    def get_batch_results(
        self, batch: BatchStatus
    ) -> Mapping[str, BatchItemResult]:
        response = self._request(
            "GET", f"/api/ai/enrichment/batches/{batch.id}/results/"
        )
        payload = self._json(response)
        raw_items = payload.get("items")
        if not isinstance(raw_items, list):
            raise ProviderResponseError(
                "MLJAR returned invalid batch results."
            )
        items = {}
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping) or "custom_id" not in raw_item:
                raise ProviderResponseError(
                    "MLJAR returned invalid batch results."
                )
            custom_id = str(raw_item["custom_id"])
            error = raw_item.get("error")
            if error:
                items[custom_id] = BatchItemResult(
                    custom_id=custom_id,
                    error=self._error_message(error),
                )
            else:
                items[custom_id] = BatchItemResult(
                    custom_id=custom_id,
                    result=self._parse_completion(raw_item),
                )
        return items

    def cancel_batch(self, batch_id: str) -> BatchStatus:
        response = self._request(
            "POST", f"/api/ai/enrichment/batches/{batch_id}/cancel/"
        )
        return self._parse_batch(self._json(response))

    def _parse_completion(self, payload: Mapping) -> CompletionResult:
        content = payload.get("content")
        if not isinstance(content, str):
            raise ProviderResponseError(
                "MLJAR returned an invalid enrichment response."
            )
        return CompletionResult(
            content=content.strip(),
            model=str(payload.get("model") or self.default_model),
            input_tokens=self._optional_int(payload.get("input_tokens")),
            output_tokens=self._optional_int(payload.get("output_tokens")),
        )

    def _parse_batch(self, payload: Mapping) -> BatchStatus:
        try:
            counts = payload.get("request_counts") or payload
            return BatchStatus(
                id=str(payload["id"]),
                status=str(payload["status"]),
                total=int(
                    counts.get("total") or counts.get("requested_items") or 0
                ),
                completed=int(
                    counts.get("completed") or counts.get("completed_items") or 0
                ),
                failed=int(
                    counts.get("failed") or counts.get("failed_items") or 0
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderResponseError(
                "MLJAR returned an invalid batch response."
            ) from exc

    def _request_id(self, request: CompletionRequest) -> str:
        return f"enrichment-{uuid.uuid4().hex}"

    def _batch_request_id(self, items: list[dict]) -> str:
        encoded = json.dumps(items, ensure_ascii=False, sort_keys=True, default=str)
        return f"enrichment-batch-{hashlib.sha256(encoded.encode()).hexdigest()[:32]}"

    def _optional_int(self, value) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ProviderResponseError(
                "MLJAR returned invalid token usage."
            ) from exc

    def _error_message(self, error) -> str:
        if isinstance(error, Mapping):
            return str(
                error.get("message")
                or error.get("code")
                or "Enrichment request failed."
            )
        return str(error)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
