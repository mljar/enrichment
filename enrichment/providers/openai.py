"""OpenAI provider."""

from __future__ import annotations

import json
import os
from typing import Mapping, Optional

import httpx

from ..exceptions import ProviderConfigurationError, ProviderResponseError
from ..models import (
    BatchItemResult,
    BatchStatus,
    CompletionRequest,
)
from .base import BatchProvider
from .openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider, BatchProvider):
    """OpenAI provider using the modern Chat Completions HTTP API."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-5-nano",
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

    def build_chat_payload(self, request: CompletionRequest) -> dict:
        payload = super().build_chat_payload(request)
        model = str(payload["model"])
        if model == "gpt-5-nano" or model.startswith("gpt-5-nano-"):
            payload["reasoning_effort"] = "minimal"
        return payload

    def submit_batch(
        self, requests: Mapping[str, CompletionRequest]
    ) -> BatchStatus:
        if not requests:
            raise ValueError("A batch must contain at least one request.")
        if len(requests) > 50_000:
            raise ValueError(
                "A single OpenAI batch can contain at most 50,000 requests."
            )

        lines = []
        models = set()
        for custom_id, request in requests.items():
            payload = self.build_chat_payload(request)
            models.add(payload["model"])
            lines.append(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": payload,
                    },
                    ensure_ascii=False,
                )
            )
        if len(models) != 1:
            raise ValueError("All requests in an OpenAI batch must use one model.")

        content = ("\n".join(lines) + "\n").encode("utf-8")
        if len(content) > 200 * 1024 * 1024:
            raise ValueError("OpenAI batch input files cannot exceed 200 MB.")

        upload = self.send_http_request(
            "POST",
            f"{self.base_url}/files",
            headers=self.auth_headers(),
            data={"purpose": "batch"},
            files={"file": ("enrichment.jsonl", content, "application/jsonl")},
        )
        try:
            input_file_id = upload.json()["id"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderResponseError(
                "OpenAI returned an invalid batch file response."
            ) from exc

        created = self.send_http_request(
            "POST",
            f"{self.base_url}/batches",
            headers={"Content-Type": "application/json", **self.auth_headers()},
            json={
                "input_file_id": input_file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "metadata": {"source": "enrichment"},
            },
        )
        return self._parse_batch(created.json())

    def get_batch(self, batch_id: str) -> BatchStatus:
        response = self.send_http_request(
            "GET",
            f"{self.base_url}/batches/{batch_id}",
            headers=self.auth_headers(),
        )
        return self._parse_batch(response.json())

    def get_batch_results(
        self, batch: BatchStatus
    ) -> Mapping[str, BatchItemResult]:
        items = {}
        if batch.output_file_id:
            output = self._get_file_content(batch.output_file_id)
            items.update(self._parse_batch_lines(output))
        if batch.error_file_id:
            errors = self._get_file_content(batch.error_file_id)
            items.update(self._parse_batch_lines(errors))
        return items

    def cancel_batch(self, batch_id: str) -> BatchStatus:
        response = self.send_http_request(
            "POST",
            f"{self.base_url}/batches/{batch_id}/cancel",
            headers=self.auth_headers(),
        )
        return self._parse_batch(response.json())

    def _get_file_content(self, file_id: str) -> str:
        response = self.send_http_request(
            "GET",
            f"{self.base_url}/files/{file_id}/content",
            headers=self.auth_headers(),
        )
        return response.text

    def _parse_batch(self, data: Mapping) -> BatchStatus:
        try:
            counts = data.get("request_counts") or {}
            return BatchStatus(
                id=str(data["id"]),
                status=str(data["status"]),
                input_file_id=data.get("input_file_id"),
                output_file_id=data.get("output_file_id"),
                error_file_id=data.get("error_file_id"),
                total=int(counts.get("total") or 0),
                completed=int(counts.get("completed") or 0),
                failed=int(counts.get("failed") or 0),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderResponseError(
                "OpenAI returned an invalid batch response."
            ) from exc

    def _parse_batch_lines(self, content: str) -> Mapping[str, BatchItemResult]:
        items = {}
        for raw_line in content.splitlines():
            if not raw_line.strip():
                continue
            try:
                line = json.loads(raw_line)
                custom_id = str(line["custom_id"])
                error = line.get("error")
                response = line.get("response") or {}
                if error:
                    message = (
                        error.get("message")
                        or error.get("code")
                        or "Batch request failed."
                    )
                    items[custom_id] = BatchItemResult(
                        custom_id=custom_id, error=str(message)
                    )
                    continue
                status_code = int(response.get("status_code") or 0)
                if status_code >= 400:
                    body = response.get("body") or {}
                    body_error = body.get("error") or {}
                    message = (
                        body_error.get("message")
                        or body_error.get("code")
                        or f"Batch request returned HTTP {status_code}."
                    )
                    items[custom_id] = BatchItemResult(
                        custom_id=custom_id,
                        error=str(message),
                    )
                    continue
                body = response["body"]
                model = str(body.get("model") or self.default_model or "")
                result = self.parse_chat_response(body, model=model)
                items[custom_id] = BatchItemResult(
                    custom_id=custom_id, result=result
                )
            except (ValueError, KeyError, TypeError) as exc:
                raise ProviderResponseError(
                    "OpenAI returned invalid batch output JSONL."
                ) from exc
        return items
