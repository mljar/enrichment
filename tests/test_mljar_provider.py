import json

import httpx
import pandas as pd
import pytest

from enrichment import (
    CompletionRequest,
    MLJARProvider,
    OpenAIProvider,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    enrich,
)
from enrichment.providers.resolver import resolve_provider


def _set_token_file(monkeypatch, tmp_path, value="account-token"):
    token_file = tmp_path / "runtime_token.txt"
    token_file.write_text(value, encoding="utf-8")
    monkeypatch.setenv("MLJAR_RUNTIME_TOKEN_FILE", str(token_file))
    return token_file


def test_resolver_prefers_mljar_token_file_over_openai_environment(
    monkeypatch, tmp_path
):
    _set_token_file(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    provider = resolve_provider()
    try:
        assert isinstance(provider, MLJARProvider)
    finally:
        provider.close()


def test_explicit_api_key_precedes_mljar_token_file(monkeypatch, tmp_path):
    _set_token_file(monkeypatch, tmp_path)

    provider = resolve_provider(api_key="explicit-openai-key")
    try:
        assert isinstance(provider, OpenAIProvider)
    finally:
        provider.close()


def test_mljar_provider_uses_runtime_licensing_url(monkeypatch, tmp_path):
    _set_token_file(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "MLJAR_RUNTIME_LICENSING_BASE_URL", "https://platform.example/"
    )

    provider = resolve_provider()
    try:
        assert isinstance(provider, MLJARProvider)
        assert provider.base_url == "https://platform.example"
    finally:
        provider.close()


@pytest.mark.parametrize("token_file_value", [None, "", "  \n"])
def test_missing_or_empty_mljar_token_falls_back_to_openai(
    monkeypatch, tmp_path, token_file_value
):
    if token_file_value is None:
        monkeypatch.setenv(
            "MLJAR_RUNTIME_TOKEN_FILE", str(tmp_path / "missing-token.txt")
        )
    else:
        _set_token_file(monkeypatch, tmp_path, token_file_value)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    provider = resolve_provider()
    try:
        assert isinstance(provider, OpenAIProvider)
    finally:
        provider.close()


def test_unreadable_token_file_does_not_fall_back_to_openai(monkeypatch, tmp_path):
    token_file = _set_token_file(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    original_read_text = type(token_file).read_text

    def failing_read_text(path, *args, **kwargs):
        if path == token_file:
            raise PermissionError("not readable")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(type(token_file), "read_text", failing_read_text)

    with pytest.raises(ProviderConfigurationError, match="could not be read"):
        resolve_provider()


def test_mljar_complete_reads_rotated_token_for_every_request(tmp_path):
    token_file = tmp_path / "runtime_token.txt"
    token_file.write_text("first-token", encoding="utf-8")
    authorization_headers = []

    def handler(request):
        authorization_headers.append(request.headers["Authorization"])
        return httpx.Response(
            200,
            json={
                "content": "category",
                "model": "gpt-5-nano",
                "input_tokens": 12,
                "output_tokens": 1,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = MLJARProvider(
        token_file=str(token_file),
        base_url="https://platform.example",
        client=client,
    )
    request = CompletionRequest(
        instructions="Classify",
        input_data={"text": "hello"},
        request_id="request-1",
    )

    first = provider.complete(request)
    token_file.write_text("second-token", encoding="utf-8")
    second = provider.complete(request)

    assert authorization_headers == [
        "Token first-token",
        "Token second-token",
    ]
    assert first.content == second.content == "category"
    assert first.input_tokens == 12
    assert first.output_tokens == 1


def test_mljar_sync_retry_reuses_request_id(tmp_path):
    token_file = tmp_path / "runtime_token.txt"
    token_file.write_text("account-token", encoding="utf-8")
    request_ids = []

    def handler(request):
        request_ids.append(request.headers["X-Request-ID"])
        if len(request_ids) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"content": "done"})

    provider = MLJARProvider(
        token_file=str(token_file),
        base_url="https://platform.example",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = enrich(
        pd.DataFrame({"text": ["hello"]}),
        "text",
        "result",
        "Process",
        provider=provider,
        max_retries=1,
        retry_base_delay=0,
        show_progress=False,
    )

    assert result.loc[0, "result"] == "done"
    assert len(request_ids) == 2
    assert request_ids[0] == request_ids[1]
    assert request_ids[0].startswith("enrichment-")


def test_mljar_authentication_error_does_not_expose_token(tmp_path):
    token_file = tmp_path / "runtime_token.txt"
    token_file.write_text("do-not-expose-this-token", encoding="utf-8")
    provider = MLJARProvider(
        token_file=str(token_file),
        base_url="https://platform.example",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    401, text=request.headers["Authorization"]
                )
            )
        ),
    )

    with pytest.raises(ProviderAuthenticationError) as caught:
        provider.complete(
            CompletionRequest(
                instructions="Classify",
                input_data={"text": "hello"},
            )
        )

    assert "do-not-expose-this-token" not in str(caught.value)
    assert "sign in again" in str(caught.value).lower()


def test_mljar_sends_only_account_authentication_header(tmp_path):
    token_file = tmp_path / "runtime_token.txt"
    token_file.write_text("account-token", encoding="utf-8")
    captured = {}

    def handler(request):
        captured["headers"] = request.headers
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"content": "positive"})

    provider = MLJARProvider(
        token_file=str(token_file),
        base_url="https://platform.example/",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = provider.complete(
        CompletionRequest(
            instructions="Classify sentiment",
            input_data={"review": "Great"},
            request_id="stable-request-id",
        )
    )

    assert result.content == "positive"
    assert captured["headers"]["Authorization"] == "Token account-token"
    assert "X-MLJAR-Device-ID" not in captured["headers"]
    assert "X-AI-Session-Token" not in captured["headers"]
    assert captured["headers"]["X-Request-ID"] == "stable-request-id"
    assert captured["payload"] == {
        "request_id": "stable-request-id",
        "instructions": "Classify sentiment",
        "input_data": {"review": "Great"},
    }


def test_mljar_batch_lifecycle(tmp_path):
    token_file = tmp_path / "runtime_token.txt"
    token_file.write_text("account-token", encoding="utf-8")
    captured = {}

    def handler(request):
        path = request.url.path
        if request.method == "POST" and path == "/api/ai/enrichment/batches/":
            captured["submit"] = json.loads(request.content)
            captured["submit_request_id"] = request.headers["X-Request-ID"]
            return httpx.Response(
                202,
                json={
                    "id": "job-1",
                    "status": "submitted",
                    "requested_items": 2,
                },
            )
        if request.method == "GET" and path == "/api/ai/enrichment/batches/job-1/":
            return httpx.Response(
                200,
                json={
                    "id": "job-1",
                    "status": "completed",
                    "requested_items": 2,
                    "completed_items": 1,
                    "failed_items": 1,
                },
            )
        if path == "/api/ai/enrichment/batches/job-1/results/":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "custom_id": "item-2",
                            "error": "invalid input",
                        },
                        {
                            "custom_id": "item-1",
                            "content": "positive",
                            "model": "gpt-5-nano",
                            "input_tokens": 15,
                            "output_tokens": 1,
                        },
                    ]
                },
            )
        if request.method == "POST" and path.endswith("/cancel/"):
            return httpx.Response(
                200,
                json={"id": "job-1", "status": "cancelling"},
            )
        raise AssertionError(f"Unexpected request: {request.method} {path}")

    provider = MLJARProvider(
        token_file=str(token_file),
        base_url="https://platform.example",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    requests = {
        "item-1": CompletionRequest(
            instructions="Classify",
            input_data={"text": "great"},
            request_id="request-1",
        ),
        "item-2": CompletionRequest(
            instructions="Classify",
            input_data={"text": "bad"},
            request_id="request-2",
        ),
    }

    submitted = provider.submit_batch(requests)
    completed = provider.get_batch(submitted.id)
    results = provider.get_batch_results(completed)
    cancelled = provider.cancel_batch(submitted.id)

    assert submitted.id == "job-1"
    assert submitted.total == 2
    assert completed.status == "completed"
    assert completed.completed == 1
    assert completed.failed == 1
    assert results["item-1"].result.content == "positive"
    assert results["item-1"].result.input_tokens == 15
    assert results["item-2"].error == "invalid input"
    assert cancelled.status == "cancelling"
    assert captured["submit"]["items"][0]["request_id"] == "request-1"
    assert captured["submit_request_id"].startswith("enrichment-batch-")
