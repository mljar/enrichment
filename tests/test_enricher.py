import threading
import time

import httpx
import pandas as pd
import pytest

from enrichment import (
    CompletionResult,
    EnrichmentError,
    OpenAICompatibleProvider,
    OpenAIProvider,
    Provider,
    ProviderConfigurationError,
    ProviderResponseError,
    ProviderTemporaryError,
    enrich,
    register_provider,
    unregister_provider,
)
from enrichment.providers.resolver import resolve_provider


class RecordingProvider(Provider):
    name = "recording"
    default_model = "test-model"

    def __init__(self, response=None):
        self.requests = []
        self.response = response
        self._lock = threading.Lock()

    def complete(self, request):
        with self._lock:
            self.requests.append(request)
        content = self.response or "|".join(str(v) for v in request.input_data.values())
        return CompletionResult(content=content, input_tokens=2, output_tokens=1)


def test_existing_single_column_api_is_preserved():
    provider = RecordingProvider()
    source = pd.DataFrame({"review": ["good", "bad"]}, index=[5, 9])

    result = enrich(
        source,
        "review",
        "sentiment",
        "Classify sentiment",
        provider=provider,
        show_progress=False,
    )

    assert list(result["sentiment"]) == ["good", "bad"]
    assert list(result.index) == [5, 9]
    assert "sentiment" not in source.columns


def test_empty_dataframe_returns_an_empty_result_column():
    provider = RecordingProvider()
    source = pd.DataFrame({"review": pd.Series(dtype="object")})

    result, report = enrich(
        source,
        "review",
        "sentiment",
        "Classify sentiment",
        provider=provider,
        show_progress=False,
        return_report=True,
    )

    assert result.empty
    assert "sentiment" in result.columns
    assert report.total_rows == 0
    assert report.unique_requests == 0
    assert provider.requests == []


def test_multiple_input_columns_are_sent_as_structured_data():
    provider = RecordingProvider()
    source = pd.DataFrame(
        {"company": ["MLJAR"], "website": ["https://mljar.com"]}
    )

    result = enrich(
        source,
        input_cols=["company", "website"],
        output_col="industry",
        prompt="Determine the industry",
        provider=provider,
        show_progress=False,
    )

    assert result.loc[0, "industry"] == "MLJAR|https://mljar.com"
    assert provider.requests[0].input_data == {
        "company": "MLJAR",
        "website": "https://mljar.com",
    }


def test_duplicate_inputs_are_requested_once_and_usage_is_not_duplicated():
    provider = RecordingProvider(response="Positive")
    source = pd.DataFrame({"review": ["same", "same", "different"]})

    result, report = enrich(
        source,
        "review",
        "sentiment",
        "Classify sentiment",
        provider=provider,
        show_progress=False,
        return_report=True,
    )

    assert list(result["sentiment"]) == ["Positive", "Positive", "Positive"]
    assert len(provider.requests) == 2
    assert report.total_rows == 3
    assert report.unique_requests == 2
    assert report.completed == 3
    assert report.input_tokens == 4
    assert report.output_tokens == 2


def test_rows_with_only_missing_inputs_are_skipped():
    provider = RecordingProvider()
    source = pd.DataFrame({"left": [None, "value"], "right": [pd.NA, None]})

    result, report = enrich(
        source,
        input_cols=["left", "right"],
        output_col="result",
        prompt="Process",
        provider=provider,
        show_progress=False,
        return_report=True,
    )

    assert pd.isna(result.loc[0, "result"])
    assert result.loc[1, "result"] == "value|None"
    assert report.skipped == 1
    assert report.completed == 1
    assert len(provider.requests) == 1


class RetryingProvider(Provider):
    name = "retrying"
    default_model = "retry-model"

    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        if self.calls <= self.failures:
            error = ProviderTemporaryError("try again")
            error.retry_after = 0
            raise error
        return CompletionResult(content="done")


def test_temporary_failures_are_retried():
    provider = RetryingProvider(failures=2)

    result, report = enrich(
        pd.DataFrame({"text": ["hello"]}),
        "text",
        "result",
        "Reply",
        provider=provider,
        max_retries=2,
        show_progress=False,
        return_report=True,
    )

    assert result.loc[0, "result"] == "done"
    assert provider.calls == 3
    assert report.retries == 2


class FailingProvider(Provider):
    name = "failing"
    default_model = "failure-model"

    def complete(self, request):
        raise ProviderResponseError("invalid request")


def test_on_error_keep_records_failures_and_uses_na():
    result, report = enrich(
        pd.DataFrame({"text": ["hello"]}, index=[42]),
        "text",
        "result",
        "Reply",
        provider=FailingProvider(),
        on_error="keep",
        show_progress=False,
        return_report=True,
    )

    assert pd.isna(result.loc[42, "result"])
    assert report.failed == 1
    assert report.errors[0].indices == [42]


def test_on_error_raise_contains_context_but_not_provider_secrets():
    with pytest.raises(EnrichmentError) as caught:
        enrich(
            pd.DataFrame({"text": ["hello"]}, index=[42]),
            "text",
            "result",
            "Reply",
            provider=FailingProvider(),
            show_progress=False,
        )

    assert caught.value.index == 42
    assert caught.value.provider == "failing"
    assert "invalid request" in str(caught.value)


class ConcurrentProvider(Provider):
    name = "concurrent"
    default_model = "concurrent-model"

    def __init__(self):
        self.active = 0
        self.maximum_active = 0
        self.lock = threading.Lock()

    def complete(self, request):
        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        return CompletionResult(content="ok")


def test_concurrency_is_bounded():
    provider = ConcurrentProvider()
    source = pd.DataFrame({"text": [f"row-{i}" for i in range(8)]})

    enrich(
        source,
        "text",
        "result",
        "Reply",
        provider=provider,
        max_concurrency=3,
        show_progress=False,
    )

    assert 1 < provider.maximum_active <= 3


def test_openai_compatible_request_and_usage():
    captured = {}

    def handler(request):
        captured["request"] = request
        return httpx.Response(
            200,
            json={
                "model": "served-model",
                "choices": [{"message": {"content": "  category  "}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        base_url="http://localhost:1234/v1/",
        api_key="secret-value",
        model="configured-model",
        headers={"X-App": "MLJAR"},
        client=client,
    )

    result, report = enrich(
        pd.DataFrame({"text": ["hello"]}),
        "text",
        "result",
        "Categorize",
        provider=provider,
        show_progress=False,
        return_report=True,
    )

    request = captured["request"]
    assert str(request.url) == "http://localhost:1234/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer secret-value"
    assert request.headers["X-App"] == "MLJAR"
    assert result.loc[0, "result"] == "category"
    assert report.input_tokens == 10
    assert report.output_tokens == 2


def test_openai_provider_uses_current_enrichment_default():
    provider = OpenAIProvider(api_key="test-key")
    try:
        assert provider.default_model == "gpt-5.4-nano"
    finally:
        provider.close()


def test_openai_compatible_error_does_not_expose_key():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(401))
    )
    provider = OpenAICompatibleProvider(
        base_url="http://localhost:1234/v1",
        api_key="do-not-leak-this",
        model="model",
        client=client,
    )

    with pytest.raises(EnrichmentError) as caught:
        enrich(
            pd.DataFrame({"text": ["hello"]}),
            "text",
            "result",
            "Reply",
            provider=provider,
            show_progress=False,
        )

    assert "do-not-leak-this" not in str(caught.value)


def test_registered_provider_has_priority_over_environment(monkeypatch):
    provider = RecordingProvider()
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    register_provider("test-runtime", provider, priority=10)
    try:
        assert resolve_provider() is provider
    finally:
        unregister_provider("test-runtime")


def test_missing_provider_configuration_has_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MLJAR_RUNTIME_TOKEN_FILE", raising=False)
    with pytest.raises(ProviderConfigurationError, match="No AI provider"):
        resolve_provider()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "At least one input column"),
        ({"input_col": "missing"}, "missing input columns"),
        ({"input_col": "text", "input_cols": ["text"]}, "either input_col"),
        ({"input_cols": "text"}, "must be a sequence"),
    ],
)
def test_input_validation(kwargs, message):
    base = {
        "df": pd.DataFrame({"text": ["hello"]}),
        "output_col": "result",
        "prompt": "Reply",
        "provider": RecordingProvider(),
        "show_progress": False,
    }
    with pytest.raises(ValueError, match=message):
        enrich(**base, **kwargs)
