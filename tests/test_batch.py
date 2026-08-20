import json

import httpx
import pandas as pd
import pytest

from enrichment import (
    BatchItemResult,
    BatchProvider,
    BatchStatus,
    CompletionResult,
    EnrichmentError,
    OpenAIProvider,
    Provider,
    enrich,
)
from enrichment.batch import EnrichmentBatchJob, _enrich_batch


class RecordingBatchProvider(BatchProvider):
    name = "recording-batch"
    default_model = "batch-model"

    def __init__(self, results=None):
        self.requests = {}
        self.results = results or {}
        self.cancelled = False

    def complete(self, request):
        raise AssertionError("Provider-side batches must not call complete().")

    def submit_batch(self, requests):
        self.requests = dict(requests)
        return BatchStatus(
            id="batch-1",
            status="in_progress",
            total=len(requests),
        )

    def get_batch(self, batch_id):
        assert batch_id == "batch-1"
        return BatchStatus(
            id=batch_id,
            status="completed",
            output_file_id="output-1",
            total=len(self.requests),
            completed=sum(item.result is not None for item in self.results.values()),
            failed=sum(item.error is not None for item in self.results.values()),
        )

    def get_batch_results(self, batch):
        assert batch.id == "batch-1"
        return self.results

    def cancel_batch(self, batch_id):
        self.cancelled = True
        return BatchStatus(id=batch_id, status="cancelling")


class AutomaticBatchProvider(RecordingBatchProvider):
    def __init__(self):
        super().__init__()
        self.concurrent_requests = []

    def complete(self, request):
        self.concurrent_requests.append(request)
        return CompletionResult(content="concurrent")

    def submit_batch(self, requests):
        self.results = {
            custom_id: BatchItemResult(
                custom_id=custom_id,
                result=CompletionResult(content="batched"),
            )
            for custom_id in requests
        }
        return super().submit_batch(requests)


def test_enrich_automatically_uses_batch_for_many_unique_inputs():
    provider = AutomaticBatchProvider()
    source = pd.DataFrame({"text": [f"row-{i}" for i in range(50)]})

    result, report = enrich(
        source,
        "text",
        "result",
        "Classify",
        provider=provider,
        show_progress=False,
        return_report=True,
    )

    assert len(provider.requests) == 50
    assert provider.concurrent_requests == []
    assert set(result["result"]) == {"batched"}
    assert report.batch_id == "batch-1"


def test_enrich_keeps_concurrent_path_for_small_jobs():
    provider = AutomaticBatchProvider()

    result, report = enrich(
        pd.DataFrame({"text": ["first", "second"]}),
        "text",
        "result",
        "Classify",
        provider=provider,
        show_progress=False,
        return_report=True,
    )

    assert len(provider.concurrent_requests) == 2
    assert provider.requests == {}
    assert set(result["result"]) == {"concurrent"}
    assert report.batch_id is None


def test_duplicate_rows_do_not_trigger_automatic_batch():
    provider = AutomaticBatchProvider()

    enrich(
        pd.DataFrame({"text": ["same"] * 50}),
        "text",
        "result",
        "Classify",
        provider=provider,
        show_progress=False,
    )

    assert len(provider.concurrent_requests) == 1
    assert provider.requests == {}


def test_enrich_can_force_batch_for_a_small_job():
    provider = AutomaticBatchProvider()

    result, report = enrich(
        pd.DataFrame({"text": ["first", "second"]}),
        "text",
        "result",
        "Classify",
        provider=provider,
        use_batch=True,
        show_progress=False,
        return_report=True,
    )

    assert len(provider.requests) == 2
    assert provider.concurrent_requests == []
    assert set(result["result"]) == {"batched"}
    assert report.batch_id == "batch-1"


def test_enrich_can_disable_batch_for_a_large_job():
    provider = AutomaticBatchProvider()

    result, report = enrich(
        pd.DataFrame({"text": [f"row-{i}" for i in range(50)]}),
        "text",
        "result",
        "Classify",
        provider=provider,
        use_batch=False,
        show_progress=False,
        return_report=True,
    )

    assert len(provider.concurrent_requests) == 50
    assert provider.requests == {}
    assert set(result["result"]) == {"concurrent"}
    assert report.batch_id is None


def test_forced_batch_requires_a_batch_provider():
    with pytest.raises(TypeError, match="does not support provider-side batches"):
        enrich(
            pd.DataFrame({"text": ["hello"]}),
            "text",
            "result",
            "Classify",
            provider=SynchronousOnlyProvider(),
            use_batch=True,
            show_progress=False,
        )


def test_use_batch_rejects_non_boolean_values():
    with pytest.raises(ValueError, match="True, False, or None"):
        enrich(
            pd.DataFrame({"text": ["hello"]}),
            "text",
            "result",
            "Classify",
            provider=SynchronousOnlyProvider(),
            use_batch="yes",
            show_progress=False,
        )


def test_batch_maps_out_of_order_results_and_deduplicates_inputs():
    provider = RecordingBatchProvider(
        results={
            "enrichment-1": BatchItemResult(
                custom_id="enrichment-1",
                result=CompletionResult(
                    content="second",
                    input_tokens=7,
                    output_tokens=2,
                ),
            ),
            "enrichment-0": BatchItemResult(
                custom_id="enrichment-0",
                result=CompletionResult(
                    content="first",
                    input_tokens=5,
                    output_tokens=1,
                ),
            ),
        }
    )
    source = pd.DataFrame({"text": ["same", "same", "different", None]})

    result, report = _enrich_batch(
        source,
        "text",
        "result",
        "Classify",
        provider=provider,
        show_progress=False,
        poll_interval=0.001,
        return_report=True,
    )

    assert list(result.loc[:2, "result"]) == ["first", "first", "second"]
    assert pd.isna(result.loc[3, "result"])
    assert len(provider.requests) == 2
    assert report.total_rows == 4
    assert report.unique_requests == 2
    assert report.completed == 3
    assert report.skipped == 1
    assert report.input_tokens == 12
    assert report.output_tokens == 3
    assert report.batch_id == "batch-1"
    assert report.batch_status == "completed"
    assert "result" not in source.columns


def test_batch_can_keep_item_failures():
    provider = RecordingBatchProvider(
        results={
            "enrichment-0": BatchItemResult(
                custom_id="enrichment-0", error="invalid input"
            )
        }
    )

    result, report = _enrich_batch(
        pd.DataFrame({"text": ["hello"]}, index=[42]),
        "text",
        "result",
        "Classify",
        provider=provider,
        show_progress=False,
        poll_interval=0.001,
        on_error="keep",
        return_report=True,
    )

    assert pd.isna(result.loc[42, "result"])
    assert report.failed == 1
    assert report.errors[0].indices == [42]
    assert report.errors[0].error == "invalid input"


def test_batch_raises_with_dataframe_context_for_item_failure():
    provider = RecordingBatchProvider(
        results={
            "enrichment-0": BatchItemResult(
                custom_id="enrichment-0", error="invalid input"
            )
        }
    )

    with pytest.raises(EnrichmentError) as caught:
        _enrich_batch(
            pd.DataFrame({"text": ["hello"]}, index=[42]),
            "text",
            "result",
            "Classify",
            provider=provider,
            show_progress=False,
            poll_interval=0.001,
        )

    assert caught.value.index == 42
    assert caught.value.provider == "recording-batch"


def test_batch_can_be_managed_without_waiting():
    provider = RecordingBatchProvider()

    job = _enrich_batch(
        pd.DataFrame({"text": ["hello"]}),
        "text",
        "result",
        "Classify",
        provider=provider,
        wait=False,
    )

    assert isinstance(job, EnrichmentBatchJob)
    assert job.id == "batch-1"
    assert job.status == "in_progress"
    assert job.cancel().status == "cancelling"
    assert provider.cancelled


class SynchronousOnlyProvider(Provider):
    name = "sync-only"

    def complete(self, request):
        return CompletionResult(content="unused")


def test_batch_rejects_provider_without_batch_support():
    with pytest.raises(TypeError, match="does not support provider-side batches"):
        _enrich_batch(
            pd.DataFrame({"text": ["hello"]}),
            "text",
            "result",
            "Classify",
            provider=SynchronousOnlyProvider(),
            wait=False,
        )


def test_openai_batch_http_workflow_and_output_parsing():
    requests = []

    def handler(request):
        requests.append(request)
        path = request.url.path
        if request.method == "POST" and path == "/v1/files":
            assert b'enrichment-0' in request.content
            assert b'gpt-5-nano' in request.content
            assert b'"reasoning_effort": "minimal"' in request.content
            return httpx.Response(200, json={"id": "file-input"})
        if request.method == "POST" and path == "/v1/batches":
            body = json.loads(request.content)
            assert body == {
                "input_file_id": "file-input",
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "metadata": {"source": "enrichment"},
            }
            return httpx.Response(
                200,
                json={
                    "id": "batch-live-shape",
                    "status": "validating",
                    "input_file_id": "file-input",
                    "request_counts": {"total": 1, "completed": 0, "failed": 0},
                },
            )
        if request.method == "GET" and path == "/v1/batches/batch-live-shape":
            return httpx.Response(
                200,
                json={
                    "id": "batch-live-shape",
                    "status": "completed",
                    "input_file_id": "file-input",
                    "output_file_id": "file-output",
                    "request_counts": {"total": 1, "completed": 1, "failed": 0},
                },
            )
        if request.method == "GET" and path == "/v1/files/file-output/content":
            return httpx.Response(
                200,
                text=json.dumps(
                    {
                        "id": "batch-request-1",
                        "custom_id": "enrichment-0",
                        "response": {
                            "status_code": 200,
                            "request_id": "request-1",
                            "body": {
                                "model": "gpt-5-nano",
                                "choices": [
                                    {"message": {"content": " positive "}}
                                ],
                                "usage": {
                                    "prompt_tokens": 20,
                                    "completion_tokens": 1,
                                },
                            },
                        },
                        "error": None,
                    }
                )
                + "\n",
            )
        raise AssertionError(f"Unexpected request: {request.method} {path}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="test-key", client=client)

    result, report = _enrich_batch(
        pd.DataFrame({"text": ["great"]}),
        "text",
        "sentiment",
        "Return sentiment",
        provider=provider,
        show_progress=False,
        poll_interval=0.001,
        return_report=True,
    )

    assert result.loc[0, "sentiment"] == "positive"
    assert report.batch_id == "batch-live-shape"
    assert report.input_tokens == 20
    assert report.output_tokens == 1
    assert requests[0].headers["Authorization"] == "Bearer test-key"
