# AI Data Enrichment for pandas

`enrichment` adds AI-generated columns to pandas DataFrames. Describe the value
you need, select one or more input columns, and call `enrich()`.

```python
import pandas as pd
from enrichment import enrich

df = pd.DataFrame(
    {
        "review": [
            "I loved the product!",
            "It arrived broken.",
            "It was okay.",
        ]
    }
)

result = enrich(
    df,
    input_col="review",
    output_col="sentiment",
    prompt="Classify sentiment as positive, negative, or neutral",
)
```

The original DataFrame is not modified. Requests run concurrently, identical
inputs are processed once, temporary provider failures are retried, and result
order always matches the source DataFrame.

## Installation

```bash
pip install enrichment
```

For the default OpenAI provider, set your API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

Then call `enrich()` without configuring a provider. You can also pass
`api_key=` directly for compatibility with earlier versions.

The default OpenAI model is `gpt-5.4-nano`, selected for fast, cost-efficient
classification and data extraction. Override it with `model=` when a task needs
a different quality or cost profile.

## Use several input columns

```python
result = enrich(
    df,
    input_cols=["company_name", "website"],
    output_col="industry",
    prompt="Determine the company's industry",
)
```

The selected values are sent as structured JSON, keeping field names and values
unambiguous.

## OpenAI-compatible providers

Use any service exposing an OpenAI-compatible Chat Completions endpoint:

```python
from enrichment import OpenAICompatibleProvider, enrich

provider = OpenAICompatibleProvider(
    base_url="http://127.0.0.1:1234/v1",
    model="local-model",
)

result = enrich(
    df,
    input_col="review",
    output_col="sentiment",
    prompt="Classify sentiment",
    provider=provider,
)
```

Remote compatible providers can receive a key and additional headers:

```python
provider = OpenAICompatibleProvider(
    base_url="https://provider.example/v1",
    api_key="your-api-key",
    model="provider/model-name",
    headers={"X-App": "My application"},
)
```

## Performance and reliability

The default concurrency is five requests. It can be adjusted for the selected
provider:

```python
result = enrich(
    df,
    input_col="text",
    output_col="topic",
    prompt="Return the main topic",
    max_concurrency=10,
    max_retries=3,
)
```

HTTP 429 responses, timeouts, network failures, and temporary server failures
are retried with backoff. A provider's `Retry-After` response is honored.

## OpenAI Batch API

For large jobs that do not need immediate results, `enrich_batch()` submits an
asynchronous OpenAI Batch API job. OpenAI processes batches within 24 hours at
a lower price than synchronous requests. Duplicate inputs are submitted once,
and output is restored to the original DataFrame order even when OpenAI returns
items out of order.

```python
from enrichment import enrich_batch

result, report = enrich_batch(
    df,
    input_col="text",
    output_col="topic",
    prompt="Return the main topic",
    show_progress=True,
    return_report=True,
)

print(report.batch_id, report.batch_status)
```

By default, the call waits for the batch and returns the enriched DataFrame.
For a long-running job, submit it first and manage it separately:

```python
job = enrich_batch(
    df,
    input_col="text",
    output_col="topic",
    prompt="Return the main topic",
    wait=False,
)

print(job.id, job.status)
job.wait(poll_interval=30)
result = job.result()
```

`job.refresh()` retrieves its current status and `job.cancel()` requests
cancellation. A single OpenAI batch supports at most 50,000 unique requests and
a 200 MB input file. Provider-side batches currently use the built-in
`OpenAIProvider`; other providers can add support by implementing
`BatchProvider`.

## Missing values and errors

Rows where all selected inputs are missing are skipped and receive `pd.NA`.
By default, enrichment stops when a request fails after retries. To keep
processing and store `pd.NA` for failures:

```python
result = enrich(
    df,
    input_col="text",
    output_col="topic",
    prompt="Return the main topic",
    on_error="keep",
)
```

## Execution report

Request a report when you need usage and failure details:

```python
result, report = enrich(
    df,
    input_col="text",
    output_col="topic",
    prompt="Return the main topic",
    return_report=True,
)

print(report.completed)
print(report.unique_requests)
print(report.retries)
print(report.input_tokens, report.output_tokens)
print(report.errors)
```

Token usage is available when the provider reports it.

## Runtime providers

Applications embedding `enrichment` can register a provider for automatic use:

```python
from enrichment import register_provider

register_provider("application-runtime", provider, priority=100)
```

Provider selection follows this order:

1. `provider=` passed to `enrich()`
2. Highest-priority registered runtime provider
3. OpenAI configured through `api_key=` or `OPENAI_API_KEY`

This hook is intended for integrations such as MLJAR Studio, where users should
be able to enrich data without entering provider configuration in every
notebook.

## Custom providers

Implement the small synchronous provider interface:

```python
from enrichment import CompletionResult, Provider


class MyProvider(Provider):
    name = "my-provider"
    default_model = "my-model"

    def complete(self, request):
        value = call_my_service(
            instructions=request.instructions,
            input_data=request.input_data,
            model=request.model or self.default_model,
        )
        return CompletionResult(content=value)
```

## API

```python
enrich(
    df,
    input_col=None,
    output_col=None,
    prompt=None,
    model=None,
    api_key=None,
    show_progress=True,
    *,
    input_cols=None,
    provider=None,
    max_concurrency=5,
    max_retries=3,
    retry_base_delay=0.5,
    on_error="raise",
    return_report=False,
)
```

`input_col` and `input_cols` are mutually exclusive.

```python
enrich_batch(
    df,
    input_col=None,
    output_col=None,
    prompt=None,
    model=None,
    api_key=None,
    show_progress=True,
    *,
    input_cols=None,
    provider=None,
    wait=True,
    poll_interval=10.0,
    timeout=None,
    on_error="raise",
    return_report=False,
)
```

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Live OpenAI tests are skipped by default because they make paid API requests.
Run them explicitly with:

```bash
RUN_LIVE_API_TESTS=1 OPENAI_API_KEY="your-api-key" python -m pytest -m live
```

The live Batch API test has a separate opt-in because completion can take
several minutes:

```bash
RUN_LIVE_BATCH_TESTS=1 OPENAI_API_KEY="your-api-key" \
  python -m pytest -m live -k batch
```
