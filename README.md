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

The original DataFrame is not modified. Identical inputs are processed once,
temporary provider failures are retried, and result order always matches the
source DataFrame. Small jobs run concurrently. Large jobs automatically use a
provider-side batch when the selected provider supports it.

## Installation

```bash
pip install enrichment
```

Inside MLJAR Studio, hosted enrichment is selected automatically from the
signed-in account. No OpenAI API key or provider configuration is required.
Studio supplies the path to its account-token file, and `enrichment` rereads
that file for every operation so login, logout, and token rotation take effect
without restarting Jupyter.

Outside MLJAR Studio, set an OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

Then call `enrich()` without configuring a provider. You can also pass
`api_key=` directly.

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

## MLJAR hosted provider

MLJAR Studio configures its hosted provider through:

```text
MLJAR_RUNTIME_TOKEN_FILE
MLJAR_RUNTIME_LICENSING_BASE_URL
```

`MLJAR_RUNTIME_TOKEN_FILE` contains a path, not the token itself. The provider
reads the account token from that file and sends it using standard token
authentication. It does not use a device header, legacy JWT, or additional AI
session token.

If the file is missing or empty, provider selection can fall back to
`OPENAI_API_KEY`. If the file contains an invalid token or cannot be read, the
operation fails with a sign-in or configuration error instead of silently
charging the OpenAI account.

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

## Automatic batch processing

You always call `enrich()`. When a job contains at least 50 unique non-empty
inputs and the selected provider supports batches, `enrich()` automatically
submits a provider-side batch. Smaller jobs use concurrent requests. Providers
without batch support continue using concurrent requests for every job.

```python
from enrichment import enrich

result, report = enrich(
    df,
    input_col="text",
    output_col="topic",
    prompt="Return the main topic",
    show_progress=True,
    return_report=True,
)

print(report.batch_id, report.batch_status)
```

Override the automatic choice when needed:

```python
# Always use a provider batch, even for a small job.
result = enrich(df, ..., use_batch=True)

# Never use a provider batch, even for a large job.
result = enrich(df, ..., use_batch=False)
```

The default `use_batch=None` selects the execution mode automatically. Forcing
batch mode requires a provider that implements `BatchProvider`.

The return value does not change: `enrich()` waits for processing and returns
the enriched DataFrame. OpenAI processes asynchronous batches within 24 hours
at 50% lower cost than synchronous requests. Output is restored to the original
DataFrame order even when the provider returns items out of order. A single
OpenAI batch supports at most 50,000 unique requests and a 200 MB input file.
Other providers can add automatic batch support by implementing
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
2. OpenAI configured explicitly through `api_key=`
3. Highest-priority registered runtime provider
4. MLJAR account token found through `MLJAR_RUNTIME_TOKEN_FILE`
5. OpenAI configured through `OPENAI_API_KEY`

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
    use_batch=None,
    on_error="raise",
    return_report=False,
)
```

`input_col` and `input_cols` are mutually exclusive. Set `use_batch=True` to
force batch processing, `False` to disable it, or leave it as `None` for
automatic selection.

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
