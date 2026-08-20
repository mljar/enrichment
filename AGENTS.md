# Agent instructions

## Purpose

`enrichment` adds AI-generated columns to pandas DataFrames. Keep the public
experience centered on the single `enrich()` function.

## Using the package

- Prefer `from enrichment import enrich`.
- Pass exactly one of `input_col` or `input_cols`.
- Always assign the returned DataFrame. `enrich()` does not modify the source
  DataFrame.
- Write a precise prompt that states the expected format or allowed values.
- Inspect the available columns and test uncertain prompts on a small sample
  before processing a large table.
- Leave `use_batch=None` unless the user explicitly requests interactive or
  batch execution. The package automatically batches 50 or more unique inputs
  when the selected provider supports it.
- In MLJAR Studio, do not ask the user for an API key and do not construct a
  provider. Studio authentication is discovered automatically.
- Outside Studio, allow normal provider resolution through `OPENAI_API_KEY`, or
  pass an explicit provider only when the user asks for a specific service.
- Never print, store in generated code, or commit API keys or Studio tokens.

Typical usage:

```python
from enrichment import enrich

result = enrich(
    df,
    input_col="review",
    output_col="sentiment",
    prompt="Classify as positive, negative, or neutral. Return one word.",
)
```

For several input columns:

```python
result = enrich(
    df,
    input_cols=["company_name", "website"],
    output_col="industry",
    prompt="Return the company's industry in 1-3 words.",
)
```

## Working on this repository

- Preserve backward compatibility of the `enrich()` API and exported provider
  classes.
- Keep provider-specific behavior behind the provider abstraction.
- Add or update tests for behavior changes. Unit tests must not make paid API
  requests; live tests remain explicitly opt-in.
- Run `python -m pytest` before finishing a code change.
- Do not commit `.env` files, credentials, build artifacts, or paid-test output.

