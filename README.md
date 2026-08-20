# enrichment

![Enrichment — enrich your data with AI](https://raw.githubusercontent.com/mljar/enrichment/main/media/enrichment-banner.webp)

**Add AI columns to your pandas DataFrame.**

[![PyPI](https://img.shields.io/pypi/v/enrichment)](https://pypi.org/project/enrichment/)
[![Python](https://img.shields.io/pypi/pyversions/enrichment)](https://pypi.org/project/enrichment/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

You have a table. You want a new column that only a human could fill in — like "is this review happy or angry?" or "what industry is this company in?".

Write what you want in normal English. `enrichment` asks an AI model for every row and gives you back a new table.

---

## What it does

**Before:**

| review |
| --- |
| I loved the product! |
| It arrived broken. |
| It was okay. |

**Your instruction:** *"Classify sentiment as positive, negative, or neutral"*

**After:**

| review | sentiment |
| --- | --- |
| I loved the product! | positive |
| It arrived broken. | negative |
| It was okay. | neutral |

That is the whole idea. One function, one sentence of instructions, one new column.

---

## Your first enrichment

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
    input_col="review",       # which column the AI reads
    output_col="sentiment",   # name of the new column
    prompt="Classify sentiment as positive, negative, or neutral",
)

print(result)
```

That's it. Four things to fill in: your table, the column to read, the column to create, and what you want.

**Two things worth knowing:**

- Your original `df` is **not changed**. You always get a new DataFrame back.
- If the same text appears twice, it is sent to the AI **once**. You don't pay twice for the same row.

---

## More things you can do

The prompt is just plain English, so you are not limited to sentiment. A few ideas:

```python
# Pull a value out of messy text
enrich(df, input_col="address", output_col="city",
       prompt="Extract the city name")

# Sort things into groups
enrich(df, input_col="ticket", output_col="team",
       prompt="Route to one team: billing, technical, or sales")

# Yes / no questions
enrich(df, input_col="email", output_col="is_spam",
       prompt="Answer yes or no: is this email spam?")

# Translate
enrich(df, input_col="comment", output_col="english",
       prompt="Translate to English")

# Clean up untidy data
enrich(df, input_col="job_title", output_col="clean_title",
       prompt="Rewrite as a standard job title, e.g. 'Software Engineer'")

# Summarize
enrich(df, input_col="article", output_col="summary",
       prompt="Summarize in one short sentence")
```

---

## Reading more than one column

Sometimes one column is not enough context. Use `input_cols` (note the **s**) and pass a list:

```python
result = enrich(
    df,
    input_cols=["company_name", "website"],
    output_col="industry",
    prompt="Determine the company's industry",
)
```

The AI sees both values together, with their column names, so it knows which is which.

Use either `input_col` or `input_cols` — not both at the same time.

---

## Tips for writing good prompts

The prompt is the most important part. Small changes make a big difference.

| Instead of | Try |
| --- | --- |
| "sentiment" | "Classify sentiment as positive, negative, or neutral" |
| "what is this about" | "Return the main topic in 1-3 words" |
| "clean this" | "Return only the phone number, digits only" |

Three simple rules:

1. **List the allowed answers.** "positive, negative, or neutral" gives you a tidy column. "How does this person feel?" gives you paragraphs.
2. **Say how long the answer should be.** "in one word", "in one short sentence".
3. **Test on a few rows first.** Run `df.head(10)` before running 10,000 rows.

```python
# Try it small first
sample = enrich(df.head(10), input_col="review", output_col="sentiment",
                prompt="Classify sentiment as positive, negative, or neutral")
print(sample)
```

---

## Big tables

You don't have to do anything special for large tables — just call `enrich()` as usual.

- **Small jobs** are sent as several requests at the same time, so they finish faster.
- **Big jobs** (50 or more unique values) are automatically sent as one batch, if your provider supports it. On OpenAI, batches cost **50% less**. They can take up to 24 hours, but usually much less.

Rows always come back in the original order, even when the provider returns them mixed up.

Want to decide yourself?

```python
result = enrich(df, ..., use_batch=True)   # always batch
result = enrich(df, ..., use_batch=False)  # never batch
result = enrich(df, ..., use_batch=None)   # default: decide for me
```

OpenAI batch limits: up to 50,000 unique rows and 200 MB per batch.

---

## When something goes wrong

Network problems happen. `enrichment` handles the common ones for you: rate limits (HTTP 429), timeouts, and temporary server errors are retried automatically with a growing wait time.

**Empty rows** are skipped and get `pd.NA` — no API call, no cost.

**Failed rows:** by default the whole job stops if a row keeps failing. If you'd rather finish the job and mark the bad rows, use `on_error="keep"`:

```python
result = enrich(
    df,
    input_col="text",
    output_col="topic",
    prompt="Return the main topic",
    on_error="keep",   # failed rows become pd.NA instead of stopping everything
)
```

You can also slow down or speed up the requests:

```python
result = enrich(
    df,
    input_col="text",
    output_col="topic",
    prompt="Return the main topic",
    max_concurrency=10,   # requests at the same time (default 5)
    max_retries=3,        # tries per row before giving up
)
```

---

## See what happened

Add `return_report=True` to get a small report along with your table. Useful for checking cost and errors.

```python
result, report = enrich(
    df,
    input_col="text",
    output_col="topic",
    prompt="Return the main topic",
    return_report=True,
)

print(report.completed)         # how many rows were filled
print(report.unique_requests)   # how many calls were actually sent
print(report.retries)           # how many retries were needed
print(report.input_tokens, report.output_tokens)   # usage, for cost
print(report.errors)            # what failed, if anything
```

Note that `enrich()` now returns **two** things, so you need two variables on the left.

---

## Choosing the model

The default OpenAI model is `gpt-5-nano`. It is fast and cheap, and it's a good fit for sorting and extracting data — which is most of what people use this for.

Need better quality on a hard task? Pick another model:

```python
result = enrich(df, ..., model="gpt-5.4")
```

---

## Using other AI providers

You are not locked into OpenAI. Anything that speaks the OpenAI Chat Completions format works — including models running on your own machine.

**A local model (nothing leaves your computer):**

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

**A hosted provider with a key:**

```python
provider = OpenAICompatibleProvider(
    base_url="https://provider.example/v1",
    api_key="your-api-key",
    model="provider/model-name",
    headers={"X-App": "My application"},
)
```

---

## All the settings

```python
enrich(
    df,
    input_col=None,          # column to read
    output_col=None,         # new column to create
    prompt=None,             # what you want, in plain English
    model=None,              # model name, provider default if empty
    api_key=None,            # key, if you don't use an env variable
    show_progress=True,      # show a progress bar
    input_cols=None,         # several columns to read (instead of input_col)
    provider=None,           # custom provider object
    max_concurrency=5,       # parallel requests
    max_retries=3,           # tries per row
    retry_base_delay=0.5,    # seconds before the first retry
    use_batch=None,          # True / False / None (automatic)
    on_error="raise",        # "raise" to stop, "keep" to fill pd.NA
    return_report=False,     # also return an execution report
)
```

---

## Questions people ask

**Do I need to know anything about AI?**
No. If you can write a sentence and use pandas, you're ready.

**Is my data sent to the internet?**
Yes, to whichever provider you choose. If your data cannot leave your machine, run a local model and pass it through `OpenAICompatibleProvider`.

**How much does it cost?**
It depends on your provider and model, not on this package. Two things keep it low: repeated values are only sent once, and big jobs use cheaper batches. Run `return_report=True` to see your exact token usage.

**Will it change my original DataFrame?**
No. You always get a new one back.

**What happens to empty cells?**
They are skipped and filled with `pd.NA`. No API call is made for them.

**Can I get the same answer every time?**
Mostly, but AI models can vary a little. For anything important, check a sample of the output yourself.

---

## Install

```bash
pip install enrichment
```

You also need pandas, which comes along with the install.

---

## Get an API key

`enrichment` does not have its own AI. It talks to an AI provider for you. The easiest one to start with is OpenAI.

1. Go to [platform.openai.com](https://platform.openai.com/api-keys) and create an account.
2. Add a payment method (you pay for what you use — usually cents for small tables).
3. Create an API key and copy it. It looks like `sk-...`.
4. Tell your computer about it:

**Mac / Linux:**

```bash
export OPENAI_API_KEY="your-api-key"
```

**Windows (PowerShell):**

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

Or, if you prefer, pass it straight to the function:

```python
enrich(df, ..., api_key="your-api-key")
```

> **Using MLJAR Studio?** You can skip this whole section. Studio signs you in and picks the provider for you. No API key needed.

---

## For advanced users

<details>
<summary><b>Writing your own provider</b></summary>

The provider interface is small and synchronous:

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

Add automatic batch support by also implementing `BatchProvider`.

</details>

<details>
<summary><b>Registering a runtime provider</b></summary>

Applications that embed `enrichment` can register a provider so users never configure anything:

```python
from enrichment import register_provider

register_provider("application-runtime", provider, priority=100)
```

Providers are chosen in this order:

1. `provider=` passed to `enrich()`
2. OpenAI configured explicitly with `api_key=`
3. Highest-priority registered runtime provider
4. MLJAR account token from `MLJAR_RUNTIME_TOKEN_FILE`
5. OpenAI configured through `OPENAI_API_KEY`

</details>

<details>
<summary><b>Development</b></summary>

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Live OpenAI tests are skipped by default because they make paid API requests:

```bash
RUN_LIVE_API_TESTS=1 OPENAI_API_KEY="your-api-key" python -m pytest -m live
```

The live Batch API test opts in separately, because it can take several minutes:

```bash
RUN_LIVE_BATCH_TESTS=1 OPENAI_API_KEY="your-api-key" \
  python -m pytest -m live -k batch
```

</details>

---

## License

Apache 2.0. See [LICENSE](LICENSE).

Made by [MLJAR](https://mljar.com). Found a bug or have an idea? [Open an issue](https://github.com/mljar/enrichment/issues).
