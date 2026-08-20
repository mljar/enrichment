"""Add AI-generated columns to pandas DataFrames.

Agent usage guidance:

* Prefer the top-level :func:`enrich` function.
* Pass ``input_col`` for one source column or ``input_cols`` for several.
* Assign the returned DataFrame; the input DataFrame is never modified.
* Make ``prompt`` explicit about allowed values, output format, and length.
* Keep ``use_batch=None`` for automatic execution. Providers that support
  batches automatically use them for 50 or more unique inputs.
* In MLJAR Studio, call :func:`enrich` without an API key or provider. Studio
  authentication is resolved automatically.
* Never place credentials in generated notebook code. Outside Studio, prefer
  environment-based configuration or an explicitly requested provider.

Example::

    from enrichment import enrich

    enriched = enrich(
        df,
        input_col="review",
        output_col="sentiment",
        prompt=(
            "Classify as positive, negative, or neutral. Return one word."
        ),
    )

See the project ``AGENTS.md`` and README for multi-column, reporting, and
provider examples.
"""

from .enricher import enrich
from .exceptions import (
    EnrichmentError,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderResponseError,
    ProviderTemporaryError,
)
from .models import (
    BatchItemResult,
    BatchStatus,
    CompletionRequest,
    CompletionResult,
    EnrichmentReport,
)
from .providers import (
    OpenAICompatibleProvider,
    OpenAIProvider,
    MLJARProvider,
    BatchProvider,
    Provider,
    register_provider,
    unregister_provider,
)

__version__ = "1.0.2"

__all__ = [
    "enrich",
    "Provider",
    "BatchProvider",
    "MLJARProvider",
    "OpenAIProvider",
    "OpenAICompatibleProvider",
    "CompletionRequest",
    "CompletionResult",
    "BatchStatus",
    "BatchItemResult",
    "EnrichmentReport",
    "EnrichmentError",
    "ProviderError",
    "ProviderAuthenticationError",
    "ProviderTemporaryError",
    "ProviderResponseError",
    "ProviderConfigurationError",
    "register_provider",
    "unregister_provider",
]
