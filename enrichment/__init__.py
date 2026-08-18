"""AI data enrichment for pandas DataFrames."""

from .enricher import enrich
from .batch import EnrichmentBatchJob, enrich_batch
from .exceptions import (
    EnrichmentError,
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
    BatchProvider,
    Provider,
    register_provider,
    unregister_provider,
)

__version__ = "0.3.0"

__all__ = [
    "enrich",
    "enrich_batch",
    "EnrichmentBatchJob",
    "Provider",
    "BatchProvider",
    "OpenAIProvider",
    "OpenAICompatibleProvider",
    "CompletionRequest",
    "CompletionResult",
    "BatchStatus",
    "BatchItemResult",
    "EnrichmentReport",
    "EnrichmentError",
    "ProviderError",
    "ProviderTemporaryError",
    "ProviderResponseError",
    "ProviderConfigurationError",
    "register_provider",
    "unregister_provider",
]
