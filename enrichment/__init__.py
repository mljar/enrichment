"""AI data enrichment for pandas DataFrames."""

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

__version__ = "1.0.0"

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
