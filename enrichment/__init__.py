"""AI data enrichment for pandas DataFrames."""

from .enricher import enrich
from .exceptions import (
    EnrichmentError,
    ProviderConfigurationError,
    ProviderError,
    ProviderResponseError,
    ProviderTemporaryError,
)
from .models import CompletionRequest, CompletionResult, EnrichmentReport
from .providers import (
    OpenAICompatibleProvider,
    OpenAIProvider,
    Provider,
    register_provider,
    unregister_provider,
)

__version__ = "0.2.0"

__all__ = [
    "enrich",
    "Provider",
    "OpenAIProvider",
    "OpenAICompatibleProvider",
    "CompletionRequest",
    "CompletionResult",
    "EnrichmentReport",
    "EnrichmentError",
    "ProviderError",
    "ProviderTemporaryError",
    "ProviderResponseError",
    "ProviderConfigurationError",
    "register_provider",
    "unregister_provider",
]
