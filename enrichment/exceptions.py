"""Exceptions raised by the enrichment package."""

from __future__ import annotations

from typing import Any, Mapping, Optional


class EnrichmentError(RuntimeError):
    """Base error raised while enriching a DataFrame."""

    def __init__(
        self,
        message: str,
        *,
        index: Any = None,
        input_data: Optional[Mapping[str, Any]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.index = index
        self.input_data = dict(input_data) if input_data is not None else None
        self.provider = provider
        self.model = model


class ProviderConfigurationError(EnrichmentError):
    """Raised when no usable provider configuration is available."""


class ProviderError(EnrichmentError):
    """Base error returned by an AI provider."""

    retryable = False


class ProviderTemporaryError(ProviderError):
    """A temporary provider error that can be retried."""

    retryable = True


class ProviderResponseError(ProviderError):
    """A permanent or malformed response from a provider."""
