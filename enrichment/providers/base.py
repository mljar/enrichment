"""Provider interface used by the enrichment engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Optional

from ..models import (
    BatchItemResult,
    BatchStatus,
    CompletionRequest,
    CompletionResult,
)


class Provider(ABC):
    """Base class for synchronous enrichment providers."""

    name = "provider"
    default_model: Optional[str] = None

    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResult:
        """Return one enrichment result."""

    def close(self) -> None:
        """Release provider resources owned by the provider, if any."""


class BatchProvider(Provider):
    """Provider capable of asynchronous batch completion requests."""

    @abstractmethod
    def submit_batch(
        self, requests: Mapping[str, CompletionRequest]
    ) -> BatchStatus:
        """Upload requests and submit a provider batch."""

    @abstractmethod
    def get_batch(self, batch_id: str) -> BatchStatus:
        """Retrieve the current batch state."""

    @abstractmethod
    def get_batch_results(
        self, batch: BatchStatus
    ) -> Mapping[str, BatchItemResult]:
        """Retrieve completed and failed batch items."""

    @abstractmethod
    def cancel_batch(self, batch_id: str) -> BatchStatus:
        """Request cancellation of a batch."""
