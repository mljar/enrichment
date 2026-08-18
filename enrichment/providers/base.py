"""Provider interface used by the enrichment engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..models import CompletionRequest, CompletionResult


class Provider(ABC):
    """Base class for synchronous enrichment providers."""

    name = "provider"
    default_model: Optional[str] = None

    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResult:
        """Return one enrichment result."""

    def close(self) -> None:
        """Release provider resources owned by the provider, if any."""
