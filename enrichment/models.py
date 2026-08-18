"""Shared request, response, and reporting models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional


@dataclass(frozen=True)
class CompletionRequest:
    """Provider-independent request for one enrichment value."""

    instructions: str
    input_data: Mapping[str, Any]
    model: Optional[str] = None
    response_schema: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class CompletionResult:
    """Provider-independent completion result."""

    content: str
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass(frozen=True)
class BatchStatus:
    """Provider-independent state of a submitted batch."""

    id: str
    status: str
    input_file_id: Optional[str] = None
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    total: int = 0
    completed: int = 0
    failed: int = 0

    @property
    def terminal(self) -> bool:
        return self.status in {"completed", "failed", "expired", "cancelled"}


@dataclass(frozen=True)
class BatchItemResult:
    """Result or error for one custom ID in a provider batch."""

    custom_id: str
    result: Optional[CompletionResult] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class EnrichmentFailure:
    """A failed enrichment request and the affected DataFrame rows."""

    indices: List[Any]
    input_data: Mapping[str, Any]
    error: str
    provider: str
    model: Optional[str]


@dataclass
class EnrichmentReport:
    """Summary of work performed by :func:`enrich`."""

    total_rows: int = 0
    unique_requests: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    retries: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    elapsed_seconds: float = 0.0
    provider: str = ""
    model: Optional[str] = None
    batch_id: Optional[str] = None
    batch_status: Optional[str] = None
    errors: List[EnrichmentFailure] = field(default_factory=list)

    def add_usage(self, result: CompletionResult) -> None:
        """Add token usage when the provider reports it."""
        if result.input_tokens is not None:
            self.input_tokens = (self.input_tokens or 0) + result.input_tokens
        if result.output_tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + result.output_tokens
