"""Provider registration and automatic resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from threading import RLock
from typing import Dict, Optional

from ..exceptions import ProviderConfigurationError
from .base import Provider
from .openai import OpenAIProvider


@dataclass(frozen=True)
class _RegisteredProvider:
    provider: Provider
    priority: int


_providers: Dict[str, _RegisteredProvider] = {}
_lock = RLock()


def register_provider(name: str, provider: Provider, *, priority: int = 0) -> None:
    """Register a runtime provider for automatic selection."""
    if not name.strip():
        raise ValueError("Provider registration name cannot be empty.")
    if not isinstance(provider, Provider):
        raise TypeError("provider must be an instance of Provider.")
    with _lock:
        _providers[name] = _RegisteredProvider(provider=provider, priority=priority)


def unregister_provider(name: str) -> None:
    """Remove a previously registered runtime provider."""
    with _lock:
        _providers.pop(name, None)


def resolve_provider(
    *,
    provider: Optional[Provider] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Provider:
    """Resolve explicit, runtime-registered, or environment configuration."""
    if provider is not None:
        return provider

    with _lock:
        if _providers:
            return max(_providers.values(), key=lambda item: item.priority).provider

    if api_key or os.getenv("OPENAI_API_KEY"):
        kwargs = {"api_key": api_key}
        if model:
            kwargs["model"] = model
        return OpenAIProvider(**kwargs)

    raise ProviderConfigurationError(
        "No AI provider is configured. Pass provider=..., provide api_key=..., "
        "or set OPENAI_API_KEY."
    )
