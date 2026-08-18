"""Public DataFrame enrichment API."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import pandas as pd

from .engine import run_enrichment
from .exceptions import EnrichmentError
from .models import EnrichmentReport
from .providers.base import Provider
from .providers.openai import OpenAIProvider
from .providers.resolver import resolve_provider


def _default_openai_client(api_key: Optional[str] = None) -> OpenAIProvider:
    """Return the default OpenAI provider (kept for API compatibility)."""
    return OpenAIProvider(api_key=api_key)


def enrich(
    df: pd.DataFrame,
    input_col: Optional[str] = None,
    output_col: Optional[str] = None,
    prompt: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    show_progress: bool = True,
    *,
    input_cols: Optional[Sequence[str]] = None,
    provider: Optional[Provider] = None,
    max_concurrency: int = 5,
    max_retries: int = 3,
    retry_base_delay: float = 0.5,
    on_error: str = "raise",
    return_report: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, EnrichmentReport]]:
    """Add an AI-generated column to a pandas DataFrame.

    ``input_col`` remains available for single-column enrichment. Use
    ``input_cols`` when the task needs values from several columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if input_col is not None and input_cols is not None:
        raise ValueError("Use either input_col or input_cols, not both.")
    if isinstance(input_cols, str):
        raise ValueError("input_cols must be a sequence of column names, not a string.")

    selected_cols = [input_col] if input_col is not None else list(input_cols or [])
    invalid_column = any(
        not isinstance(column, str) or not column for column in selected_cols
    )
    if not selected_cols or invalid_column:
        raise ValueError("At least one input column must be provided.")
    if len(set(selected_cols)) != len(selected_cols):
        raise ValueError("Input column names must be unique.")
    missing_cols = [column for column in selected_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame is missing input columns: {missing_cols}.")
    if not isinstance(output_col, str) or not output_col:
        raise ValueError("output_col must be a non-empty string.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")
    if not isinstance(max_concurrency, int) or max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1.")
    if not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries cannot be negative.")
    if retry_base_delay < 0:
        raise ValueError("retry_base_delay cannot be negative.")
    if on_error not in {"raise", "keep"}:
        raise ValueError("on_error must be either 'raise' or 'keep'.")

    selected_provider = resolve_provider(
        provider=provider,
        api_key=api_key,
        model=model,
    )
    enriched, report = run_enrichment(
        df,
        input_cols=selected_cols,
        output_col=output_col,
        prompt=prompt.strip(),
        provider=selected_provider,
        model=model,
        show_progress=show_progress,
        max_concurrency=max_concurrency,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        on_error=on_error,
    )
    return (enriched, report) if return_report else enriched
