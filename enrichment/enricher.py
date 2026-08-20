"""Public DataFrame enrichment API."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import pandas as pd

from .engine import _prepare_work, run_enrichment
from .exceptions import EnrichmentError
from .models import EnrichmentReport
from .providers.base import BatchProvider, Provider
from .providers.openai import OpenAIProvider
from .providers.resolver import resolve_provider


AUTO_BATCH_THRESHOLD = 50


def _default_openai_client(api_key: Optional[str] = None) -> OpenAIProvider:
    """Return the default OpenAI provider (kept for API compatibility)."""
    return OpenAIProvider(api_key=api_key)


def _validate_common_inputs(
    df: pd.DataFrame,
    input_col: Optional[str],
    input_cols: Optional[Sequence[str]],
    output_col: Optional[str],
    prompt: Optional[str],
) -> Tuple[Sequence[str], str, str]:
    """Validate arguments shared by interactive and batch enrichment."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if input_col is not None and input_cols is not None:
        raise ValueError("Use either input_col or input_cols, not both.")
    if isinstance(input_cols, str):
        raise ValueError(
            "input_cols must be a sequence of column names, not a string."
        )

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
    return selected_cols, output_col, prompt.strip()


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
    use_batch: Optional[bool] = None,
    on_error: str = "raise",
    return_report: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, EnrichmentReport]]:
    """Add an AI-generated column to a pandas DataFrame.

    ``input_col`` remains available for single-column enrichment. Use
    ``input_cols`` when the task needs values from several columns.
    """
    selected_cols, output_col, prompt = _validate_common_inputs(
        df, input_col, input_cols, output_col, prompt
    )
    if not isinstance(max_concurrency, int) or max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1.")
    if not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries cannot be negative.")
    if retry_base_delay < 0:
        raise ValueError("retry_base_delay cannot be negative.")
    if use_batch is not None and not isinstance(use_batch, bool):
        raise ValueError("use_batch must be True, False, or None.")
    if on_error not in {"raise", "keep"}:
        raise ValueError("on_error must be either 'raise' or 'keep'.")

    selected_provider = resolve_provider(
        provider=provider,
        api_key=api_key,
        model=model,
    )
    prepared_work = _prepare_work(df, selected_cols)
    work_items, _ = prepared_work
    should_use_batch = use_batch is True or (
        use_batch is None
        and isinstance(selected_provider, BatchProvider)
        and len(work_items) >= AUTO_BATCH_THRESHOLD
    )
    if should_use_batch:
        if not isinstance(selected_provider, BatchProvider):
            raise TypeError(
                f"Provider '{selected_provider.name}' does not support "
                "provider-side batches."
            )
        # Import locally to avoid a module cycle around shared input validation.
        # Users still get the same synchronous DataFrame result.
        from .batch import _enrich_batch

        return _enrich_batch(
            df,
            output_col=output_col,
            prompt=prompt,
            model=model,
            show_progress=show_progress,
            input_cols=selected_cols,
            provider=selected_provider,
            on_error=on_error,
            return_report=return_report,
            _prepared_work=prepared_work,
        )

    enriched, report = run_enrichment(
        df,
        input_cols=selected_cols,
        output_col=output_col,
        prompt=prompt,
        provider=selected_provider,
        model=model,
        show_progress=show_progress,
        max_concurrency=max_concurrency,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        on_error=on_error,
        prepared_work=prepared_work,
    )
    return (enriched, report) if return_report else enriched
