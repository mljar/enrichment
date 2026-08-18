"""Concurrent DataFrame enrichment engine."""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from threading import Event
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from tqdm.auto import tqdm

from .exceptions import EnrichmentError, ProviderTemporaryError
from .models import (
    CompletionRequest,
    CompletionResult,
    EnrichmentFailure,
    EnrichmentReport,
)
from .providers.base import Provider


@dataclass
class _WorkItem:
    input_data: Mapping[str, Any]
    positions: List[int]
    indices: List[Any]


@dataclass
class _WorkOutcome:
    result: Optional[CompletionResult] = None
    error: Optional[BaseException] = None
    retries: int = 0


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if not pd.api.types.is_scalar(value):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _json_value(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _prepare_work(
    df: pd.DataFrame, input_cols: Sequence[str]
) -> Tuple[List[_WorkItem], List[int]]:
    work_by_key: Dict[str, _WorkItem] = {}
    skipped_positions: List[int] = []

    for position, (index, row) in enumerate(df.loc[:, input_cols].iterrows()):
        input_data = {column: _json_value(row[column]) for column in input_cols}
        if all(value is None for value in input_data.values()):
            skipped_positions.append(position)
            continue
        key = json.dumps(input_data, ensure_ascii=False, sort_keys=True, default=str)
        item = work_by_key.get(key)
        if item is None:
            work_by_key[key] = _WorkItem(
                input_data=input_data,
                positions=[position],
                indices=[index],
            )
        else:
            item.positions.append(position)
            item.indices.append(index)

    return list(work_by_key.values()), skipped_positions


def _run_request(
    provider: Provider,
    request: CompletionRequest,
    *,
    max_retries: int,
    retry_base_delay: float,
    cancelled: Event,
) -> _WorkOutcome:
    retries = 0
    while not cancelled.is_set():
        try:
            return _WorkOutcome(result=provider.complete(request), retries=retries)
        except ProviderTemporaryError as exc:
            if retries >= max_retries:
                return _WorkOutcome(error=exc, retries=retries)
            retries += 1
            retry_after = getattr(exc, "retry_after", None)
            delay = (
                float(retry_after)
                if retry_after is not None
                else retry_base_delay * (2 ** (retries - 1)) + random.uniform(0, 0.1)
            )
            if cancelled.wait(max(0.0, delay)):
                break
        except Exception as exc:  # Provider plug-ins may raise their own errors.
            return _WorkOutcome(error=exc, retries=retries)
    return _WorkOutcome(
        error=EnrichmentError("Enrichment was cancelled."), retries=retries
    )


def run_enrichment(
    df: pd.DataFrame,
    *,
    input_cols: Sequence[str],
    output_col: str,
    prompt: str,
    provider: Provider,
    model: Optional[str],
    show_progress: bool,
    max_concurrency: int,
    max_retries: int,
    retry_base_delay: float,
    on_error: str,
) -> Tuple[pd.DataFrame, EnrichmentReport]:
    """Run enrichment and return the copied DataFrame and execution report."""
    started = time.monotonic()
    report = EnrichmentReport(
        total_rows=len(df),
        provider=provider.name,
        model=model or provider.default_model,
    )
    work_items, skipped_positions = _prepare_work(df, input_cols)
    report.unique_requests = len(work_items)
    report.skipped = len(skipped_positions)

    results: List[Any] = [pd.NA] * len(df)
    cancelled = Event()
    progress = tqdm(total=len(df), desc="Enriching", disable=not show_progress)
    progress.update(len(skipped_positions))

    worker_count = min(max_concurrency, max(1, len(work_items)))
    executor = ThreadPoolExecutor(max_workers=worker_count)
    futures: Dict[Future[_WorkOutcome], _WorkItem] = {}
    try:
        for item in work_items:
            request = CompletionRequest(
                instructions=prompt,
                input_data=item.input_data,
                model=model,
            )
            future = executor.submit(
                _run_request,
                provider,
                request,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
                cancelled=cancelled,
            )
            futures[future] = item

        for future in as_completed(futures):
            item = futures[future]
            outcome = future.result()
            report.retries += outcome.retries
            if outcome.error is None and outcome.result is not None:
                for position in item.positions:
                    results[position] = outcome.result.content
                report.completed += len(item.positions)
                report.add_usage(outcome.result)
            else:
                error = outcome.error or EnrichmentError(
                    "Unknown provider error."
                )
                failure = EnrichmentFailure(
                    indices=item.indices,
                    input_data=item.input_data,
                    error=str(error),
                    provider=provider.name,
                    model=model or provider.default_model,
                )
                report.failed += len(item.positions)
                report.errors.append(failure)
                if on_error == "raise":
                    cancelled.set()
                    for pending in futures:
                        pending.cancel()
                    row_label = item.indices[0]
                    raise EnrichmentError(
                        (
                            f"Enrichment failed for DataFrame index {row_label!r} "
                            f"using provider '{provider.name}': {error}"
                        ),
                        index=row_label,
                        input_data=item.input_data,
                        provider=provider.name,
                        model=model or provider.default_model,
                    ) from error
            progress.update(len(item.positions))
    finally:
        cancelled.set()
        executor.shutdown(wait=True, cancel_futures=True)
        progress.close()
        report.elapsed_seconds = time.monotonic() - started

    enriched = df.copy()
    enriched[output_col] = pd.Series(
        results, index=enriched.index, dtype="object"
    )
    return enriched, report
