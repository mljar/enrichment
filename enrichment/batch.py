"""Provider-side batch enrichment for large DataFrames."""

from __future__ import annotations

import time
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
from tqdm.auto import tqdm

from .engine import _WorkItem, _prepare_work
from .enricher import _validate_common_inputs
from .exceptions import EnrichmentError
from .models import (
    BatchItemResult,
    BatchStatus,
    CompletionRequest,
    EnrichmentFailure,
    EnrichmentReport,
)
from .providers.base import BatchProvider, Provider
from .providers.resolver import resolve_provider


class EnrichmentBatchJob:
    """A submitted provider batch that can be polled and materialized."""

    def __init__(
        self,
        *,
        provider: BatchProvider,
        batch: BatchStatus,
        source: pd.DataFrame,
        output_col: str,
        work_items: Mapping[str, _WorkItem],
        skipped_positions: Sequence[int],
        model: Optional[str],
    ) -> None:
        self.provider = provider
        self.batch = batch
        self.source = source.copy()
        self.output_col = output_col
        self.work_items = dict(work_items)
        self.skipped_positions = list(skipped_positions)
        self.model = model or provider.default_model
        self._started = time.monotonic()

    @property
    def id(self) -> str:
        return self.batch.id

    @property
    def status(self) -> str:
        return self.batch.status

    def refresh(self) -> BatchStatus:
        """Refresh and return the provider batch state."""
        if self.id == "local-empty":
            return self.batch
        self.batch = self.provider.get_batch(self.id)
        return self.batch

    def wait(
        self,
        *,
        poll_interval: float = 10.0,
        timeout: Optional[float] = None,
        show_progress: bool = True,
    ) -> "EnrichmentBatchJob":
        """Poll until the batch reaches a terminal state."""
        if poll_interval <= 0:
            raise ValueError("poll_interval must be greater than zero.")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout cannot be negative.")

        progress = tqdm(
            total=len(self.work_items),
            desc="Batch enrichment",
            disable=not show_progress,
        )
        try:
            while True:
                batch = self.refresh()
                progress.n = min(batch.completed + batch.failed, progress.total)
                progress.refresh()
                if batch.terminal:
                    return self
                if timeout is not None and time.monotonic() - self._started >= timeout:
                    raise TimeoutError(
                        f"Batch {self.id} did not finish within {timeout} seconds."
                    )
                time.sleep(poll_interval)
        finally:
            progress.close()

    def cancel(self) -> BatchStatus:
        """Request cancellation and return the new provider state."""
        if self.id == "local-empty" or self.batch.terminal:
            return self.batch
        self.batch = self.provider.cancel_batch(self.id)
        return self.batch

    def result(
        self,
        *,
        on_error: str = "raise",
        return_report: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, EnrichmentReport]]:
        """Download results and add them to a copy of the source DataFrame."""
        if on_error not in {"raise", "keep"}:
            raise ValueError("on_error must be either 'raise' or 'keep'.")
        if not self.batch.terminal:
            raise EnrichmentError(
                f"Batch {self.id} is still {self.batch.status}; call wait() first."
            )

        provider_items: Mapping[str, BatchItemResult] = {}
        if self.id != "local-empty":
            provider_items = self.provider.get_batch_results(self.batch)

        values = [pd.NA] * len(self.source)
        report = EnrichmentReport(
            total_rows=len(self.source),
            unique_requests=len(self.work_items),
            skipped=len(self.skipped_positions),
            provider=self.provider.name,
            model=self.model,
            batch_id=self.id,
            batch_status=self.batch.status,
            elapsed_seconds=time.monotonic() - self._started,
        )

        for custom_id, work_item in self.work_items.items():
            item = provider_items.get(custom_id)
            if item is not None and item.result is not None:
                for position in work_item.positions:
                    values[position] = item.result.content
                report.completed += len(work_item.positions)
                report.add_usage(item.result)
                continue

            error = (
                item.error
                if item is not None and item.error
                else f"No result was returned for {custom_id}."
            )
            failure = EnrichmentFailure(
                indices=work_item.indices,
                input_data=work_item.input_data,
                error=error,
                provider=self.provider.name,
                model=self.model,
            )
            report.failed += len(work_item.positions)
            report.errors.append(failure)
            if on_error == "raise":
                raise EnrichmentError(
                    (
                        f"Batch enrichment failed for DataFrame index "
                        f"{work_item.indices[0]!r}: {error}"
                    ),
                    index=work_item.indices[0],
                    input_data=work_item.input_data,
                    provider=self.provider.name,
                    model=self.model,
                )

        enriched = self.source.copy()
        enriched[self.output_col] = pd.Series(
            values, index=enriched.index, dtype="object"
        )
        return (enriched, report) if return_report else enriched


def enrich_batch(
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
    wait: bool = True,
    poll_interval: float = 10.0,
    timeout: Optional[float] = None,
    on_error: str = "raise",
    return_report: bool = False,
) -> Union[
    EnrichmentBatchJob,
    pd.DataFrame,
    Tuple[pd.DataFrame, EnrichmentReport],
]:
    """Enrich a DataFrame through a provider-side asynchronous batch."""
    selected_cols, output_col, prompt = _validate_common_inputs(
        df, input_col, input_cols, output_col, prompt
    )
    if on_error not in {"raise", "keep"}:
        raise ValueError("on_error must be either 'raise' or 'keep'.")
    if not wait and return_report:
        raise ValueError("return_report requires wait=True.")

    selected_provider = resolve_provider(
        provider=provider,
        api_key=api_key,
        model=model,
    )
    if not isinstance(selected_provider, BatchProvider):
        raise TypeError(
            f"Provider '{selected_provider.name}' does not support "
            "provider-side batches."
        )

    work, skipped_positions = _prepare_work(df, selected_cols)
    work_items: Dict[str, _WorkItem] = {
        f"enrichment-{position}": item for position, item in enumerate(work)
    }
    if work_items:
        requests = {
            custom_id: CompletionRequest(
                instructions=prompt,
                input_data=item.input_data,
                model=model,
            )
            for custom_id, item in work_items.items()
        }
        batch = selected_provider.submit_batch(requests)
    else:
        batch = BatchStatus(id="local-empty", status="completed")

    job = EnrichmentBatchJob(
        provider=selected_provider,
        batch=batch,
        source=df,
        output_col=output_col,
        work_items=work_items,
        skipped_positions=skipped_positions,
        model=model,
    )
    if not wait:
        return job
    job.wait(
        poll_interval=poll_interval,
        timeout=timeout,
        show_progress=show_progress,
    )
    return job.result(on_error=on_error, return_report=return_report)
