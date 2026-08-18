"""Opt-in tests that make a real, paid OpenAI API request."""

import os

import pandas as pd
import pytest

from enrichment import enrich


pytestmark = pytest.mark.live


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_API_TESTS") != "1",
    reason="Set RUN_LIVE_API_TESTS=1 to enable paid API tests.",
)
def test_live_openai_enrichment():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.fail("OPENAI_API_KEY is required for live API tests.")

    result, report = enrich(
        pd.DataFrame({"text": ["This is an excellent product."]}),
        input_col="text",
        output_col="sentiment",
        prompt="Return exactly one lowercase word: positive, negative, or neutral",
        show_progress=False,
        return_report=True,
    )

    assert result.loc[0, "sentiment"].strip().lower() == "positive"
    assert report.provider == "openai"
    assert report.model == "gpt-5.4-nano"
    assert report.completed == 1
    assert report.failed == 0
    assert report.input_tokens is not None
    assert report.output_tokens is not None
