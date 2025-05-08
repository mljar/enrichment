import pandas as pd
import pytest
from enrichment.enricher import enrich

class DummyResp:
    def __init__(self, text):
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": text})})]

class DummyClient:
    @staticmethod
    def create(model, messages):
        txt = messages[0]["content"]
        if "good" in txt.lower():
            return DummyResp("Positive")
        return DummyResp("Negative")

@pytest.fixture(autouse=True)
def patch_openai(monkeypatch):
    import openai
    monkeypatch.setattr(openai, "ChatCompletion", DummyClient)


def test_enrich_with_progress_flag_false():
    df = pd.DataFrame({"review": ["This is good", "Terrible"]})
    prompt = "Classify sentiment"
    # Hide progress bar
    out = enrich(df, "review", "sentiment", prompt, show_progress=False)
    assert list(out["sentiment"]) == ["Positive", "Negative"]


def test_enrich_default():
    df = pd.DataFrame({"review": ["OK review", "Worst review"]})
    prompt = "Classify sentiment"
    out = enrich(df, "review", "sentiment", prompt)
    # Default behavior shows progress; logic unchanged
    assert list(out["sentiment"]) == ["Negative", "Negative"]