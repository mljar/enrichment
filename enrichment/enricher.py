import os
import pandas as pd
import openai
from tqdm import tqdm
from typing import Optional


def _default_openai_client(api_key: Optional[str] = None) -> None:
    """
    Configure OpenAI API key.
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY.")
    openai.api_key = key


def _get_response(prompt: str, model: str) -> str:
    """
    Internal helper to call OpenAI with a prompt and return content.
    """
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()


def enrich(
    df: pd.DataFrame,
    input_col: str,
    output_col: str,
    prompt: str,
    model: str = "gpt-4.1",
    api_key: Optional[str] = None,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Enriches `df` by iterating over each row and applying a prompt, with an optional progress bar.

    Args:
        df: pandas DataFrame to enrich.
        input_col: column with input text.
        output_col: new column name for results.
        prompt: template string describing the analysis to perform on each input.
        model: OpenAI model (default to gpt-4.1).
        api_key: OpenAI API key.
        show_progress: whether to display a tqdm progress bar (default True).

    Returns:
        DataFrame with `output_col` added.
    """
    _default_openai_client(api_key)
    texts = df[input_col].astype(str)
    iterator = tqdm(texts, desc="Enriching") if show_progress else texts

    results = []
    for txt in iterator:
        full_prompt = (
            f"{prompt}. Run on input: {txt}. "
            "Respond with only the output, without any explanations or reasoning."
        )
        results.append(_get_response(full_prompt, model))

    df_copy = df.copy()
    df_copy[output_col] = results
    return df_copy