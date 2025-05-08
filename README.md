# Data Enrichment with AI for Pandas DataFrame 🐼💰

The `enrichment` library lets you give your pandas DataFrame a new column data created with AI. Just tell it what you need in plain English. Need sentiment labels, clean addresses, language tags, or keywords? One call to `enrich()` adds a new column of results.

Why you’ll love it:
- **Easy to use:** Pass in your DataFrame, describe what you want, and you’re done.
- **Flexible:** Works with any OpenAI-style model—pick the balance you like between speed and cost.
- **Ready for real work:** Built-in batching, progress bars, and smooth pandas integration.
- **Anything you can name:** From sentiment and topics to translations and more—if you can write it, `enrich()` can handle it.

Give it a try and watch your data come to life!

## Examples

Below are several common enrichment tasks. Each example uses a DataFrame and demonstrates how to call `enrich` with an appropriate prompt, along with expected output.

### 1. Sentiment Analysis
```python
import pandas as pd
from enrichment import enrich

# Sample reviews
df_sentiment = pd.DataFrame({
    "review": [
        "I loved the product, it was fantastic!",
        "Terrible experience, will not buy again.",
        "It was okay, nothing special.",
        "Absolutely amazing, exceeded expectations!",
        "Worst purchase ever, very disappointed."
    ]
})

# Perform sentiment analysis
enriched_sentiment = enrich(
    df_sentiment,
    input_col="review",
    output_col="sentiment",
    prompt="Classify sentiment"
)
print(enriched_sentiment)
```
Expected output:
```plaintext
                                       review sentiment
0      I loved the product, it was fantastic!  Positive
1    Terrible experience, will not buy again.  Negative
2               It was okay, nothing special.   Neutral
3  Absolutely amazing, exceeded expectations!  Positive
4     Worst purchase ever, very disappointed.  Negative

```

### 2. Address Format Standardization & Cleaning
```python
import pandas as pd
from enrichment import enrich

# Sample addresses
df_address = pd.DataFrame({
    "address": [
        "123 main st., Apt#4, new york, ny",
        "456 Elm Street Suite 5 Chicago IL",
        "789 Broadway Blvd Los Angeles,CA",
        "101 first avenue,San Francisco CA",
        "202 Second St. Apt. 10, Boston MA"
    ]
})

# Standardize and clean addresses
enriched_address = enrich(
    df_address,
    input_col="address",
    output_col="clean_address",
    prompt="Standardize and clean this address to a consistent format"
)
print(enriched_address)
```
Expected output:
```plaintext
                             address                                clean_address
0  123 main st., Apt#4, new york, ny             123 Main St, Apt 4, New York, NY
1  456 Elm Street Suite 5 Chicago IL               456 Elm St, Ste 5, Chicago, IL
2   789 Broadway Blvd Los Angeles,CA           789 Broadway Blvd, Los Angeles, CA
3  101 first avenue,San Francisco CA             101 First Ave, San Francisco, CA
4  202 Second St. Apt. 10, Boston MA  202 Second Street, Apartment 10, Boston, MA

```

### 3. Keyword Extraction
```python
import pandas as pd
from enrichment import enrich

# Sample text paragraphs
df_keywords = pd.DataFrame({
    "text": [
        "ChatGPT is a powerful language model developed by OpenAI.",
        "Python's pandas library is excellent for data manipulation.",
        "The Eiffel Tower is one of the most famous landmarks in Paris.",
        "Machine learning and AI are transforming industries.",
        "Renewable energy sources include solar, wind, and hydroelectric power."
    ]
})

# Extract keywords
enriched_keywords = enrich(
    df_keywords,
    input_col="text",
    output_col="keywords",
    prompt="Extract the top 3 keywords"
)
print(enriched_keywords)
```
Expected output:
```plaintext
                                                text                          keywords
0  ChatGPT is a powerful language model developed...   ChatGPT, language model, OpenAI
1  Python's pandas library is excellent for data ...        pandas, data, manipulation
2  The Eiffel Tower is one of the most famous lan...    Eiffel Tower, landmarks, Paris
3  Machine learning and AI are transforming indus...  Machine learning, AI, industries
4  Renewable energy sources include solar, wind, ...     renewable energy, solar, wind
```

### 4. Language Detection
```python
import pandas as pd
from enrichment import enrich

# Sample multilingual sentences
df_language = pd.DataFrame({
    "sentence": [
        "Bonjour, comment ça va?",
        "Hello, how are you?",
        "Hola, ¿cómo estás?",
        "Hallo, wie geht es dir?",
        "Ciao, come stai?",
        "Cześć, jak się masz?"
    ]
})


# Detect language
enriched_language = enrich(
    df_language,
    input_col="sentence",
    output_col="language",
    prompt="Detect the language of this sentence"
)
print(enriched_language)
```
Expected output:
```plaintext
                  sentence language
0  Bonjour, comment ça va?   French
1      Hello, how are you?  English
2       Hola, ¿cómo estás?  Spanish
3  Hallo, wie geht es dir?   German
4         Ciao, come stai?  Italian
5     Cześć, jak się masz?   Polish
```

### 5. Text Classification
```python
import pandas as pd
from enrichment import enrich

# Sample news headlines
df_classify = pd.DataFrame({
    "headline": [
        "Local team wins championship after dramatic final",
        "New species of bird discovered in Amazon rainforest",
        "Stock markets rally as tech stocks soar",
        "Study reveals health benefits of green tea",
        "Government announces new climate policy"
    ]
})

# Classify headlines into categories: sports, science, finance, health, politics
enriched_classify = enrich(
    df_classify,
    input_col="headline",
    output_col="category",
    prompt="Classify this headline into one of: sports, science, finance, health, politics"
)
print(enriched_classify)
```
Expected output:
```plaintext
                                            headline  category
0  Local team wins championship after dramatic final    sports
1  New species of bird discovered in Amazon rainf...   science
2            Stock markets rally as tech stocks soar   finance
3         Study reveals health benefits of green tea    health
4            Government announces new climate policy  politics

```

## Installation

Use the following command to install this package:

```bash
pip install enrichment
```

## API Reference

```python
enrich(
    df: pandas.DataFrame,
    input_col: str,
    output_col: str,
    prompt: str,
    model: str = "gpt-4.1",
    api_key: Optional[str] = None,
    show_progress: bool = True
) -> pandas.DataFrame
```

| Parameter      | Type                 | Description                                                                                         |
| -------------- | -------------------- | --------------------------------------------------------------------------------------------------- |
| `df`           | `DataFrame`          | The source DataFrame containing text data.                                                         |
| `input_col`    | `str`                | Column name in `df` holding the input text.                                                       |
| `output_col`   | `str`                | Name of the new column to store enrichment results.                                               |
| `prompt`       | `str`                | Task description; model receives `prompt` + input row, forced to output only the result.           |
| `model`        | `str`, optional      | OpenAI model to use (default: `gpt-4.1`).                                                          |
| `api_key`      | `str`, optional      | OpenAI API key. If omitted, reads from `OPENAI_API_KEY` environment variable.                      |
| `show_progress`| `bool`, optional     | Display a tqdm progress bar (default: `True`; set `False` to hide).                                |

**Returns:**

- A new `pandas.DataFrame` identical to `df` but with `output_col` populated by model responses.

**Raises:**

- `ValueError` if the model key is not set via `api_key` or `OPENAI_API_KEY`.

## Running Tests

Make sure you have `pytest` installed, then run:

```bash
python -m pytest
```
