"""
Enrichment package

Provides utilities to enrich pandas DataFrames by adding new columns
based on OpenAI-driven analysis.
"""

from .enricher import enrich, _default_openai_client

__version__ = "0.1.0"