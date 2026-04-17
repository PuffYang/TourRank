"""Shared Azure GPT-4o client utilities for DRB evaluation.

Provides ``create_chat_completion_text`` and ``resolve_azure_gpt4o_model``
used by ``deep_research_bench_eval/run_eval.py``.

Environment Variables:
    AZURE_OPENAI_API_KEY   – Azure OpenAI API key (required).
    AZURE_OPENAI_ENDPOINT  – Azure OpenAI endpoint URL (required).
    OPENAI_API_VERSION     – API version string (default ``2024-06-01``).
"""

from __future__ import annotations

import os
from typing import Any

from openai import AzureOpenAI

# ---------------------------------------------------------------------------
# Module-level client cache (one per (key, version, endpoint) triple)
# ---------------------------------------------------------------------------
_CLIENT_CACHE: dict[tuple[str, str, str], AzureOpenAI] = {}

# Model name normalization map – the runway proxy may require exact names.
_MODEL_ALIASES: dict[str, str] = {
    "gpt-4-o": "gpt-4o",
    "gpt4o": "gpt-4o",
    "gpt-4-o-mini": "gpt-4o-mini",
    "gpt4o-mini": "gpt-4o-mini",
}


def resolve_azure_gpt4o_model(model: str | None = None) -> str:
    """Normalize a model name to the canonical Azure deployment name."""
    if not model:
        return "gpt-4o"
    canonical = _MODEL_ALIASES.get(model.lower().strip(), model)
    return canonical


def _get_client(
    api_key: str | None = None,
    api_version: str | None = None,
    azure_endpoint: str | None = None,
) -> AzureOpenAI:
    api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    api_version = api_version or os.environ.get("OPENAI_API_VERSION", "2024-06-01")
    azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")

    if not api_key:
        raise ValueError(
            "Azure OpenAI API key not provided. "
            "Set AZURE_OPENAI_API_KEY environment variable."
        )
    if not azure_endpoint:
        raise ValueError(
            "Azure OpenAI endpoint not provided. "
            "Set AZURE_OPENAI_ENDPOINT environment variable."
        )

    cache_key = (api_key, api_version, azure_endpoint)
    client = _CLIENT_CACHE.get(cache_key)
    if client is None:
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )
        _CLIENT_CACHE[cache_key] = client
    return client


def create_chat_completion_text(
    messages: list[dict[str, str]],
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 5000,
    timeout: int = 600,
    **kwargs: Any,
) -> tuple[str, dict[str, Any]]:
    """Call Azure GPT-4o and return ``(response_text, raw_usage_dict)``.

    This is the primary interface consumed by ``run_eval.py``.
    """
    client = _get_client()
    resolved_model = resolve_azure_gpt4o_model(model)

    response = client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        timeout=timeout,
    )

    content = response.choices[0].message.content or ""
    usage: dict[str, Any] = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return content, usage
