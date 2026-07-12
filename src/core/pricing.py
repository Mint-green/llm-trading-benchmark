"""
Model pricing constants for API cost calculation.

Prices per 1M tokens (input / output). Update when provider pricing changes.
All values in USD.
"""

from __future__ import annotations

# Pricing per 1 million tokens
MODEL_PRICING: dict[str, dict[str, float]] = {
    "deepseek-v4-pro": {
        "input": 0.28,    # $0.28 per 1M input tokens
        "output": 1.10,   # $1.10 per 1M output tokens
        "provider": "DeepSeek",
    },
    "deepseek-chat": {
        "input": 0.07,
        "output": 0.28,
        "provider": "DeepSeek",
    },
    "mimo-v2.5-pro": {
        "input": 0.55,
        "output": 2.19,
        "provider": "Xiaomi MIMO",
    },
}

DEFAULT_PRICING = {"input": 0.50, "output": 1.50}


def get_pricing(model_name: str) -> dict[str, float]:
    """Get pricing dict for a model. Falls back to DEFAULT_PRICING."""
    for key in MODEL_PRICING:
        if key in model_name or model_name in key:
            return MODEL_PRICING[key]
    return dict(DEFAULT_PRICING)


def compute_api_cost(
    prompt_tokens: int, completion_tokens: int, model_name: str,
) -> float:
    """Compute API cost in USD for a single LLM call."""
    pricing = get_pricing(model_name)
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


def compute_total_api_cost(
    total_prompt_tokens: int, total_completion_tokens: int, model_name: str,
) -> float:
    """Compute total API cost across all calls."""
    return compute_api_cost(total_prompt_tokens, total_completion_tokens, model_name)
