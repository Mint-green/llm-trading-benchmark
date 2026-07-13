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
    "qwen-max": {
        "input": 0.17,     # ¥1.2/1M ≈ $0.17 (7.25 FX)
        "output": 0.99,    # ¥7.2/1M ≈ $0.99
        "provider": "Alibaba Bailian",
    },
    "qwen-flash": {
        "input": 0.17,     # ¥1.2/1M ≈ $0.17
        "output": 0.99,    # ¥7.2/1M ≈ $0.99
        "provider": "Alibaba Bailian",
    },
    "qwen3.6-max": {
        "input": 1.24,     # ¥9/1M ≈ $1.24
        "output": 7.45,    # ¥54/1M ≈ $7.45
        "provider": "Alibaba Bailian",
    },
}

DEFAULT_PRICING = {"input": 0.50, "output": 1.50}


def get_pricing(model_name: str) -> dict[str, float]:
    """Get pricing dict for a model. Falls back to DEFAULT_PRICING."""
    for key in MODEL_PRICING:
        if key in model_name or model_name in key:
            return MODEL_PRICING[key]
    return dict(DEFAULT_PRICING)


def classify_rejection(error_message: str) -> str:
    """Classify a rejection error message into a standardized code."""
    if not error_message:
        return ""
    msg = error_message.lower()
    if "market" in msg and ("closed" in msg or "rule" in msg):
        return "MARKET_CLOSED"
    if "daily" in msg and "limit" in msg:
        return "DAILY_LIMIT"
    if "cooling" in msg:
        return "COOLING_PERIOD"
    if "position" in msg and "limit" in msg:
        return "POSITION_LIMIT"
    if "market exposure" in msg:
        return "MARKET_EXPOSURE"
    if "cash" in msg and ("reserve" in msg or "5%" in msg):
        return "CASH_RESERVE"
    if "t+1" in msg.lower():
        return "T1_RESTRICTION"
    if "rounds to 0" in msg or "lot" in msg:
        return "LOT_ROUNDING"
    if "sell" in msg and ("limit" in msg or "max" in msg):
        return "SELL_LIMIT"
    if "insufficient" in msg:
        return "INSUFFICIENT_FUNDS"
    if "price" in msg and "unavailable" in msg:
        return "PRICE_UNAVAILABLE"
    if "notional_too_small" in msg or "contract" in msg:
        return "CONTRACT_SIZE"
    return "OTHER"


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
