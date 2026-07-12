"""USD-normalized trade values for metrics and behavior analysis."""

from __future__ import annotations

from src.core.types import Market, TradeResult


MARKET_CURRENCY = {
    Market.US: "USD",
    Market.HK: "HKD",
    Market.CN: "CNY",
    Market.CRYPTO: "USD",
    Market.GOLD: "USD",
    Market.FUTURES: "USD",
}


def trade_cost_usd(trade: TradeResult, fx_rates: dict[str, float]) -> float:
    return _trade_value_usd(trade, "cost", fx_rates)


def trade_fees_usd(trade: TradeResult, fx_rates: dict[str, float]) -> float:
    return _trade_value_usd(trade, "fees", fx_rates)


def _trade_value_usd(
    trade: TradeResult,
    field: str,
    fx_rates: dict[str, float],
) -> float:
    metadata_key = f"{field}_usd"
    if metadata_key in trade.metadata:
        return float(trade.metadata[metadata_key] or 0.0)

    amount = float(getattr(trade, field) or 0.0)
    currency = str(
        trade.metadata.get("currency")
        or MARKET_CURRENCY.get(trade.order.market, "USD")
    )
    if currency == "USD":
        return amount
    rate = float(fx_rates.get(currency, 0.0) or 0.0)
    if rate <= 0:
        raise ValueError(f"Missing FX rate for {currency} trade metrics")
    return amount / rate
