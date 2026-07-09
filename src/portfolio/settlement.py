"""
SettlementEngine — handles post-trade settlement.

Implements T+1 for A-shares (sell-side).
Other markets: immediate settlement.
"""

from __future__ import annotations
from collections import defaultdict

from src.core.types import Market, TradeResult, OrderSide
from src.core.interfaces import ISettlementEngine


class SettlementEngine(ISettlementEngine):
    """Handles settlement with T+1 support for CN market."""

    def __init__(self):
        # Tracks buy timestamps per position key for T+1 enforcement
        # key: "MARKET:SYMBOL", value: list of (timestamp, quantity)
        self._buy_history: dict[str, list[tuple[str, float]]] = defaultdict(list)

    def settle(self, result: TradeResult, timestamp: str) -> None:
        """Record settlement. For buys, track the acquisition timestamp."""
        if not result.success:
            return

        order = result.order
        key = f"{order.market.value}:{order.symbol}"

        if order.side == OrderSide.BUY:
            self._buy_history[key].append((timestamp, order.quantity))
        elif order.side == OrderSide.SELL:
            # Reduce buy history (FIFO)
            remaining = order.quantity
            while remaining > 0 and self._buy_history[key]:
                ts, qty = self._buy_history[key][0]
                if qty <= remaining:
                    remaining -= qty
                    self._buy_history[key].pop(0)
                else:
                    self._buy_history[key][0] = (ts, qty - remaining)
                    remaining = 0

    def get_sellable_quantity(self, key: str, timestamp: str) -> float:
        """How many shares can be sold at this timestamp (T+1 aware).

        For CN market: shares bought today cannot be sold until tomorrow.
        Other markets: all shares are immediately sellable.
        """
        market_str = key.split(":")[0]
        market = Market(market_str) if market_str in [m.value for m in Market] else None

        if market != Market.CN:
            # Non-CN: all shares immediately sellable
            return sum(qty for _, qty in self._buy_history.get(key, []))

        # CN T+1: only shares bought before today can be sold
        today = timestamp[:10]  # "YYYY-MM-DD"
        sellable = 0
        for buy_ts, qty in self._buy_history.get(key, []):
            buy_date = buy_ts[:10]
            if buy_date < today:
                sellable += qty

        return sellable

    def get_frozen_keys(self, timestamp: str) -> set[str]:
        """Get position keys that are frozen due to T+1 (CN shares bought today).

        Returns set of keys like "CN:sh.600519" that cannot be sold.
        """
        today = timestamp[:10]
        frozen = set()
        for key, history in self._buy_history.items():
            market_str = key.split(":")[0]
            if market_str != "CN":
                continue
            # If any shares were bought today, this position has frozen shares
            for buy_ts, qty in history:
                if buy_ts[:10] == today and qty > 0:
                    frozen.add(key)
                    break
        return frozen

    def reset(self) -> None:
        """Clear all settlement state."""
        self._buy_history.clear()
