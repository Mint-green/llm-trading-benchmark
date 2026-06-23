"""
ConstraintEngine — validates orders against position limits.

Checks: single position limit, market exposure limit, crypto exposure limit, cash availability.
"""

from __future__ import annotations

from src.core.types import Market, Position
from src.core.interfaces import IConstraintEngine
from src.core.config import Config


class ConstraintEngine(IConstraintEngine):
    """Validates orders against portfolio constraints."""

    def __init__(self, config: Config, cooling_hours: float = 2.0, max_daily_trades: int = 25):
        self._limits = config.position_limits
        self._cooling_hours = cooling_hours
        self._max_daily_trades = max_daily_trades
        # Track last buy time per position: {key: timestamp_str}
        self._last_buy: dict[str, str] = {}
        # Track daily trade count: {date_str: count}
        self._daily_trades: dict[str, int] = {}
        # Track SELLs per decision point to prevent panic selling
        self._sells_this_decision: int = 0
        self._last_decision_ts: str = ""

    def reset_decision_state(self, timestamp: str) -> None:
        """Reset per-decision counters (call at start of each decision point)."""
        if timestamp[:16] != self._last_decision_ts[:16]:
            self._sells_this_decision = 0
            self._last_decision_ts = timestamp

    def record_buy(self, key: str, timestamp: str) -> None:
        """Record a buy for cooling period tracking."""
        self._last_buy[key] = timestamp

    def record_trade(self, timestamp: str) -> None:
        """Record a trade for daily limit tracking."""
        date = timestamp[:10]
        self._daily_trades[date] = self._daily_trades.get(date, 0) + 1

    @property
    def daily_buys_remaining(self) -> int:
        """How many BUYs are left today (for LLM context)."""
        today = max(self._daily_trades.keys(), default="")
        used = self._daily_trades.get(today, 0)
        return max(0, self._max_daily_trades - used)

    def check_daily_limit(self, timestamp: str) -> tuple[bool, str]:
        """Check if daily trade limit is exceeded."""
        date = timestamp[:10]
        count = self._daily_trades.get(date, 0)
        if count >= self._max_daily_trades:
            return False, (
                f"daily trade limit reached ({count}/{self._max_daily_trades}). "
                f"Wait for the next trading day."
            )
        return True, "ok"

    def _hours_since(self, timestamp: str, reference: str) -> float:
        """Calculate hours between two timestamp strings."""
        from datetime import datetime
        try:
            t1 = datetime.strptime(timestamp[:16], "%Y-%m-%d %H:%M")
            t2 = datetime.strptime(reference[:16], "%Y-%m-%d %H:%M")
            return (t1 - t2).total_seconds() / 3600
        except ValueError:
            return 999  # can't parse, allow trade

    def validate_buy(
        self, symbol: str, market: Market, quantity: int, price: float,
        current_nav: float, current_positions: dict[str, Position],
    ) -> tuple[bool, str]:
        """Validate a buy order against all constraints."""
        if quantity <= 0:
            return False, "quantity must be positive"
        if price <= 0:
            return False, "price must be positive"
        if current_nav <= 0:
            return False, "NAV is zero or negative"

        cost = price * quantity

        # 1. Cash availability (min cash ratio)
        available_cash = current_nav * (1 - self._limits.min_cash_ratio)
        # Subtract committed cash from pending buys (not implemented yet)
        if cost > available_cash:
            return False, f"insufficient cash (need {cost:.0f}, available {available_cash:.0f})"

        # 2. Single position limit
        key = f"{market.value}:{symbol}"
        existing = current_positions.get(key)
        existing_value = existing.market_value if existing else 0.0
        new_position_value = existing_value + cost
        max_position = current_nav * self._limits.max_single_position

        if new_position_value > max_position:
            return False, (
                f"breaches {self._limits.max_single_position*100:.0f}% single position limit "
                f"(existing={existing_value:.0f}, new={cost:.0f}, max={max_position:.0f})"
            )

        # 3. Market exposure limit
        market_exposure = sum(
            p.market_value for k, p in current_positions.items()
            if p.market == market
        )
        new_market_exposure = market_exposure + cost
        max_market = current_nav * self._limits.max_market_exposure

        if new_market_exposure > max_market:
            return False, (
                f"breaches {self._limits.max_market_exposure*100:.0f}% market exposure limit "
                f"(current={market_exposure:.0f}, new={cost:.0f}, max={max_market:.0f})"
            )

        # 4. Crypto exposure limit
        if market == Market.CRYPTO:
            crypto_exposure = sum(
                p.market_value for k, p in current_positions.items()
                if p.market == Market.CRYPTO
            )
            new_crypto_exposure = crypto_exposure + cost
            max_crypto = current_nav * self._limits.max_crypto_exposure

            if new_crypto_exposure > max_crypto:
                return False, (
                    f"breaches {self._limits.max_crypto_exposure*100:.0f}% crypto exposure limit"
                )

        return True, "ok"

    def validate_sell(
        self, key: str, quantity: int,
        current_positions: dict[str, Position],
        timestamp: str = "",
    ) -> tuple[bool, str]:
        """Validate a sell order."""
        if quantity <= 0:
            return False, "quantity must be positive"

        pos = current_positions.get(key)
        if pos is None:
            return False, f"no position in {key}"

        if quantity > pos.quantity:
            return False, f"sell quantity {quantity} > held {pos.quantity}"

        # Reject micro-sells: allocation_pct < 2% are cost-inefficient noise
        if quantity <= 0:
            return False, "sell quantity too small (allocation_pct < 2% — costs exceed benefit)"

        # Cooling period: prevent selling within N hours of purchase
        if key in self._last_buy and timestamp:
            hours = self._hours_since(timestamp, self._last_buy[key])
            if hours < self._cooling_hours:
                return False, (
                    f"cooling period: bought {hours:.1f}h ago, "
                    f"must hold >= {self._cooling_hours:.0f}h. "
                    f"Short-term flipping is proven to lose money after costs."
                )

        return True, "ok"
