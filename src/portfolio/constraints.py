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

    def __init__(self, config: Config):
        self._limits = config.position_limits

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
    ) -> tuple[bool, str]:
        """Validate a sell order."""
        if quantity <= 0:
            return False, "quantity must be positive"

        pos = current_positions.get(key)
        if pos is None:
            return False, f"no position in {key}"

        if quantity > pos.quantity:
            return False, f"sell quantity {quantity} > held {pos.quantity}"

        return True, "ok"
