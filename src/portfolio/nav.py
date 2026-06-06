"""
NavEngine — computes Net Asset Value across multi-currency accounts.

Current implementation: fixed FX rates from config.
Future: inject IFxProvider for dynamic rates.
"""

from __future__ import annotations

from src.core.types import Position
from src.core.interfaces import INavEngine


class NavEngine(INavEngine):
    """Computes NAV = cash + sum(position_value * fx_rate)."""

    def __init__(self, fx_rates: dict[str, float]):
        """
        Args:
            fx_rates: mapping from currency code to USD rate (e.g. {"HKD": 7.8} means 1 USD = 7.8 HKD)
        """
        self._fx_rates = fx_rates

    def compute_nav(
        self, cash: float, positions: list[Position],
        fx_rates: dict[str, float] | None = None,
    ) -> float:
        """Compute total NAV in USD.

        cash is already in USD.
        Position values are in USD (already converted at trade time).
        """
        rates = fx_rates or self._fx_rates

        # Position values are stored in USD (converted at execution time)
        position_value = sum(p.market_value for p in positions)

        return cash + position_value

    def convert_to_usd(self, amount: float, currency: str) -> float:
        """Convert foreign currency amount to USD."""
        if currency == "USD":
            return amount
        rate = self._fx_rates.get(currency)
        if rate is None or rate == 0:
            raise ValueError(f"Unknown FX rate for {currency}")
        return amount / rate

    def convert_from_usd(self, amount_usd: float, currency: str) -> float:
        """Convert USD amount to foreign currency."""
        if currency == "USD":
            return amount_usd
        rate = self._fx_rates.get(currency)
        if rate is None:
            raise ValueError(f"Unknown FX rate for {currency}")
        return amount_usd * rate

    def update_rates(self, new_rates: dict[str, float]) -> None:
        """Update FX rates (for dynamic FX module integration)."""
        self._fx_rates.update(new_rates)
