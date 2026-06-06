"""
MarketRuleEngine — enforces market-specific trading rules.

Handles: trading hours, lunch breaks, T+1 settlement, limit up/down, halt detection.
"""

from __future__ import annotations

from src.core.types import Market, OrderSide, LimitStatus
from src.core.interfaces import IMarketRuleEngine
from src.core.config import Config
from ..data.asset_status import AssetStatusProvider


class MarketRuleEngine(IMarketRuleEngine):
    """Enforces market-specific trading rules."""

    def __init__(self, config: Config, asset_status: AssetStatusProvider):
        self._config = config
        self._asset_status = asset_status

    def can_trade(
        self, market: Market, symbol: str, side: OrderSide, timestamp: str,
    ) -> tuple[bool, str]:
        """Check if a trade is allowed.

        Checks: market hours, lunch break, (future: halt, limit up/down).
        """
        # Crypto: always tradable
        if market == Market.CRYPTO:
            return True, "ok"

        # Check market hours (including lunch breaks)
        tradable, reason = self._asset_status.get_status(market, symbol, timestamp)
        if not tradable:
            return False, reason

        # Additional rules would be checked here:
        # - Halted assets: requires external halt data
        # - Limit up/down: requires real-time price vs reference
        # - T+1: checked by SettlementEngine

        return True, "ok"
