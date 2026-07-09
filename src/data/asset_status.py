"""
AssetStatusProvider — determines if an asset is tradable at a given timestamp.

Checks: trading hours, lunch breaks, halted status (missing bars), limit up/down.
"""

from __future__ import annotations

from src.core.types import Market, OrderSide, LimitStatus, Tradability
from src.core.interfaces import IAssetStatusProvider
from src.core.config import Config


class AssetStatusProvider(IAssetStatusProvider):
    """Determines asset tradability based on market rules and price data."""

    def __init__(self, config: Config):
        self._config = config

    def get_status(
        self, market: Market, symbol: str, timestamp: str,
    ) -> tuple[bool, str]:
        """Check if an asset can be traded (market hours only).

        Returns (is_tradable, reason).
        """
        if market == Market.CRYPTO:
            return True, "crypto_24_7"

        if not self._is_market_open(market, timestamp):
            return False, "market_closed"

        return True, "tradable"

    def is_tradable_with_data(
        self, market: Market, symbol: str, timestamp: str,
        has_bar: bool, price: float,
    ) -> tuple[bool, str]:
        """Full tradability check including halt detection.

        Args:
            has_bar: whether a bar exists at this timestamp
            price: current price (0 if no bar)
        """
        # Market hours
        if market != Market.CRYPTO:
            if not self._is_market_open(market, timestamp):
                return False, "market_closed"

        # Halt: no bar at this timestamp (market is open but stock has no data)
        if not has_bar:
            return False, "halted_no_bar"

        # Price unavailable
        if price <= 0:
            return False, "price_unavailable"

        return True, "tradable"

    def get_limit_status(
        self, market: Market, symbol: str,
        current_price: float, reference_price: float,
    ) -> LimitStatus:
        """Determine limit status based on price movement."""
        if market != Market.CN:
            return LimitStatus.NORMAL

        if reference_price <= 0:
            return LimitStatus.NORMAL

        change_pct = (current_price - reference_price) / reference_price * 100

        is_star = symbol.startswith("sh.688")
        limit = self._config.cn_st_limit_up_pct if is_star else self._config.cn_limit_up_pct
        threshold = self._config.cn_limit_near_threshold

        if change_pct >= limit:
            return LimitStatus.LIMIT_UP
        elif change_pct >= limit - threshold:
            return LimitStatus.NEAR_LIMIT_UP
        elif change_pct <= -limit:
            return LimitStatus.LIMIT_DOWN
        elif change_pct <= -(limit - threshold):
            return LimitStatus.NEAR_LIMIT_DOWN

        return LimitStatus.NORMAL

    def _is_market_open(self, market: Market, timestamp: str) -> bool:
        market_hours = self._config.market_hours.get(market)
        if market_hours is None:
            return True

        time_part = self._extract_time(timestamp)
        for open_t, close_t in market_hours.sessions:
            if open_t <= time_part < close_t:
                return True

        return False

    @staticmethod
    def _extract_time(timestamp: str) -> str:
        s = timestamp.replace("T", " ").replace("+00:00", "").strip()
        if len(s) >= 16:
            return s[11:16]
        return s
