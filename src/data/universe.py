"""
UniverseRegistry — provides the investable universe for each market.

Reads constituent lists from the getStockData project's Python files.
Does NOT modify source data.
"""

from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

from src.core.types import Market, AssetInfo
from src.core.interfaces import IUniverseRegistry
from src.core.config import Config


# Mapping from Market to the constituent file and variable name
_CONSTITUENT_MAP = {
    Market.US: ("US_shares.py", "NASDAQ100_COMPONENTS"),
    Market.HK: ("HK_shares.py", "HK_STOCKS"),
    Market.CN: ("A_shares.py", "A_SHARE_SSE50_CONSTITUENTS"),
    Market.CRYPTO: ("Crypto_shares.py", "CRYPTO_STOCKS"),
    Market.GOLD: ("Gold_shares.py", "GOLD_STOCKS"),
    Market.FUTURES: ("Futures_shares.py", "FUTURES_STOCKS"),
}

# Ticker suffix per market (for building full ticker from raw symbol)
_SUFFIX_MAP = {
    Market.US: ".US",
    Market.HK: ".HK",
    Market.CN: "",  # CN symbols already have "sh." or "sz." prefix
    Market.CRYPTO: ".CC",
    Market.GOLD: "",
    Market.FUTURES: "",
}


class UniverseRegistry(IUniverseRegistry):
    """Loads universe from getStockData constituent files."""

    def __init__(self, config: Config, constituents_dir: str | None = None):
        self._config = config
        self._constituents_dir = constituents_dir or self._find_constituents_dir()
        self._cache: dict[Market, list[AssetInfo]] = {}

    def _find_constituents_dir(self) -> str:
        """Locate the constituents directory relative to stock_data_dir."""
        # stock_data_dir is like ".../getStockData/data", constituents is ".../getStockData/constituents"
        parent = Path(self._config.stock_data_dir).parent
        candidates = [
            parent / "constituents",
            parent / "getStockData" / "constituents",
        ]
        for c in candidates:
            if c.is_dir():
                return str(c)
        raise FileNotFoundError(
            f"Cannot find constituents directory. "
            f"Looked in: {[str(c) for c in candidates]}"
        )

    def _load_symbols(self, market: Market) -> list[str]:
        """Load symbol list from the constituent Python file."""
        filename, varname = _CONSTITUENT_MAP[market]
        filepath = Path(self._constituents_dir) / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Constituent file not found: {filepath}")

        # Load the Python module dynamically
        spec = importlib.util.spec_from_file_location(f"const_{market.value}", str(filepath))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        symbols = getattr(module, varname, None)
        if symbols is None:
            raise AttributeError(f"Variable {varname} not found in {filepath}")

        return list(symbols)

    def get_assets(self, market: Market) -> list[AssetInfo]:
        if market in self._cache:
            return self._cache[market]

        symbols = self._load_symbols(market)
        if market == Market.GOLD:
            allowed = set(getattr(self._config.gold, "allowed_symbols", ("XAUUSD.FOREX",)))
            symbols = [sym for sym in symbols if sym in allowed]
        assets = []
        for sym in symbols:
            # Determine sector (placeholder — real sector data would come from a registry)
            assets.append(AssetInfo(
                ticker=sym,
                name=sym,  # name resolved later if needed
                market=market,
                sector="",
                asset_class="crypto" if market == Market.CRYPTO else "futures" if market == Market.FUTURES else "gold_spot" if market == Market.GOLD else "equity",
            ))

        self._cache[market] = assets
        return assets

    def get_asset(self, ticker: str) -> AssetInfo | None:
        for market in Market:
            for asset in self.get_assets(market):
                if asset.ticker == ticker:
                    return asset
        return None

    def get_symbols(self, market: Market) -> list[str]:
        """Convenience: return just the ticker list."""
        return [a.ticker for a in self.get_assets(market)]
