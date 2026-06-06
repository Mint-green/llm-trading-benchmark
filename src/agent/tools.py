"""
ToolSystem — provides tools the agent can call during rounds 2-7.

Tools:
  - market_overview(): Current market index status
  - query_stock(ticker): Detailed stock data
  - query_macro(): Macro economic context (placeholder)
  - query_fx(): Current FX rates
  - query_position(): Detailed position info
  - query_history(ticker, days): Historical price data
  - query_news(ticker): News (reserved)
"""

from __future__ import annotations

from src.core.types import Market, PortfolioSnapshot, IndicatorSnapshot
from src.core.interfaces import IToolSystem
from src.data.provider import MarketDataProvider
from src.data.features import FeatureGenerator


class ToolSystem(IToolSystem):
    """Provides queryable tools for the agent."""

    def __init__(
        self,
        data_provider: MarketDataProvider,
        feature_gen: FeatureGenerator,
        portfolio_snapshot_fn,  # callable that returns PortfolioSnapshot
    ):
        self._data = data_provider
        self._features = feature_gen
        self._get_snapshot = portfolio_snapshot_fn

    def get_tool_descriptions(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "market_overview",
                    "description": "Get current market status and index overview",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_stock",
                    "description": "Get detailed stock data including recent bars and indicators",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string", "description": "Stock ticker, e.g. AAPL.US, 0700.HK"},
                        },
                        "required": ["ticker"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_macro",
                    "description": "Get macro economic context (interest rates, inflation, etc.)",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_fx",
                    "description": "Get current FX rates",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_position",
                    "description": "Get detailed position and portfolio info",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_history",
                    "description": "Get historical price data for a stock",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "days": {"type": "integer", "description": "Number of days of history", "default": 5},
                        },
                        "required": ["ticker"],
                    },
                },
            },
        ]

    def execute_tool(
        self, name: str, args: dict, timestamp: str,
    ) -> str:
        if name == "market_overview":
            return self._market_overview(timestamp)
        elif name == "query_stock":
            return self._query_stock(args.get("ticker", ""), timestamp)
        elif name == "query_macro":
            return self._query_macro()
        elif name == "query_fx":
            return self._query_fx()
        elif name == "query_position":
            return self._query_position(timestamp)
        elif name == "query_history":
            return self._query_history(args.get("ticker", ""), args.get("days", 5), timestamp)
        elif name == "query_news":
            return "(news not available in backtest mode)"
        else:
            return f"Unknown tool: {name}"

    def _market_overview(self, timestamp: str) -> str:
        snapshot = self._get_snapshot()
        lines = ["Market Overview:"]
        for market in [Market.US, Market.HK, Market.CN, Market.CRYPTO]:
            symbols = self._data.get_universe_symbols(market)
            exposure = snapshot.market_exposure.get(market, 0)
            lines.append(f"  {market.value}: {len(symbols)} stocks, exposure=${exposure:,.0f}")
        return "\n".join(lines)

    def _query_stock(self, ticker: str, timestamp: str) -> str:
        if not ticker:
            return "Error: ticker is required"

        # Determine market from ticker suffix
        market = self._ticker_to_market(ticker)
        if market is None:
            return f"Error: cannot determine market for {ticker}"

        bars = self._data.load_bars(market, ticker, "2025-10-01", timestamp)
        if not bars:
            return f"No data for {ticker}"

        # Compute indicators
        snap = self._features.compute(bars, timestamp)
        if snap is None:
            return f"Insufficient data for {ticker} (need more bars)"

        # Format recent bars
        recent = bars[-5:]
        lines = [f"[{ticker}] Detailed:"]
        lines.append(f"  Price: ${snap.price:.2f}")
        lines.append(f"  RSI: {snap.rsi:.1f}  ATR%: {snap.atr_pct:.2f}%  Trend: {snap.trend}")
        lines.append(f"  Chg: 5m={snap.chg_5m:+.2f}% 1h={snap.chg_1h:+.2f}% 1d={snap.chg_1d:+.2f}%")
        lines.append(f"  BB pos: {snap.bb_position:.2f}  RelVol: {snap.rel_volume:.1f}x")
        lines.append("  Recent bars:")
        for b in recent:
            lines.append(f"    {b.timestamp}: O={b.open:.2f} H={b.high:.2f} L={b.low:.2f} C={b.close:.2f}")

        return "\n".join(lines)

    def _query_macro(self) -> str:
        # Placeholder - would integrate with macro data source
        return "Macro: (no live macro data in backtest mode. Assume stable conditions.)"

    def _query_fx(self) -> str:
        snapshot = self._get_snapshot()
        lines = ["FX Rates (per 1 USD):"]
        for currency, rate in snapshot.fx_rates.items():
            if currency != "USD":
                lines.append(f"  USD/{currency}: {rate:.2f}")
        return "\n".join(lines)

    def _query_position(self, timestamp: str) -> str:
        snapshot = self._get_snapshot()
        lines = [f"Portfolio @ {timestamp}:"]
        lines.append(f"  NAV: ${snapshot.total_nav:,.2f}  Cash: ${snapshot.cash:,.2f}")

        if not snapshot.positions:
            lines.append("  No positions")
        else:
            for key, pos in snapshot.positions.items():
                pnl = pos.unrealized_pnl
                pnl_pct = ((pos.current_price - pos.avg_cost) / pos.avg_cost * 100) if pos.avg_cost > 0 else 0
                lines.append(
                    f"  {pos.symbol}({pos.market.value}): {pos.quantity}sh "
                    f"avg=${pos.avg_cost:.2f} now=${pos.current_price:.2f} "
                    f"PnL=${pnl:+.2f} ({pnl_pct:+.1f}%)"
                )

        return "\n".join(lines)

    def _query_history(self, ticker: str, days: int, timestamp: str) -> str:
        if not ticker:
            return "Error: ticker is required"

        market = self._ticker_to_market(ticker)
        if market is None:
            return f"Error: cannot determine market for {ticker}"

        # Approximate: 48 bars/day for equities, 288 for crypto
        bars_per_day = 288 if market == Market.CRYPTO else 48
        lookback_bars = days * bars_per_day

        bars = self._data.load_bars(market, ticker, "2025-10-01", timestamp)
        if not bars:
            return f"No data for {ticker}"

        recent = bars[-lookback_bars:]
        if not recent:
            return f"No recent data for {ticker}"

        first = recent[0]
        last = recent[-1]
        change_pct = (last.close - first.close) / first.close * 100 if first.close > 0 else 0

        high = max(b.high for b in recent)
        low = min(b.low for b in recent)

        lines = [f"[{ticker}] {days}-day history:"]
        lines.append(f"  Range: ${low:.2f} - ${high:.2f}")
        lines.append(f"  Change: {change_pct:+.2f}% ({first.close:.2f} -> {last.close:.2f})")
        lines.append(f"  Bars: {len(recent)}")

        return "\n".join(lines)

    @staticmethod
    def _ticker_to_market(ticker: str) -> Market | None:
        if ticker.endswith(".US"):
            return Market.US
        elif ticker.endswith(".HK"):
            return Market.HK
        elif ticker.endswith(".CC") or "-" in ticker:
            return Market.CRYPTO
        elif ticker.startswith("sh.") or ticker.startswith("sz."):
            return Market.CN
        return None
