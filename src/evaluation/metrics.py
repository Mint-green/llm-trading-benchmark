"""
MetricsEngine — computes benchmark performance metrics.

Metrics:
  - Total Return
  - Sharpe Ratio
  - Sortino Ratio
  - Max Drawdown
  - Volatility (annualized)
  - Win Rate
  - Average Holding Period
  - Turnover Rate
"""

from __future__ import annotations
import math
from typing import Any

from src.core.types import PortfolioSnapshot, TradeResult, OrderSide
from src.core.interfaces import IMetricsEngine


class MetricsEngine(IMetricsEngine):
    """Computes benchmark performance metrics."""

    def __init__(self, risk_free_rate: float = 0.05):
        """risk_free_rate: annualized (e.g. 0.05 = 5%)"""
        self._risk_free = risk_free_rate

    def compute(
        self, portfolio_history: list[PortfolioSnapshot], trades: list[TradeResult],
    ) -> dict[str, float]:
        """Compute all metrics from portfolio history and trade log."""
        if len(portfolio_history) < 2:
            return self._empty_metrics()

        navs = [s.total_nav for s in portfolio_history]
        initial_nav = navs[0]
        final_nav = navs[-1]

        # Returns per period
        returns = []
        for i in range(1, len(navs)):
            if navs[i - 1] > 0:
                returns.append((navs[i] - navs[i - 1]) / navs[i - 1])

        # Total return
        total_return = (final_nav - initial_nav) / initial_nav if initial_nav > 0 else 0.0

        # Annualization factor (assuming 5-min snapshots, ~48 per day, ~252 trading days)
        periods_per_year = 48 * 252  # ~12,096
        actual_periods = len(returns)
        ann_factor = periods_per_year / max(actual_periods, 1)

        # Volatility (annualized)
        vol = self._std(returns) * math.sqrt(periods_per_year) if returns else 0.0

        # Sharpe Ratio
        excess_return = total_return * ann_factor - self._risk_free
        sharpe = excess_return / vol if vol > 0 else 0.0

        # Sortino Ratio (downside deviation)
        downside = [r for r in returns if r < 0]
        downside_std = self._std(downside) * math.sqrt(periods_per_year) if downside else 0.0
        sortino = excess_return / downside_std if downside_std > 0 else 0.0

        # Max Drawdown
        max_dd = self._max_drawdown(navs)

        # Trade statistics
        successful_trades = [t for t in trades if t.success]
        buy_trades = [t for t in successful_trades if t.order.side == OrderSide.BUY]
        sell_trades = [t for t in successful_trades if t.order.side == OrderSide.SELL]

        # Win rate (simplified: sell price > buy price for matched trades)
        win_rate = self._compute_win_rate(buy_trades, sell_trades)

        # Turnover
        total_traded = sum(t.cost for t in successful_trades)
        avg_nav = sum(navs) / len(navs) if navs else 1.0
        turnover = total_traded / avg_nav if avg_nav > 0 else 0.0

        return {
            "total_return": round(total_return * 100, 4),  # percentage
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(max_dd * 100, 4),  # percentage
            "volatility": round(vol * 100, 4),  # percentage
            "total_trades": len(successful_trades),
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "win_rate": round(win_rate * 100, 2),  # percentage
            "turnover": round(turnover, 4),
            "initial_nav": initial_nav,
            "final_nav": final_nav,
            "total_return_usd": round(final_nav - initial_nav, 2),
        }

    @staticmethod
    def _std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    @staticmethod
    def _max_drawdown(navs: list[float]) -> float:
        if not navs:
            return 0.0
        peak = navs[0]
        max_dd = 0.0
        for nav in navs:
            if nav > peak:
                peak = nav
            dd = (peak - nav) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    @staticmethod
    def _compute_win_rate(buys: list[TradeResult], sells: list[TradeResult]) -> float:
        """Simplified win rate: match sells to buys by symbol."""
        if not sells:
            return 0.0

        # Build buy price lookup
        buy_prices: dict[str, list[float]] = {}
        for t in buys:
            sym = t.order.symbol
            buy_prices.setdefault(sym, []).append(t.price)

        wins = 0
        for sell in sells:
            sym = sell.order.symbol
            if sym in buy_prices and buy_prices[sym]:
                avg_buy = sum(buy_prices[sym]) / len(buy_prices[sym])
                if sell.price > avg_buy:
                    wins += 1
                buy_prices[sym].pop(0)  # FIFO matching

        return wins / len(sells) if sells else 0.0

    @staticmethod
    def _empty_metrics() -> dict[str, float]:
        return {k: 0.0 for k in [
            "total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown",
            "volatility", "total_trades", "buy_trades", "sell_trades",
            "win_rate", "turnover", "initial_nav", "final_nav", "total_return_usd",
        ]}
