"""
MetricsEngine — computes benchmark performance metrics.

Metrics:
  - Total Return, Sharpe, Sortino, Max Drawdown, Volatility
  - Win Rate, Turnover, Average Holding Period

Enhanced metrics (v3):
  - Behavior: constraint_hits, rejected_orders, adjusted_orders, tool_usage
  - PnL Attribution: by_market, by_asset_type, by_symbol
  - Efficiency: fees, slippage, cash_drag
"""

from __future__ import annotations
import math
from collections import defaultdict
from typing import Any

from src.core.types import PortfolioSnapshot, TradeResult, OrderSide, Market
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
        failed_trades = [t for t in trades if not t.success]
        buy_trades = [t for t in successful_trades if t.order.side == OrderSide.BUY]
        sell_trades = [t for t in successful_trades if t.order.side == OrderSide.SELL]

        # Win rate (simplified: sell price > buy price for matched trades)
        win_rate = self._compute_win_rate(buy_trades, sell_trades)

        # Turnover
        total_traded = sum(t.cost for t in successful_trades)
        avg_nav = sum(navs) / len(navs) if navs else 1.0
        turnover = total_traded / avg_nav if avg_nav > 0 else 0.0

        # Fees and slippage
        total_fees = sum(t.fees for t in successful_trades)

        # PnL attribution by market
        pnl_by_market = self._compute_pnl_by_market(portfolio_history, successful_trades)

        return {
            # Core metrics
            "total_return": round(total_return * 100, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(max_dd * 100, 4),
            "volatility": round(vol * 100, 4),
            # Trade metrics
            "total_trades": len(successful_trades),
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "rejected_orders": len(failed_trades),
            "win_rate": round(win_rate * 100, 2),
            "turnover": round(turnover, 4),
            # NAV
            "initial_nav": initial_nav,
            "final_nav": final_nav,
            "total_return_usd": round(final_nav - initial_nav, 2),
            # Efficiency
            "total_fees_usd": round(total_fees, 2),
        }

    def compute_behavior_metrics(
        self,
        decisions: list[dict],
        trades: list[TradeResult],
        tool_calls: list[dict],
    ) -> dict[str, Any]:
        """Compute behavior metrics.

        Args:
            decisions: list of decision records
            trades: list of trade results
            tool_calls: list of tool call records
        """
        successful = [t for t in trades if t.success]
        failed = [t for t in trades if not t.success]

        # Constraint hits (from failed trade reasons)
        constraint_hits = sum(1 for t in failed if "constraint" in t.error.lower())
        tail_guard_hits = sum(1 for t in failed if "tail_guard" in t.error.lower())
        cooling_hits = sum(1 for t in failed if "cooling" in t.error.lower())

        # Tool usage
        total_tool_calls = len(tool_calls)
        tool_names = defaultdict(int)
        for tc in tool_calls:
            tool_names[tc.get("tool_name", "unknown")] += 1

        # Decision types
        decision_types = defaultdict(int)
        for d in decisions:
            decision_types[d.get("decision_type", "unknown")] += 1

        # Turnover level
        total_traded = sum(t.cost for t in successful)
        avg_nav = 100000  # placeholder
        turnover_ratio = total_traded / avg_nav if avg_nav > 0 else 0
        if turnover_ratio > 5:
            turnover_level = "high"
        elif turnover_ratio > 2:
            turnover_level = "moderate"
        else:
            turnover_level = "low"

        return {
            "constraint_hits": constraint_hits,
            "tail_guard_hits": tail_guard_hits,
            "cooling_hits": cooling_hits,
            "rejected_orders": len(failed),
            "adjusted_orders": sum(1 for t in successful if t.order.quantity != getattr(t, 'requested_quantity', t.order.quantity)),
            "total_tool_calls": total_tool_calls,
            "tool_usage": dict(tool_names),
            "decision_types": dict(decision_types),
            "turnover_level": turnover_level,
        }

    def compute_pnl_attribution(
        self,
        portfolio_history: list[PortfolioSnapshot],
        trades: list[TradeResult],
    ) -> dict[str, Any]:
        """Compute PnL attribution.

        Returns:
            {
                "total_pnl_usd": float,
                "by_market": {market: pnl},
                "by_symbol": {symbol: pnl},
                "fees_slippage_usd": float,
            }
        """
        if len(portfolio_history) < 2:
            return {"total_pnl_usd": 0, "by_market": {}, "by_symbol": {}, "fees_slippage_usd": 0}

        initial_nav = portfolio_history[0].total_nav
        final_nav = portfolio_history[-1].total_nav
        total_pnl = final_nav - initial_nav

        # PnL by market (from final snapshot positions)
        by_market: dict[str, float] = defaultdict(float)
        by_symbol: dict[str, float] = {}

        final_snap = portfolio_history[-1]
        for key, pos in final_snap.positions.items():
            pnl = pos.unrealized_pnl
            market = pos.market.value
            by_market[market] += pnl
            by_symbol[pos.symbol] = pnl

        # Add realized PnL from trades
        successful = [t for t in trades if t.success]
        fees_total = sum(t.fees for t in successful)

        return {
            "total_pnl_usd": round(total_pnl, 2),
            "by_market": {k: round(v, 2) for k, v in by_market.items()},
            "by_symbol": {k: round(v, 2) for k, v in sorted(by_symbol.items(), key=lambda x: x[1], reverse=True)[:10]},
            "fees_slippage_usd": round(fees_total, 2),
        }

    @staticmethod
    def _compute_pnl_by_market(
        history: list[PortfolioSnapshot], trades: list[TradeResult],
    ) -> dict[str, float]:
        """Compute PnL by market from final snapshot."""
        if not history:
            return {}
        final = history[-1]
        pnl: dict[str, float] = defaultdict(float)
        for key, pos in final.positions.items():
            pnl[pos.market.value] += pos.unrealized_pnl
        return dict(pnl)

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
