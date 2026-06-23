"""
SummaryEngine — generates session and daily summaries.

Session Summary:
  Triggered after market close (e.g., HK 08:05 UTC, US 21:05 UTC)
  Summarizes market session, model actions, open positions, risk notes

Daily Summary:
  Triggered at benchmark boundary (00:00 UTC)
  Summarizes full day: NAV, decisions, what worked/failed, carryover positions

Both summaries use a fixed summarizer (not the trading model) for fairness.
"""

from __future__ import annotations
from datetime import datetime

from src.core.types import (
    Market, SessionSummary, DailySummary,
    PortfolioSnapshot, Decision, TradeResult,
    ExecutionFeedback,
)


class SummaryEngine:
    """Generates session and daily summaries."""

    def __init__(self):
        self._session_summaries: list[SessionSummary] = []
        self._daily_summaries: list[DailySummary] = []

    def generate_session_summary(
        self,
        market: Market,
        timestamp: str,
        snapshot: PortfolioSnapshot,
        decisions: list[dict],
        trades: list[TradeResult],
        plans: list[dict],
    ) -> SessionSummary:
        """Generate a session summary for a closing market.

        Args:
            market: which market just closed
            timestamp: close timestamp
            snapshot: current portfolio snapshot
            decisions: list of decision records for this session
            trades: list of trade results for this session
            plans: list of active plans for this market
        """
        # Filter to this market
        market_decisions = [d for d in decisions if d.get("market") == market.value]
        market_trades = [t for t in trades if t.order.market == market]

        # Build model actions
        model_actions = []
        for d in market_decisions:
            action = d.get("action", "hold")
            symbol = d.get("symbol", "")
            if action != "hold":
                model_actions.append(f"{action} {symbol}")

        # Build open positions
        open_positions = []
        for key, pos in snapshot.positions.items():
            if pos.market == market:
                plan = next((p for p in plans if p.get("symbol") == pos.symbol), None)
                open_positions.append({
                    "symbol": pos.symbol,
                    "plan": plan.get("note", "") if plan else "",
                })

        # Compute market PnL
        market_pnl = 0.0
        for key, pos in snapshot.positions.items():
            if pos.market == market:
                market_pnl += pos.unrealized_pnl

        # Build risk notes
        risk_notes = []
        nav = snapshot.total_nav
        if nav > 0:
            exposure = snapshot.market_exposure.get(market, 0)
            exposure_pct = exposure / nav
            if exposure_pct > 0.4:
                risk_notes.append(f"{market.value} exposure at {exposure_pct:.1%} — near limit")

        summary = SessionSummary(
            market=market.value,
            session_date=timestamp[:10],
            market_read=f"{market.value} session closed. PnL: ${market_pnl:+,.0f}",
            model_actions=model_actions[:5],
            open_positions=open_positions,
            risk_notes=risk_notes,
            created_at=timestamp,
        )

        self._session_summaries.append(summary)
        return summary

    def generate_daily_summary(
        self,
        date: str,
        nav_start: float,
        nav_end: float,
        all_decisions: list[dict],
        all_trades: list[TradeResult],
        session_summaries: list[SessionSummary],
        snapshot: PortfolioSnapshot,
        plans: list[dict],
    ) -> DailySummary:
        """Generate a daily global summary at benchmark boundary.

        Args:
            date: date string (YYYY-MM-DD)
            nav_start: NAV at start of day
            nav_end: NAV at end of day
            all_decisions: all decisions made today
            all_trades: all trades executed today
            session_summaries: session summaries from today
            snapshot: final portfolio snapshot
            plans: all active plans
        """
        daily_return = (nav_end - nav_start) / nav_start if nav_start > 0 else 0.0

        # Major decisions
        major_decisions = []
        for d in all_decisions:
            action = d.get("action", "hold")
            if action in ("trade", "rebalance"):
                symbol = d.get("symbol", "")
                side = d.get("side", "")
                major_decisions.append(f"{side.upper()} {symbol}")

        # What worked / failed
        what_worked = []
        what_failed = []
        for t in all_trades:
            if t.success:
                pnl = t.order.quantity * (t.price - t.order.quantity) if t.order.side.value == "buy" else 0
                # Simplified: just record successful trades
                what_worked.append(f"Executed {t.order.side.value} {t.order.symbol}")
            else:
                what_failed.append(f"Failed {t.order.side.value} {t.order.symbol}: {t.error}")

        # Carryover positions
        carryover = []
        for key, pos in snapshot.positions.items():
            plan = next((p for p in plans if p.get("symbol") == pos.symbol), None)
            carryover.append({
                "symbol": pos.symbol,
                "plan": plan.get("note", "") if plan else "",
            })

        # Behavior stats
        total_trades = len(all_trades)
        successful = sum(1 for t in all_trades if t.success)
        rejected = sum(1 for t in all_trades if not t.success)

        behavior = {
            "trades": total_trades,
            "successful": successful,
            "rejected": rejected,
            "decisions": len(all_decisions),
        }

        # Build market read from session summaries
        market_reads = [s.market_read for s in session_summaries]
        market_read = "; ".join(market_reads) if market_reads else "No market activity"

        summary = DailySummary(
            date=date,
            nav_start=nav_start,
            nav_end=nav_end,
            daily_return_pct=daily_return,
            market_read=market_read,
            major_decisions=major_decisions[:5],
            what_worked=what_worked[:3],
            what_failed=what_failed[:3],
            carryover_positions=carryover,
            avoid_next_day=[],
            behavior=behavior,
            created_at=f"{date} 00:00",
        )

        self._daily_summaries.append(summary)
        return summary

    def get_latest_session_summary(self, market: str) -> SessionSummary | None:
        """Get latest session summary for a market."""
        for s in reversed(self._session_summaries):
            if s.market == market:
                return s
        return None

    def get_previous_daily_summary(self) -> DailySummary | None:
        """Get the most recent daily summary."""
        return self._daily_summaries[-1] if self._daily_summaries else None
