"""
MemoryManager — manages structured memory for LLM agent.

Memory components:
  daily_thesis       — daily market thesis (one active version)
  active_plans       — trading plans for held positions
  watchlist          — observation list with conditions
  avoid_list         — cooldown/blocked list
  recent_activity    — recent decisions and feedback
  execution_feedback — trade execution results
  session_summary    — per-market session summary
  daily_summary      — daily global summary
  rolling_behavior_notes — recent behavior patterns
"""

from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta

from src.core.types import (
    ActivePlan, PlanTrigger, PlanAction,
    WatchlistItem, AvoidItem, DailyThesis,
    RecentActivity, ExecutionFeedback,
    SessionSummary, DailySummary, MemoryState,
    PortfolioTarget, DecisionType,
    Market, TriggerType,
)


class MemoryManager:
    """Manages all structured memory components."""

    def __init__(self):
        # Daily thesis
        self._thesis: DailyThesis | None = None
        self._thesis_history: list[DailyThesis] = []

        # Plans
        self._plans: dict[str, ActivePlan] = {}  # symbol -> plan

        # Watchlist
        self._watchlist: dict[str, WatchlistItem] = {}  # symbol -> item

        # Avoid list
        self._avoid: dict[str, AvoidItem] = {}  # symbol -> item

        # Recent activity
        self._recent_decisions: list[dict] = []  # last N non-HOLD decisions
        self._recent_focused: list[dict] = []    # last N focused decisions
        self._recent_feedback: list[ExecutionFeedback] = []  # last N feedbacks
        self._recent_risk_changes: list[dict] = []  # last N risk changes

        # Summaries
        self._session_summaries: dict[str, SessionSummary] = {}  # market -> latest
        self._daily_summaries: list[DailySummary] = []

        # Behavior notes
        self._behavior_notes: list[str] = []

    # --- Daily Thesis ---

    def update_thesis(
        self, text: str, confidence: float, timestamp: str, expires_bars: int = 18,
    ) -> DailyThesis:
        """Update the daily thesis. Old version goes to history."""
        if self._thesis:
            self._thesis_history.append(self._thesis)

        self._thesis = DailyThesis(
            text=text,
            confidence=confidence,
            version=len(self._thesis_history) + 1,
            created_at=timestamp,
            expires_at=self._compute_expiry(timestamp, expires_bars),
        )
        return self._thesis

    def get_thesis(self) -> DailyThesis | None:
        """Get current active thesis."""
        return self._thesis

    def expire_thesis(self, timestamp: str) -> None:
        """Expire thesis if past expiry."""
        if self._thesis and self._thesis.expires_at and timestamp >= self._thesis.expires_at:
            self._thesis_history.append(self._thesis)
            self._thesis = None

    # --- Plans ---

    def create_plan(
        self,
        symbol: str,
        entry_price: float,
        entry_reason: str,
        timestamp: str,
        triggers: list[PlanTrigger] | None = None,
        pct_nav: float = 0.0,
    ) -> ActivePlan:
        """Create a new active plan for a position."""
        plan_id = f"plan_{symbol}_{timestamp.replace(':', '').replace('-', '').replace(' ', '_')}"
        plan = ActivePlan(
            plan_id=plan_id,
            symbol=symbol,
            status="active",
            entry_time=timestamp,
            entry_price=entry_price,
            current_pct_nav=pct_nav,
            entry_reason=entry_reason,
            last_review_time=timestamp,
            last_review_price=entry_price,
            peak_since_entry=entry_price,
            peak_since_last_review=entry_price,
            triggers=triggers or [],
        )
        self._plans[symbol] = plan
        return plan

    def update_plan(
        self,
        symbol: str,
        plan_action: PlanAction,
        triggers: list[PlanTrigger] | None = None,
        note: str = "",
        intended_horizon_bars: int = 36,
        timestamp: str = "",
        current_price: float = 0.0,
    ) -> ActivePlan | None:
        """Update a plan based on LLM's plan_update."""
        plan = self._plans.get(symbol)

        if plan_action == PlanAction.NO_CHANGE:
            return plan

        if plan_action == PlanAction.CREATE:
            plan = self.create_plan(
                symbol, current_price, note, timestamp, triggers,
            )
            if note:
                plan.plan_note = note
            plan.intended_horizon_bars = intended_horizon_bars
            return plan

        if plan is None:
            # Create new plan if doesn't exist
            plan = self.create_plan(
                symbol, current_price, note, timestamp, triggers,
            )
            if note:
                plan.plan_note = note
            plan.intended_horizon_bars = intended_horizon_bars
            return plan

        if plan_action == PlanAction.UPDATE:
            plan.plan_version += 1
            plan.last_review_time = timestamp or plan.last_review_time
            plan.last_review_price = current_price or plan.last_review_price
            plan.peak_since_last_review = current_price or plan.peak_since_last_review
            plan.intended_horizon_bars = intended_horizon_bars
            if note:
                plan.plan_note = note
            if triggers:
                plan.triggers = triggers
            return plan

        if plan_action == PlanAction.CLOSE:
            plan.status = "closed"
            closed_plan = self._plans.pop(symbol, None)
            return closed_plan

        return plan

    def get_plan(self, symbol: str) -> ActivePlan | None:
        """Get active plan for a symbol."""
        return self._plans.get(symbol)

    def get_all_plans(self) -> dict[str, ActivePlan]:
        """Get all active plans."""
        return dict(self._plans)

    def update_plan_peak(self, symbol: str, current_price: float) -> None:
        """Update peak prices for a plan."""
        plan = self._plans.get(symbol)
        if plan:
            if current_price > plan.peak_since_entry:
                plan.peak_since_entry = current_price
            if current_price > plan.peak_since_last_review:
                plan.peak_since_last_review = current_price

    def close_plan(self, symbol: str, timestamp: str) -> ActivePlan | None:
        """Close a plan (position sold)."""
        plan = self._plans.pop(symbol, None)
        if plan:
            plan.status = "closed"
        return plan

    # --- Watchlist ---

    def add_watch(
        self,
        symbol: str,
        reason: str,
        condition: dict | None = None,
        expires_bars: int = 24,
        timestamp: str = "",
    ) -> WatchlistItem:
        """Add or update a watchlist item."""
        item = WatchlistItem(
            symbol=symbol,
            reason=reason,
            desired_condition=condition or {},
            created_at=timestamp,
            expires_at=self._compute_expiry(timestamp, expires_bars),
        )
        self._watchlist[symbol] = item
        return item

    def remove_watch(self, symbol: str) -> WatchlistItem | None:
        """Remove a watchlist item."""
        return self._watchlist.pop(symbol, None)

    def get_watchlist(self) -> list[WatchlistItem]:
        """Get all active watchlist items."""
        return list(self._watchlist.values())

    def expire_watchlist(self, timestamp: str) -> list[WatchlistItem]:
        """Expire watchlist items past expiry. Returns expired items."""
        expired = []
        to_remove = []
        for sym, item in self._watchlist.items():
            if item.expires_at and timestamp >= item.expires_at:
                expired.append(item)
                to_remove.append(sym)
        for sym in to_remove:
            del self._watchlist[sym]
        return expired

    def watch_to_plan(self, symbol: str) -> WatchlistItem | None:
        """Convert a watchlist item to a plan (when bought)."""
        return self._watchlist.pop(symbol, None)

    # --- Avoid List ---

    def add_avoid(
        self,
        symbol: str,
        reason: str,
        expires_bars: int = 12,
        timestamp: str = "",
    ) -> AvoidItem:
        """Add or update an avoid item."""
        item = AvoidItem(
            symbol=symbol,
            reason=reason,
            created_at=timestamp,
            expires_at=self._compute_expiry(timestamp, expires_bars),
        )
        self._avoid[symbol] = item
        return item

    def remove_avoid(self, symbol: str) -> AvoidItem | None:
        """Remove an avoid item."""
        return self._avoid.pop(symbol, None)

    def get_avoid_list(self) -> list[AvoidItem]:
        """Get all active avoid items."""
        return list(self._avoid.values())

    def expire_avoid(self, timestamp: str) -> list[AvoidItem]:
        """Expire avoid items past expiry. Returns expired items."""
        expired = []
        to_remove = []
        for sym, item in self._avoid.items():
            if item.expires_at and timestamp >= item.expires_at:
                expired.append(item)
                to_remove.append(sym)
        for sym in to_remove:
            del self._avoid[sym]
        return expired

    # --- Apply LLM Updates ---

    def apply_memory_updates(self, updates: dict, timestamp: str = "") -> None:
        """Apply memory_updates from LLM output.

        Expected format:
        {
            "daily_thesis": "market is bullish" | null,
            "add_watch": [{"symbol": "AAPL.US", "reason": "RSI recovering"}],
            "add_avoid": [{"symbol": "TSLA.US", "reason": "high volatility"}],
            "remove_watch": ["AAPL.US"],
            "remove_avoid": ["TSLA.US"],
        }
        """
        if not updates:
            return

        # Update daily thesis
        thesis_text = updates.get("daily_thesis")
        if thesis_text:
            self._thesis = DailyThesis(
                text=thesis_text,
                created_at=timestamp,
            )

        # Add watchlist items
        for item in updates.get("add_watch", []):
            if isinstance(item, dict):
                symbol = item.get("symbol", "")
                reason = item.get("reason", "")
            elif isinstance(item, str):
                # Handle simple string format: "AAPL.US: RSI recovering"
                parts = item.split(":", 1)
                symbol = parts[0].strip()
                reason = parts[1].strip() if len(parts) > 1 else ""
            else:
                continue
            if symbol:
                self.add_watch(symbol, reason, timestamp=timestamp)

        # Add avoid items
        for item in updates.get("add_avoid", []):
            if isinstance(item, dict):
                symbol = item.get("symbol", "")
                reason = item.get("reason", "")
            elif isinstance(item, str):
                parts = item.split(":", 1)
                symbol = parts[0].strip()
                reason = parts[1].strip() if len(parts) > 1 else ""
            else:
                continue
            if symbol:
                self.add_avoid(symbol, reason, timestamp=timestamp)

        # Remove watchlist items
        for symbol in updates.get("remove_watch", []):
            if isinstance(symbol, str) and symbol:
                self.remove_watch(symbol)

        # Remove avoid items
        for symbol in updates.get("remove_avoid", []):
            if isinstance(symbol, str) and symbol:
                self.remove_avoid(symbol)

    def apply_plan_updates(self, updates: list[dict], timestamp: str = "") -> None:
        """Apply plan_updates from LLM output.

        Supports both the structured v3 format:
            {"symbol": "AAPL.US", "plan_action": "update", "triggers": [...], "plan_note": "..."}
        and the older shorthand format:
            {"symbol": "AAPL.US", "action": "update", "stop_loss": 150, "take_profit": 180}
        """
        if not updates:
            return

        for update in updates:
            if not isinstance(update, dict):
                continue

            symbol = update.get("symbol", "")
            if not symbol:
                continue

            action = self._parse_plan_action(update.get("plan_action", update.get("action", "no_change")))
            triggers = self._parse_plan_triggers(update.get("triggers", []))
            note = update.get("plan_note", update.get("note", "")) or ""
            legacy_levels = self._format_legacy_plan_levels(update)
            if legacy_levels:
                note = f"{note}; {legacy_levels}" if note else legacy_levels
            intended_horizon_bars = update.get("intended_horizon_bars", 36)
            current_price = update.get("current_price", update.get("entry_price", 0.0)) or 0.0

            self.update_plan(
                symbol=symbol,
                plan_action=action,
                triggers=triggers or None,
                note=note,
                intended_horizon_bars=intended_horizon_bars,
                timestamp=timestamp,
                current_price=float(current_price),
            )

    @staticmethod
    def _parse_plan_action(raw_action) -> PlanAction:
        if isinstance(raw_action, PlanAction):
            return raw_action
        try:
            return PlanAction(str(raw_action).lower())
        except ValueError:
            return PlanAction.NO_CHANGE

    @staticmethod
    def _parse_plan_triggers(raw_triggers) -> list[PlanTrigger]:
        triggers: list[PlanTrigger] = []
        for raw in raw_triggers or []:
            if isinstance(raw, PlanTrigger):
                triggers.append(raw)
                continue
            if not isinstance(raw, dict):
                continue
            trigger_type = raw.get("trigger_type", raw.get("type", ""))
            try:
                parsed_type = TriggerType(trigger_type)
            except ValueError:
                continue
            triggers.append(PlanTrigger(
                trigger_type=parsed_type,
                direction=raw.get("direction", ""),
                anchor=raw.get("anchor", ""),
                threshold_pct=raw.get("threshold_pct", 0.0) or 0.0,
                atr_multiple=raw.get("atr_multiple", 0.0) or 0.0,
                operator=raw.get("operator", ""),
                since=raw.get("since", ""),
                bars=raw.get("bars", 0) or 0,
                peak_anchor=raw.get("peak_anchor", ""),
                atr_source=raw.get("atr_source", ""),
            ))
        return triggers

    @staticmethod
    def _format_legacy_plan_levels(update: dict) -> str:
        parts = []
        if "stop_loss" in update:
            parts.append(f"legacy_stop_loss={update['stop_loss']}")
        if "take_profit" in update:
            parts.append(f"legacy_take_profit={update['take_profit']}")
        return ", ".join(parts)

    # --- Recent Activity ---

    def record_decision(
        self, decision_type: DecisionType, summary: str, timestamp: str,
    ) -> None:
        """Record a decision in recent activity."""
        entry = {"type": decision_type.value, "summary": summary, "timestamp": timestamp}

        if decision_type == DecisionType.FULL_DECISION:
            self._recent_decisions.append(entry)
            if len(self._recent_decisions) > 3:
                self._recent_decisions = self._recent_decisions[-3:]
        elif decision_type in (DecisionType.FOCUSED_POSITION, DecisionType.FOCUSED_MARKET_RISK):
            self._recent_focused.append(entry)
            if len(self._recent_focused) > 2:
                self._recent_focused = self._recent_focused[-2:]

    def record_feedback(self, feedback: ExecutionFeedback) -> None:
        """Record execution feedback."""
        self._recent_feedback.append(feedback)
        if len(self._recent_feedback) > 3:
            self._recent_feedback = self._recent_feedback[-3:]

    def record_risk_change(self, summary: str, timestamp: str) -> None:
        """Record a risk state change."""
        self._recent_risk_changes.append({"summary": summary, "timestamp": timestamp})
        if len(self._recent_risk_changes) > 1:
            self._recent_risk_changes = self._recent_risk_changes[-1:]

    def get_recent_activity(self) -> RecentActivity:
        """Get recent activity summary for prompt injection."""
        return RecentActivity(
            non_hold_decisions=[d["summary"] for d in self._recent_decisions],
            focused_decisions=[d["summary"] for d in self._recent_focused],
            execution_feedback=[f"{f.symbol}: {f.status}" for f in self._recent_feedback],
            risk_state_changes=[r["summary"] for r in self._recent_risk_changes],
        )

    # --- Summaries ---

    def save_session_summary(self, summary: SessionSummary) -> None:
        """Save a session summary."""
        self._session_summaries[summary.market] = summary

    def get_session_summary(self, market: str) -> SessionSummary | None:
        """Get latest session summary for a market."""
        return self._session_summaries.get(market)

    def save_daily_summary(self, summary: DailySummary) -> None:
        """Save a daily summary."""
        self._daily_summaries.append(summary)

    def get_previous_daily_summary(self) -> DailySummary | None:
        """Get the most recent daily summary."""
        return self._daily_summaries[-1] if self._daily_summaries else None

    # --- Behavior Notes ---

    def add_behavior_note(self, note: str) -> None:
        """Add a rolling behavior note."""
        self._behavior_notes.append(note)
        if len(self._behavior_notes) > 5:
            self._behavior_notes = self._behavior_notes[-5:]

    def get_behavior_notes(self) -> list[str]:
        """Get recent behavior notes."""
        return list(self._behavior_notes[-3:])

    # --- Full Memory State ---

    def get_memory_state(self, is_first_decision: bool = False) -> MemoryState:
        """Get complete memory state for prompt injection."""
        return MemoryState(
            previous_daily_summary=self.get_previous_daily_summary() if is_first_decision else None,
            daily_thesis=self._thesis,
            recent_activity=self.get_recent_activity(),
            watchlist=self.get_watchlist(),
            avoid_list=self.get_avoid_list(),
            recent_feedback=list(self._recent_feedback),
            rolling_behavior_notes=self.get_behavior_notes(),
        )

    # --- Helpers ---

    @staticmethod
    def _compute_expiry(timestamp: str, bars: int) -> str:
        """Compute expiry timestamp (bars * 5min from now)."""
        try:
            dt = datetime.strptime(timestamp[:16], "%Y-%m-%d %H:%M")
            expiry = dt + timedelta(minutes=bars * 5)
            return expiry.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return ""
