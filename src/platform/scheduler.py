"""
DecisionScheduler — determines what type of decision to make at each timestamp.

Decision types:
  auto_hold                    — no LLM call, system records hold
  full_decision                — full market review
  focused_position_decision    — single position event
  focused_market_or_risk_decision — market/risk event

Schedule (per market):
  normal:            every 30min
  open_window:       first 30min after open, every 15min
  close_window:      last 30min before close, every 15min
  tail_guard:        last 15min before close, block new buys/increases

Market close times (UTC):
  CN:   07:00
  HK:   08:00
  US:   21:00
  CRYPTO: 24/7 (no close)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from src.core.types import (
    Market, DecisionType, RiskMode,
)
from src.core.config import Config, DecisionScheduleConfig, TailGuardConfig

# Import TriggerEvent from trigger_engine (not types)
from src.portfolio.trigger_engine import TriggerEvent


@dataclass
class DecisionRequest:
    """A request for a decision at a specific timestamp."""
    timestamp: str
    decision_type: DecisionType
    priority: str = "P3"  # P0=system, P1=must, P2=can merge, P3=info
    scope_market: str = ""
    scope_symbols: list[str] = field(default_factory=list)
    trigger_events: list[TriggerEvent] = field(default_factory=list)
    tail_guard_active: bool = False
    tail_guard_markets: list[str] = field(default_factory=list)


class DecisionScheduler:
    """Determines what type of decision to make at each timestamp."""

    # Market close times (UTC HH:MM)
    MARKET_CLOSE = {
        Market.CN: "07:00",
        Market.HK: "08:00",
        Market.US: "21:00",
    }

    # Market open times (UTC HH:MM)
    MARKET_OPEN = {
        Market.CN: "01:30",
        Market.HK: "01:30",
        Market.US: "14:30",
    }

    def __init__(self, config: Config):
        self._config = config
        self._schedule = config.decision_schedule
        self._tail_guard = config.tail_guard

        # Track last decision time per type
        self._last_full_decision: str = ""
        self._last_market_decision: dict[str, str] = {}  # market -> timestamp
        self._last_focused: dict[str, str] = {}  # symbol -> timestamp

        # Track which markets have had their close-window decisions
        self._close_window_done: dict[str, bool] = {}  # market -> done

    def reset(self) -> None:
        """Reset state for a new benchmark day."""
        self._last_full_decision = ""
        self._last_market_decision = {}
        self._last_focused = {}
        self._close_window_done = {}

    def schedule(
        self,
        timestamp: str,
        open_markets: list[Market],
        closed_markets: list[Market],
        trigger_events: list[TriggerEvent] | None = None,
        risk_mode: RiskMode = RiskMode.GREEN,
    ) -> DecisionRequest:
        """Determine what decision to make at this timestamp.

        Args:
            timestamp: current UTC timestamp
            open_markets: currently open markets
            closed_markets: currently closed markets
            trigger_events: detected trigger events
            risk_mode: current risk mode

        Returns:
            DecisionRequest with decision type and context
        """
        trigger_events = trigger_events or []
        time_part = timestamp[11:16] if len(timestamp) >= 16 else ""

        # Check tail guard status
        tail_guard_active, tail_guard_markets = self._check_tail_guard(time_part, open_markets)

        # Priority 0: System forced events (not handled here, handled by EventDetector)

        # Priority 1: Focused position events from triggers
        position_events = [e for e in trigger_events if e.priority in ("P1", "P2")]
        if position_events:
            # Group by priority
            p1_events = [e for e in position_events if e.priority == "P1"]
            if p1_events:
                return DecisionRequest(
                    timestamp=timestamp,
                    decision_type=DecisionType.FOCUSED_POSITION,
                    priority="P1",
                    scope_symbols=[e.symbol for e in p1_events],
                    trigger_events=p1_events,
                    tail_guard_active=tail_guard_active,
                    tail_guard_markets=tail_guard_markets,
                )

        # Priority 2: Close window decisions (30min before close, every 15min)
        for market in open_markets:
            if self._in_close_window(time_part, market):
                if not self._close_window_done.get(market.value, False):
                    self._close_window_done[market.value] = True
                    return DecisionRequest(
                        timestamp=timestamp,
                        decision_type=DecisionType.FULL_DECISION,
                        priority="P2",
                        scope_market=market.value,
                        tail_guard_active=tail_guard_active,
                        tail_guard_markets=tail_guard_markets,
                    )

        # Priority 2: Open window decisions (30min after open, every 15min)
        for market in open_markets:
            if self._in_open_window(time_part, market):
                return DecisionRequest(
                    timestamp=timestamp,
                    decision_type=DecisionType.FULL_DECISION,
                    priority="P2",
                    scope_market=market.value,
                    tail_guard_active=tail_guard_active,
                    tail_guard_markets=tail_guard_markets,
                )

        # Priority 3: Normal full decision (every 30min)
        if self._is_normal_decision_time(time_part):
            # Check if we already did a full decision recently
            if not self._last_full_decision or self._minutes_since(timestamp, self._last_full_decision) >= 25:
                self._last_full_decision = timestamp
                return DecisionRequest(
                    timestamp=timestamp,
                    decision_type=DecisionType.FULL_DECISION,
                    priority="P3",
                    tail_guard_active=tail_guard_active,
                    tail_guard_markets=tail_guard_markets,
                )

        # Priority 3: Focused events (P2 triggers that weren't P1)
        p2_events = [e for e in trigger_events if e.priority == "P2"]
        if p2_events:
            return DecisionRequest(
                timestamp=timestamp,
                decision_type=DecisionType.FOCUSED_POSITION,
                priority="P2",
                scope_symbols=[e.symbol for e in p2_events],
                trigger_events=p2_events,
                tail_guard_active=tail_guard_active,
                tail_guard_markets=tail_guard_markets,
            )

        # Default: auto_hold
        return DecisionRequest(
            timestamp=timestamp,
            decision_type=DecisionType.AUTO_HOLD,
            priority="P3",
            tail_guard_active=tail_guard_active,
            tail_guard_markets=tail_guard_markets,
        )

    def _in_open_window(self, time_part: str, market: Market) -> bool:
        """Check if we're in the open window (first 30min after market open)."""
        if not self._schedule.open_window.enabled:
            return False

        open_time = self.MARKET_OPEN.get(market)
        if not open_time:
            return False

        try:
            open_dt = datetime.strptime(open_time, "%H:%M")
            current_dt = datetime.strptime(time_part, "%H:%M")
            minutes_after = (current_dt - open_dt).total_seconds() / 60

            if minutes_after < 0 or minutes_after > self._schedule.open_window.minutes_after_open:
                return False

            # Check if we're at an interval point
            interval = self._schedule.open_window.interval_minutes
            if minutes_after == 0:
                return True  # exact open time
            if minutes_after % interval == 0:
                return True

            return False
        except ValueError:
            return False

    def _in_close_window(self, time_part: str, market: Market) -> bool:
        """Check if we're in the close window (last 30min before market close)."""
        if not self._schedule.close_window.enabled:
            return False

        close_time = self.MARKET_CLOSE.get(market)
        if not close_time:
            return False

        try:
            close_dt = datetime.strptime(close_time, "%H:%M")
            current_dt = datetime.strptime(time_part, "%H:%M")
            minutes_before = (close_dt - current_dt).total_seconds() / 60

            if minutes_before < 0 or minutes_before > self._schedule.close_window.minutes_before_close:
                return False

            # Check if we're at an interval point
            interval = self._schedule.close_window.interval_minutes
            if minutes_before == 0:
                return self._schedule.close_window.include_close_time
            if minutes_before % interval == 0:
                return True

            return False
        except ValueError:
            return False

    def _check_tail_guard(self, time_part: str, open_markets: list[Market]) -> tuple[bool, list[str]]:
        """Check if tail guard is active for any market."""
        if not self._tail_guard.enabled:
            return False, []

        active_markets = []
        for market in open_markets:
            close_time = self.MARKET_CLOSE.get(market)
            if not close_time:
                continue

            try:
                close_dt = datetime.strptime(close_time, "%H:%M")
                current_dt = datetime.strptime(time_part, "%H:%M")
                minutes_before = (close_dt - current_dt).total_seconds() / 60

                if 0 <= minutes_before <= self._tail_guard.minutes_before_close:
                    active_markets.append(market.value)
            except ValueError:
                continue

        return len(active_markets) > 0, active_markets

    def _is_normal_decision_time(self, time_part: str) -> bool:
        """Check if this is a normal decision time (every 30min)."""
        try:
            dt = datetime.strptime(time_part, "%H:%M")
            return dt.minute % self._schedule.normal_interval_minutes == 0
        except ValueError:
            return False

    @staticmethod
    def _minutes_since(timestamp: str, reference: str) -> int:
        """Calculate minutes between two timestamps."""
        try:
            t1 = datetime.strptime(timestamp[:16], "%Y-%m-%d %H:%M")
            t2 = datetime.strptime(reference[:16], "%Y-%m-%d %H:%M")
            return int((t1 - t2).total_seconds() / 60)
        except ValueError:
            return 999

    def is_tail_guard_active(self, timestamp: str, market: Market) -> bool:
        """Check if tail guard is active for a specific market."""
        time_part = timestamp[11:16] if len(timestamp) >= 16 else ""
        close_time = self.MARKET_CLOSE.get(market)
        if not close_time:
            return False

        try:
            close_dt = datetime.strptime(close_time, "%H:%M")
            current_dt = datetime.strptime(time_part, "%H:%M")
            minutes_before = (close_dt - current_dt).total_seconds() / 60
            return 0 <= minutes_before <= self._tail_guard.minutes_before_close
        except ValueError:
            return False

    def get_open_markets(self, timestamp: str) -> list[Market]:
        """Get list of open markets at this timestamp."""
        time_part = timestamp[11:16] if len(timestamp) >= 16 else ""
        open_markets = []

        # Check weekday (stock markets closed on weekends)
        try:
            dt = datetime.strptime(timestamp[:16], "%Y-%m-%d %H:%M")
            is_weekend = dt.weekday() >= 5
        except ValueError:
            is_weekend = False

        if not is_weekend:
            for market, open_time in self.MARKET_OPEN.items():
                close_time = self.MARKET_CLOSE.get(market)
                if not close_time:
                    continue
                if open_time <= time_part < close_time:
                    open_markets.append(market)

        # Crypto always open
        open_markets.append(Market.CRYPTO)

        return open_markets

    def get_closed_markets(self, timestamp: str) -> list[Market]:
        """Get list of closed markets at this timestamp."""
        open_markets = set(self.get_open_markets(timestamp))
        return [m for m in [Market.US, Market.HK, Market.CN, Market.CRYPTO] if m not in open_markets]
