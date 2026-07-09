"""
Phase 2 unit tests — verify decision scheduler, event detector, and tail guard.
"""

import pytest
from src.core.types import (
    Market, DecisionType, RiskMode, TriggerType,
    PortfolioSnapshot, Position, ActivePlan, PlanTrigger,
)
from src.core.config import Config, DecisionScheduleConfig, OpenWindowConfig, CloseWindowConfig, TailGuardConfig
from src.portfolio.trigger_engine import TriggerEngine, TriggerEvent
from src.platform.scheduler import DecisionScheduler, DecisionRequest
from src.platform.event_detector import EventDetector
from src.portfolio.constraints import ConstraintEngine


# ============================================================
# Test: DecisionScheduler
# ============================================================

class TestDecisionScheduler:
    def setup_method(self):
        self.config = Config()
        self.scheduler = DecisionScheduler(self.config)

    def test_auto_hold_default(self):
        """Non-decision time should return auto_hold."""
        # 10:05 is not a 30min boundary
        request = self.scheduler.schedule(
            timestamp="2026-01-07 10:05",
            open_markets=[Market.US],
            closed_markets=[Market.HK, Market.CN],
        )
        assert request.decision_type == DecisionType.AUTO_HOLD

    def test_normal_decision_30min(self):
        """30min boundaries should trigger full decision."""
        # 10:00 is a 30min boundary
        request = self.scheduler.schedule(
            timestamp="2026-01-07 10:00",
            open_markets=[Market.US],
            closed_markets=[Market.HK, Market.CN],
        )
        assert request.decision_type == DecisionType.FULL_DECISION

    def test_open_window(self):
        """First 30min after US open (14:30) should trigger at 15min intervals."""
        # US opens at 14:30, so 14:45 should be in open window
        request = self.scheduler.schedule(
            timestamp="2026-01-07 14:45",
            open_markets=[Market.US],
            closed_markets=[Market.HK, Market.CN],
        )
        assert request.decision_type == DecisionType.FULL_DECISION

    def test_close_window(self):
        """Last 30min before US close (21:00) should trigger at 15min intervals."""
        # US closes at 21:00, so 20:30 should be in close window
        request = self.scheduler.schedule(
            timestamp="2026-01-07 20:30",
            open_markets=[Market.US],
            closed_markets=[Market.HK, Market.CN],
        )
        assert request.decision_type == DecisionType.FULL_DECISION

    def test_tail_guard_active(self):
        """Last 15min before US close should have tail guard active."""
        # US closes at 21:00, so 20:45 should have tail guard
        request = self.scheduler.schedule(
            timestamp="2026-01-07 20:45",
            open_markets=[Market.US],
            closed_markets=[Market.HK, Market.CN],
        )
        assert request.tail_guard_active is True
        assert "US" in request.tail_guard_markets

    def test_tail_guard_not_active_normal(self):
        """Normal time should not have tail guard."""
        request = self.scheduler.schedule(
            timestamp="2026-01-07 15:00",
            open_markets=[Market.US],
            closed_markets=[Market.HK, Market.CN],
        )
        assert request.tail_guard_active is False

    def test_focused_position_from_triggers(self):
        """P1 trigger events should trigger focused position decision."""
        events = [TriggerEvent(
            symbol="AAPL.US",
            plan_id="p1",
            trigger_type=TriggerType.PNL_PCT,
            priority="P1",
            trigger_detail={"operator": "<=", "current_pnl_pct": -0.03},
            actual_value=-0.03,
            threshold=-0.025,
        )]
        request = self.scheduler.schedule(
            timestamp="2026-01-07 10:05",
            open_markets=[Market.US],
            closed_markets=[Market.HK, Market.CN],
            trigger_events=events,
        )
        assert request.decision_type == DecisionType.FOCUSED_POSITION
        assert request.priority == "P1"
        assert "AAPL.US" in request.scope_symbols

    def test_get_open_markets(self):
        """Should return correct open markets."""
        # US hours
        markets = self.scheduler.get_open_markets("2026-01-07 15:00")
        assert Market.US in markets
        assert Market.CRYPTO in markets

        # HK hours (but US closed)
        markets = self.scheduler.get_open_markets("2026-01-07 03:00")
        assert Market.HK in markets
        assert Market.CN in markets
        assert Market.CRYPTO in markets

    def test_get_closed_markets(self):
        """Should return correct closed markets."""
        # US hours — HK/CN closed
        closed = self.scheduler.get_closed_markets("2026-01-07 15:00")
        assert Market.HK in closed
        assert Market.CN in closed
        assert Market.CRYPTO not in closed  # crypto always open

    def test_weekend_all_closed(self):
        """Weekend should have no stock markets open."""
        # Saturday
        markets = self.scheduler.get_open_markets("2026-01-10 15:00")
        assert Market.US not in markets
        assert Market.HK not in markets
        assert Market.CN not in markets
        assert Market.CRYPTO in markets


# ============================================================
# Test: Tail Guard in ConstraintEngine
# ============================================================

class TestTailGuard:
    def setup_method(self):
        self.config = Config()
        self.constraints = ConstraintEngine(self.config)

    def test_tail_guard_blocks_new_buy(self):
        """Tail guard should block new buys."""
        self.constraints.set_tail_guard(True, ["US"])

        positions = {}  # no existing position
        ok, reason = self.constraints.validate_buy(
            "AAPL.US", Market.US, 100, 150.0, 100000.0, positions,
        )
        assert ok is False
        assert "tail_guard" in reason

    def test_tail_guard_blocks_increase(self):
        """Tail guard should block position increases."""
        self.constraints.set_tail_guard(True, ["US"])

        positions = {
            "US:AAPL": Position(symbol="AAPL", market=Market.US, quantity=10, avg_cost=150.0),
        }
        ok, reason = self.constraints.validate_buy(
            "AAPL.US", Market.US, 100, 150.0, 100000.0, positions,
        )
        assert ok is False
        assert "tail_guard" in reason

    def test_tail_guard_allows_other_markets(self):
        """Tail guard for US should not block HK."""
        self.constraints.set_tail_guard(True, ["US"])

        positions = {}
        ok, reason = self.constraints.validate_buy(
            "0700.HK", Market.HK, 100, 300.0, 100000.0, positions,
        )
        # Should pass tail guard check (may fail for other reasons)
        assert "tail_guard" not in reason

    def test_tail_guard_disabled(self):
        """When tail guard is disabled, all buys should pass."""
        self.constraints.set_tail_guard(False, [])

        positions = {}
        ok, reason = self.constraints.validate_buy(
            "AAPL.US", Market.US, 100, 150.0, 100000.0, positions,
        )
        # Should pass tail guard check
        assert "tail_guard" not in reason


# ============================================================
# Test: EventDetector
# ============================================================

class TestEventDetector:
    def setup_method(self):
        self.config = Config()
        self.trigger_engine = TriggerEngine()
        # EventDetector needs features — we'll test with mock
        # For now, test the logic indirectly through scheduler

    def test_trigger_event_to_decision(self):
        """Trigger events should flow through scheduler to decision."""
        scheduler = DecisionScheduler(self.config)

        events = [
            TriggerEvent(
                symbol="AAPL.US",
                plan_id="p1",
                trigger_type=TriggerType.TRAILING_DRAWDOWN_PCT,
                priority="P1",
                trigger_detail={"drawdown_pct": 0.03},
                actual_value=0.03,
                threshold=0.02,
            ),
        ]

        request = scheduler.schedule(
            timestamp="2026-01-07 10:05",
            open_markets=[Market.US],
            closed_markets=[Market.HK, Market.CN],
            trigger_events=events,
        )

        assert request.decision_type == DecisionType.FOCUSED_POSITION
        assert "AAPL.US" in request.scope_symbols


# ============================================================
# Test: Config
# ============================================================

class TestConfigPhase2:
    def test_decision_schedule_config(self):
        config = Config()
        schedule = config.decision_schedule
        assert schedule.normal_interval_minutes == 30
        assert schedule.open_window.enabled is True
        assert schedule.open_window.minutes_after_open == 30
        assert schedule.open_window.interval_minutes == 15
        assert schedule.close_window.enabled is True
        assert schedule.close_window.minutes_before_close == 30
        assert schedule.close_window.interval_minutes == 15

    def test_tail_guard_config(self):
        config = Config()
        tg = config.tail_guard
        assert tg.enabled is True
        assert tg.minutes_before_close == 15
        assert tg.block_new_buy is True
        assert tg.block_increase_position is True
        assert tg.allow_reduce_close_hold is True

    def test_market_close_rule_config(self):
        config = Config()
        mcr = config.market_close_rule
        assert mcr.at_or_after_close_no_same_session_trade is True
        assert mcr.final_bar_usable_for_summary is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
