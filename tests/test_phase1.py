"""
Phase 1 unit tests — verify core data structures, triggers, memory, and context.
"""

import pytest
from src.core.types import (
    Market, OrderSide, CandidateBucket, DecisionType, PlanAction, TriggerType,
    RiskMode, PortfolioTarget, ActivePlan, PlanTrigger,
    WatchlistItem, AvoidItem, DailyThesis, RecentActivity,
    ExecutionFeedback, SessionSummary, DailySummary, MemoryState,
    CandidateInBucket, CandidateBuckets,
    Position, PortfolioSnapshot, TradeOrder,
)
from src.core.config import Config, TriggerConfig, CryptoTriggerConfig
from src.data.screener import Screener, CandidateScore
from src.portfolio.trigger_engine import TriggerEngine, TriggerEvent
from src.agent.memory_manager import MemoryManager
from src.agent.context import ContextBuilder


# ============================================================
# Test: Types
# ============================================================

class TestTypes:
    def test_candidate_bucket_enum(self):
        assert CandidateBucket.HELD_POSITIONS.value == "held_positions"
        assert CandidateBucket.TREND_LEADERS.value == "trend_leaders"
        assert len(CandidateBucket) == 8

    def test_decision_type_enum(self):
        assert DecisionType.AUTO_HOLD.value == "auto_hold"
        assert DecisionType.FULL_DECISION.value == "full_decision"
        assert DecisionType.FOCUSED_POSITION.value == "focused_position_decision"
        assert DecisionType.FOCUSED_MARKET_RISK.value == "focused_market_or_risk_decision"

    def test_plan_action_enum(self):
        assert PlanAction.CREATE.value == "create"
        assert PlanAction.UPDATE.value == "update"
        assert PlanAction.CLOSE.value == "close"
        assert PlanAction.NO_CHANGE.value == "no_change"

    def test_trigger_type_enum(self):
        assert len(TriggerType) == 9
        assert TriggerType.PRICE_MOVE_PCT.value == "price_move_pct"
        assert TriggerType.BARS_ELAPSED.value == "bars_elapsed"

    def test_risk_mode_enum(self):
        assert RiskMode.GREEN.value == "GREEN"
        assert RiskMode.YELLOW.value == "YELLOW"
        assert RiskMode.RED.value == "RED"

    def test_portfolio_target(self):
        target = PortfolioTarget(symbol="AAPL.US", target_pct_nav=0.05)
        assert target.symbol == "AAPL.US"
        assert target.target_pct_nav == 0.05
        assert target.asset_type == "equity"
        assert target.priority == 1

    def test_active_plan(self):
        plan = ActivePlan(
            plan_id="plan_1",
            symbol="AAPL.US",
            entry_price=150.0,
            triggers=[],
        )
        assert plan.status == "active"
        assert plan.peak_since_entry == 0.0

    def test_plan_trigger(self):
        trigger = PlanTrigger(
            trigger_type=TriggerType.PRICE_MOVE_PCT,
            direction="down",
            threshold_pct=0.02,
        )
        assert trigger.trigger_type == TriggerType.PRICE_MOVE_PCT
        assert trigger.threshold_pct == 0.02

    def test_memory_state(self):
        state = MemoryState()
        assert state.previous_daily_summary is None
        assert state.daily_thesis is None
        assert len(state.watchlist) == 0
        assert len(state.avoid_list) == 0


# ============================================================
# Test: Config
# ============================================================

class TestConfig:
    def test_trigger_config_defaults(self):
        config = TriggerConfig()
        assert config.price_move_pct == 0.02
        assert config.atr_move_multiple == 1.5
        assert config.pnl_pct_threshold == -0.03
        assert config.trailing_drawdown_pct == 0.02
        assert config.trailing_atr_multiple == 2.0
        assert config.bars_elapsed == 6

    def test_crypto_trigger_config_defaults(self):
        config = CryptoTriggerConfig()
        assert config.price_move_pct == 0.05
        assert config.pnl_pct_threshold == -0.05
        assert config.trailing_drawdown_pct == 0.03

    def test_config_has_new_fields(self):
        config = Config()
        assert config.benchmark_boundary_utc == "00:00"
        assert config.session_summary_minutes_after_close == 5
        assert config.trigger_config.price_move_pct == 0.02
        assert config.crypto_trigger_config.price_move_pct == 0.05
        assert config.tail_guard.enabled is True
        assert config.decision_schedule.normal_interval_minutes == 30


# ============================================================
# Test: TriggerEngine
# ============================================================

class TestTriggerEngine:
    def setup_method(self):
        self.engine = TriggerEngine()

    def test_price_move_triggered(self):
        plan = ActivePlan(
            plan_id="p1",
            symbol="AAPL.US",
            last_review_price=100.0,
            triggers=[PlanTrigger(
                trigger_type=TriggerType.PRICE_MOVE_PCT,
                direction="down",
                anchor="last_review_price",
                threshold_pct=0.02,
            )],
        )
        # Price dropped 3% — should trigger
        events = self.engine.evaluate_plan(
            plan=plan,
            current_price=97.0,
            current_pnl_pct=-0.03,
            current_atr=1.0,
            bars_since_review=6,
            market_regime=RiskMode.GREEN,
            asset_tradable=True,
            market=Market.US,
        )
        assert len(events) == 1
        assert events[0].trigger_type == TriggerType.PRICE_MOVE_PCT
        assert events[0].actual_value == pytest.approx(0.03, abs=0.001)

    def test_price_move_not_triggered(self):
        plan = ActivePlan(
            plan_id="p1",
            symbol="AAPL.US",
            last_review_price=100.0,
            triggers=[PlanTrigger(
                trigger_type=TriggerType.PRICE_MOVE_PCT,
                direction="down",
                anchor="last_review_price",
                threshold_pct=0.02,
            )],
        )
        # Price dropped 1% — should NOT trigger
        events = self.engine.evaluate_plan(
            plan=plan,
            current_price=99.0,
            current_pnl_pct=-0.01,
            current_atr=1.0,
            bars_since_review=2,
            market_regime=RiskMode.GREEN,
            asset_tradable=True,
            market=Market.US,
        )
        assert len(events) == 0

    def test_pnl_pct_triggered(self):
        plan = ActivePlan(
            plan_id="p1",
            symbol="AAPL.US",
            last_review_price=100.0,
            triggers=[PlanTrigger(
                trigger_type=TriggerType.PNL_PCT,
                operator="<=",
                threshold_pct=-0.03,
            )],
        )
        events = self.engine.evaluate_plan(
            plan=plan,
            current_price=97.0,
            current_pnl_pct=-0.03,
            current_atr=1.0,
            bars_since_review=6,
            market_regime=RiskMode.GREEN,
            asset_tradable=True,
            market=Market.US,
        )
        assert len(events) == 1
        assert events[0].trigger_type == TriggerType.PNL_PCT

    def test_pnl_pct_cooldown_suppresses_trigger(self):
        plan = ActivePlan(
            plan_id="p1",
            symbol="AAPL.US",
            last_review_price=100.0,
            triggers=[PlanTrigger(
                trigger_type=TriggerType.PNL_PCT,
                operator="<=",
                threshold_pct=-0.03,
            )],
        )
        # Within cooldown (6 bars) — should NOT trigger
        events = self.engine.evaluate_plan(
            plan=plan,
            current_price=97.0,
            current_pnl_pct=-0.03,
            current_atr=1.0,
            bars_since_review=3,
            market_regime=RiskMode.GREEN,
            asset_tradable=True,
            market=Market.US,
        )
        assert len(events) == 0

    def test_trailing_drawdown_triggered(self):
        plan = ActivePlan(
            plan_id="p1",
            symbol="AAPL.US",
            last_review_price=100.0,
            peak_since_entry=105.0,
            triggers=[PlanTrigger(
                trigger_type=TriggerType.TRAILING_DRAWDOWN_PCT,
                anchor="peak_since_entry",
                threshold_pct=0.02,
            )],
        )
        # Price at 102, peak 105, drawdown = 2.86% — should trigger
        events = self.engine.evaluate_plan(
            plan=plan,
            current_price=102.0,
            current_pnl_pct=-0.02,
            current_atr=1.0,
            bars_since_review=6,
            market_regime=RiskMode.GREEN,
            asset_tradable=True,
            market=Market.US,
        )
        assert len(events) == 1
        assert events[0].trigger_type == TriggerType.TRAILING_DRAWDOWN_PCT

    def test_bars_elapsed_triggered(self):
        plan = ActivePlan(
            plan_id="p1",
            symbol="AAPL.US",
            last_review_price=100.0,
            triggers=[PlanTrigger(
                trigger_type=TriggerType.BARS_ELAPSED,
                since="last_review",
                bars=6,
            )],
        )
        events = self.engine.evaluate_plan(
            plan=plan,
            current_price=100.0,
            current_pnl_pct=0.0,
            current_atr=1.0,
            bars_since_review=7,
            market_regime=RiskMode.GREEN,
            asset_tradable=True,
            market=Market.US,
        )
        assert len(events) == 1
        assert events[0].trigger_type == TriggerType.BARS_ELAPSED

    def test_regime_change_red(self):
        plan = ActivePlan(
            plan_id="p1",
            symbol="AAPL.US",
            last_review_price=100.0,
            triggers=[PlanTrigger(trigger_type=TriggerType.REGIME_CHANGE)],
        )
        events = self.engine.evaluate_plan(
            plan=plan,
            current_price=100.0,
            current_pnl_pct=0.0,
            current_atr=1.0,
            bars_since_review=1,
            market_regime=RiskMode.RED,
            asset_tradable=True,
            market=Market.US,
        )
        assert len(events) == 1
        assert events[0].priority == "P1"

    def test_asset_status_change(self):
        plan = ActivePlan(
            plan_id="p1",
            symbol="AAPL.US",
            last_review_price=100.0,
            triggers=[PlanTrigger(trigger_type=TriggerType.ASSET_STATUS_CHANGE)],
        )
        events = self.engine.evaluate_plan(
            plan=plan,
            current_price=100.0,
            current_pnl_pct=0.0,
            current_atr=1.0,
            bars_since_review=1,
            market_regime=RiskMode.GREEN,
            asset_tradable=False,
            market=Market.US,
        )
        assert len(events) == 1

    def test_crypto_wider_thresholds(self):
        engine = TriggerEngine()
        plan = ActivePlan(
            plan_id="p1",
            symbol="BTC",
            last_review_price=50000.0,
            triggers=[PlanTrigger(
                trigger_type=TriggerType.PRICE_MOVE_PCT,
                direction="down",
                anchor="last_review_price",
                threshold_pct=0.05,  # crypto default
            )],
        )
        # 4% drop — should NOT trigger for crypto
        events = engine.evaluate_plan(
            plan=plan,
            current_price=48000.0,
            current_pnl_pct=-0.04,
            current_atr=2.0,
            bars_since_review=2,
            market_regime=RiskMode.GREEN,
            asset_tradable=True,
            market=Market.CRYPTO,
        )
        assert len(events) == 0

        # 6% drop — should trigger
        events = engine.evaluate_plan(
            plan=plan,
            current_price=47000.0,
            current_pnl_pct=-0.06,
            current_atr=2.0,
            bars_since_review=6,
            market_regime=RiskMode.GREEN,
            asset_tradable=True,
            market=Market.CRYPTO,
        )
        assert len(events) == 1

    def test_trailing_drawdown_negative_threshold_clamped(self):
        """Model set -0.03 (negative) — should be clamped to abs(0.03) = 3%."""
        plan = ActivePlan(
            plan_id="p1",
            symbol="AAPL.US",
            last_review_price=100.0,
            peak_since_entry=105.0,
            triggers=[PlanTrigger(
                trigger_type=TriggerType.TRAILING_DRAWDOWN_PCT,
                threshold_pct=-0.03,  # negative — wrong!
                anchor="peak_since_entry",
            )],
        )
        engine = TriggerEngine()
        # Drawdown 0.95% (105→104) — should NOT trigger (threshold clamped to 3%)
        events = engine.evaluate_plan(
            plan=plan,
            current_price=104.0,
            current_pnl_pct=-0.01,
            current_atr=1.0,
            bars_since_review=6,
            market_regime=RiskMode.GREEN,
            asset_tradable=True,
            market=Market.US,
        )
        assert len(events) == 0

        # Drawdown 4.76% (105→100) — should trigger
        events = engine.evaluate_plan(
            plan=plan,
            current_price=100.0,
            current_pnl_pct=-0.05,
            current_atr=1.0,
            bars_since_review=6,
            market_regime=RiskMode.GREEN,
            asset_tradable=True,
            market=Market.US,
        )
        assert len(events) == 1
        assert events[0].trigger_type == TriggerType.TRAILING_DRAWDOWN_PCT

    def test_make_default_triggers(self):
        triggers = TriggerEngine.make_default_triggers(
            entry_price=100.0,
            atr_at_entry=1.5,
            config=TriggerConfig(),
        )
        assert len(triggers) == 3
        types = {t.trigger_type for t in triggers}
        assert TriggerType.PNL_PCT in types
        assert TriggerType.TRAILING_DRAWDOWN_PCT in types
        assert TriggerType.BARS_ELAPSED in types


# ============================================================
# Test: MemoryManager
# ============================================================

class TestMemoryManager:
    def setup_method(self):
        self.mm = MemoryManager()

    def test_thesis_update(self):
        thesis = self.mm.update_thesis("HK bullish", 0.7, "2026-01-07 10:00")
        assert thesis.text == "HK bullish"
        assert thesis.version == 1

        # Update again — version increments
        thesis2 = self.mm.update_thesis("CN weak", 0.5, "2026-01-07 11:00")
        assert thesis2.version == 2
        assert self.mm.get_thesis().text == "CN weak"

    def test_plan_create_and_close(self):
        plan = self.mm.create_plan("AAPL.US", 150.0, "trend", "2026-01-07 10:00")
        assert plan.symbol == "AAPL.US"
        assert plan.status == "active"

        retrieved = self.mm.get_plan("AAPL.US")
        assert retrieved is not None

        self.mm.close_plan("AAPL.US", "2026-01-07 11:00")
        assert self.mm.get_plan("AAPL.US") is None

    def test_plan_update(self):
        self.mm.create_plan("AAPL.US", 150.0, "trend", "2026-01-07 10:00")
        updated = self.mm.update_plan(
            "AAPL.US", PlanAction.UPDATE, note="holding well",
            current_price=155.0, timestamp="2026-01-07 11:00",
        )
        assert updated.plan_version == 2
        assert updated.plan_note == "holding well"

    def test_watchlist_add_remove(self):
        self.mm.add_watch("AAPL.US", "good trend", timestamp="2026-01-07 10:00")
        assert len(self.mm.get_watchlist()) == 1

        self.mm.remove_watch("AAPL.US")
        assert len(self.mm.get_watchlist()) == 0

    def test_watchlist_expire(self):
        self.mm.add_watch("AAPL.US", "good", expires_bars=1, timestamp="2026-01-07 10:00")
        # After 5 minutes (1 bar)
        expired = self.mm.expire_watchlist("2026-01-07 10:05")
        assert len(expired) == 1
        assert len(self.mm.get_watchlist()) == 0

    def test_avoid_list(self):
        self.mm.add_avoid("AAPL.US", "weak exit", timestamp="2026-01-07 10:00")
        assert len(self.mm.get_avoid_list()) == 1

        self.mm.remove_avoid("AAPL.US")
        assert len(self.mm.get_avoid_list()) == 0

    def test_recent_activity(self):
        self.mm.record_decision(DecisionType.FULL_DECISION, "bought AAPL", "2026-01-07 10:00")
        self.mm.record_decision(DecisionType.FULL_DECISION, "sold TSLA", "2026-01-07 10:30")
        self.mm.record_decision(DecisionType.FOCUSED_POSITION, "stop review AAPL", "2026-01-07 10:15")

        activity = self.mm.get_recent_activity()
        assert len(activity.non_hold_decisions) == 2
        assert len(activity.focused_decisions) == 1

    def test_memory_state(self):
        self.mm.update_thesis("bullish", 0.7, "2026-01-07 10:00")
        self.mm.add_watch("AAPL.US", "trend", timestamp="2026-01-07 10:00")
        self.mm.add_avoid("TSLA.US", "weak", timestamp="2026-01-07 10:00")

        state = self.mm.get_memory_state()
        assert state.daily_thesis is not None
        assert len(state.watchlist) == 1
        assert len(state.avoid_list) == 1

    def test_session_summary(self):
        summary = SessionSummary(
            market="HK",
            session_date="2026-01-07",
            market_read="HK opened strong",
        )
        self.mm.save_session_summary(summary)
        retrieved = self.mm.get_session_summary("HK")
        assert retrieved.market_read == "HK opened strong"

    def test_daily_summary(self):
        summary = DailySummary(
            date="2026-01-07",
            nav_start=1000000,
            nav_end=1003200,
            daily_return_pct=0.0032,
        )
        self.mm.save_daily_summary(summary)
        prev = self.mm.get_previous_daily_summary()
        assert prev.daily_return_pct == 0.0032


# ============================================================
# Test: CandidateBuckets
# ============================================================

class TestCandidateBuckets:
    def test_candidate_in_bucket(self):
        c = CandidateInBucket(
            bucket=CandidateBucket.TREND_LEADERS,
            ticker="AAPL.US",
            market=Market.US,
            price=150.0,
            score=0.85,
        )
        assert c.bucket == CandidateBucket.TREND_LEADERS
        assert c.tradable is True

    def test_candidate_buckets_structure(self):
        buckets = CandidateBuckets(
            held_positions=[],
            exit_watch=[],
            trend_leaders=[],
            pullback_continuation=[],
            oversold_reversal=[],
            low_vol_defensive=[],
            crypto_candidates=[],
            blocked_or_warning=[],
        )
        assert len(buckets.held_positions) == 0
        assert len(buckets.trend_leaders) == 0


# ============================================================
# Test: PortfolioTarget conversion
# ============================================================

class TestPortfolioTarget:
    def test_target_dataclass(self):
        target = PortfolioTarget(
            symbol="AAPL.US",
            target_pct_nav=0.05,
            reason="trend continuation",
        )
        assert target.symbol == "AAPL.US"
        assert target.target_pct_nav == 0.05
        assert target.priority == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
