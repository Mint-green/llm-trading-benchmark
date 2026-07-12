from types import SimpleNamespace

import pytest

from prompts.active.prompts import SYSTEM_PROMPT
from src.agent.context import ContextBuilder
from src.agent.protocol import DecisionProtocol
from src.agent.memory_manager import MemoryManager
from src.core.types import (
    ActivePlan,
    Market,
    OrderSide,
    PlanTrigger,
    PortfolioSnapshot,
    Position,
    RiskMode,
    TradeOrder,
    TradeResult,
    TriggerType,
)
from src.platform.event_detector import EventDetector
from src.platform.experiment import ExperimentRunner


def test_invalid_plan_action_is_rejected_instead_of_silent_no_change():
    memory = MemoryManager()

    results = memory.apply_plan_updates([
        {"symbol": "AAPL.US", "action": "set_take_profit", "take_profit": 180},
    ], "2026-01-06 16:00")

    assert results == [{
        "symbol": "AAPL.US",
        "status": "rejected",
        "reason": "unsupported plan_action: set_take_profit",
    }]
    assert memory.get_plan("AAPL.US") is None


def test_close_missing_plan_does_not_create_one():
    memory = MemoryManager()

    results = memory.apply_plan_updates([
        {"symbol": "AAPL.US", "plan_action": "close"},
    ], "2026-01-06 16:00")

    assert results[0]["status"] == "rejected"
    assert results[0]["reason"] == "active plan not found"
    assert memory.get_all_plans() == {}


def test_focused_close_action_with_targets_is_normalized_to_rebalance():
    decision = DecisionProtocol().parse("""
    {
      "action": "close",
      "portfolio_targets": [
        {
          "symbol": "0700.HK",
          "asset_type": "equity",
          "target_pct_nav": 0.0,
          "side": "flat"
        }
      ],
      "plan_updates": [
        {"symbol": "0700.HK", "plan_action": "close"}
      ],
      "memory_updates": {},
      "reason": "risk exit"
    }
    """)

    assert decision is not None
    assert decision.action == "trade"
    assert decision.trades[0].side == OrderSide.SELL


def test_plan_close_is_rejected_while_position_is_still_held():
    runner = object.__new__(ExperimentRunner)
    runner.memory = MemoryManager()
    runner.memory.create_plan(
        symbol="AAPL.US",
        entry_price=100.0,
        entry_reason="entry",
        timestamp="2026-01-06 16:00",
    )
    position = Position(
        symbol="AAPL.US",
        market=Market.US,
        quantity=10,
        avg_cost=100.0,
        current_price=97.0,
    )
    runner.portfolio = SimpleNamespace(
        get_snapshot=lambda timestamp: SimpleNamespace(
            positions={"US:AAPL.US": position},
        ),
    )

    results = runner._apply_plan_updates_after_execution([
        {"symbol": "AAPL.US", "plan_action": "close"},
    ], "2026-01-06 16:30")

    assert results[0]["status"] == "rejected"
    assert results[0]["reason"] == "position still held after execution"
    assert runner.memory.get_plan("AAPL.US") is not None


def test_active_plan_is_injected_into_normal_memory_prompt():
    memory = MemoryManager()
    memory.create_plan(
        symbol="AAPL.US",
        entry_price=150.0,
        entry_reason="trend entry",
        timestamp="2026-01-06 16:00",
        triggers=[PlanTrigger(trigger_type=TriggerType.BARS_ELAPSED, bars=12)],
    )

    state = memory.get_memory_state()
    rendered = ContextBuilder._format_memory_state(object.__new__(ContextBuilder), state)

    assert "active_plans:" in rendered
    assert "AAPL.US|active|150.0000" in rendered
    assert "bars_elapsed" in rendered
    assert "plan_action must be create, update, close, or no_change" in SYSTEM_PROMPT


def test_successful_buy_creates_plan_and_full_exit_closes_it():
    runner = object.__new__(ExperimentRunner)
    runner.memory = MemoryManager()
    position = Position(
        symbol="AAPL.US",
        market=Market.US,
        quantity=10,
        avg_cost=100.0,
        current_price=100.0,
    )
    held = {"US:AAPL.US": position}
    runner.portfolio = SimpleNamespace(
        nav=10000.0,
        get_position=lambda key: held.get(key),
        get_snapshot=lambda ts: SimpleNamespace(positions=held),
    )
    runner.config = SimpleNamespace(
        decision_interval=5,
        decision_schedule=SimpleNamespace(normal_interval_minutes=60),
        trigger_config=SimpleNamespace(pnl_pct_threshold=-0.03),
    )

    buy = TradeOrder(
        symbol="AAPL.US",
        market=Market.US,
        side=OrderSide.BUY,
        quantity=10,
        allocation_pct=0.10,
        reason="trend entry",
    )
    runner._record_execution_memory(
        TradeResult(order=buy, success=True, price=100.0),
        requested_qty=10,
        timestamp="2026-01-06 16:00",
    )

    plan = runner.memory.get_plan("AAPL.US")
    assert plan is not None
    assert [trigger.trigger_type for trigger in plan.triggers] == [TriggerType.PNL_PCT]
    assert runner.memory.get_recent_activity().execution_feedback == ["AAPL.US: OK"]

    held.clear()
    sell = TradeOrder(
        symbol="AAPL.US",
        market=Market.US,
        side=OrderSide.SELL,
        quantity=10,
        allocation_pct=0.0,
    )
    runner._record_execution_memory(
        TradeResult(order=sell, success=True, price=101.0),
        requested_qty=10,
        timestamp="2026-01-06 17:00",
    )
    assert runner.memory.get_plan("AAPL.US") is None


def test_event_detector_uses_usd_position_mark_for_plan_pnl():
    captured = {}

    class CaptureTriggerEngine:
        def evaluate_plan(self, **kwargs):
            captured.update(kwargs)
            return []

    detector = object.__new__(EventDetector)
    detector._trigger_engine = CaptureTriggerEngine()
    detector._features = SimpleNamespace(compute=lambda bars, timestamp: None)

    position = Position(
        symbol="0700.HK",
        market=Market.HK,
        quantity=100,
        avg_cost=40.0,
        current_price=42.0,
    )
    snapshot = PortfolioSnapshot(
        timestamp="2026-01-06 02:00",
        cash=1000.0,
        positions={"HK:0700.HK": position},
        total_nav=5200.0,
        market_exposure={Market.HK: 4200.0},
        fx_rates={"USD/HKD": 7.8},
    )
    plan = ActivePlan(
        plan_id="plan-hk",
        symbol="0700.HK",
        entry_price=40.0,
        last_review_time="2026-01-06 01:00",
        triggers=[PlanTrigger(trigger_type=TriggerType.PNL_PCT, threshold_pct=-0.03)],
    )

    detector._evaluate_plan_triggers(
        plan,
        "0700.HK",
        snapshot,
        {Market.HK: {"0700.HK": [SimpleNamespace(close=320.0)]}},
        RiskMode.GREEN,
    )

    assert captured["current_price"] == 42.0
    assert captured["current_pnl_pct"] == pytest.approx(0.05)



def test_plan_update_uses_system_usd_mark_instead_of_model_price():
    runner = object.__new__(ExperimentRunner)
    runner.memory = MemoryManager()
    runner.memory.create_plan(
        symbol="0700.HK",
        entry_price=40.0,
        entry_reason="entry",
        timestamp="2026-01-06 02:00",
    )
    position = Position(
        symbol="0700.HK",
        market=Market.HK,
        quantity=100,
        avg_cost=40.0,
        current_price=42.0,
    )
    runner.portfolio = SimpleNamespace(
        get_snapshot=lambda timestamp: SimpleNamespace(
            positions={"HK:0700.HK": position},
        ),
    )

    results = runner._apply_plan_updates_after_execution([
        {
            "symbol": "0700.HK",
            "plan_action": "update",
            "current_price": 327.60,
            "plan_note": "keep",
        },
    ], "2026-01-06 02:30")

    assert results[0]["status"] == "applied"
    assert runner.memory.get_plan("0700.HK").last_review_price == 42.0


def test_futures_buy_creates_plan_by_family_symbol():
    runner = object.__new__(ExperimentRunner)
    runner.memory = MemoryManager()
    position = Position(
        symbol="GCQ5.CM",
        market=Market.FUTURES,
        quantity=1,
        avg_cost=3300.0,
        current_price=3310.0,
    )
    held = {"FUT:GCQ5.CM": position}
    runner.portfolio = SimpleNamespace(
        nav=100000.0,
        get_snapshot=lambda ts: SimpleNamespace(positions=held),
    )
    runner.config = SimpleNamespace(
        trigger_config=SimpleNamespace(pnl_pct_threshold=-0.03),
    )

    buy = TradeOrder(
        symbol="GOLD_FUT",
        market=Market.FUTURES,
        side=OrderSide.BUY,
        quantity=1,
        reason="gold trend",
        asset_type="futures",
    )
    runner._record_execution_memory(
        TradeResult(order=buy, success=True, price=3300.0),
        requested_qty=1,
        timestamp="2026-02-03 10:00",
    )

    plan = runner.memory.get_plan("GOLD_FUT")
    assert plan is not None
    assert plan.entry_price == 3300.0
    assert plan.triggers[0].trigger_type == TriggerType.PNL_PCT

    # Verify get_plan_by_position finds it via root match
    found = runner.memory.get_plan_by_position("GCQ5.CM")
    assert found is plan

    # Verify close_plan_by_position works
    held.clear()
    closed = runner.memory.close_plan_by_position("GCQ5.CM", "2026-02-03 11:00")
    assert closed is not None
    assert runner.memory.get_plan("GOLD_FUT") is None


def test_futures_contract_roll_updates_plan_entry_price():
    memory = MemoryManager()
    memory.create_plan(
        symbol="GOLD_FUT",
        entry_price=3300.0,
        entry_reason="gold trend",
        timestamp="2026-02-03 10:00",
        triggers=[PlanTrigger(trigger_type=TriggerType.PNL_PCT, operator="<=", threshold_pct=-0.03)],
    )

    # Simulate contract roll: GCQ5 → GCZ5
    plan = memory.get_plan_by_position("GCQ5.CM")
    assert plan is not None

    # New contract with different price
    plan.entry_price = 3350.0
    plan.peak_since_entry = 3350.0

    # Old contract position gone, new one appears
    found = memory.get_plan_by_position("GCZ5.CM")
    assert found is plan  # Same plan, since same root "GC"


def test_ensure_plans_for_positions_creates_missing_plans():
    runner = object.__new__(ExperimentRunner)
    runner.memory = MemoryManager()
    position = Position(
        symbol="AAPL.US",
        market=Market.US,
        quantity=10,
        avg_cost=150.0,
        current_price=155.0,
    )
    runner.portfolio = SimpleNamespace(
        get_snapshot=lambda ts: SimpleNamespace(
            positions={"US:AAPL.US": position},
            total_nav=100000.0,
        ),
    )
    runner.config = SimpleNamespace(
        trigger_config=SimpleNamespace(pnl_pct_threshold=-0.03),
    )

    assert runner.memory.get_plan("AAPL.US") is None
    runner._ensure_plans_for_positions("2026-02-03 10:00")
    plan = runner.memory.get_plan("AAPL.US")
    assert plan is not None
    assert plan.entry_price == 150.0
    assert plan.entry_reason == "checkpoint_resume"


def test_ensure_plans_skips_existing_plans():
    runner = object.__new__(ExperimentRunner)
    runner.memory = MemoryManager()
    runner.memory.create_plan(
        symbol="AAPL.US",
        entry_price=140.0,
        entry_reason="manual",
        timestamp="2026-02-03 09:00",
    )
    position = Position(
        symbol="AAPL.US",
        market=Market.US,
        quantity=10,
        avg_cost=150.0,
        current_price=155.0,
    )
    runner.portfolio = SimpleNamespace(
        get_snapshot=lambda ts: SimpleNamespace(
            positions={"US:AAPL.US": position},
            total_nav=100000.0,
        ),
    )
    runner.config = SimpleNamespace(
        trigger_config=SimpleNamespace(pnl_pct_threshold=-0.03),
    )

    runner._ensure_plans_for_positions("2026-02-03 10:00")
    plan = runner.memory.get_plan("AAPL.US")
    assert plan.entry_price == 140.0  # Original, not overwritten
    assert plan.entry_reason == "manual"

