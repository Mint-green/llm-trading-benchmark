from pathlib import Path
from src.agent.context import ContextBuilder
from src.agent.runner import AgentRunner
from src.platform.experiment import ExperimentRunner
from src.platform.scheduler import DecisionScheduler
from src.portfolio.constraints import ConstraintEngine
from src.portfolio.portfolio import PortfolioEngine
from src.portfolio.nav import NavEngine
from src.agent.memory_manager import MemoryManager
from src.data.screener import Screener
from src.core.config import Config, DecisionScheduleConfig, OpenWindowConfig, CloseWindowConfig
from src.core.types import Decision, DecisionType, DailySummary, Market, OrderSide, PortfolioSnapshot, Position, TradeOrder, TradeResult, TriggerType


class _Loader:
    def load_instruction_template(self):
        return "Round {round_num}/{max_rounds}. Decide."

    def load_final_round_instruction(self):
        return "FINAL ROUND ({max_rounds}/{max_rounds}). Decide now."


class _Context:
    _loader = _Loader()

    def build(self, *args, **kwargs):
        return [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "base context"},
        ]


class _Tools:
    def execute_tool(self, name, args, timestamp):
        if name == "screen_universe":
            return "SCREEN_RESULT WDC.US strong_continuation recent_score=+2"
        if name == "query_market_overview":
            return "OVERVIEW_RESULT US breadth strong"
        return "UNKNOWN"


class _CapturingRunner(AgentRunner):
    def __init__(self):
        self._config = Config(max_agent_rounds=3)
        self._model_name = "mimo-v2.5-pro"
        self._context = _Context()
        self._tools = _Tools()
        self._price_lookup = None
        self._portfolio = None
        self.final_messages = []
        self.tool_round = 0

    def _call_llm_with_tools(self, messages):
        self.tool_round += 1
        if self.tool_round == 1:
            return "", [{"id": "call_1", "function": {"name": "screen_universe", "arguments": "{}"}}], 0, 0, ""
        return "", [{"id": "call_2", "function": {"name": "query_market_overview", "arguments": "{}"}}], 0, 0, ""

    def _call_llm(self, messages):
        self.final_messages = messages
        combined = "\n".join(m["content"] for m in messages)
        assert "SCREEN_RESULT" in combined
        assert "OVERVIEW_RESULT" in combined
        return '{"action":"hold","reason":"history retained"}', 0, 0, ""


def test_market_timing_marks_last_chance_and_tail_guard():
    at_2030 = ContextBuilder._format_market_timing("2026-01-06 20:30", ["US", "CRYPTO"])
    assert "US|open|30|15|last_chance_buy_now_or_skip" in at_2030

    at_2045 = ContextBuilder._format_market_timing("2026-01-06 20:45", ["US", "CRYPTO"])
    assert "US|open|15|0|tail_guard_active_no_new_buys" in at_2045


def test_agent_runner_retains_all_tool_results_for_final_round():
    runner = _CapturingRunner()
    decision, rounds = runner.run(
        timestamp="2026-01-06 20:30",
        snapshot=object(),
        market_data="",
        stock_data="",
        alerts="",
        news="",
        pre_built_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "base v3 context"},
        ],
    )

    assert isinstance(decision, Decision)
    assert decision.action == "hold"
    assert decision.reason == "history retained"
    assert len(rounds) == 3

def test_trade_feedback_guard_filters_same_market_replacement_buys():
    decision = Decision(
        action="trade",
        reason="rebalance after stop",
        trades=[
            TradeOrder("3993.HK", Market.HK, OrderSide.BUY, allocation_pct=0.04),
            TradeOrder("ISRG.US", Market.US, OrderSide.BUY, allocation_pct=0.04),
            TradeOrder("9988.HK", Market.HK, OrderSide.SELL, quantity=100),
        ],
    )

    filtered = AgentRunner._apply_trade_feedback_guards(
        decision,
        "AUTO SELL 9626.HK(HK): PnL=-3.6% hit stop-loss",
    )

    assert filtered.action == "trade"
    assert [trade.symbol for trade in filtered.trades] == ["ISRG.US", "9988.HK"]
    assert "filtered 1 same-market BUY" in filtered.reason


def test_trade_feedback_guard_turns_all_filtered_buys_into_hold():
    decision = Decision(
        action="trade",
        reason="replace stopped position",
        trades=[TradeOrder("3993.HK", Market.HK, OrderSide.BUY, allocation_pct=0.04)],
    )

    filtered = AgentRunner._apply_trade_feedback_guards(
        decision,
        "AUTO SELL 9626.HK(HK): PnL=-3.6% hit stop-loss",
    )

    assert filtered.action == "hold"
    assert filtered.trades == []
    assert "filtered 1 same-market BUY" in filtered.reason

def test_light_decision_context_marks_24h_scope():
    context = ContextBuilder._format_decision_context(
        ContextBuilder.__new__(ContextBuilder),
        "2026-01-10 04:00",
        "light_decision",
        ["CRYPTO"],
        ["US", "HK", "CN"],
        0,
        48,
    )

    assert "decision_type: light_decision" in context
    assert "scope: 24h_assets_only" in context
    assert "allowed_trade_markets: ['CRYPTO', 'GOLD', 'FUTURES']" in context


def test_market_overview_denied_for_full_and_light_decisions():
    runner = _CapturingRunner()
    calls = [{"function": {"name": "query_market_overview", "arguments": "{}"}}]
    messages = [{"role": "user", "content": "[DECISION_CONTEXT]\ndecision_type: full_decision"}]

    result, records = runner._execute_tool_calls(calls, "2026-01-06 15:00", messages)

    assert "query_market_overview denied" in result
    assert records[0]["name"] == "query_market_overview"
    assert "denied" in records[0]["result"]


def test_light_decision_boundary_respects_position_frequency():
    assert ExperimentRunner._is_light_decision_boundary("2026-01-10 01:00", 60)
    assert ExperimentRunner._is_light_decision_boundary("2026-01-10 23:00", 60)
    assert not ExperimentRunner._is_light_decision_boundary("2026-01-10 01:05", 60)

    assert ExperimentRunner._is_light_decision_boundary("2026-01-10 04:00", 240)
    assert not ExperimentRunner._is_light_decision_boundary("2026-01-10 02:00", 240)

def test_cooling_blocked_sell_is_filtered_before_execution():
    runner = object.__new__(AgentRunner)
    constraints = ConstraintEngine(Config())
    constraints.record_buy("US:SHOP.US", "2026-01-06 19:00")
    runner._portfolio = type("PortfolioRef", (), {"_constraints": constraints})()

    snapshot = PortfolioSnapshot(
        timestamp="2026-01-06 20:30",
        cash=95_000.0,
        positions={
            "US:SHOP.US": Position(
                symbol="SHOP.US",
                market=Market.US,
                quantity=24,
                avg_cost=100.0,
                current_price=101.0,
            ),
        },
        total_nav=100_000.0,
        market_exposure={Market.US: 2_424.0},
        fx_rates={"USD": 1.0},
    )

    trades, reason = runner._filter_cooling_blocked_sells(
        [TradeOrder("SHOP.US", Market.US, OrderSide.SELL, quantity=24)],
        snapshot,
        "2026-01-06 20:30",
    )

    assert trades == []
    assert "cooling-blocked SELL" in reason
    assert "SHOP.US(US)" in reason

def test_zero_lot_buy_is_filtered_before_execution():
    runner = object.__new__(AgentRunner)
    execution = type(
        "ExecutionRef",
        (),
        {"_round_lots": staticmethod(lambda market, symbol, quantity, side: 0 if market == Market.CN else quantity)},
    )()
    runner._portfolio = type("PortfolioRef", (), {"_execution": execution})()

    trades, reason = runner._filter_zero_lot_buys([
        TradeOrder("sh.688256", Market.CN, OrderSide.BUY, quantity=20),
        TradeOrder("AAPL.US", Market.US, OrderSide.BUY, quantity=5),
        TradeOrder("SHOP.US", Market.US, OrderSide.SELL, quantity=3),
    ])

    assert [trade.symbol for trade in trades] == ["AAPL.US", "SHOP.US"]
    assert "zero-lot BUY" in reason
    assert "sh.688256(CN, requested 20)" in reason


def test_daily_buy_remaining_is_timestamp_scoped():
    constraints = ConstraintEngine(Config(), max_daily_trades=2)
    constraints.record_trade("2026-01-06 15:00")
    constraints.record_trade("2026-01-06 16:00")

    assert constraints.daily_buys_remaining_at("2026-01-06 20:00") == 0
    assert constraints.daily_buys_remaining_at("2026-01-07 01:30") == 2


def test_auto_stop_loss_sell_skips_cooling_rejection():
    runner = object.__new__(ExperimentRunner)
    constraints = ConstraintEngine(Config())
    constraints.record_buy("US:MSTR.US", "2026-01-09 17:00")
    runner.portfolio = type("PortfolioRef", (), {"_constraints": constraints})()
    snapshot = PortfolioSnapshot(
        timestamp="2026-01-09 18:15",
        cash=95_000.0,
        positions={
            "US:MSTR.US": Position(
                symbol="MSTR.US",
                market=Market.US,
                quantity=24,
                avg_cost=100.0,
                current_price=96.0,
            ),
        },
        total_nav=100_000.0,
        market_exposure={Market.US: 2_304.0},
        fx_rates={"USD": 1.0},
    )

    assert runner._is_auto_sell_cooling_blocked("US:MSTR.US", 24, snapshot, "2026-01-09 18:15")
    assert not runner._is_auto_sell_cooling_blocked("US:MSTR.US", 24, snapshot, "2026-01-09 19:00")


def test_post_stop_loss_market_cooldown_filters_same_market_buys():
    runner = object.__new__(ExperimentRunner)
    runner._stop_loss_buy_pause_until = {Market.HK: "2026-01-07 02:30"}
    decision = Decision(
        action="trade",
        reason="replace after stop",
        trades=[
            TradeOrder("0291.HK", Market.HK, OrderSide.BUY, allocation_pct=0.03),
            TradeOrder("AAPL.US", Market.US, OrderSide.BUY, allocation_pct=0.03),
        ],
    )

    filtered = runner._filter_stop_loss_cooldown_buys(decision, "2026-01-07 02:00")

    assert filtered.action == "trade"
    assert [trade.symbol for trade in filtered.trades] == ["AAPL.US"]
    assert "post-stop-loss BUY" in filtered.reason
    assert "0291.HK(HK, until 2026-01-07 02:30)" in filtered.reason


def test_light_decision_blocks_new_crypto_buy_without_existing_position():
    decision = Decision(
        action="trade",
        reason="new 24h opportunity",
        trades=[TradeOrder("ETC-USD.CC", Market.CRYPTO, OrderSide.BUY, allocation_pct=0.03)],
    )

    filtered = ExperimentRunner._restrict_light_decision_trades(
        decision, allow_new_crypto_buys=False,
    )

    assert filtered.action == "hold"
    assert filtered.trades == []
    assert "new crypto BUY" in filtered.reason


def test_light_decision_allows_crypto_sell_and_existing_position_buy():
    decision = Decision(
        action="trade",
        reason="manage crypto",
        trades=[
            TradeOrder("BTC-USD.CC", Market.CRYPTO, OrderSide.SELL, quantity=1),
            TradeOrder("ETC-USD.CC", Market.CRYPTO, OrderSide.BUY, allocation_pct=0.03),
            TradeOrder("AAPL.US", Market.US, OrderSide.BUY, allocation_pct=0.03),
        ],
    )

    filtered = ExperimentRunner._restrict_light_decision_trades(
        decision, allow_new_crypto_buys=True,
    )

    assert filtered.action == "trade"
    assert [t.symbol for t in filtered.trades] == ["BTC-USD.CC", "ETC-USD.CC"]
    assert "out-of-scope" in filtered.reason



def test_default_full_decision_schedule_is_30min():
    scheduler = DecisionScheduler(Config())

    first = scheduler.schedule(
        "2026-01-06 16:00",
        open_markets=[Market.US, Market.CRYPTO],
        closed_markets=[Market.HK, Market.CN],
    )
    middle = scheduler.schedule(
        "2026-01-06 16:15",
        open_markets=[Market.US, Market.CRYPTO],
        closed_markets=[Market.HK, Market.CN],
    )
    second = scheduler.schedule(
        "2026-01-06 16:30",
        open_markets=[Market.US, Market.CRYPTO],
        closed_markets=[Market.HK, Market.CN],
    )

    assert first.decision_type == DecisionType.FULL_DECISION
    assert middle.decision_type == DecisionType.AUTO_HOLD
    assert second.decision_type == DecisionType.FULL_DECISION


def test_full_decision_schedule_can_remain_60min_when_configured():
    scheduler = DecisionScheduler(Config(
        decision_schedule=DecisionScheduleConfig(normal_interval_minutes=60),
    ))

    first = scheduler.schedule(
        "2026-01-06 16:00",
        open_markets=[Market.US, Market.CRYPTO],
        closed_markets=[Market.HK, Market.CN],
    )
    second = scheduler.schedule(
        "2026-01-06 16:30",
        open_markets=[Market.US, Market.CRYPTO],
        closed_markets=[Market.HK, Market.CN],
    )

    assert first.decision_type == DecisionType.FULL_DECISION
    assert second.decision_type == DecisionType.AUTO_HOLD


def test_config_loads_decision_schedule_from_toml():
    config_path = Path("tmp_runs/test_config_schedule.toml")
    config_path.parent.mkdir(exist_ok=True)
    config_path.write_text(
        '''
[data]
base_dir = "D:/Projects/claw/getStockData"

[decision_schedule]
normal_interval_minutes = 60

[decision_schedule.open_window]
enabled = true
minutes_after_open = 20
interval_minutes = 10
include_open_plus_30 = false

[decision_schedule.close_window]
enabled = false
minutes_before_close = 45
interval_minutes = 15
include_close_time = true
''',
        encoding="utf-8",
    )

    config = Config.load_from_toml(str(config_path))

    assert config.decision_schedule.normal_interval_minutes == 60
    assert config.decision_schedule.open_window.minutes_after_open == 20
    assert config.decision_schedule.open_window.interval_minutes == 10
    assert config.decision_schedule.open_window.include_open_plus_30 is False
    assert config.decision_schedule.close_window.enabled is False
    assert config.decision_schedule.close_window.minutes_before_close == 45
    assert config.decision_schedule.close_window.include_close_time is True


def test_plan_updates_accept_structured_and_legacy_formats():
    memory = MemoryManager()

    memory.apply_plan_updates([
        {
            "symbol": "AAPL.US",
            "plan_action": "create",
            "current_price": 150.0,
            "plan_note": "trend entry",
            "triggers": [
                {
                    "type": "price_move_pct",
                    "direction": "down",
                    "anchor": "last_review_price",
                    "threshold_pct": 0.02,
                }
            ],
        }
    ], "2026-01-06 16:00")
    plan = memory.get_plan("AAPL.US")

    assert plan is not None
    assert plan.entry_price == 150.0
    assert plan.plan_note == "trend entry"
    assert plan.triggers[0].trigger_type == TriggerType.PRICE_MOVE_PCT
    assert plan.triggers[0].threshold_pct == 0.02

    memory.apply_plan_updates([
        {
            "symbol": "AAPL.US",
            "action": "update",
            "stop_loss": 145.0,
            "take_profit": 165.0,
        }
    ], "2026-01-06 16:30")
    plan = memory.get_plan("AAPL.US")

    assert plan.plan_version == 2
    assert "legacy_stop_loss=145.0" in plan.plan_note
    assert "legacy_take_profit=165.0" in plan.plan_note


def test_daily_summary_injection_is_consumed_once():
    runner = object.__new__(ExperimentRunner)
    runner._pending_daily_summary_injection = True
    memory = MemoryManager()
    memory.save_daily_summary(DailySummary(
        date="2026-01-06",
        nav_start=100000.0,
        nav_end=100200.0,
        daily_return_pct=0.002,
        market_read="first day bullish",
    ))

    first_state = memory.get_memory_state(
        is_first_decision=runner._consume_daily_summary_injection(),
    )
    second_state = memory.get_memory_state(
        is_first_decision=runner._consume_daily_summary_injection(),
    )

    assert first_state.previous_daily_summary is not None
    assert first_state.previous_daily_summary.market_read == "first day bullish"
    assert second_state.previous_daily_summary is None


def test_daily_rollover_summary_skips_backtest_start_day():
    runner = object.__new__(ExperimentRunner)
    runner.config = Config(backtest_start="2026-01-06")
    decisions = [{"timestamp": "2026-01-06 00:00", "action": "hold"}]

    assert not runner._should_generate_daily_rollover_summary("2026-01-06 00:05", decisions)
    assert runner._should_generate_daily_rollover_summary("2026-01-07 00:05", decisions)
    assert not runner._should_generate_daily_rollover_summary("2026-01-07 00:10", decisions)
    assert not runner._should_generate_daily_rollover_summary("2026-01-07 00:05", [])


def test_trade_decision_preserves_memory_and_plan_updates_after_filters():
    runner = object.__new__(AgentRunner)
    runner._portfolio = None
    snapshot = PortfolioSnapshot(
        timestamp="2026-01-06 16:00",
        cash=100000.0,
        positions={},
        total_nav=100000.0,
        market_exposure={},
        fx_rates={"USD": 1.0},
    )
    decision = Decision(
        action="trade",
        trades=[TradeOrder("AAPL.US", Market.US, OrderSide.BUY, quantity=5)],
        reason="buy with plan",
        memory_updates={"daily_thesis": "risk-on"},
        plan_updates=[{"symbol": "AAPL.US", "plan_action": "create"}],
    )

    resolved = runner._finalize_trade_decision(decision, snapshot, "2026-01-06 16:00")

    assert resolved.action == "trade"
    assert resolved.memory_updates == {"daily_thesis": "risk-on"}
    assert resolved.plan_updates == [{"symbol": "AAPL.US", "plan_action": "create"}]


def test_trade_decision_preserves_updates_when_all_trades_filtered():
    runner = object.__new__(AgentRunner)
    execution = type(
        "ExecutionRef",
        (),
        {"_round_lots": staticmethod(lambda market, symbol, quantity, side: 0)},
    )()
    runner._portfolio = type("PortfolioRef", (), {"_execution": execution})()
    snapshot = PortfolioSnapshot(
        timestamp="2026-01-06 16:00",
        cash=100000.0,
        positions={},
        total_nav=100000.0,
        market_exposure={},
        fx_rates={"USD": 1.0},
    )
    decision = Decision(
        action="trade",
        trades=[TradeOrder("sh.688256", Market.CN, OrderSide.BUY, quantity=20)],
        reason="too small",
        memory_updates={"daily_thesis": "be selective"},
        plan_updates=[{"symbol": "sh.688256", "plan_action": "create"}],
    )

    resolved = runner._finalize_trade_decision(decision, snapshot, "2026-01-06 16:00")

    assert resolved.action == "hold"
    assert "zero-lot BUY" in resolved.reason
    assert resolved.memory_updates == {"daily_thesis": "be selective"}
    assert resolved.plan_updates == [{"symbol": "sh.688256", "plan_action": "create"}]


def test_screener_uses_pct_nav_for_position_buckets():
    buckets = Screener(features=object())._classify_buckets(
        [],
        held_positions={
            "AAPL.US": {
                "market": Market.US,
                "price": 100.0,
                "pnl_pct": -0.07,
                "pct_nav": 0.12,
            }
        },
        exit_watch_positions={
            "MSFT.US": {
                "market": Market.US,
                "price": 200.0,
                "pnl_pct": 0.05,
                "pct_nav": 0.08,
            }
        },
        open_market_set={Market.US, Market.HK, Market.CN, Market.CRYPTO},
    )

    assert buckets.held_positions[0].pct_nav == 0.12
    assert buckets.exit_watch[0].pct_nav == 0.08


class _AlwaysTradeRules:
    def can_trade(self, market, symbol, side, timestamp):
        return True, "ok"


class _AlwaysSellableSettlement:
    def get_sellable_quantity(self, key, timestamp):
        return 10

    def settle(self, result, timestamp):
        return None


class _PassthroughExecution:
    def execute(self, order, price, timestamp):
        return TradeResult(order=order, success=True, price=price, cost=price * order.quantity, fees=0.0)


def test_daily_buy_limit_does_not_block_sell_orders():
    config = Config(initial_cash=100000.0)
    constraints = ConstraintEngine(config, max_daily_trades=1)
    constraints.record_trade("2026-01-06 15:00")
    portfolio = PortfolioEngine(
        config,
        NavEngine(config.fx_rates),
        constraints,
        _PassthroughExecution(),
        _AlwaysSellableSettlement(),
        _AlwaysTradeRules(),
    )
    portfolio._positions["US:AAPL.US"] = Position(
        symbol="AAPL.US",
        market=Market.US,
        quantity=10,
        avg_cost=100.0,
        current_price=100.0,
    )

    result = portfolio.process_order(
        TradeOrder("AAPL.US", Market.US, OrderSide.SELL, quantity=5),
        101.0,
        "2026-01-06 16:00",
    )

    assert result.success is True
    assert result.error == ""


def test_model_configs_keep_five_minute_scanner_interval():
    assert 'decision_interval = 5' in Path('config/mimo.toml').read_text(encoding='utf-8')
    assert 'decision_interval = 5' in Path('config/deepseek.toml').read_text(encoding='utf-8')
