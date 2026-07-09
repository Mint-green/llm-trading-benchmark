import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.agent.tools import ToolSystem
from src.core.config import Config
from src.core.types import Market, OrderSide, TradeOrder
from src.data.features import FeatureGenerator
from src.data.futures_candidates import FuturesCandidateBuilder
from src.data.futures_resolver import FuturesContractResolver
from src.data.provider import MarketDataProvider
from src.portfolio.futures import FuturesAccount


import os
DATA_ROOT = Path(os.environ.get("STOCK_DATA_ROOT", os.path.expanduser("~/Desktop/getStockData")))
DATA_DB = DATA_ROOT / "data" / "FUTURES_stock.db"
pytestmark = pytest.mark.skipif(not DATA_DB.exists(), reason="FUTURES_stock.db is not available")


def _config(**futures_overrides):
    cfg = Config()
    if futures_overrides:
        cfg = replace(cfg, futures=replace(cfg.futures, **futures_overrides))
    return cfg


def test_resolver_maps_gc_to_one_actual_contract_without_future_liquidity():
    cfg = _config()
    data = MarketDataProvider(cfg)
    resolver = FuturesContractResolver(cfg, data)

    resolved = resolver.resolve("GC.FUT", "2026-02-03 15:00")

    assert resolved.continuous_symbol == "GC.FUT"
    assert resolved.contract_ticker.startswith("GC")
    assert resolved.price and resolved.price > 0
    assert resolved.notional_per_contract == pytest.approx(resolved.price * resolved.multiplier)
    assert resolved.previous_session_dollar_volume is not None
    assert resolved.selection_method in {
        "previous_session_liquidity_safe",
        "fallback_liquidity_all_candidates",
    }
    assert resolved.roll_status in {"normal", "near_roll_window", "forced_near_expiry"}
    data.close()


def test_query_futures_contract_tool_returns_margin_and_contract_mapping():
    cfg = _config()
    data = MarketDataProvider(cfg)
    tools = ToolSystem(data, FeatureGenerator(), lambda: None)

    output = tools.execute_tool(
        "query_futures_contract",
        {"continuous_symbol": "GC.FUT", "fields": ["actual_contract", "price", "notional", "initial_margin", "roll_status"]},
        "2026-02-03 15:00",
    )
    payload = json.loads(output)

    assert payload["continuous_symbol"] == "GC.FUT"
    assert payload["actual_contract"].startswith("GC")
    assert payload["price"] > 0
    assert payload["notional_per_contract"] > 0
    assert payload["initial_margin"] > 0
    data.close()


def test_futures_macro_candidate_uses_actual_contract_and_margin_fields():
    cfg = _config()
    data = MarketDataProvider(cfg)
    resolver = FuturesContractResolver(cfg, data)
    builder = FuturesCandidateBuilder(data, FeatureGenerator(), resolver)

    rows = builder.build("2026-02-03 15:00", nav=1_000_000, symbols=["GC.FUT"])

    assert len(rows) == 1
    row = rows[0]
    assert row.market == Market.FUTURES
    assert row.asset_type == "futures"
    assert row.actual_contract.startswith("GC")
    assert row.notional_per_contract > 0
    assert row.initial_margin > 0
    assert row.one_contract_notional_pct_nav > 0
    data.close()


def test_100k_account_rejects_standard_gc_contract_when_target_floors_to_zero():
    cfg = _config(max_risk_budget_pct_nav=0.50)
    data = MarketDataProvider(cfg)
    resolver = FuturesContractResolver(cfg, data)
    account = FuturesAccount(cfg, data, resolver, cash_usd=100_000)
    order = TradeOrder(
        symbol="GC.FUT",
        market=Market.FUTURES,
        side=OrderSide.BUY,
        asset_type="futures",
        action="OPEN_OR_INCREASE",
        futures_side="long",
        target_notional_pct_nav=4.0,
        max_margin_pct_nav=0.50,
        risk_budget_pct_nav=0.50,
    )

    result = account.process_order(order, "2026-02-03 15:00")

    assert not result.success
    assert result.error == "target_notional_too_small_for_one_contract"
    data.close()


def test_futures_open_and_mark_to_market_updates_cash_once():
    cfg = _config(
        max_abs_notional_pct_nav=1.00,
        max_total_abs_notional_pct_nav=2.00,
        max_margin_pct_nav=0.50,
        max_total_margin_pct_nav=0.50,
        max_risk_budget_pct_nav=0.50,
    )
    data = MarketDataProvider(cfg)
    resolver = FuturesContractResolver(cfg, data)
    account = FuturesAccount(cfg, data, resolver, cash_usd=1_000_000)
    order = TradeOrder(
        symbol="GC.FUT",
        market=Market.FUTURES,
        side=OrderSide.BUY,
        asset_type="futures",
        action="OPEN_OR_INCREASE",
        futures_side="long",
        target_notional_pct_nav=0.50,
        max_margin_pct_nav=0.50,
        risk_budget_pct_nav=0.50,
    )

    opened = account.process_order(order, "2026-02-03 15:00")
    assert opened.success, opened.error
    assert opened.order.quantity == 1
    assert account.margin_locked > 0
    cash_after_open = account.cash_usd
    pos = account.positions["FUTURES:GC.FUT"]
    previous_mark = pos.previous_mark_price

    marks = account.mark_to_market("2026-02-03 16:00")

    assert marks
    mark = marks[0]
    expected_delta = (mark.current_price - previous_mark) * pos.multiplier * pos.contracts
    assert mark.pnl_delta == pytest.approx(expected_delta)
    assert account.cash_usd == pytest.approx(cash_after_open + expected_delta)
    assert account.nav == pytest.approx(account.cash_usd)
    data.close()



def test_roll_closes_old_contract_and_opens_resolved_new_contract():
    cfg = _config(
        max_abs_notional_pct_nav=1.00,
        max_total_abs_notional_pct_nav=2.00,
        max_margin_pct_nav=0.50,
        max_total_margin_pct_nav=0.50,
        max_risk_budget_pct_nav=0.50,
    )
    data = MarketDataProvider(cfg)
    resolver = FuturesContractResolver(cfg, data)
    account = FuturesAccount(cfg, data, resolver, cash_usd=1_000_000)
    order = TradeOrder(
        symbol="GC.FUT",
        market=Market.FUTURES,
        side=OrderSide.BUY,
        asset_type="futures",
        action="OPEN_OR_INCREASE",
        futures_side="long",
        target_notional_pct_nav=0.50,
        max_margin_pct_nav=0.50,
        risk_budget_pct_nav=0.50,
    )

    opened = account.process_order(order, "2026-02-03 15:00")
    assert opened.success, opened.error
    old_contract = opened.metadata["actual_contract"]
    assert old_contract == "GCJ6"

    account.mark_to_market("2026-04-01 15:00")

    assert account.roll_history
    event = account.roll_history[-1]
    assert event["status"] == "rolled"
    assert event["old_contract"] == "GCJ6"
    assert event["new_contract"] == "GCM6"
    assert event["roll_cost"] > 0
    assert event["roll_gap"] == pytest.approx(event["new_open_price"] - event["old_close_price"])
    pos = account.positions["FUTURES:GC.FUT"]
    assert pos.contract_ticker == "GCM6"
    data.close()


def test_portfolio_sync_reserves_margin_without_reducing_nav():
    from src.portfolio.constraints import ConstraintEngine
    from src.portfolio.execution import ExecutionEngine
    from src.portfolio.market_rules import MarketRuleEngine
    from src.portfolio.nav import NavEngine
    from src.portfolio.portfolio import PortfolioEngine
    from src.portfolio.settlement import SettlementEngine
    from src.data.asset_status import AssetStatusProvider

    cfg = _config()
    nav = NavEngine(cfg.fx_rates)
    portfolio = PortfolioEngine(
        cfg,
        nav,
        ConstraintEngine(cfg),
        ExecutionEngine(cfg, nav),
        SettlementEngine(),
        MarketRuleEngine(cfg, AssetStatusProvider(cfg)),
    )

    portfolio.sync_futures_state(
        cash_usd=cfg.initial_cash,
        positions={},
        margin_locked=12_000,
        margin_state="OK",
    )
    snapshot = portfolio.get_snapshot("2026-02-03 15:00")

    assert snapshot.total_nav == pytest.approx(cfg.initial_cash)
    assert snapshot.futures_margin_locked == 12_000
    assert portfolio.reserved_usd == 12_000
    assert portfolio.ensure_cash("USD", 89_000, "2026-02-03 15:00") is False
    assert portfolio.ensure_cash("USD", 88_000, "2026-02-03 15:00") is True



def test_short_futures_uses_sell_side_open_and_buy_side_close_slippage_when_enabled():
    cfg = _config(
        allow_short=True,
        max_abs_notional_pct_nav=1.00,
        max_total_abs_notional_pct_nav=2.00,
        max_margin_pct_nav=0.50,
        max_total_margin_pct_nav=0.50,
        max_risk_budget_pct_nav=0.50,
    )
    data = MarketDataProvider(cfg)
    resolver = FuturesContractResolver(cfg, data)
    account = FuturesAccount(cfg, data, resolver, cash_usd=1_000_000)
    resolved = resolver.resolve("GC.FUT", "2026-02-03 15:00")
    open_bar = data.get_next_executable_futures_bar("GC.FUT", resolved.contract_ticker, "2026-02-03 15:00")
    assert open_bar is not None
    order = TradeOrder(
        symbol="GC.FUT",
        market=Market.FUTURES,
        side=OrderSide.BUY,
        asset_type="futures",
        action="OPEN_OR_INCREASE",
        futures_side="short",
        target_notional_pct_nav=0.50,
        max_margin_pct_nav=0.50,
        risk_budget_pct_nav=0.50,
    )

    opened = account.process_order(order, "2026-02-03 15:00")

    assert opened.success, opened.error
    assert opened.order.futures_side == "short"
    expected_open = round(open_bar.open * (1 - cfg.futures.slippage_bps / 10_000) / cfg.futures.gc_tick_size) * cfg.futures.gc_tick_size
    assert opened.price == pytest.approx(expected_open)

    close_bar = data.get_next_executable_futures_bar("GC.FUT", opened.metadata["actual_contract"], "2026-02-03 15:05")
    assert close_bar is not None
    closed = account.process_order(
        TradeOrder(
            symbol="GC.FUT",
            market=Market.FUTURES,
            side=OrderSide.SELL,
            asset_type="futures",
            action="CLOSE",
            futures_side="flat",
        ),
        "2026-02-03 15:05",
    )

    assert closed.success, closed.error
    expected_close = round(close_bar.open * (1 + cfg.futures.slippage_bps / 10_000) / cfg.futures.gc_tick_size) * cfg.futures.gc_tick_size
    assert closed.price == pytest.approx(expected_close)
    data.close()


def test_futures_logger_persists_trade_metadata_marks_and_rolls():
    from src.platform.logging import ExperimentLogger

    cfg = _config(
        max_abs_notional_pct_nav=1.00,
        max_total_abs_notional_pct_nav=2.00,
        max_margin_pct_nav=0.50,
        max_total_margin_pct_nav=0.50,
        max_risk_budget_pct_nav=0.50,
    )
    data = MarketDataProvider(cfg)
    resolver = FuturesContractResolver(cfg, data)
    account = FuturesAccount(cfg, data, resolver, cash_usd=1_000_000)
    logger = ExperimentLogger(":memory:")
    logger.init_run(cfg.to_dict(), model="test", start_date="2026-02-03", end_date="2026-04-01")

    opened = account.process_order(TradeOrder(
        symbol="GC.FUT", market=Market.FUTURES, side=OrderSide.BUY,
        asset_type="futures", action="OPEN_OR_INCREASE", futures_side="long",
        target_notional_pct_nav=0.50, max_margin_pct_nav=0.50, risk_budget_pct_nav=0.50,
    ), "2026-02-03 15:00")
    assert opened.success, opened.error
    logger.log_trade(opened, "2026-02-03 15:00")
    marks = account.mark_to_market("2026-04-01 15:00")
    for mark in marks:
        logger.log_futures_mark(mark, account.cash_usd)
    for event in account.roll_history:
        logger.log_futures_roll_event(event)

    conn = logger._conn
    trade_meta = conn.execute("SELECT metadata FROM trades WHERE symbol='GC.FUT'").fetchone()[0]
    mark_count = conn.execute("SELECT COUNT(*) FROM futures_marks").fetchone()[0]
    roll_count = conn.execute("SELECT COUNT(*) FROM futures_roll_events").fetchone()[0]

    assert "actual_contract" in trade_meta
    assert mark_count >= 1
    assert roll_count >= 1
    logger.close()
    data.close()



def test_runner_combines_and_logs_futures_account_trades_once():
    from src.platform.experiment import ExperimentRunner
    from src.platform.logging import ExperimentLogger
    from src.portfolio.constraints import ConstraintEngine
    from src.portfolio.execution import ExecutionEngine
    from src.portfolio.market_rules import MarketRuleEngine
    from src.portfolio.nav import NavEngine
    from src.portfolio.portfolio import PortfolioEngine
    from src.portfolio.settlement import SettlementEngine
    from src.data.asset_status import AssetStatusProvider

    cfg = _config(
        max_abs_notional_pct_nav=1.00,
        max_total_abs_notional_pct_nav=2.00,
        max_margin_pct_nav=0.50,
        max_total_margin_pct_nav=0.50,
        max_risk_budget_pct_nav=0.50,
    )
    data = MarketDataProvider(cfg)
    resolver = FuturesContractResolver(cfg, data)
    account = FuturesAccount(cfg, data, resolver, cash_usd=1_000_000)
    nav = NavEngine(cfg.fx_rates)
    portfolio = PortfolioEngine(
        cfg, nav, ConstraintEngine(cfg), ExecutionEngine(cfg, nav),
        SettlementEngine(), MarketRuleEngine(cfg, AssetStatusProvider(cfg)),
    )
    logger = ExperimentLogger(":memory:")
    logger.init_run(cfg.to_dict(), model="test", start_date="2026-02-03", end_date="2026-04-01")

    runner = object.__new__(ExperimentRunner)
    runner.portfolio = portfolio
    runner.futures_account = account
    runner.logger = logger
    runner._logged_futures_trade_count = 0

    opened = account.process_order(TradeOrder(
        symbol="GC.FUT", market=Market.FUTURES, side=OrderSide.BUY,
        asset_type="futures", action="OPEN_OR_INCREASE", futures_side="long",
        target_notional_pct_nav=0.50, max_margin_pct_nav=0.50, risk_budget_pct_nav=0.50,
    ), "2026-02-03 15:00")
    assert opened.success, opened.error

    logged = runner._log_new_futures_account_trades("2026-02-03 15:00")
    assert len(logged) == 1
    assert runner._log_new_futures_account_trades("2026-02-03 15:00") == []
    assert len(runner._all_trade_history()) == 1

    account.mark_to_market("2026-04-01 15:00")
    roll_logged = runner._log_new_futures_account_trades("2026-04-01 15:00")
    assert len(roll_logged) >= 2
    rows = logger._conn.execute("SELECT COUNT(*) FROM trades WHERE market='FUTURES'").fetchone()[0]
    assert rows == len(account.trade_history)
    assert len(runner._all_trade_history()) == len(account.trade_history)
    logger.close()
    data.close()


def test_futures_margin_breach_force_liquidation_enters_trade_history():
    cfg = _config(
        max_abs_notional_pct_nav=1.00,
        max_total_abs_notional_pct_nav=2.00,
        max_margin_pct_nav=0.50,
        max_total_margin_pct_nav=0.50,
        max_risk_budget_pct_nav=0.50,
    )
    data = MarketDataProvider(cfg)
    resolver = FuturesContractResolver(cfg, data)
    account = FuturesAccount(cfg, data, resolver, cash_usd=1_000_000)
    opened = account.process_order(TradeOrder(
        symbol="GC.FUT", market=Market.FUTURES, side=OrderSide.BUY,
        asset_type="futures", action="OPEN_OR_INCREASE", futures_side="long",
        target_notional_pct_nav=0.50, max_margin_pct_nav=0.50, risk_budget_pct_nav=0.50,
    ), "2026-02-03 15:00")
    assert opened.success, opened.error

    account.cash_usd = 1_000
    account.force_liquidate("2026-02-03 16:00")

    forced = [t for t in account.trade_history if t.metadata.get("forced_liquidation")]
    assert forced
    assert not account.positions
    assert account.margin_state == "OK"
    data.close()
