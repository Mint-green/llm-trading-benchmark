from __future__ import annotations

import pytest

from src.core.config import Config
from src.core.types import Market, OrderSide, PortfolioSnapshot, TradeOrder
from src.evaluation.behavior import BehaviorAnalyzer
from src.evaluation.metrics import MetricsEngine
from src.portfolio.execution import ExecutionEngine
from src.portfolio.nav import NavEngine


def test_execution_keeps_local_values_and_records_usd_audit_values() -> None:
    config = Config(
        fx_rates={"USD": 1.0, "HKD": 8.0, "CNY": 7.0},
        commission_bps={Market.HK: 5.0},
        slippage_bps={Market.HK: 5.0},
    )
    result = ExecutionEngine(config, NavEngine(config.fx_rates)).execute(
        TradeOrder("0700.HK", Market.HK, OrderSide.BUY, quantity=100),
        price=80.0,
        timestamp="2026-02-05 01:30",
    )

    assert result.success
    assert result.cost == 8000.0
    assert result.fees == 8.0
    assert result.metadata == {
        "currency": "HKD",
        "cost_local": 8000.0,
        "fees_local": 8.0,
        "cost_usd": 1000.0,
        "fees_usd": 1.0,
    }


def test_metrics_and_behavior_use_usd_values() -> None:
    config = Config(
        fx_rates={"USD": 1.0, "HKD": 8.0},
        commission_bps={Market.HK: 5.0},
        slippage_bps={Market.HK: 5.0},
    )
    trade = ExecutionEngine(config, NavEngine(config.fx_rates)).execute(
        TradeOrder("0700.HK", Market.HK, OrderSide.BUY, quantity=100),
        price=80.0,
        timestamp="2026-02-05 01:30",
    )
    history = [
        PortfolioSnapshot("2026-02-05 01:30", 100_000, {}, 100_000, {}, {}),
        PortfolioSnapshot("2026-02-05 01:35", 100_000, {}, 100_000, {}, {}),
    ]

    metrics = MetricsEngine(fx_rates=config.fx_rates).compute(history, [trade])
    behavior = BehaviorAnalyzer(fx_rates=config.fx_rates).analyze([], [trade])

    assert metrics["turnover"] == pytest.approx(0.01)
    assert metrics["total_fees_usd"] == 1.0
    assert behavior["trade_analysis"]["total_fees_usd"] == 1.0
