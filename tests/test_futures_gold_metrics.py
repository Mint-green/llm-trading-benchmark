import pytest

from src.core.types import Market, OrderSide, PortfolioSnapshot, TradeOrder, TradeResult
from src.evaluation.metrics import MetricsEngine


def test_metrics_append_futures_and_gold_fields_without_changing_core_metrics():
    history = [
        PortfolioSnapshot(
            timestamp="2026-02-03 15:00",
            cash=100_000,
            positions={},
            total_nav=100_000,
            market_exposure={},
            fx_rates={},
            futures_margin_locked=0,
            futures_margin_state="OK",
            futures_pnl_delta=0,
        ),
        PortfolioSnapshot(
            timestamp="2026-02-03 16:00",
            cash=101_000,
            positions={},
            total_nav=101_000,
            market_exposure={Market.GOLD: 3_000},
            fx_rates={},
            futures_margin_locked=12_000,
            futures_margin_state="WARNING",
            futures_pnl_delta=1_000,
        ),
    ]
    trades = [
        TradeResult(
            order=TradeOrder(
                symbol="GC.FUT", market=Market.FUTURES, side=OrderSide.BUY,
                quantity=1, asset_type="futures",
            ),
            success=True,
            price=4900,
            cost=490000,
            fees=2.5,
            metadata={"roll_trade": True, "pnl_delta": 25.0},
        ),
        TradeResult(
            order=TradeOrder(
                symbol="XAUUSD.FOREX", market=Market.GOLD, side=OrderSide.BUY,
                quantity=1.0, asset_type="gold_spot",
            ),
            success=True,
            price=3000,
            cost=3000,
            fees=1.5,
        ),
        TradeResult(
            order=TradeOrder(
                symbol="GC.FUT", market=Market.FUTURES, side=OrderSide.BUY,
                quantity=0, asset_type="futures",
            ),
            success=False,
            error="risk_budget_exceeded",
        ),
    ]

    metrics = MetricsEngine().compute(history, trades)

    assert metrics["total_return"] == pytest.approx(0.01)
    assert metrics["futures_trades"] == 1
    assert metrics["futures_rejected_orders"] == 1
    assert metrics["futures_roll_trades"] == 1
    assert metrics["max_futures_margin_pct_nav"] == pytest.approx(11.8812)
    assert metrics["futures_margin_warning_count"] == 1
    assert metrics["futures_mark_pnl_usd"] == 1000
    assert metrics["futures_close_pnl_usd"] == 25
    assert metrics["futures_pnl_delta_usd"] == 1025
    assert metrics["gold_trades"] == 1
    assert metrics["gold_fees_usd"] == 1.5
