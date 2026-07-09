"""V5 regression tests for portfolio target execution semantics."""

from src.agent.runner import AgentRunner
from src.core.types import Market, OrderSide, PortfolioSnapshot, Position, TradeOrder


def test_zero_target_sell_resolves_to_full_position_quantity():
    runner = object.__new__(AgentRunner)
    runner._price_lookup = None

    snapshot = PortfolioSnapshot(
        timestamp="2026-01-06 15:00",
        cash=95_000.0,
        positions={
            "US:AAPL.US": Position(
                symbol="AAPL.US",
                market=Market.US,
                quantity=42,
                avg_cost=100.0,
                current_price=110.0,
            )
        },
        total_nav=100_000.0,
        market_exposure={Market.US: 4_620.0},
        fx_rates={"USD": 1.0},
    )
    close_order = TradeOrder(
        symbol="AAPL.US",
        market=Market.US,
        side=OrderSide.SELL,
        allocation_pct=0,
        reason="close position",
    )

    resolved = AgentRunner._resolve_trades(runner, [close_order], snapshot)

    assert len(resolved) == 1
    assert resolved[0].side == OrderSide.SELL
    assert resolved[0].quantity == 42
    assert resolved[0].allocation_pct == 0