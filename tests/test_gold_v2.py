from dataclasses import replace
from pathlib import Path

import pytest

from src.agent.tools import ToolSystem
from src.core.config import Config
from src.core.types import Market, OrderSide, PortfolioTarget, TradeOrder
from src.data.features import FeatureGenerator
from src.data.provider import MarketDataProvider
from src.data.universe import UniverseRegistry
from src.portfolio.constraints import ConstraintEngine
from src.portfolio.execution import ExecutionEngine
from src.portfolio.market_rules import MarketRuleEngine
from src.portfolio.nav import NavEngine
from src.portfolio.portfolio import PortfolioEngine
from src.portfolio.settlement import SettlementEngine
from src.data.asset_status import AssetStatusProvider


import os
DATA_ROOT = Path(os.environ.get("STOCK_DATA_ROOT", os.path.expanduser("~/Desktop/getStockData")))
DATA_DB = DATA_ROOT / "data" / "GOLD_stock.db"
pytestmark = pytest.mark.skipif(not DATA_DB.exists(), reason="GOLD_stock.db is not available")


def _portfolio(cfg: Config) -> PortfolioEngine:
    nav = NavEngine(cfg.fx_rates)
    return PortfolioEngine(
        cfg,
        nav,
        ConstraintEngine(cfg),
        ExecutionEngine(cfg, nav),
        SettlementEngine(),
        MarketRuleEngine(cfg, AssetStatusProvider(cfg)),
    )


def test_gold_provider_filters_to_xauusd_and_exposes_bid_ask():
    cfg = Config()
    data = MarketDataProvider(cfg)

    symbols = data.get_universe_symbols(Market.GOLD)
    all_bars = data.load_all_bars(Market.GOLD, "2026-02-03", "2026-02-03")
    quote = data.get_gold_bid_ask("XAUUSD.FOREX", "2026-02-03 15:00")

    assert symbols == ["XAUUSD.FOREX"]
    assert set(all_bars) == {"XAUUSD.FOREX"}
    assert quote["bid"] is not None
    assert quote["bid"].close > 0
    # ASK exists in the current data set; keep this explicit so GLD/ASK filtering stays tested.
    assert quote["ask"] is not None
    assert quote["ask"].close > 0
    data.close()


def test_gold_universe_ignores_gld_us_even_if_constituent_file_contains_it():
    cfg = Config()
    universe = UniverseRegistry(cfg)

    assets = universe.get_assets(Market.GOLD)

    assert [a.ticker for a in assets] == ["XAUUSD.FOREX"]
    assert assets[0].asset_class == "gold_spot"


def test_query_asset_handles_xauusd_as_gold_market():
    cfg = Config()
    data = MarketDataProvider(cfg)
    tools = ToolSystem(data, FeatureGenerator(), lambda: None)

    out = tools.execute_tool(
        "query_asset",
        {"symbol": "XAUUSD.FOREX", "fields": ["quote", "cost"], "recent_bar_count": 0},
        "2026-02-03 15:00",
    )

    assert "[ASSET] XAUUSD.FOREX" in out
    assert "Price:" in out
    assert "~5 bps" in out
    data.close()


def test_gold_fractional_spot_trade_executes_without_integer_lot_rounding():
    cfg = Config()
    data = MarketDataProvider(cfg)
    portfolio = _portfolio(cfg)
    bar = data.get_last_completed_bar(Market.GOLD, "XAUUSD.FOREX", "2026-02-03 15:00")
    assert bar is not None

    order = TradeOrder(
        symbol="XAUUSD.FOREX",
        market=Market.GOLD,
        side=OrderSide.BUY,
        quantity=0.5,
        asset_type="gold_spot",
        reason="test fractional gold",
    )
    result = portfolio.process_order(order, bar.close, "2026-02-03 15:00")

    assert result.success, result.error
    assert result.order.quantity == pytest.approx(0.5)
    pos = portfolio.get_position("GOLD:XAUUSD.FOREX")
    assert pos is not None
    assert pos.quantity == pytest.approx(0.5)
    data.close()


def test_gold_target_conversion_uses_fractional_ounces():
    cfg = Config()
    portfolio = _portfolio(cfg)
    target = PortfolioTarget(
        symbol="XAUUSD.FOREX",
        asset_type="gold_spot",
        target_pct_nav=0.03,
        reason="small gold allocation",
    )

    converted = portfolio.convert_targets_to_orders(
        [target],
        prices={"XAUUSD.FOREX": 3000.0},
        markets={"XAUUSD.FOREX": Market.GOLD},
    )

    assert not converted.skipped
    assert len(converted.orders) == 1
    assert converted.orders[0].quantity == pytest.approx(1.0)
    assert converted.orders[0].market == Market.GOLD
