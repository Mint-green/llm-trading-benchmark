"""V5 tests for screen_universe signal quality."""

from src.agent.tools import ToolSystem
from src.core.types import IndicatorSnapshot, Market, PortfolioSnapshot


class ScreenDataProvider:
    def __init__(self, symbols_by_market):
        self.symbols_by_market = symbols_by_market

    def get_universe_symbols(self, market):
        return self.symbols_by_market.get(market, [])

    def load_bars(self, market, symbol, start, end):
        return [object()]


class ScreenFeatureGenerator:
    def __init__(self, snapshots):
        self.snapshots = snapshots

    def compute(self, bars, timestamp):
        return self.snapshots.pop(0)


def snap(
    symbol_price=100.0,
    chg_1h=0.5,
    chg_1d=1.0,
    rsi=55.0,
    trend="UU",
    setup="strong_continuation",
    recent_score=1,
    ret_30m=0.2,
    rsi_d1h=2.0,
    bb_position=0.5,
):
    return IndicatorSnapshot(
        timestamp="2026-01-06 02:00",
        price=symbol_price,
        chg_5m=0.1,
        chg_1h=chg_1h,
        chg_1d=chg_1d,
        rel_volume=1.0,
        rsi=rsi,
        atr_pct=0.8,
        trend=trend,
        bb_position=bb_position,
        high_low_pos=0.5,
        ret_30m=ret_30m,
        rsi_d1h=rsi_d1h,
        trend6="UUUUU",
        setup=setup,
        recent_score=recent_score,
    )


def portfolio():
    return PortfolioSnapshot(
        timestamp="2026-01-06 02:00",
        cash=100_000.0,
        positions={},
        total_nav=100_000.0,
        market_exposure={},
        fx_rates={"USD": 1.0, "CNY": 7.25, "HKD": 7.8},
    )


def test_screen_universe_outputs_v4_setup_fields_and_risk_tag():
    provider = ScreenDataProvider({Market.HK: ["GOOD.HK", "HOT.HK"]})
    features = ScreenFeatureGenerator([
        snap(chg_1h=0.8, rsi=54, setup="strong_continuation", recent_score=2),
        snap(chg_1h=3.2, rsi=64, setup="strong_continuation", recent_score=2, bb_position=0.9),
    ])
    tools = ToolSystem(provider, features, portfolio)

    result = tools.execute_tool(
        "screen_universe",
        {"market": "HK", "bucket": "trend_leaders", "limit": 2},
        "2026-01-06 02:00",
    )

    assert "setup|recent_score|risk" in result
    assert "GOOD.HK" in result
    assert "HOT.HK" in result
    assert "extended_intraday" in result
    assert result.splitlines()[2].startswith("GOOD.HK|")


def test_screen_universe_keeps_cn_min_lot_filter():
    provider = ScreenDataProvider({Market.CN: ["EXPENSIVE.CN", "OK.CN"]})
    features = ScreenFeatureGenerator([
        snap(symbol_price=400.0),  # 100 shares costs more than 4.5% NAV in USD.
        snap(symbol_price=20.0),
    ])
    tools = ToolSystem(provider, features, portfolio)

    result = tools.execute_tool(
        "screen_universe",
        {"market": "CN", "bucket": "trend_leaders", "limit": 5},
        "2026-01-06 02:00",
    )

    assert "OK.CN" in result
    assert "EXPENSIVE.CN" not in result