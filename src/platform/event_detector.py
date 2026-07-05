"""
EventDetector — detects events that trigger decisions.

Event categories:
  Portfolio Events:   position PnL, trailing stop, bars elapsed, sellable status
  Market Events:      open/close, regime change, volatility spike, breadth collapse
  Risk Events:        cash below reserve, exposure near max, margin warning
  Memory Events:      watchlist condition, avoid expired, plan expired

Priority levels:
  P0: System forced (margin breach, hard constraint, forced liquidation)
  P1: Must focused decision (stop review, trailing stop, severe adverse, regime RED)
  P2: Can merge focused decision (take-profit, scheduled review, watchlist)
  P3: Info only (low impact, minor fluctuation)
"""

from __future__ import annotations
from dataclasses import dataclass

from src.core.types import (
    Market, RiskMode, TriggerType,
    ActivePlan, PortfolioSnapshot,
    CandidateBuckets,
)
from src.core.config import Config
from src.portfolio.trigger_engine import TriggerEngine, TriggerEvent
from src.data.features import FeatureGenerator


@dataclass
class MarketEvent:
    """A detected market-level event."""
    event_type: str  # regime_change, market_open, market_close, volatility_spike
    market: Market
    priority: str  # P0, P1, P2, P3
    detail: dict


@dataclass
class RiskEvent:
    """A detected risk event."""
    event_type: str  # cash_below_reserve, exposure_near_max, margin_warning
    priority: str
    detail: dict


class EventDetector:
    """Detects events that trigger decisions."""

    def __init__(
        self,
        config: Config,
        trigger_engine: TriggerEngine,
        features: FeatureGenerator,
    ):
        self._config = config
        self._trigger_engine = trigger_engine
        self._features = features

        # State tracking
        self._prev_regime: dict[str, RiskMode] = {}  # market -> regime
        self._prev_open_state: dict[str, bool] = {}   # market -> was open

    def detect(
        self,
        timestamp: str,
        snapshot: PortfolioSnapshot,
        all_bars: dict[Market, dict[str, list]],
        plans: dict[str, ActivePlan],
        open_markets: list[Market],
        closed_markets: list[Market],
        risk_mode: RiskMode = RiskMode.GREEN,
        lightweight: bool = False,
    ) -> tuple[list[TriggerEvent], list[MarketEvent], list[RiskEvent]]:
        """Detect all events at this timestamp.

        Args:
            lightweight: if True, skip expensive volatility spike detection
                (used for AUTO_HOLD pre-check to save ~0.2s per timestamp)

        Returns:
            (trigger_events, market_events, risk_events)
        """
        trigger_events: list[TriggerEvent] = []
        market_events: list[MarketEvent] = []
        risk_events: list[RiskEvent] = []

        # 1. Detect market events (skip volatility spike in lightweight mode)
        if lightweight:
            # Only detect open/close transitions (cheap)
            for market in open_markets:
                was_open = self._prev_open_state.get(market.value, True)
                if not was_open:
                    market_events.append(MarketEvent(
                        event_type="market_open", market=market, priority="P3",
                        detail={"timestamp": timestamp},
                    ))
                self._prev_open_state[market.value] = True
            for market in closed_markets:
                was_open = self._prev_open_state.get(market.value, True)
                if was_open:
                    market_events.append(MarketEvent(
                        event_type="market_close", market=market, priority="P3",
                        detail={"timestamp": timestamp},
                    ))
                self._prev_open_state[market.value] = False
        else:
            market_events.extend(self._detect_market_events(timestamp, open_markets, closed_markets, all_bars))

        # 2. Detect risk events (cheap, always run)
        risk_events.extend(self._detect_risk_events(snapshot, risk_mode))

        # 3. Detect plan trigger events (only if plans exist)
        for symbol, plan in plans.items():
            events = self._evaluate_plan_triggers(
                plan, symbol, snapshot, all_bars, risk_mode,
            )
            trigger_events.extend(events)

        return trigger_events, market_events, risk_events

    def _detect_market_events(
        self,
        timestamp: str,
        open_markets: list[Market],
        closed_markets: list[Market],
        all_bars: dict[Market, dict[str, list]],
    ) -> list[MarketEvent]:
        """Detect market-level events."""
        events: list[MarketEvent] = []

        # Market open/close transitions
        for market in open_markets:
            was_open = self._prev_open_state.get(market.value, True)
            if not was_open:
                events.append(MarketEvent(
                    event_type="market_open",
                    market=market,
                    priority="P3",
                    detail={"timestamp": timestamp},
                ))
            self._prev_open_state[market.value] = True

        for market in closed_markets:
            was_open = self._prev_open_state.get(market.value, True)
            if was_open:
                events.append(MarketEvent(
                    event_type="market_close",
                    market=market,
                    priority="P3",
                    detail={"timestamp": timestamp},
                ))
            self._prev_open_state[market.value] = False

        # Volatility spike detection (for open markets)
        for market in open_markets:
            if market == Market.CRYPTO:
                continue
            vol_event = self._detect_volatility_spike(market, all_bars, timestamp)
            if vol_event:
                events.append(vol_event)

        return events

    def _detect_volatility_spike(
        self, market: Market, all_bars: dict[Market, dict[str, list]], timestamp: str,
    ) -> MarketEvent | None:
        """Detect if market volatility spiked (> 2x normal)."""
        market_bars = all_bars.get(market, {})
        if not market_bars:
            return None

        # Compute average ATR across market
        atr_values = []
        for symbol, bars in market_bars.items():
            snap = self._features.compute(bars, timestamp)
            if snap and snap.atr_pct > 0:
                atr_values.append(snap.atr_pct)

        if not atr_values:
            return None

        avg_atr = sum(atr_values) / len(atr_values)

        # Spike if average ATR > 2% (for stocks)
        if avg_atr > 2.0:
            return MarketEvent(
                event_type="volatility_spike",
                market=market,
                priority="P2",
                detail={"avg_atr_pct": avg_atr, "threshold": 2.0},
            )

        return None

    def _detect_risk_events(
        self, snapshot: PortfolioSnapshot, risk_mode: RiskMode,
    ) -> list[RiskEvent]:
        """Detect risk events."""
        events: list[RiskEvent] = []
        nav = snapshot.total_nav

        if nav <= 0:
            return events

        # Cash below reserve
        cash_pct = snapshot.cash / nav
        min_cash = self._config.position_limits.min_cash_ratio
        if cash_pct < min_cash:
            events.append(RiskEvent(
                event_type="cash_below_reserve",
                priority="P1",
                detail={"cash_pct": cash_pct, "min_required": min_cash},
            ))

        # Market exposure near max
        for market, exposure in snapshot.market_exposure.items():
            exposure_pct = exposure / nav
            max_pct = self._config.position_limits.max_market_exposure
            if exposure_pct > max_pct * 0.9:  # 90% of limit
                events.append(RiskEvent(
                    event_type="exposure_near_max",
                    priority="P2",
                    detail={
                        "market": market.value,
                        "exposure_pct": exposure_pct,
                        "max_pct": max_pct,
                    },
                ))

        # Crypto exposure near max
        crypto_exposure = sum(
            v for m, v in snapshot.market_exposure.items() if m == Market.CRYPTO
        )
        crypto_pct = crypto_exposure / nav
        max_crypto = self._config.position_limits.max_crypto_exposure
        if crypto_pct > max_crypto * 0.9:
            events.append(RiskEvent(
                event_type="crypto_exposure_near_max",
                priority="P2",
                detail={"crypto_pct": crypto_pct, "max_pct": max_crypto},
            ))

        # Futures margin state
        if snapshot.futures_margin_state in ("WARNING", "BREACH"):
            events.append(RiskEvent(
                event_type="futures_margin_" + snapshot.futures_margin_state.lower(),
                priority="P0" if snapshot.futures_margin_state == "BREACH" else "P1",
                detail={
                    "margin_state": snapshot.futures_margin_state,
                    "margin_locked": snapshot.futures_margin_locked,
                    "pnl_delta": snapshot.futures_pnl_delta,
                },
            ))

        # Regime change to RED
        if risk_mode == RiskMode.RED:
            events.append(RiskEvent(
                event_type="regime_red",
                priority="P1",
                detail={"risk_mode": "RED"},
            ))

        return events

    def _evaluate_plan_triggers(
        self,
        plan: ActivePlan,
        symbol: str,
        snapshot: PortfolioSnapshot,
        all_bars: dict[Market, dict[str, list]],
        risk_mode: RiskMode,
    ) -> list[TriggerEvent]:
        """Evaluate triggers for a single plan."""
        # Find position
        pos = None
        for key, p in snapshot.positions.items():
            if p.symbol == symbol:
                pos = p
                break

        if pos is None:
            return []

        # Get price data
        bars = all_bars.get(pos.market, {}).get(symbol, [])
        if not bars:
            return []

        # Get current price and indicators
        current_price = 0.0
        current_atr = 0.0
        for bar in reversed(bars):
            if bar.timestamp <= snapshot.timestamp:
                current_price = bar.close
                break

        snap = self._features.compute(bars, snapshot.timestamp)
        if snap:
            current_atr = snap.atr_pct

        # Compute PnL
        pnl_pct = 0.0
        if pos.avg_cost > 0:
            pnl_pct = (current_price - pos.avg_cost) / pos.avg_cost

        # Compute bars since review
        bars_since_review = self._compute_bars_since(plan.last_review_time, snapshot.timestamp)

        # Check tradability
        asset_tradable = True  # TODO: check from AssetStatusProvider

        # Evaluate triggers
        return self._trigger_engine.evaluate_plan(
            plan=plan,
            current_price=current_price,
            current_pnl_pct=pnl_pct,
            current_atr=current_atr,
            bars_since_review=bars_since_review,
            market_regime=risk_mode,
            asset_tradable=asset_tradable,
            market=pos.market,
        )

    @staticmethod
    def _compute_bars_since(last_review: str, current: str) -> int:
        """Compute number of 5-min bars between two timestamps."""
        try:
            from datetime import datetime
            t1 = datetime.strptime(last_review[:16], "%Y-%m-%d %H:%M")
            t2 = datetime.strptime(current[:16], "%Y-%m-%d %H:%M")
            minutes = (t2 - t1).total_seconds() / 60
            return max(0, int(minutes / 5))
        except ValueError:
            return 0

    def reset(self) -> None:
        """Reset state for a new benchmark day."""
        self._prev_regime = {}
        self._prev_open_state = {}
