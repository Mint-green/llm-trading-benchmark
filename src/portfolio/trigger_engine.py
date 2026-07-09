"""
TriggerEngine — evaluates structured triggers for plan monitoring.

9 trigger types:
  price_move_pct       — price moved X% from anchor
  atr_move             — price moved X*ATR from anchor
  pnl_pct              — position PnL hit threshold
  trailing_drawdown_pct — price dropped X% from peak
  trailing_atr         — price dropped X*ATR from peak
  bars_elapsed         — N bars since last review
  regime_change        — market regime changed
  asset_status_change  — asset became non-tradable
  margin_risk_change   — margin risk level changed
"""

from __future__ import annotations
from dataclasses import dataclass

from src.core.types import (
    TriggerType, PlanTrigger, ActivePlan,
    Market, RiskMode,
)
from src.core.config import TriggerConfig, CryptoTriggerConfig


@dataclass
class TriggerEvent:
    """A triggered event that needs attention."""
    symbol: str
    plan_id: str
    trigger_type: TriggerType
    priority: str  # P0, P1, P2, P3
    trigger_detail: dict
    actual_value: float
    threshold: float


class TriggerEngine:
    """Evaluates structured triggers and detects events."""

    def __init__(
        self,
        trigger_config: TriggerConfig | None = None,
        crypto_trigger_config: CryptoTriggerConfig | None = None,
    ):
        self._config = trigger_config or TriggerConfig()
        self._crypto_config = crypto_trigger_config or CryptoTriggerConfig()

    def evaluate_plan(
        self,
        plan: ActivePlan,
        current_price: float,
        current_pnl_pct: float,
        current_atr: float,
        bars_since_review: int,
        market_regime: RiskMode,
        asset_tradable: bool,
        market: Market,
    ) -> list[TriggerEvent]:
        """Evaluate all triggers for a plan and return triggered events.

        Args:
            plan: the active plan to evaluate
            current_price: current price in local currency
            current_pnl_pct: position PnL as fraction (-0.025 = -2.5%)
            current_atr: current ATR as % of price
            bars_since_review: bars elapsed since last review
            market_regime: current risk mode
            asset_tradable: whether asset is currently tradable
            market: which market (for crypto-specific thresholds)
        """
        events: list[TriggerEvent] = []
        config = self._crypto_config if market == Market.CRYPTO else self._config

        for trigger in plan.triggers:
            event = self._evaluate_single(
                trigger, plan, current_price, current_pnl_pct,
                current_atr, bars_since_review, market_regime,
                asset_tradable, config,
            )
            if event is not None:
                events.append(event)

        return events

    def _evaluate_single(
        self,
        trigger: PlanTrigger,
        plan: ActivePlan,
        current_price: float,
        current_pnl_pct: float,
        current_atr: float,
        bars_since_review: int,
        market_regime: RiskMode,
        asset_tradable: bool,
        config: TriggerConfig | CryptoTriggerConfig,
    ) -> TriggerEvent | None:
        """Evaluate a single trigger."""

        if trigger.trigger_type == TriggerType.PRICE_MOVE_PCT:
            return self._eval_price_move(trigger, plan, current_price, config)
        elif trigger.trigger_type == TriggerType.ATR_MOVE:
            return self._eval_atr_move(trigger, plan, current_price, current_atr, config)
        elif trigger.trigger_type == TriggerType.PNL_PCT:
            return self._eval_pnl_pct(trigger, plan, current_pnl_pct, config)
        elif trigger.trigger_type == TriggerType.TRAILING_DRAWDOWN_PCT:
            return self._eval_trailing_drawdown(trigger, plan, current_price, config)
        elif trigger.trigger_type == TriggerType.TRAILING_ATR:
            return self._eval_trailing_atr(trigger, plan, current_price, current_atr, config)
        elif trigger.trigger_type == TriggerType.BARS_ELAPSED:
            return self._eval_bars_elapsed(trigger, plan, bars_since_review)
        elif trigger.trigger_type == TriggerType.REGIME_CHANGE:
            return self._eval_regime_change(trigger, plan, market_regime)
        elif trigger.trigger_type == TriggerType.ASSET_STATUS_CHANGE:
            return self._eval_asset_status(trigger, plan, asset_tradable)
        elif trigger.trigger_type == TriggerType.MARGIN_RISK_CHANGE:
            return None  # TODO: implement margin risk detection

        return None

    def _eval_price_move(
        self, trigger: PlanTrigger, plan: ActivePlan,
        current_price: float, config,
    ) -> TriggerEvent | None:
        """Evaluate price_move_pct trigger."""
        anchor_price = self._get_anchor_price(trigger.anchor, plan)
        if anchor_price <= 0:
            return None

        move_pct = abs(current_price - anchor_price) / anchor_price
        threshold = trigger.threshold_pct or config.price_move_pct

        if move_pct >= threshold:
            direction = "up" if current_price > anchor_price else "down"
            if trigger.direction and direction != trigger.direction:
                return None
            return TriggerEvent(
                symbol=plan.symbol,
                plan_id=plan.plan_id,
                trigger_type=TriggerType.PRICE_MOVE_PCT,
                priority="P2",
                trigger_detail={
                    "direction": direction,
                    "anchor": trigger.anchor,
                    "anchor_price": anchor_price,
                },
                actual_value=move_pct,
                threshold=threshold,
            )
        return None

    def _eval_atr_move(
        self, trigger: PlanTrigger, plan: ActivePlan,
        current_price: float, current_atr: float, config,
    ) -> TriggerEvent | None:
        """Evaluate atr_move trigger."""
        anchor_price = self._get_anchor_price(trigger.anchor, plan)
        if anchor_price <= 0:
            return None

        price_diff = abs(current_price - anchor_price)
        atr_value = anchor_price * (current_atr / 100) if current_atr > 0 else 0
        if atr_value <= 0:
            return None

        atr_multiple = price_diff / atr_value
        threshold = trigger.atr_multiple or config.atr_move_multiple

        if atr_multiple >= threshold:
            direction = "up" if current_price > anchor_price else "down"
            return TriggerEvent(
                symbol=plan.symbol,
                plan_id=plan.plan_id,
                trigger_type=TriggerType.ATR_MOVE,
                priority="P1",
                trigger_detail={
                    "direction": direction,
                    "anchor": trigger.anchor,
                    "atr_multiple": atr_multiple,
                },
                actual_value=atr_multiple,
                threshold=threshold,
            )
        return None

    def _eval_pnl_pct(
        self, trigger: PlanTrigger, plan: ActivePlan,
        current_pnl_pct: float, config,
    ) -> TriggerEvent | None:
        """Evaluate pnl_pct trigger."""
        threshold = trigger.threshold_pct
        if isinstance(config, CryptoTriggerConfig):
            threshold = threshold or config.pnl_pct_threshold
        else:
            threshold = threshold or config.pnl_pct_threshold

        operator = trigger.operator or "<="
        triggered = False
        if operator == "<=":
            triggered = current_pnl_pct <= threshold
        elif operator == ">=":
            triggered = current_pnl_pct >= threshold

        if triggered:
            return TriggerEvent(
                symbol=plan.symbol,
                plan_id=plan.plan_id,
                trigger_type=TriggerType.PNL_PCT,
                priority="P1",
                trigger_detail={
                    "operator": operator,
                    "current_pnl_pct": current_pnl_pct,
                },
                actual_value=current_pnl_pct,
                threshold=threshold,
            )
        return None

    def _eval_trailing_drawdown(
        self, trigger: PlanTrigger, plan: ActivePlan,
        current_price: float, config,
    ) -> TriggerEvent | None:
        """Evaluate trailing_drawdown_pct trigger."""
        peak = plan.peak_since_entry
        if peak <= 0:
            return None

        drawdown_pct = (peak - current_price) / peak
        threshold = trigger.threshold_pct or config.trailing_drawdown_pct

        if drawdown_pct >= threshold:
            return TriggerEvent(
                symbol=plan.symbol,
                plan_id=plan.plan_id,
                trigger_type=TriggerType.TRAILING_DRAWDOWN_PCT,
                priority="P1",
                trigger_detail={
                    "peak": peak,
                    "current_price": current_price,
                    "drawdown_pct": drawdown_pct,
                },
                actual_value=drawdown_pct,
                threshold=threshold,
            )
        return None

    def _eval_trailing_atr(
        self, trigger: PlanTrigger, plan: ActivePlan,
        current_price: float, current_atr: float, config,
    ) -> TriggerEvent | None:
        """Evaluate trailing_atr trigger."""
        peak = plan.peak_since_entry
        if peak <= 0:
            return None

        price_diff = peak - current_price
        atr_value = peak * (current_atr / 100) if current_atr > 0 else 0
        if atr_value <= 0:
            return None

        atr_multiple = price_diff / atr_value
        threshold = trigger.atr_multiple or config.trailing_atr_multiple

        if atr_multiple >= threshold:
            return TriggerEvent(
                symbol=plan.symbol,
                plan_id=plan.plan_id,
                trigger_type=TriggerType.TRAILING_ATR,
                priority="P1",
                trigger_detail={
                    "peak": peak,
                    "atr_multiple": atr_multiple,
                },
                actual_value=atr_multiple,
                threshold=threshold,
            )
        return None

    def _eval_bars_elapsed(
        self, trigger: PlanTrigger, plan: ActivePlan,
        bars_since_review: int,
    ) -> TriggerEvent | None:
        """Evaluate bars_elapsed trigger."""
        threshold = trigger.bars or self._config.bars_elapsed

        if bars_since_review >= threshold:
            return TriggerEvent(
                symbol=plan.symbol,
                plan_id=plan.plan_id,
                trigger_type=TriggerType.BARS_ELAPSED,
                priority="P2",
                trigger_detail={
                    "since": trigger.since or "last_review",
                    "bars_elapsed": bars_since_review,
                },
                actual_value=bars_since_review,
                threshold=threshold,
            )
        return None

    def _eval_regime_change(
        self, trigger: PlanTrigger, plan: ActivePlan,
        market_regime: RiskMode,
    ) -> TriggerEvent | None:
        """Evaluate regime_change trigger."""
        # Trigger if regime is RED
        if market_regime == RiskMode.RED:
            return TriggerEvent(
                symbol=plan.symbol,
                plan_id=plan.plan_id,
                trigger_type=TriggerType.REGIME_CHANGE,
                priority="P1",
                trigger_detail={"new_regime": market_regime.value},
                actual_value=2.0,  # RED = 2
                threshold=2.0,
            )
        return None

    def _eval_asset_status(
        self, trigger: PlanTrigger, plan: ActivePlan,
        asset_tradable: bool,
    ) -> TriggerEvent | None:
        """Evaluate asset_status_change trigger."""
        if not asset_tradable:
            return TriggerEvent(
                symbol=plan.symbol,
                plan_id=plan.plan_id,
                trigger_type=TriggerType.ASSET_STATUS_CHANGE,
                priority="P1",
                trigger_detail={"tradable": False},
                actual_value=0.0,
                threshold=1.0,
            )
        return None

    def _get_anchor_price(self, anchor: str, plan: ActivePlan) -> float:
        """Get anchor price from plan based on anchor type."""
        if anchor == "last_review_price":
            return plan.last_review_price
        elif anchor == "entry_price":
            return plan.entry_price
        elif anchor == "peak_since_entry":
            return plan.peak_since_entry
        elif anchor == "peak_since_last_review":
            return plan.peak_since_last_review
        return plan.last_review_price  # default

    @staticmethod
    def make_default_triggers(
        entry_price: float,
        atr_at_entry: float,
        config: TriggerConfig,
    ) -> list[PlanTrigger]:
        """Create default triggers for a new plan.

        Default triggers:
        - pnl_pct <= -2.5% (stop loss review)
        - trailing_drawdown_pct 2% (from peak)
        - bars_elapsed 6 (30min review)
        """
        return [
            PlanTrigger(
                trigger_type=TriggerType.PNL_PCT,
                operator="<=",
                threshold_pct=config.pnl_pct_threshold,
            ),
            PlanTrigger(
                trigger_type=TriggerType.TRAILING_DRAWDOWN_PCT,
                anchor="peak_since_entry",
                threshold_pct=config.trailing_drawdown_pct,
            ),
            PlanTrigger(
                trigger_type=TriggerType.BARS_ELAPSED,
                since="last_review",
                bars=config.bars_elapsed,
            ),
        ]
