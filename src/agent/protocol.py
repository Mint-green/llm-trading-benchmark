"""
DecisionProtocol — parses LLM output into structured decisions.

Handles: JSON extraction, malformed output, action validation.
Supports v3 format: portfolio_targets, plan_updates, memory_updates.
"""

from __future__ import annotations
import json
import re

from src.core.types import (
    Market, OrderSide, TradeOrder, Decision,
    PortfolioTarget, PlanAction, PlanTrigger, TriggerType,
)


class DecisionProtocol:
    """Parses LLM JSON output into Decision objects."""

    def parse(self, text: str) -> Decision | None:
        """Parse LLM text into a Decision. Returns None on failure."""
        data = self._extract_json(text)
        if data is None:
            return None

        action = data.get("action", "").lower()

        # Extract memory_updates and plan_updates (v3 format)
        memory_updates = data.get("memory_updates", {})
        plan_updates = data.get("plan_updates", [])

        # v3 format: rebalance with portfolio_targets
        if action == "rebalance":
            targets = data.get("portfolio_targets", [])
            trades = self._targets_to_trades(targets)
            return Decision(
                action="trade",
                trades=trades,
                reason=data.get("reason", ""),
                queries=[],
                memory_updates=memory_updates,
                plan_updates=plan_updates,
            )

        # v2 format: trade with trades list
        if action == "trade":
            trades = []
            for t in data.get("trades", []):
                trade = self._parse_trade(t)
                if trade:
                    trades.append(trade)
            return Decision(
                action="trade",
                trades=trades,
                reason=data.get("reason", ""),
                memory_updates=memory_updates,
                plan_updates=plan_updates,
            )

        elif action == "hold":
            return Decision(
                action="hold",
                reason=data.get("reason", ""),
                memory_updates=memory_updates,
                plan_updates=plan_updates,
            )

        elif action == "query":
            queries = data.get("queries", [])
            return Decision(
                action="query",
                queries=queries,
                reason="",
                memory_updates=memory_updates,
                plan_updates=plan_updates,
            )

        return None

    def parse_plan_updates(self, text: str) -> list[dict]:
        """Parse plan_updates from LLM output."""
        data = self._extract_json(text)
        if data is None:
            return []

        updates = data.get("plan_updates", [])
        result = []
        for u in updates:
            symbol = u.get("symbol", "")
            if not symbol:
                continue
            plan_action = u.get("plan_action", "no_change")
            try:
                action = PlanAction(plan_action)
            except ValueError:
                action = PlanAction.NO_CHANGE

            triggers = []
            for t in u.get("triggers", []):
                trigger_type = t.get("type", "")
                try:
                    tt = TriggerType(trigger_type)
                except ValueError:
                    continue
                triggers.append(PlanTrigger(
                    trigger_type=tt,
                    direction=t.get("direction", ""),
                    anchor=t.get("anchor", ""),
                    threshold_pct=t.get("threshold_pct", 0),
                    atr_multiple=t.get("atr_multiple", 0),
                    operator=t.get("operator", ""),
                    since=t.get("since", ""),
                    bars=t.get("bars", 0),
                ))

            result.append({
                "symbol": symbol,
                "plan_action": action,
                "triggers": triggers,
                "intended_horizon_bars": u.get("intended_horizon_bars", 36),
                "plan_note": u.get("plan_note", ""),
            })

        return result

    def parse_memory_updates(self, text: str) -> dict:
        """Parse memory_updates from LLM output."""
        data = self._extract_json(text)
        if data is None:
            return {}

        return data.get("memory_updates", {})

    def _targets_to_trades(self, targets: list[dict]) -> list[TradeOrder]:
        """Convert portfolio_targets to TradeOrders."""
        trades = []
        for t in targets:
            symbol = t.get("symbol", "")
            if not symbol:
                continue

            market = self._ticker_to_market(symbol)
            if market is None:
                continue

            asset_type = t.get("asset_type", "equity")
            reason = t.get("reason", "")

            if market == Market.FUTURES or asset_type == "futures":
                fut_side = (t.get("side") or "long").lower()
                target_notional = t.get("target_notional_pct_nav")
                action = t.get("action", "")
                if fut_side == "flat" or not target_notional or target_notional <= 0:
                    side = OrderSide.SELL
                    action = action or "CLOSE"
                else:
                    side = OrderSide.BUY
                    action = action or "OPEN_OR_INCREASE"
                trades.append(TradeOrder(
                    symbol=symbol,
                    market=Market.FUTURES,
                    side=side,
                    quantity=0,
                    allocation_pct=None,
                    reason=reason,
                    asset_type="futures",
                    action=action,
                    futures_side=fut_side,
                    target_notional_pct_nav=target_notional,
                    max_margin_pct_nav=t.get("max_margin_pct_nav"),
                    risk_budget_pct_nav=t.get("risk_budget_pct_nav"),
                    unit_hint=t.get("unit_hint", {}),
                    risk_trigger=t.get("risk_trigger", ""),
                ))
                continue

            target_pct = t.get("target_pct_nav", 0)

            # Determine side from target_pct
            # Positive = buy, zero/negative = sell
            if target_pct > 0:
                side = OrderSide.BUY
                allocation_pct = target_pct
            else:
                side = OrderSide.SELL
                allocation_pct = abs(target_pct) if target_pct < 0 else 0

            trades.append(TradeOrder(
                symbol=symbol,
                market=market,
                side=side,
                quantity=0,  # will be resolved later
                allocation_pct=allocation_pct,
                reason=reason,
                asset_type=asset_type,
            ))

        return trades

    def _extract_json(self, text: str) -> dict | None:
        """Extract JSON from LLM output (handles markdown, extra text)."""
        text = text.strip()

        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # From markdown code blocks
        for m in re.finditer(r"```(?:json)?\s*([\s\S]+?)```", text):
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                continue

        # Find JSON object in text
        for m in re.finditer(r"\{[\s\S]*\}", text):
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue

        return None

    def _parse_trade(self, data: dict) -> TradeOrder | None:
        """Parse a single trade entry (v2 format)."""
        ticker = data.get("ticker", "")
        side_str = data.get("side", "").lower()

        if not ticker or side_str not in ("buy", "sell"):
            return None

        side = OrderSide.BUY if side_str == "buy" else OrderSide.SELL

        # Determine market from ticker
        market = self._ticker_to_market(ticker)
        if market is None:
            return None

        # Sizing: allocation_pct (preferred), pct_nav (legacy), or quantity
        allocation_pct = data.get("allocation_pct") or data.get("pct_nav")
        quantity = data.get("quantity")

        reason = data.get("reason", "")

        return TradeOrder(
            symbol=ticker,
            market=market,
            side=side,
            quantity=quantity or 0,
            allocation_pct=allocation_pct,
            reason=reason,
        )

    @staticmethod
    def _ticker_to_market(ticker: str) -> Market | None:
        if ticker == "XAUUSD.FOREX":
            return Market.GOLD
        if ticker.endswith(".US"):
            return Market.US
        elif ticker.endswith(".HK"):
            return Market.HK
        elif ticker.endswith(".FUT"):
            return Market.FUTURES
        elif ticker.endswith(".CC") or "-" in ticker:
            return Market.CRYPTO
        elif ticker.startswith("sh.") or ticker.startswith("sz."):
            return Market.CN
        return None
