"""
DecisionProtocol — parses LLM output into structured decisions.

Handles: JSON extraction, malformed output, action validation.
"""

from __future__ import annotations
import json
import re

from src.core.types import Market, OrderSide, TradeOrder, Decision


class DecisionProtocol:
    """Parses LLM JSON output into Decision objects."""

    def parse(self, text: str) -> Decision | None:
        """Parse LLM text into a Decision. Returns None on failure."""
        data = self._extract_json(text)
        if data is None:
            return None

        action = data.get("action", "").lower()

        if action == "trade":
            trades = []
            for t in data.get("trades", []):
                trade = self._parse_trade(t)
                if trade:
                    trades.append(trade)
            return Decision(action="trade", trades=trades, reason=data.get("reason", ""))

        elif action == "hold":
            return Decision(action="hold", reason=data.get("reason", ""))

        elif action == "query":
            queries = data.get("queries", [])
            return Decision(action="query", queries=queries, reason="")

        return None

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
        """Parse a single trade entry."""
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
        if ticker.endswith(".US"):
            return Market.US
        elif ticker.endswith(".HK"):
            return Market.HK
        elif ticker.endswith(".CC") or "-" in ticker:
            return Market.CRYPTO
        elif ticker.startswith("sh.") or ticker.startswith("sz."):
            return Market.CN
        return None
