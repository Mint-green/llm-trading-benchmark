"""
BehaviorAnalyzer — analyzes agent behavior patterns.

Tracks: tool usage, rejected orders, constraint triggers, decision patterns.
Does NOT participate in scoring (as per design doc).
"""

from __future__ import annotations
from collections import Counter
from typing import Any

from src.core.types import AgentRound, TradeResult, Decision
from src.core.interfaces import IBehaviorAnalyzer
from src.evaluation.trade_values import trade_fees_usd


class BehaviorAnalyzer(IBehaviorAnalyzer):
    """Analyzes agent behavior patterns."""

    def __init__(self, fx_rates: dict[str, float] | None = None):
        self._fx_rates = fx_rates or {
            "USD": 1.0, "HKD": 7.8, "CNY": 7.25, "JPY": 155.0,
        }

    def analyze(
        self, rounds: list[AgentRound], trades: list[TradeResult],
    ) -> dict[str, Any]:
        """Analyze agent behavior from round history and trade log."""
        return {
            "decision_patterns": self._decision_patterns(rounds),
            "tool_usage": self._tool_usage(rounds),
            "trade_analysis": self._trade_analysis(trades),
            "latency_stats": self._latency_stats(rounds),
            "round_efficiency": self._round_efficiency(rounds),
        }

    def _decision_patterns(self, rounds: list[AgentRound]) -> dict:
        actions = Counter()
        reasons = []
        for r in rounds:
            actions[r.decision.action] += 1
            if r.decision.reason:
                reasons.append(r.decision.reason[:100])

        return {
            "action_counts": dict(actions),
            "total_decisions": len(rounds),
            "top_reasons": Counter(reasons).most_common(5),
        }

    def _tool_usage(self, rounds: list[AgentRound]) -> dict:
        tool_calls = Counter()
        query_rounds = 0
        for r in rounds:
            if r.decision.action == "query":
                query_rounds += 1
                for q in r.decision.queries:
                    tool_name = q.get("tool", "unknown")
                    tool_calls[tool_name] += 1

        return {
            "query_rounds": query_rounds,
            "tool_calls": dict(tool_calls),
            "avg_queries_per_round": query_rounds / max(len(rounds), 1),
        }

    def _trade_analysis(self, trades: list[TradeResult]) -> dict:
        successful = [t for t in trades if t.success]
        rejected = [t for t in trades if not t.success]

        rejection_reasons = Counter()
        for t in rejected:
            # Extract first part of error
            reason = t.error.split(":")[0] if t.error else "unknown"
            rejection_reasons[reason] += 1

        markets_traded = Counter()
        for t in successful:
            markets_traded[t.order.market.value] += 1

        return {
            "total_orders": len(trades),
            "successful": len(successful),
            "rejected": len(rejected),
            "rejection_rate": len(rejected) / max(len(trades), 1) * 100,
            "rejection_reasons": dict(rejection_reasons),
            "markets_traded": dict(markets_traded),
            "total_fees_usd": sum(
                trade_fees_usd(t, self._fx_rates) for t in successful
            ),
        }

    def _latency_stats(self, rounds: list[AgentRound]) -> dict:
        latencies = [r.latency_ms for r in rounds if r.latency_ms > 0]
        if not latencies:
            return {"avg_ms": 0, "max_ms": 0, "min_ms": 0, "total_ms": 0}

        return {
            "avg_ms": round(sum(latencies) / len(latencies), 1),
            "max_ms": round(max(latencies), 1),
            "min_ms": round(min(latencies), 1),
            "total_ms": round(sum(latencies), 1),
            "total_calls": len(latencies),
        }

    def _round_efficiency(self, rounds: list[AgentRound]) -> dict:
        """How many rounds did the agent need to make a decision?"""
        # Group rounds by timestamp (each timestamp = one decision cycle)
        # For now, just analyze round distribution
        round_counts = Counter()
        for r in rounds:
            round_counts[r.decision.action] += 1

        return {
            "action_distribution": dict(round_counts),
            "total_rounds": len(rounds),
        }
