"""
Leaderboard — ranks benchmark results.

Two ranking modes:
  1. Profit Leaderboard: sorted by total return
  2. Composite Leaderboard: weighted score

Composite weights (from design doc):
  - Return: 45%
  - Risk-adjusted (Sharpe): 20%
  - Drawdown: 15%
  - Stability (1/volatility): 10%
  - Efficiency (1/turnover): 5%
  - Discipline (win rate): 5%
"""

from __future__ import annotations

from src.core.types import BenchmarkResult
from src.core.interfaces import ILeaderboard


class Leaderboard(ILeaderboard):
    """Ranks benchmark results."""

    def rank(self, results: list[BenchmarkResult]) -> list[dict]:
        """Rank results by composite score."""
        if not results:
            return []

        scored = []
        for r in results:
            score = self._composite_score(r)
            scored.append({
                "model": r.model_name,
                "total_return": r.total_return,
                "sharpe": r.sharpe_ratio,
                "max_drawdown": r.max_drawdown,
                "win_rate": r.win_rate,
                "composite_score": round(score, 4),
                "dataset_version": r.dataset_version,
            })

        scored.sort(key=lambda x: x["composite_score"], reverse=True)
        for i, entry in enumerate(scored):
            entry["rank"] = i + 1

        return scored

    def rank_by_return(self, results: list[BenchmarkResult]) -> list[dict]:
        """Rank by total return only."""
        sorted_results = sorted(results, key=lambda r: r.total_return, reverse=True)
        return [
            {
                "rank": i + 1,
                "model": r.model_name,
                "total_return": r.total_return,
                "dataset_version": r.dataset_version,
            }
            for i, r in enumerate(sorted_results)
        ]

    @staticmethod
    def _composite_score(r: BenchmarkResult) -> float:
        """Compute weighted composite score."""
        # Normalize each metric to [0, 1] range (approximate)
        # These are rough normalization ranges based on typical market performance

        # Return: -50% to +100% → 0 to 1
        return_score = max(0, min(1, (r.total_return + 50) / 150))

        # Sharpe: -2 to +4 → 0 to 1
        sharpe_score = max(0, min(1, (r.sharpe_ratio + 2) / 6))

        # Max drawdown: 0% to 50% → 1 to 0 (lower is better)
        dd_score = max(0, 1 - r.max_drawdown / 50)

        # Win rate: already 0-100
        win_score = r.win_rate / 100

        # Stability: inverse of volatility (capped at 100%)
        vol_score = max(0, 1 - min(r.max_drawdown, 100) / 100)

        # Efficiency: inverse of turnover (rough)
        eff_score = max(0, 1 - min(r.total_trades / 1000, 1))

        # Weighted sum
        return (
            0.45 * return_score
            + 0.20 * sharpe_score
            + 0.15 * dd_score
            + 0.10 * vol_score
            + 0.05 * eff_score
            + 0.05 * win_score
        )
