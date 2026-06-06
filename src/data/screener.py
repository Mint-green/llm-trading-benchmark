"""
Candidate Generation Layer — 8-factor composite scoring + market quota + random explore.

Scoring factors (designed for style coverage, not strategy injection):
  0.20  Liquidity      — can you trade it (volume_rank, turnover_rank)
  0.15  Momentum       — direction (mixed 1h/1d/5d returns)
  0.15  Volatility     — trading opportunity space (ATR percentile)
  0.10  Reversal       — oversold bounce potential (RSI<35 + negative 5d)
  0.10  Trend          — trend state (UU/UD/DU/DD)
  0.15  Market Activity — is something happening (rel_vol, abnormal_move, sector_heat)
  0.05  Recency        — recent breakout/volume spike
  0.10  Random Explore — injected after deterministic ranking (5-10% replacement)
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Any

from src.core.types import Market, IndicatorSnapshot, OHLCVBar
from src.core.interfaces import IFeatureGenerator
from .sectors import get_sector


@dataclass
class CandidateScore:
    """Scoring result for a single stock."""
    ticker: str
    market: Market
    sector: str
    price: float
    chg_1h: float
    chg_1d: float
    chg_5d: float
    volume_rank: float      # 0-1 percentile
    volatility_rank: float  # 0-1 percentile
    atr: float
    rsi: float
    trend: str
    tradable: bool
    limit_status: str
    # Sub-scores
    liquidity: float
    momentum: float
    volatility: float
    reversal: float
    trend_score: float
    market_activity: float
    recency: float
    composite: float


class Screener:
    """8-factor composite screener for Candidate Generation Layer."""

    def __init__(self, features: IFeatureGenerator, seed: int = 42):
        self._features = features
        self._rng = random.Random(seed)

    def screen(
        self,
        all_bars: dict[Market, dict[str, list[OHLCVBar]]],
        timestamp: str,
        held_tickers: dict[str, Market] | None = None,
        quotas: dict[Market, int] | None = None,
        competition_slots: int = 25,
        total_target: int = 65,
    ) -> list[CandidateScore]:
        """Screen all stocks and return ranked candidates.

        Args:
            all_bars: {market: {symbol: [bars]}}
            timestamp: decision timestamp
            held_tickers: {symbol: market} for current holdings (forced into candidates)
            quotas: guaranteed slots per market {Market.US: 15, ...}
            competition_slots: open competition slots (filled by top scorers across all markets)
            total_target: total candidates to return
        """
        if quotas is None:
            quotas = {Market.US: 15, Market.HK: 12, Market.CN: 10, Market.CRYPTO: 3}
        if held_tickers is None:
            held_tickers = {}

        # Step 1: Compute indicators and raw scores for ALL stocks
        all_scores: list[CandidateScore] = []
        for market, market_bars in all_bars.items():
            for symbol, bars in market_bars.items():
                score = self._score_stock(market, symbol, bars, timestamp)
                if score is not None:
                    all_scores.append(score)

        if not all_scores:
            return []

        # Step 2: Compute cross-market percentile ranks
        self._compute_percentile_ranks(all_scores)

        # Step 3: Compute final composite scores
        for s in all_scores:
            s.composite = self._compute_composite(s)

        # Step 4: Allocate by quota + competition
        selected = self._allocate(all_scores, quotas, competition_slots, total_target, held_tickers)

        # Step 5: Random explore — replace 5-10% with random non-selected stocks
        selected = self._random_explore(all_scores, selected, total_target)

        # Step 6: Sort by composite score
        selected.sort(key=lambda x: x.composite, reverse=True)

        return selected

    def _score_stock(
        self, market: Market, symbol: str,
        bars: list[OHLCVBar], timestamp: str,
    ) -> CandidateScore | None:
        """Compute raw scores for a single stock."""
        snap = self._features.compute(bars, timestamp)
        if snap is None:
            return None

        # Compute 5d return
        chg_5d = self._compute_chg(bars, timestamp, 48 * 5)  # ~5 trading days

        # Compute turnover rank proxy (volume * price)
        avg_volume = self._avg_volume(bars, timestamp, 20)
        rel_volume = snap.rel_volume

        # Trend classification
        trend = snap.trend

        # Recency: check for recent breakout or volume spike
        recency = self._compute_recency(bars, timestamp)

        return CandidateScore(
            ticker=symbol,
            market=market,
            sector=get_sector(symbol),
            price=snap.price,
            chg_1h=snap.chg_1h,
            chg_1d=snap.chg_1d,
            chg_5d=chg_5d,
            volume_rank=0.0,  # computed later as percentile
            volatility_rank=0.0,  # computed later as percentile
            atr=snap.atr_pct,
            rsi=snap.rsi,
            trend=trend,
            tradable=True,  # placeholder
            limit_status="normal",  # placeholder
            liquidity=0.0,
            momentum=0.0,
            volatility=0.0,
            reversal=0.0,
            trend_score=0.0,
            market_activity=0.0,
            recency=recency,
            composite=0.0,
        )

    def _compute_percentile_ranks(self, scores: list[CandidateScore]) -> None:
        """Compute cross-market percentile ranks for volume and volatility."""
        # Volume percentile (using rel_volume from indicators)
        volumes = [s.market_activity for s in scores]  # placeholder, will use actual volume
        atrs = [s.atr for s in scores]

        # Sort and assign percentiles
        self._assign_percentile(scores, 'atr', 'volatility_rank')

    def _assign_percentile(self, scores: list[CandidateScore], value_attr: str, rank_attr: str) -> None:
        """Assign percentile rank (0-1) based on a value attribute."""
        sorted_scores = sorted(scores, key=lambda s: getattr(s, value_attr))
        n = len(sorted_scores)
        for i, s in enumerate(sorted_scores):
            setattr(s, rank_attr, i / max(n - 1, 1))

    def _compute_composite(self, s: CandidateScore) -> float:
        """Compute 8-factor composite score."""

        # 1. Liquidity (0.20): 0.7 * volume_rank + 0.3 * turnover_rank
        # Using volume_rank as proxy for both (turnover_rank not available)
        liquidity = s.volume_rank

        # 2. Momentum (0.15): 0.4*1h + 0.3*1d + 0.3*5d
        mom_1h = self._normalize_return(s.chg_1h, -5, 5)
        mom_1d = self._normalize_return(s.chg_1d, -10, 10)
        mom_5d = self._normalize_return(s.chg_5d, -20, 20)
        momentum = 0.4 * mom_1h + 0.3 * mom_1d + 0.3 * mom_5d

        # 3. Volatility (0.15): ATR percentile (already 0-1)
        volatility = s.volatility_rank

        # 4. Reversal (0.10): 0.6*oversold + 0.4*negative_return
        oversold = 0.0
        if s.rsi < 20:
            oversold = 1.0
        elif s.rsi < 35:
            oversold = (35 - s.rsi) / 15  # linear 0-1
        neg_return = self._normalize_return(-s.chg_5d, 0, 20)  # positive = declined
        neg_return = max(0, min(1, neg_return))
        reversal = 0.6 * oversold + 0.4 * neg_return

        # 5. Trend (0.10): UU=1.0, UD=0.7, DU=0.3, DD=0.0
        trend_map = {"UU": 1.0, "UD": 0.7, "DU": 0.3, "DD": 0.0}
        trend_score = trend_map.get(s.trend, 0.5)

        # 6. Market Activity (0.15): 0.5*rel_vol + 0.3*|1d_return| + 0.2*sector_heat
        # rel_volume not stored in CandidateScore, use volume_rank as proxy
        abnormal = abs(s.chg_1d) / 10  # normalize
        abnormal = min(1, abnormal)
        sector_heat = 0.5  # placeholder (no sector data)
        market_activity = 0.5 * s.volume_rank + 0.3 * abnormal + 0.2 * sector_heat

        # 7. Recency (0.05): already computed
        recency = s.recency

        # 8. Random (0.10): added later in _random_explore
        # Not included in deterministic composite

        composite = (
            0.20 * liquidity
            + 0.15 * momentum
            + 0.15 * volatility
            + 0.10 * reversal
            + 0.10 * trend_score
            + 0.15 * market_activity
            + 0.05 * recency
            # random 0.10 added during allocation
        )

        # Store sub-scores for debugging
        s.liquidity = liquidity
        s.momentum = momentum
        s.volatility = volatility
        s.reversal = reversal
        s.trend_score = trend_score
        s.market_activity = market_activity
        s.recency = recency

        return composite

    def _allocate(
        self, all_scores: list[CandidateScore],
        quotas: dict[Market, int],
        competition_slots: int,
        total_target: int,
        held_tickers: dict[str, Market] | None = None,
    ) -> list[CandidateScore]:
        """Allocate candidates: holdings (forced) + guaranteed quotas + open competition."""
        selected: list[CandidateScore] = []
        selected_tickers: set[str] = set()

        # Sort all by composite
        all_scores.sort(key=lambda x: x.composite, reverse=True)

        # Step 0: Force current holdings into candidates (not counted against quota)
        if held_tickers:
            for s in all_scores:
                if s.ticker in held_tickers and s.ticker not in selected_tickers:
                    s.sector = s.sector or "HELD"  # mark as held
                    selected.append(s)
                    selected_tickers.add(s.ticker)

        # Step 1: Fill guaranteed quotas per market
        for market, quota in quotas.items():
            market_stocks = [s for s in all_scores if s.market == market and s.ticker not in selected_tickers]
            for s in market_stocks[:quota]:
                selected.append(s)
                selected_tickers.add(s.ticker)

        # Step 2: Fill remaining slots with top scorers across all markets
        remaining = total_target - len(selected)
        for s in all_scores:
            if remaining <= 0:
                break
            if s.ticker not in selected_tickers:
                selected.append(s)
                selected_tickers.add(s.ticker)
                remaining -= 1

        return selected

    def _random_explore(
        self, all_scores: list[CandidateScore],
        selected: list[CandidateScore],
        total_target: int,
    ) -> list[CandidateScore]:
        """Replace 5-10% of candidates with random non-selected stocks."""
        selected_tickers = {s.ticker for s in selected}
        non_selected = [s for s in all_scores if s.ticker not in selected_tickers]

        if not non_selected:
            return selected

        # Replace 5-10% (round to at least 1)
        n_replace = max(1, int(len(selected) * self._rng.uniform(0.05, 0.10)))
        n_replace = min(n_replace, len(non_selected))

        # Remove last N from selected (lowest scores)
        selected = selected[:-n_replace] if n_replace < len(selected) else selected

        # Add random picks
        random_picks = self._rng.sample(non_selected, n_replace)
        selected.extend(random_picks)

        return selected

    # --- Helper methods ---

    @staticmethod
    def _normalize_return(value: float, min_val: float, max_val: float) -> float:
        """Normalize a return value to 0-1 range."""
        if max_val == min_val:
            return 0.5
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    @staticmethod
    def _compute_chg(bars: list[OHLCVBar], timestamp: str, lookback: int) -> float:
        """Compute change over lookback bars."""
        relevant = [b for b in bars if b.timestamp <= timestamp]
        if len(relevant) < lookback + 1:
            return 0.0
        current = relevant[-1].close
        past = relevant[-(lookback + 1)].close
        if past <= 0:
            return 0.0
        return (current - past) / past * 100

    @staticmethod
    def _avg_volume(bars: list[OHLCVBar], timestamp: str, period: int) -> float:
        """Compute average volume over last N bars."""
        relevant = [b for b in bars if b.timestamp <= timestamp]
        if len(relevant) < period:
            return 0.0
        return sum(b.volume for b in relevant[-period:]) / period

    def _compute_recency(self, bars: list[OHLCVBar], timestamp: str) -> float:
        """Compute recency score: recent breakout or volume spike.

        Checks last 10 bars for:
        - New 20-bar high (breakout)
        - Volume > 2x average (volume spike)
        """
        relevant = [b for b in bars if b.timestamp <= timestamp]
        if len(relevant) < 30:
            return 0.0

        last_10 = relevant[-10:]
        prev_20 = relevant[-30:-10]

        prev_high = max(b.high for b in prev_20)
        avg_vol = sum(b.volume for b in prev_20) / len(prev_20) if prev_20 else 1

        score = 0.0

        # Breakout: any bar in last 10 exceeded previous 20-bar high
        for b in last_10:
            if b.high > prev_high * 1.01:  # 1% above previous high
                score += 0.5
                break

        # Volume spike: any bar in last 10 had > 2x average volume
        for b in last_10:
            if avg_vol > 0 and b.volume > avg_vol * 2:
                score += 0.5
                break

        return min(1.0, score)
