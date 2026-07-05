"""
Candidate Generation Layer — 8-factor composite scoring + 9-bucket classification.

Scoring factors (designed for style coverage, not strategy injection):
  0.19  Liquidity      — can you trade it (volume_rank, turnover_rank)
  0.14  Momentum       — direction (mixed 1h/1d/5d returns)
  0.14  Volatility     — trading opportunity space (ATR percentile)
  0.10  Reversal       — oversold bounce potential (RSI<35 + negative 5d)
  0.10  Trend          — trend state (UU/UD/DU/DD)
  0.14  Market Activity — is something happening (rel_vol, abnormal_move, sector_heat)
  0.05  Recency        — recent breakout/volume spike
  0.10  Random Explore — injected after deterministic ranking (5-10% replacement)
  0.03  Cost Efficiency — nudge cheaper markets up

Buckets:
  held_positions       — current holdings (always shown)
  exit_watch           — positions with weak signals
  trend_leaders        — strong trend + volume
  pullback_continuation — good trend, short-term pullback
  oversold_reversal    — RI<35, potential bounce
  low_vol_defensive    — low volatility, defensive
  crypto_candidates    — crypto only
  blocked_or_warning   — non-tradable, limit-locked
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Any

from src.core.types import (
    Market, IndicatorSnapshot, OHLCVBar,
    CandidateBucket, CandidateInBucket, CandidateBuckets,
)
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
    recent_bars: str = ""  # last 6 bars' 10-min % changes, e.g. "+0.1 -0.2 +0.3 ..."
    # V4 trend variables
    ret_30m: float = 0.0
    rsi_d1h: float = 0.0
    trend6: str = ""
    setup: str = ""
    recent_score: int = 0


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
        competition_slots: int = 12,
        total_target: int = 30,
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
            quotas = {Market.US: 8, Market.HK: 5, Market.CN: 5, Market.CRYPTO: 2}
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

    # Bucket limits for prompt size control
    BUCKET_LIMITS = {
        "trend_leaders": 5,
        "pullback_continuation": 5,
        "oversold_reversal": 10,
        "low_vol_defensive": 5,
        "crypto_candidates": 8,
        "blocked_or_warning": 5,
    }

    def screen_into_buckets(
        self,
        all_bars: dict[Market, dict[str, list[OHLCVBar]]],
        timestamp: str,
        held_positions: dict[str, dict] | None = None,
        exit_watch_positions: dict[str, dict] | None = None,
        open_markets: list[Market] | None = None,
    ) -> CandidateBuckets:
        """Screen all stocks and classify into 9 buckets.

        Args:
            all_bars: {market: {symbol: [bars]}}
            timestamp: decision timestamp
            held_positions: {symbol: {market, pnl_pct, pct_nav, hold_bars, sellable, plan_status}}
            exit_watch_positions: {symbol: {market, pnl_pct, pct_nav, reason}}
            open_markets: list of open markets (only these appear in new candidates)
        """
        if held_positions is None:
            held_positions = {}
        if exit_watch_positions is None:
            exit_watch_positions = {}
        if open_markets is None:
            open_markets = [Market.US, Market.HK, Market.CN, Market.CRYPTO]

        open_market_set = set(open_markets)

        # Step 1: Score stocks from open markets + held positions from closed markets
        all_scores: list[CandidateScore] = []
        for market, market_bars in all_bars.items():
            is_open = market in open_market_set
            for symbol, bars in market_bars.items():
                # Skip closed market stocks unless they're held positions
                if not is_open and symbol not in held_positions:
                    continue
                score = self._score_stock(market, symbol, bars, timestamp)
                if score is not None:
                    if not is_open:
                        score.tradable = False
                    all_scores.append(score)

        if not all_scores:
            return CandidateBuckets(
                held_positions=[], exit_watch=[], trend_leaders=[],
                pullback_continuation=[], oversold_reversal=[],
                low_vol_defensive=[], crypto_candidates=[], blocked_or_warning=[],
            )

        # Step 2: Compute percentile ranks and composite scores
        self._compute_percentile_ranks(all_scores)
        for s in all_scores:
            s.composite = self._compute_composite(s)

        # Step 3: Classify into buckets
        buckets = self._classify_buckets(
            all_scores, held_positions, exit_watch_positions,
            open_market_set=open_market_set,
        )

        # Step 4: Apply bucket limits
        buckets = self._apply_bucket_limits(buckets)

        return buckets

    def _apply_bucket_limits(self, buckets: CandidateBuckets) -> CandidateBuckets:
        """Apply row limits to each bucket."""
        return CandidateBuckets(
            held_positions=buckets.held_positions,
            exit_watch=buckets.exit_watch[:10],
            trend_leaders=buckets.trend_leaders[:self.BUCKET_LIMITS["trend_leaders"]],
            pullback_continuation=buckets.pullback_continuation[:self.BUCKET_LIMITS["pullback_continuation"]],
            oversold_reversal=buckets.oversold_reversal[:self.BUCKET_LIMITS["oversold_reversal"]],
            low_vol_defensive=buckets.low_vol_defensive[:self.BUCKET_LIMITS["low_vol_defensive"]],
            crypto_candidates=buckets.crypto_candidates[:self.BUCKET_LIMITS["crypto_candidates"]],
            blocked_or_warning=buckets.blocked_or_warning[:self.BUCKET_LIMITS["blocked_or_warning"]],
        )

    def _classify_buckets(
        self,
        all_scores: list[CandidateScore],
        held_positions: dict[str, dict],
        exit_watch_positions: dict[str, dict],
        open_market_set: set[Market] | None = None,
    ) -> CandidateBuckets:
        """Classify scored stocks into 9 buckets.

        Closed market stocks only appear in: held_positions, exit_watch, blocked_or_warning.
        They do NOT appear in: trend_leaders, pullback_continuation, oversold_reversal, low_vol_defensive, crypto_candidates.
        """
        if open_market_set is None:
            open_market_set = {Market.US, Market.HK, Market.CN, Market.CRYPTO}

        # Sort by composite for trend_leaders
        sorted_by_score = sorted(all_scores, key=lambda x: x.composite, reverse=True)

        # Build held_positions bucket
        held_bucket: list[CandidateInBucket] = []
        held_tickers: set[str] = set()
        for sym, info in held_positions.items():
            held_tickers.add(sym)
            held_bucket.append(CandidateInBucket(
                bucket=CandidateBucket.HELD_POSITIONS,
                ticker=sym,
                market=info.get("market", Market.US),
                price=info.get("price", 0.0),
                score=info.get("score", 0.0),
                pnl_pct=info.get("pnl_pct", 0.0),
                pct_nav=info.get("pct_nav", 0.0),
                hold_bars=info.get("hold_bars", 0),
                sellable=info.get("sellable", True),
                tradable=info.get("tradable", True),
                plan_status=info.get("plan_status", ""),
                risk_note=info.get("risk_note", ""),
            ))

        # Build exit_watch bucket
        exit_bucket: list[CandidateInBucket] = []
        exit_tickers: set[str] = set()
        for sym, info in exit_watch_positions.items():
            exit_tickers.add(sym)
            exit_bucket.append(CandidateInBucket(
                bucket=CandidateBucket.EXIT_WATCH,
                ticker=sym,
                market=info.get("market", Market.US),
                price=info.get("price", 0.0),
                score=info.get("score", 0.0),
                pnl_pct=info.get("pnl_pct", 0.0),
                pct_nav=info.get("pct_nav", 0.0),
                reason=info.get("reason", ""),
                allowed_action=info.get("allowed_action", "reduce_or_close"),
            ))

        # Classify non-held stocks into buckets
        trend_bucket: list[CandidateInBucket] = []
        pullback_bucket: list[CandidateInBucket] = []
        oversold_bucket: list[CandidateInBucket] = []
        defensive_bucket: list[CandidateInBucket] = []
        crypto_bucket: list[CandidateInBucket] = []
        blocked_bucket: list[CandidateInBucket] = []

        for s in sorted_by_score:
            if s.ticker in held_tickers or s.ticker in exit_tickers:
                continue

            # Closed market stocks only go to blocked_or_warning
            is_closed_market = s.market not in open_market_set

            # Blocked or warning: non-tradable, limit-locked, OR closed market
            if not s.tradable or s.limit_status != "normal" or is_closed_market:
                reason = "closed_market" if is_closed_market else (
                    f"limit_status={s.limit_status}" if not s.tradable else s.limit_status
                )
                blocked_bucket.append(CandidateInBucket(
                    bucket=CandidateBucket.BLOCKED_OR_WARNING,
                    ticker=s.ticker,
                    market=s.market,
                    price=s.price,
                    score=s.composite,
                    tradable=False if is_closed_market else s.tradable,
                    reason=reason,
                    allowed_action="closed" if is_closed_market else "check_tradability",
                ))
                continue

            # Skip closed market stocks from new candidate buckets
            if is_closed_market:
                continue

            # Crypto candidates
            if s.market == Market.CRYPTO:
                crypto_bucket.append(CandidateInBucket(
                    bucket=CandidateBucket.CRYPTO_CANDIDATES,
                    ticker=s.ticker,
                    market=s.market,
                    price=s.price,
                    score=s.composite,
                    chg_1h=s.chg_1h,
                    chg_1d=s.chg_1d,
                    chg_5d=s.chg_5d,
                    rsi=s.rsi,
                    trend=s.trend,
                    volatility=s.volatility_rank,
                    liquidity=s.volume_rank,
                    ret_30m=s.ret_30m,
                    rsi_d1h=s.rsi_d1h,
                    trend6=s.trend6,
                    setup=s.setup,
                    recent_score=s.recent_score,
                ))
                continue

            # Trend leaders: high score + strong trend
            if s.composite >= 0.5 and s.trend in ("UU", "UD"):
                trend_bucket.append(CandidateInBucket(
                    bucket=CandidateBucket.TREND_LEADERS,
                    ticker=s.ticker,
                    market=s.market,
                    price=s.price,
                    score=s.composite,
                    chg_1h=s.chg_1h,
                    chg_1d=s.chg_1d,
                    chg_5d=s.chg_5d,
                    rsi=s.rsi,
                    trend=s.trend,
                    cost_bps=self._market_cost_bps(s.market),
                    recent_bars=s.recent_bars,
                    ret_30m=s.ret_30m,
                    rsi_d1h=s.rsi_d1h,
                    trend6=s.trend6,
                    setup=s.setup,
                    recent_score=s.recent_score,
                ))
                continue

            # Pullback continuation: good trend but short-term pullback
            if s.trend in ("UU", "UD") and s.chg_1d < 0 and s.rsi > 35:
                pullback_bucket.append(CandidateInBucket(
                    bucket=CandidateBucket.PULLBACK_CONTINUATION,
                    ticker=s.ticker,
                    market=s.market,
                    price=s.price,
                    score=s.composite,
                    chg_1d=s.chg_1d,
                    chg_5d=s.chg_5d,
                    rsi=s.rsi,
                    trend=s.trend,
                    pullback_note=f"1d={s.chg_1d:+.1f}%",
                    ret_30m=s.ret_30m,
                    rsi_d1h=s.rsi_d1h,
                    trend6=s.trend6,
                    setup=s.setup,
                    recent_score=s.recent_score,
                ))
                continue

            # Oversold reversal: RSI < 35, negative 5d
            if s.rsi < 35 and s.chg_5d < 0:
                oversold_bucket.append(CandidateInBucket(
                    bucket=CandidateBucket.OVERSOLD_REVERSAL,
                    ticker=s.ticker,
                    market=s.market,
                    price=s.price,
                    score=s.composite,
                    chg_1d=s.chg_1d,
                    chg_5d=s.chg_5d,
                    rsi=s.rsi,
                    trend=s.trend,
                    stabilization="RSI recovering" if s.rsi > 25 else "deeply oversold",
                    risk_note="High risk — reversal may fail",
                    ret_30m=s.ret_30m,
                    rsi_d1h=s.rsi_d1h,
                    trend6=s.trend6,
                    setup=s.setup,
                    recent_score=s.recent_score,
                ))
                continue

            # Low vol defensive: low ATR, low drawdown
            if s.atr < 0.5:  # ATR% < 0.5%
                defensive_bucket.append(CandidateInBucket(
                    bucket=CandidateBucket.LOW_VOL_DEFENSIVE,
                    ticker=s.ticker,
                    market=s.market,
                    price=s.price,
                    score=s.composite,
                    chg_5d=s.chg_5d,
                    atr_pct=s.atr,
                    drawdown_pct=0.0,  # TODO: compute from peak
                    cost_bps=self._market_cost_bps(s.market),
                    ret_30m=s.ret_30m,
                    rsi_d1h=s.rsi_d1h,
                    trend6=s.trend6,
                    setup=s.setup,
                    recent_score=s.recent_score,
                ))
                continue

            # Default: if good trend, put in trend_leaders; otherwise skip
            if s.composite >= 0.4:
                trend_bucket.append(CandidateInBucket(
                    bucket=CandidateBucket.TREND_LEADERS,
                    ticker=s.ticker,
                    market=s.market,
                    price=s.price,
                    score=s.composite,
                    chg_1h=s.chg_1h,
                    chg_1d=s.chg_1d,
                    chg_5d=s.chg_5d,
                    rsi=s.rsi,
                    trend=s.trend,
                    cost_bps=self._market_cost_bps(s.market),
                    recent_bars=s.recent_bars,
                ))

        # Sort each bucket by score
        trend_bucket.sort(key=lambda x: x.score, reverse=True)
        pullback_bucket.sort(key=lambda x: x.score, reverse=True)
        oversold_bucket.sort(key=lambda x: x.score, reverse=True)
        defensive_bucket.sort(key=lambda x: x.score, reverse=True)
        crypto_bucket.sort(key=lambda x: x.score, reverse=True)

        return CandidateBuckets(
            held_positions=held_bucket,
            exit_watch=exit_bucket,
            trend_leaders=trend_bucket,
            pullback_continuation=pullback_bucket,
            oversold_reversal=oversold_bucket,
            low_vol_defensive=defensive_bucket,
            crypto_candidates=crypto_bucket,
            blocked_or_warning=blocked_bucket,
        )

    @staticmethod
    def _market_cost_bps(market: Market) -> float:
        """Approximate round-trip cost in bps per market."""
        return {
            Market.US: 16.0,   # 3+5+5+3 commission+slippage both sides
            Market.HK: 40.0,   # higher fees + stamp
            Market.CN: 26.0,   # commission + tax
            Market.CRYPTO: 40.0,  # spread + slippage
            Market.GOLD: 10.0,    # spot spread/slippage estimate
        }.get(market, 30.0)

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

        # Recent 6-bar trajectory (10-min % changes)
        recent_bars = self._compute_recent_bars(bars, timestamp, n=6)

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
            recent_bars=recent_bars,
            # V4 trend variables
            ret_30m=snap.ret_30m,
            rsi_d1h=snap.rsi_d1h,
            trend6=snap.trend6,
            setup=snap.setup,
            recent_score=snap.recent_score,
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

        # 9. Cost efficiency (0.03 weight): nudge cheaper markets up
        # US=8bps CN=13bps HK=20bps Crypto=20bps round-trip per side
        cost_map = {"US": 1.0, "CN": 0.6, "HK": 0.2, "CRYPTO": 0.5}
        cost_efficiency = cost_map.get(s.market.value, 0.5)

        # 10. RSI penalty: penalize stocks with RSI > 60 (overbought risk)
        # RSI 30-55 is ideal (penalty=0), RSI 55-65 is acceptable (penalty=0.2), RSI > 65 is risky (penalty=0.5)
        rsi_penalty = 0.0
        if s.rsi > 65:
            rsi_penalty = 0.5  # strong penalty for overbought
        elif s.rsi > 60:
            rsi_penalty = 0.2  # moderate penalty
        elif s.rsi < 30:
            rsi_penalty = 0.3  # penalty for deeply oversold (falling knife risk)

        composite = (
            0.19 * liquidity
            + 0.14 * momentum
            + 0.14 * volatility
            + 0.10 * reversal
            + 0.10 * trend_score
            + 0.14 * market_activity
            + 0.05 * recency
            + 0.03 * cost_efficiency
            - 0.10 * rsi_penalty  # RSI penalty
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

    @staticmethod
    def _compute_recent_bars(bars: list[OHLCVBar], timestamp: str, n: int = 6) -> str:
        """Compute last N bars' 10-min % changes as a compact string.
        Returns e.g. "+0.1 -0.3 +0.2 +0.1 -0.1 +0.4"
        """
        relevant = [b for b in bars if b.timestamp <= timestamp]
        if len(relevant) < n + 1:
            return "N/A"
        recent = relevant[-(n + 1):]
        parts = []
        for i in range(1, len(recent)):
            if recent[i - 1].close > 0:
                chg = (recent[i].close - recent[i - 1].close) / recent[i - 1].close * 100
                parts.append(f"{chg:+.1f}")
            else:
                parts.append("0.0")
        return " ".join(parts)

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
