"""
FeatureGenerator — computes technical indicators from OHLCV bars.

Implements Wilder-smoothed RSI/ATR, EMA, Bollinger Bands, and trend detection.
Online-style: processes bars incrementally via sliding window deques.
"""

from __future__ import annotations
from collections import deque

from src.core.types import OHLCVBar, IndicatorSnapshot
from src.core.interfaces import IFeatureGenerator


class FeatureGenerator(IFeatureGenerator):
    """Computes technical indicators from OHLCV data."""

    def __init__(
        self,
        rsi_period: int = 14,
        atr_period: int = 14,
        ema_short: int = 9,
        ema_long: int = 21,
        bb_period: int = 20,
        bb_std: float = 2.0,
        volume_period: int = 20,
    ):
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.ema_short_period = ema_short
        self.ema_long_period = ema_long
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.volume_period = volume_period

    def compute(self, bars: list[OHLCVBar], timestamp: str) -> IndicatorSnapshot | None:
        """Compute indicators for the latest bar up to timestamp.

        Expects bars sorted by timestamp ascending.
        Returns None if insufficient data.
        """
        # Filter bars up to timestamp
        relevant = [b for b in bars if b.timestamp <= timestamp]
        if len(relevant) < self.rsi_period + 1:
            return None

        closes = [b.close for b in relevant]
        highs = [b.high for b in relevant]
        lows = [b.low for b in relevant]
        volumes = [b.volume for b in relevant]

        latest = relevant[-1]
        price = latest.close

        # 5-minute change
        chg_5m = ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 and closes[-2] != 0 else 0.0

        # 1-hour change (12 bars)
        chg_1h = self._change_pct(closes, 12)

        # 1-day change (~48 bars for equities, ~288 for crypto)
        chg_1d = self._change_pct(closes, 48)

        # RSI (Wilder smoothing)
        rsi = self._compute_rsi(closes)

        # ATR (Wilder smoothing)
        atr = self._compute_atr(highs, lows, closes)
        atr_pct = (atr / price * 100) if price > 0 else 0.0

        # EMA short/long for trend
        ema_s = self._ema(closes, self.ema_short_period)
        ema_l = self._ema(closes, self.ema_long_period)

        # Trend: compare EMA short vs long at last two points
        ema_s_prev = self._ema(closes[:-1], self.ema_short_period)
        ema_l_prev = self._ema(closes[:-1], self.ema_long_period)
        trend_short = "U" if ema_s >= ema_l else "D"
        trend_long = "U" if ema_s_prev >= ema_l_prev else "D"
        trend = trend_long + trend_short  # e.g. "UU", "UD", "DU", "DD"

        # Bollinger Band position
        bb_pos = self._bb_position(closes)

        # Relative volume
        rel_vol = self._relative_volume(volumes)

        # High-low position (intraday)
        hl_pos = self._high_low_position(latest, relevant[-min(48, len(relevant)):])

        return IndicatorSnapshot(
            timestamp=latest.timestamp,
            price=price,
            chg_5m=round(chg_5m, 4),
            chg_1h=round(chg_1h, 4),
            chg_1d=round(chg_1d, 4),
            rel_volume=round(rel_vol, 2),
            rsi=round(rsi, 2),
            atr_pct=round(atr_pct, 4),
            trend=trend,
            bb_position=round(bb_pos, 4),
            high_low_pos=round(hl_pos, 4),
        )

    def compute_batch(
        self, bars: list[OHLCVBar], timestamps: list[str],
    ) -> dict[str, IndicatorSnapshot]:
        """Compute indicators for multiple timestamps efficiently."""
        result = {}
        for ts in timestamps:
            snap = self.compute(bars, ts)
            if snap:
                result[ts] = snap
        return result

    # --- Indicator implementations ---

    @staticmethod
    def _change_pct(closes: list[float], lookback: int) -> float:
        if len(closes) < lookback + 1:
            return 0.0
        prev = closes[-(lookback + 1)]
        if prev == 0:
            return 0.0
        return (closes[-1] - prev) / prev * 100

    def _compute_rsi(self, closes: list[float]) -> float:
        """Wilder-smoothed RSI."""
        if len(closes) < self.rsi_period + 1:
            return 50.0

        # Calculate price changes
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

        # Seed with SMA of first `period` changes
        seed = deltas[:self.rsi_period]
        avg_gain = sum(d for d in seed if d > 0) / self.rsi_period
        avg_loss = sum(-d for d in seed if d < 0) / self.rsi_period

        # Wilder smoothing for remaining
        for d in deltas[self.rsi_period:]:
            gain = d if d > 0 else 0
            loss = -d if d < 0 else 0
            avg_gain = (avg_gain * (self.rsi_period - 1) + gain) / self.rsi_period
            avg_loss = (avg_loss * (self.rsi_period - 1) + loss) / self.rsi_period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _compute_atr(self, highs: list[float], lows: list[float], closes: list[float]) -> float:
        """Wilder-smoothed ATR."""
        if len(closes) < self.atr_period + 1:
            return 0.0

        # True Range
        trs = []
        for i in range(1, len(closes)):
            h, l, pc = highs[i], lows[i], closes[i - 1]
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)

        if len(trs) < self.atr_period:
            return 0.0

        # Seed with SMA
        atr = sum(trs[:self.atr_period]) / self.atr_period

        # Wilder smoothing
        for tr in trs[self.atr_period:]:
            atr = (atr * (self.atr_period - 1) + tr) / self.atr_period

        return atr

    def _ema(self, values: list[float], period: int) -> float:
        """Exponential Moving Average."""
        if len(values) < period:
            return values[-1] if values else 0.0

        # Seed with SMA
        sma = sum(values[:period]) / period
        multiplier = 2.0 / (period + 1)

        ema = sma
        for v in values[period:]:
            ema = (v - ema) * multiplier + ema

        return ema

    def _bb_position(self, closes: list[float]) -> float:
        """Bollinger Band position: 0 = lower band, 1 = upper band."""
        if len(closes) < self.bb_period:
            return 0.5

        window = closes[-self.bb_period:]
        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        std = variance ** 0.5

        if std == 0:
            return 0.5

        upper = mean + self.bb_std * std
        lower = mean - self.bb_std * std
        band_width = upper - lower

        if band_width == 0:
            return 0.5

        pos = (closes[-1] - lower) / band_width
        return max(0.0, min(1.0, pos))

    def _relative_volume(self, volumes: list[float]) -> float:
        """Volume relative to 20-bar average."""
        if len(volumes) < self.volume_period + 1:
            return 1.0

        current = volumes[-1]
        avg = sum(volumes[-(self.volume_period + 1):-1]) / self.volume_period

        if avg == 0:
            return 1.0
        return current / avg

    @staticmethod
    def _high_low_position(latest: OHLCVBar, window: list[OHLCVBar]) -> float:
        """Position within the high-low range of the window."""
        high = max(b.high for b in window)
        low = min(b.low for b in window)

        if high == low:
            return 0.5

        return (latest.close - low) / (high - low)
