"""Futures candidate bucket construction."""

from __future__ import annotations

from src.core.types import CandidateBucket, CandidateInBucket, Market
from src.data.features import FeatureGenerator
from src.data.futures_resolver import FuturesContractResolver
from src.data.provider import MarketDataProvider


class FuturesCandidateBuilder:
    """Build compact futures_macro rows for prompt context."""

    def __init__(self, data: MarketDataProvider, features: FeatureGenerator, resolver: FuturesContractResolver):
        self._data = data
        self._features = features
        self._resolver = resolver

    def build(self, timestamp: str, nav: float, symbols: list[str] | None = None) -> list[CandidateInBucket]:
        symbols = symbols or ["GC.FUT"]
        rows: list[CandidateInBucket] = []
        for symbol in symbols:
            resolved = self._resolver.resolve(symbol, timestamp)
            if not resolved.contract_ticker or resolved.price is None:
                continue
            bars = self._data.load_futures_bars(symbol, resolved.contract_ticker, "2025-10-01", timestamp)
            snap = self._features.compute(bars, timestamp)
            if snap is None:
                continue
            notional = resolved.notional_per_contract or 0.0
            one_notional_pct = notional / nav if nav > 0 else 0.0
            one_margin_pct = resolved.initial_margin / nav if nav > 0 else 0.0
            risk_note = ""
            if one_notional_pct > 0.5:
                risk_note = "one_contract_large_vs_nav"
            if resolved.roll_status != "normal":
                risk_note = (risk_note + ";" if risk_note else "") + resolved.roll_status
            liquidity_note = f"prev_dollar_vol={resolved.previous_session_dollar_volume or 0:.0f}"
            rows.append(CandidateInBucket(
                bucket=CandidateBucket.BLOCKED_OR_WARNING,
                ticker=symbol,
                market=Market.FUTURES,
                price=snap.price,
                score=max(0.0, snap.recent_score + (1 if snap.trend in ("UU", "UD") else 0)),
                chg_1h=snap.chg_1h,
                chg_1d=snap.chg_1d,
                chg_5d=0.0,
                rsi=snap.rsi,
                trend=snap.trend,
                atr_pct=snap.atr_pct,
                ret_30m=snap.ret_30m,
                rsi_d1h=snap.rsi_d1h,
                trend6=snap.trend6,
                setup=snap.setup,
                recent_score=snap.recent_score,
                asset_type="futures",
                actual_contract=resolved.contract_ticker,
                notional_per_contract=notional,
                initial_margin=resolved.initial_margin,
                maintenance_margin=resolved.maintenance_margin,
                one_contract_notional_pct_nav=one_notional_pct,
                one_contract_margin_pct_nav=one_margin_pct,
                roll_status=resolved.roll_status,
                days_to_expiry=resolved.days_to_expiry or 0,
                liquidity_note=liquidity_note,
                risk_note=risk_note,
            ))
        return rows
