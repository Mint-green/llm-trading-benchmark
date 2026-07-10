"""Futures candidate bucket construction."""

from __future__ import annotations

import math

from src.core.futures_specs import (
    get_futures_family_spec,
    get_futures_product_spec,
    is_futures_family_symbol,
)
from src.core.types import CandidateBucket, CandidateInBucket, Market, FuturesResolvedContract
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
        symbols = symbols or list(self._resolver._config.futures.allowed_symbols)
        rows: list[CandidateInBucket] = []
        for symbol in symbols:
            if is_futures_family_symbol(symbol):
                row = self._build_family_row(symbol, timestamp, nav)
                if row is not None:
                    rows.append(row)
            else:
                row = self._build_contract_row(symbol, timestamp, nav)
                if row is not None:
                    rows.append(row)
        return rows

    def _build_contract_row(self, symbol: str, timestamp: str, nav: float) -> CandidateInBucket | None:
        resolved = self._resolver.resolve(symbol, timestamp)
        if not resolved.contract_ticker or resolved.price is None:
            return None
        snap = self._snapshot(symbol, resolved.contract_ticker, timestamp)
        if snap is None:
            return None
        notional = resolved.notional_per_contract or 0.0
        one_notional_pct = notional / nav if nav > 0 else 0.0
        one_margin_pct = resolved.initial_margin / nav if nav > 0 else 0.0
        product = get_futures_product_spec(symbol)
        notes = []
        if product is not None:
            notes.append(product.macro_role)
        notes.extend(self._risk_notes(one_notional_pct, one_margin_pct, resolved.roll_status))
        return CandidateInBucket(
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
            liquidity_note=f"prev_dollar_vol={resolved.previous_session_dollar_volume or 0:.0f}",
            risk_note=";".join(notes),
            signal_symbol=symbol,
        )

    def _build_family_row(self, family_symbol: str, timestamp: str, nav: float) -> CandidateInBucket | None:
        family = get_futures_family_spec(family_symbol)
        if family is None:
            return None

        variants = []
        for variant_symbol in family.variants:
            resolved = self._resolver.resolve(variant_symbol, timestamp)
            if not resolved.contract_ticker or resolved.price is None:
                variants.append((variant_symbol, resolved, None, "not_tradable"))
                continue
            variants.append((variant_symbol, resolved, self._snapshot(variant_symbol, resolved.contract_ticker, timestamp), ""))

        resolved_variants = [(sym, res, snap) for sym, res, snap, note in variants if snap is not None]
        if not resolved_variants:
            return None

        signal_symbol, signal_resolved, signal_snap = max(
            resolved_variants,
            key=lambda item: (item[1].previous_session_dollar_volume or 0.0, item[1].previous_session_volume or 0.0),
        )

        standard = self._variant_summary(family.standard_symbol, variants, nav) if family.standard_symbol else "n/a"
        micro = self._variant_summary(family.micro_symbol, variants, nav) if family.micro_symbol else "n/a"
        guidance = self._family_guidance(family.standard_symbol, family.micro_symbol, variants, nav)
        pilot_target_pct = self._pilot_target_pct_nav(variants, nav)

        one_notional_pct = (signal_resolved.notional_per_contract or 0.0) / nav if nav > 0 else 0.0
        one_margin_pct = signal_resolved.initial_margin / nav if nav > 0 else 0.0
        notes = [family.macro_role]
        notes.extend(self._risk_notes(one_notional_pct, one_margin_pct, signal_resolved.roll_status))

        return CandidateInBucket(
            bucket=CandidateBucket.BLOCKED_OR_WARNING,
            ticker=family.family_symbol,
            market=Market.FUTURES,
            price=signal_snap.price,
            score=max(0.0, signal_snap.recent_score + (1 if signal_snap.trend in ("UU", "UD") else 0)),
            chg_1h=signal_snap.chg_1h,
            chg_1d=signal_snap.chg_1d,
            chg_5d=0.0,
            rsi=signal_snap.rsi,
            trend=signal_snap.trend,
            atr_pct=signal_snap.atr_pct,
            ret_30m=signal_snap.ret_30m,
            rsi_d1h=signal_snap.rsi_d1h,
            trend6=signal_snap.trend6,
            setup=signal_snap.setup,
            recent_score=signal_snap.recent_score,
            asset_type="futures_family",
            actual_contract=signal_resolved.contract_ticker,
            notional_per_contract=signal_resolved.notional_per_contract or 0.0,
            initial_margin=signal_resolved.initial_margin,
            maintenance_margin=signal_resolved.maintenance_margin,
            one_contract_notional_pct_nav=one_notional_pct,
            one_contract_margin_pct_nav=one_margin_pct,
            roll_status=signal_resolved.roll_status,
            days_to_expiry=signal_resolved.days_to_expiry or 0,
            liquidity_note=f"signal={signal_symbol};prev_dollar_vol={signal_resolved.previous_session_dollar_volume or 0:.0f}",
            risk_note=";".join(notes),
            signal_symbol=signal_symbol,
            standard_variant=standard,
            micro_variant=micro,
            execution_guidance=guidance,
            pilot_target_pct_nav=pilot_target_pct,
        )

    def _snapshot(self, symbol: str, contract_ticker: str, timestamp: str):
        bars = self._data.load_futures_bars(symbol, contract_ticker, "2025-10-01", timestamp)
        return self._features.compute(bars, timestamp)

    def _variant_summary(self, symbol: str | None, variants: list[tuple[str, FuturesResolvedContract, object, str]], nav: float) -> str:
        if not symbol:
            return "n/a"
        found = next((item for item in variants if item[0] == symbol), None)
        if found is None:
            return "n/a"
        _, resolved, snap, note = found
        product = get_futures_product_spec(symbol)
        variant = product.variant if product is not None else "unknown"
        if snap is None or not resolved.contract_ticker or not resolved.notional_per_contract:
            return f"{symbol}:{variant}:not_tradable:{resolved.selection_method or note}"
        notional_pct = resolved.notional_per_contract / nav if nav > 0 else 0.0
        margin_pct = resolved.initial_margin / nav if nav > 0 else 0.0
        max_contracts = self._max_contracts_allowed(resolved, nav)
        notes = self._risk_notes(notional_pct, margin_pct, resolved.roll_status)
        risk = ",".join(notes) if notes else "ok"
        return (
            f"{symbol}:{variant}:{resolved.contract_ticker}:"
            f"{notional_pct*100:.1f}%NAV:{margin_pct*100:.1f}%margin:max{max_contracts}:{risk}"
        )

    def _pilot_target_pct_nav(self, variants: list[tuple[str, FuturesResolvedContract, object, str]], nav: float) -> float:
        notionals = [
            res.notional_per_contract
            for sym, res, snap, note in variants
            if snap is not None and res.notional_per_contract and nav > 0
        ]
        if not notionals:
            return 0.0
        raw = min(notionals) / nav * 1.05
        if raw > 0.10:
            return 0.0
        return max(0.01, math.ceil(raw * 100) / 100)

    def _family_guidance(self, standard_symbol: str | None, micro_symbol: str | None, variants: list[tuple[str, FuturesResolvedContract, object, str]], nav: float) -> str:
        if not micro_symbol:
            return "standard_only;do_not_split_family_view"
        std = next((res for sym, res, snap, note in variants if sym == standard_symbol and snap is not None), None)
        micro = next((res for sym, res, snap, note in variants if sym == micro_symbol and snap is not None), None)
        if micro is None:
            return "standard_only_micro_unavailable;do_not_split_family_view"
        if std is None or not std.notional_per_contract:
            return "micro_available;do_not_split_family_view"
        threshold = (std.notional_per_contract / nav) * 0.8 if nav > 0 else 0.0
        return f"auto_size;prefer_micro_below_{threshold*100:.1f}%target;do_not_split_family_view"

    def _max_contracts_allowed(self, resolved: FuturesResolvedContract, nav: float) -> int:
        if nav <= 0 or not resolved.notional_per_contract or resolved.initial_margin <= 0:
            return 0
        by_symbol = self._resolver._config.futures.max_contracts_per_symbol
        by_notional = math.floor(nav * self._resolver._config.futures.max_abs_notional_pct_nav / resolved.notional_per_contract)
        by_margin = math.floor(nav * self._resolver._config.futures.max_margin_pct_nav / resolved.initial_margin)
        return max(0, min(by_symbol, by_notional, by_margin))

    def _risk_notes(self, one_notional_pct: float, one_margin_pct: float, roll_status: str) -> list[str]:
        notes: list[str] = []
        if one_notional_pct > self._resolver._config.futures.max_abs_notional_pct_nav:
            notes.append("one_contract_exceeds_notional_cap")
        elif one_notional_pct > 0.5:
            notes.append("large_but_allowed_if_setup_strong")
        if one_margin_pct > self._resolver._config.futures.max_margin_pct_nav:
            notes.append("one_contract_exceeds_margin_cap")
        if roll_status != "normal":
            notes.append(roll_status)
        return notes
