"""
Futures contract resolution for continuous symbols.

The resolver maps a continuous symbol such as GC.FUT to one actual contract
at a decision timestamp without using future bars or future liquidity.
"""

from __future__ import annotations

from datetime import datetime, date

from src.core.config import Config
from src.core.types import FuturesContractSpec, FuturesResolvedContract
from src.data.provider import MarketDataProvider


class FuturesContractResolver:
    """Point-in-time continuous-to-actual futures contract resolver."""

    def __init__(self, config: Config, data_provider: MarketDataProvider):
        self._config = config
        self._data = data_provider
        self._spec_cache: dict[str, list[FuturesContractSpec]] = {}

    def resolve(self, continuous_symbol: str, timestamp: str) -> FuturesResolvedContract:
        if not self._config.futures.enabled:
            return self._empty(continuous_symbol, timestamp, "expired_or_invalid", "futures_disabled")
        if continuous_symbol not in self._config.futures.allowed_symbols:
            return self._empty(continuous_symbol, timestamp, "expired_or_invalid", "symbol_not_allowed")

        specs = self._load_specs(continuous_symbol)
        ts_date = self._parse_date(timestamp)
        candidates: list[tuple[FuturesContractSpec, int | None, float, float]] = []

        for spec in specs:
            expiry = self._parse_optional_date(spec.expiry_date)
            if expiry is not None and expiry < ts_date:
                continue
            if not self._data.has_futures_bar_at_or_before(continuous_symbol, spec.contract_ticker, timestamp):
                continue
            days_to_expiry = (expiry - ts_date).days if expiry is not None else None
            dollar_volume, volume = self._data.get_previous_session_liquidity(
                continuous_symbol, spec.contract_ticker, timestamp,
            )
            candidates.append((spec, days_to_expiry, dollar_volume, volume))

        if not candidates:
            return self._empty(continuous_symbol, timestamp, "no_active_contract", "no_candidates")

        safe = [c for c in candidates if c[1] is None or c[1] > self._config.futures.roll_days_before_expiry]
        if safe:
            ranked = sorted(safe, key=lambda c: (c[2], c[3], -(c[1] or 9999)), reverse=True)
            chosen, days, dollar_volume, volume = ranked[0]
            roll_status = "normal"
            method = "previous_session_liquidity_safe"
        else:
            ranked = sorted(candidates, key=lambda c: (c[2], c[3], -(c[1] or 9999)), reverse=True)
            chosen, days, dollar_volume, volume = ranked[0]
            roll_status = "forced_near_expiry"
            method = "fallback_liquidity_all_candidates"

        if days is not None and days <= self._config.futures.roll_days_before_expiry and roll_status == "normal":
            roll_status = "near_roll_window"

        bar = self._data.get_last_completed_futures_bar(continuous_symbol, chosen.contract_ticker, timestamp)
        price = bar.close if bar else None
        notional = price * chosen.multiplier if price is not None else None
        return FuturesResolvedContract(
            continuous_symbol=continuous_symbol,
            contract_ticker=chosen.contract_ticker,
            timestamp=timestamp,
            expiry_date=chosen.expiry_date,
            days_to_expiry=days,
            roll_status=roll_status,
            selection_method=method,
            price=price,
            multiplier=chosen.multiplier,
            tick_size=chosen.tick_size,
            tick_value=chosen.tick_value,
            initial_margin=chosen.initial_margin,
            maintenance_margin=chosen.maintenance_margin,
            notional_per_contract=notional,
            previous_session_dollar_volume=dollar_volume,
            previous_session_volume=volume,
        )

    def _load_specs(self, continuous_symbol: str) -> list[FuturesContractSpec]:
        cached = self._spec_cache.get(continuous_symbol)
        if cached is not None:
            return cached
        raw = self._data.load_futures_contracts(continuous_symbol)
        specs = [self._build_spec(r) for r in raw]
        self._spec_cache[continuous_symbol] = specs
        return specs

    def _build_spec(self, row: dict) -> FuturesContractSpec:
        root = row["root_symbol"]
        if root == "GC":
            multiplier = self._config.futures.gc_multiplier
            tick_size = self._config.futures.gc_tick_size
            tick_value = self._config.futures.gc_tick_value
            initial_margin = self._config.futures.gc_initial_margin
            maintenance_margin = self._config.futures.gc_maintenance_margin
        else:
            multiplier = 1.0
            tick_size = 0.01
            tick_value = 0.01
            initial_margin = 0.0
            maintenance_margin = 0.0
        return FuturesContractSpec(
            root_symbol=root,
            continuous_symbol=row["continuous_symbol"],
            contract_ticker=row["contract_ticker"],
            exchange=row.get("exchange", ""),
            multiplier=multiplier,
            tick_size=tick_size,
            tick_value=tick_value,
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            expiry_date=row.get("expiry_date"),
            status=row.get("status"),
            bars_count=row.get("bars_count"),
            date_range=row.get("date_range"),
        )

    @staticmethod
    def _empty(symbol: str, timestamp: str, status: str, method: str) -> FuturesResolvedContract:
        return FuturesResolvedContract(
            continuous_symbol=symbol,
            contract_ticker="",
            timestamp=timestamp,
            roll_status=status,
            selection_method=method,
        )

    @staticmethod
    def _parse_date(timestamp: str) -> date:
        return datetime.strptime(timestamp[:10], "%Y-%m-%d").date()

    @staticmethod
    def _parse_optional_date(value: str | None) -> date | None:
        if not value:
            return None
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
