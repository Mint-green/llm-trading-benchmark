from __future__ import annotations

import os
import time

import pytest

from src.core.config import Config
from src.core.types import Market, OHLCVBar
from src.data.features import FeatureGenerator
from src.data.futures_candidates import FuturesCandidateBuilder
from src.data.futures_resolver import FuturesContractResolver
from src.data.provider import MarketDataProvider


class LegacyFuturesProvider(MarketDataProvider):
    """Pre-optimization futures SQL retained only for differential testing."""

    def load_futures_bars(self, symbol, contract, start, end):
        conn = self._get_conn(Market.FUTURES)
        table = self._config.db_tables[Market.FUTURES]
        start_dt = self._normalize_timestamp(start)
        end_dt = self._normalize_timestamp(end, end_of_day=True)
        rows = conn.execute(
            f"""
            SELECT datetime_utc, open, high, low, close, volume FROM {table}
            WHERE symbol = ? AND contract_ticker = ?
              AND REPLACE(REPLACE(datetime_utc, 'T', ' '), '+00:00', '') >= ?
              AND REPLACE(REPLACE(datetime_utc, 'T', ' '), '+00:00', '') <= ?
            ORDER BY datetime_utc
            """,
            (symbol, contract, start_dt, end_dt),
        ).fetchall()
        return [
            OHLCVBar(self._normalize_output_ts(ts), o, h, low, close, volume or 0)
            for ts, o, h, low, close, volume in rows
            if close is not None and close > 0
        ]

    def get_last_completed_futures_bar(self, symbol, contract, timestamp):
        conn = self._get_conn(Market.FUTURES)
        table = self._config.db_tables[Market.FUTURES]
        ts = self._normalize_timestamp(timestamp)
        row = conn.execute(
            f"""
            SELECT datetime_utc, open, high, low, close, volume FROM {table}
            WHERE symbol = ? AND contract_ticker = ?
              AND REPLACE(REPLACE(datetime_utc, 'T', ' '), '+00:00', '') <= ?
              AND close IS NOT NULL AND close > 0
            ORDER BY datetime_utc DESC LIMIT 1
            """,
            (symbol, contract, ts),
        ).fetchone()
        if row is None:
            return None
        return OHLCVBar(self._normalize_output_ts(row[0]), *row[1:5], row[5] or 0)

    def get_previous_session_liquidity(self, symbol, contract, timestamp):
        conn = self._get_conn(Market.FUTURES)
        table = self._config.db_tables[Market.FUTURES]
        date = self._normalize_timestamp(timestamp)[:10]
        previous = conn.execute(
            f"""
            SELECT MAX(SUBSTR(REPLACE(REPLACE(datetime_utc, 'T', ' '), '+00:00', ''), 1, 10))
            FROM {table}
            WHERE symbol = ? AND contract_ticker = ?
              AND SUBSTR(REPLACE(REPLACE(datetime_utc, 'T', ' '), '+00:00', ''), 1, 10) < ?
            """,
            (symbol, contract, date),
        ).fetchone()[0]
        if not previous:
            return 0.0, 0.0
        row = conn.execute(
            f"""
            SELECT COALESCE(SUM(dollar_volume), 0), COALESCE(SUM(volume), 0)
            FROM {table}
            WHERE symbol = ? AND contract_ticker = ?
              AND SUBSTR(REPLACE(REPLACE(datetime_utc, 'T', ' '), '+00:00', ''), 1, 10) = ?
            """,
            (symbol, contract, previous),
        ).fetchone()
        return float(row[0] or 0.0), float(row[1] or 0.0)


def _config_with_futures_data() -> Config:
    config = Config.load_from_toml("config/config.toml")
    if not os.path.exists(config.db_paths[Market.FUTURES]):
        pytest.skip("FUTURES_stock.db is unavailable")
    return config


def test_optimized_futures_candidates_match_legacy_queries() -> None:
    config = _config_with_futures_data()
    legacy = LegacyFuturesProvider(config)
    optimized = MarketDataProvider(config)
    try:
        legacy_rows = FuturesCandidateBuilder(
            legacy, FeatureGenerator(), FuturesContractResolver(config, legacy),
        ).build("2026-02-05 14:30", 1_000_000)
        optimized_rows = FuturesCandidateBuilder(
            optimized, FeatureGenerator(), FuturesContractResolver(config, optimized),
        ).build("2026-02-05 14:30", 1_000_000)
        assert optimized_rows == legacy_rows
    finally:
        legacy.close()
        optimized.close()


def test_warm_futures_candidate_build_stays_under_one_second() -> None:
    config = _config_with_futures_data()
    provider = MarketDataProvider(config)
    builder = FuturesCandidateBuilder(
        provider, FeatureGenerator(), FuturesContractResolver(config, provider),
    )
    try:
        builder.build("2026-02-05 14:30", 1_000_000)
        started = time.perf_counter()
        rows = builder.build("2026-02-05 15:00", 1_000_000)
        elapsed = time.perf_counter() - started
        assert rows
        assert elapsed < 1.0
    finally:
        provider.close()
