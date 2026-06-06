"""
MarketDataProvider — reads OHLCV data from getStockData SQLite databases.

Handles schema differences between A_stock (baostock) and US/HK/CRYPTO (EODHD).
Read-only access to source databases.

PATH CONFIG: paths are defined in Config (core/config.py):
  - STOCK_DATA_DIR = "D:/Projects/claw/getStockData/data"
  - DB_PATHS: {Market: "path/to/db"}
  - DB_TABLES: {Market: "table_name"}
"""

from __future__ import annotations
import sqlite3
from datetime import datetime

from src.core.types import Market, OHLCVBar
from src.core.interfaces import IMarketDataProvider
from src.core.config import Config


class MarketDataProvider(IMarketDataProvider):
    """Reads 5-minute OHLCV data from getStockData SQLite databases."""

    def __init__(self, config: Config):
        self._config = config
        self._connections: dict[Market, sqlite3.Connection] = {}
        self._universe_cache: dict[Market, list[str]] = {}

    def _get_conn(self, market: Market) -> sqlite3.Connection:
        if market not in self._connections:
            path = self._config.db_paths[market]
            self._connections[market] = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        return self._connections[market]

    def close(self) -> None:
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()

    def get_universe_symbols(self, market: Market) -> list[str]:
        if market in self._universe_cache:
            return self._universe_cache[market]

        conn = self._get_conn(market)
        table = self._config.db_tables[market]
        rows = conn.execute(
            f"SELECT DISTINCT symbol FROM {table} ORDER BY symbol"
        ).fetchall()
        symbols = [r[0] for r in rows]
        self._universe_cache[market] = symbols
        return symbols

    def load_bars(
        self, market: Market, symbol: str,
        start: str, end: str,
    ) -> list[OHLCVBar]:
        conn = self._get_conn(market)
        table = self._config.db_tables[market]

        # Normalize datetime format for comparison
        # CN data has "T" separator and "+00:00" suffix; strip for comparison
        start_dt = self._normalize_timestamp(start)
        end_dt = self._normalize_timestamp(end, end_of_day=True)

        query = f"""
            SELECT datetime_utc, open, high, low, close, volume
            FROM {table}
            WHERE symbol = ?
              AND REPLACE(REPLACE(datetime_utc, 'T', ' '), '+00:00', '') >= ?
              AND REPLACE(REPLACE(datetime_utc, 'T', ' '), '+00:00', '') <= ?
            ORDER BY datetime_utc
        """
        rows = conn.execute(query, (symbol, start_dt, end_dt)).fetchall()

        bars = []
        for row in rows:
            ts_raw, o, h, l, c, v = row
            if c is None or c <= 0:
                continue
            ts = self._normalize_output_ts(ts_raw)
            bars.append(OHLCVBar(
                timestamp=ts, open=o, high=h, low=l, close=c, volume=v or 0,
            ))
        return bars

    def load_all_bars(
        self, market: Market, start: str, end: str,
    ) -> dict[str, list[OHLCVBar]]:
        conn = self._get_conn(market)
        table = self._config.db_tables[market]

        start_dt = self._normalize_timestamp(start)
        end_dt = self._normalize_timestamp(end, end_of_day=True)

        query = f"""
            SELECT symbol, datetime_utc, open, high, low, close, volume
            FROM {table}
            WHERE REPLACE(REPLACE(datetime_utc, 'T', ' '), '+00:00', '') >= ?
              AND REPLACE(REPLACE(datetime_utc, 'T', ' '), '+00:00', '') <= ?
            ORDER BY symbol, datetime_utc
        """
        rows = conn.execute(query, (start_dt, end_dt)).fetchall()

        result: dict[str, list[OHLCVBar]] = {}
        for row in rows:
            sym, ts_raw, o, h, l, c, v = row
            if c is None or c <= 0:
                continue
            ts = self._normalize_output_ts(ts_raw)
            bar = OHLCVBar(
                timestamp=ts, open=o, high=h, low=l, close=c, volume=v or 0,
            )
            result.setdefault(sym, []).append(bar)

        return result

    @staticmethod
    def _normalize_timestamp(ts: str, end_of_day: bool = False) -> str:
        """Normalize timestamp to a format SQLite can compare.

        Input can be: '2026-01-05', '2026-01-05 01:30', '2026-01-05T01:30:00+00:00'
        Output: 'YYYY-MM-DD HH:MM:SS+00:00' or 'YYYY-MM-DD HH:MM:SS'
        """
        if "T" in ts and "+" in ts:
            return ts  # already ISO format with timezone
        if "T" in ts:
            return ts.replace("T", " ")
        if len(ts) == 10:  # date only
            return ts + (" 23:59:59" if end_of_day else " 00:00:00")
        if len(ts) == 16:  # "YYYY-MM-DD HH:MM"
            return ts + ":00"
        return ts

    @staticmethod
    def _normalize_output_ts(ts_raw: str) -> str:
        """Normalize DB timestamp to consistent output format.

        Input: '2025-10-09T01:35:00+00:00' or '2025-10-02 01:30:00+00:00'
        Output: 'YYYY-MM-DD HH:MM' (without seconds, without timezone)
        """
        # Strip timezone suffix
        s = ts_raw.replace("+00:00", "").strip()
        # Handle T separator
        s = s.replace("T", " ")
        # Truncate to "YYYY-MM-DD HH:MM"
        if len(s) >= 16:
            return s[:16]
        return s
