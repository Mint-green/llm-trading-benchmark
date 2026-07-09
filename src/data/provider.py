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
from typing import Any

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
            self._connections[market] = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
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
        symbols = self._filter_tradable_symbols(market, symbols)
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

        allowed_symbols = None
        if market == Market.GOLD:
            allowed_symbols = set(getattr(self._config.gold, "allowed_symbols", ("XAUUSD.FOREX",)))

        result: dict[str, list[OHLCVBar]] = {}
        for row in rows:
            sym, ts_raw, o, h, l, c, v = row
            if allowed_symbols is not None and sym not in allowed_symbols:
                continue
            if c is None or c <= 0:
                continue
            ts = self._normalize_output_ts(ts_raw)
            bar = OHLCVBar(
                timestamp=ts, open=o, high=h, low=l, close=c, volume=v or 0,
            )
            result.setdefault(sym, []).append(bar)

        return result


    def get_last_completed_bar(self, market: Market, symbol: str, timestamp: str) -> OHLCVBar | None:
        """Return latest completed regular-market bar at or before timestamp."""
        conn = self._get_conn(market)
        table = self._config.db_tables[market]
        ts = self._normalize_timestamp(timestamp)
        row = conn.execute(
            f"""
            SELECT datetime_utc, open, high, low, close, volume
            FROM {table}
            WHERE symbol = ?
              AND REPLACE(REPLACE(datetime_utc, 'T', ' '), '+00:00', '') <= ?
              AND close IS NOT NULL AND close > 0
            ORDER BY datetime_utc DESC
            LIMIT 1
            """,
            (symbol, ts),
        ).fetchone()
        if row is None:
            return None
        ts_raw, o, h, l, c, v = row
        return OHLCVBar(self._normalize_output_ts(ts_raw), o, h, l, c, v or 0)

    def get_gold_bid_ask(self, symbol: str, timestamp: str) -> dict[str, OHLCVBar | None]:
        """Return point-in-time bid and ask bars for XAUUSD spot when available."""
        ask_symbol = getattr(self._config.gold, "ask_symbol", f"{symbol}.ASK")
        return {
            "bid": self.get_last_completed_bar(Market.GOLD, symbol, timestamp),
            "ask": self.get_last_completed_bar(Market.GOLD, ask_symbol, timestamp),
        }

    def _filter_tradable_symbols(self, market: Market, symbols: list[str]) -> list[str]:
        if market == Market.GOLD:
            allowed = set(getattr(self._config.gold, "allowed_symbols", ("XAUUSD.FOREX",)))
            return [sym for sym in symbols if sym in allowed]
        return symbols


    def load_futures_contracts(self, continuous_symbol: str) -> list[dict[str, Any]]:
        """Load actual futures contracts for a continuous symbol such as GC.FUT."""
        root = continuous_symbol.split(".")[0]
        conn = self._get_conn(Market.FUTURES)
        rows = conn.execute(
            """
            SELECT symbol, contract_ticker, exchange, roll_type, expiry_date,
                   status, bars_count, date_range
            FROM futures_contracts
            WHERE symbol = ?
            ORDER BY expiry_date, contract_ticker
            """,
            (root,),
        ).fetchall()
        return [
            {
                "root_symbol": row[0],
                "continuous_symbol": continuous_symbol,
                "contract_ticker": row[1],
                "exchange": row[2] or "",
                "roll_type": row[3] or "",
                "expiry_date": row[4],
                "status": row[5] or "",
                "bars_count": row[6] or 0,
                "date_range": row[7] or "",
            }
            for row in rows
        ]

    def load_futures_bars(
        self, continuous_symbol: str, contract_ticker: str,
        start: str, end: str,
    ) -> list[OHLCVBar]:
        """Load bars for one actual futures contract, never a mixed continuous series."""
        conn = self._get_conn(Market.FUTURES)
        table = self._config.db_tables[Market.FUTURES]
        start_dt = self._normalize_timestamp(start)
        end_dt = self._normalize_timestamp(end, end_of_day=True)
        rows = conn.execute(
            f"""
            SELECT datetime_utc, open, high, low, close, volume
            FROM {table}
            WHERE symbol = ?
              AND contract_ticker = ?
              AND REPLACE(REPLACE(datetime_utc, 'T', ' '), '+00:00', '') >= ?
              AND REPLACE(REPLACE(datetime_utc, 'T', ' '), '+00:00', '') <= ?
            ORDER BY datetime_utc
            """,
            (continuous_symbol, contract_ticker, start_dt, end_dt),
        ).fetchall()
        bars: list[OHLCVBar] = []
        for ts_raw, o, h, l, c, v in rows:
            if c is None or c <= 0:
                continue
            bars.append(OHLCVBar(
                timestamp=self._normalize_output_ts(ts_raw),
                open=o, high=h, low=l, close=c, volume=v or 0,
            ))
        return bars

    def get_last_completed_futures_bar(
        self, continuous_symbol: str, contract_ticker: str, timestamp: str,
    ) -> OHLCVBar | None:
        """Return the latest bar at or before timestamp for an actual contract."""
        conn = self._get_conn(Market.FUTURES)
        table = self._config.db_tables[Market.FUTURES]
        ts = self._normalize_timestamp(timestamp)
        row = conn.execute(
            f"""
            SELECT datetime_utc, open, high, low, close, volume
            FROM {table}
            WHERE symbol = ?
              AND contract_ticker = ?
              AND REPLACE(REPLACE(datetime_utc, 'T', ' '), '+00:00', '') <= ?
              AND close IS NOT NULL AND close > 0
            ORDER BY datetime_utc DESC
            LIMIT 1
            """,
            (continuous_symbol, contract_ticker, ts),
        ).fetchone()
        if row is None:
            return None
        ts_raw, o, h, l, c, v = row
        return OHLCVBar(self._normalize_output_ts(ts_raw), o, h, l, c, v or 0)

    def get_next_executable_futures_bar(
        self, continuous_symbol: str, contract_ticker: str, timestamp: str,
    ) -> OHLCVBar | None:
        """Return the first bar strictly after timestamp for next-bar execution."""
        conn = self._get_conn(Market.FUTURES)
        table = self._config.db_tables[Market.FUTURES]
        ts = self._normalize_timestamp(timestamp)
        row = conn.execute(
            f"""
            SELECT datetime_utc, open, high, low, close, volume
            FROM {table}
            WHERE symbol = ?
              AND contract_ticker = ?
              AND REPLACE(REPLACE(datetime_utc, 'T', ' '), '+00:00', '') > ?
              AND close IS NOT NULL AND close > 0
            ORDER BY datetime_utc ASC
            LIMIT 1
            """,
            (continuous_symbol, contract_ticker, ts),
        ).fetchone()
        if row is None:
            return None
        ts_raw, o, h, l, c, v = row
        return OHLCVBar(self._normalize_output_ts(ts_raw), o, h, l, c, v or 0)

    def has_futures_bar_at_or_before(
        self, continuous_symbol: str, contract_ticker: str, timestamp: str,
    ) -> bool:
        return self.get_last_completed_futures_bar(continuous_symbol, contract_ticker, timestamp) is not None

    def get_previous_session_liquidity(
        self, continuous_symbol: str, contract_ticker: str, timestamp: str,
    ) -> tuple[float, float]:
        """Return dollar volume and volume from the latest completed session before timestamp's date."""
        conn = self._get_conn(Market.FUTURES)
        table = self._config.db_tables[Market.FUTURES]
        ts = self._normalize_timestamp(timestamp)
        date_part = ts[:10]
        prev_date = conn.execute(
            f"""
            SELECT MAX(SUBSTR(REPLACE(REPLACE(datetime_utc, 'T', ' '), '+00:00', ''), 1, 10))
            FROM {table}
            WHERE symbol = ? AND contract_ticker = ?
              AND SUBSTR(REPLACE(REPLACE(datetime_utc, 'T', ' '), '+00:00', ''), 1, 10) < ?
            """,
            (continuous_symbol, contract_ticker, date_part),
        ).fetchone()[0]
        if not prev_date:
            return 0.0, 0.0
        row = conn.execute(
            f"""
            SELECT COALESCE(SUM(dollar_volume), 0), COALESCE(SUM(volume), 0)
            FROM {table}
            WHERE symbol = ? AND contract_ticker = ?
              AND SUBSTR(REPLACE(REPLACE(datetime_utc, 'T', ' '), '+00:00', ''), 1, 10) = ?
            """,
            (continuous_symbol, contract_ticker, prev_date),
        ).fetchone()
        return float(row[0] or 0.0), float(row[1] or 0.0)

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
