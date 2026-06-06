"""
IndexProvider — reads index/ETF data for market summary.

Data sources:
  - FOREX_stock.db → index_5min table (NDX.INDX, HSI.INDX)
  - A_stock.db → stock_5min table (sh.510300 CSI300 ETF)

Configuration (edit these paths/mappings at the top):
"""

from __future__ import annotations
import sqlite3
import os
from src.core.types import Market

# ============================================================
# PATH CONFIG — 数据源路径（修改这里适配不同环境）
# ============================================================

# getStockData 项目根目录
STOCK_DATA_ROOT = "D:/Projects/claw/getStockData"

# 数据库路径
FOREX_DB_PATH = os.path.join(STOCK_DATA_ROOT, "data", "FOREX_stock.db")
A_STOCK_DB_PATH = os.path.join(STOCK_DATA_ROOT, "data", "A_stock.db")

# ============================================================
# INDEX MAPPING — 市场→指数/ETF 映射（修改这里更换指数）
# ============================================================

INDEX_CONFIG: dict[Market, dict] = {
    Market.US: {
        "symbol": "NDX.INDX",          # NASDAQ-100（无标普500数据时用此代替）
        "source": "forex_db",           # forex_db = FOREX_stock.db / index_5min
        "table": "index_5min",
        "name": "NDX",
    },
    Market.HK: {
        "symbol": "HSI.INDX",           # 恒生指数
        "source": "forex_db",
        "table": "index_5min",
        "name": "HSI",
    },
    Market.CN: {
        "symbol": "sh.510300",          # 沪深300 ETF（覆盖2026年1月至今）
        "source": "a_stock_db",         # a_stock_db = A_stock.db / stock_5min
        "table": "stock_5min",
        "name": "CSI300",
    },
}

# Crypto 无指数数据
CRYPTO_INDEX_NAME = "BTC"


class IndexProvider:
    """Provides index/ETF price data for market summary."""

    def __init__(self):
        self._conns: dict[str, sqlite3.Connection] = {}

    def _get_conn(self, source: str) -> sqlite3.Connection:
        if source not in self._conns:
            if source == "forex_db":
                self._conns[source] = sqlite3.connect(f"file:{FOREX_DB_PATH}?mode=ro", uri=True)
            elif source == "a_stock_db":
                self._conns[source] = sqlite3.connect(f"file:{A_STOCK_DB_PATH}?mode=ro", uri=True)
        return self._conns[source]

    def close(self) -> None:
        for conn in self._conns.values():
            conn.close()
        self._conns.clear()

    def get_index_return(self, market: Market, timestamp: str, lookback_minutes: int) -> float | None:
        """Get index return over lookback_minutes at the given timestamp."""
        cfg = INDEX_CONFIG.get(market)
        if not cfg:
            return None

        conn = self._get_conn(cfg["source"])
        ts_clean = self._normalize_ts(timestamp)

        # Get current price
        current = self._query_price(conn, cfg["table"], cfg["symbol"], ts_clean)
        if not current or current <= 0:
            return None

        # Get price N minutes ago
        from datetime import datetime, timedelta
        try:
            dt = datetime.strptime(ts_clean[:19], "%Y-%m-%d %H:%M:%S")
            past_ts = (dt - timedelta(minutes=lookback_minutes)).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

        past = self._query_price(conn, cfg["table"], cfg["symbol"], past_ts)
        if not past or past <= 0:
            return None

        return (current - past) / past * 100

    def get_all_index_returns(self, timestamp: str) -> dict[Market, dict]:
        """Get Index1H and Index1D for all markets."""
        result = {}
        for market, cfg in INDEX_CONFIG.items():
            ret_1h = self.get_index_return(market, timestamp, 60)
            ret_1d = self.get_index_return(market, timestamp, 1440)
            result[market] = {
                "symbol": cfg["name"],
                "return_1h": ret_1h,
                "return_1d": ret_1d,
            }
        return result

    def _query_price(self, conn: sqlite3.Connection, table: str, symbol: str, ts: str) -> float | None:
        """Get close price at or before timestamp."""
        row = conn.execute(
            f"SELECT close FROM {table} WHERE symbol = ? AND REPLACE(REPLACE(datetime_utc, 'T', ' '), '+00:00', '') <= ? ORDER BY datetime_utc DESC LIMIT 1",
            (symbol, ts),
        ).fetchone()
        return row[0] if row else None

    @staticmethod
    def _normalize_ts(ts: str) -> str:
        s = ts.replace("T", " ").replace("+00:00", "").strip()
        if len(s) == 16:
            s += ":00"
        return s
