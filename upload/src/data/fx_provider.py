"""
FxProvider — reads real-time FX rates from FOREX_stock.db.

Configuration (edit these paths/mappings at the top):
"""

from __future__ import annotations
import sqlite3
import os

# ============================================================
# PATH CONFIG — 数据源路径
# ============================================================

STOCK_DATA_ROOT = os.environ.get("STOCK_DATA_ROOT", os.path.expanduser("~/Desktop/getStockData"))
FOREX_DB_PATH = os.path.join(STOCK_DATA_ROOT, "data", "FOREX_stock.db")

# ============================================================
# FX PAIR CONFIG — 货币对映射
# ============================================================

# 可用货币对（EODHD 格式: BASEQUOTE.FOREX）
AVAILABLE_PAIRS = [
    "EURUSD.FOREX",
    "GBPUSD.FOREX",
    "USDJPY.FOREX",
    "USDCHF.FOREX",
    "AUDUSD.FOREX",
    "USDCAD.FOREX",
    "NZDUSD.FOREX",
    "USDCNH.FOREX",   # 美元/离岸人民币
    "USDHKD.FOREX",   # 美元/港币
]

# 默认汇率（数据库查不到时的 fallback）
DEFAULT_RATES = {
    "USD": 1.0,
    "HKD": 7.80,
    "CNY": 7.25,
    "JPY": 155.0,
    "EUR": 0.85,
    "GBP": 0.74,
}


class FxProvider:
    """Provides FX rates from historical 5min data."""

    def __init__(self):
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(f"file:{FOREX_DB_PATH}?mode=ro", uri=True)
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def get_rate(self, from_currency: str, to_currency: str, timestamp: str = "") -> float:
        """Get FX rate: 1 from_currency = X to_currency."""
        if from_currency == to_currency:
            return 1.0

        # Try direct pair (e.g. USDHKD.FOREX)
        pair = f"{from_currency}{to_currency}.FOREX"
        rate = self._get_rate_from_db(pair, timestamp)
        if rate is not None:
            return rate

        # Try inverse pair (e.g. for USD→EUR, try EURUSD and invert)
        pair_inv = f"{to_currency}{from_currency}.FOREX"
        rate_inv = self._get_rate_from_db(pair_inv, timestamp)
        if rate_inv is not None and rate_inv > 0:
            return 1.0 / rate_inv

        # Try via USD (cross rate)
        if from_currency != "USD" and to_currency != "USD":
            rate_from = self.get_rate(from_currency, "USD", timestamp)
            rate_to = self.get_rate("USD", to_currency, timestamp)
            if rate_from > 0 and rate_to > 0:
                return rate_from * rate_to

        # Fallback to defaults
        return DEFAULT_RATES.get(to_currency, 0.0) / DEFAULT_RATES.get(from_currency, 1.0)

    def get_all_rates(self, timestamp: str = "") -> dict[str, float]:
        """Get all USD-based rates at a timestamp. Returns {currency: rate_per_usd}."""
        rates = {"USD": 1.0}
        for currency in ["HKD", "CNY", "JPY", "EUR", "GBP"]:
            rate = self.get_rate("USD", currency, timestamp)
            if rate > 0:
                rates[currency] = rate
            else:
                rates[currency] = DEFAULT_RATES.get(currency, 1.0)
        return rates

    def _get_rate_from_db(self, pair: str, timestamp: str) -> float | None:
        """Get the close price of a FX pair at or before the timestamp."""
        conn = self._get_conn()

        if timestamp:
            ts_clean = timestamp.replace("T", " ").replace("+00:00", "").strip()
            if len(ts_clean) == 16:
                ts_clean += ":00"

            row = conn.execute(
                "SELECT close FROM forex_5min WHERE symbol = ? AND REPLACE(REPLACE(datetime_utc, 'T', ' '), '+00:00', '') <= ? ORDER BY datetime_utc DESC LIMIT 1",
                (pair, ts_clean),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT close FROM forex_5min WHERE symbol = ? ORDER BY datetime_utc DESC LIMIT 1",
                (pair,),
            ).fetchone()

        return row[0] if row else None
