"""
Shared types, enums, and data structures for the benchmark system.

All modules use these types for consistency. No business logic here.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================
# Enums
# ============================================================

class Market(str, Enum):
    US = "US"
    HK = "HK"
    CN = "CN"
    CRYPTO = "CRYPTO"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"


class AssetStatus(str, Enum):
    ACTIVE = "active"
    HALTED = "halted"
    DELISTED = "delisted"
    PENDING_DELIST = "pending_delist"


class LimitStatus(str, Enum):
    NORMAL = "normal"
    NEAR_LIMIT_UP = "near_limit_up"
    LIMIT_UP = "limit_up"
    NEAR_LIMIT_DOWN = "near_limit_down"
    LIMIT_DOWN = "limit_down"


class Tradability(str, Enum):
    TRADABLE = "tradable"
    NOT_TRADABLE = "not_tradable"
    BUY_ONLY = "buy_only"    # e.g. limit-down: can buy, cannot sell
    SELL_ONLY = "sell_only"  # e.g. limit-up: can sell, cannot buy


# ============================================================
# Market configuration dataclasses
# ============================================================

@dataclass(frozen=True)
class MarketHours:
    """Trading hours in UTC. Supports lunch breaks."""
    market: Market
    sessions: list[tuple[str, str]]  # list of (open_time, close_time) in HH:MM UTC
    timezone: str                     # display timezone name (e.g. "EST", "HKT")


@dataclass(frozen=True)
class PositionLimit:
    """Position limit configuration."""
    max_single_position: float    # fraction of NAV (e.g. 0.25 = 25%)
    max_market_exposure: float    # fraction of NAV
    max_crypto_exposure: float    # fraction of NAV
    min_cash_ratio: float         # fraction of NAV


# ============================================================
# Asset data
# ============================================================

@dataclass
class AssetInfo:
    """Basic asset information from universe registry."""
    ticker: str           # e.g. "AAPL.US", "0700.HK", "sh.600519"
    name: str
    market: Market
    sector: str = ""
    asset_class: str = "equity"  # equity, crypto, commodity_etf


@dataclass
class OHLCVBar:
    """Single 5-minute OHLCV bar."""
    timestamp: str        # UTC ISO format
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float = 0.0   # turnover (CN market has this)


@dataclass
class IndicatorSnapshot:
    """Computed technical indicators for a single bar."""
    timestamp: str
    price: float
    chg_5m: float         # 5-minute change %
    chg_1h: float         # 1-hour change %
    chg_1d: float         # 1-day change %
    rel_volume: float     # relative volume vs 20-bar avg
    rsi: float
    atr_pct: float        # ATR as % of price
    trend: str            # "UU", "UD", "DU", "DD"
    bb_position: float    # Bollinger Band position [0,1]
    high_low_pos: float   # intraday high-low position [0,1]


# ============================================================
# Trading data
# ============================================================

@dataclass
class TradeOrder:
    """A trade order submitted by the agent."""
    symbol: str
    market: Market
    side: OrderSide
    quantity: int = 0            # shares/units (resolved from allocation_pct)
    allocation_pct: float | None = None  # fraction of NAV (0.05 = 5%)
    reason: str = ""
    order_type: OrderType = OrderType.MARKET


@dataclass
class TradeResult:
    """Result of executing a trade order."""
    order: TradeOrder
    success: bool
    price: float = 0.0
    cost: float = 0.0     # total cost (buy) or proceeds (sell)
    fees: float = 0.0
    error: str = ""


@dataclass
class Position:
    """A held position."""
    symbol: str
    market: Market
    quantity: int
    avg_cost: float       # average cost per share in USD
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio state."""
    timestamp: str
    cash: float                          # USD
    positions: dict[str, Position]       # key: "MARKET:SYMBOL"
    total_nav: float
    market_exposure: dict[Market, float] # market -> exposure in USD
    fx_rates: dict[str, float]           # "USD/JPY" etc.
    frozen_keys: set[str] = field(default_factory=set)  # T+1 frozen (CN bought today)


# ============================================================
# Agent decision
# ============================================================

@dataclass
class Decision:
    """Agent's trading decision."""
    action: str                          # "trade", "hold", "query"
    trades: list[TradeOrder] = field(default_factory=list)
    queries: list[dict[str, str]] = field(default_factory=list)
    reason: str = ""


@dataclass
class AgentRound:
    """One round of agent interaction."""
    round_num: int
    decision: Decision
    tool_results: str = ""
    llm_response: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0


# ============================================================
# Benchmark result
# ============================================================

@dataclass
class BenchmarkResult:
    """Complete benchmark run result."""
    model_name: str
    dataset_version: str
    start_date: str
    end_date: str
    initial_nav: float
    final_nav: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    avg_holding_bars: float
    total_decisions: int
    rejected_orders: int
    total_llm_tokens: int
    total_llm_calls: int
    decision_log: list[dict[str, Any]]
    portfolio_history: list[PortfolioSnapshot]
