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
    GOLD = "GOLD"
    FUTURES = "FUTURES"


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


class CandidateBucket(str, Enum):
    """Candidate bucket types for structured candidate presentation."""
    HELD_POSITIONS = "held_positions"
    EXIT_WATCH = "exit_watch"
    TREND_LEADERS = "trend_leaders"
    PULLBACK_CONTINUATION = "pullback_continuation"
    OVERSOLD_REVERSAL = "oversold_reversal"
    LOW_VOL_DEFENSIVE = "low_vol_defensive"
    CRYPTO_CANDIDATES = "crypto_candidates"
    BLOCKED_OR_WARNING = "blocked_or_warning"


class DecisionType(str, Enum):
    """Decision types for the scheduling system."""
    AUTO_HOLD = "auto_hold"
    FULL_DECISION = "full_decision"
    LIGHT_DECISION = "light_decision"
    FOCUSED_POSITION = "focused_position_decision"
    FOCUSED_MARKET_RISK = "focused_market_or_risk_decision"


class PlanAction(str, Enum):
    """Plan update actions."""
    CREATE = "create"
    UPDATE = "update"
    CLOSE = "close"
    NO_CHANGE = "no_change"


class TriggerType(str, Enum):
    """Structured trigger types for plan monitoring."""
    PRICE_MOVE_PCT = "price_move_pct"
    ATR_MOVE = "atr_move"
    PNL_PCT = "pnl_pct"
    TRAILING_DRAWDOWN_PCT = "trailing_drawdown_pct"
    TRAILING_ATR = "trailing_atr"
    BARS_ELAPSED = "bars_elapsed"
    REGIME_CHANGE = "regime_change"
    ASSET_STATUS_CHANGE = "asset_status_change"
    MARGIN_RISK_CHANGE = "margin_risk_change"


class RiskMode(str, Enum):
    """Market risk regime."""
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


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
    # V4 trend variables
    ret_30m: float = 0.0      # 30-minute return (6 bars)
    rsi_d1h: float = 0.0      # RSI change in last hour
    trend6: str = ""           # 6-bar trend pattern (e.g., "↑↑→↑↑↑")
    setup: str = ""            # setup classification
    recent_score: int = 0      # short-term state score (-2 to +2)


# ============================================================
# Trading data
# ============================================================

@dataclass
class TradeOrder:
    """A trade order submitted by the agent."""
    symbol: str
    market: Market
    side: OrderSide
    quantity: float = 0.0        # shares/units/contracts (resolved from allocation_pct)
    allocation_pct: float | None = None  # fraction of NAV (0.05 = 5%)
    reason: str = ""
    order_type: OrderType = OrderType.MARKET
    asset_type: str = "equity"
    action: str = ""                     # futures: OPEN_OR_INCREASE, REDUCE, CLOSE, HOLD
    futures_side: str = ""               # futures: long, short, flat
    target_notional_pct_nav: float | None = None
    max_margin_pct_nav: float | None = None
    risk_budget_pct_nav: float | None = None
    unit_hint: dict[str, Any] = field(default_factory=dict)
    risk_trigger: str = ""


@dataclass
class TradeResult:
    """Result of executing a trade order."""
    order: TradeOrder
    success: bool
    price: float = 0.0
    cost: float = 0.0     # total cost (buy) or proceeds (sell)
    fees: float = 0.0
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """A held position."""
    symbol: str
    market: Market
    quantity: float
    avg_cost: float       # average cost per share/unit in USD
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
    futures_positions: dict[str, FuturesPosition] = field(default_factory=dict)
    futures_margin_locked: float = 0.0
    futures_margin_state: str = "OK"
    futures_pnl_delta: float = 0.0


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
    # V3 memory and plan updates
    memory_updates: dict = field(default_factory=dict)
    plan_updates: list[dict] = field(default_factory=list)


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


# ============================================================
# Candidate bucket types
# ============================================================

@dataclass
class CandidateInBucket:
    """A candidate placed in a specific bucket."""
    bucket: CandidateBucket
    ticker: str
    market: Market
    price: float
    score: float  # composite score from screener
    chg_1h: float = 0.0
    chg_1d: float = 0.0
    chg_5d: float = 0.0
    rsi: float = 50.0
    trend: str = ""
    tradable: bool = True
    # Extra fields depending on bucket
    pnl_pct: float = 0.0       # for held_positions, exit_watch
    pct_nav: float = 0.0       # for held_positions, exit_watch
    hold_bars: int = 0         # for held_positions
    sellable: bool = True      # for held_positions
    plan_status: str = ""      # for held_positions
    risk_note: str = ""        # for held_positions, blocked_or_warning
    reason: str = ""           # for exit_watch, blocked_or_warning
    allowed_action: str = ""   # for exit_watch, blocked_or_warning
    cost_bps: float = 0.0      # for trend_leaders, low_vol_defensive
    pullback_note: str = ""    # for pullback_continuation
    stabilization: str = ""    # for oversold_reversal
    atr_pct: float = 0.0       # for low_vol_defensive
    drawdown_pct: float = 0.0  # for low_vol_defensive
    volatility: float = 0.0    # for crypto_candidates
    liquidity: float = 0.0     # for crypto_candidates
    recent_bars: str = ""      # for trend_leaders
    # V4 trend variables
    ret_30m: float = 0.0       # 30-minute return
    rsi_d1h: float = 0.0       # RSI change in last hour
    trend6: str = ""           # 6-bar trend pattern
    setup: str = ""            # setup classification
    recent_score: int = 0      # short-term state score (-2 to +2)
    asset_type: str = "equity"
    actual_contract: str = ""
    notional_per_contract: float = 0.0
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    one_contract_notional_pct_nav: float = 0.0
    one_contract_margin_pct_nav: float = 0.0
    roll_status: str = ""
    days_to_expiry: int = 0
    liquidity_note: str = ""
    signal_symbol: str = ""
    standard_variant: str = ""
    micro_variant: str = ""
    execution_guidance: str = ""
    pilot_target_pct_nav: float = 0.0


@dataclass
class CandidateBuckets:
    """All candidate buckets for a decision point."""
    held_positions: list[CandidateInBucket]
    exit_watch: list[CandidateInBucket]
    trend_leaders: list[CandidateInBucket]
    pullback_continuation: list[CandidateInBucket]
    oversold_reversal: list[CandidateInBucket]
    low_vol_defensive: list[CandidateInBucket]
    crypto_candidates: list[CandidateInBucket]
    blocked_or_warning: list[CandidateInBucket]
    futures_macro: list[CandidateInBucket] = field(default_factory=list)


# ============================================================
# Portfolio target (target_pct_nav system)
# ============================================================

@dataclass
class PortfolioTarget:
    """LLM's target for a position (target_pct_nav system)."""
    symbol: str
    asset_type: str = "equity"  # equity, crypto, gold_spot, oil_proxy, cash
    target_pct_nav: float = 0.0  # fraction of NAV (0.03 = 3%)
    side: str = ""               # futures: long, short, flat
    target_notional_pct_nav: float | None = None
    max_margin_pct_nav: float | None = None
    risk_budget_pct_nav: float | None = None
    unit_hint: dict[str, Any] = field(default_factory=dict)
    risk_trigger: str = ""
    priority: int = 1
    max_cost_bps: float = 35.0
    reason: str = ""


# ============================================================
# Futures system
# ============================================================

@dataclass(frozen=True)
class FuturesContractSpec:
    """Static and point-in-time metadata for an actual futures contract."""
    root_symbol: str
    continuous_symbol: str
    contract_ticker: str
    exchange: str = ""
    currency: str = "USD"
    multiplier: float = 1.0
    tick_size: float = 0.01
    tick_value: float = 0.01
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    expiry_date: str | None = None
    first_notice_date: str | None = None
    last_trade_date: str | None = None
    roll_start_date: str | None = None
    status: str | None = None
    bars_count: int | None = None
    date_range: str | None = None


@dataclass(frozen=True)
class FuturesResolvedContract:
    """Resolver output for a continuous futures symbol at one timestamp."""
    continuous_symbol: str
    contract_ticker: str
    timestamp: str
    expiry_date: str | None = None
    days_to_expiry: int | None = None
    roll_status: str = "normal"
    selection_method: str = ""
    price: float | None = None
    multiplier: float = 1.0
    tick_size: float = 0.01
    tick_value: float = 0.01
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    notional_per_contract: float | None = None
    previous_session_dollar_volume: float | None = None
    previous_session_volume: float | None = None


@dataclass
class FuturesPosition:
    """Variation-margin futures position state."""
    continuous_symbol: str
    contract_ticker: str
    side: str
    contracts: int
    avg_entry_price: float
    previous_mark_price: float
    current_price: float
    multiplier: float
    tick_size: float
    initial_margin_per_contract: float
    maintenance_margin_per_contract: float
    margin_locked: float
    realized_pnl: float = 0.0
    cumulative_variation_pnl: float = 0.0
    opened_at: str = ""
    updated_at: str = ""

    @property
    def side_sign(self) -> int:
        return -1 if self.side == "short" else 1

    @property
    def notional(self) -> float:
        return self.contracts * self.current_price * self.multiplier


@dataclass
class FuturesMarkResult:
    """Mark-to-market result for one futures position."""
    timestamp: str
    continuous_symbol: str
    contract_ticker: str
    previous_mark_price: float
    current_price: float
    pnl_delta: float
    cumulative_variation_pnl: float
    margin_locked: float
    margin_state: str


# ============================================================
# Plan system
# ============================================================

@dataclass
class PlanTrigger:
    """A structured trigger for plan monitoring."""
    trigger_type: TriggerType
    direction: str = ""        # "up" or "down"
    anchor: str = ""           # "last_review_price", "peak_since_entry"
    threshold_pct: float = 0.0
    atr_multiple: float = 0.0
    operator: str = ""         # "<=", ">=" for pnl_pct
    since: str = ""            # "last_review" for bars_elapsed
    bars: int = 0              # for bars_elapsed
    peak_anchor: str = ""      # for trailing_drawdown_pct
    atr_source: str = ""       # for atr_move


@dataclass
class ActivePlan:
    """An active trading plan for a position."""
    plan_id: str
    symbol: str
    position_id: str = ""
    status: str = "active"  # active, closed
    side: str = "long"
    entry_time: str = ""
    entry_price: float = 0.0
    current_pct_nav: float = 0.0
    entry_reason: str = ""
    plan_version: int = 1
    last_review_time: str = ""
    last_review_price: float = 0.0
    atr_at_review: float = 0.0
    peak_since_entry: float = 0.0
    peak_since_last_review: float = 0.0
    intended_horizon_bars: int = 36
    plan_note: str = ""
    triggers: list[PlanTrigger] = field(default_factory=list)


# ============================================================
# Memory system
# ============================================================

@dataclass
class WatchlistItem:
    """An item on the watchlist."""
    symbol: str
    reason: str = ""
    desired_condition: dict = field(default_factory=dict)  # e.g. {"type": "rsi_range", "min": 45, "max": 65}
    source_event_id: str = ""
    created_at: str = ""
    expires_at: str = ""
    # V4 structured fields
    current_price: float = 0.0
    current_rsi: float = 0.0
    met: str = "unknown"  # "yes", "no", "unknown"
    tradable: bool = True
    action_hint: str = "keep_watch"  # "keep_watch", "consider_buy", "remove"


@dataclass
class AvoidItem:
    """An item on the avoid/cooldown list."""
    symbol: str
    reason: str = ""
    source_event_id: str = ""
    created_at: str = ""
    expires_at: str = ""


@dataclass
class DailyThesis:
    """The daily market thesis."""
    text: str = ""
    confidence: float = 0.0
    version: int = 1
    created_at: str = ""
    expires_at: str = ""


@dataclass
class RecentActivity:
    """Recent activity summary for prompt injection."""
    non_hold_decisions: list[str] = field(default_factory=list)  # last 3
    focused_decisions: list[str] = field(default_factory=list)   # last 2
    execution_feedback: list[str] = field(default_factory=list)  # last 3
    risk_state_changes: list[str] = field(default_factory=list)  # last 1


@dataclass
class ExecutionFeedback:
    """Feedback from a trade execution."""
    symbol: str
    requested_target_pct_nav: float
    filled_target_pct_nav: float
    status: str  # OK, ADJUSTED, FAILED
    reason: str = ""
    fees_usd: float = 0.0
    slippage_usd: float = 0.0
    timestamp: str = ""


@dataclass
class SessionSummary:
    """Summary for a market session (e.g., HK close)."""
    market: str
    session_date: str
    market_read: str = ""
    model_actions: list[str] = field(default_factory=list)
    open_positions: list[dict] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    created_at: str = ""


@dataclass
class DailySummary:
    """Daily global summary at benchmark boundary."""
    date: str
    nav_start: float = 0.0
    nav_end: float = 0.0
    daily_return_pct: float = 0.0
    market_read: str = ""
    major_decisions: list[str] = field(default_factory=list)
    what_worked: list[str] = field(default_factory=list)
    what_failed: list[str] = field(default_factory=list)
    carryover_positions: list[dict] = field(default_factory=list)
    avoid_next_day: list[dict] = field(default_factory=list)
    behavior: dict = field(default_factory=dict)
    created_at: str = ""


@dataclass
class MemoryState:
    """Complete memory state injected into prompt."""
    previous_daily_summary: DailySummary | None = None
    daily_thesis: DailyThesis | None = None
    recent_activity: RecentActivity = field(default_factory=RecentActivity)
    watchlist: list[WatchlistItem] = field(default_factory=list)
    avoid_list: list[AvoidItem] = field(default_factory=list)
    recent_feedback: list[ExecutionFeedback] = field(default_factory=list)
    rolling_behavior_notes: list[str] = field(default_factory=list)
    active_plans: dict[str, ActivePlan] = field(default_factory=dict)
