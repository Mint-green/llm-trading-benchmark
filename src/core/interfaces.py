"""
Core interfaces (abstract base classes) for the benchmark system.

All modules implement these interfaces. Dependency injection is used
to wire them together. No concrete implementations here.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from .types import (
    Market, OrderSide, AssetInfo, OHLCVBar, IndicatorSnapshot,
    TradeOrder, TradeResult, Position, PortfolioSnapshot,
    Decision, AgentRound, BenchmarkResult,
)


# ============================================================
# Data Layer Interfaces
# ============================================================

class IMarketDataProvider(ABC):
    """Provides historical market data (OHLCV bars)."""

    @abstractmethod
    def load_bars(
        self, market: Market, symbol: str,
        start: str, end: str,
    ) -> list[OHLCVBar]:
        """Load OHLCV bars for a symbol in a date range."""
        ...

    @abstractmethod
    def load_all_bars(
        self, market: Market, start: str, end: str,
    ) -> dict[str, list[OHLCVBar]]:
        """Load OHLCV bars for all symbols in a market."""
        ...

    @abstractmethod
    def get_universe_symbols(self, market: Market) -> list[str]:
        """Get all symbols in the universe for a market."""
        ...


class IFeatureGenerator(ABC):
    """Computes technical indicators from OHLCV data."""

    @abstractmethod
    def compute(
        self, bars: list[OHLCVBar], timestamp: str,
    ) -> IndicatorSnapshot | None:
        """Compute indicators for the latest bar up to timestamp."""
        ...


class IUniverseRegistry(ABC):
    """Provides the investable universe."""

    @abstractmethod
    def get_assets(self, market: Market) -> list[AssetInfo]:
        """Get all assets in a market's universe."""
        ...

    @abstractmethod
    def get_asset(self, ticker: str) -> AssetInfo | None:
        """Get asset info by ticker."""
        ...


class IAssetStatusProvider(ABC):
    """Provides real-time asset tradability status."""

    @abstractmethod
    def get_status(
        self, market: Market, symbol: str, timestamp: str,
    ) -> tuple[bool, str]:
        """Return (is_tradable, reason) for an asset at a timestamp."""
        ...


# ============================================================
# Portfolio & Trading Interfaces
# ============================================================

class IPortfolioEngine(ABC):
    """Manages positions and cash."""

    @abstractmethod
    def get_snapshot(self, timestamp: str) -> PortfolioSnapshot:
        ...

    @abstractmethod
    def get_position(self, key: str) -> Position | None:
        ...

    @abstractmethod
    def execute_buy(
        self, symbol: str, market: Market, quantity: int, price: float, fees: float,
    ) -> None:
        ...

    @abstractmethod
    def execute_sell(
        self, symbol: str, market: Market, quantity: int, price: float, fees: float,
    ) -> None:
        ...

    @property
    @abstractmethod
    def cash(self) -> float:
        ...

    @property
    @abstractmethod
    def nav(self) -> float:
        ...


class INavEngine(ABC):
    """Computes NAV across multi-currency accounts."""

    @abstractmethod
    def compute_nav(
        self, cash: float, positions: list[Position],
        fx_rates: dict[str, float],
    ) -> float:
        ...


class IConstraintEngine(ABC):
    """Validates orders against position limits."""

    @abstractmethod
    def validate_buy(
        self, symbol: str, market: Market, quantity: int, price: float,
        current_nav: float, current_positions: dict[str, Position],
    ) -> tuple[bool, str]:
        """Return (ok, reason) for a buy order."""
        ...

    @abstractmethod
    def validate_sell(
        self, key: str, quantity: int,
        current_positions: dict[str, Position],
    ) -> tuple[bool, str]:
        """Return (ok, reason) for a sell order."""
        ...


class IMarketRuleEngine(ABC):
    """Enforces market-specific trading rules."""

    @abstractmethod
    def can_trade(
        self, market: Market, symbol: str, side: OrderSide, timestamp: str,
    ) -> tuple[bool, str]:
        """Check if a trade is allowed (hours, halt, limit, T+1, etc.)."""
        ...


class IExecutionEngine(ABC):
    """Handles order execution with cost model."""

    @abstractmethod
    def execute(
        self, order: TradeOrder, price: float, timestamp: str,
    ) -> TradeResult:
        """Execute an order and return the result with fees."""
        ...


class ISettlementEngine(ABC):
    """Handles post-trade settlement (T+1, etc.)."""

    @abstractmethod
    def settle(self, result: TradeResult, timestamp: str) -> None:
        ...

    @abstractmethod
    def get_sellable_quantity(self, key: str, timestamp: str) -> int:
        """How many shares can be sold at this timestamp (T+1 aware)."""
        ...


# ============================================================
# Agent Interfaces
# ============================================================

class IContextBuilder(ABC):
    """Builds LLM prompt context."""

    @abstractmethod
    def build(
        self, timestamp: str, snapshot: PortfolioSnapshot,
        market_data: str, stock_data: str, alerts: str,
        news: str, round_num: int, tool_results: str,
        trade_feedback: str = "",
    ) -> list[dict[str, str]]:
        """Build messages array for the LLM."""
        ...


class IToolSystem(ABC):
    """Provides tools the agent can call."""

    @abstractmethod
    def get_tool_descriptions(self) -> list[dict]:
        ...

    @abstractmethod
    def execute_tool(
        self, name: str, args: dict, timestamp: str,
    ) -> str:
        ...


class IAgentRunner(ABC):
    """Runs the multi-round agent decision loop."""

    @abstractmethod
    def run(
        self, timestamp: str, snapshot: PortfolioSnapshot,
        market_data: str, stock_data: str, alerts: str, news: str,
    ) -> tuple[Decision, list[AgentRound]]:
        """Run agent loop and return final decision + round history."""
        ...


# ============================================================
# Evaluation Interfaces
# ============================================================

class IMetricsEngine(ABC):
    """Computes benchmark metrics."""

    @abstractmethod
    def compute(
        self, portfolio_history: list[PortfolioSnapshot], trades: list[TradeResult],
    ) -> dict[str, float]:
        ...


class ILeaderboard(ABC):
    """Ranks benchmark results."""

    @abstractmethod
    def rank(self, results: list[BenchmarkResult]) -> list[dict]:
        ...


class IBehaviorAnalyzer(ABC):
    """Analyzes agent behavior patterns."""

    @abstractmethod
    def analyze(
        self, rounds: list[AgentRound], trades: list[TradeResult],
    ) -> dict[str, Any]:
        ...


# ============================================================
# Platform Interfaces
# ============================================================

class IExperimentLogger(ABC):
    """Logs experiment data."""

    @abstractmethod
    def log_decision(self, timestamp: str, decision: Decision, snapshot: PortfolioSnapshot, decision_type: str = "full_decision") -> None:
        ...

    @abstractmethod
    def log_trade(self, result: TradeResult, timestamp: str = "") -> None:
        ...

    @abstractmethod
    def log_round(self, round_data: AgentRound) -> None:
        ...

    @abstractmethod
    def save_results(self, result: BenchmarkResult) -> str:
        """Save complete results, return path."""
        ...
