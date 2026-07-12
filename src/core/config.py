"""
Configuration system for the benchmark.

Loads from TOML file with sensible defaults.
All paths, API keys, and parameters are centralized here.
"""

from __future__ import annotations
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from .types import Market, MarketHours, PositionLimit
from src.core.futures_specs import DEFAULT_ALLOWED_FUTURES_SYMBOLS


@dataclass(frozen=True)
class TriggerLimitsConfig:
    """Valid ranges for trigger thresholds. Values outside are clamped."""
    pnl_pct_min: float = -0.10         # max stop-loss -10%
    pnl_pct_max: float = 0.20          # max take-profit +20%
    price_move_min: float = -0.20      # max drop -20%
    price_move_max: float = 0.20       # max rise +20%
    trailing_drawdown_min: float = 0.001  # must be positive, min 0.1%
    trailing_drawdown_max: float = 0.20   # max 20%


@dataclass(frozen=True)
class TriggerConfig:
    """Trigger thresholds for plan monitoring. Market-specific overrides possible."""
    price_move_pct: float = 0.02       # 2% price move triggers review
    atr_move_multiple: float = 1.5     # 1.5x ATR move triggers review
    pnl_pct_threshold: float = -0.03   # -3% PnL triggers stop review
    trailing_drawdown_pct: float = 0.02  # 2% drawdown from peak
    trailing_atr_multiple: float = 2.0   # 2x ATR from peak
    bars_elapsed: int = 6              # review every 6 bars (30min at 5min/bar)
    cooldown_bars: int = 6             # cooldown after focused review (30min)


@dataclass(frozen=True)
class CryptoTriggerConfig:
    """Crypto-specific trigger thresholds (wider due to higher volatility)."""
    price_move_pct: float = 0.05       # 5%
    pnl_pct_threshold: float = -0.05   # -5%
    trailing_drawdown_pct: float = 0.03  # 3%


@dataclass(frozen=True)
class StopLossConfig:
    """Hard stop-loss thresholds. When PnL hits this, force-sell without LLM."""
    hard_stop_pct: float = -0.05       # -5% for stocks/futures
    crypto_hard_stop_pct: float = -0.08  # -8% for crypto


@dataclass(frozen=True)
class OpenWindowConfig:
    """Decision frequency during market open window."""
    enabled: bool = True
    minutes_after_open: int = 30
    interval_minutes: int = 15
    include_open_plus_30: bool = True


@dataclass(frozen=True)
class CloseWindowConfig:
    """Decision frequency during market close window."""
    enabled: bool = True
    minutes_before_close: int = 30
    interval_minutes: int = 15
    include_close_time: bool = False


@dataclass(frozen=True)
class DecisionScheduleConfig:
    """Full decision scheduling configuration."""
    normal_interval_minutes: int = 30
    open_window: OpenWindowConfig = field(default_factory=OpenWindowConfig)
    close_window: CloseWindowConfig = field(default_factory=CloseWindowConfig)


@dataclass(frozen=True)
class TailGuardConfig:
    """Tail guard configuration for close window."""
    enabled: bool = True
    minutes_before_close: int = 15
    block_new_buy: bool = True
    block_increase_position: bool = True
    allow_reduce_close_hold: bool = True


@dataclass(frozen=True)
class MarketCloseRuleConfig:
    """Market close rule configuration."""
    at_or_after_close_no_same_session_trade: bool = True
    final_bar_usable_for_summary: bool = True


@dataclass(frozen=True)
class GoldConfig:
    """Gold spot configuration using XAUUSD.FOREX from GOLD_stock.db."""
    enabled: bool = True
    allowed_symbols: tuple[str, ...] = ("XAUUSD.FOREX",)
    ask_symbol: str = "XAUUSD.FOREX.ASK"
    max_exposure_pct_nav: float = 0.25

@dataclass(frozen=True)
class FuturesConfig:
    """Conservative futures configuration for the supported futures basket."""
    enabled: bool = True
    allowed_symbols: tuple[str, ...] = DEFAULT_ALLOWED_FUTURES_SYMBOLS
    allow_short: bool = False
    max_contracts_per_symbol: int = 1
    max_abs_notional_pct_nav: float = 1.00
    max_total_abs_notional_pct_nav: float = 1.00
    max_margin_pct_nav: float = 0.10
    max_total_margin_pct_nav: float = 0.20
    max_risk_budget_pct_nav: float = 0.01
    roll_days_before_expiry: int = 5
    force_close_days_before_expiry: int = 2
    commission_per_contract: float = 2.50
    slippage_bps: float = 2.0
    min_dollar_volume_lookback: float = 0.0
    liquidity_lookback_bars: int = 12
    execution_price_mode: str = "next_bar_open"
    gc_multiplier: float = 100.0
    gc_tick_size: float = 0.1
    gc_tick_value: float = 10.0
    gc_initial_margin: float = 12000.0
    gc_maintenance_margin: float = 11000.0


@dataclass(frozen=True)
class Config:
    """Benchmark configuration. Immutable after creation."""

    # --- Data paths ---
    stock_data_dir: str = "D:/Projects/claw/getStockData/data"

    # --- Active model ---
    model_name: str = "deepseek-v4-pro"

    # --- Database paths ---
    @property
    def db_paths(self) -> dict[Market, str]:
        return {
            Market.US: os.path.join(self.stock_data_dir, "US_stock.db").replace("\\", "/"),
            Market.HK: os.path.join(self.stock_data_dir, "HK_stock.db").replace("\\", "/"),
            Market.CN: os.path.join(self.stock_data_dir, "A_stock.db").replace("\\", "/"),
            Market.CRYPTO: os.path.join(self.stock_data_dir, "CRYPTO_stock.db").replace("\\", "/"),
            Market.GOLD: os.path.join(self.stock_data_dir, "GOLD_stock.db").replace("\\", "/"),
            Market.FUTURES: os.path.join(self.stock_data_dir, "FUTURES_stock.db").replace("\\", "/"),
        }

    @property
    def db_tables(self) -> dict[Market, str]:
        return {
            Market.US: "us_5min",
            Market.HK: "hk_5min",
            Market.CN: "stock_5min",
            Market.CRYPTO: "crypto_5min",
            Market.GOLD: "gold_5min",
            Market.FUTURES: "futures_5min",
        }

    # --- LLM Global Settings ---
    temperature: float = 0.3
    thinking_enabled: bool = False

    # --- LLM API (mimo-v2.5-pro) ---
    mimo_pro_api_key: str = ""
    mimo_pro_base_url: str = "https://token-plan-cn.xiaomimimo.com/v1"
    mimo_pro_model: str = "mimo-v2.5-pro"
    mimo_pro_max_tokens: int = 4096
    mimo_pro_timeout: float = 180.0

    # --- LLM API (deepseek-v4-pro) ---
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_max_tokens: int = 4096
    deepseek_timeout: float = 180.0

    # --- Benchmark parameters ---
    backtest_start: str = "2026-02-03"
    backtest_end: str = "2026-02-09"
    decision_interval: int = 5   # minutes between decisions
    snapshot_interval: int = 60  # minutes between portfolio snapshots
    dataset_version: str = "2026-06-05"

    # --- Agent ---
    max_agent_rounds: int = 4  # Global hard cap; v3 decision types may use lower caps
    full_decision_max_rounds: int = 3
    max_decisions: int = 0     # 0 = unlimited

    # --- Decision Schedule ---
    decision_schedule: DecisionScheduleConfig = field(default_factory=DecisionScheduleConfig)
    tail_guard: TailGuardConfig = field(default_factory=TailGuardConfig)
    market_close_rule: MarketCloseRuleConfig = field(default_factory=MarketCloseRuleConfig)
    futures: FuturesConfig = field(default_factory=FuturesConfig)
    gold: GoldConfig = field(default_factory=GoldConfig)

    # --- Trigger Thresholds ---
    trigger_config: TriggerConfig = field(default_factory=TriggerConfig)
    crypto_trigger_config: CryptoTriggerConfig = field(default_factory=CryptoTriggerConfig)
    trigger_limits: TriggerLimitsConfig = field(default_factory=TriggerLimitsConfig)
    stop_loss: StopLossConfig = field(default_factory=StopLossConfig)

    # --- Time System ---
    benchmark_boundary_utc: str = "00:00"  # daily summary at 00:00 UTC
    session_summary_minutes_after_close: int = 5  # session summary 5min after market close

    # --- Portfolio ---
    initial_cash: float = 100_000.0
    position_limits: PositionLimit = field(
        default_factory=lambda: PositionLimit(
            max_single_position=0.25,
            max_market_exposure=0.50,
            max_crypto_exposure=0.25,
            min_cash_ratio=0.05,
        )
    )

    # --- Cost model (basis points) ---
    commission_bps: dict[Market, float] = field(
        default_factory=lambda: {
            Market.US: 3.0,
            Market.HK: 5.0,
            Market.CN: 3.0,
            Market.CRYPTO: 10.0,
            Market.GOLD: 0.0,
            Market.FUTURES: 0.0,
        }
    )
    slippage_bps: dict[Market, float] = field(
        default_factory=lambda: {
            Market.US: 5.0,
            Market.HK: 5.0,
            Market.CN: 5.0,
            Market.CRYPTO: 10.0,
            Market.GOLD: 5.0,
            Market.FUTURES: 2.0,
        }
    )
    cn_sell_tax_bps: float = 5.0
    fx_fee_bps: float = 5.0

    # --- FX rates (fixed for now) ---
    fx_rates: dict[str, float] = field(
        default_factory=lambda: {
            "USD": 1.0,
            "HKD": 7.80,
            "CNY": 7.25,
            "JPY": 155.0,
        }
    )

    # --- Market hours (UTC) ---
    @property
    def market_hours(self) -> dict[Market, MarketHours]:
        return {
            Market.US: MarketHours(
                market=Market.US,
                sessions=[("14:30", "21:00")],
                timezone="EST",
            ),
            Market.HK: MarketHours(
                market=Market.HK,
                sessions=[("01:30", "04:00"), ("05:00", "08:00")],
                timezone="HKT",
            ),
            Market.CN: MarketHours(
                market=Market.CN,
                sessions=[("01:30", "03:30"), ("05:00", "07:00")],
                timezone="CST",
            ),
        }

    # --- Asset limits ---
    cn_limit_up_pct: float = 10.0
    cn_st_limit_up_pct: float = 20.0
    cn_limit_near_threshold: float = 0.5

    # --- Universe size per market ---
    universe_size: dict[Market, int] = field(
        default_factory=lambda: {
            Market.US: 50,
            Market.HK: 50,
            Market.CN: 50,
            Market.CRYPTO: 18,
            Market.GOLD: 1,
            Market.FUTURES: 9,
        }
    )

    @classmethod
    def load_from_toml(cls, config_path: str = "config/config.toml") -> "Config":
        """Load config from TOML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Copy template: cp config/template.toml config/config.toml"
            )

        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Extract sections
        data_cfg = data.get("data", {})
        backtest_cfg = data.get("backtest", {})
        model_cfg = data.get("model", {})
        api_cfg = model_cfg.get("api", {})
        portfolio_cfg = data.get("portfolio", {})
        limits_cfg = portfolio_cfg.get("position_limits", {})
        costs_cfg = data.get("costs", {})
        agent_cfg = data.get("agent", {})
        schedule_cfg = data.get("decision_schedule", {})
        open_window_cfg = schedule_cfg.get("open_window", {})
        close_window_cfg = schedule_cfg.get("close_window", {})
        futures_section_present = "futures" in data
        gold_section_present = "gold" in data
        futures_cfg = data.get("futures", {})
        gold_cfg = data.get("gold", {})
        trigger_cfg = data.get("trigger", {})
        crypto_trigger_cfg = trigger_cfg.get("crypto", {})
        stop_loss_cfg = data.get("stop_loss", {})

        # Determine which API to use based on model name
        model_name = model_cfg.get("name", "mimo-v2.5-pro")

        # Build config kwargs
        base_dir = data_cfg.get("base_dir", "D:/Projects/claw/getStockData")
        kwargs = {
            # Data
            "stock_data_dir": os.path.join(base_dir, "data").replace("\\", "/"),
            # Active model
            "model_name": model_name,
            # Backtest
            "backtest_start": backtest_cfg.get("start", "2026-02-03"),
            "backtest_end": backtest_cfg.get("end", "2026-02-09"),
            "decision_interval": backtest_cfg.get("decision_interval", 5),
            "max_decisions": backtest_cfg.get("max_decisions", 0),
            # Model
            "temperature": model_cfg.get("temperature", 0.3),
            "thinking_enabled": model_cfg.get("thinking_enabled", False),
            # Portfolio
            "initial_cash": portfolio_cfg.get("initial_cash", 100000),
            "position_limits": PositionLimit(
                max_single_position=limits_cfg.get("max_single", 0.25),
                max_market_exposure=limits_cfg.get("max_market", 0.50),
                max_crypto_exposure=limits_cfg.get("max_crypto", 0.25),
                min_cash_ratio=limits_cfg.get("min_cash", 0.05),
            ),
            # Costs
            "cn_sell_tax_bps": costs_cfg.get("cn_sell_tax_bps", 5),
            "fx_fee_bps": costs_cfg.get("fx_fee_bps", 5),
            "commission_bps": {
                Market.US: costs_cfg.get("commission_bps", {}).get("US", 3),
                Market.HK: costs_cfg.get("commission_bps", {}).get("HK", 5),
                Market.CN: costs_cfg.get("commission_bps", {}).get("CN", 3),
                Market.CRYPTO: costs_cfg.get("commission_bps", {}).get("CRYPTO", 10),
                Market.GOLD: costs_cfg.get("commission_bps", {}).get("GOLD", 0),
                Market.FUTURES: costs_cfg.get("commission_bps", {}).get("FUTURES", 0),
            },
            "slippage_bps": {
                Market.US: costs_cfg.get("slippage_bps", {}).get("US", 5),
                Market.HK: costs_cfg.get("slippage_bps", {}).get("HK", 5),
                Market.CN: costs_cfg.get("slippage_bps", {}).get("CN", 5),
                Market.CRYPTO: costs_cfg.get("slippage_bps", {}).get("CRYPTO", 10),
                Market.GOLD: costs_cfg.get("slippage_bps", {}).get("GOLD", gold_cfg.get("slippage_bps", 5)),
                Market.FUTURES: costs_cfg.get("slippage_bps", {}).get("FUTURES", futures_cfg.get("slippage_bps", 2)),
            },
            # Agent
            "max_agent_rounds": agent_cfg.get("max_rounds", 4),
            "full_decision_max_rounds": agent_cfg.get("full_decision_max_rounds", 3),
            # Decision schedule
            "gold": GoldConfig(
                enabled=gold_cfg.get("enabled", gold_section_present),
                allowed_symbols=tuple(gold_cfg.get("allowed_symbols", ["XAUUSD.FOREX"])),
                ask_symbol=gold_cfg.get("ask_symbol", "XAUUSD.FOREX.ASK"),
                max_exposure_pct_nav=gold_cfg.get("max_exposure_pct_nav", 0.25),
            ),
            "futures": FuturesConfig(
                enabled=futures_cfg.get("enabled", futures_section_present),
                allowed_symbols=tuple(futures_cfg.get("allowed_symbols", list(DEFAULT_ALLOWED_FUTURES_SYMBOLS))),
                allow_short=futures_cfg.get("allow_short", False),
                max_contracts_per_symbol=futures_cfg.get("max_contracts_per_symbol", 1),
                max_abs_notional_pct_nav=futures_cfg.get("max_abs_notional_pct_nav", 1.00),
                max_total_abs_notional_pct_nav=futures_cfg.get("max_total_abs_notional_pct_nav", 1.00),
                max_margin_pct_nav=futures_cfg.get("max_margin_pct_nav", 0.10),
                max_total_margin_pct_nav=futures_cfg.get("max_total_margin_pct_nav", 0.20),
                max_risk_budget_pct_nav=futures_cfg.get("max_risk_budget_pct_nav", 0.01),
                roll_days_before_expiry=futures_cfg.get("roll_days_before_expiry", 5),
                force_close_days_before_expiry=futures_cfg.get("force_close_days_before_expiry", 2),
                commission_per_contract=futures_cfg.get("commission_per_contract", 2.50),
                slippage_bps=futures_cfg.get("slippage_bps", 2),
                min_dollar_volume_lookback=futures_cfg.get("min_dollar_volume_lookback", 0),
                liquidity_lookback_bars=futures_cfg.get("liquidity_lookback_bars", 12),
                execution_price_mode=futures_cfg.get("execution_price_mode", "next_bar_open"),
                gc_multiplier=futures_cfg.get("gc_multiplier", 100.0),
                gc_tick_size=futures_cfg.get("gc_tick_size", 0.1),
                gc_tick_value=futures_cfg.get("gc_tick_value", 10.0),
                gc_initial_margin=futures_cfg.get("gc_initial_margin", 12000.0),
                gc_maintenance_margin=futures_cfg.get("gc_maintenance_margin", 11000.0),
            ),
            "decision_schedule": DecisionScheduleConfig(
                normal_interval_minutes=schedule_cfg.get("normal_interval_minutes", 30),
                open_window=OpenWindowConfig(
                    enabled=open_window_cfg.get("enabled", True),
                    minutes_after_open=open_window_cfg.get("minutes_after_open", 30),
                    interval_minutes=open_window_cfg.get("interval_minutes", 15),
                    include_open_plus_30=open_window_cfg.get("include_open_plus_30", True),
                ),
                close_window=CloseWindowConfig(
                    enabled=close_window_cfg.get("enabled", True),
                    minutes_before_close=close_window_cfg.get("minutes_before_close", 30),
                    interval_minutes=close_window_cfg.get("interval_minutes", 15),
                    include_close_time=close_window_cfg.get("include_close_time", False),
                ),
            ),
        }

        # Trigger and stop-loss configs
        trigger_limits_cfg = trigger_cfg.get("limits", {})
        kwargs["trigger_config"] = TriggerConfig(
            price_move_pct=trigger_cfg.get("price_move_pct", 0.02),
            atr_move_multiple=trigger_cfg.get("atr_move_multiple", 1.5),
            pnl_pct_threshold=trigger_cfg.get("pnl_pct_threshold", -0.03),
            trailing_drawdown_pct=trigger_cfg.get("trailing_drawdown_pct", 0.02),
            trailing_atr_multiple=trigger_cfg.get("trailing_atr_multiple", 2.0),
            bars_elapsed=trigger_cfg.get("bars_elapsed", 6),
            cooldown_bars=trigger_cfg.get("cooldown_bars", 6),
        )
        kwargs["crypto_trigger_config"] = CryptoTriggerConfig(
            price_move_pct=crypto_trigger_cfg.get("price_move_pct", 0.05),
            pnl_pct_threshold=crypto_trigger_cfg.get("pnl_pct_threshold", -0.05),
            trailing_drawdown_pct=crypto_trigger_cfg.get("trailing_drawdown_pct", 0.03),
        )
        kwargs["trigger_limits"] = TriggerLimitsConfig(
            pnl_pct_min=trigger_limits_cfg.get("pnl_pct_min", -0.10),
            pnl_pct_max=trigger_limits_cfg.get("pnl_pct_max", 0.20),
            price_move_min=trigger_limits_cfg.get("price_move_min", -0.20),
            price_move_max=trigger_limits_cfg.get("price_move_max", 0.20),
            trailing_drawdown_min=trigger_limits_cfg.get("trailing_drawdown_min", 0.001),
            trailing_drawdown_max=trigger_limits_cfg.get("trailing_drawdown_max", 0.20),
        )
        kwargs["stop_loss"] = StopLossConfig(
            hard_stop_pct=stop_loss_cfg.get("hard_stop_pct", -0.05),
            crypto_hard_stop_pct=stop_loss_cfg.get("crypto_hard_stop_pct", -0.08),
        )

        # Load all API keys (both models)
        kwargs["mimo_pro_api_key"] = api_cfg.get("mimo_v2_5_pro_api_key", "")
        kwargs["mimo_pro_base_url"] = api_cfg.get("mimo_v2_5_pro_base_url", "https://token-plan-cn.xiaomimimo.com/v1")
        kwargs["mimo_pro_model"] = "mimo-v2.5-pro"
        kwargs["mimo_pro_max_tokens"] = model_cfg.get("max_tokens", 4096)
        kwargs["mimo_pro_timeout"] = model_cfg.get("timeout", 180)

        kwargs["deepseek_api_key"] = api_cfg.get("deepseek_v4_pro_api_key", "")
        kwargs["deepseek_base_url"] = api_cfg.get("deepseek_v4_pro_base_url", "https://api.deepseek.com")
        kwargs["deepseek_model"] = "deepseek-v4-pro"
        kwargs["deepseek_max_tokens"] = model_cfg.get("max_tokens", 4096)
        kwargs["deepseek_timeout"] = model_cfg.get("timeout", 180)

        return cls(**kwargs)

    def to_dict(self) -> dict:
        """Serialize config for logging."""
        return {
            "backtest_start": self.backtest_start,
            "backtest_end": self.backtest_end,
            "decision_interval": self.decision_interval,
            "initial_cash": self.initial_cash,
            "max_agent_rounds": self.max_agent_rounds,
            "full_decision_max_rounds": self.full_decision_max_rounds,
            "temperature": self.temperature,
            "thinking_enabled": self.thinking_enabled,
            "decision_schedule": {
                "normal_interval_minutes": self.decision_schedule.normal_interval_minutes,
                "open_window_interval_minutes": self.decision_schedule.open_window.interval_minutes,
                "close_window_interval_minutes": self.decision_schedule.close_window.interval_minutes,
            },
            "position_limits": {
                "max_single": self.position_limits.max_single_position,
                "max_market": self.position_limits.max_market_exposure,
                "max_crypto": self.position_limits.max_crypto_exposure,
                "min_cash": self.position_limits.min_cash_ratio,
            },
            "gold": {
                "enabled": self.gold.enabled,
                "allowed_symbols": list(self.gold.allowed_symbols),
                "ask_symbol": self.gold.ask_symbol,
                "max_exposure_pct_nav": self.gold.max_exposure_pct_nav,
            },
            "futures": {
                "enabled": self.futures.enabled,
                "allowed_symbols": list(self.futures.allowed_symbols),
                "max_contracts_per_symbol": self.futures.max_contracts_per_symbol,
                "max_margin_pct_nav": self.futures.max_margin_pct_nav,
                "max_abs_notional_pct_nav": self.futures.max_abs_notional_pct_nav,
            },
        }
