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


@dataclass(frozen=True)
class Config:
    """Benchmark configuration. Immutable after creation."""

    # --- Data paths ---
    stock_data_dir: str = "D:/Projects/claw/getStockData/data"

    # --- Database paths ---
    @property
    def db_paths(self) -> dict[Market, str]:
        return {
            Market.US: os.path.join(self.stock_data_dir, "US_stock.db").replace("\\", "/"),
            Market.HK: os.path.join(self.stock_data_dir, "HK_stock.db").replace("\\", "/"),
            Market.CN: os.path.join(self.stock_data_dir, "A_stock.db").replace("\\", "/"),
            Market.CRYPTO: os.path.join(self.stock_data_dir, "CRYPTO_stock.db").replace("\\", "/"),
        }

    @property
    def db_tables(self) -> dict[Market, str]:
        return {
            Market.US: "us_5min",
            Market.HK: "hk_5min",
            Market.CN: "stock_5min",
            Market.CRYPTO: "crypto_5min",
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
    decision_interval: int = 60  # minutes between decisions
    snapshot_interval: int = 60  # minutes between portfolio snapshots
    dataset_version: str = "2026-06-05"

    # --- Agent ---
    max_agent_rounds: int = 8  # Round1=context, 2-7=tools, 8=mandatory decision
    max_decisions: int = 0     # 0 = unlimited

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
        }
    )
    slippage_bps: dict[Market, float] = field(
        default_factory=lambda: {
            Market.US: 5.0,
            Market.HK: 5.0,
            Market.CN: 5.0,
            Market.CRYPTO: 10.0,
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

        # Determine which API to use based on model name
        model_name = model_cfg.get("name", "mimo-v2.5-pro")

        # Build config kwargs
        base_dir = data_cfg.get("base_dir", "D:/Projects/claw/getStockData")
        kwargs = {
            # Data
            "stock_data_dir": os.path.join(base_dir, "data").replace("\\", "/"),
            # Backtest
            "backtest_start": backtest_cfg.get("start", "2026-02-03"),
            "backtest_end": backtest_cfg.get("end", "2026-02-09"),
            "decision_interval": backtest_cfg.get("interval", 60),
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
            },
            "slippage_bps": {
                Market.US: costs_cfg.get("slippage_bps", {}).get("US", 5),
                Market.HK: costs_cfg.get("slippage_bps", {}).get("HK", 5),
                Market.CN: costs_cfg.get("slippage_bps", {}).get("CN", 5),
                Market.CRYPTO: costs_cfg.get("slippage_bps", {}).get("CRYPTO", 10),
            },
            # Agent
            "max_agent_rounds": agent_cfg.get("max_rounds", 8),
        }

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
            "temperature": self.temperature,
            "thinking_enabled": self.thinking_enabled,
            "position_limits": {
                "max_single": self.position_limits.max_single_position,
                "max_market": self.position_limits.max_market_exposure,
                "max_crypto": self.position_limits.max_crypto_exposure,
                "min_cash": self.position_limits.min_cash_ratio,
            },
        }
