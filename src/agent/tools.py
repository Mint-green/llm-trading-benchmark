"""
ToolSystem — provides read-only query tools for the agent.

7 tools (v3 design):
  - screen_universe:    Screen universe by market, bucket, filters
  - query_asset:        Query detailed asset info
  - query_position:     Query position details
  - query_history:      Query historical price data
  - query_market_overview: Query market-level overview
  - query_fx:           Query FX rates and cash balances
  - query_futures_contract: Query futures contract info (reserved)

All tools are read-only. Trade execution happens via final JSON output.
"""

from __future__ import annotations

import json

from src.core.types import Market, PortfolioSnapshot, IndicatorSnapshot
from src.core.interfaces import IToolSystem
from src.data.provider import MarketDataProvider
from src.data.features import FeatureGenerator
from src.data.futures_resolver import FuturesContractResolver
from src.core.futures_specs import get_futures_family_spec, get_futures_product_spec


# Tool schemas for function calling
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "screen_universe",
            "description": "Screen the point-in-time investable universe using market, bucket, filters, and sorting rules. Read-only. Does not change portfolio or memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "market": {
                        "type": "string",
                        "enum": ["US", "CN", "HK", "CRYPTO", "GOLD", "FUTURES", "ALL"],
                        "description": "Market to screen"
                    },
                    "bucket": {
                        "type": "string",
                        "enum": [
                            "trend_leaders", "pullback_continuation",
                            "oversold_reversal", "low_vol_defensive",
                            "volume_breakout", "custom",
                        ],
                        "description": "Candidate bucket type"
                    },
                    "filters": {
                        "type": "object",
                        "properties": {
                            "rsi_min": {"type": "number"},
                            "rsi_max": {"type": "number"},
                            "min_1d_return": {"type": "number"},
                            "min_5d_return": {"type": "number"},
                            "max_atr_pct": {"type": "number"},
                            "tradable_now": {"type": "boolean"},
                            "exclude_current_positions": {"type": "boolean"},
                        },
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["score_total", "trend_score", "volume_rank", "cost_efficiency"],
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 30,
                        "description": "Max results to return"
                    },
                },
                "required": ["market", "bucket", "limit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_asset",
            "description": "Query point-in-time details for a specific tradable asset. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Asset ticker, e.g. AAPL.US, 0700.HK"},
                    "fields": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "quote", "recent_bars", "indicators",
                                "tradability", "cost", "lot_info",
                            ],
                        },
                        "description": "Which fields to include"
                    },
                    "recent_bar_count": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 48,
                        "description": "Number of recent bars to include"
                    },
                },
                "required": ["symbol", "fields"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_position",
            "description": "Query current position, plan, PnL, tradability, and execution history for a symbol. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Position ticker"},
                    "include_plan": {"type": "boolean", "description": "Include active plan details"},
                    "include_recent_executions": {"type": "boolean", "description": "Include recent trade history"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_history",
            "description": "Query historical price and indicator data available at the decision timestamp. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Asset ticker"},
                    "lookback_bars": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                        "description": "Number of bars to look back"
                    },
                    "bar_size": {
                        "type": "string",
                        "enum": ["5m", "15m", "1h", "1d"],
                        "description": "Bar granularity"
                    },
                    "fields": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["ohlcv", "returns", "rsi", "atr", "ema"],
                        },
                        "description": "Which fields to include"
                    },
                },
                "required": ["symbol", "lookback_bars", "bar_size"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_market_overview",
            "description": "Query point-in-time market overview, regime, breadth, volatility, and open/close status. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "markets": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["US", "CN", "HK", "CRYPTO", "GOLD", "FUTURES"],
                        },
                        "description": "Markets to query"
                    },
                    "include_regime": {"type": "boolean", "description": "Include regime status"},
                    "include_breadth": {"type": "boolean", "description": "Include breadth data"},
                },
                "required": ["markets"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_fx",
            "description": "Query point-in-time FX rates, cash balances, and conversion costs. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_currency": {"type": "string", "description": "Base currency (e.g. USD)"},
                    "quote_currency": {"type": "string", "description": "Quote currency (e.g. HKD)"},
                    "include_cash_balances": {"type": "boolean", "description": "Include current cash positions"},
                    "include_conversion_cost": {"type": "boolean", "description": "Include FX conversion cost"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_futures_contract",
            "description": "Query current actual futures contract mapped from a continuous symbol, including margin, notional, liquidity, and roll status. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "continuous_symbol": {"type": "string", "description": "Continuous futures symbol"},
                    "fields": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["actual_contract", "price", "multiplier", "tick_size", "tick_value", "notional", "initial_margin", "maintenance_margin", "volume", "dollar_volume", "roll_status", "expiry_date", "days_to_expiry", "selection_method"],
                        },
                    },
                },
                "required": ["continuous_symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_futures_family",
            "description": "Query a futures exposure family with signal contract and standard/micro tradable variants. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Futures family symbol such as GOLD_FUT or OIL_FUT"},
                    "fields": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["signal", "signal_features", "pilot_target_pct_nav", "tradable_variants", "roll_status", "risk", "current_family_exposure"],
                        },
                    },
                },
                "required": ["symbol"],
            },
        },
    },]


class ToolSystem(IToolSystem):
    """Provides queryable tools for the agent."""

    def __init__(
        self,
        data_provider: MarketDataProvider,
        feature_gen: FeatureGenerator,
        portfolio_snapshot_fn,  # callable that returns PortfolioSnapshot
    ):
        self._data = data_provider
        self._features = feature_gen
        self._get_snapshot = portfolio_snapshot_fn
        self._futures_resolver = None
        if hasattr(data_provider, "_config") and hasattr(data_provider, "load_futures_contracts"):
            self._futures_resolver = FuturesContractResolver(data_provider._config, data_provider)

    def get_tool_descriptions(self) -> list[dict]:
        """Return tool schemas for function calling."""
        return TOOL_SCHEMAS

    def execute_tool(
        self, name: str, args: dict, timestamp: str,
    ) -> str:
        """Execute a tool and return result as string."""
        if name == "screen_universe":
            return self._screen_universe(args, timestamp)
        elif name == "query_asset":
            return self._query_asset(args, timestamp)
        elif name == "query_position":
            return self._query_position(args, timestamp)
        elif name == "query_history":
            return self._query_history(args, timestamp)
        elif name == "query_market_overview":
            return self._query_market_overview(args, timestamp)
        elif name == "query_fx":
            return self._query_fx(args, timestamp)
        elif name == "query_futures_contract":
            return self._query_futures_contract(args, timestamp)
        elif name == "query_futures_family":
            return self._query_futures_family(args, timestamp)
        else:
            return f"Unknown tool: {name}"


    def _query_futures_family(self, args: dict, timestamp: str) -> str:
        """Query family-level futures variants for the model."""
        family_symbol = args.get("symbol", "GOLD_FUT")
        fields = set(args.get("fields") or [])
        family = get_futures_family_spec(family_symbol)
        if family is None or self._futures_resolver is None:
            return json.dumps({"symbol": family_symbol, "error": "unknown futures family"}, ensure_ascii=False)

        variants = []
        signal = None
        signal_features = None
        for symbol in family.variants:
            product = get_futures_product_spec(symbol)
            resolved = self._futures_resolver.resolve(symbol, timestamp)
            item = {
                "variant": product.variant if product else "standard",
                "symbol": symbol,
                "actual_contract": resolved.contract_ticker,
                "price": resolved.price,
                "multiplier": resolved.multiplier,
                "one_contract_notional_usd": resolved.notional_per_contract,
                "initial_margin_usd": resolved.initial_margin,
                "roll_status": resolved.roll_status,
                "days_to_expiry": resolved.days_to_expiry,
                "previous_session_dollar_volume": resolved.previous_session_dollar_volume,
                "tradable": bool(resolved.contract_ticker and resolved.price),
                "selection_method": resolved.selection_method,
            }
            variants.append(item)
            if item["tradable"] and (signal is None or (item["previous_session_dollar_volume"] or 0) > (signal.get("previous_session_dollar_volume") or 0)):
                signal = item
                signal_features = self._futures_signal_features(symbol, resolved.contract_ticker, timestamp)

        pilot_target_pct_nav = self._futures_pilot_target_pct_nav(variants)

        snapshot = self._get_snapshot()
        positions = []
        if snapshot is not None:
            for pos in getattr(snapshot, "futures_positions", {}).values():
                product = get_futures_product_spec(pos.continuous_symbol)
                if product and product.family_symbol == family.family_symbol:
                    positions.append({
                        "execution_symbol": pos.continuous_symbol,
                        "actual_contract": pos.contract_ticker,
                        "side": pos.side,
                        "contracts": pos.contracts,
                        "notional_usd": pos.notional,
                        "margin_locked": pos.margin_locked,
                    })
        payload = {
            "symbol": family.family_symbol,
            "underlying": family.underlying,
            "signal": signal,
            "signal_features": signal_features,
            "pilot_target_pct_nav": pilot_target_pct_nav,
            "tradable_variants": variants,
            "current_family_exposure": {"positions": positions},
            "risk": {"risk_note": "standard and micro variants are the same family view; execution auto-selects one variant"},
        }
        if fields:
            keep = {"symbol", "underlying"} | fields
            payload = {k: v for k, v in payload.items() if k in keep}
        return json.dumps(payload, ensure_ascii=False)

    def _futures_signal_features(self, symbol: str, contract: str, timestamp: str) -> dict | None:
        if not contract:
            return None
        bars = self._data.load_futures_bars(symbol, contract, "2025-10-01", timestamp)
        snap = self._features.compute(bars, timestamp)
        if snap is None:
            return None
        return {
            "trend": snap.trend,
            "setup": snap.setup,
            "recent_score": snap.recent_score,
            "rsi": round(snap.rsi, 1),
            "chg_1h_pct": round(snap.chg_1h, 2),
            "chg_1d_pct": round(snap.chg_1d, 2),
            "ret_30m_pct": round(snap.ret_30m, 2),
            "atr_pct": round(snap.atr_pct, 2),
        }

    def _futures_pilot_target_pct_nav(self, variants: list[dict]) -> float:
        snapshot = self._get_snapshot()
        nav = getattr(snapshot, "total_nav", 0.0) if snapshot is not None else 0.0
        notionals = [
            float(item["one_contract_notional_usd"])
            for item in variants
            if item.get("tradable") and item.get("one_contract_notional_usd")
        ]
        if nav <= 0 or not notionals:
            return 0.0
        import math
        raw = min(notionals) / nav * 1.05
        return min(0.10, max(0.01, math.ceil(raw * 100) / 100))

    def _query_futures_contract(self, args: dict, timestamp: str) -> str:
        """Query point-in-time actual contract mapping and futures risk metadata."""
        symbol = args.get("continuous_symbol", "GC.FUT")
        fields = set(args.get("fields") or [])
        if self._futures_resolver is None:
            return "(futures not available in current version)"
        resolved = self._futures_resolver.resolve(symbol, timestamp)
        if not resolved.contract_ticker:
            return json.dumps({
                "continuous_symbol": symbol,
                "timestamp": timestamp,
                "status": resolved.roll_status,
                "selection_method": resolved.selection_method,
                "error": "no active futures contract",
            }, ensure_ascii=False)

        payload = {
            "continuous_symbol": symbol,
            "actual_contract": resolved.contract_ticker,
            "timestamp": timestamp,
            "price": resolved.price,
            "multiplier": resolved.multiplier,
            "tick_size": resolved.tick_size,
            "tick_value": resolved.tick_value,
            "notional_per_contract": resolved.notional_per_contract,
            "initial_margin": resolved.initial_margin,
            "maintenance_margin": resolved.maintenance_margin,
            "previous_session_dollar_volume": resolved.previous_session_dollar_volume,
            "previous_session_volume": resolved.previous_session_volume,
            "roll_status": resolved.roll_status,
            "expiry_date": resolved.expiry_date,
            "days_to_expiry": resolved.days_to_expiry,
            "selection_method": resolved.selection_method,
        }
        if fields:
            aliases = {"notional": "notional_per_contract", "margin": "initial_margin", "volume": "previous_session_volume", "dollar_volume": "previous_session_dollar_volume"}
            keep = {"continuous_symbol", "timestamp"}
            for f in fields:
                keep.add(aliases.get(f, f))
            payload = {k: v for k, v in payload.items() if k in keep}
        return json.dumps(payload, ensure_ascii=False)

    def _screen_universe(self, args: dict, timestamp: str) -> str:
        """Screen universe by market, bucket, filters."""
        market_str = args.get("market", "ALL")
        bucket = args.get("bucket", "trend_leaders")
        limit = args.get("limit", 10)
        filters = args.get("filters", {})
        sort_by = args.get("sort_by", "score_total")

        # Determine markets to screen
        if market_str == "ALL":
            markets = [Market.US, Market.HK, Market.CN, Market.CRYPTO, Market.GOLD]
        else:
            market_map = {"US": Market.US, "HK": Market.HK, "CN": Market.CN, "CRYPTO": Market.CRYPTO, "GOLD": Market.GOLD, "FUTURES": Market.FUTURES}
            markets = [market_map.get(market_str, Market.US)]

        snapshot = self._get_snapshot()
        nav = snapshot.total_nav if snapshot else 0
        fx_rates = snapshot.fx_rates if snapshot else {}

        # Load bars and compute indicators
        results = []
        for market in markets:
            symbols = self._data.get_universe_symbols(market)
            for symbol in symbols:
                bars = self._data.load_bars(market, symbol, "2025-10-01", timestamp)
                if not bars:
                    continue

                snap = self._features.compute(bars, timestamp)
                if snap is None:
                    continue

                # Apply filters
                rsi_min = filters.get("rsi_min", 0)
                rsi_max = filters.get("rsi_max", 100)
                if snap.rsi < rsi_min or snap.rsi > rsi_max:
                    continue

                if market == Market.CN and nav > 0:
                    cny_per_usd = fx_rates.get("CNY", 7.25)
                    min_lot_usd = (snap.price * 100) / cny_per_usd
                    if min_lot_usd > nav * 0.045:
                        continue

                risk = self._screen_risk_tag(snap)
                results.append({
                    "symbol": symbol,
                    "market": market.value,
                    "price": snap.price,
                    "rsi": snap.rsi,
                    "trend": snap.trend,
                    "chg_1h": snap.chg_1h,
                    "chg_1d": snap.chg_1d,
                    "atr_pct": snap.atr_pct,
                    "bb_position": snap.bb_position,
                    "ret_30m": snap.ret_30m,
                    "rsi_d1h": snap.rsi_d1h,
                    "trend6": snap.trend6,
                    "setup": snap.setup,
                    "recent_score": snap.recent_score,
                    "risk": risk,
                    "screen_score": self._screen_score(snap, bucket, sort_by, risk),
                })

        # Sort by bucket-aware setup quality instead of raw absolute movement.
        results.sort(key=lambda x: x["screen_score"], reverse=True)

        results = results[:limit]

        # Format output
        lines = [f"[SCREEN] {market_str} | bucket={bucket} | {len(results)} results"]
        lines.append("symbol|mkt|price|score|rsi|trend|1h_chg|1d_chg|atr%|rsi_d1h|ret_30m|trend6|setup|recent_score|risk")
        for r in results:
            lines.append(
                f"{r['symbol']}|{r['market']}|{r['price']:.2f}|"
                f"{r['screen_score']:.2f}|{r['rsi']:.0f}|{r['trend']}|"
                f"{r['chg_1h']:+.2f}|{r['chg_1d']:+.2f}|{r['atr_pct']:.2f}|"
                f"{r['rsi_d1h']:+.0f}|{r['ret_30m']:+.1f}|{r['trend6']}|"
                f"{r['setup']}|{r['recent_score']:+d}|{r['risk']}"
            )
        return "\n".join(lines)

    @staticmethod
    def _screen_risk_tag(snap: IndicatorSnapshot) -> str:
        """Short, non-blocking risk tag for tool output."""
        if snap.rsi > 70:
            return "overbought"
        if snap.chg_1h > 2.0 and (snap.rsi > 62 or snap.bb_position > 0.85):
            return "extended_intraday"
        if snap.rsi < 30 and snap.ret_30m < 0:
            return "falling_knife"
        return ""

    @staticmethod
    def _screen_score(
        snap: IndicatorSnapshot, bucket: str, sort_by: str, risk: str,
    ) -> float:
        """Bucket-aware ranking score for screen_universe.

        The score is only for ordering tool results; it does not remove
        candidates, so stronger models can still choose extended winners.
        """
        setup_rank = {
            "strong_continuation": 4.0,
            "pullback_stabilizing": 3.5,
            "oversold_rebounding": 3.0,
            "weak_actionable": 1.5,
            "weak_no_signal": 0.0,
            "falling_knife": -2.5,
            "extended_overbought": -2.5,
        }.get(snap.setup, 0.0)

        trend_bonus = 1.0 if snap.trend == "UU" else 0.4 if snap.trend in ("UD", "DU") else -0.5
        risk_penalty = 1.5 if risk == "extended_intraday" else 2.5 if risk else 0.0

        if sort_by == "volume_rank":
            return abs(snap.chg_1h) + abs(snap.chg_1d) - risk_penalty
        if sort_by == "cost_efficiency":
            return setup_rank + snap.recent_score * 0.6 - snap.atr_pct * 0.1 - risk_penalty

        if bucket == "pullback_continuation":
            pullback_fit = 1.5 if snap.setup == "pullback_stabilizing" else 0.0
            not_worsening = 0.8 if snap.rsi_d1h >= 0 and snap.ret_30m >= -0.3 else -0.8
            return pullback_fit + setup_rank + not_worsening + snap.recent_score * 0.6 - risk_penalty

        if bucket == "oversold_reversal":
            oversold_fit = 1.5 if snap.setup == "oversold_rebounding" else 0.0
            rsi_fit = 0.8 if 25 <= snap.rsi <= 45 else -0.5
            return oversold_fit + setup_rank + rsi_fit + snap.recent_score * 0.7 - risk_penalty

        if bucket == "low_vol_defensive":
            return setup_rank + trend_bonus + snap.recent_score * 0.5 - snap.atr_pct * 0.3 - risk_penalty

        if bucket == "volume_breakout":
            return setup_rank + trend_bonus + snap.recent_score * 0.7 + max(snap.chg_1h, 0) * 0.2 - risk_penalty

        # trend_leaders/custom/default: prefer actionable setup over raw movement.
        return setup_rank + trend_bonus + snap.recent_score * 0.8 + max(snap.chg_1d, 0) * 0.05 - risk_penalty
    def _query_asset(self, args: dict, timestamp: str) -> str:
        """Query detailed asset info."""
        symbol = args.get("symbol", "")
        fields = args.get("fields", ["quote", "indicators"])
        bar_count = args.get("recent_bar_count", 5)

        if not symbol:
            return "Error: symbol is required"

        market = self._ticker_to_market(symbol)
        if market is None:
            return f"Error: cannot determine market for {symbol}"

        bars = self._data.load_bars(market, symbol, "2025-10-01", timestamp)
        if not bars:
            return f"No data for {symbol}"

        snap = self._features.compute(bars, timestamp)
        if snap is None:
            return f"Insufficient data for {symbol}"

        lines = [f"[ASSET] {symbol} @ {timestamp}"]

        if "quote" in fields:
            lines.append(f"  Price: {snap.price:.2f}")
            lines.append(f"  Chg: 5m={snap.chg_5m:+.2f}% 1h={snap.chg_1h:+.2f}% 1d={snap.chg_1d:+.2f}%")

        if "indicators" in fields:
            lines.append(f"  RSI: {snap.rsi:.1f}  ATR%: {snap.atr_pct:.2f}%  Trend: {snap.trend}")
            lines.append(f"  BB pos: {snap.bb_position:.2f}  RelVol: {snap.rel_volume:.1f}x")

        if "recent_bars" in fields and bar_count > 0:
            recent = bars[-bar_count:]
            lines.append(f"  Recent {len(recent)} bars:")
            for b in recent:
                lines.append(f"    {b.timestamp}: O={b.open:.2f} H={b.high:.2f} L={b.low:.2f} C={b.close:.2f}")

        if "tradability" in fields:
            lines.append(f"  Tradable: yes")  # TODO: check AssetStatusProvider

        if "cost" in fields:
            cost_map = {"US": "3+5 bps", "HK": "15-30 bps", "CN": "5-15 bps", "CRYPTO": "5-20 bps", "GOLD": "~5 bps"}
            lines.append(f"  Est. cost: {cost_map.get(market.value, 'unknown')}")

        return "\n".join(lines)

    def _query_position(self, args: dict, timestamp: str) -> str:
        """Query position details."""
        symbol = args.get("symbol", "")
        include_plan = args.get("include_plan", False)
        include_executions = args.get("include_recent_executions", False)

        snapshot = self._get_snapshot()

        # Find position
        pos = None
        for key, p in snapshot.positions.items():
            if p.symbol == symbol:
                pos = p
                break

        if pos is None:
            return f"No position in {symbol}"

        pnl_pct = ((pos.current_price - pos.avg_cost) / pos.avg_cost * 100) if pos.avg_cost > 0 else 0
        pos_pct = pos.market_value / snapshot.total_nav * 100 if snapshot.total_nav > 0 else 0

        lines = [f"[POSITION] {symbol}"]
        lines.append(f"  Market: {pos.market.value}")
        lines.append(f"  Quantity: {pos.quantity}")
        lines.append(f"  Avg cost: {pos.avg_cost:.2f}")
        lines.append(f"  Current: {pos.current_price:.2f}")
        lines.append(f"  PnL: {pnl_pct:+.1f}% (${pos.unrealized_pnl:+.2f})")
        lines.append(f"  % NAV: {pos_pct:.1f}%")

        if include_plan:
            lines.append(f"  Plan: (no active plan)")  # TODO: get from MemoryManager

        if include_executions:
            lines.append(f"  Recent executions: (not available)")

        return "\n".join(lines)

    def _query_history(self, args: dict, timestamp: str) -> str:
        """Query historical price data."""
        symbol = args.get("symbol", "")
        lookback = args.get("lookback_bars", 48)
        bar_size = args.get("bar_size", "5m")
        fields = args.get("fields", ["ohlcv"])

        if not symbol:
            return "Error: symbol is required"

        market = self._ticker_to_market(symbol)
        if market is None:
            return f"Error: cannot determine market for {symbol}"

        bars = self._data.load_bars(market, symbol, "2025-10-01", timestamp)
        if not bars:
            return f"No data for {symbol}"

        # Adjust lookback based on bar_size
        if bar_size == "15m":
            lookback = lookback * 3
        elif bar_size == "1h":
            lookback = lookback * 12
        elif bar_size == "1d":
            lookback = lookback * 48

        recent = bars[-lookback:]
        if not recent:
            return f"No recent data for {symbol}"

        first = recent[0]
        last = recent[-1]
        change_pct = (last.close - first.close) / first.close * 100 if first.close > 0 else 0

        lines = [f"[HISTORY] {symbol} | {bar_size} | {len(recent)} bars"]
        lines.append(f"  Range: {first.timestamp} → {last.timestamp}")
        lines.append(f"  Change: {change_pct:+.2f}%")

        if "ohlcv" in fields:
            lines.append("  Last 5 bars:")
            for b in recent[-5:]:
                lines.append(f"    {b.timestamp}: O={b.open:.2f} H={b.high:.2f} L={b.low:.2f} C={b.close:.2f}")

        if "returns" in fields:
            returns = []
            for i in range(1, min(6, len(recent))):
                r = (recent[-i].close - recent[-i-1].close) / recent[-i-1].close * 100 if recent[-i-1].close > 0 else 0
                returns.append(f"{r:+.2f}%")
            lines.append(f"  Recent returns: {', '.join(returns)}")

        return "\n".join(lines)

    def _query_market_overview(self, args: dict, timestamp: str) -> str:
        """Query market overview."""
        markets = args.get("markets", ["US", "HK", "CN", "CRYPTO", "GOLD"])
        include_regime = args.get("include_regime", True)
        include_breadth = args.get("include_breadth", True)

        market_map = {"US": Market.US, "HK": Market.HK, "CN": Market.CN, "CRYPTO": Market.CRYPTO, "GOLD": Market.GOLD, "FUTURES": Market.FUTURES}
        snapshot = self._get_snapshot()

        lines = [f"[MARKET_OVERVIEW] @ {timestamp}"]

        for market_str in markets:
            market = market_map.get(market_str)
            if market is None:
                continue

            symbols = self._data.get_universe_symbols(market)
            exposure = snapshot.market_exposure.get(market, 0)

            # Compute market stats
            up_count = 0
            down_count = 0
            total_chg = 0.0
            count = 0

            for symbol in symbols[:50]:  # sample first 50
                bars = self._data.load_bars(market, symbol, "2025-10-01", timestamp)
                if not bars:
                    continue
                snap = self._features.compute(bars, timestamp)
                if snap is None:
                    continue
                if snap.chg_1d > 0:
                    up_count += 1
                else:
                    down_count += 1
                total_chg += snap.chg_1d
                count += 1

            avg_chg = total_chg / count if count > 0 else 0

            lines.append(f"  {market_str}: {len(symbols)} stocks, exposure=${exposure:,.0f}")
            if include_breadth:
                lines.append(f"    Breadth: {up_count}↑/{down_count}↓, avg_chg={avg_chg:+.2f}%")
            if include_regime:
                regime = "GREEN" if avg_chg > -1 else "YELLOW" if avg_chg > -3 else "RED"
                lines.append(f"    Regime: {regime}")

        return "\n".join(lines)

    def _query_fx(self, args: dict, timestamp: str) -> str:
        """Query FX rates."""
        snapshot = self._get_snapshot()
        include_balances = args.get("include_cash_balances", False)
        include_cost = args.get("include_conversion_cost", False)

        lines = [f"[FX] @ {timestamp}"]
        lines.append("  Rates (per 1 USD):")
        for currency, rate in snapshot.fx_rates.items():
            if currency != "USD":
                lines.append(f"    USD/{currency}: {rate:.4f}")

        if include_balances:
            lines.append(f"  Cash: ${snapshot.cash:,.2f}")

        if include_cost:
            lines.append(f"  FX fee: ~5 bps per conversion")

        return "\n".join(lines)

    @staticmethod
    def _ticker_to_market(ticker: str) -> Market | None:
        """Determine market from ticker suffix."""
        if ticker == "XAUUSD.FOREX":
            return Market.GOLD
        if ticker.endswith(".US"):
            return Market.US
        elif ticker.endswith(".HK"):
            return Market.HK
        elif ticker.endswith(".FUT"):
            return Market.FUTURES
        elif ticker.endswith(".CC") or "-" in ticker:
            return Market.CRYPTO
        elif ticker.startswith("sh.") or ticker.startswith("sz."):
            return Market.CN
        return None
