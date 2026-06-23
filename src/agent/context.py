"""
ContextBuilder — assembles LLM prompt context.

4-layer structure (v3):
  Layer 1 [MARKET_RULES]:  Market rules and trading costs (stable, cacheable)
  Layer 2 [UNIVERSE]:      Universe summary + candidate buckets
  Layer 3 [OPEN MARKETS]:  Current market open/closed status
  Layer 4 [PORTFOLIO]:     Current portfolio state + memory

Prompts are loaded from prompts/active/prompts.py via PromptLoader.
"""

from __future__ import annotations

from src.core.types import (
    PortfolioSnapshot, IndicatorSnapshot, Market,
    CandidateBuckets, CandidateInBucket, CandidateBucket,
    MemoryState, DecisionType, RiskMode,
    ActivePlan,
)
from src.core.interfaces import IContextBuilder
from src.data.screener import CandidateScore
from .prompt_loader import PromptLoader


class ContextBuilder(IContextBuilder):
    """Builds LLM prompt messages from portfolio state and market data."""

    def __init__(self, prompt_dir: str | None = None, max_rounds: int = 4):
        self._loader = PromptLoader(prompt_dir)
        self._max_rounds = max_rounds

    def build(
        self, timestamp: str, snapshot: PortfolioSnapshot,
        market_data: str, stock_data: str, alerts: str,
        news: str, round_num: int, tool_results: str,
        trade_feedback: str = "", buy_quota_remaining: int = -1,
    ) -> list[dict[str, str]]:
        """Build messages with cache-friendly ordering.

        Order (most stable → most volatile):
          1. MARKET_RULES    — never changes
          2. UNIVERSE        — never changes
          3. OPEN MARKETS    — changes by time of day
          4. MARKET_SUMMARY  — rolling, every decision
          5. CANDIDATES      — rolling, every decision
          6. PORTFOLIO       — changes after trades
          7. MEMORY_STATE    — changes per decision
          8. TRADE_FEEDBACK  — from previous attempt (retry mode)
          9. TOOL_RESULTS    — changes per round
          10. NEWS           — changes per decision
          11. TIMESTAMP/ROUND/INSTRUCTION — changes per round
        """
        user_blocks = []

        # --- Stable prefix (cacheable across decisions) ---

        # Layer 1: Market Rules (never change during backtest → API cache hit)
        user_blocks.append(self._build_market_rules())
        user_blocks.append("")

        # Layer 2: Universe summary (stable)
        if market_data.strip():
            user_blocks.append(market_data)
            user_blocks.append("")

        # Layer 3: Open markets (changes by time of day)
        open_markets = self._get_open_markets(timestamp)
        user_blocks.append(f"[OPEN MARKETS] {open_markets}")
        user_blocks.append("")

        # FX rates (changes with timestamp)
        fx_info = self._build_fx_and_rules(snapshot)
        user_blocks.append(fx_info)
        user_blocks.append("")

        # Layer 4: Market Summary + Candidates (rolling, changes every decision)
        if stock_data.strip():
            user_blocks.append(stock_data)
            user_blocks.append("")

        # --- Volatile suffix (changes per round/decision) ---

        # Portfolio (changes after trades)
        user_blocks.append(self._format_portfolio(snapshot, buy_quota_remaining))
        user_blocks.append("")

        # Trade feedback from previous attempt (retry mode)
        if trade_feedback.strip():
            user_blocks.append(trade_feedback)
            user_blocks.append("")

        # Tool results from previous round (changes per round)
        if tool_results:
            user_blocks.append(tool_results)
            user_blocks.append("")

        # News (changes per decision)
        if news.strip() and "no news" not in news.lower():
            user_blocks.append(news)
            user_blocks.append("")

        # Alerts
        if alerts.strip() and "(no alerts)" not in alerts:
            user_blocks.append(alerts)
            user_blocks.append("")

        # Timestamp + Round + Instruction (most volatile, always last)
        max_r = self._max_rounds
        user_blocks.append(f"[TIMESTAMP] {timestamp}")
        user_blocks.append(f"[ROUND] {round_num}/{max_r}")

        # Instruction
        if round_num >= max_r:
            template = self._loader.load_final_round_instruction()
        else:
            template = self._loader.load_instruction_template()
        instruction = template.replace("{round_num}", str(round_num)).replace("{max_rounds}", str(max_r))
        user_blocks.append(instruction)

        user_content = "\n".join(user_blocks)
        system_prompt = self._loader.load_system_prompt()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def build_full_decision(
        self,
        timestamp: str,
        snapshot: PortfolioSnapshot,
        market_summary: str,
        buckets: CandidateBuckets,
        memory_state: MemoryState,
        risk_mode: RiskMode,
        open_markets: list[str],
        closed_markets: list[str],
        benchmark_day: int,
        bar_index: int,
        round_num: int,
        tool_results: str = "",
    ) -> list[dict[str, str]]:
        """Build full decision prompt (v3 style).

        4-layer structure with bucketed candidates and memory state.
        """
        user_blocks = []

        # --- Layer 1: MARKET_RULES (stable) ---
        user_blocks.append(self._build_market_rules())
        user_blocks.append("")

        # --- Layer 2: MARKET_SUMMARY ---
        if market_summary.strip():
            user_blocks.append(market_summary)
            user_blocks.append("")

        # --- Layer 3: OPEN MARKETS ---
        open_str = ", ".join(open_markets) if open_markets else "none"
        closed_str = ", ".join(closed_markets) if closed_markets else "none"
        user_blocks.append(f"[OPEN MARKETS] OPEN: {open_str} | CLOSED: {closed_str}")
        user_blocks.append("")

        # --- Layer 4: PORTFOLIO ---
        user_blocks.append(self._format_portfolio_v2(snapshot, risk_mode))
        user_blocks.append("")

        # --- MEMORY_STATE ---
        user_blocks.append(self._format_memory_state(memory_state))
        user_blocks.append("")

        # --- CANDIDATE_BUCKETS ---
        user_blocks.append(self._format_candidate_buckets(buckets))
        user_blocks.append("")

        # --- DECISION_CONTEXT ---
        user_blocks.append(self._format_decision_context(
            timestamp, "full_decision", open_markets, closed_markets,
            benchmark_day, bar_index,
        ))
        user_blocks.append("")

        # --- Tool results ---
        if tool_results:
            user_blocks.append(tool_results)
            user_blocks.append("")

        # --- Round + Instruction ---
        max_r = self._max_rounds
        user_blocks.append(f"[ROUND] {round_num}/{max_r}")
        if round_num >= max_r:
            user_blocks.append(self._loader.load_final_round_instruction())
        else:
            instruction = self._loader.load_instruction_template()
            instruction = instruction.replace("{round_num}", str(round_num)).replace("{max_rounds}", str(max_r))
            user_blocks.append(instruction)

        user_content = "\n".join(user_blocks)
        system_prompt = self._loader.load_system_prompt()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def build_focused_position_decision(
        self,
        timestamp: str,
        snapshot: PortfolioSnapshot,
        symbol: str,
        plan: ActivePlan | None,
        trigger_detail: dict,
        priority: str,
        round_num: int,
        allowed_actions: str = "hold, reduce, close",
        tool_results: str = "",
    ) -> list[dict[str, str]]:
        """Build focused position decision prompt."""
        user_blocks = []

        user_blocks.append("[OBJECTIVE]")
        user_blocks.append("Handle one focused position event. Do not re-evaluate the whole market. Output JSON only.")
        user_blocks.append("")

        user_blocks.append(f"[DECISION_CONTEXT]")
        user_blocks.append(f'decision_type: focused_position_decision')
        user_blocks.append(f'timestamp_utc: {timestamp}')
        user_blocks.append(f'scope: [{symbol}]')
        user_blocks.append("")

        user_blocks.append("[TRIGGER]")
        user_blocks.append(f'symbol: {symbol}')
        user_blocks.append(f'priority: {priority}')
        user_blocks.append(f'detail: {trigger_detail}')
        user_blocks.append("")

        # Position info
        pos = snapshot.positions.get(f"*:{symbol}") or self._find_position(snapshot, symbol)
        if pos:
            pnl_pct = ((pos.current_price - pos.avg_cost) / pos.avg_cost * 100) if pos.avg_cost > 0 else 0
            pos_pct = pos.market_value / snapshot.total_nav * 100 if snapshot.total_nav > 0 else 0
            user_blocks.append("[POSITION]")
            user_blocks.append(f'{pos.symbol}|{pos.market.value}|{pos_pct:.1f}%|{pos.avg_cost:.2f}|{pos.current_price:.2f}|{pnl_pct:+.1f}%|sellable')
            user_blocks.append("")

        if plan:
            user_blocks.append("[PREVIOUS_PLAN]")
            user_blocks.append(f'entry_reason: {plan.entry_reason}')
            user_blocks.append(f'last_review: {plan.last_review_time} @ {plan.last_review_price:.2f}')
            user_blocks.append(f'horizon: {plan.intended_horizon_bars} bars')
            user_blocks.append(f'note: {plan.plan_note}')
            user_blocks.append("")

        user_blocks.append(f"[ALLOWED_ACTIONS] {allowed_actions}")
        user_blocks.append("")

        if tool_results:
            user_blocks.append(tool_results)
            user_blocks.append("")

        max_r = self._max_rounds
        user_blocks.append(f"[ROUND] {round_num}/{max_r}")
        if round_num >= max_r:
            user_blocks.append(self._loader.load_final_round_instruction())

        user_content = "\n".join(user_blocks)
        system_prompt = self._loader.load_system_prompt()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def build_focused_market_decision(
        self,
        timestamp: str,
        snapshot: PortfolioSnapshot,
        event_type: str,
        event_detail: dict,
        scope_market: str,
        round_num: int,
        allowed_actions: str = "hold, reduce, close",
        tool_results: str = "",
    ) -> list[dict[str, str]]:
        """Build focused market/risk decision prompt."""
        user_blocks = []

        user_blocks.append("[OBJECTIVE]")
        user_blocks.append("Handle a market/risk event. Do not do general stock picking unless explicitly allowed. Output JSON only.")
        user_blocks.append("")

        user_blocks.append("[DECISION_CONTEXT]")
        user_blocks.append(f'decision_type: focused_market_or_risk_decision')
        user_blocks.append(f'timestamp_utc: {timestamp}')
        user_blocks.append(f'event_type: {event_type}')
        user_blocks.append(f'scope_market: {scope_market}')
        user_blocks.append("")

        user_blocks.append("[EVENT]")
        user_blocks.append(str(event_detail))
        user_blocks.append("")

        # Relevant positions
        relevant = {k: v for k, v in snapshot.positions.items() if v.market.value == scope_market}
        if relevant:
            user_blocks.append("[RELEVANT_POSITIONS]")
            for key, pos in relevant.items():
                pnl_pct = ((pos.current_price - pos.avg_cost) / pos.avg_cost * 100) if pos.avg_cost > 0 else 0
                user_blocks.append(f'{pos.symbol}|{pos.market.value}|{pos.quantity}sh|{pnl_pct:+.1f}%')
            user_blocks.append("")

        user_blocks.append(f"[ALLOWED_ACTIONS] {allowed_actions}")
        user_blocks.append("")

        if tool_results:
            user_blocks.append(tool_results)
            user_blocks.append("")

        max_r = self._max_rounds
        user_blocks.append(f"[ROUND] {round_num}/{max_r}")
        if round_num >= max_r:
            user_blocks.append(self._loader.load_final_round_instruction())

        user_content = "\n".join(user_blocks)
        system_prompt = self._loader.load_system_prompt()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def build_universe_layer(
        universe: dict[Market, list[str]],
    ) -> str:
        """Build Layer 1: Universe list (ticker + market only)."""
        lines = ["[UNIVERSE] All investable stocks (use query_stock for details):"]
        for market in [Market.US, Market.HK, Market.CN, Market.CRYPTO]:
            tickers = universe.get(market, [])
            if tickers:
                display = ", ".join(tickers)
                lines.append(f"  {market.value}({len(tickers)}): {display}")
        return "\n".join(lines)

    @staticmethod
    def build_candidate_layer(
        candidates: list[CandidateScore],
        detail_count: int = 10,
    ) -> tuple[str, str]:
        """Build Layer 2 (candidate table) and Layer 3 (detail cards).

        Returns: (candidate_table_str, detail_cards_str)
        """
        if not candidates:
            return "(no candidates)", "(no stock data)"

        # Layer 2: Candidate table
        table_lines = ["[CANDIDATES] Top screened stocks (ranked by composite score). You may also trade any stock in [UNIVERSE] — use query_stock for details:"]
        table_lines.append(
            "rank|ticker|mkt|score|price|1h_chg|1d_chg|5d_chg|rsi|trend|recent_6bars"
        )

        for i, c in enumerate(candidates):
            held_mark = " *" if c.sector == "HELD" else ""
            recent = c.recent_bars if c.recent_bars else "N/A"
            table_lines.append(
                f"{i+1:2d}|{c.ticker}{held_mark}|{c.market.value}|{c.composite:.2f}|"
                f"{c.price:.2f}|{c.chg_1h:+.2f}|{c.chg_1d:+.2f}|{c.chg_5d:+.2f}|"
                f"{c.rsi:.0f}|{c.trend}|{recent}"
            )

        # Detail layer is deprecated — Candidate table already contains all metrics.
        # If detail_count > 0, generate compact cards (for backward compatibility).
        detail_lines = []
        if detail_count > 0:
            detail_lines.append("[STOCK_DATA] Top {} detailed (Compact Card):".format(min(detail_count, len(candidates))))
            detail_lines.append("sym|mkt|px|c5m|c1h|c1d|rvol|rsi|atr%|trend|bbp|hlp")
            for c in candidates[:detail_count]:
                card = (
                    f"{c.ticker}|{c.market.value}|{c.price:.2f}|"
                    f"{c.chg_1h:+.2f}|{c.chg_1h:+.2f}|{c.chg_1d:+.2f}|"
                    f"{c.volume_rank:.1f}x|{c.rsi:.0f}|{c.atr:.2f}|"
                    f"{c.trend}|0.50|0.50"
                )
                detail_lines.append(card)

        return "\n".join(table_lines), "\n".join(detail_lines)

    @staticmethod
    def build_trade_feedback(results: list[dict]) -> str:
        """Build trade feedback section with OK/ADJUSTED/FAILED types.

        Args:
            results: list of {"type": "ok"|"adjusted"|"failed", "symbol": str,
                     "market": str, "side": str, "quantity": int,
                     "adjusted_qty": int (for adjusted), "error": str (for failed)}
        """
        if not results:
            return ""

        lines = ["[TRADE_FEEDBACK]"]
        has_failures = False

        for r in results:
            t = r.get("type", "failed")
            side = r.get("side", "?").upper()
            sym = r.get("symbol", "?")
            mkt = r.get("market", "?")
            qty = r.get("quantity", 0)

            if t == "ok":
                lines.append(f"  OK: {side} {qty} {sym}({mkt})")
            elif t == "adjusted":
                adj_qty = r.get("adjusted_qty", qty)
                reason = r.get("reason", "lot rounding")
                lines.append(f"  ADJUSTED: {side} {qty}→{adj_qty} {sym}({mkt}) — {reason}")
            elif t == "failed":
                has_failures = True
                error = r.get("error", "unknown")
                msg = ContextBuilder._humanize_error(error, r)
                lines.append(f"  FAILED: {side} {qty} {sym}({mkt}) — {msg}")

        if has_failures:
            lines.append("")
            lines.append("IMPORTANT: The FAILED trades above CANNOT be executed as-is. Do NOT resubmit them unchanged.")
            lines.append("Either: (a) fix the issue (reduce quantity, wait for market open, etc.), or (b) drop the trade entirely.")
        return "\n".join(lines)

    @staticmethod
    def _humanize_error(error: str, context: dict = None) -> str:
        """Convert system error to human-readable message."""
        if "T+1" in error:
            # Extract sellable quantity from error
            import re
            m = re.search(r"can only sell (\d+) of (\d+)", error)
            if m:
                return f"T+1 restriction: shares bought today cannot be sold until tomorrow. Max sellable today = {m.group(1)}."
            return "T+1 restriction: shares bought today cannot be sold until tomorrow."

        if "insufficient" in error and "funds" in error:
            import re
            currency = "USD"
            m = re.search(r"insufficient (\w+) funds", error)
            if m:
                currency = m.group(1)
            return f"Not enough {currency}. Try reducing allocation or selling other positions first."

        if "25% single position" in error or "single position limit" in error:
            return "Would exceed 25% NAV per position limit. Reduce allocation_pct."

        if "50% market exposure" in error or "market exposure limit" in error:
            import re
            m = re.search(r"current=(\d+)", error)
            current = m.group(1) if m else "?"
            return f"Would exceed 50% market exposure limit. Current exposure too high to add more."

        if "min cash reserve" in error or "cash reserve" in error:
            return "Would breach 5% minimum cash reserve. Cannot deploy more capital."

        if "market_closed" in error or "market_rule" in error:
            return "Market is closed. Check [OPEN MARKETS] for trading hours."

        if "price unavailable" in error:
            return "No price data available for this asset at current timestamp."

        return error

    @staticmethod
    def _build_fx_and_rules(snapshot: PortfolioSnapshot) -> str:
        """Build FX rates info."""
        lines = ["[FX_RATES] Current rates (per 1 USD):"]
        for currency, rate in snapshot.fx_rates.items():
            if currency != "USD":
                lines.append(f"  USD/{currency}: {rate:.4f}")
        lines.append("  FX fee: ~0.05% per conversion")
        return "\n".join(lines)

    @staticmethod
    def _build_market_rules() -> str:
        """Build market rules and trading costs (stable, cacheable)."""
        return """[MARKET_RULES]
US: Currency=USD; Settlement=T+0; Lot=1 share min; No price limits; Halted/delisted non-tradable.
HK: Currency=HKD; Settlement=T+0; Variable lots; Lot rounding auto; No price limits; Halted non-tradable.
CN: Currency=CNY; Buy=T+0; Sell=T+1; Lot=100; Price limit ±10% (STAR ±20%); Halted/limit-locked non-tradable.
Crypto: Currency=USD; 24/7; Fractional ok; Max exposure=25% NAV.
Global: Multi-currency accounts; Auto FX if balance insufficient; Local currency consumed first; Violations reject orders.

[TRADING_COSTS] All prices include costs and slippage.
US: ~0.02%/trade commission, low slippage
HK: ~0.15%-0.30% (commission + fees + stamp), variable lots add cost
CN: ~0.05%-0.15% (commission + taxes), sell costs higher
Crypto: ~0.05%-0.20% (spread + slippage), liquidity dependent
FX: ~0.10%-0.30% per conversion
Avoid trades where expected edge < transaction costs."""

    @staticmethod
    def _get_open_markets(timestamp: str) -> str:
        time_part = timestamp[-5:] if len(timestamp) >= 16 else timestamp[11:16]
        open_list = []
        closed_list = []
        if ("01:30" <= time_part < "04:00") or ("05:00" <= time_part < "08:00"):
            open_list.append("HK")
        else:
            closed_list.append("HK")
        if ("01:30" <= time_part < "03:30") or ("05:00" <= time_part < "07:00"):
            open_list.append("CN")
        else:
            closed_list.append("CN")
        if "14:30" <= time_part < "21:00":
            open_list.append("US")
        else:
            closed_list.append("US")
        open_list.append("CRYPTO")
        result = "OPEN: " + ", ".join(open_list)
        if closed_list:
            result += " | CLOSED: " + ", ".join(closed_list) + " (DO NOT trade these)"
        return result

    @staticmethod
    def build_market_summary_from_universe(
        all_bars: dict[Market, dict[str, list]],
        features,  # FeatureGenerator
        timestamp: str,
        index_returns: dict | None = None,
        open_markets: list[Market] | None = None,
    ) -> str:
        """Build market summary from ALL stocks in each market (full universe).

        Fields: Market|Open|Regime|TradeAllowed|Universe1H|Universe1D|Breadth|Vol

        Args:
            index_returns: {Market: {"symbol": str, "return_1h": float, "return_1d": float}}
            open_markets: list of open markets
        """
        if open_markets is None:
            open_markets = [Market.US, Market.HK, Market.CN, Market.CRYPTO]

        open_market_set = set(open_markets)
        index_defaults = {
            Market.US: "NDX",
            Market.HK: "HSI",
            Market.CN: "SSE50",
            Market.CRYPTO: "BTC",
        }

        lines = ["[MARKET_SUMMARY]"]
        lines.append("Market|Open|Regime|TradeAllowed|Universe1H|Universe1D|Breadth|Vol")

        for market in [Market.US, Market.HK, Market.CN, Market.CRYPTO]:
            market_bars = all_bars.get(market, {})
            is_open = market in open_market_set
            open_str = "yes" if is_open else "no"
            trade_allowed = "yes" if is_open else "no"

            if not market_bars:
                idx = index_defaults.get(market, "?")
                lines.append(f"{market.value}|{open_str}|N/A|{trade_allowed}|N/A|N/A|N/A|N/A")
                continue

            # Compute indicators for ALL stocks in this market
            chg_1h_list = []
            chg_1d_list = []
            rsi_list = []
            atr_list = []

            for symbol, bars in market_bars.items():
                snap = features.compute(bars, timestamp)
                if snap is None:
                    continue
                chg_1h_list.append(snap.chg_1h)
                chg_1d_list.append(snap.chg_1d)
                rsi_list.append(snap.rsi)
                atr_list.append(snap.atr_pct)

            n = len(chg_1h_list)
            if n == 0:
                lines.append(f"{market.value}|{open_str}|N/A|{trade_allowed}|N/A|N/A|N/A|N/A")
                continue

            avg_1h = sum(chg_1h_list) / n
            avg_1d = sum(chg_1d_list) / n
            up_1h = sum(1 for v in chg_1h_list if v > 0)
            down_1h = n - up_1h
            bull_count = sum(1 for v in rsi_list if v > 60)
            bull_ratio = bull_count / n
            avg_atr = sum(atr_list) / n

            # Compute regime: GREEN/YELLOW/RED based on breadth ratio and avg change
            # Use up/(up+down) as breadth ratio, not bull_ratio (RSI > 60)
            total = up_1h + (n - up_1h)
            breadth_ratio = up_1h / total if total > 0 else 0.5

            # Regime logic:
            # GREEN: breadth > 55% AND avg_1h > 0%
            # RED: breadth < 40% OR avg_1h < -0.5%
            # YELLOW: everything else
            if breadth_ratio > 0.55 and avg_1h > 0:
                regime = "GREEN"
            elif breadth_ratio < 0.40 or avg_1h < -0.5:
                regime = "RED"
            else:
                regime = "YELLOW"

            lines.append(
                f"{market.value}|{open_str}|{regime}|{trade_allowed}|"
                f"{avg_1h:+.2f}|{avg_1d:+.2f}|"
                f"{up_1h}/{down_1h}|{avg_atr:.3f}"
            )

        return "\n".join(lines)

    def _format_portfolio(self, snap: PortfolioSnapshot, buy_quota_remaining: int = -1) -> str:
        lines = ["[PORTFOLIO]"]
        nav = snap.total_nav
        cash = snap.cash
        cash_pct = cash / nav * 100 if nav > 0 else 0
        available = max(0, cash - nav * 0.05)

        lines.append(f"NAV: ${nav:,.2f}")
        lines.append(f"Cash: ${cash:,.2f} ({cash_pct:.1f}% of NAV)")
        lines.append(f"Available for new positions: ${available:,.2f}")
        if buy_quota_remaining >= 0:
            lines.append(f"BUY quota remaining today: {buy_quota_remaining}")

        if snap.positions:
            lines.append("")
            lines.append("Positions:")
            frozen_positions = []
            for key, pos in snap.positions.items():
                pos_value = pos.market_value
                pos_pct = pos_value / nav * 100 if nav > 0 else 0
                # Skip ghost/residual positions (quantity <= 0 or allocation < 0.5% NAV)
                if pos.quantity <= 0:
                    continue
                if pos_pct < 0.5:
                    continue  # too small to meaningfully trade — ignore
                pnl_pct = ((pos.current_price - pos.avg_cost) / pos.avg_cost * 100) if pos.avg_cost > 0 else 0
                lines.append(
                    f"  {pos.symbol}({pos.market.value}): {pos.quantity}sh "
                    f"@ ${pos.avg_cost:.2f} -> ${pos.current_price:.2f} "
                    f"PnL={pnl_pct:+.1f}% (${pos_value:,.0f} = {pos_pct:.1f}% NAV)"
                )
                if key in snap.frozen_keys:
                    frozen_positions.append(pos)

            if frozen_positions:
                lines.append("")
                lines.append("Frozen (cannot sell today — T+1 restriction):")
                for pos in frozen_positions:
                    lines.append(f"  {pos.symbol}({pos.market.value}): {pos.quantity}sh — bought today, sellable tomorrow")
        else:
            lines.append("Positions: (none)")

        if snap.market_exposure:
            exposure_str = ", ".join(
                f"{m.value}=${v:,.0f}({v/nav*100:.1f}%)"
                for m, v in snap.market_exposure.items()
            )
            lines.append(f"Market exposure: {exposure_str}")

        return "\n".join(lines)

    @staticmethod
    def _find_position(snapshot: PortfolioSnapshot, symbol: str):
        """Find a position by symbol across all markets."""
        for key, pos in snapshot.positions.items():
            if pos.symbol == symbol:
                return pos
        return None

    def _format_portfolio_v2(self, snap: PortfolioSnapshot, risk_mode: RiskMode) -> str:
        """Format portfolio section (v3 style)."""
        lines = ["[PORTFOLIO]"]
        nav = snap.total_nav
        cash = snap.cash
        cash_pct = cash / nav * 100 if nav > 0 else 0

        lines.append(f"NAV_USD={nav:,.2f}")
        lines.append(f"cash_pct_nav={cash_pct:.1f}%")
        lines.append(f"risk_mode={risk_mode.value}")
        lines.append("")

        if snap.positions:
            lines.append("symbol|mkt|pct_nav|pnl_pct|hold_bars|trend|rsi|sellable|plan_status|risk_note")
            for key, pos in snap.positions.items():
                if pos.quantity <= 0:
                    continue
                pos_pct = pos.market_value / nav * 100 if nav > 0 else 0
                if pos_pct < 0.5:
                    continue
                pnl_pct = ((pos.current_price - pos.avg_cost) / pos.avg_cost * 100) if pos.avg_cost > 0 else 0
                frozen = key in snap.frozen_keys
                sellable = "no" if frozen else "yes"
                lines.append(
                    f"{pos.symbol}|{pos.market.value}|{pos_pct:.1f}%|{pnl_pct:+.1f}%|"
                    f"0|||{sellable}||"
                )
        else:
            lines.append("No positions")

        return "\n".join(lines)

    def _format_memory_state(self, memory: MemoryState) -> str:
        """Format memory state section."""
        lines = ["[MEMORY_STATE]"]

        # Previous daily summary
        if memory.previous_daily_summary:
            ds = memory.previous_daily_summary
            lines.append(f"previous_daily_summary: {ds.market_read}")
            if ds.major_decisions:
                lines.append(f"  major_decisions: {', '.join(ds.major_decisions[:3])}")

        # Daily thesis
        if memory.daily_thesis:
            lines.append(f"daily_thesis: {memory.daily_thesis.text} (conf={memory.daily_thesis.confidence:.2f})")

        # Recent activity
        ra = memory.recent_activity
        if ra.non_hold_decisions:
            lines.append(f"recent_decisions: {'; '.join(ra.non_hold_decisions)}")
        if ra.execution_feedback:
            lines.append(f"recent_feedback: {'; '.join(ra.execution_feedback)}")

        # Watchlist (table format with status)
        if memory.watchlist:
            lines.append("watchlist:")
            lines.append("symbol|condition|met|action_hint")
            for w in memory.watchlist:
                condition = w.reason or ""
                # Simple met check: if condition mentions RSI, we'd need current RSI
                # For now, mark as unknown if not computable
                met = "unknown"
                action_hint = "keep_watch"
                lines.append(f"{w.symbol}|{condition}|{met}|{action_hint}")

        # Avoid list
        if memory.avoid_list:
            lines.append("avoid_list:")
            for a in memory.avoid_list:
                lines.append(f"  {a.symbol}: {a.reason}")

        # Behavior notes
        if memory.rolling_behavior_notes:
            lines.append(f"behavior_notes: {'; '.join(memory.rolling_behavior_notes)}")

        return "\n".join(lines)

    def _format_candidate_buckets(self, buckets: CandidateBuckets) -> str:
        """Format candidate buckets section."""
        lines = ["[CANDIDATE_BUCKETS]"]

        # Helper to format a bucket
        def fmt_bucket(name: str, items: list[CandidateInBucket], fields: str):
            if not items:
                return
            lines.append(f"")
            lines.append(f"# {name}")
            lines.append(fields)
            for c in items:
                lines.append(self._format_candidate_line(c, name))

        # held_positions
        fmt_bucket("held_positions", buckets.held_positions,
                   "symbol|mkt|price|pnl_pct|trend|rsi|risk_note|suggested_action")

        # exit_watch
        fmt_bucket("exit_watch", buckets.exit_watch,
                   "symbol|mkt|price|pnl_pct|rsi|reason|action")

        # trend_leaders (shortened)
        fmt_bucket("trend_leaders", buckets.trend_leaders,
                   "symbol|mkt|price|score|1d_pct|5d_pct|rsi|trend|cost|risk")

        # pullback_continuation (shortened)
        fmt_bucket("pullback_continuation", buckets.pullback_continuation,
                   "symbol|mkt|price|score|1d_pct|5d_pct|rsi|trend|risk")

        # oversold_reversal (shortened)
        fmt_bucket("oversold_reversal", buckets.oversold_reversal,
                   "symbol|mkt|price|score|1d_pct|5d_pct|rsi|trend|risk")

        # low_vol_defensive (shortened)
        fmt_bucket("low_vol_defensive", buckets.low_vol_defensive,
                   "symbol|mkt|price|score|1d_pct|rsi|atr%|cost|risk")

        # crypto_candidates (shortened)
        fmt_bucket("crypto_candidates", buckets.crypto_candidates,
                   "symbol|price|score|1d_pct|rsi|vol|risk")

        # blocked_or_warning
        fmt_bucket("blocked_or_warning", buckets.blocked_or_warning,
                   "symbol|mkt|reason|action")

        return "\n".join(lines)

    @staticmethod
    def _risk_tag(c: CandidateInBucket, bucket_name: str) -> str:
        """Generate short risk tag for a candidate."""
        if c.risk_note and "closed" in c.risk_note.lower():
            return "closed_market"
        if c.risk_note and "reversal" in c.risk_note.lower():
            return "high_reversal_risk"
        if bucket_name == "crypto_candidates":
            return "crypto_beta"
        if c.rsi and c.rsi < 25:
            return "deeply_oversold"
        if c.rsi and c.rsi > 70:
            return "overbought"
        return ""

    def _format_candidate_line(self, c: CandidateInBucket, bucket_name: str) -> str:
        """Format a single candidate line based on bucket type (shortened format)."""
        risk = self._risk_tag(c, bucket_name)

        if bucket_name == "held_positions":
            return (f"{c.ticker}|{c.market.value}|{c.price:.2f}|{c.pnl_pct:+.1f}%|"
                    f"{c.trend}|{c.risk_note}|hold")

        if bucket_name == "exit_watch":
            return (f"{c.ticker}|{c.market.value}|{c.price:.2f}|{c.pnl_pct:+.1f}%|"
                    f"||{c.reason}|{c.allowed_action}")

        if bucket_name == "trend_leaders":
            return (f"{c.ticker}|{c.market.value}|{c.price:.2f}|{c.score:.2f}|"
                    f"{c.chg_1d:+.1f}|{c.chg_5d:+.1f}|{c.rsi:.0f}|{c.trend}|"
                    f"{c.cost_bps:.0f}bps|{risk}")

        if bucket_name == "pullback_continuation":
            return (f"{c.ticker}|{c.market.value}|{c.price:.2f}|{c.score:.2f}|"
                    f"{c.chg_1d:+.1f}|{c.chg_5d:+.1f}|{c.rsi:.0f}|{c.trend}|{risk}")

        if bucket_name == "oversold_reversal":
            return (f"{c.ticker}|{c.market.value}|{c.price:.2f}|{c.score:.2f}|"
                    f"{c.chg_1d:+.1f}|{c.chg_5d:+.1f}|{c.rsi:.0f}|{c.trend}|{risk}")

        if bucket_name == "low_vol_defensive":
            return (f"{c.ticker}|{c.market.value}|{c.price:.2f}|{c.score:.2f}|"
                    f"{c.chg_5d:+.1f}|{c.rsi:.0f}|{c.atr_pct:.2f}|"
                    f"{c.cost_bps:.0f}bps|{risk}")

        if bucket_name == "crypto_candidates":
            return (f"{c.ticker}|{c.price:.2f}|{c.score:.2f}|"
                    f"{c.chg_1d:+.1f}|{c.rsi:.0f}|{c.volatility:.2f}|{risk}")
            return (f"{c.ticker}|{c.price:.2f}|{c.score:.2f}|"
                    f"{c.chg_1h:+.2f}|{c.chg_1d:+.2f}|{c.chg_5d:+.2f}|"
                    f"{c.rsi:.0f}|{c.volatility:.2f}|{c.liquidity:.2f}|{c.risk_note}")

        if bucket_name == "blocked_or_warning":
            return f"{c.ticker}|{c.market.value}|{c.reason}|{c.allowed_action}"

        return ""

    def _format_decision_context(
        self, timestamp: str, decision_type: str,
        open_markets: list[str], closed_markets: list[str],
        benchmark_day: int, bar_index: int,
    ) -> str:
        """Format decision context section."""
        lines = ["[DECISION_CONTEXT]"]
        lines.append(f'decision_type: {decision_type}')
        lines.append(f'timestamp_utc: {timestamp}')
        lines.append(f'benchmark_day: {benchmark_day}')
        lines.append(f'bar_index: {bar_index}')
        lines.append(f'open_markets: {open_markets}')
        lines.append(f'closed_markets: {closed_markets}')
        return "\n".join(lines)
