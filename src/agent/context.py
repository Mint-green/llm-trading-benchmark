"""
ContextBuilder — assembles LLM prompt context.

3-layer structure:
  Layer 1 [UNIVERSE]:     Full list of investable tickers (names only)
  Layer 2 [CANDIDATES]:   Screened 60-70 candidates with scores and key metrics
  Layer 3 [STOCK_DATA]:   Top 10 detailed Compact Cards

Prompts are loaded from prompts/active/prompts.py via PromptLoader.
"""

from __future__ import annotations

from src.core.types import PortfolioSnapshot, IndicatorSnapshot, Market
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
          1. UNIVERSE        — never changes
          2. OPEN MARKETS    — changes by time of day
          3. MARKET_SUMMARY  — rolling, every decision
          4. CANDIDATES      — rolling, every decision
          5. PORTFOLIO       — changes after trades
          6. TRADE_FEEDBACK  — from previous attempt (retry mode)
          7. TOOL_RESULTS    — changes per round
          8. NEWS            — changes per decision
          9. TIMESTAMP/ROUND/INSTRUCTION — changes per round
        """
        user_blocks = []

        # --- Stable prefix (cacheable across decisions) ---

        # Layer 1: Market Rules + Universe (never change during backtest → API cache hit)
        user_blocks.append(self._build_market_rules())
        user_blocks.append("")

        if market_data.strip():
            user_blocks.append(market_data)
            user_blocks.append("")

        # Open markets (changes by time of day, stable within same hour)
        open_markets = self._get_open_markets(timestamp)
        user_blocks.append(f"[OPEN MARKETS] {open_markets}")
        user_blocks.append("")

        # FX rates (changes with timestamp)
        fx_info = self._build_fx_and_rules(snapshot)
        user_blocks.append(fx_info)
        user_blocks.append("")

        # Layer 2: Market Summary + Candidates (rolling, changes every decision)
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
    ) -> str:
        """Build market summary from ALL stocks in each market (full universe).

        Fields: Market|Index|Index1H|Index1D|UniverseAvg1H|UniverseAvg1D|
                UpDown1H|UpDown1D|BullRatio|Volatility|Status

        Args:
            index_returns: {Market: {"symbol": str, "return_1h": float, "return_1d": float}}
        """
        index_defaults = {
            Market.US: "NDX",
            Market.HK: "HSI",
            Market.CN: "SSE50",
            Market.CRYPTO: "BTC",
        }

        lines = ["[MARKET_SUMMARY]"]
        lines.append("Market|Index|Index1H|Index1D|UniverseAvg1H|UniverseAvg1D|UpDown1H|UpDown1D|BullRatio|Volatility|Status")

        for market in [Market.US, Market.HK, Market.CN, Market.CRYPTO]:
            market_bars = all_bars.get(market, {})
            if not market_bars:
                idx = index_defaults.get(market, "?")
                lines.append(f"{market.value}|{idx}|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|CLOSED")
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
                idx = index_defaults.get(market, "?")
                lines.append(f"{market.value}|{idx}|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A|N/A")
                continue

            avg_1h = sum(chg_1h_list) / n
            avg_1d = sum(chg_1d_list) / n
            up_1h = sum(1 for v in chg_1h_list if v > 0)
            down_1h = n - up_1h
            up_1d = sum(1 for v in chg_1d_list if v > 0)
            down_1d = n - up_1d
            bull_count = sum(1 for v in rsi_list if v > 60)
            bull_ratio = bull_count / n
            avg_atr = sum(atr_list) / n
            status = "OPEN"  # would use AssetStatusProvider

            idx = index_defaults.get(market, "?")
            # Get real index returns if available
            idx_1h = "N/A"
            idx_1d = "N/A"
            if index_returns and market in index_returns:
                r = index_returns[market]
                if r.get("return_1h") is not None:
                    idx_1h = f"{r['return_1h']:+.2f}"
                if r.get("return_1d") is not None:
                    idx_1d = f"{r['return_1d']:+.2f}"

            lines.append(
                f"{market.value}|{idx}|{idx_1h}|{idx_1d}|"
                f"{avg_1h:+.2f}|{avg_1d:+.2f}|"
                f"{up_1h}/{down_1h}|{up_1d}/{down_1d}|"
                f"{bull_ratio:.2f}|{avg_atr:.3f}|{status}"
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
