"""
AgentRunner — manages multi-round decision flow with LLM.

Flow per timestamp:
  Round 1: Receive full context
  Round 2: Use tools (function calling)
  Round 3: Mandatory decision (JSON only) for v3 full/light decisions

Supports two modes:
  1. Legacy mode: prompt-based tool calls (query action)
  2. v3 mode: native function calling + pre-built messages
"""

from __future__ import annotations
import json
import time
from openai import OpenAI

from src.core.types import (
    Market, OrderSide, PortfolioSnapshot, Decision, AgentRound, TradeOrder,
)
from src.core.interfaces import IAgentRunner, IContextBuilder, IToolSystem
from src.core.config import Config
from src.portfolio.portfolio import PortfolioEngine


class AgentRunner(IAgentRunner):
    """Runs the multi-round agent decision loop."""

    def __init__(
        self,
        config: Config,
        context_builder: IContextBuilder,
        tool_system: IToolSystem,
        portfolio: PortfolioEngine,
        model: str = "mimo-v2.5-pro",
        price_lookup: object = None,
    ):
        self._config = config
        self._model_name = model
        self._context = context_builder
        self._tools = tool_system
        self._price_lookup = price_lookup
        self._portfolio = portfolio

        if "deepseek" in model:
            self._client = OpenAI(
                api_key=config.deepseek_api_key,
                base_url=config.deepseek_base_url,
                timeout=config.deepseek_timeout,
            )
            self._api_model = config.deepseek_model
            self._max_tokens = config.deepseek_max_tokens
        else:
            # Default to mimo-pro
            self._client = OpenAI(
                api_key=config.mimo_pro_api_key,
                base_url=config.mimo_pro_base_url,
                timeout=config.mimo_pro_timeout,
            )
            self._api_model = config.mimo_pro_model
            self._max_tokens = config.mimo_pro_max_tokens

    def run(
        self, timestamp: str, snapshot: PortfolioSnapshot,
        market_data: str, stock_data: str, alerts: str, news: str,
        trade_feedback: str = "", buy_quota_remaining: int = -1,
        pre_built_messages: list[dict] | None = None,
    ) -> tuple[Decision, list[AgentRound]]:
        """Run agent loop. Returns final decision and round history.

        Args:
            pre_built_messages: If provided, use these messages instead of building from context.
                               Used for v3-style prompts (full_decision, focused_position, etc.)
        """
        rounds: list[AgentRound] = []
        tool_results = ""
        tool_history: list[str] = []
        base_messages = list(pre_built_messages) if pre_built_messages else None
        if base_messages and trade_feedback.strip():
            base_messages.append({
                "role": "user",
                "content": "[TRADE_FEEDBACK]\n" + trade_feedback.strip() +
                "\n[SYSTEM] Recent forced stop-loss or failed execution reduces risk appetite. Do not immediately replace a stopped position unless the setup is exceptional."
            })
        final_decision = Decision(action="hold", reason="max rounds reached")
        use_tools = pre_built_messages is not None  # Enable function calling for v3 mode
        decision_type = self._decision_type_from_messages(base_messages or [])
        effective_max_rounds = self._effective_max_rounds(decision_type)

        for round_num in range(1, effective_max_rounds + 1):
            start_time = time.time()

            # Build context
            if base_messages and round_num == 1:
                messages = base_messages
            elif base_messages:
                messages = list(base_messages)
                if tool_history:
                    messages.append({"role": "user", "content": "\n\n".join(tool_history)})

                max_r = effective_max_rounds
                if round_num >= max_r:
                    instruction = self._context._loader.load_final_round_instruction()
                else:
                    instruction = self._context._loader.load_instruction_template()
                    instruction = instruction.replace("{round_num}", str(round_num)).replace("{max_rounds}", str(max_r))
                messages.append({"role": "user", "content": f"[ROUND] {round_num}/{max_r}\n{instruction}"})
            else:
                messages = self._context.build(
                    timestamp, snapshot, market_data, stock_data,
                    alerts, news, round_num, tool_results,
                    trade_feedback=trade_feedback,
                    buy_quota_remaining=buy_quota_remaining,
                )

            # Call LLM (with or without tools)
            if use_tools and round_num < effective_max_rounds:
                response_text, tool_calls, prompt_tokens, completion_tokens, reasoning = self._call_llm_with_tools(messages)
            else:
                response_text, prompt_tokens, completion_tokens, reasoning = self._call_llm(messages)
                tool_calls = None

            latency = (time.time() - start_time) * 1000

            # Handle function calling response
            if tool_calls:
                # Execute tool calls
                tool_results, tool_records = self._execute_tool_calls(tool_calls, timestamp, messages)

                # Add controller reminder based on round number
                if round_num == 1:
                    tool_results += "\n\n[SYSTEM] You have used 1 tool round. If no concrete trade or risk action is justified now, produce final JSON. Do not continue broad exploration."
                elif round_num == 2:
                    tool_results += "\n\n[SYSTEM] You have used 2 tool rounds. Further exploration is discouraged. Produce final JSON unless one specific required field is still missing."
                elif round_num >= 3:
                    tool_results += "\n\n[SYSTEM] Tool exploration budget exhausted. Final JSON only. Reject any further tool calls."
                if base_messages:
                    tool_history.append(tool_results)

                # Record round
                decision = Decision(action="query", reason="tool calls")
                round_data = AgentRound(
                    round_num=round_num,
                    decision=decision,
                    tool_results=tool_results,
                    llm_response=response_text or f"[{len(tool_calls)} tool calls]",
                    latency_ms=latency,
                    tokens_used=prompt_tokens + completion_tokens,
                )
                round_data._prompt_tokens = prompt_tokens
                round_data._completion_tokens = completion_tokens
                round_data._reasoning = reasoning
                round_data._tool_records = tool_records  # Store for logging
                rounds.append(round_data)

                continue


            # Parse response
            decision = self._parse_decision(response_text)
            decision = self._apply_trade_feedback_guards(decision, trade_feedback)

            round_data = AgentRound(
                round_num=round_num,
                decision=decision,
                tool_results=tool_results,
                llm_response=response_text,
                latency_ms=latency,
                tokens_used=prompt_tokens + completion_tokens,
            )
            rounds.append(round_data)

            # Store token info and reasoning for external logging
            round_data._prompt_tokens = prompt_tokens
            round_data._completion_tokens = completion_tokens
            round_data._reasoning = reasoning

            if decision.action == "trade":
                final_decision = self._finalize_trade_decision(decision, snapshot, timestamp)
                break

            elif decision.action == "query" and round_num < effective_max_rounds:
                # Execute queries (legacy mode)
                tool_results = self._execute_queries(decision.queries, timestamp)
                continue

            elif decision.action == "hold":
                final_decision = decision
                break

            else:
                # Unknown action or query on last round → hold
                final_decision = Decision(action="hold", reason=f"invalid action: {decision.action}")
                break

        return final_decision, rounds

    def _finalize_trade_decision(
        self, decision: Decision, snapshot: PortfolioSnapshot, timestamp: str,
    ) -> Decision:
        """Resolve and filter trades while preserving model memory/plan updates."""
        resolved_trades = self._resolve_trades(decision.trades, snapshot)
        resolved_trades, filter_reason = self._filter_cooling_blocked_sells(
            resolved_trades, snapshot, timestamp,
        )
        resolved_trades, lot_filter_reason = self._filter_zero_lot_buys(resolved_trades)
        resolved_trades, buy_filter_reason = self._filter_constraint_blocked_buys(
            resolved_trades, snapshot,
        )
        resolved_trades, futures_filter_reason = self._filter_existing_futures_buys(
            resolved_trades, snapshot,
        )
        reason = decision.reason
        for extra_reason in (filter_reason, lot_filter_reason, buy_filter_reason, futures_filter_reason):
            if extra_reason:
                reason = f"{reason} | {extra_reason}".strip()
        if not resolved_trades:
            return Decision(
                action="hold",
                reason=reason,
                memory_updates=decision.memory_updates,
                plan_updates=decision.plan_updates,
            )

        return Decision(
            action="trade",
            trades=resolved_trades,
            reason=reason,
            memory_updates=decision.memory_updates,
            plan_updates=decision.plan_updates,
        )

    @staticmethod
    def _apply_trade_feedback_guards(decision: Decision, trade_feedback: str) -> Decision:
        """Apply deterministic guards derived from just-executed trade feedback."""
        if decision.action != "trade" or not decision.trades or "AUTO SELL" not in trade_feedback:
            return decision

        stopped_markets = {
            market
            for market in Market
            if f"({market.value})" in trade_feedback
        }
        if not stopped_markets:
            return decision

        filtered_trades = [
            trade for trade in decision.trades
            if not (trade.side == OrderSide.BUY and trade.market in stopped_markets)
        ]
        removed = len(decision.trades) - len(filtered_trades)
        if removed == 0:
            return decision

        reason = (
            f"{decision.reason} | filtered {removed} same-market BUY(s) after auto stop-loss"
        ).strip()
        if not filtered_trades:
            return Decision(
                action="hold",
                reason=reason,
                memory_updates=decision.memory_updates,
                plan_updates=decision.plan_updates,
            )

        return Decision(
            action="trade",
            trades=filtered_trades,
            reason=reason,
            memory_updates=decision.memory_updates,
            plan_updates=decision.plan_updates,
        )

    def _filter_cooling_blocked_sells(
        self, trades: list[TradeOrder], snapshot: PortfolioSnapshot, timestamp: str,
    ) -> tuple[list[TradeOrder], str]:
        """Drop LLM SELLs that would be rejected by the cooling-period constraint."""
        if not trades or self._portfolio is None:
            return trades, ""

        constraints = getattr(self._portfolio, "_constraints", None)
        if constraints is None:
            return trades, ""

        filtered: list[TradeOrder] = []
        blocked = []
        for trade in trades:
            if trade.side != OrderSide.SELL:
                filtered.append(trade)
                continue

            key = f"{trade.market.value}:{trade.symbol}"
            ok, reason = constraints.validate_sell(
                key, trade.quantity, snapshot.positions, timestamp=timestamp,
            )
            if not ok and reason.startswith("cooling period"):
                blocked.append(f"{trade.symbol}({trade.market.value})")
                continue
            filtered.append(trade)

        if not blocked:
            return trades, ""
        return filtered, f"filtered {len(blocked)} cooling-blocked SELL(s): {', '.join(blocked)}"

    def _filter_constraint_blocked_buys(
        self, trades: list[TradeOrder], snapshot: PortfolioSnapshot,
    ) -> tuple[list[TradeOrder], str]:
        """Drop BUY orders that current portfolio constraints would reject."""
        if not trades or self._portfolio is None:
            return trades, ""

        constraints = getattr(self._portfolio, "_constraints", None)
        if constraints is None:
            return trades, ""

        filtered: list[TradeOrder] = []
        blocked = []
        for trade in trades:
            if trade.side != OrderSide.BUY or trade.market == Market.FUTURES or trade.asset_type == "futures":
                filtered.append(trade)
                continue

            price = self._get_price_from_snapshot(trade.symbol, trade.market, snapshot)
            if price is None or price <= 0:
                filtered.append(trade)
                continue

            to_usd = getattr(self._portfolio, "_to_usd", None)
            constraint_price = to_usd(price, trade.market) if to_usd else price
            ok, reason = constraints.validate_buy(
                trade.symbol, trade.market, trade.quantity, constraint_price,
                snapshot.total_nav, snapshot.positions,
            )
            if not ok:
                blocked.append(f"{trade.symbol}({trade.market.value}: {reason})")
                continue
            filtered.append(trade)

        if not blocked:
            return trades, ""
        return filtered, f"filtered {len(blocked)} constraint-blocked BUY(s): {', '.join(blocked)}"

    @staticmethod
    def _filter_existing_futures_buys(
        trades: list[TradeOrder], snapshot: PortfolioSnapshot,
    ) -> tuple[list[TradeOrder], str]:
        """Drop futures BUYs that would try to increase an existing family position."""
        if not trades or not snapshot.futures_positions:
            return trades, ""

        filtered: list[TradeOrder] = []
        blocked = []
        for trade in trades:
            is_futures_buy = (
                trade.side == OrderSide.BUY
                and (trade.market == Market.FUTURES or trade.asset_type == "futures")
            )
            key = f"FUTURES:{trade.symbol}"
            if is_futures_buy and key in snapshot.futures_positions:
                blocked.append(trade.symbol)
                continue
            filtered.append(trade)

        if not blocked:
            return trades, ""
        return filtered, f"filtered {len(blocked)} existing-futures BUY(s): {', '.join(blocked)}"

    def _filter_zero_lot_buys(self, trades: list[TradeOrder]) -> tuple[list[TradeOrder], str]:
        """Drop BUY orders that execution lot rounding would reduce to zero."""
        if not trades or self._portfolio is None:
            return trades, ""

        execution = getattr(self._portfolio, "_execution", None)
        round_lots = getattr(execution, "_round_lots", None)
        if round_lots is None:
            return trades, ""

        filtered: list[TradeOrder] = []
        blocked = []
        for trade in trades:
            if trade.side != OrderSide.BUY:
                filtered.append(trade)
                continue

            rounded_qty = round_lots(trade.market, trade.symbol, trade.quantity, trade.side)
            if rounded_qty <= 0:
                blocked.append(f"{trade.symbol}({trade.market.value}, requested {trade.quantity})")
                continue
            filtered.append(trade)

        if not blocked:
            return trades, ""
        return filtered, f"filtered {len(blocked)} zero-lot BUY(s): {', '.join(blocked)}"

    def _effective_max_rounds(self, decision_type: str) -> int:
        """Cap v3 decisions by type to control latency and exploratory churn."""
        if decision_type == "light_decision":
            return 1
        if decision_type == "full_decision":
            return min(self._config.max_agent_rounds, self._config.full_decision_max_rounds)
        return self._config.max_agent_rounds

    @staticmethod
    def _decision_type_from_messages(messages: list[dict]) -> str:
        """Extract decision_type from the prompt context, if present."""
        for msg in messages:
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if not isinstance(content, str) or "decision_type:" not in content:
                continue
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("decision_type:"):
                    value = line.split(":", 1)[1].strip()
                    return value.strip('"').strip("'")
        return ""

    def _call_llm(self, messages: list[dict]) -> tuple[str, int, int, str]:
        """Call LLM and return (response_text, prompt_tokens, completion_tokens, reasoning_content)."""
        extra = {}
        if not self._config.thinking_enabled:
            extra["extra_body"] = {"thinking": {"type": "disabled"}}

        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self._api_model,
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=self._config.temperature,
                    **extra,
                )
                msg = resp.choices[0].message
                content = msg.content or ""
                reasoning = getattr(msg, "reasoning_content", None) or ""
                if content:
                    usage = resp.usage
                    prompt_tokens = usage.prompt_tokens if usage else 0
                    completion_tokens = usage.completion_tokens if usage else 0
                    return content, prompt_tokens, completion_tokens, reasoning
            except Exception as e:
                print(f"  [LLM] Attempt {attempt+1} failed: {e}")
                time.sleep(3)
        return "", 0, 0, ""

    def _call_llm_with_tools(self, messages: list[dict]) -> tuple[str, list | None, int, int, str]:
        """Call LLM with function calling support.

        Returns:
            (response_text, tool_calls_or_none, prompt_tokens, completion_tokens, reasoning)
        """
        extra = {}
        if not self._config.thinking_enabled:
            extra["extra_body"] = {"thinking": {"type": "disabled"}}

        # Get tool schemas. Full/light prompts already include market breadth/open status,
        # so do not expose broad market-overview tools on those decision types.
        tools = self._tools.get_tool_descriptions()
        decision_type = self._decision_type_from_messages(messages)
        if decision_type in ("full_decision", "light_decision"):
            tools = [
                tool for tool in tools
                if tool.get("function", {}).get("name") != "query_market_overview"
            ]

        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self._api_model,
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=self._config.temperature,
                    tools=tools,
                    tool_choice="auto",
                    **extra,
                )
                msg = resp.choices[0].message
                content = msg.content or ""
                reasoning = getattr(msg, "reasoning_content", None) or ""
                usage = resp.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0

                # Check for tool calls
                if msg.tool_calls:
                    tool_calls = []
                    for tc in msg.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {}
                        tool_calls.append({
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": json.dumps(args),
                            },
                        })
                    return content, tool_calls, prompt_tokens, completion_tokens, reasoning

                # No tool calls — return content
                return content, None, prompt_tokens, completion_tokens, reasoning

            except Exception as e:
                print(f"  [LLM] Attempt {attempt+1} failed: {e}")
                time.sleep(3)

        return "", None, 0, 0, ""

    def _execute_tool_calls(
        self, tool_calls: list[dict], timestamp: str, messages: list[dict] | None = None,
    ) -> tuple[str, list[dict]]:
        """Execute tool calls and return formatted results + tool call records.

        Returns:
            (formatted_results, tool_call_records)
            tool_call_records: list of {name, args, result, latency_ms}
        """
        lines = ["[TOOL_RESULT]"]
        records = []
        decision_type = self._decision_type_from_messages(messages or [])
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}

            start_time = time.time()
            if name == "query_market_overview" and decision_type in ("full_decision", "light_decision"):
                result = "query_market_overview denied: current MARKET_SUMMARY already provides breadth and open status for full/light decisions."
            else:
                result = self._tools.execute_tool(name, args, timestamp)
            latency = (time.time() - start_time) * 1000

            records.append({
                "name": name,
                "args": args,
                "result": result,
                "latency_ms": latency,
            })

            # Format result
            if len(result) < 120 and "\n" not in result:
                lines.append(f"[{name}] {result}")
            else:
                lines.append(f"[{name}]")
                for rline in result.split("\n"):
                    lines.append(f"  {rline}")

        return "\n".join(lines), records

    def _parse_decision(self, text: str) -> Decision:
        """Parse LLM output into Decision."""
        from .protocol import DecisionProtocol
        protocol = DecisionProtocol()
        decision = protocol.parse(text)
        if decision is None:
            return Decision(action="hold", reason="parse error")
        return decision

    def _resolve_trades(
        self, trades: list[TradeOrder], snapshot: PortfolioSnapshot,
    ) -> list[TradeOrder]:
        """Resolve allocation_pct to share quantities using current prices."""
        resolved = []
        for trade in trades:
            if trade.quantity > 0 and trade.allocation_pct is None:
                # Already has explicit quantity — cap sell at held quantity
                if trade.side.value == "sell":
                    key = f"{trade.market.value}:{trade.symbol}"
                    pos = snapshot.positions.get(key)
                    if pos:
                        trade = TradeOrder(
                            symbol=trade.symbol,
                            market=trade.market,
                            side=trade.side,
                            quantity=min(trade.quantity, pos.quantity),
                            allocation_pct=trade.allocation_pct,
                            reason=trade.reason,
                            asset_type=trade.asset_type,
                            action=trade.action,
                            futures_side=trade.futures_side,
                            target_notional_pct_nav=trade.target_notional_pct_nav,
                            max_margin_pct_nav=trade.max_margin_pct_nav,
                            risk_budget_pct_nav=trade.risk_budget_pct_nav,
                            unit_hint=trade.unit_hint,
                            risk_trigger=trade.risk_trigger,
                        )
                resolved.append(trade)
                continue

            if trade.allocation_pct is not None:
                # target_pct_nav=0 is the v3 "close position" signal.
                # Resolve it to the full held quantity instead of a zero-share sell.
                if trade.side.value == "sell" and trade.allocation_pct == 0:
                    key = f"{trade.market.value}:{trade.symbol}"
                    pos = snapshot.positions.get(key)
                    if pos and pos.quantity > 0:
                        resolved.append(TradeOrder(
                            symbol=trade.symbol,
                            market=trade.market,
                            side=trade.side,
                            quantity=pos.quantity,
                            allocation_pct=trade.allocation_pct,
                            reason=trade.reason,
                            asset_type=trade.asset_type,
                            action=trade.action,
                            futures_side=trade.futures_side,
                            target_notional_pct_nav=trade.target_notional_pct_nav,
                            max_margin_pct_nav=trade.max_margin_pct_nav,
                            risk_budget_pct_nav=trade.risk_budget_pct_nav,
                            unit_hint=trade.unit_hint,
                            risk_trigger=trade.risk_trigger,
                        ))
                        continue
                # Cap allocation_pct at 25% (hard limit)
                alloc = min(trade.allocation_pct, 0.25)

                # Get current price from snapshot positions or data
                price = self._get_price_from_snapshot(trade.symbol, trade.market, snapshot)
                if price and price > 0:
                    # allocation_pct of NAV -> USD amount -> shares
                    nav = snapshot.total_nav
                    usd_amount = nav * alloc
                    from src.portfolio.portfolio import MARKET_CURRENCY
                    currency = MARKET_CURRENCY.get(trade.market, "USD")
                    local_amount = self._portfolio._nav.convert_from_usd(usd_amount, currency)
                    if trade.market in (Market.CRYPTO, Market.GOLD):
                        quantity = round(local_amount / price, 8)
                    else:
                        quantity = int(local_amount / price)

                    # Cap sell at held quantity
                    if trade.side.value == "sell":
                        key = f"{trade.market.value}:{trade.symbol}"
                        pos = snapshot.positions.get(key)
                        if pos:
                            quantity = min(quantity, pos.quantity)

                    if quantity > 0:
                        resolved.append(TradeOrder(
                            symbol=trade.symbol,
                            market=trade.market,
                            side=trade.side,
                            quantity=quantity,
                            allocation_pct=alloc,
                            reason=trade.reason,
                            asset_type=trade.asset_type,
                            action=trade.action,
                            futures_side=trade.futures_side,
                            target_notional_pct_nav=trade.target_notional_pct_nav,
                            max_margin_pct_nav=trade.max_margin_pct_nav,
                            risk_budget_pct_nav=trade.risk_budget_pct_nav,
                            unit_hint=trade.unit_hint,
                            risk_trigger=trade.risk_trigger,
                        ))
                        continue

            # Fallback: keep original (will likely fail validation)
            resolved.append(trade)

        return resolved

    def _get_price_from_snapshot(self, symbol: str, market: Market, snapshot: PortfolioSnapshot) -> float | None:
        """Get current price from snapshot positions or price lookup."""
        # Check held positions first
        key = f"{market.value}:{symbol}"
        pos = snapshot.positions.get(key)
        if pos and pos.current_price > 0:
            return pos.current_price

        # Use price lookup (from all_bars)
        if self._price_lookup:
            try:
                return self._price_lookup(symbol, market, snapshot.timestamp)
            except Exception:
                pass
        return None

    def _execute_queries(self, queries: list[dict], timestamp: str) -> str:
        """Execute tool queries and return formatted results.

        Format:
          [TOOL_RESULT]
          [query_stock][AAPL.US] Price: $264.62, RSI: 43...
          [query_stock][MSFT.US] Price: $471.26, RSI: 52...
          [query_news][policy] Fed announces rate decision...
        """
        lines = ["[TOOL_RESULT]"]
        for q in queries:
            tool_name = q.get("tool", "")
            args = q.get("args", {})
            # Format args as compact string (e.g. "AAPL.US" or "keyword(policy)")
            arg_str = self._format_args(tool_name, args)
            result = self._tools.execute_tool(tool_name, args, timestamp)
            # Single-line if short, multi-line if long
            if len(result) < 120 and "\n" not in result:
                lines.append(f"[{tool_name}][{arg_str}] {result}")
            else:
                lines.append(f"[{tool_name}][{arg_str}]")
                for rline in result.split("\n"):
                    lines.append(f"  {rline}")
        return "\n".join(lines)

    @staticmethod
    def _format_args(tool_name: str, args: dict) -> str:
        """Format tool args into a compact display string."""
        if not args:
            return ""
        if tool_name == "query_stock":
            return args.get("ticker", "")
        elif tool_name == "query_history":
            return f"{args.get('ticker','')},{args.get('days','')}d"
        elif tool_name == "query_news":
            return args.get("keyword", "")
        else:
            return ",".join(f"{k}={v}" for k, v in args.items())
