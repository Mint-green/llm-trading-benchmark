"""
AgentRunner — manages multi-round decision flow with LLM.

Flow per timestamp:
  Round 1: Receive full context
  Rounds 2-3: Use tools (function calling)
  Round 4: Mandatory decision (JSON only)

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
        final_decision = Decision(action="hold", reason="max rounds reached")
        use_tools = pre_built_messages is not None  # Enable function calling for v3 mode

        for round_num in range(1, self._config.max_agent_rounds + 1):
            start_time = time.time()

            # Build context
            if pre_built_messages and round_num == 1:
                messages = pre_built_messages
            else:
                messages = self._context.build(
                    timestamp, snapshot, market_data, stock_data,
                    alerts, news, round_num, tool_results,
                    trade_feedback=trade_feedback,
                    buy_quota_remaining=buy_quota_remaining,
                )

            # Call LLM (with or without tools)
            if use_tools and round_num < self._config.max_agent_rounds:
                response_text, tool_calls, prompt_tokens, completion_tokens, reasoning = self._call_llm_with_tools(messages)
            else:
                response_text, prompt_tokens, completion_tokens, reasoning = self._call_llm(messages)
                tool_calls = None

            latency = (time.time() - start_time) * 1000

            # Handle function calling response
            if tool_calls:
                # Execute tool calls
                tool_results, tool_records = self._execute_tool_calls(tool_calls, timestamp)

                # Add controller reminder based on round number
                if round_num == 1:
                    tool_results += "\n\n[SYSTEM] You have used 1 tool round. If no concrete trade or risk action is justified now, produce final JSON. Do not continue broad exploration."
                elif round_num == 2:
                    tool_results += "\n\n[SYSTEM] You have used 2 tool rounds. Further exploration is discouraged. Produce final JSON unless one specific required field is still missing."
                elif round_num >= 3:
                    tool_results += "\n\n[SYSTEM] Tool exploration budget exhausted. Final JSON only. Reject any further tool calls."

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

                # Append tool results to messages for next round
                if pre_built_messages:
                    pre_built_messages = list(pre_built_messages)  # copy
                    pre_built_messages.append({"role": "assistant", "content": response_text, "tool_calls": [
                        {
                            "id": tc.get("id", f"call_{i}"),
                            "type": "function",
                            "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                        }
                        for i, tc in enumerate(tool_calls)
                    ]})
                    pre_built_messages.append({"role": "tool", "tool_call_id": "all", "content": tool_results})
                continue

            # Parse response
            decision = self._parse_decision(response_text)

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
                # Resolve pct_nav to quantities and validate
                resolved_trades = self._resolve_trades(decision.trades, snapshot)
                final_decision = Decision(
                    action="trade",
                    trades=resolved_trades,
                    reason=decision.reason,
                )
                break

            elif decision.action == "query" and round_num < self._config.max_agent_rounds:
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

        # Get tool schemas
        tools = self._tools.get_tool_descriptions()

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

    def _execute_tool_calls(self, tool_calls: list[dict], timestamp: str) -> tuple[str, list[dict]]:
        """Execute tool calls and return formatted results + tool call records.

        Returns:
            (formatted_results, tool_call_records)
            tool_call_records: list of {name, args, result, latency_ms}
        """
        lines = ["[TOOL_RESULT]"]
        records = []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}

            start_time = time.time()
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
                        )
                resolved.append(trade)
                continue

            if trade.allocation_pct is not None:
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
