"""
Debug script — captures one complete decision cycle for analysis.
Outputs: request body, prompt, responses, tool calls, final decision.
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import Config
from src.core.types import DecisionType, Market
from src.agent.runner import AgentRunner
from src.agent.context import ContextBuilder
from src.agent.tools import ToolSystem
from src.data.provider import MarketDataProvider
from src.data.features import FeatureGenerator
from src.data.universe import UniverseRegistry
from src.data.screener import Screener
from src.data.fx_provider import FxProvider
from src.data.index_provider import IndexProvider
from src.data.asset_status import AssetStatusProvider
from src.portfolio.nav import NavEngine
from src.portfolio.constraints import ConstraintEngine
from src.portfolio.market_rules import MarketRuleEngine
from src.portfolio.execution import ExecutionEngine
from src.portfolio.settlement import SettlementEngine
from src.portfolio.portfolio import PortfolioEngine
from src.agent.memory_manager import MemoryManager
from src.platform.scheduler import DecisionScheduler


def capture_one_decision(model_name: str, output_file: str):
    """Capture one complete decision cycle."""
    config = Config.load_from_toml()

    # Setup
    data_provider = MarketDataProvider(config)
    universe = UniverseRegistry(config)
    features = FeatureGenerator()
    asset_status = AssetStatusProvider(config)
    screener = Screener(features)
    fx_provider = FxProvider()
    index_provider = IndexProvider()

    nav_engine = NavEngine(config.fx_rates)
    constraints = ConstraintEngine(config)
    market_rules = MarketRuleEngine(config, asset_status)
    execution = ExecutionEngine(config, nav_engine)
    settlement = SettlementEngine()
    portfolio = PortfolioEngine(
        config, nav_engine, constraints,
        execution, settlement, market_rules,
    )

    context_builder = ContextBuilder(max_rounds=config.max_agent_rounds)
    tool_system = ToolSystem(
        data_provider, features,
        lambda: portfolio.get_snapshot(""),
    )

    def price_lookup(symbol, market, timestamp):
        return None

    agent = AgentRunner(
        config, context_builder, tool_system, portfolio,
        model=model_name, price_lookup=price_lookup,
    )

    memory = MemoryManager()
    scheduler = DecisionScheduler(config)

    # Add mock memory state to simulate mid-day decision
    memory.update_thesis("US market GREEN regime, favor oversold reversals in tech", 0.7, "2026-02-03 15:00")
    memory.add_watch("TEAM.US", "RSI 29, waiting for RSI > 35", timestamp="2026-02-03 15:00")
    memory.add_watch("WDAY.US", "RSI 26, deeply oversold", timestamp="2026-02-03 15:00")
    memory.add_avoid("MSTR.US", "high volatility crypto proxy", timestamp="2026-02-03 16:00")
    memory.record_decision(DecisionType.FULL_DECISION, "hold: no clear entry signal", "2026-02-03 15:00")
    memory.record_decision(DecisionType.FULL_DECISION, "hold: waiting for oversold reversal confirmation", "2026-02-03 16:00")
    memory.record_decision(DecisionType.FULL_DECISION, "hold: TEAM approaching RSI 35 threshold", "2026-02-03 17:00")

    # Load data
    from datetime import datetime, timedelta
    warmup_start = (datetime.strptime(config.backtest_start, "%Y-%m-%d") - timedelta(days=14)).strftime("%Y-%m-%d")

    all_bars = {}
    for market in [Market.US, Market.HK, Market.CN, Market.CRYPTO]:
        all_bars[market] = data_provider.load_all_bars(market, warmup_start, config.backtest_end)

    # Pick a timestamp with open markets (later in the day for memory state)
    ts = "2026-02-03 18:00"  # Later decision with potential memory state

    # Update prices
    prices = {}
    for key, pos in portfolio._positions.items():
        bars = all_bars.get(pos.market, {}).get(pos.symbol, [])
        if bars:
            for bar in reversed(bars):
                if bar.timestamp <= ts:
                    prices[key] = nav_engine.convert_to_usd(bar.close, "USD")
                    break
    portfolio.update_prices(prices)

    snapshot = portfolio.get_snapshot(ts)

    # Build context
    open_markets = scheduler.get_open_markets(ts)
    closed_markets = scheduler.get_closed_markets(ts)

    # Build buckets
    held_info = {}
    buckets = screener.screen_into_buckets(
        all_bars, ts, held_positions=held_info, exit_watch_positions={},
    )

    try:
        index_returns = index_provider.get_all_index_returns(ts)
    except:
        index_returns = {}
    market_summary = ContextBuilder.build_market_summary_from_universe(
        all_bars, features, ts, index_returns=index_returns,
    )
    memory_state = memory.get_memory_state()

    from src.core.types import RiskMode
    messages = context_builder.build_full_decision(
        timestamp=ts,
        snapshot=snapshot,
        market_summary=market_summary,
        buckets=buckets,
        memory_state=memory_state,
        risk_mode=RiskMode.GREEN,
        open_markets=[m.value for m in open_markets],
        closed_markets=[m.value for m in closed_markets],
        benchmark_day=0,
        bar_index=0,
        round_num=1,
    )

    # Capture log
    log = []
    log.append(f"# Decision Cycle Debug Log — {model_name}")
    log.append(f"Timestamp: {ts}")
    log.append(f"Open markets: {[m.value for m in open_markets]}")
    log.append(f"Closed markets: {[m.value for m in closed_markets]}")
    log.append(f"Max rounds: {config.max_agent_rounds}")
    log.append("")

    # Log system prompt
    log.append("## System Prompt")
    log.append("```")
    log.append(messages[0]["content"][:2000])
    log.append("```")
    log.append("")

    # Log user prompt (full)
    log.append("## User Prompt (full)")
    log.append("```")
    log.append(messages[1]["content"])
    log.append("```")
    log.append("")

    # Run agent and capture each round
    log.append("## Decision Rounds")
    log.append("")

    # Monkey-patch to capture API calls
    original_call = agent._call_llm_with_tools
    original_execute = agent._execute_tool_calls

    round_count = [0]

    def capturing_call(msgs):
        round_count[0] += 1
        result = original_call(msgs)
        response_text, tool_calls, prompt_tokens, completion_tokens, reasoning = result

        log.append(f"### Round {round_count[0]}")
        log.append(f"Prompt tokens: {prompt_tokens}")
        log.append(f"Completion tokens: {completion_tokens}")
        log.append(f"")
        log.append(f"**Response:**")
        log.append("```")
        log.append(response_text[:1000] if response_text else "(empty)")
        log.append("```")
        log.append("")

        if tool_calls:
            log.append(f"**Tool Calls ({len(tool_calls)}):**")
            for tc in tool_calls:
                func = tc.get("function", {})
                log.append(f"- {func.get('name', '?')}: {func.get('arguments', '{}')[:200]}")
            log.append("")

        return result

    def capturing_execute(tool_calls, timestamp):
        result_text, records = original_execute(tool_calls, timestamp)

        log.append(f"**Tool Results:**")
        log.append("```")
        log.append(result_text[:2000])
        log.append("```")
        log.append("")

        return result_text, records

    agent._call_llm_with_tools = capturing_call
    agent._execute_tool_calls = capturing_execute

    # Run decision
    decision, rounds = agent.run(
        ts, snapshot, "", "", "", "",
        pre_built_messages=messages,
    )

    # Log final decision
    log.append("## Final Decision")
    log.append(f"Action: {decision.action}")
    log.append(f"Reason: {decision.reason}")
    if decision.trades:
        log.append("Trades:")
        for t in decision.trades:
            log.append(f"  - {t.side.value} {t.symbol} ({t.allocation_pct:.1%})")
    log.append("")

    log.append(f"Total rounds: {round_count[0]}")
    log.append(f"Total LLM calls: {len(rounds)}")

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log))

    print(f"Saved to {output_file}")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
    output = sys.argv[2] if len(sys.argv) > 2 else f"output/debug_{model}.md"
    capture_one_decision(model, output)
