"""
ExperimentRunner — orchestrates a complete benchmark run.

Wires all modules together and runs the backtest loop.
"""

from __future__ import annotations
import json
import os
import time
from typing import Any

from src.core.config import Config
from src.core.types import (
    Market, OrderSide, PortfolioSnapshot, TradeOrder, Decision,
    AgentRound, BenchmarkResult, DecisionType, RiskMode,
)
from src.data.provider import MarketDataProvider
from src.data.universe import UniverseRegistry
from src.data.features import FeatureGenerator
from src.data.asset_status import AssetStatusProvider
from src.data.screener import Screener
from src.data.fx_provider import FxProvider
from src.data.index_provider import IndexProvider
from src.portfolio.nav import NavEngine
from src.portfolio.constraints import ConstraintEngine
from src.portfolio.market_rules import MarketRuleEngine
from src.portfolio.execution import ExecutionEngine
from src.portfolio.settlement import SettlementEngine
from src.portfolio.portfolio import PortfolioEngine
from src.portfolio.trigger_engine import TriggerEngine
from src.agent.context import ContextBuilder
from src.agent.tools import ToolSystem
from src.agent.runner import AgentRunner
from src.agent.memory_manager import MemoryManager
from src.evaluation.metrics import MetricsEngine
from src.evaluation.behavior import BehaviorAnalyzer
from .logging import ExperimentLogger
from .scheduler import DecisionScheduler, DecisionRequest
from .event_detector import EventDetector


class ExperimentRunner:
    """Orchestrates a complete benchmark experiment."""

    def __init__(self, config: Config, model: str = "mimo-v2.5-pro", db_path: str = "output/results/benchmark.db"):
        self.config = config
        self._model = model
        self._db_path = db_path
        self._last_light_decision = ""
        self._setup()

    def _setup(self) -> None:
        """Wire all modules together."""
        # Data layer
        self.data_provider = MarketDataProvider(self.config)
        self.universe = UniverseRegistry(self.config)
        self.features = FeatureGenerator()
        self.asset_status = AssetStatusProvider(self.config)
        self.screener = Screener(self.features)
        forex_db = os.path.join(self.config.stock_data_dir, "FOREX_stock.db")
        self.fx_provider = FxProvider(db_path=forex_db)
        self.index_provider = IndexProvider(data_dir=self.config.stock_data_dir)

        # Portfolio layer
        self.nav_engine = NavEngine(self.config.fx_rates)
        self.constraints = ConstraintEngine(self.config)
        self.market_rules = MarketRuleEngine(self.config, self.asset_status)
        self.execution = ExecutionEngine(self.config, self.nav_engine)
        self.settlement = SettlementEngine()
        self.portfolio = PortfolioEngine(
            self.config, self.nav_engine, self.constraints,
            self.execution, self.settlement, self.market_rules,
        )

        # Trigger engine
        self.trigger_engine = TriggerEngine(
            self.config.trigger_config, self.config.crypto_trigger_config,
        )

        # Memory manager
        self.memory = MemoryManager()

        # Decision scheduler
        self.scheduler = DecisionScheduler(self.config)

        # Event detector
        self.event_detector = EventDetector(self.config, self.trigger_engine, self.features)

        # Agent layer
        self.context_builder = ContextBuilder(max_rounds=self.config.max_agent_rounds)
        self.tool_system = ToolSystem(
            self.data_provider, self.features,
            lambda: self.portfolio.get_snapshot(""),
        )
        # Price lookup: get latest price from bars data
        def price_lookup(symbol: str, market: Market, timestamp: str) -> float | None:
            bars = self._all_bars_cache.get(market, {}).get(symbol, [])
            for bar in reversed(bars):
                if bar.timestamp <= timestamp:
                    return bar.close
            return None

        self.agent = AgentRunner(
            self.config, self.context_builder, self.tool_system, self.portfolio,
            model=self._model, price_lookup=price_lookup,
        )

        # Evaluation
        self.metrics = MetricsEngine()
        self.behavior = BehaviorAnalyzer()

        # Logging
        self.logger = ExperimentLogger(self._db_path)

        # Cache for price lookup
        self._all_bars_cache: dict[Market, dict[str, list]] = {}

        # Risk mode
        self._risk_mode = RiskMode.GREEN

    def run(self) -> BenchmarkResult:
        """Run the complete benchmark."""
        print(f"Starting benchmark: {self.config.backtest_start} → {self.config.backtest_end}")
        print(f"Model: {self.agent._api_model}")
        print(f"Initial NAV: ${self.config.initial_cash:,.2f}")

        # Initialize logger with enhanced run tracking
        self.logger.init_run(
            config_dict=self.config.to_dict(),
            model=self._model,
            start_date=self.config.backtest_start,
            end_date=self.config.backtest_end,
            interval_min=self.config.decision_interval,
            initial_cash=self.config.initial_cash,
            thinking_enabled=self.config.thinking_enabled,
            total_decisions=0,  # will be updated after timestamps generated
        )

        # Load all data upfront with warmup period for indicator computation
        warmup_days = 14  # need ~14 bars for RSI warmup, load 14 days of history
        from datetime import datetime, timedelta
        warmup_start = (datetime.strptime(self.config.backtest_start, "%Y-%m-%d") - timedelta(days=warmup_days)).strftime("%Y-%m-%d")

        print(f"Loading market data (warmup from {warmup_start})...")
        all_bars: dict[Market, dict[str, list]] = {}
        for market in [Market.US, Market.HK, Market.CN, Market.CRYPTO]:
            all_bars[market] = self.data_provider.load_all_bars(
                market, warmup_start, self.config.backtest_end,
            )
            total_bars = sum(len(b) for b in all_bars[market].values())
            print(f"  {market.value}: {len(all_bars[market])} symbols, {total_bars} bars")

        # Cache for price lookup
        self._all_bars_cache = all_bars

        # Generate timestamps (every 5 minutes)
        timestamps = self._generate_timestamps()
        print(f"Decision points: {len(timestamps)}")

        # Run backtest
        all_rounds: list[AgentRound] = []
        all_decisions: list[dict] = []
        decision_count = 0
        portfolio_history: list[PortfolioSnapshot] = []  # collect snapshots for metrics

        # Daily summary tracking
        daily_start_nav = self.config.initial_cash
        daily_decisions: list[dict] = []
        daily_session_summaries: list = []
        last_daily_summary_date = ""

        fx_rates = dict(self.config.fx_rates)  # initial FX rates

        # Timing accumulators
        _t_fx = _t_reset = _t_prices = _t_snap = _t_markets = _t_sched = _t_event = _t_ctx = _t_llm = _t_log = 0.0

        for i, ts in enumerate(timestamps):
            import time as _time

            # Update FX rates at this timestamp
            _t0 = _time.time()
            new_rates = self.fx_provider.get_all_rates(ts)
            if new_rates:
                fx_rates = new_rates
                self.nav_engine.update_rates(fx_rates)
            _t_fx += _time.time() - _t0

            # Reset per-decision state
            _t0 = _time.time()
            self.portfolio._constraints.reset_decision_state(ts)
            _t_reset += _time.time() - _t0

            # Update portfolio prices
            _t0 = _time.time()
            self._update_prices(ts, all_bars)
            _t_prices += _time.time() - _t0

            # Snapshot (with real FX rates)
            _t0 = _time.time()
            snapshot = self.portfolio.get_snapshot(ts)
            snapshot.fx_rates = fx_rates
            _t_snap += _time.time() - _t0

            # Get open/closed markets
            _t0 = _time.time()
            open_markets = self.scheduler.get_open_markets(ts)
            closed_markets = self.scheduler.get_closed_markets(ts)
            _t_markets += _time.time() - _t0

            # Daily summary at 00:05 UTC (before any market logic)
            from datetime import datetime as _dt
            time_part = ts[11:16] if len(ts) >= 16 else ""
            if time_part == "00:05" and daily_decisions:
                from src.evaluation.summary_engine import SummaryEngine
                from datetime import timedelta as _td
                summary_date = ts[:10]
                prev_date = (_dt.strptime(summary_date, "%Y-%m-%d") - _td(days=1)).strftime("%Y-%m-%d")
                engine = SummaryEngine()
                nav_end = self.portfolio.nav
                daily_summary = engine.generate_daily_summary(
                    date=prev_date,
                    nav_start=daily_start_nav,
                    nav_end=nav_end,
                    all_decisions=daily_decisions,
                    all_trades=list(self.portfolio.trade_history),
                    session_summaries=daily_session_summaries,
                    snapshot=snapshot,
                    plans=[],
                )
                self.memory.save_daily_summary(daily_summary)
                print(f"  Daily summary: {prev_date} | return={daily_summary.daily_return_pct:+.2%} | NAV=${nav_end:,.0f}")
                daily_start_nav = nav_end
                daily_decisions = []
                daily_session_summaries = []
                last_daily_summary_date = summary_date

            # Skip weekends (stock markets closed)
            try:
                _d = _dt.strptime(ts[:16], "%Y-%m-%d %H:%M")
                if _d.weekday() >= 5:
                    continue
            except ValueError:
                continue

            # Session summary at market close + 5min (after weekend skip)
            just_closed = self._just_closed_market(ts)
            if just_closed:
                summary = self._generate_session_summary(just_closed, ts, snapshot)
                if summary:
                    daily_session_summaries.append(summary)

            # --- Decision scheduling ---
            _t0 = _time.time()
            plans = self.memory.get_all_plans()

            # Quick scheduler pre-check (no state update, no event detection)
            has_stock_market = any(m in open_markets for m in [Market.US, Market.HK, Market.CN])
            has_crypto_position = any(
                pos.market == Market.CRYPTO for pos in snapshot.positions.values()
            )
            might_need_decision = has_stock_market and self.scheduler.needs_decision(ts, open_markets)
            _t_sched += _time.time() - _t0

            # Crypto-only period: schedule light_decision at lower frequency
            if not has_stock_market:
                # Check if it's time for a light_decision
                light_interval = 60 if has_crypto_position else 240  # 1h or 4h
                time_part = ts[11:16] if len(ts) >= 16 else ""
                # Only trigger at :00 of each hour (or every 4th hour)
                if time_part == "00:00" or (light_interval == 240 and int(time_part[:2]) % 4 == 0):
                    if not self._last_light_decision or \
                       self.scheduler._minutes_since(ts, self._last_light_decision) >= light_interval:
                        self._last_light_decision = ts
                        # Create light decision request
                        decision_request = DecisionRequest(
                            timestamp=ts,
                            decision_type=DecisionType.LIGHT_DECISION,
                            priority="P3",
                        )
                        # Fall through to light decision processing
                    else:
                        # Not time yet — skip
                        self.logger.log_snapshot(snapshot)
                        portfolio_history.append(snapshot)
                        if i % 12 == 0:
                            print(f"[{ts}] auto_hold (crypto-wait) | NAV=${self.portfolio.nav:,.0f}", flush=True)
                        continue
                else:
                    # Not at a light_decision boundary — skip
                    self.logger.log_snapshot(snapshot)
                    portfolio_history.append(snapshot)
                    if i % 12 == 0:
                        print(f"[{ts}] auto_hold (crypto-wait) | NAV=${self.portfolio.nav:,.0f}", flush=True)
                    continue

            # AUTO_HOLD fast path — timestamp doesn't need a decision (stock market)
            elif not might_need_decision:
                # If no plans exist, no triggers possible — skip entirely
                if not plans:
                    self.logger.log_snapshot(snapshot)
                    portfolio_history.append(snapshot)
                    if i % 12 == 0:
                        print(f"[{ts}] auto_hold | NAV=${self.portfolio.nav:,.0f}", flush=True)
                    continue

                # Check plan triggers only (lightweight, no volatility spike)
                _t0 = _time.time()
                trigger_events, _, _ = self.event_detector.detect(
                    ts, snapshot, all_bars, plans, open_markets, closed_markets, self._risk_mode,
                    lightweight=True,
                )
                _t_event += _time.time() - _t0
                decision_request = self.scheduler.schedule(
                    ts, open_markets, closed_markets, trigger_events, self._risk_mode,
                )
                if decision_request.decision_type == DecisionType.AUTO_HOLD:
                    # Still AUTO_HOLD after trigger check — skip
                    self._update_plan_peaks(plans, snapshot)
                    self.memory.expire_thesis(ts)
                    self.memory.expire_watchlist(ts)
                    self.memory.expire_avoid(ts)
                    self.logger.log_snapshot(snapshot)
                    portfolio_history.append(snapshot)
                    if i % 12 == 0:
                        print(f"[{ts}] auto_hold | NAV=${self.portfolio.nav:,.0f}", flush=True)
                    continue
                # else: trigger event found — fall through to full processing

            else:
                # Non-AUTO_HOLD from pre-check: run full event detection
                _t0 = _time.time()
                trigger_events, market_events, risk_events = self.event_detector.detect(
                    ts, snapshot, all_bars, plans, open_markets, closed_markets, self._risk_mode,
                    lightweight=True,
                )
                _t_event += _time.time() - _t0
                decision_request = self.scheduler.schedule(
                    ts, open_markets, closed_markets, trigger_events, self._risk_mode,
                )

            # --- Common processing for non-AUTO_HOLD ---
            self._update_plan_peaks(plans, snapshot)
            self.memory.expire_thesis(ts)
            self.memory.expire_watchlist(ts)
            self.memory.expire_avoid(ts)

            self.portfolio._constraints.set_tail_guard(
                decision_request.tail_guard_active,
                decision_request.tail_guard_markets,
            )

            # Final AUTO_HOLD check (in case scheduler changed decision)
            if decision_request.decision_type == DecisionType.AUTO_HOLD:
                self.logger.log_snapshot(snapshot)
                portfolio_history.append(snapshot)
                if i % 12 == 0:
                    print(f"[{ts}] auto_hold | NAV=${self.portfolio.nav:,.0f}", flush=True)
                continue

            # Build alerts (position RSI extremes only — actionable signals)
            alerts = self._build_alerts(ts, all_bars)

            # Get index returns
            index_returns = self.index_provider.get_all_index_returns(ts)

            # Build context based on decision type
            _t_ctx_start = _time.time()
            if decision_request.decision_type == DecisionType.LIGHT_DECISION:
                # Light decision for crypto-only periods
                # Build simplified context with only crypto candidates
                held_info = {}
                for key, pos in snapshot.positions.items():
                    if pos.market == Market.CRYPTO:
                        pnl_pct = ((pos.current_price - pos.avg_cost) / pos.avg_cost) if pos.avg_cost > 0 else 0
                        pos_pct = pos.market_value / snapshot.total_nav if snapshot.total_nav > 0 else 0
                        held_info[pos.symbol] = {
                            "market": pos.market,
                            "price": pos.current_price,
                            "score": 0.0,
                            "pnl_pct": pnl_pct,
                            "pct_nav": pos_pct,
                            "hold_bars": 0,
                            "sellable": key not in snapshot.frozen_keys,
                            "tradable": True,  # crypto is always tradable
                            "plan_status": "",
                            "risk_note": "",
                        }

                # Only screen crypto market
                buckets = self.screener.screen_into_buckets(
                    all_bars, ts, held_positions=held_info,
                    open_markets=[Market.CRYPTO],
                )

                # Build crypto-only market summary
                market_summary = ContextBuilder.build_market_summary_from_universe(
                    all_bars, self.features, ts, index_returns=index_returns,
                    open_markets=[Market.CRYPTO],
                )

                memory_state = self.memory.get_memory_state()

                # Build light decision context
                messages = self.context_builder.build_full_decision(
                    timestamp=ts,
                    snapshot=snapshot,
                    market_summary=market_summary,
                    buckets=buckets,
                    memory_state=memory_state,
                    risk_mode=self._risk_mode,
                    open_markets=["CRYPTO"],
                    closed_markets=["US", "HK", "CN"],
                    benchmark_day=0,
                    bar_index=i,
                    round_num=1,
                )

            elif decision_request.decision_type == DecisionType.FULL_DECISION:
                # Use v3-style bucketed context
                held_info = {}
                exit_info = {}
                open_market_set = set(open_markets)
                for key, pos in snapshot.positions.items():
                    pnl_pct = ((pos.current_price - pos.avg_cost) / pos.avg_cost) if pos.avg_cost > 0 else 0
                    pos_pct = pos.market_value / snapshot.total_nav if snapshot.total_nav > 0 else 0
                    is_tradable = pos.market in open_market_set
                    held_info[pos.symbol] = {
                        "market": pos.market,
                        "price": pos.current_price,
                        "score": 0.0,
                        "pnl_pct": pnl_pct,
                        "pct_nav": pos_pct,
                        "hold_bars": 0,
                        "sellable": key not in snapshot.frozen_keys,
                        "tradable": is_tradable,
                        "plan_status": "",
                        "risk_note": "market_closed" if not is_tradable else "",
                    }

                buckets = self.screener.screen_into_buckets(
                    all_bars, ts, held_positions=held_info, exit_watch_positions=exit_info,
                    open_markets=open_markets,
                )

                market_summary = ContextBuilder.build_market_summary_from_universe(
                    all_bars, self.features, ts, index_returns=index_returns,
                    open_markets=open_markets,
                )

                memory_state = self.memory.get_memory_state()

                messages = self.context_builder.build_full_decision(
                    timestamp=ts,
                    snapshot=snapshot,
                    market_summary=market_summary,
                    buckets=buckets,
                    memory_state=memory_state,
                    risk_mode=self._risk_mode,
                    open_markets=[m.value for m in open_markets],
                    closed_markets=[m.value for m in closed_markets],
                    benchmark_day=0,
                    bar_index=i,
                    round_num=1,
                )
            else:
                # Focused decision — use simpler context
                universe_dict = {
                    Market.US: self.universe.get_symbols(Market.US),
                    Market.HK: self.universe.get_symbols(Market.HK),
                    Market.CN: self.universe.get_symbols(Market.CN),
                    Market.CRYPTO: self.universe.get_symbols(Market.CRYPTO),
                }
                universe_str = ContextBuilder.build_universe_layer(universe_dict)

                market_summary = ContextBuilder.build_market_summary_from_universe(
                    all_bars, self.features, ts, index_returns=index_returns,
                )

                held_tickers = {pos.symbol: pos.market for pos in snapshot.positions.values()}
                candidates = self.screener.screen(all_bars, ts, held_tickers=held_tickers)
                candidate_str, _ = ContextBuilder.build_candidate_layer(candidates, detail_count=0)
                stock_context = market_summary + "\n\n" + candidate_str

                # Build focused prompt
                if decision_request.scope_symbols:
                    symbol = decision_request.scope_symbols[0]
                    plan = self.memory.get_plan(symbol)
                    trigger_detail = {}
                    if decision_request.trigger_events:
                        te = decision_request.trigger_events[0]
                        trigger_detail = te.trigger_detail

                    messages = self.context_builder.build_focused_position_decision(
                        timestamp=ts,
                        snapshot=snapshot,
                        symbol=symbol,
                        plan=plan,
                        trigger_detail=trigger_detail,
                        priority=decision_request.priority,
                        round_num=1,
                    )
                else:
                    messages = self.context_builder.build_focused_market_decision(
                        timestamp=ts,
                        snapshot=snapshot,
                        event_type=decision_request.trigger_events[0].trigger_type.value if decision_request.trigger_events else "unknown",
                        event_detail=decision_request.trigger_events[0].trigger_detail if decision_request.trigger_events else {},
                        scope_market=decision_request.scope_market,
                        round_num=1,
                    )

            _t_ctx = _time.time() - _t_ctx_start

            # Run agent
            trade_feedback = ""
            decision = Decision(action="hold", reason="no decision")
            rounds = []

            for attempt in range(2):  # max 2 attempts (1 initial + 1 retry)
                # Use the agent's run method with pre-built context
                _t_llm_start = _time.time()
                decision, rounds = self.agent.run(
                    ts, snapshot, "", "", "", "",
                    trade_feedback=trade_feedback,
                    buy_quota_remaining=self.portfolio._constraints.daily_buys_remaining,
                    pre_built_messages=messages,
                )
                _t_llm += _time.time() - _t_llm_start
                all_rounds.extend(rounds)

                # Log LLM call details, agent rounds, and tool calls
                for r in rounds:
                    prompt_t = getattr(r, '_prompt_tokens', 0)
                    compl_t = getattr(r, '_completion_tokens', 0)
                    reasoning = getattr(r, '_reasoning', '')
                    self.logger.log_llm_call(
                        ts, r.round_num, self.agent._api_model,
                        prompt_t, compl_t, r.latency_ms,
                        reasoning=reasoning, response=r.llm_response,
                    )
                    # Log agent round
                    self.logger.log_round(r, timestamp=ts)
                    # Log tool calls (if any)
                    tool_records = getattr(r, '_tool_records', [])
                    for tr in tool_records:
                        self.logger.log_tool_call(
                            timestamp=ts,
                            tool_name=tr["name"],
                            tool_args=tr["args"],
                            tool_result=tr["result"],
                            latency_ms=tr["latency_ms"],
                        )

                # Execute trades and collect results
                if decision.action == "trade":
                    feedback_results = []
                    has_failures = False
                    for trade in decision.trades:
                        requested_qty = trade.quantity
                        price = self._get_price(trade.symbol, trade.market, ts, all_bars)
                        if price:
                            result = self.portfolio.process_order(trade, price, ts)
                            self.logger.log_trade(result, ts)
                            if result.success:
                                executed_qty = result.order.quantity
                                if executed_qty != requested_qty:
                                    feedback_results.append({
                                        "type": "adjusted",
                                        "symbol": trade.symbol,
                                        "market": trade.market.value,
                                        "side": trade.side.value,
                                        "quantity": requested_qty,
                                        "adjusted_qty": executed_qty,
                                        "reason": f"lot rounding (requested {requested_qty}, executed {executed_qty})",
                                    })
                                else:
                                    feedback_results.append({
                                        "type": "ok",
                                        "symbol": trade.symbol,
                                        "market": trade.market.value,
                                        "side": trade.side.value,
                                        "quantity": executed_qty,
                                    })
                            else:
                                has_failures = True
                                feedback_results.append({
                                    "type": "failed",
                                    "symbol": trade.symbol,
                                    "market": trade.market.value,
                                    "side": trade.side.value,
                                    "quantity": trade.quantity,
                                    "error": result.error,
                                })
                        else:
                            has_failures = True
                            feedback_results.append({
                                "type": "failed",
                                "symbol": trade.symbol,
                                "market": trade.market.value,
                                "side": trade.side.value,
                                "quantity": trade.quantity,
                                "error": "price unavailable",
                            })

                    # Only retry if there are FAILED trades (not just ADJUSTED)
                    if not has_failures or attempt == 1:
                        break

                    trade_feedback = ContextBuilder.build_trade_feedback(feedback_results)
                    snapshot = self.portfolio.get_snapshot(ts)
                    snapshot.fx_rates = fx_rates
                else:
                    break

            # Log
            self.logger.log_decision(ts, decision, snapshot)
            self.logger.log_snapshot(self.portfolio.get_snapshot(ts))
            portfolio_history.append(self.portfolio.get_snapshot(ts))

            # Calculate total LLM calls and latency for this decision
            calls_this_decision = len(rounds)
            latency_ms = sum(r.latency_ms for r in rounds)

            # Log decision event (v3)
            execution_result = json.dumps({
                "trades": [{"symbol": t.symbol, "side": t.side.value, "qty": t.quantity} for t in decision.trades],
            }) if decision.trades else "{}"
            self.logger.log_decision_event(
                timestamp=ts,
                decision_type=decision_request.decision_type.value,
                raw_output=rounds[-1].llm_response if rounds else "",
                parsed_output=json.dumps({"action": decision.action, "reason": decision.reason}),
                execution_result=execution_result,
                latency_ms=int(latency_ms),
            )

            decision_count += 1

            # Build trades string
            trades_str = ""
            if decision.action == "trade" and decision.trades:
                trades_parts = []
                for t in decision.trades:
                    pct = f"({t.allocation_pct:.1%})" if t.allocation_pct else ""
                    trades_parts.append(f"{t.side.value.upper()} {t.symbol}{pct}")
                trades_str = ", ".join(trades_parts)

            # Collect decision records
            decision_record = {
                "timestamp": ts,
                "action": decision.action,
                "symbol": trades_str or "hold",
                "market": decision_request.scope_market,
            }
            all_decisions.append(decision_record)
            daily_decisions.append(decision_record)

            # Record decision in memory
            self.memory.record_decision(
                decision_request.decision_type,
                f"{decision.action}: {trades_str or 'hold'}",
                ts,
            )

            # Apply LLM memory_updates and plan_updates
            if decision.memory_updates:
                self.memory.apply_memory_updates(decision.memory_updates, ts)
            if decision.plan_updates:
                self.memory.apply_plan_updates(decision.plan_updates, ts)

            # Structured progress output
            print(
                f"[{ts}] {decision_request.decision_type.value} | "
                f"calls={calls_this_decision} | latency={latency_ms/1000:.1f}s | "
                f"action={decision.action} | trades={trades_str} | "
                f"NAV=${self.portfolio.nav:,.0f}",
                flush=True,
            )

            # Update progress in database (every decision)
            total_trades = len(self.portfolio.trade_history)
            successful_trades = sum(1 for t in self.portfolio.trade_history if t.success)
            self.logger.update_progress(
                last_decision_ts=ts,
                decisions_made=decision_count,
                current_nav=self.portfolio.nav,
                total_trades=total_trades,
                successful_trades=successful_trades,
            )

            if self.config.max_decisions > 0 and decision_count >= self.config.max_decisions:
                print(f"  Reached max decisions ({self.config.max_decisions})")
                break

        # Daily summary for the last day (if not already generated at 00:05)
        if daily_decisions:
            from src.evaluation.summary_engine import SummaryEngine
            last_date = timestamps[-1][:10] if timestamps else ""
            engine = SummaryEngine()
            nav_end = self.portfolio.nav
            daily_summary = engine.generate_daily_summary(
                date=last_date,
                nav_start=daily_start_nav,
                nav_end=nav_end,
                all_decisions=daily_decisions,
                all_trades=list(self.portfolio.trade_history),
                session_summaries=daily_session_summaries,
                snapshot=self.portfolio.get_snapshot(timestamps[-1] if timestamps else ""),
                plans=[],
            )
            self.memory.save_daily_summary(daily_summary)
            print(f"  Daily summary (final): {last_date} | return={daily_summary.daily_return_pct:+.2%} | NAV=${nav_end:,.0f}")

        # Timing summary
        n_ts = len(timestamps)
        n_dec = decision_count
        print(f"\n=== Timing Breakdown ({n_ts} timestamps, {n_dec} decisions) ===")
        print(f"  FX rates:     {_t_fx:.1f}s ({_t_fx/n_ts*1000:.1f}ms/ts)")
        print(f"  Snapshot:     {_t_snap:.1f}s ({_t_snap/n_ts*1000:.1f}ms/ts)")
        print(f"  Scheduler:    {_t_sched:.1f}s ({_t_sched/n_ts*1000:.1f}ms/ts)")
        if n_dec > 0:
            print(f"  Context:      {_t_ctx:.1f}s ({_t_ctx/n_dec:.1f}s/decision)")
            print(f"  LLM calls:    {_t_llm:.1f}s ({_t_llm/n_dec:.1f}s/decision)")
        _t_total = _t_fx + _t_snap + _t_sched + _t_ctx + _t_llm
        print(f"  TOTAL:        {_t_total:.1f}s ({_t_total/60:.1f}min)")

        # Final snapshot
        final_snapshot = self.portfolio.get_snapshot(timestamps[-1] if timestamps else "")

        # Append final snapshot to portfolio_history for accurate metrics
        if portfolio_history:
            # Only append if final snapshot is different from last in history
            if portfolio_history[-1].timestamp != final_snapshot.timestamp:
                portfolio_history.append(final_snapshot)
        else:
            portfolio_history = [final_snapshot]

        # Compute results
        trades = self.portfolio.trade_history
        metrics = self.metrics.compute(portfolio_history, trades)
        behavior = self.behavior.analyze(all_rounds, trades)

        result = BenchmarkResult(
            model_name=self.agent._api_model,
            dataset_version=self.config.dataset_version,
            start_date=self.config.backtest_start,
            end_date=self.config.backtest_end,
            initial_nav=self.config.initial_cash,
            final_nav=final_snapshot.total_nav,
            total_return=metrics["total_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            total_trades=metrics["total_trades"],
            win_rate=metrics["win_rate"],
            avg_holding_bars=0,
            total_decisions=decision_count,
            rejected_orders=behavior["trade_analysis"]["rejected"],
            total_llm_tokens=0,
            total_llm_calls=len(all_rounds),
            decision_log=all_decisions,
            portfolio_history=portfolio_history,
        )

        self.logger.save_results(result)
        self.logger.mark_completed()
        self.logger.close()

        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Model: {result.model_name}")
        print(f"Period: {result.start_date} → {result.end_date}")
        print(f"Initial NAV: ${result.initial_nav:,.2f}")
        print(f"Final NAV:   ${result.final_nav:,.2f}")
        print(f"Return:      {result.total_return:+.2%}")
        print(f"Sharpe:      {result.sharpe_ratio:.4f}")
        print(f"Max DD:      {result.max_drawdown:.2f}%")
        print(f"Trades:      {result.total_trades}")
        print(f"Win Rate:    {result.win_rate:.1f}%")
        print(f"Decisions:   {result.total_decisions}")
        print(f"LLM Calls:   {result.total_llm_calls}")
        print(f"Behavior:    {behavior['trade_analysis']}")
        print("=" * 60)

        return result

    def _any_market_open(self, ts: str) -> bool:
        for market in [Market.US, Market.HK, Market.CN, Market.CRYPTO]:
            if market == Market.CRYPTO:
                return True  # always open
            ok, _ = self.asset_status.get_status(market, "", ts)
            if ok:
                return True
        return False

    def _any_stock_market_open(self, ts: str) -> bool:
        """Check if any non-crypto market is open (respects trading days)."""
        # Weekend check: stock markets closed Sat/Sun, crypto remains 24/7
        from datetime import datetime
        try:
            dt = datetime.strptime(ts[:16], "%Y-%m-%d %H:%M")
            if dt.weekday() >= 5:  # Saturday=5, Sunday=6
                return False
        except ValueError:
            pass
        for market in [Market.US, Market.HK, Market.CN]:
            ok, _ = self.asset_status.get_status(market, "", ts)
            if ok:
                return True
        return False

    def _is_end_of_session(self, ts: str) -> bool:
        """Check if this is the session summary trigger point."""
        return self._just_closed_market(ts) is not None

    def _just_closed_market(self, ts: str) -> Market | None:
        """Return the market that just closed (5min ago), or None."""
        time_part = ts[11:16] if len(ts) >= 16 else ""
        close_map = {"07:05": Market.CN, "08:05": Market.HK, "21:05": Market.US}
        return close_map.get(time_part)

    def _generate_session_summary(self, market: Market, timestamp: str, snapshot: PortfolioSnapshot):
        """Generate session summary for a closing market. Returns the summary."""
        from src.core.types import SessionSummary
        summary = SessionSummary(
            market=market.value,
            session_date=timestamp[:10],
            market_read=f"{market.value} session closed at {timestamp}",
            model_actions=[],
            open_positions=[
                {"symbol": pos.symbol, "plan": ""}
                for key, pos in snapshot.positions.items()
                if pos.market == market
            ],
            risk_notes=[],
            created_at=timestamp,
        )
        self.memory.save_session_summary(summary)
        print(f"  Session summary: {market.value} closed at {timestamp}")
        return summary

    def _build_stock_data(self, ts: str, all_bars: dict) -> str:
        """Build compact card format stock data."""
        lines = ["[STOCK_DATA]"]
        for market in [Market.US, Market.HK, Market.CN, Market.CRYPTO]:
            symbols = self.universe.get_symbols(market)
            for sym in symbols[:5]:  # top 5 per market for now
                bars = all_bars.get(market, {}).get(sym, [])
                if not bars:
                    continue
                snap = self.features.compute(bars, ts)
                if snap is None:
                    continue
                card = (
                    f"{sym}|{market.value}|{snap.price:.2f}|"
                    f"{snap.chg_5m:+.2f}|{snap.chg_1h:+.2f}|{snap.chg_1d:+.2f}|"
                    f"{snap.rel_volume:.1f}x|{snap.rsi:.0f}|{snap.atr_pct:.2f}|"
                    f"{snap.trend}|{snap.bb_position:.2f}|{snap.high_low_pos:.2f}"
                )
                lines.append(card)
        return "\n".join(lines)

    def _build_alerts(self, ts: str, all_bars: dict) -> str:
        """Generate alerts for positions only — actionable RSI extremes.

        Only checks currently held positions, not the full universe.
        RSI extremes on stocks we don't own are noise, not signals.
        """
        alerts = []
        for key, pos in self.portfolio._positions.items():
            if pos.quantity <= 0:
                continue
            bars = all_bars.get(pos.market, {}).get(pos.symbol, [])
            if not bars:
                continue
            snap = self.features.compute(bars, ts)
            if snap is None:
                continue
            if snap.rsi > 80:
                alerts.append(f"ALERT: {pos.symbol} RSI={snap.rsi:.0f} OVERBOUGHT — consider taking profit")
            elif snap.rsi < 20:
                alerts.append(f"ALERT: {pos.symbol} RSI={snap.rsi:.0f} OVERSOLD — consider adding or holding")
        return "\n".join(alerts[:5]) if alerts else "(no alerts)"

    def _build_market_overview(self, ts: str, all_bars: dict) -> str:
        lines = []
        for market in [Market.US, Market.HK, Market.CN, Market.CRYPTO]:
            n = len(all_bars.get(market, {}))
            lines.append(f"{market.value}: {n} stocks")
        return "\n".join(lines)

    def _update_plan_peaks(self, plans: dict, snapshot) -> None:
        """Update plan peak prices for trailing stop tracking."""
        for symbol, plan in plans.items():
            for key, p in snapshot.positions.items():
                if p.symbol == symbol:
                    self.memory.update_plan_peak(symbol, p.current_price)
                    break

    def _update_prices(self, ts: str, all_bars: dict) -> None:
        """Update portfolio position prices."""
        prices = {}
        for key, pos in self.portfolio._positions.items():
            bars = all_bars.get(pos.market, {}).get(pos.symbol, [])
            if bars:
                # Find latest bar at or before ts
                for bar in reversed(bars):
                    if bar.timestamp <= ts:
                        prices[key] = self.nav_engine.convert_to_usd(bar.close, self._market_currency(pos.market))
                        break
        self.portfolio.update_prices(prices)

    def _get_price(self, symbol: str, market: Market, ts: str, all_bars: dict) -> float | None:
        bars = all_bars.get(market, {}).get(symbol, [])
        for bar in reversed(bars):
            if bar.timestamp <= ts:
                return bar.close
        return None

    def _generate_timestamps(self) -> list[str]:
        """Generate 5-minute timestamps for the backtest period."""
        from datetime import datetime, timedelta
        start = datetime.strptime(self.config.backtest_start, "%Y-%m-%d")
        end = datetime.strptime(self.config.backtest_end, "%Y-%m-%d")

        timestamps = []
        current = start
        while current <= end:
            for hour in range(24):
                for minute in range(0, 60, self.config.decision_interval):
                    ts = current.replace(hour=hour, minute=minute)
                    timestamps.append(ts.strftime("%Y-%m-%d %H:%M"))
            current += timedelta(days=1)

        return timestamps

    @staticmethod
    def _market_currency(market: Market) -> str:
        return {Market.US: "USD", Market.HK: "HKD", Market.CN: "CNY", Market.CRYPTO: "USD"}.get(market, "USD")
