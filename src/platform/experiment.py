"""
ExperimentRunner — orchestrates a complete benchmark run.

Wires all modules together and runs the backtest loop.
"""

from __future__ import annotations
import os
import time
from typing import Any

from src.core.config import Config
from src.core.types import (
    Market, OrderSide, PortfolioSnapshot, TradeOrder, Decision,
    AgentRound, BenchmarkResult,
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
from src.agent.context import ContextBuilder
from src.agent.tools import ToolSystem
from src.agent.runner import AgentRunner
from src.evaluation.metrics import MetricsEngine
from src.evaluation.behavior import BehaviorAnalyzer
from .logging import ExperimentLogger


class ExperimentRunner:
    """Orchestrates a complete benchmark experiment."""

    def __init__(self, config: Config, model: str = "mimo-v2.5-pro", db_path: str = "output/results/benchmark.db"):
        self.config = config
        self._model = model
        self._db_path = db_path
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

        fx_rates = dict(self.config.fx_rates)  # initial FX rates

        for i, ts in enumerate(timestamps):
            # Update FX rates at this timestamp
            new_rates = self.fx_provider.get_all_rates(ts)
            if new_rates:
                fx_rates = new_rates
                self.nav_engine.update_rates(fx_rates)

            # Reset per-decision state
            self.portfolio._constraints.reset_decision_state(ts)

            # Update portfolio prices
            self._update_prices(ts, all_bars)

            # Snapshot (with real FX rates)
            snapshot = self.portfolio.get_snapshot(ts)
            snapshot.fx_rates = fx_rates

            # Check if any stock market is open (skip crypto-only periods)
            if not self._any_stock_market_open(ts):
                continue

            # Skip end-of-session (20:00 UTC = last US hour) to avoid impulsive closing trades
            if self._is_end_of_session(ts):
                if self._any_stock_market_open(ts):
                    print(
                        f"[{ts}] decision={decision_count+1}/{len(timestamps)} | "
                        f"calls=0 | latency=0.0s | action=hold (end of session) | "
                        f"trades= | NAV=${self.portfolio.nav:,.0f}",
                        flush=True,
                    )
                continue

            # Build alerts (position RSI extremes only — actionable signals)
            alerts = self._build_alerts(ts, all_bars)

            # Get index returns
            index_returns = self.index_provider.get_all_index_returns(ts)

            # Build 3-layer context using screener
            held_tickers = {pos.symbol: pos.market for pos in snapshot.positions.values()}
            candidates = self.screener.screen(all_bars, ts, held_tickers=held_tickers)

            # Full universe list (restored per system-design.md Layer 1)
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
            candidate_str, _ = ContextBuilder.build_candidate_layer(candidates, detail_count=0)
            stock_context = market_summary + "\n\n" + candidate_str

            # Run agent with retry mechanism
            trade_feedback = ""
            decision = Decision(action="hold", reason="no decision")
            rounds = []

            for attempt in range(2):  # max 2 attempts (1 initial + 1 retry)
                decision, rounds = self.agent.run(
                    ts, snapshot, universe_str, stock_context, alerts, "",
                    trade_feedback=trade_feedback,
                    buy_quota_remaining=self.portfolio._constraints.daily_buys_remaining,
                )
                all_rounds.extend(rounds)

                # Log LLM call details
                for r in rounds:
                    prompt_t = getattr(r, '_prompt_tokens', 0)
                    compl_t = getattr(r, '_completion_tokens', 0)
                    reasoning = getattr(r, '_reasoning', '')
                    self.logger.log_llm_call(
                        ts, r.round_num, self.agent._api_model,
                        prompt_t, compl_t, r.latency_ms,
                        reasoning=reasoning, response=r.llm_response,
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
                                    # Lot rounding adjusted the quantity
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

                    # Build feedback for retry (include OK/ADJUSTED/FAILED)
                    trade_feedback = ContextBuilder.build_trade_feedback(feedback_results)
                    # Update snapshot for retry (portfolio state changed)
                    snapshot = self.portfolio.get_snapshot(ts)
                    snapshot.fx_rates = fx_rates
                else:
                    # Hold or other action, no retry needed
                    break

            # Log
            self.logger.log_decision(ts, decision, snapshot)
            self.logger.log_snapshot(self.portfolio.get_snapshot(ts))
            portfolio_history.append(self.portfolio.get_snapshot(ts))

            decision_count += 1

            # Calculate total LLM calls for this decision
            calls_this_decision = len(rounds)

            # Calculate total latency for this decision
            latency_ms = sum(r.latency_ms for r in rounds)

            # Build trades string
            trades_str = ""
            if decision.action == "trade" and decision.trades:
                trades_parts = []
                for t in decision.trades:
                    pct = f"({t.allocation_pct:.1%})" if t.allocation_pct else ""
                    trades_parts.append(f"{t.side.value.upper()} {t.symbol}{pct}")
                trades_str = ", ".join(trades_parts)

            # Structured progress output
            print(
                f"[{ts}] decision={decision_count}/{len(timestamps)} | "
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

        # Final snapshot
        final_snapshot = self.portfolio.get_snapshot(timestamps[-1] if timestamps else "")

        # Compute results
        if not portfolio_history:
            portfolio_history = [final_snapshot]
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
        print(f"Return:      {result.total_return:+.2f}%")
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
        """Check if this is the last decision point before market close.
        20:00 UTC = last US trading hour. Avoid impulsive end-of-day trades."""
        time_part = ts[11:16] if len(ts) >= 16 else ""
        return time_part == "20:00"

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
