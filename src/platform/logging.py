"""
ExperimentLogger — SQLite-based experiment logging.

Tables:
  - benchmark_runs: metadata for each experiment run
  - decisions: all agent decisions with timestamps
  - trades: all trade executions
  - portfolio_snapshots: periodic portfolio state
  - agent_rounds: LLM interaction details
"""

from __future__ import annotations
import json
import os
import sqlite3
from datetime import datetime

from src.core.types import Decision, PortfolioSnapshot, TradeResult, AgentRound, BenchmarkResult
from src.core.interfaces import IExperimentLogger


class ExperimentLogger(IExperimentLogger):
    """SQLite-based experiment logger."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._run_id: str | None = None

    def init_run(
        self,
        config_dict: dict,
        dataset_version: str = "",
        model: str = "",
        start_date: str = "",
        end_date: str = "",
        interval_min: int = 60,
        initial_cash: float = 100000,
        thinking_enabled: bool = False,
        total_decisions: int = 0,
    ) -> str:
        """Initialize a new experiment run. Returns run_id."""
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._create_tables()

        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        now = datetime.now().isoformat()
        self._conn.execute(
            """INSERT INTO benchmark_runs (
                run_id, model, start_date, end_date, interval_min, initial_cash,
                thinking_enabled, config, dataset_version, status, created_at,
                decisions_made, total_decisions, current_nav,
                total_trades, successful_trades, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self._run_id, model, start_date, end_date, interval_min, initial_cash,
                thinking_enabled, json.dumps(config_dict), dataset_version, "running", now,
                0, total_decisions, initial_cash,
                0, 0, now,
            ),
        )
        self._conn.commit()
        return self._run_id

    def _create_tables(self) -> None:
        self._conn.executescript("""
            -- Core tables (existing)
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                run_id TEXT PRIMARY KEY,
                model TEXT,
                model_id TEXT,
                model_version TEXT,
                benchmark_id TEXT,
                dataset_version TEXT,
                prompt_version TEXT,
                tool_version TEXT,
                start_date TEXT,
                end_date TEXT,
                interval_min INTEGER,
                initial_cash REAL,
                thinking_enabled BOOLEAN,
                config TEXT,
                status TEXT DEFAULT 'running',
                created_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                last_decision_ts TEXT,
                decisions_made INTEGER DEFAULT 0,
                total_decisions INTEGER,
                current_nav REAL,
                total_trades INTEGER DEFAULT 0,
                successful_trades INTEGER DEFAULT 0,
                started_at TEXT,
                finished_at TEXT,
                result TEXT
            );

            CREATE TABLE IF NOT EXISTS llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                decision_timestamp TEXT,
                round_num INTEGER,
                model TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                latency_ms REAL,
                reasoning TEXT,
                response TEXT,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                timestamp TEXT,
                decision_type TEXT DEFAULT 'full_decision',
                action TEXT,
                trades TEXT,
                reason TEXT,
                portfolio_nav REAL,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                timestamp TEXT,
                symbol TEXT,
                market TEXT,
                side TEXT,
                quantity INTEGER,
                price REAL,
                cost REAL,
                fees REAL,
                success INTEGER,
                error TEXT,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                timestamp TEXT,
                cash REAL,
                nav REAL,
                positions TEXT,
                market_exposure TEXT,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS agent_rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                decision_timestamp TEXT,
                round_num INTEGER,
                action TEXT,
                llm_response TEXT,
                tool_results TEXT,
                latency_ms REAL,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            );

            -- v3 tables: Decision events (enhanced decisions)
            CREATE TABLE IF NOT EXISTS decision_events (
                event_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                model_id TEXT,
                decision_timestamp TEXT NOT NULL,
                decision_type TEXT NOT NULL,
                prompt_hash TEXT,
                prompt_snapshot TEXT,
                tool_schema_hash TEXT,
                raw_model_output TEXT,
                parsed_output TEXT,
                validation_result TEXT,
                execution_result TEXT,
                state_diff TEXT,
                token_usage TEXT,
                latency_ms INTEGER,
                created_at TEXT NOT NULL
            );

            -- v3 tables: Tool calls
            CREATE TABLE IF NOT EXISTS tool_calls (
                tool_call_id TEXT PRIMARY KEY,
                event_id TEXT,
                run_id TEXT NOT NULL,
                decision_timestamp TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                tool_args TEXT NOT NULL,
                tool_result TEXT NOT NULL,
                result_hash TEXT,
                latency_ms INTEGER,
                created_at TEXT NOT NULL
            );

            -- v3 tables: Active plans
            CREATE TABLE IF NOT EXISTS active_plans (
                plan_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                model_id TEXT,
                symbol TEXT NOT NULL,
                position_id TEXT,
                status TEXT NOT NULL,
                side TEXT,
                entry_time TEXT,
                entry_price REAL,
                current_pct_nav REAL,
                entry_reason TEXT,
                plan_version INTEGER NOT NULL,
                last_review_time TEXT,
                last_review_price REAL,
                atr_at_review REAL,
                peak_since_entry REAL,
                peak_since_last_review REAL,
                intended_horizon_bars INTEGER,
                plan_note TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            -- v3 tables: Plan versions
            CREATE TABLE IF NOT EXISTS plan_versions (
                plan_version_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                source_event_id TEXT,
                plan_snapshot TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            -- v3 tables: Plan triggers
            CREATE TABLE IF NOT EXISTS plan_triggers (
                trigger_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                status TEXT NOT NULL,
                trigger_type TEXT NOT NULL,
                trigger_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                triggered_at TEXT,
                archived_at TEXT
            );

            -- v3 tables: Watchlist items
            CREATE TABLE IF NOT EXISTS watchlist_items (
                watch_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                model_id TEXT,
                symbol TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT,
                desired_condition_json TEXT,
                source_event_id TEXT,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                archived_at TEXT
            );

            -- v3 tables: Avoid items
            CREATE TABLE IF NOT EXISTS avoid_items (
                avoid_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                model_id TEXT,
                symbol TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT,
                source_event_id TEXT,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                archived_at TEXT
            );

            -- v3 tables: Daily thesis versions
            CREATE TABLE IF NOT EXISTS daily_thesis_versions (
                thesis_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                model_id TEXT,
                benchmark_day TEXT NOT NULL,
                version INTEGER NOT NULL,
                text TEXT NOT NULL,
                confidence REAL,
                source_event_id TEXT,
                created_at TEXT NOT NULL,
                expires_at TEXT
            );

            -- v3 tables: Summaries
            CREATE TABLE IF NOT EXISTS summaries (
                summary_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                model_id TEXT,
                summary_type TEXT NOT NULL,
                market TEXT,
                benchmark_day TEXT,
                source_start TEXT NOT NULL,
                source_end TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                summarizer_model TEXT,
                prompt_hash TEXT,
                created_at TEXT NOT NULL
            );

            -- v3 tables: Metrics daily
            CREATE TABLE IF NOT EXISTS metrics_daily (
                metrics_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                benchmark_day TEXT NOT NULL,
                nav_start_usd REAL,
                nav_end_usd REAL,
                daily_return_pct REAL,
                max_drawdown_pct REAL,
                turnover_pct REAL,
                fees_usd REAL,
                slippage_usd REAL,
                rejected_orders INTEGER,
                adjusted_orders INTEGER,
                constraint_hits INTEGER,
                tool_calls INTEGER,
                pnl_by_market TEXT,
                pnl_by_asset_type TEXT,
                pnl_by_symbol TEXT,
                attribution_json TEXT,
                created_at TEXT NOT NULL
            );
        """)

    def log_decision(self, timestamp: str, decision: Decision, snapshot: PortfolioSnapshot) -> None:
        trades_json = json.dumps([
            {"symbol": t.symbol, "market": t.market.value, "side": t.side.value,
             "quantity": t.quantity, "reason": t.reason}
            for t in decision.trades
        ])
        self._conn.execute(
            "INSERT INTO decisions (run_id, timestamp, action, trades, reason, portfolio_nav) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (self._run_id, timestamp, decision.action, trades_json,
             decision.reason, snapshot.total_nav),
        )
        self._conn.commit()

    def log_trade(self, result: TradeResult, timestamp: str = "") -> None:
        self._conn.execute(
            "INSERT INTO trades (run_id, timestamp, symbol, market, side, quantity, price, cost, fees, success, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (self._run_id, timestamp,
             result.order.symbol, result.order.market.value,
             result.order.side.value, result.order.quantity,
             result.price, result.cost, result.fees,
             1 if result.success else 0, result.error),
        )
        self._conn.commit()

    def log_llm_call(
        self, timestamp: str, round_num: int, model: str,
        prompt_tokens: int, completion_tokens: int, latency_ms: float,
        reasoning: str = "", response: str = "",
    ) -> None:
        self._conn.execute(
            "INSERT INTO llm_calls (run_id, decision_timestamp, round_num, model, prompt_tokens, completion_tokens, total_tokens, latency_ms, reasoning, response) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (self._run_id, timestamp, round_num, model,
             prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, latency_ms,
             reasoning, response),
        )
        self._conn.commit()

    def log_round(self, round_data: AgentRound, timestamp: str = "") -> None:
        """Log an agent round (v3: includes timestamp)."""
        self._conn.execute(
            "INSERT INTO agent_rounds (run_id, decision_timestamp, round_num, action, llm_response, tool_results, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (self._run_id, timestamp, round_data.round_num,
             round_data.decision.action, round_data.llm_response,
             round_data.tool_results, round_data.latency_ms),
        )
        self._conn.commit()

    def log_tool_call(
        self, timestamp: str, tool_name: str, tool_args: dict,
        tool_result: str, latency_ms: float = 0,
    ) -> None:
        """Log a tool call (v3)."""
        tool_call_id = f"tc_{self._run_id}_{timestamp}_{tool_name}_{id(tool_args)}"
        args_json = json.dumps(tool_args)
        result_hash = str(hash(tool_result))
        self._conn.execute(
            "INSERT INTO tool_calls (tool_call_id, run_id, decision_timestamp, tool_name, tool_args, tool_result, result_hash, latency_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (tool_call_id, self._run_id, timestamp, tool_name, args_json,
             tool_result, result_hash, latency_ms, datetime.now().isoformat()),
        )
        self._conn.commit()

    def log_decision_event(
        self, timestamp: str, decision_type: str,
        prompt_snapshot: str = "", raw_output: str = "",
        parsed_output: str = "", execution_result: str = "",
        token_usage: str = "", latency_ms: int = 0,
    ) -> None:
        """Log a decision event (v3)."""
        event_id = f"evt_{self._run_id}_{timestamp}_{id(raw_output)}"
        self._conn.execute(
            "INSERT INTO decision_events (event_id, run_id, decision_timestamp, decision_type, "
            "prompt_snapshot, raw_model_output, parsed_output, execution_result, "
            "token_usage, latency_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (event_id, self._run_id, timestamp, decision_type,
             prompt_snapshot, raw_output, parsed_output, execution_result,
             token_usage, latency_ms, datetime.now().isoformat()),
        )
        self._conn.commit()

    def log_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        positions_json = json.dumps({
            k: {"qty": p.quantity, "avg": p.avg_cost, "px": p.current_price}
            for k, p in snapshot.positions.items()
        })
        exposure_json = json.dumps({
            m.value: v for m, v in snapshot.market_exposure.items()
        })
        self._conn.execute(
            "INSERT INTO portfolio_snapshots (run_id, timestamp, cash, nav, positions, market_exposure) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (self._run_id, snapshot.timestamp, snapshot.cash,
             snapshot.total_nav, positions_json, exposure_json),
        )
        self._conn.commit()

    def update_progress(
        self,
        last_decision_ts: str,
        decisions_made: int,
        current_nav: float,
        total_trades: int = 0,
        successful_trades: int = 0,
    ) -> None:
        """Update run progress (called after each decision)."""
        self._conn.execute(
            """UPDATE benchmark_runs SET
                last_decision_ts = ?,
                decisions_made = ?,
                current_nav = ?,
                total_trades = ?,
                successful_trades = ?
            WHERE run_id = ?""",
            (last_decision_ts, decisions_made, current_nav, total_trades, successful_trades, self._run_id),
        )
        self._conn.commit()

    def mark_completed(self) -> None:
        """Mark run as completed."""
        self._conn.execute(
            "UPDATE benchmark_runs SET status = 'completed', completed_at = ?, finished_at = ? WHERE run_id = ?",
            (datetime.now().isoformat(), datetime.now().isoformat(), self._run_id),
        )
        self._conn.commit()

    def mark_failed(self, error_message: str) -> None:
        """Mark run as failed."""
        self._conn.execute(
            "UPDATE benchmark_runs SET status = 'failed', error_message = ?, completed_at = ? WHERE run_id = ?",
            (error_message, datetime.now().isoformat(), self._run_id),
        )
        self._conn.commit()

    def save_results(self, result: BenchmarkResult) -> str:
        result_json = json.dumps({
            "model": result.model_name,
            "total_return": result.total_return,
            "sharpe": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
        })
        self._conn.execute(
            "UPDATE benchmark_runs SET finished_at = ?, result = ?, status = 'completed', completed_at = ? WHERE run_id = ?",
            (datetime.now().isoformat(), result_json, datetime.now().isoformat(), self._run_id),
        )
        self._conn.commit()
        return self._db_path

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @staticmethod
    def get_latest_run(db_path: str) -> dict | None:
        """Get the latest run from database."""
        if not os.path.exists(db_path):
            return None
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM benchmark_runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    @staticmethod
    def get_running_run(db_path: str) -> dict | None:
        """Get the latest running (incomplete) run."""
        if not os.path.exists(db_path):
            return None
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM benchmark_runs WHERE status = 'running' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    @staticmethod
    def list_runs(db_path: str, limit: int = 10) -> list[dict]:
        """List recent runs."""
        if not os.path.exists(db_path):
            return []
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM benchmark_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
