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
import hashlib
import json
import os
import sqlite3
from datetime import datetime

from src.core.types import (
    AgentRound, BenchmarkResult, Decision, FuturesMarkResult, Market,
    PortfolioSnapshot, TradeResult,
)
from src.core.interfaces import IExperimentLogger
from src.core.pricing import compute_total_api_cost


class ExperimentLogger(IExperimentLogger):
    """SQLite-based experiment logger."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._run_id: str | None = None
        self._in_event = False
        # Accumulators for cost/efficiency computation
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_trading_fees: float = 0.0

    def init_run(
        self,
        config_dict: dict,
        run_id: str | None = None,
        dataset_version: str = "",
        prompt_version: str = "",
        tool_version: str = "",
        code_version: str = "",
        config_hash: str = "",
        benchmark_id: str = "",
        model: str = "",
        start_date: str = "",
        end_date: str = "",
        interval_min: int = 60,
        initial_cash: float = 100000,
        thinking_enabled: bool = False,
        total_decisions: int = 0,
        run_mode: str = "fresh",
        parent_run_id: str = "",
        parent_checkpoint_id: str = "",
    ) -> str:
        """Initialize a new experiment run. Returns run_id."""
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        # Some Windows/sandboxed filesystems reject DELETE journal cleanup.
        self._conn.execute("PRAGMA journal_mode=TRUNCATE")
        self._create_tables()

        self._run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        now = datetime.now().isoformat()
        self._conn.execute(
            """INSERT INTO benchmark_runs (
                run_id, model, start_date, end_date, interval_min, initial_cash,
                thinking_enabled, config, dataset_version, status, created_at,
                decisions_made, total_decisions, current_nav,
                total_trades, successful_trades, started_at, prompt_version,
                tool_version, code_version, config_hash, benchmark_id,
                run_mode, parent_run_id, parent_checkpoint_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self._run_id, model, start_date, end_date, interval_min, initial_cash,
                thinking_enabled, json.dumps(config_dict), dataset_version, "running", now,
                0, total_decisions, initial_cash,
                0, 0, now, prompt_version, tool_version, code_version, config_hash,
                benchmark_id, run_mode, parent_run_id, parent_checkpoint_id,
            ),
        )
        self._commit()
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
                code_version TEXT,
                config_hash TEXT,
                run_mode TEXT DEFAULT 'fresh',
                parent_run_id TEXT,
                parent_checkpoint_id TEXT,
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
                quantity REAL,
                price REAL,
                cost REAL,
                fees REAL,
                success INTEGER,
                error TEXT,
                metadata TEXT,
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
                futures_positions TEXT,
                futures_margin_locked REAL DEFAULT 0,
                futures_margin_state TEXT DEFAULT 'OK',
                futures_pnl_delta REAL DEFAULT 0,
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

            CREATE TABLE IF NOT EXISTS futures_marks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                timestamp TEXT,
                continuous_symbol TEXT,
                actual_contract TEXT,
                previous_mark_price REAL,
                current_price REAL,
                pnl_delta REAL,
                cumulative_variation_pnl REAL,
                cash_usd_after REAL,
                margin_locked REAL,
                margin_state TEXT,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS futures_roll_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                timestamp TEXT,
                continuous_symbol TEXT,
                old_contract TEXT,
                new_contract TEXT,
                old_contracts REAL,
                new_contracts REAL,
                old_close_price REAL,
                new_open_price REAL,
                roll_gap REAL,
                roll_cost REAL,
                selection_method TEXT,
                status TEXT,
                reject_reason TEXT,
                event_json TEXT,
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
            CREATE TABLE IF NOT EXISTS run_checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                event_seq INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                next_timestamp TEXT NOT NULL,
                next_timestamp_index INTEGER NOT NULL,
                state_schema_version INTEGER NOT NULL,
                state_blob BLOB NOT NULL,
                state_hash TEXT NOT NULL,
                previous_checkpoint_hash TEXT,
                config_hash TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                code_version TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(run_id, event_seq),
                FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
            );

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
        self._ensure_column("trades", "metadata", "TEXT")
        self._ensure_column("portfolio_snapshots", "futures_positions", "TEXT")
        self._ensure_column("portfolio_snapshots", "futures_margin_locked", "REAL DEFAULT 0")
        self._ensure_column("portfolio_snapshots", "futures_margin_state", "TEXT DEFAULT 'OK'")
        self._ensure_column("portfolio_snapshots", "futures_pnl_delta", "REAL DEFAULT 0")
        self._ensure_column("benchmark_runs", "code_version", "TEXT")
        self._ensure_column("benchmark_runs", "config_hash", "TEXT")
        self._ensure_column("benchmark_runs", "run_mode", "TEXT DEFAULT 'fresh'")
        self._ensure_column("benchmark_runs", "parent_run_id", "TEXT")
        self._ensure_column("benchmark_runs", "parent_checkpoint_id", "TEXT")
        self._ensure_column("run_checkpoints", "event_id", "TEXT")
        self._ensure_column("run_checkpoints", "next_timestamp", "TEXT")
        self._ensure_column("run_checkpoints", "previous_checkpoint_hash", "TEXT")
        self._ensure_column("run_checkpoints", "dataset_version", "TEXT")
        self._ensure_column("run_checkpoints", "status", "TEXT")
        # Enhancement columns for cost/efficiency analytics
        self._ensure_column("benchmark_runs", "api_cost_total", "REAL DEFAULT 0")
        self._ensure_column("benchmark_runs", "trading_fees_total", "REAL DEFAULT 0")
        self._ensure_column("benchmark_runs", "slippage_total", "REAL DEFAULT 0")
        self._ensure_column("benchmark_runs", "total_cost", "REAL DEFAULT 0")
        self._ensure_column("benchmark_runs", "total_prompt_tokens", "INTEGER DEFAULT 0")
        self._ensure_column("benchmark_runs", "total_completion_tokens", "INTEGER DEFAULT 0")
        self._ensure_column("benchmark_runs", "avg_latency_ms", "REAL DEFAULT 0")
        self._ensure_column("benchmark_runs", "tokens_per_decision", "REAL DEFAULT 0")
        self._ensure_column("benchmark_runs", "cost_per_decision", "REAL DEFAULT 0")
        self._ensure_column("benchmark_runs", "return_per_dollar_cost", "REAL DEFAULT 0")
        # Trade P&L tracking columns
        self._ensure_column("trades", "rejection_code", "TEXT DEFAULT ''")
        self._ensure_column("trades", "buy_timestamp", "TEXT DEFAULT ''")
        self._ensure_column("trades", "holding_minutes", "INTEGER DEFAULT 0")
        self._ensure_column("trades", "realized_pnl", "REAL DEFAULT 0")
        self._ensure_column("trades", "realized_pnl_pct", "REAL DEFAULT 0")

    def _commit(self) -> None:
        if not self._in_event:
            self._conn.commit()

    def begin_event(self) -> None:
        if self._in_event:
            raise RuntimeError("An event transaction is already active")
        self._conn.execute("BEGIN")
        self._in_event = True

    def rollback_event(self) -> None:
        if self._in_event:
            self._conn.rollback()
            self._in_event = False

    def commit_checkpoint(
        self,
        *,
        event_seq: int,
        timestamp: str,
        event_type: str,
        next_timestamp: str,
        next_timestamp_index: int,
        state_schema_version: int,
        state_blob: bytes,
        state_hash: str,
        config_hash: str,
        dataset_version: str,
        code_version: str,
    ) -> str:
        if not self._in_event:
            raise RuntimeError("Checkpoint must commit an active event transaction")
        checkpoint_id = f"cp_{self._run_id}_{event_seq:08d}"
        event_id = f"evt_{self._run_id}_{event_seq:08d}"
        previous = self._conn.execute(
            """SELECT state_hash FROM run_checkpoints
               WHERE run_id = ? AND status = 'COMMITTED'
               ORDER BY event_seq DESC LIMIT 1""",
            (self._run_id,),
        ).fetchone()
        previous_hash = previous[0] if previous else None
        self._conn.execute(
            """INSERT INTO run_checkpoints (
                checkpoint_id, run_id, event_id, event_seq, timestamp, event_type,
                next_timestamp, next_timestamp_index, state_schema_version, state_blob,
                state_hash, previous_checkpoint_hash, config_hash,
                dataset_version, code_version, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                checkpoint_id, self._run_id, event_id, event_seq, timestamp,
                event_type, next_timestamp, next_timestamp_index,
                state_schema_version, state_blob, state_hash, previous_hash, config_hash,
                dataset_version, code_version, "COMMITTED",
                datetime.now().isoformat(),
            ),
        )
        self._conn.commit()
        self._in_event = False
        return checkpoint_id

    def verify_checkpoint_chain(self) -> None:
        rows = self._conn.execute(
            """SELECT state_blob, state_hash, previous_checkpoint_hash
               FROM run_checkpoints
               WHERE run_id = ? AND status = 'COMMITTED'
               ORDER BY event_seq""",
            (self._run_id,),
        ).fetchall()
        previous_hash = None
        for blob, state_hash, linked_hash in rows:
            if hashlib.sha256(blob).hexdigest() != state_hash:
                raise ValueError("Checkpoint state hash mismatch")
            if linked_hash != previous_hash:
                raise ValueError("Checkpoint hash chain mismatch")
            previous_hash = state_hash
    def get_current_run(self) -> dict:
        row = self._conn.execute(
            "SELECT * FROM benchmark_runs WHERE run_id = ?",
            (self._run_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Run not found: {self._run_id}")
        columns = [item[0] for item in self._conn.execute(
            "SELECT * FROM benchmark_runs LIMIT 0"
        ).description]
        return dict(zip(columns, row))

    def set_initial_state_nav(self, nav: float) -> None:
        self._conn.execute(
            """UPDATE benchmark_runs SET initial_cash = ?, current_nav = ?
               WHERE run_id = ?""",
            (nav, nav, self._run_id),
        )
        self._conn.commit()

    def mark_running(self) -> None:
        self._conn.execute(
            """UPDATE benchmark_runs SET status = 'running', error_message = NULL
               WHERE run_id = ?""",
            (self._run_id,),
        )
        self._conn.commit()

    def prepare_extend(
        self,
        *,
        config_dict: dict,
        config_hash: str,
        end_date: str,
    ) -> None:
        self._conn.execute(
            """UPDATE benchmark_runs
               SET config = ?, config_hash = ?, end_date = ?,
                   status = 'running', completed_at = NULL,
                   finished_at = NULL, result = NULL
               WHERE run_id = ?""",
            (
                json.dumps(config_dict, ensure_ascii=False),
                config_hash,
                end_date,
                self._run_id,
            ),
        )
        self._conn.commit()
    def load_latest_checkpoint(self) -> dict | None:
        row = self._conn.execute(
            """SELECT * FROM run_checkpoints
               WHERE run_id = ? AND status = 'COMMITTED'
               ORDER BY event_seq DESC LIMIT 1""",
            (self._run_id,),
        ).fetchone()
        if row is None:
            return None
        columns = [item[0] for item in self._conn.execute(
            "SELECT * FROM run_checkpoints LIMIT 0"
        ).description]
        return dict(zip(columns, row))

    def attach_run(self, run_id: str) -> None:
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=TRUNCATE")
        self._create_tables()
        exists = self._conn.execute(
            "SELECT 1 FROM benchmark_runs WHERE run_id = ?", (run_id,),
        ).fetchone()
        if exists is None:
            raise ValueError(f"Run not found: {run_id}")
        self._run_id = run_id

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        cols = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in cols:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def log_decision(self, timestamp: str, decision: Decision, snapshot: PortfolioSnapshot, decision_type: str = "full_decision") -> None:
        trades_json = json.dumps([
            {"symbol": t.symbol, "market": t.market.value, "side": t.side.value,
             "quantity": t.quantity, "reason": t.reason}
            for t in decision.trades
        ])
        self._conn.execute(
            "INSERT INTO decisions (run_id, timestamp, decision_type, action, trades, reason, portfolio_nav) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (self._run_id, timestamp, decision_type, decision.action, trades_json,
             decision.reason, snapshot.total_nav),
        )
        self._commit()

    def log_trade(self, result: TradeResult, timestamp: str = "",
                  buy_timestamp: str = "", holding_minutes: int = 0,
                  realized_pnl: float = 0.0, realized_pnl_pct: float = 0.0,
                  rejection_code: str = "") -> None:
        if result.success:
            self._total_trading_fees += result.fees
        self._conn.execute(
            "INSERT INTO trades (run_id, timestamp, symbol, market, side, quantity, "
            "price, cost, fees, success, error, metadata, "
            "rejection_code, buy_timestamp, holding_minutes, realized_pnl, realized_pnl_pct) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (self._run_id, timestamp,
             result.order.symbol, result.order.market.value,
             result.order.side.value, result.order.quantity,
             result.price, result.cost, result.fees,
             1 if result.success else 0, result.error,
             json.dumps(result.metadata, ensure_ascii=False),
             rejection_code, buy_timestamp, holding_minutes, realized_pnl, realized_pnl_pct),
        )
        self._commit()

    def log_llm_call(
        self, timestamp: str, round_num: int, model: str,
        prompt_tokens: int, completion_tokens: int, latency_ms: float,
        reasoning: str = "", response: str = "",
    ) -> None:
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._conn.execute(
            "INSERT INTO llm_calls (run_id, decision_timestamp, round_num, model, prompt_tokens, completion_tokens, total_tokens, latency_ms, reasoning, response) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (self._run_id, timestamp, round_num, model,
             prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, latency_ms,
             reasoning, response),
        )
        self._commit()

    def log_round(self, round_data: AgentRound, timestamp: str = "") -> None:
        """Log an agent round (v3: includes timestamp)."""
        self._conn.execute(
            "INSERT INTO agent_rounds (run_id, decision_timestamp, round_num, action, llm_response, tool_results, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (self._run_id, timestamp, round_data.round_num,
             round_data.decision.action, round_data.llm_response,
             round_data.tool_results, round_data.latency_ms),
        )
        self._commit()

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
        self._commit()

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
        self._commit()

    def log_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        positions_json = json.dumps({
            k: {"qty": p.quantity, "avg": p.avg_cost, "px": p.current_price}
            for k, p in snapshot.positions.items()
        })
        exposure_json = json.dumps({
            m.value: v for m, v in snapshot.market_exposure.items()
        })
        futures_json = json.dumps({
            k: {
                "actual_contract": p.contract_ticker,
                "side": p.side,
                "contracts": p.contracts,
                "price": p.current_price,
                "margin_locked": p.margin_locked,
                "cum_pnl": p.cumulative_variation_pnl,
            }
            for k, p in snapshot.futures_positions.items()
        })
        self._conn.execute(
            "INSERT INTO portfolio_snapshots (run_id, timestamp, cash, nav, positions, market_exposure, futures_positions, futures_margin_locked, futures_margin_state, futures_pnl_delta) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (self._run_id, snapshot.timestamp, snapshot.cash,
             snapshot.total_nav, positions_json, exposure_json, futures_json,
             snapshot.futures_margin_locked, snapshot.futures_margin_state,
             snapshot.futures_pnl_delta),
        )
        self._commit()

    def log_futures_mark(self, mark: FuturesMarkResult, cash_usd_after: float) -> None:
        self._conn.execute(
            "INSERT INTO futures_marks (run_id, timestamp, continuous_symbol, actual_contract, previous_mark_price, current_price, pnl_delta, cumulative_variation_pnl, cash_usd_after, margin_locked, margin_state) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (self._run_id, mark.timestamp, mark.continuous_symbol, mark.contract_ticker,
             mark.previous_mark_price, mark.current_price, mark.pnl_delta,
             mark.cumulative_variation_pnl, cash_usd_after, mark.margin_locked,
             mark.margin_state),
        )
        self._commit()

    def log_futures_roll_event(self, event: dict) -> None:
        self._conn.execute(
            "INSERT INTO futures_roll_events (run_id, timestamp, continuous_symbol, old_contract, new_contract, old_contracts, new_contracts, old_close_price, new_open_price, roll_gap, roll_cost, selection_method, status, reject_reason, event_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (self._run_id, event.get("timestamp", ""), event.get("continuous_symbol", ""),
             event.get("old_contract", ""), event.get("new_contract", ""),
             event.get("old_contracts", 0), event.get("new_contracts", 0),
             event.get("old_close_price", 0.0), event.get("new_open_price", 0.0),
             event.get("roll_gap", 0.0), event.get("roll_cost", 0.0),
             event.get("selection_method", ""), event.get("status", ""),
             event.get("reject_reason", ""), json.dumps(event, ensure_ascii=False)),
        )
        self._commit()

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
        self._commit()

    def mark_completed(self) -> None:
        """Mark run as completed."""
        self._conn.execute(
            "UPDATE benchmark_runs SET status = 'completed', completed_at = ?, finished_at = ? WHERE run_id = ?",
            (datetime.now().isoformat(), datetime.now().isoformat(), self._run_id),
        )
        self._commit()

    def mark_failed(self, error_message: str) -> None:
        """Mark run as failed."""
        self._conn.execute(
            "UPDATE benchmark_runs SET status = 'failed', error_message = ?, completed_at = ? WHERE run_id = ?",
            (error_message, datetime.now().isoformat(), self._run_id),
        )
        self._commit()

    def save_results(self, result: BenchmarkResult) -> str:
        """Save final results with cost and efficiency metrics."""
        # Compute API cost from accumulated tokens
        api_cost = compute_total_api_cost(
            self._total_prompt_tokens, self._total_completion_tokens,
            result.model_name,
        )
        total_cost = api_cost + self._total_trading_fees

        # Compute efficiency metrics
        decisions_made = max(result.total_decisions, 1)
        total_tokens = self._total_prompt_tokens + self._total_completion_tokens
        tokens_per_decision = round(total_tokens / decisions_made, 2)
        cost_per_decision = round(total_cost / decisions_made, 6)
        return_usd = result.final_nav - result.initial_nav
        return_per_dollar = round(return_usd / total_cost, 4) if total_cost > 0 else 0.0
        avg_latency = self._compute_avg_latency()

        result_json = json.dumps({
            "model": result.model_name,
            "total_return": result.total_return,
            "sharpe": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "api_cost_total": round(api_cost, 4),
            "trading_fees_total": round(self._total_trading_fees, 4),
            "total_cost": round(total_cost, 4),
            "return_per_dollar_cost": return_per_dollar,
            "tokens_per_decision": tokens_per_decision,
            "cost_per_decision": cost_per_decision,
        })

        self._conn.execute(
            """UPDATE benchmark_runs SET
                finished_at = ?, result = ?, status = 'completed', completed_at = ?,
                api_cost_total = ?, trading_fees_total = ?, total_cost = ?,
                total_prompt_tokens = ?, total_completion_tokens = ?,
                avg_latency_ms = ?,
                tokens_per_decision = ?, cost_per_decision = ?,
                return_per_dollar_cost = ?
            WHERE run_id = ?""",
            (datetime.now().isoformat(), result_json, datetime.now().isoformat(),
             round(api_cost, 4), round(self._total_trading_fees, 4), round(total_cost, 4),
             self._total_prompt_tokens, self._total_completion_tokens,
             round(avg_latency, 2),
             tokens_per_decision, cost_per_decision,
             return_per_dollar,
             self._run_id),
        )
        self._commit()
        return self._db_path

    def _compute_avg_latency(self) -> float:
        """Compute average LLM call latency from llm_calls."""
        row = self._conn.execute(
            "SELECT AVG(latency_ms) FROM llm_calls WHERE run_id = ?",
            (self._run_id,),
        ).fetchone()
        return row[0] if row and row[0] else 0.0

    def load_resume_records(
        self,
    ) -> tuple[list[PortfolioSnapshot], list[AgentRound], list[dict]]:
        snapshot_rows = self._conn.execute(
            """SELECT timestamp, cash, nav, market_exposure,
                      futures_margin_locked, futures_margin_state,
                      futures_pnl_delta
               FROM portfolio_snapshots
               WHERE run_id = ? ORDER BY id""",
            (self._run_id,),
        ).fetchall()
        snapshots = []
        for row in snapshot_rows:
            exposure_raw = json.loads(row[3] or "{}")
            exposure = {Market(key): value for key, value in exposure_raw.items()}
            snapshots.append(PortfolioSnapshot(
                timestamp=row[0],
                cash=row[1],
                positions={},
                total_nav=row[2],
                market_exposure=exposure,
                fx_rates={},
                futures_margin_locked=row[4] or 0.0,
                futures_margin_state=row[5] or "OK",
                futures_pnl_delta=row[6] or 0.0,
            ))

        tool_rows = self._conn.execute(
            """SELECT decision_timestamp, tool_name FROM tool_calls
               WHERE run_id = ? ORDER BY created_at, tool_call_id""",
            (self._run_id,),
        ).fetchall()
        tools_by_timestamp: dict[str, list[dict[str, str]]] = {}
        for timestamp, tool_name in tool_rows:
            tools_by_timestamp.setdefault(timestamp, []).append({"tool": tool_name})
        round_rows = self._conn.execute(
            """SELECT decision_timestamp, round_num, action, llm_response,
                      tool_results, latency_ms
               FROM agent_rounds WHERE run_id = ? ORDER BY id""",
            (self._run_id,),
        ).fetchall()
        rounds = [
            AgentRound(
                round_num=row[1],
                decision=Decision(
                    action=row[2] or "hold",
                    queries=tools_by_timestamp.get(row[0], []),
                ),
                llm_response=row[3] or "",
                tool_results=row[4] or "",
                latency_ms=row[5] or 0.0,
            )
            for row in round_rows
        ]

        decision_rows = self._conn.execute(
            """SELECT timestamp, action, trades, decision_type
               FROM decisions WHERE run_id = ? ORDER BY id""",
            (self._run_id,),
        ).fetchall()
        decisions = [
            {
                "timestamp": row[0],
                "action": row[1],
                "symbol": row[2] or "hold",
                "market": row[3] or "",
            }
            for row in decision_rows
        ]
        return snapshots, rounds, decisions
    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @staticmethod
    def read_checkpoint(
        db_path: str, checkpoint_id: str | None = None,
    ) -> tuple[dict, dict]:
        if not os.path.exists(db_path):
            raise FileNotFoundError(db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        chain_rows = conn.execute(
            """SELECT state_blob, state_hash, previous_checkpoint_hash
               FROM run_checkpoints WHERE status = 'COMMITTED'
               ORDER BY event_seq"""
        ).fetchall()
        previous_hash = None
        for item in chain_rows:
            if hashlib.sha256(item["state_blob"]).hexdigest() != item["state_hash"]:
                conn.close()
                raise ValueError("Parent checkpoint state hash mismatch")
            if item["previous_checkpoint_hash"] != previous_hash:
                conn.close()
                raise ValueError("Parent checkpoint hash chain mismatch")
            previous_hash = item["state_hash"]
        if checkpoint_id:
            checkpoint = conn.execute(
                "SELECT * FROM run_checkpoints WHERE checkpoint_id = ? AND status = 'COMMITTED'",
                (checkpoint_id,),
            ).fetchone()
        else:
            checkpoint = conn.execute(
                "SELECT * FROM run_checkpoints WHERE status = 'COMMITTED' ORDER BY event_seq DESC LIMIT 1"
            ).fetchone()
        if checkpoint is None:
            conn.close()
            raise ValueError("Checkpoint not found")
        run = conn.execute(
            "SELECT * FROM benchmark_runs WHERE run_id = ?",
            (checkpoint["run_id"],),
        ).fetchone()
        conn.close()
        if run is None:
            raise ValueError("Parent run metadata not found")
        return dict(checkpoint), dict(run)
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
