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
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                run_id TEXT PRIMARY KEY,
                model TEXT,
                start_date TEXT,
                end_date TEXT,
                interval_min INTEGER,
                initial_cash REAL,
                thinking_enabled BOOLEAN,
                config TEXT,
                dataset_version TEXT,
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

    def log_round(self, round_data: AgentRound) -> None:
        self._conn.execute(
            "INSERT INTO agent_rounds (run_id, decision_timestamp, round_num, action, llm_response, tool_results, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (self._run_id, "", round_data.round_num,
             round_data.decision.action, round_data.llm_response,
             round_data.tool_results, round_data.latency_ms),
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
