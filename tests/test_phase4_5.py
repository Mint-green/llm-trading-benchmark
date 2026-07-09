"""
Phase 4+5 unit tests — verify SummaryEngine, Metrics enhancement, and Replay.
"""

import pytest
import json
import os
import tempfile
from src.core.types import (
    Market, OrderSide, PortfolioSnapshot, Position,
    TradeOrder, TradeResult, SessionSummary, DailySummary,
)
from src.evaluation.summary_engine import SummaryEngine
from src.evaluation.metrics import MetricsEngine
from src.platform.replay import ReplayEngine


# ============================================================
# Test: SummaryEngine
# ============================================================

class TestSummaryEngine:
    def setup_method(self):
        self.engine = SummaryEngine()

    def test_session_summary_generation(self):
        snapshot = PortfolioSnapshot(
            timestamp="2026-01-07 08:00",
            cash=50000.0,
            positions={},
            total_nav=100000.0,
            market_exposure={Market.HK: 20000},
            fx_rates={},
        )

        summary = self.engine.generate_session_summary(
            market=Market.HK,
            timestamp="2026-01-07 08:00",
            snapshot=snapshot,
            decisions=[],
            trades=[],
            plans=[],
        )

        assert summary.market == "HK"
        assert summary.session_date == "2026-01-07"
        assert "HK" in summary.market_read

    def test_daily_summary_generation(self):
        snapshot = PortfolioSnapshot(
            timestamp="2026-01-07 00:00",
            cash=50000.0,
            positions={},
            total_nav=100320.0,
            market_exposure={},
            fx_rates={},
        )

        summary = self.engine.generate_daily_summary(
            date="2026-01-07",
            nav_start=100000.0,
            nav_end=100320.0,
            all_decisions=[{"action": "trade", "symbol": "AAPL", "side": "buy"}],
            all_trades=[],
            session_summaries=[],
            snapshot=snapshot,
            plans=[],
        )

        assert summary.date == "2026-01-07"
        assert summary.daily_return_pct == pytest.approx(0.0032, abs=0.0001)
        assert len(summary.major_decisions) > 0

    def test_get_latest_session_summary(self):
        # Add summaries
        self.engine._session_summaries.append(SessionSummary(
            market="HK", session_date="2026-01-06", market_read="HK day 1",
        ))
        self.engine._session_summaries.append(SessionSummary(
            market="HK", session_date="2026-01-07", market_read="HK day 2",
        ))

        latest = self.engine.get_latest_session_summary("HK")
        assert latest.session_date == "2026-01-07"

    def test_get_previous_daily_summary(self):
        assert self.engine.get_previous_daily_summary() is None

        self.engine._daily_summaries.append(DailySummary(
            date="2026-01-06", nav_start=100000, nav_end=100100, daily_return_pct=0.001,
        ))
        self.engine._daily_summaries.append(DailySummary(
            date="2026-01-07", nav_start=100100, nav_end=100300, daily_return_pct=0.002,
        ))

        prev = self.engine.get_previous_daily_summary()
        assert prev.date == "2026-01-07"


# ============================================================
# Test: MetricsEngine Enhancement
# ============================================================

class TestMetricsEnhancement:
    def setup_method(self):
        self.engine = MetricsEngine()

    def test_behavior_metrics(self):
        decisions = [
            {"decision_type": "full_decision"},
            {"decision_type": "focused_position_decision"},
        ]
        trades = [
            TradeResult(
                order=TradeOrder(symbol="AAPL", market=Market.US, side=OrderSide.BUY, quantity=100),
                success=True, price=150.0, cost=15000, fees=5.0,
            ),
            TradeResult(
                order=TradeOrder(symbol="TSLA", market=Market.US, side=OrderSide.BUY, quantity=50),
                success=False, error="constraint: single position limit",
            ),
        ]
        tool_calls = [
            {"tool_name": "query_asset"},
            {"tool_name": "screen_universe"},
        ]

        behavior = self.engine.compute_behavior_metrics(decisions, trades, tool_calls)

        assert behavior["rejected_orders"] == 1
        assert behavior["constraint_hits"] == 1
        assert behavior["total_tool_calls"] == 2
        assert "query_asset" in behavior["tool_usage"]

    def test_pnl_attribution(self):
        history = [
            PortfolioSnapshot(
                timestamp="2026-01-07 00:00", cash=50000.0,
                positions={}, total_nav=100000.0,
                market_exposure={}, fx_rates={},
            ),
            PortfolioSnapshot(
                timestamp="2026-01-07 23:55", cash=40000.0,
                positions={
                    "US:AAPL": Position(
                        symbol="AAPL", market=Market.US, quantity=100,
                        avg_cost=150.0, current_price=155.0, unrealized_pnl=500.0,
                    ),
                },
                total_nav=100500.0,
                market_exposure={Market.US: 15500},
                fx_rates={},
            ),
        ]

        attribution = self.engine.compute_pnl_attribution(history, [])

        assert attribution["total_pnl_usd"] == 500.0
        assert "US" in attribution["by_market"]
        assert attribution["by_market"]["US"] == 500.0


# ============================================================
# Test: ReplayEngine
# ============================================================

class TestReplayEngine:
    def setup_method(self):
        # Create a temporary database
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.tmp.name
        self.tmp.close()

        # Create tables
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                run_id TEXT PRIMARY KEY, model TEXT, status TEXT,
                start_date TEXT, end_date TEXT, initial_cash REAL,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS decision_events (
                event_id TEXT PRIMARY KEY, run_id TEXT,
                decision_timestamp TEXT, decision_type TEXT,
                execution_result TEXT, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT,
                timestamp TEXT, action TEXT, trades TEXT, reason TEXT,
                portfolio_nav REAL
            );
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT,
                timestamp TEXT, symbol TEXT, market TEXT, side TEXT,
                quantity INTEGER, price REAL, cost REAL, fees REAL,
                success INTEGER, error TEXT
            );
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT,
                timestamp TEXT, cash REAL, nav REAL,
                positions TEXT, market_exposure TEXT
            );
            CREATE TABLE IF NOT EXISTS tool_calls (
                tool_call_id TEXT PRIMARY KEY, event_id TEXT,
                run_id TEXT, decision_timestamp TEXT,
                tool_name TEXT, tool_args TEXT, tool_result TEXT,
                result_hash TEXT, latency_ms INTEGER, created_at TEXT
            );
        """)
        conn.close()

    def teardown_method(self):
        os.unlink(self.db_path)

    def test_replay_empty_run(self):
        engine = ReplayEngine(self.db_path)
        result = engine.replay("nonexistent")
        assert result.success is True
        assert result.events_replayed == 0

    def test_verify_state(self):
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO benchmark_runs (run_id, model, status, start_date, end_date, initial_cash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("test_run", "test_model", "completed", "2026-01-07", "2026-01-08", 100000, "2026-01-07T00:00:00"),
        )
        conn.execute(
            "INSERT INTO decisions (run_id, timestamp, action, portfolio_nav) VALUES (?, ?, ?, ?)",
            ("test_run", "2026-01-07 10:00", "hold", 100000),
        )
        conn.execute(
            "INSERT INTO portfolio_snapshots (run_id, timestamp, cash, nav) VALUES (?, ?, ?, ?)",
            ("test_run", "2026-01-07 10:00", 50000, 100000),
        )
        conn.commit()
        conn.close()

        engine = ReplayEngine(self.db_path)
        result = engine.verify_state("test_run")

        assert result["consistent"] is True
        assert len(result["checks"]) > 0

    def test_get_run_summary(self):
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO benchmark_runs (run_id, model, status, start_date, end_date, initial_cash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("test_run", "test_model", "completed", "2026-01-07", "2026-01-08", 100000, "2026-01-07T00:00:00"),
        )
        conn.commit()
        conn.close()

        engine = ReplayEngine(self.db_path)
        summary = engine.get_run_summary("test_run")

        assert summary["run_id"] == "test_run"
        assert summary["model"] == "test_model"
        assert summary["status"] == "completed"


# ============================================================
# Test: Database Schema
# ============================================================

class TestDatabaseSchema:
    def test_schema_creation(self):
        """Test that all v3 tables are created."""
        import sqlite3
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = tmp.name
        tmp.close()

        try:
            conn = sqlite3.connect(db_path)

            # Create tables using the same SQL as logging.py
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS benchmark_runs (run_id TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS llm_calls (id INTEGER PRIMARY KEY AUTOINCREMENT);
                CREATE TABLE IF NOT EXISTS decisions (id INTEGER PRIMARY KEY AUTOINCREMENT);
                CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT);
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (id INTEGER PRIMARY KEY AUTOINCREMENT);
                CREATE TABLE IF NOT EXISTS agent_rounds (id INTEGER PRIMARY KEY AUTOINCREMENT);
                CREATE TABLE IF NOT EXISTS decision_events (event_id TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS tool_calls (tool_call_id TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS active_plans (plan_id TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS plan_versions (plan_version_id TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS plan_triggers (trigger_id TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS watchlist_items (watch_id TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS avoid_items (avoid_id TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS daily_thesis_versions (thesis_id TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS summaries (summary_id TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS metrics_daily (metrics_id TEXT PRIMARY KEY);
            """)

            # Check tables exist
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            table_names = {t[0] for t in tables}

            expected = {
                "benchmark_runs", "llm_calls", "decisions", "trades",
                "portfolio_snapshots", "agent_rounds", "decision_events",
                "tool_calls", "active_plans", "plan_versions", "plan_triggers",
                "watchlist_items", "avoid_items", "daily_thesis_versions",
                "summaries", "metrics_daily",
            }

            assert expected.issubset(table_names)

            conn.close()
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
