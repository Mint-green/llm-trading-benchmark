from __future__ import annotations

import os
import sqlite3

import pytest

from src.core.config import Config
from src.core.types import Market
from src.platform.experiment import ExperimentRunner
from src.platform.logging import ExperimentLogger
from src.platform.run_identity import generate_run_id


def test_interrupted_run_resumes_without_duplicate_events(tmp_path, monkeypatch) -> None:
    config = Config(
        backtest_start="2026-02-03",
        backtest_end="2026-02-03",
        initial_cash=1_000_000,
        mimo_pro_api_key="unused-test-key",
    )
    if not all(os.path.exists(config.db_paths[market]) for market in (
        Market.US, Market.HK, Market.CN, Market.CRYPTO, Market.GOLD,
    )):
        pytest.skip("Local market databases are unavailable")

    timestamps = ["2026-02-03 00:05", "2026-02-03 00:10"]
    run_id = generate_run_id("mimo-v2.5-pro")
    db_path = tmp_path / f"{run_id}.db"
    first = ExperimentRunner(
        config, model="mimo-v2.5-pro", db_path=str(db_path), run_id=run_id,
    )
    monkeypatch.setattr(first, "_generate_timestamps", lambda: timestamps)
    original_commit = first.logger.commit_checkpoint
    commits = 0

    def interrupt_after_first_commit(**kwargs):
        nonlocal commits
        result = original_commit(**kwargs)
        commits += 1
        if commits == 1:
            raise RuntimeError("simulated process interruption")
        return result

    monkeypatch.setattr(first.logger, "commit_checkpoint", interrupt_after_first_commit)
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        first.run()
    first._close_resources()

    resumed = ExperimentRunner(
        config,
        model="mimo-v2.5-pro",
        db_path=str(db_path),
        run_id=run_id,
        resume=True,
    )
    monkeypatch.setattr(resumed, "_generate_timestamps", lambda: timestamps)
    result = resumed.run()

    assert result.final_nav == pytest.approx(1_000_000)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM run_checkpoints").fetchone()[0] == 2
        snapshot_timestamps = conn.execute(
            "SELECT timestamp FROM portfolio_snapshots ORDER BY id"
        ).fetchall()
        status = conn.execute("SELECT status FROM benchmark_runs").fetchone()[0]
    assert snapshot_timestamps == [(timestamps[0],), (timestamps[1],)]
    assert status == "completed"


def test_completed_run_can_extend_in_place(tmp_path, monkeypatch) -> None:
    base = Config(
        backtest_start="2026-02-03",
        backtest_end="2026-02-03",
        initial_cash=1_000_000,
        mimo_pro_api_key="unused-test-key",
    )
    if not os.path.exists(base.db_paths[Market.US]):
        pytest.skip("Local market databases are unavailable")
    run_id = generate_run_id("mimo-v2.5-pro")
    db_path = tmp_path / f"{run_id}.db"
    first_timestamp = "2026-02-03 00:05"

    first = ExperimentRunner(
        base, model="mimo-v2.5-pro", db_path=str(db_path), run_id=run_id,
    )
    monkeypatch.setattr(first, "_generate_timestamps", lambda: [first_timestamp])
    first.run()

    extended_config = Config(**{
        **base.__dict__,
        "backtest_end": "2026-02-04",
    })
    second_timestamp = "2026-02-03 00:10"
    extended = ExperimentRunner(
        extended_config,
        model="mimo-v2.5-pro",
        db_path=str(db_path),
        run_id=run_id,
        resume=True,
        extend=True,
    )
    monkeypatch.setattr(
        extended,
        "_generate_timestamps",
        lambda: [first_timestamp, second_timestamp],
    )
    extended.run()

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT run_id, end_date, status FROM benchmark_runs"
        ).fetchone()
        timestamps = conn.execute(
            "SELECT timestamp FROM portfolio_snapshots ORDER BY id"
        ).fetchall()
    assert row == (run_id, "2026-02-04", "completed")
    assert timestamps == [(first_timestamp,), (second_timestamp,)]


def test_fork_creates_independent_child_run(tmp_path, monkeypatch) -> None:
    config = Config(
        backtest_start="2026-02-03",
        backtest_end="2026-02-03",
        initial_cash=1_000_000,
        mimo_pro_api_key="unused-test-key",
    )
    if not os.path.exists(config.db_paths[Market.US]):
        pytest.skip("Local market databases are unavailable")

    first_timestamp = "2026-02-03 00:05"
    second_timestamp = "2026-02-03 00:10"
    parent_run_id = generate_run_id("mimo-v2.5-pro")
    parent_db = tmp_path / f"{parent_run_id}.db"
    parent = ExperimentRunner(
        config,
        model="mimo-v2.5-pro",
        db_path=str(parent_db),
        run_id=parent_run_id,
    )
    monkeypatch.setattr(parent, "_generate_timestamps", lambda: [first_timestamp])
    parent.run()

    checkpoint, parent_metadata = ExperimentLogger.read_checkpoint(str(parent_db))
    child_run_id = generate_run_id("mimo-v2.5-pro")
    child_db = tmp_path / f"{child_run_id}.db"
    child = ExperimentRunner(
        config,
        model="mimo-v2.5-pro",
        db_path=str(child_db),
        run_id=child_run_id,
        fork_checkpoint=checkpoint,
        parent_run_id=parent_run_id,
    )
    monkeypatch.setattr(
        child,
        "_generate_timestamps",
        lambda: [first_timestamp, second_timestamp],
    )
    child.run()

    with sqlite3.connect(parent_db) as conn:
        parent_snapshots = conn.execute(
            "SELECT timestamp FROM portfolio_snapshots ORDER BY id"
        ).fetchall()
    with sqlite3.connect(child_db) as conn:
        child_metadata = conn.execute(
            """SELECT run_mode, parent_run_id, parent_checkpoint_id, status
               FROM benchmark_runs"""
        ).fetchone()
        child_snapshots = conn.execute(
            "SELECT timestamp FROM portfolio_snapshots ORDER BY id"
        ).fetchall()

    assert parent_metadata["run_id"] == parent_run_id
    assert parent_snapshots == [(first_timestamp,)]
    assert child_metadata == (
        "fork", parent_run_id, checkpoint["checkpoint_id"], "completed",
    )
    assert child_snapshots == [(second_timestamp,)]
