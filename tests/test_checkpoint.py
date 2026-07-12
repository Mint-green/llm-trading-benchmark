from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest

from src.platform.checkpoint import (
    capture_runtime_state,
    decode_checkpoint,
    encode_checkpoint,
    restore_runtime_state,
)
from src.core.types import PortfolioSnapshot
from src.platform.logging import ExperimentLogger


def _component(**values):
    return SimpleNamespace(**values)


def _runner():
    runner = _component(
        portfolio=_component(
            _cash={"USD": 100.0}, _positions={}, _futures_positions={},
            _futures_margin_locked=0.0, _futures_margin_state="OK",
            _futures_pnl_delta=0.0, _reserved_usd=0.0,
            _trade_history=[], _fx_log=[],
        ),
        constraints=_component(
            _last_buy={}, _daily_trades={}, _sells_this_decision=0,
            _last_decision_ts="", _tail_guard_active=False,
            _tail_guard_markets=[],
        ),
        settlement=_component(_buy_history=defaultdict(list)),
        futures_account=_component(
            cash_usd=100.0, positions={}, trade_history=[], roll_history=[],
            margin_state="OK",
        ),
        scheduler=_component(
            _last_full_decision="", _last_market_decision={}, _last_focused={},
        ),
        event_detector=_component(_prev_regime={}, _prev_open_state={}),
        nav_engine=_component(_fx_rates={"USD": 1.0}),
        memory=_component(_plans={}, _recent_decisions=[]),
        _last_light_decision="",
        _risk_mode="GREEN",
        _stop_loss_buy_pause_until={},
        _stop_loss_recent_by_market={},
        _pending_daily_summary_injection=False,
        _logged_futures_roll_count=0,
        _logged_futures_trade_count=0,
    )
    return runner


def test_checkpoint_round_trip_restores_complete_mutable_state() -> None:
    runner = _runner()
    state = capture_runtime_state(runner, {"next_timestamp_index": 42})
    blob, digest = encode_checkpoint(state)

    runner.portfolio._cash["USD"] = 0.0
    runner.constraints._daily_trades["2026-02-05"] = 99
    runner.memory._plans["AAPL.US"] = "changed"
    loop = restore_runtime_state(runner, decode_checkpoint(blob, digest))

    assert runner.portfolio._cash == {"USD": 100.0}
    assert runner.constraints._daily_trades == {}
    assert runner.memory._plans == {}
    assert loop == {"next_timestamp_index": 42}


def test_checkpoint_rejects_tampering() -> None:
    blob, digest = encode_checkpoint({"schema_version": 1})
    tampered = blob[:-1] + bytes([blob[-1] ^ 1])
    with pytest.raises(ValueError, match="hash mismatch"):
        decode_checkpoint(tampered, digest)


def test_logger_commits_event_rows_and_checkpoint_atomically(tmp_path) -> None:
    db_path = tmp_path / "checkpoint.db"
    logger = ExperimentLogger(str(db_path))
    logger.init_run(run_id="run-1", config_dict={})
    snapshot = PortfolioSnapshot("2026-02-05 14:30", 100.0, {}, 100.0, {}, {})

    logger.begin_event()
    logger.log_snapshot(snapshot)
    logger.rollback_event()
    count = logger._conn.execute(
        "SELECT COUNT(*) FROM portfolio_snapshots"
    ).fetchone()[0]
    assert count == 0

    logger.begin_event()
    logger.log_snapshot(snapshot)
    blob, digest = encode_checkpoint({"schema_version": 1, "loop": {}})
    logger.commit_checkpoint(
        event_seq=1,
        timestamp=snapshot.timestamp,
        event_type="timestamp",
        next_timestamp="2026-02-05 14:35",
        next_timestamp_index=1,
        state_schema_version=1,
        state_blob=blob,
        state_hash=digest,
        config_hash="config",
        dataset_version="dataset",
        code_version="code",
    )

    count = logger._conn.execute(
        "SELECT COUNT(*) FROM portfolio_snapshots"
    ).fetchone()[0]
    assert count == 1
    assert logger.load_latest_checkpoint()["event_seq"] == 1

    logger.begin_event()
    second_blob, second_digest = encode_checkpoint({
        "schema_version": 1, "loop": {"next_timestamp_index": 2},
    })
    logger.commit_checkpoint(
        event_seq=2,
        timestamp="2026-02-05 14:35",
        event_type="timestamp",
        next_timestamp="",
        next_timestamp_index=2,
        state_schema_version=1,
        state_blob=second_blob,
        state_hash=second_digest,
        config_hash="config",
        dataset_version="dataset",
        code_version="code",
    )
    logger.verify_checkpoint_chain()
    logger._conn.execute(
        """UPDATE run_checkpoints SET previous_checkpoint_hash = 'tampered'
           WHERE event_seq = 2"""
    )
    logger._conn.commit()
    with pytest.raises(ValueError, match="hash chain mismatch"):
        logger.verify_checkpoint_chain()
    logger.close()
