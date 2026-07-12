from __future__ import annotations

import sqlite3

from src.platform.logging import ExperimentLogger
from src.platform.run_registry import sync_run_registry


def test_registry_syncs_flat_run_artifacts_idempotently(tmp_path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    artifact = runs / "run-1.db"
    logger = ExperimentLogger(str(artifact))
    logger.init_run(
        run_id="run-1",
        config_dict={"initial_cash": 1_000_000},
        config_hash="config-hash",
        model="mimo-v2.5-pro",
        start_date="2026-02-03",
        end_date="2026-02-09",
    )
    logger.mark_completed()
    logger.close()
    registry = tmp_path / "run_registry.db"

    assert sync_run_registry(runs, registry) == 1
    assert sync_run_registry(runs, registry) == 1

    with sqlite3.connect(registry) as conn:
        rows = conn.execute(
            "SELECT run_id, artifact_path, status FROM run_registry"
        ).fetchall()
    assert rows == [("run-1", "run-1.db", "completed")]
