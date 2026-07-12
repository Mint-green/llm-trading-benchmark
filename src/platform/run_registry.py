"""Serial synchronization of per-run metadata into a lightweight registry."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync_run_registry(runs_dir: str | Path, registry_path: str | Path) -> int:
    runs_root = Path(runs_dir).resolve()
    registry = Path(registry_path).resolve()
    registry.parent.mkdir(parents=True, exist_ok=True)
    target = sqlite3.connect(registry)
    target.execute("""
        CREATE TABLE IF NOT EXISTS run_registry (
            run_id TEXT PRIMARY KEY,
            artifact_path TEXT NOT NULL UNIQUE,
            artifact_checksum TEXT NOT NULL,
            model TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            status TEXT NOT NULL,
            config_json TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            dataset_version TEXT,
            prompt_version TEXT,
            tool_version TEXT,
            code_version TEXT,
            run_mode TEXT NOT NULL,
            parent_run_id TEXT,
            parent_checkpoint_id TEXT,
            created_at TEXT NOT NULL,
            finished_at TEXT
        )
    """)
    synced = 0
    try:
        for artifact in sorted(runs_root.glob("*.db")):
            if artifact.resolve() == registry:
                continue
            try:
                source = sqlite3.connect(f"file:{artifact.as_posix()}?mode=ro", uri=True)
                source.row_factory = sqlite3.Row
                row = source.execute(
                    "SELECT * FROM benchmark_runs ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
                source.close()
            except sqlite3.Error:
                continue
            if row is None:
                continue
            data = dict(row)
            target.execute(
                """INSERT INTO run_registry (
                    run_id, artifact_path, artifact_checksum, model,
                    start_date, end_date, status, config_json, config_hash,
                    dataset_version, prompt_version, tool_version, code_version,
                    run_mode, parent_run_id, parent_checkpoint_id,
                    created_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    artifact_path=excluded.artifact_path,
                    artifact_checksum=excluded.artifact_checksum,
                    model=excluded.model,
                    start_date=excluded.start_date,
                    status=excluded.status,
                    end_date=excluded.end_date,
                    config_json=excluded.config_json,
                    config_hash=excluded.config_hash,
                    dataset_version=excluded.dataset_version,
                    prompt_version=excluded.prompt_version,
                    tool_version=excluded.tool_version,
                    code_version=excluded.code_version,
                    run_mode=excluded.run_mode,
                    parent_run_id=excluded.parent_run_id,
                    parent_checkpoint_id=excluded.parent_checkpoint_id,
                    finished_at=excluded.finished_at
                """,
                (
                    data["run_id"], artifact.relative_to(runs_root).as_posix(),
                    _checksum(artifact), data.get("model") or "",
                    data.get("start_date") or "", data.get("end_date") or "",
                    data.get("status") or "", data.get("config") or "{}",
                    data.get("config_hash") or "", data.get("dataset_version"),
                    data.get("prompt_version"), data.get("tool_version"),
                    data.get("code_version"), data.get("run_mode") or "fresh",
                    data.get("parent_run_id"), data.get("parent_checkpoint_id"),
                    data.get("created_at") or "", data.get("finished_at"),
                ),
            )
            synced += 1
        target.commit()
    finally:
        target.close()
    return synced
