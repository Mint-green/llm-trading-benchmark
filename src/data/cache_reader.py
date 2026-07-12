"""Read-only shared derived cache for behavior-preserving backtest reuse."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import asdict
from pathlib import Path

from src.core.types import FuturesResolvedContract, IndicatorSnapshot

CACHE_SCHEMA_VERSION = 1


def futures_cache_namespace(futures_config: dict, code_version: str) -> str:
    payload = json.dumps(
        {
            "futures_config": futures_config,
            "code_version": code_version,
            "schema_version": CACHE_SCHEMA_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class DerivedCacheReader:
    def __init__(
        self,
        path: str | Path,
        dataset_version: str,
        namespace: str,
    ):
        self.path = Path(path)
        self.dataset_version = dataset_version
        self.namespace = namespace
        self._conn: sqlite3.Connection | None = None
        if self.path.exists():
            uri = f"file:{self.path.as_posix()}?mode=ro&immutable=1"
            try:
                conn = sqlite3.connect(uri, uri=True)
                row = conn.execute(
                    "SELECT value FROM cache_metadata "
                    "WHERE key = 'schema_version'"
                ).fetchone()
                if row is not None and int(row[0]) == CACHE_SCHEMA_VERSION:
                    self._conn = conn
                else:
                    conn.close()
            except (sqlite3.Error, ValueError):
                self._conn = None

    @property
    def available(self) -> bool:
        return self._conn is not None

    def get_futures_resolution(
        self, symbol: str, timestamp: str,
    ) -> FuturesResolvedContract | None:
        if self._conn is None:
            return None
        row = self._conn.execute(
            """SELECT payload FROM futures_resolutions
               WHERE dataset_version = ? AND namespace = ?
                 AND symbol = ? AND timestamp = ?""",
            (self.dataset_version, self.namespace, symbol, timestamp),
        ).fetchone()
        return FuturesResolvedContract(**json.loads(row[0])) if row else None

    def get_futures_feature(
        self, symbol: str, contract: str, timestamp: str,
    ) -> IndicatorSnapshot | None:
        if self._conn is None:
            return None
        row = self._conn.execute(
            """SELECT payload FROM futures_features
               WHERE dataset_version = ? AND namespace = ?
                 AND symbol = ? AND contract = ? AND timestamp = ?""",
            (
                self.dataset_version, self.namespace,
                symbol, contract, timestamp,
            ),
        ).fetchone()
        return IndicatorSnapshot(**json.loads(row[0])) if row else None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class DerivedCacheWriter:
    """Single-process offline writer; benchmark workers never use this class."""

    def __init__(
        self,
        path: str | Path,
        dataset_version: str,
        namespace: str,
    ):
        self.path = Path(path)
        self.dataset_version = dataset_version
        self.namespace = namespace
        os.makedirs(self.path.parent, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS cache_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS futures_resolutions (
                dataset_version TEXT NOT NULL,
                namespace TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (dataset_version, namespace, symbol, timestamp)
            );
            CREATE TABLE IF NOT EXISTS futures_features (
                dataset_version TEXT NOT NULL,
                namespace TEXT NOT NULL,
                symbol TEXT NOT NULL,
                contract TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (
                    dataset_version, namespace, symbol, contract, timestamp
                )
            );
        """)
        self._conn.execute(
            "INSERT OR REPLACE INTO cache_metadata VALUES (?, ?)",
            ("schema_version", str(CACHE_SCHEMA_VERSION)),
        )
        self._conn.commit()

    def put_futures_resolution(
        self, value: FuturesResolvedContract,
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO futures_resolutions
               VALUES (?, ?, ?, ?, ?)""",
            (
                self.dataset_version, self.namespace,
                value.continuous_symbol, value.timestamp,
                json.dumps(asdict(value), sort_keys=True),
            ),
        )

    def put_futures_feature(
        self,
        symbol: str,
        contract: str,
        timestamp: str,
        value: IndicatorSnapshot,
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO futures_features
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                self.dataset_version, self.namespace, symbol, contract,
                timestamp, json.dumps(asdict(value), sort_keys=True),
            ),
        )

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()
