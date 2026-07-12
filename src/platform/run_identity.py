"""Run identifiers and reproducibility fingerprints."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import string
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


_SUFFIX_ALPHABET = string.ascii_lowercase + string.digits


@dataclass(frozen=True)
class VersionMetadata:
    dataset_version: str
    prompt_version: str
    tool_version: str
    code_version: str
    config_hash: str
    benchmark_id: str


def reproducible_config_dict(config: Any) -> dict[str, Any]:
    """Serialize all non-secret config fields into stable JSON-compatible data."""

    def normalize(value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: normalize(getattr(value, field.name))
                for field in fields(value)
                if not field.name.endswith("_api_key")
            }
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Mapping):
            return {
                str(normalize(key)): normalize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted(normalize(item) for item in value)
        if isinstance(value, Path):
            return value.as_posix()
        return value

    result = normalize(config)
    if not isinstance(result, dict):
        raise TypeError("config must serialize to a dictionary")
    return result


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hash(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def build_version_metadata(
    project_root: str | Path,
    config_dict: dict[str, Any],
    dataset_version: str,
) -> VersionMetadata:
    root = Path(project_root).resolve()
    prompt_path = root / "prompts" / "active" / "prompts.py"
    tool_path = root / "src" / "agent" / "tools.py"
    code_paths = list((root / "src").rglob("*.py"))
    runner_path = root / "runners" / "run_backtest.py"
    if runner_path.exists():
        code_paths.append(runner_path)

    config_hash = stable_json_hash(config_dict)
    benchmark_hash = stable_json_hash({
        "config_hash": config_hash,
        "dataset_version": dataset_version,
    })
    return VersionMetadata(
        dataset_version=dataset_version,
        prompt_version=_file_hash(prompt_path),
        tool_version=_file_hash(tool_path),
        code_version=_tree_hash(code_paths, root),
        config_hash=config_hash,
        benchmark_id=f"benchmark-{benchmark_hash[:16]}",
    )


def generate_run_id(
    model: str,
    *,
    now: datetime | None = None,
    random_suffix: str | None = None,
) -> str:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    timestamp = current.strftime("%Y%m%dT%H%M%S") + f"{current.microsecond // 1000:03d}Z"
    model_slug = re.sub(r"[^a-z0-9_-]+", "-", model.lower().replace(".", "_")).strip("-_") or "model"
    suffix = random_suffix or "".join(secrets.choice(_SUFFIX_ALPHABET) for _ in range(6))
    if not re.fullmatch(r"[a-z0-9]{6}", suffix):
        raise ValueError("random_suffix must contain exactly 6 lowercase letters or digits")
    return f"{timestamp}-{model_slug}-{suffix}"


def resolve_run_output(
    explicit_output: str | None,
    run_id: str,
    runs_dir: str | Path = "artifacts/runs",
) -> Path:
    if explicit_output:
        return Path(explicit_output)
    return Path(runs_dir) / f"{run_id}.db"
