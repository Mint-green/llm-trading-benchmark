from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone

from src.platform.logging import ExperimentLogger
from src.platform.run_identity import (
    build_version_metadata,
    generate_run_id,
    reproducible_config_dict,
    resolve_run_output,
    stable_json_hash,
)


def test_generate_run_id_is_readable_and_collision_resistant() -> None:
    run_id = generate_run_id(
        "MiMo/v2.5 Pro",
        now=datetime(2026, 7, 10, 23, 12, 0, 123456, tzinfo=timezone.utc),
        random_suffix="a7k9x2",
    )

    assert run_id == "20260710T231200123Z-mimo-v2_5-pro-a7k9x2"
    assert re.fullmatch(r"\d{8}T\d{9}Z-[a-z0-9_-]+-[a-z0-9]{6}", run_id)


def test_default_output_uses_run_id_but_explicit_output_is_preserved(tmp_path) -> None:
    run_id = "20260710T231200123Z-mimo-v2_5-pro-a7k9x2"

    assert resolve_run_output(None, run_id, tmp_path) == tmp_path / f"{run_id}.db"
    explicit = tmp_path / "named-regression.db"
    assert resolve_run_output(str(explicit), run_id, tmp_path) == explicit


def test_reproducible_config_includes_nested_settings_but_excludes_api_keys() -> None:
    from src.core.config import Config, FuturesConfig

    config = Config(
        mimo_pro_api_key="secret-mimo",
        deepseek_api_key="secret-ds",
        futures=FuturesConfig(commission_per_contract=3.25),
    )

    serialized = reproducible_config_dict(config)

    assert serialized["futures"]["commission_per_contract"] == 3.25
    assert serialized["commission_bps"]["US"] == 3.0
    assert "mimo_pro_api_key" not in serialized
    assert "deepseek_api_key" not in serialized
    assert "secret-mimo" not in str(serialized)
    assert "secret-ds" not in str(serialized)

def test_dynamic_nav_rates_do_not_mutate_frozen_config() -> None:
    from src.core.config import Config
    from src.portfolio.nav import NavEngine

    config = Config(fx_rates={"USD": 1.0, "HKD": 7.8})
    before = stable_json_hash(reproducible_config_dict(config))
    nav = NavEngine(config.fx_rates)

    nav.update_rates({"HKD": 7.9})

    assert config.fx_rates == {"USD": 1.0, "HKD": 7.8}
    assert stable_json_hash(reproducible_config_dict(config)) == before


def test_version_metadata_is_deterministic_and_changes_with_inputs(tmp_path) -> None:
    (tmp_path / "src" / "agent").mkdir(parents=True)
    (tmp_path / "prompts" / "active").mkdir(parents=True)
    (tmp_path / "runners").mkdir()
    (tmp_path / "src" / "agent" / "tools.py").write_text("TOOLS = 1\n", encoding="utf-8")
    (tmp_path / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "prompts" / "active" / "prompts.py").write_text("PROMPT = 'a'\n", encoding="utf-8")
    (tmp_path / "runners" / "run_backtest.py").write_text("RUNNER = 1\n", encoding="utf-8")

    config = {"initial_cash": 1_000_000, "markets": ["US", "FUTURES"]}
    first = build_version_metadata(tmp_path, config, "dataset-v1")
    second = build_version_metadata(tmp_path, config, "dataset-v1")
    assert first == second
    assert first.config_hash == stable_json_hash(config)
    assert len(first.prompt_version) == 64
    assert len(first.tool_version) == 64
    assert len(first.code_version) == 64
    assert first.benchmark_id.startswith("benchmark-")

    (tmp_path / "src" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    changed = build_version_metadata(tmp_path, config, "dataset-v1")
    assert changed.code_version != first.code_version
    assert changed.prompt_version == first.prompt_version


def test_logger_persists_run_id_and_all_version_fields(tmp_path) -> None:
    db_path = tmp_path / "result.db"
    run_id = "20260710T231200123Z-deepseek-v4-pro-b2c3d4"
    logger = ExperimentLogger(str(db_path))

    returned = logger.init_run(
        run_id=run_id,
        config_dict={"initial_cash": 1_000_000},
        dataset_version="dataset-v1",
        prompt_version="prompt-hash",
        tool_version="tool-hash",
        code_version="code-hash",
        config_hash="config-hash",
        benchmark_id="benchmark-id",
        model="deepseek-v4-pro",
    )
    logger.close()

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT run_id, dataset_version, prompt_version, tool_version, "
            "code_version, config_hash, benchmark_id FROM benchmark_runs"
        ).fetchone()

    assert returned == run_id
    assert row == (
        run_id,
        "dataset-v1",
        "prompt-hash",
        "tool-hash",
        "code-hash",
        "config-hash",
        "benchmark-id",
    )
