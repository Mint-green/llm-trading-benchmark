"""
Main entry point for running the benchmark.

Usage:
    python runners/run_backtest.py --model mimo-v2.5-pro --start 2026-02-03 --end 2026-02-09
    python runners/run_backtest.py --resume
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.config import Config
from src.platform.experiment import ExperimentRunner
from src.platform.logging import ExperimentLogger
from src.platform.run_identity import generate_run_id, resolve_run_output


def main():
    parser = argparse.ArgumentParser(description="Multi-Market LLM Trading Benchmark")
    parser.add_argument("--model", default=None, help="Model name (e.g., mimo-v2.5-pro, deepseek-v4-pro)")
    parser.add_argument("--start", default=None, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--interval", type=int, default=None, help="Timestamp/scanner interval in minutes")
    parser.add_argument("--initial-cash", type=float, default=None, help="Initial cash amount")
    parser.add_argument("--max-decisions", type=int, default=None, help="Max decisions (0=unlimited)")
    parser.add_argument("--max-rounds", type=int, default=None, help="Max LLM rounds per decision")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    parser.add_argument("--config", default="config/config.toml", help="Config file path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output database path (default: artifacts/runs/<run_id>.db)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from last interrupted run")
    parser.add_argument(
        "--extend-end",
        default=None,
        help="Extend an existing run to a later end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--fork-from-checkpoint",
        default=None,
        metavar="PARENT_DB",
        help="Fork a new run from a parent run database checkpoint",
    )
    parser.add_argument(
        "--checkpoint-id",
        default=None,
        help="Parent checkpoint id (default: latest committed checkpoint)",
    )
    args = parser.parse_args()
    if args.checkpoint_id and not args.fork_from_checkpoint:
        parser.error("--checkpoint-id requires --fork-from-checkpoint")

    # Handle resume/extend/fork mode
    resume_run = None
    fork_checkpoint = None
    parent_run = None
    selected_modes = sum(bool(value) for value in (
        args.resume, args.extend_end, args.fork_from_checkpoint,
    ))
    if selected_modes > 1:
        parser.error("--resume, --extend-end and --fork-from-checkpoint are mutually exclusive")
    if args.fork_from_checkpoint:
        fork_checkpoint, parent_run = ExperimentLogger.read_checkpoint(
            args.fork_from_checkpoint, args.checkpoint_id,
        )
        if args.start and args.start != parent_run["start_date"]:
            parser.error("Fork start date must match the parent run start date")
        args.start = parent_run["start_date"]
        print(
            f"Forking parent run {parent_run['run_id']} "
            f"from {fork_checkpoint['checkpoint_id']}"
        )
    if args.resume or args.extend_end:
        if not args.output:
            parser.error("--resume/--extend-end requires --output <existing-run.db>")
        resume_run = (
            ExperimentLogger.get_running_run(args.output)
            if args.resume
            else ExperimentLogger.get_latest_run(args.output)
        )
        if not resume_run:
            print("No matching run found.")
            return 1
        mode = "Extending" if args.extend_end else "Resuming"
        print(
            f"{mode} run {resume_run['run_id']} "
            f"from {resume_run['last_decision_ts']}"
        )
        if args.extend_end:
            args.end = args.extend_end

    # Load config from TOML
    try:
        config = Config.load_from_toml(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Override config with CLI arguments
    overrides = {}
    if args.model is not None:
        overrides["model_name"] = args.model
    if args.start is not None:
        overrides["backtest_start"] = args.start
    if args.end is not None:
        overrides["backtest_end"] = args.end
    if args.interval is not None:
        overrides["decision_interval"] = args.interval
    if args.initial_cash is not None:
        overrides["initial_cash"] = args.initial_cash
    if args.max_decisions is not None:
        overrides["max_decisions"] = args.max_decisions
    if args.max_rounds is not None:
        overrides["max_agent_rounds"] = args.max_rounds
    if args.thinking:
        overrides["thinking_enabled"] = True

    # Apply overrides
    if overrides:
        config = Config(**{**config.__dict__, **overrides})

    if (
        fork_checkpoint is not None
        and config.backtest_end < fork_checkpoint["timestamp"][:10]
    ):
        parser.error("Fork end date cannot be earlier than the checkpoint date")

    # Determine model name
    model_name = (
        str(resume_run["model"])
        if resume_run is not None
        else (args.model or config.model_name)
    )

    # Run benchmark
    run_id = (
        str(resume_run["run_id"])
        if resume_run is not None
        else generate_run_id(model_name)
    )
    output_path = resolve_run_output(args.output, run_id)
    print(f"Run ID: {run_id}")
    print(f"Output: {output_path}")
    runner = ExperimentRunner(
        config,
        model=model_name,
        db_path=str(output_path),
        run_id=run_id,
        resume=resume_run is not None,
        extend=args.extend_end is not None,
        fork_checkpoint=fork_checkpoint,
        parent_run_id=str(parent_run["run_id"]) if parent_run else "",
    )
    result = runner.run()

    return 0 if result.total_return >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
