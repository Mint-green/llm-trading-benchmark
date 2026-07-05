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
    parser.add_argument("--output", default="output/results/benchmark.db", help="Output database path")
    parser.add_argument("--resume", action="store_true", help="Resume from last interrupted run")
    args = parser.parse_args()

    # Handle resume mode
    if args.resume:
        run = ExperimentLogger.get_running_run(args.output)
        if not run:
            print("No running run found to resume.")
            return 1
        print(f"Resuming run {run['run_id']} from {run['last_decision_ts']}")
        # TODO: Implement resume logic (load positions from database)
        print("Resume not yet implemented. Starting new run instead.")

    # Load config from TOML
    try:
        config = Config.load_from_toml(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Override config with CLI arguments
    overrides = {}
    if args.model is not None:
        # Determine model type and update API config
        # This is a simplified version - in production, you'd load from config
        pass
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

    # Determine model name
    model_name = args.model or config.model_name

    # Run benchmark
    runner = ExperimentRunner(config, model=model_name, db_path=args.output)
    result = runner.run()

    return 0 if result.total_return >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
