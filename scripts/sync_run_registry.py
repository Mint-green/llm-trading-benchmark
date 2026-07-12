from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.platform.run_registry import sync_run_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronize run metadata registry")
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--registry", default="artifacts/run_registry.db")
    args = parser.parse_args()
    count = sync_run_registry(args.runs_dir, args.registry)
    print(f"Synchronized {count} run artifact(s) into {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
