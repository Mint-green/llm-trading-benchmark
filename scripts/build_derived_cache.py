"""Offline builder for the shared, read-only futures derived cache."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.config import Config
from src.core.futures_specs import get_futures_family_spec
from src.data.cache_reader import DerivedCacheWriter, futures_cache_namespace
from src.data.features import FeatureGenerator
from src.data.futures_resolver import FuturesContractResolver
from src.data.provider import MarketDataProvider
from src.platform.run_identity import (
    build_version_metadata,
    reproducible_config_dict,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build shared futures cache")
    parser.add_argument("--config", default="config/config.toml")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")

    config = Config.load_from_toml(args.config)
    project_root = Path(__file__).resolve().parents[1]
    config_dict = reproducible_config_dict(config)
    versions = build_version_metadata(
        project_root, config_dict, config.dataset_version,
    )
    namespace = futures_cache_namespace(
        config_dict["futures"], versions.code_version,
    )
    output = Path(args.output) if args.output else (
        project_root / "artifacts" / "cache"
        / f"derived_{config.dataset_version}.db"
    )
    writer = DerivedCacheWriter(output, config.dataset_version, namespace)
    provider = MarketDataProvider(config)
    features = FeatureGenerator()
    resolver = FuturesContractResolver(config, provider)

    symbols = []
    for allowed in config.futures.allowed_symbols:
        family = get_futures_family_spec(allowed)
        symbols.extend(family.variants if family else (allowed,))
    symbols = list(dict.fromkeys(symbols))

    # Pre-load bars for all contract tickers to populate history cache.
    # This avoids repeated DB queries during resolver calls.
    preload_start = "2025-10-01"
    preload_end = args.end
    all_specs = []
    for symbol in symbols:
        raw = provider.load_futures_contracts(symbol)
        for row in raw:
            ct = row["contract_ticker"]
            all_specs.append((symbol, ct))
            # This populates _futures_contract_history_cache
            provider.load_futures_bars(symbol, ct, preload_start, preload_end)
    print(f"Pre-loaded bars for {len(all_specs)} contracts", flush=True)

    current = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d") + timedelta(days=1)
    built = 0
    commit_every = max(len(symbols) * 100, 1)
    next_cache_eviction = current + timedelta(days=5)
    try:
        while current < end:
            timestamp = current.strftime("%Y-%m-%d %H:%M")
            for symbol in symbols:
                resolved = resolver.resolve(symbol, timestamp)
                writer.put_futures_resolution(resolved)
                if resolved.contract_ticker:
                    bars = provider.load_futures_bars(
                        symbol, resolved.contract_ticker,
                        "2025-10-01", timestamp,
                    )
                    snapshot = features.compute(bars, timestamp)
                    if snapshot is not None:
                        writer.put_futures_feature(
                            symbol, resolved.contract_ticker,
                            timestamp, snapshot,
                        )
                built += 1
            if built % commit_every == 0:
                writer.commit()
                print(f"cached rows: {built} at {timestamp}", flush=True)
            if current >= next_cache_eviction:
                provider.clear_futures_caches()
                resolver.clear_caches()
                next_cache_eviction = current + timedelta(days=5)
            current += timedelta(minutes=args.interval)
        writer.commit()
    finally:
        writer.close()
        provider.close()
    print(f"Cache ready: {output} ({built} symbol-timestamps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
