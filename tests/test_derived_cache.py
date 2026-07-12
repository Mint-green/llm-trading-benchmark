from __future__ import annotations

from src.core.types import FuturesResolvedContract, IndicatorSnapshot
from src.data.cache_reader import DerivedCacheReader, DerivedCacheWriter


def test_shared_cache_round_trip_and_namespace_invalidation(tmp_path) -> None:
    path = tmp_path / "derived.db"
    resolved = FuturesResolvedContract(
        continuous_symbol="MGC.FUT",
        contract_ticker="MGCJ6",
        timestamp="2026-02-05 14:30",
        price=4900.0,
    )
    feature = IndicatorSnapshot(
        timestamp="2026-02-05 14:30",
        price=4900.0,
        chg_5m=0.1,
        chg_1h=0.5,
        chg_1d=1.0,
        rel_volume=1.2,
        rsi=55.0,
        atr_pct=0.4,
        trend="UU",
        bb_position=0.6,
        high_low_pos=0.7,
    )
    writer = DerivedCacheWriter(path, "dataset-v1", "namespace-a")
    writer.put_futures_resolution(resolved)
    writer.put_futures_feature(
        "MGC.FUT", "MGCJ6", "2026-02-05 14:30", feature,
    )
    writer.close()

    reader = DerivedCacheReader(path, "dataset-v1", "namespace-a")
    assert reader.get_futures_resolution(
        "MGC.FUT", "2026-02-05 14:30",
    ) == resolved
    assert reader.get_futures_feature(
        "MGC.FUT", "MGCJ6", "2026-02-05 14:30",
    ) == feature
    reader.close()

    stale = DerivedCacheReader(path, "dataset-v1", "namespace-b")
    assert stale.get_futures_resolution(
        "MGC.FUT", "2026-02-05 14:30",
    ) is None
    stale.close()
