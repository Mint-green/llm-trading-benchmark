"""Supported futures product and family specifications.

Product specs are contract-level constants. Family specs group standard and
micro variants that express the same underlying futures exposure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FuturesProductSpec:
    root_symbol: str
    continuous_symbol: str
    display_name: str
    macro_role: str
    multiplier: float
    tick_size: float
    tick_value: float
    initial_margin: float
    maintenance_margin: float
    family_symbol: str
    variant: str = "standard"


@dataclass(frozen=True)
class FuturesFamilySpec:
    family_symbol: str
    underlying: str
    exposure_group: str
    macro_role: str
    standard_symbol: str | None = None
    micro_symbol: str | None = None

    @property
    def variants(self) -> tuple[str, ...]:
        return tuple(sym for sym in (self.standard_symbol, self.micro_symbol) if sym)


SUPPORTED_FUTURES_SPECS: dict[str, FuturesProductSpec] = {
    "GC": FuturesProductSpec(
        root_symbol="GC",
        continuous_symbol="GC.FUT",
        display_name="Gold futures",
        macro_role="gold / real rates / safe haven",
        multiplier=100.0,
        tick_size=0.1,
        tick_value=10.0,
        initial_margin=12_000.0,
        maintenance_margin=11_000.0,
        family_symbol="GOLD_FUT",
    ),
    "MGC": FuturesProductSpec(
        root_symbol="MGC",
        continuous_symbol="MGC.FUT",
        display_name="Micro gold futures",
        macro_role="gold / real rates / safe haven",
        multiplier=10.0,
        tick_size=0.1,
        tick_value=1.0,
        initial_margin=1_200.0,
        maintenance_margin=1_100.0,
        family_symbol="GOLD_FUT",
        variant="micro",
    ),
    "CL": FuturesProductSpec(
        root_symbol="CL",
        continuous_symbol="CL.FUT",
        display_name="Crude oil futures",
        macro_role="oil / inflation / geopolitics",
        multiplier=1_000.0,
        tick_size=0.01,
        tick_value=10.0,
        initial_margin=7_500.0,
        maintenance_margin=6_800.0,
        family_symbol="OIL_FUT",
    ),
    "MCL": FuturesProductSpec(
        root_symbol="MCL",
        continuous_symbol="MCL.FUT",
        display_name="Micro crude oil futures",
        macro_role="oil / inflation / geopolitics",
        multiplier=100.0,
        tick_size=0.01,
        tick_value=1.0,
        initial_margin=750.0,
        maintenance_margin=680.0,
        family_symbol="OIL_FUT",
        variant="micro",
    ),
    "ES": FuturesProductSpec(
        root_symbol="ES",
        continuous_symbol="ES.FUT",
        display_name="S&P 500 futures",
        macro_role="US equity beta / S&P 500",
        multiplier=50.0,
        tick_size=0.25,
        tick_value=12.5,
        initial_margin=16_000.0,
        maintenance_margin=14_500.0,
        family_symbol="SP500_FUT",
    ),
    "MES": FuturesProductSpec(
        root_symbol="MES",
        continuous_symbol="MES.FUT",
        display_name="Micro S&P 500 futures",
        macro_role="US equity beta / S&P 500",
        multiplier=5.0,
        tick_size=0.25,
        tick_value=1.25,
        initial_margin=1_600.0,
        maintenance_margin=1_450.0,
        family_symbol="SP500_FUT",
        variant="micro",
    ),
    "NQ": FuturesProductSpec(
        root_symbol="NQ",
        continuous_symbol="NQ.FUT",
        display_name="Nasdaq 100 futures",
        macro_role="US growth / Nasdaq beta",
        multiplier=20.0,
        tick_size=0.25,
        tick_value=5.0,
        initial_margin=22_000.0,
        maintenance_margin=20_000.0,
        family_symbol="NASDAQ_FUT",
    ),
    "MNQ": FuturesProductSpec(
        root_symbol="MNQ",
        continuous_symbol="MNQ.FUT",
        display_name="Micro Nasdaq 100 futures",
        macro_role="US growth / Nasdaq beta",
        multiplier=2.0,
        tick_size=0.25,
        tick_value=0.5,
        initial_margin=2_200.0,
        maintenance_margin=2_000.0,
        family_symbol="NASDAQ_FUT",
        variant="micro",
    ),
    "RTY": FuturesProductSpec(
        root_symbol="RTY",
        continuous_symbol="RTY.FUT",
        display_name="Russell 2000 futures",
        macro_role="US small-cap beta",
        multiplier=50.0,
        tick_size=0.1,
        tick_value=5.0,
        initial_margin=8_000.0,
        maintenance_margin=7_300.0,
        family_symbol="RUSSELL_FUT",
    ),
    "M2K": FuturesProductSpec(
        root_symbol="M2K",
        continuous_symbol="M2K.FUT",
        display_name="Micro Russell 2000 futures",
        macro_role="US small-cap beta",
        multiplier=5.0,
        tick_size=0.1,
        tick_value=0.5,
        initial_margin=800.0,
        maintenance_margin=730.0,
        family_symbol="RUSSELL_FUT",
        variant="micro",
    ),
    "ZN": FuturesProductSpec(
        root_symbol="ZN",
        continuous_symbol="ZN.FUT",
        display_name="10Y Treasury Note futures",
        macro_role="US rates / duration",
        multiplier=1_000.0,
        tick_size=0.015625,
        tick_value=15.625,
        initial_margin=2_400.0,
        maintenance_margin=2_200.0,
        family_symbol="UST10Y_FUT",
    ),
    "6E": FuturesProductSpec(
        root_symbol="6E",
        continuous_symbol="6E.FUT",
        display_name="Euro FX futures",
        macro_role="EUR / USD FX",
        multiplier=125_000.0,
        tick_size=0.00005,
        tick_value=6.25,
        initial_margin=3_000.0,
        maintenance_margin=2_700.0,
        family_symbol="EUR_FX_FUT",
    ),
    "M6E": FuturesProductSpec(
        root_symbol="M6E",
        continuous_symbol="M6E.FUT",
        display_name="Micro Euro FX futures",
        macro_role="EUR / USD FX",
        multiplier=12_500.0,
        tick_size=0.0001,
        tick_value=1.25,
        initial_margin=300.0,
        maintenance_margin=270.0,
        family_symbol="EUR_FX_FUT",
        variant="micro",
    ),
    "6J": FuturesProductSpec(
        root_symbol="6J",
        continuous_symbol="6J.FUT",
        display_name="Japanese yen futures",
        macro_role="JPY / USD / safe haven FX",
        multiplier=12_500_000.0,
        tick_size=0.0000005,
        tick_value=6.25,
        initial_margin=3_000.0,
        maintenance_margin=2_700.0,
        family_symbol="JPY_FX_FUT",
    ),
    "MJY": FuturesProductSpec(
        root_symbol="MJY",
        continuous_symbol="MJY.FUT",
        display_name="Micro Japanese yen futures",
        macro_role="JPY / USD / safe haven FX",
        multiplier=1_250_000.0,
        tick_size=0.000001,
        tick_value=1.25,
        initial_margin=300.0,
        maintenance_margin=270.0,
        family_symbol="JPY_FX_FUT",
        variant="micro",
    ),
    "BTC": FuturesProductSpec(
        root_symbol="BTC",
        continuous_symbol="BTC.FUT",
        display_name="Bitcoin futures",
        macro_role="bitcoin / crypto beta",
        multiplier=5.0,
        tick_size=5.0,
        tick_value=25.0,
        initial_margin=90_000.0,
        maintenance_margin=82_000.0,
        family_symbol="BTC_FUT",
    ),
    "MBT": FuturesProductSpec(
        root_symbol="MBT",
        continuous_symbol="MBT.FUT",
        display_name="Micro Bitcoin futures",
        macro_role="bitcoin / crypto beta",
        multiplier=0.1,
        tick_size=5.0,
        tick_value=0.5,
        initial_margin=1_800.0,
        maintenance_margin=1_650.0,
        family_symbol="BTC_FUT",
        variant="micro",
    ),
}

SUPPORTED_FUTURES_FAMILIES: dict[str, FuturesFamilySpec] = {
    "GOLD_FUT": FuturesFamilySpec("GOLD_FUT", "Gold", "GOLD_FUT", "gold / real rates / safe haven", "GC.FUT", "MGC.FUT"),
    "OIL_FUT": FuturesFamilySpec("OIL_FUT", "WTI Crude Oil", "OIL_FUT", "oil / inflation / geopolitics", "CL.FUT", "MCL.FUT"),
    "SP500_FUT": FuturesFamilySpec("SP500_FUT", "S&P 500", "SP500_FUT", "US equity beta / S&P 500", "ES.FUT", "MES.FUT"),
    "NASDAQ_FUT": FuturesFamilySpec("NASDAQ_FUT", "Nasdaq 100", "NASDAQ_FUT", "US growth / Nasdaq beta", "NQ.FUT", "MNQ.FUT"),
    "RUSSELL_FUT": FuturesFamilySpec("RUSSELL_FUT", "Russell 2000", "RUSSELL_FUT", "US small-cap beta", "RTY.FUT", "M2K.FUT"),
    "UST10Y_FUT": FuturesFamilySpec("UST10Y_FUT", "10Y Treasury Note", "UST10Y_FUT", "US rates / duration", "ZN.FUT", None),
    "EUR_FX_FUT": FuturesFamilySpec("EUR_FX_FUT", "Euro FX", "EUR_FX_FUT", "EUR / USD FX", "6E.FUT", "M6E.FUT"),
    "JPY_FX_FUT": FuturesFamilySpec("JPY_FX_FUT", "Japanese Yen FX", "JPY_FX_FUT", "JPY / USD / safe haven FX", "6J.FUT", "MJY.FUT"),
    "BTC_FUT": FuturesFamilySpec("BTC_FUT", "Bitcoin", "BTC_FUT", "bitcoin / crypto beta", "BTC.FUT", "MBT.FUT"),
}


DEFAULT_ALLOWED_FUTURES_SYMBOLS: tuple[str, ...] = tuple(SUPPORTED_FUTURES_FAMILIES.keys())


def get_futures_product_spec(root_or_continuous_symbol: str) -> FuturesProductSpec | None:
    root = root_or_continuous_symbol.split(".")[0]
    return SUPPORTED_FUTURES_SPECS.get(root)


def get_futures_family_spec(family_or_symbol: str) -> FuturesFamilySpec | None:
    if family_or_symbol in SUPPORTED_FUTURES_FAMILIES:
        return SUPPORTED_FUTURES_FAMILIES[family_or_symbol]
    product = get_futures_product_spec(family_or_symbol)
    if product is None:
        return None
    return SUPPORTED_FUTURES_FAMILIES.get(product.family_symbol)


def is_futures_family_symbol(symbol: str) -> bool:
    return symbol in SUPPORTED_FUTURES_FAMILIES


def futures_family_variants(family_or_symbol: str) -> tuple[str, ...]:
    family = get_futures_family_spec(family_or_symbol)
    return family.variants if family is not None else ()


def futures_symbol_allowed(symbol: str, allowed_symbols: tuple[str, ...]) -> bool:
    if symbol in allowed_symbols:
        return True
    product = get_futures_product_spec(symbol)
    if product is None:
        return False
    return product.family_symbol in allowed_symbols