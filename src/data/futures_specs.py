"""Compatibility re-export for futures specs.

The source of truth lives in src.core.futures_specs.
"""

from src.core.futures_specs import (  # noqa: F401
    DEFAULT_ALLOWED_FUTURES_SYMBOLS,
    SUPPORTED_FUTURES_SPECS,
    SUPPORTED_FUTURES_FAMILIES,
    FuturesProductSpec,
    FuturesFamilySpec,
    get_futures_product_spec,
    get_futures_family_spec,
    is_futures_family_symbol,
    futures_family_variants,
    futures_symbol_allowed,
)