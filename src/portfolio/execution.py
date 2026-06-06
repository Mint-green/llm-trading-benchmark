"""
ExecutionEngine — handles order execution with cost model and lot rounding.

Lot sizes loaded from getStockData project:
  US: 1 share (no lot restriction)
  CN: 100 shares per lot (fixed)
  HK: varies by stock (loaded from HK_shares.py)
  Crypto: none (fractional ok)
"""

from __future__ import annotations
import math
import sys
import os

from src.core.types import Market, TradeOrder, TradeResult, OrderSide
from src.core.interfaces import IExecutionEngine
from src.core.config import Config
from ..portfolio.nav import NavEngine


# Load lot sizes from getStockData
_STOCKDATA_DIR = "D:/Projects/claw/getStockData"
_get_lot_size = None
HK_LOT_SIZES: dict[str, int] = {}
DEFAULT_HK_LOT = 100

try:
    if _STOCKDATA_DIR not in sys.path:
        sys.path.insert(0, _STOCKDATA_DIR)
    from data_fetchers.fetch_template import get_lot_size as _get_lot_size
    from constituents.HK_shares import HK_STOCKS_LOT
    HK_LOT_SIZES = HK_STOCKS_LOT
except ImportError:
    pass


class ExecutionEngine(IExecutionEngine):
    """Executes orders with lot rounding and cost model."""

    def __init__(self, config: Config, nav_engine: NavEngine):
        self._config = config
        self._nav = nav_engine

    def execute(
        self, order: TradeOrder, price: float, timestamp: str,
    ) -> TradeResult:
        """Execute an order with lot rounding and fee calculation.

        Price is in local currency. Fees in local currency.
        """
        if price <= 0:
            return TradeResult(order=order, success=False, error="invalid price")

        # Lot rounding
        rounded_qty = self._round_lots(order.market, order.symbol, order.quantity, order.side)
        if rounded_qty <= 0:
            return TradeResult(order=order, success=False, error="quantity rounds to 0 after lot adjustment")

        market = order.market
        notional = price * rounded_qty

        # Compute fees in basis points
        commission_bps = self._config.commission_bps.get(market, 5.0)
        slippage_bps = self._config.slippage_bps.get(market, 5.0)
        total_bps = commission_bps + slippage_bps

        # CN sell-side stamp duty
        if market == Market.CN and order.side == OrderSide.SELL:
            total_bps += self._config.cn_sell_tax_bps

        fees_local = notional * (total_bps / 10_000)

        return TradeResult(
            order=TradeOrder(
                symbol=order.symbol,
                market=order.market,
                side=order.side,
                quantity=rounded_qty,
                reason=order.reason,
                order_type=order.order_type,
            ),
            success=True,
            price=price,
            cost=notional,
            fees=fees_local,
        )

    @staticmethod
    def _round_lots(market: Market, symbol: str, quantity: int, side: OrderSide) -> int:
        """Round quantity to valid lot size for the market.

        Buy: round DOWN (can't buy partial lots)
        Sell: don't round (can sell any quantity held)
        """
        if side == OrderSide.SELL:
            return quantity  # sell any quantity

        if market in (Market.US, Market.CRYPTO):
            return max(1, math.floor(quantity))

        if market == Market.CN:
            lots = math.floor(quantity / 100)
            return lots * 100

        if market == Market.HK:
            # Use getStockData's lot size if available, else fallback to HK_LOT_SIZES dict
            if _get_lot_size:
                lot_size = _get_lot_size("HK", symbol)
            else:
                lot_size = HK_LOT_SIZES.get(symbol, DEFAULT_HK_LOT)
            if lot_size is None:
                lot_size = DEFAULT_HK_LOT
            lots = math.floor(quantity / lot_size)
            return lots * lot_size

        return max(1, math.floor(quantity))
