"""
PortfolioEngine — manages multi-currency cash, positions, and trade execution.

Cash accounts: USD, HKD, CNY, JPY (extensible).
FX conversion happens at trade time. All conversions logged.
NAV unified in USD.
"""

from __future__ import annotations
import copy
from collections import defaultdict
from dataclasses import dataclass, field

from src.core.types import (
    Market, OrderSide, Position, PortfolioSnapshot, TradeOrder, TradeResult,
    PortfolioTarget,
)
from src.core.interfaces import IPortfolioEngine, IConstraintEngine, IExecutionEngine, ISettlementEngine, IMarketRuleEngine
from src.core.config import Config
from ..portfolio.nav import NavEngine


@dataclass
class FxLog:
    """Record of an FX conversion."""
    timestamp: str
    from_currency: str
    to_currency: str
    from_amount: float
    to_amount: float
    rate: float


@dataclass
class TargetConversionResult:
    """Result of converting portfolio targets to trade orders."""
    orders: list[TradeOrder]
    skipped: list[dict]  # {"symbol": str, "reason": str}


# Market -> local currency
MARKET_CURRENCY = {
    Market.US: "USD",
    Market.HK: "HKD",
    Market.CN: "CNY",
    Market.CRYPTO: "USD",
    Market.GOLD: "USD",
    Market.FUTURES: "USD",
}


class PortfolioEngine(IPortfolioEngine):
    """Manages multi-currency cash accounts, positions, and trade execution."""

    def __init__(
        self,
        config: Config,
        nav_engine: NavEngine,
        constraint_engine: IConstraintEngine,
        execution_engine: IExecutionEngine,
        settlement_engine: ISettlementEngine,
        market_rule_engine: IMarketRuleEngine,
    ):
        self._config = config
        self._nav = nav_engine
        self._constraints = constraint_engine
        self._execution = execution_engine
        self._settlement = settlement_engine
        self._market_rules = market_rule_engine

        # Multi-currency cash accounts
        self._cash: dict[str, float] = {
            "USD": config.initial_cash,
            "HKD": 0.0,
            "CNY": 0.0,
            "JPY": 0.0,
        }

        self._positions: dict[str, Position] = {}  # key: "MARKET:SYMBOL"
        self._futures_positions: dict = {}
        self._futures_margin_locked: float = 0.0
        self._futures_margin_state: str = "OK"
        self._futures_pnl_delta: float = 0.0
        self._reserved_usd: float = 0.0
        self._trade_history: list[TradeResult] = []
        self._fx_log: list[FxLog] = []

    @property
    def cash(self) -> float:
        """Total cash in USD (all currencies converted)."""
        return sum(
            self._nav.convert_to_usd(amount, currency)
            for currency, amount in self._cash.items()
        )

    def get_cash(self, currency: str) -> float:
        """Get cash in a specific currency."""
        return self._cash.get(currency, 0.0)

    @property
    def nav(self) -> float:
        return self._nav.compute_nav(self.cash, list(self._positions.values()))

    def get_position(self, key: str) -> Position | None:
        return self._positions.get(key)

    def get_snapshot(self, timestamp: str) -> PortfolioSnapshot:
        market_exposure: dict[Market, float] = defaultdict(float)
        for pos in self._positions.values():
            market_exposure[pos.market] += pos.market_value

        # Deep copy positions to prevent mutation from later _update_prices
        positions_copy = {k: copy.copy(v) for k, v in self._positions.items()}

        # Get T+1 frozen keys (CN positions bought today)
        frozen_keys = self._settlement.get_frozen_keys(timestamp)

        return PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            positions=positions_copy,
            total_nav=self.nav,
            market_exposure=dict(market_exposure),
            fx_rates=dict(self._config.fx_rates),
            frozen_keys=frozen_keys,
            futures_positions=dict(self._futures_positions),
            futures_margin_locked=self._futures_margin_locked,
            futures_margin_state=self._futures_margin_state,
            futures_pnl_delta=self._futures_pnl_delta,
        )

    def ensure_cash(self, currency: str, amount_needed: float, timestamp: str) -> bool:
        """Ensure enough cash in the given currency. Auto-convert from USD if needed.

        FX fee is applied on the USD side (we pay more USD to get the same foreign amount).
        Returns True if successful, False if insufficient total funds.
        """
        current = self._cash.get(currency, 0.0)
        if currency == "USD":
            current = max(0.0, current - self._reserved_usd)
        if current >= amount_needed:
            return True
        if currency == "USD":
            return False

        deficit = amount_needed - current
        # Try to convert from USD (include FX fee)
        usd_needed = self._nav.convert_to_usd(deficit, currency)
        fx_fee = usd_needed * (self._config.fx_fee_bps / 10_000)
        usd_total = usd_needed + fx_fee
        usd_available = max(0.0, self._cash["USD"] - self._reserved_usd)

        if usd_available < usd_total:
            return False  # insufficient funds overall

        # Execute FX conversion (deduct USD + fee, credit foreign currency)
        self._cash["USD"] -= usd_total
        self._cash[currency] = self._cash.get(currency, 0.0) + deficit

        # Log FX conversion
        rate = self._config.fx_rates.get(currency, 1.0)
        self._fx_log.append(FxLog(
            timestamp=timestamp,
            from_currency="USD",
            to_currency=currency,
            from_amount=usd_needed,
            to_amount=deficit,
            rate=rate,
        ))
        return True

    def execute_buy(
        self, symbol: str, market: Market, quantity: float, price: float, fees: float,
    ) -> None:
        """Execute a buy. Price in local currency, stored as USD in position."""
        currency = MARKET_CURRENCY.get(market, "USD")
        cost_local = price * quantity + fees

        # Deduct from local currency account
        self._cash[currency] -= cost_local

        # Convert price to USD for position storage
        price_usd = self._to_usd(price, market)

        key = f"{market.value}:{symbol}"
        existing = self._positions.get(key)

        if existing:
            total_qty = existing.quantity + quantity
            total_cost = existing.avg_cost * existing.quantity + price_usd * quantity
            existing.avg_cost = total_cost / total_qty
            existing.quantity = total_qty
            existing.current_price = price_usd
        else:
            self._positions[key] = Position(
                symbol=symbol,
                market=market,
                quantity=quantity,
                avg_cost=price_usd,
                current_price=price_usd,
            )

    def execute_sell(
        self, symbol: str, market: Market, quantity: float, price: float, fees: float,
    ) -> None:
        """Execute a sell. Price in local currency."""
        currency = MARKET_CURRENCY.get(market, "USD")
        proceeds_local = price * quantity - fees

        # Add to local currency account
        self._cash[currency] += proceeds_local

        key = f"{market.value}:{symbol}"
        pos = self._positions[key]
        pos.quantity -= quantity
        pos.current_price = self._to_usd(price, market)

        if pos.quantity == 0:
            del self._positions[key]

    def process_order(
        self, order: TradeOrder, price: float, timestamp: str,
    ) -> TradeResult:
        """Full order processing pipeline: rules → constraints → FX → execution → settlement."""
        result = None

        # 1. Sell limit per decision (max 3 SELLs)
        if order.side == OrderSide.SELL:
            if self._constraints._sells_this_decision >= 3:
                result = TradeResult(
                    order=order, success=False,
                    error="limit: max 3 SELLs per decision. Sell gradually across decisions."
                )
                self._trade_history.append(result)
                return result
            self._constraints._sells_this_decision += 1

        # 2. Daily BUY limit; SELLs remain allowed for risk reduction.
        if order.side == OrderSide.BUY:
            ok, reason = self._constraints.check_daily_limit(timestamp)
            if not ok:
                result = TradeResult(order=order, success=False, error=f"limit: {reason}")
                self._trade_history.append(result)
                return result

        # 2. Market rules check
        can_trade, rule_reason = self._market_rules.can_trade(
            order.market, order.symbol, order.side, timestamp,
        )
        if not can_trade:
            result = TradeResult(order=order, success=False, error=f"market_rule: {rule_reason}")
            self._trade_history.append(result)
            return result

        # 3. Constraint check (price in USD) — BEFORE FX conversion
        price_usd = self._to_usd(price, order.market)
        if order.side == OrderSide.BUY:
            ok, reason = self._constraints.validate_buy(
                order.symbol, order.market, order.quantity, price_usd,
                self.nav, self._positions,
            )
            if not ok:
                result = TradeResult(order=order, success=False, error=f"constraint: {reason}")
                self._trade_history.append(result)
                return result
        else:
            key = f"{order.market.value}:{order.symbol}"
            sellable = self._settlement.get_sellable_quantity(key, timestamp)
            if order.quantity > sellable:
                result = TradeResult(
                    order=order, success=False,
                    error=f"T+1: can only sell {sellable} of {order.quantity}",
                )
                self._trade_history.append(result)
                return result
            ok, reason = self._constraints.validate_sell(
                key, order.quantity, self._positions, timestamp=timestamp,
            )
            if not ok:
                result = TradeResult(order=order, success=False, error=f"constraint: {reason}")
                self._trade_history.append(result)
                return result

        # 4. Ensure local currency funds (for buys) — AFTER constraint check
        currency = MARKET_CURRENCY.get(order.market, "USD")
        if order.side == OrderSide.BUY:
            cost_estimate = price * order.quantity * 1.002  # rough fee estimate
            if not self.ensure_cash(currency, cost_estimate, timestamp):
                result = TradeResult(order=order, success=False, error=f"insufficient {currency} funds")
                self._trade_history.append(result)
                return result

        # 5. Execution
        result = self._execution.execute(order, price, timestamp)
        if not result.success:
            self._trade_history.append(result)
            return result

        # 6. Update portfolio (in local currency) — use rounded quantity from execution
        if order.side == OrderSide.BUY:
            self.execute_buy(order.symbol, order.market, result.order.quantity, result.price, result.fees)
            # Record buy time for cooling period
            key = f"{order.market.value}:{order.symbol}"
            self._constraints.record_buy(key, timestamp)
            # Record trade for daily limit (only BUYs count; SELLs always allowed)
            self._constraints.record_trade(timestamp)
        else:
            self.execute_sell(order.symbol, order.market, result.order.quantity, result.price, result.fees)

        # 7. Settlement
        self._settlement.settle(result, timestamp)

        # 8. Record
        self._trade_history.append(result)
        return result

    def _to_usd(self, price: float, market: Market) -> float:
        currency = MARKET_CURRENCY.get(market, "USD")
        return self._nav.convert_to_usd(price, currency)

    @property
    def trade_history(self) -> list[TradeResult]:
        return list(self._trade_history)

    @property
    def fx_log(self) -> list[FxLog]:
        return list(self._fx_log)


    def sync_futures_state(self, cash_usd: float, positions: dict, margin_locked: float, margin_state: str, pnl_delta: float = 0.0) -> None:
        """Sync futures account cash and reserved margin into portfolio state."""
        self._cash["USD"] = cash_usd
        self._futures_positions = dict(positions)
        self._futures_margin_locked = margin_locked
        self._futures_margin_state = margin_state
        self._futures_pnl_delta = pnl_delta
        self._reserved_usd = margin_locked

    @property
    def reserved_usd(self) -> float:
        return self._reserved_usd

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for all positions. prices in USD."""
        for key, price in prices.items():
            pos = self._positions.get(key)
            if pos:
                pos.current_price = price
                pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity

    def convert_targets_to_orders(
        self,
        targets: list[PortfolioTarget],
        prices: dict[str, float],  # symbol -> price in local currency
        markets: dict[str, Market],  # symbol -> market
    ) -> TargetConversionResult:
        """Convert portfolio targets (target_pct_nav) to trade orders.

        For each target:
        1. Compute current % NAV for existing position
        2. Compute delta = target - current
        3. Convert delta to quantity (BUY or SELL)
        4. Return list of TradeOrders

        Args:
            targets: LLM's portfolio targets
            prices: current prices in local currency
            markets: symbol -> market mapping
        """
        nav = self.nav
        if nav <= 0:
            return TargetConversionResult(orders=[], skipped=[{"symbol": "*", "reason": "NAV is zero"}])

        orders: list[TradeOrder] = []
        skipped: list[dict] = []

        # Sort by priority (higher priority first)
        sorted_targets = sorted(targets, key=lambda t: t.priority, reverse=True)

        for target in sorted_targets:
            sym = target.symbol
            market = markets.get(sym)
            if market is None:
                skipped.append({"symbol": sym, "reason": "unknown market"})
                continue

            price = prices.get(sym)
            if price is None or price <= 0:
                skipped.append({"symbol": sym, "reason": "price unavailable"})
                continue

            # Current position
            key = f"{market.value}:{sym}"
            pos = self._positions.get(key)
            current_value_usd = pos.market_value if pos else 0.0
            current_pct_nav = current_value_usd / nav if nav > 0 else 0.0

            # Target value
            target_value_usd = nav * target.target_pct_nav
            delta_value_usd = target_value_usd - current_value_usd

            # Convert to local currency
            currency = MARKET_CURRENCY.get(market, "USD")
            price_usd = self._to_usd(price, market)

            if abs(delta_value_usd) < nav * 0.005:  # < 0.5% NAV, skip
                skipped.append({"symbol": sym, "reason": "delta too small (< 0.5% NAV)"})
                continue

            if delta_value_usd > 0:
                # BUY
                if market in (Market.CRYPTO, Market.GOLD):
                    quantity = round(delta_value_usd / price_usd, 8) if price_usd > 0 else 0
                else:
                    quantity = int(delta_value_usd / price_usd) if price_usd > 0 else 0
                if quantity > 0:
                    orders.append(TradeOrder(
                        symbol=sym,
                        market=market,
                        side=OrderSide.BUY,
                        quantity=quantity,
                        allocation_pct=target.target_pct_nav,
                        reason=target.reason,
                    ))
            elif delta_value_usd < 0:
                # SELL
                if pos is None:
                    skipped.append({"symbol": sym, "reason": "no position to sell"})
                    continue
                if market in (Market.CRYPTO, Market.GOLD):
                    quantity = min(pos.quantity, round(abs(delta_value_usd) / price_usd, 8)) if price_usd > 0 else 0
                else:
                    quantity = min(pos.quantity, int(abs(delta_value_usd) / price_usd)) if price_usd > 0 else 0
                if quantity > 0:
                    orders.append(TradeOrder(
                        symbol=sym,
                        market=market,
                        side=OrderSide.SELL,
                        quantity=quantity,
                        allocation_pct=target.target_pct_nav,
                        reason=target.reason,
                    ))

        return TargetConversionResult(orders=orders, skipped=skipped)
