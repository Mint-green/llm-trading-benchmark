"""Standalone futures execution, margin, and mark-to-market logic."""

from __future__ import annotations

import math

from src.core.config import Config
from src.core.types import (
    FuturesMarkResult,
    FuturesPosition,
    Market,
    OrderSide,
    TradeOrder,
    TradeResult,
)
from src.data.futures_resolver import FuturesContractResolver
from src.data.provider import MarketDataProvider


class FuturesAccount:
    """Variation-margin futures account for benchmark simulation."""

    def __init__(self, config: Config, data: MarketDataProvider, resolver: FuturesContractResolver, cash_usd: float):
        self._config = config
        self._data = data
        self._resolver = resolver
        self.cash_usd = cash_usd
        self.positions: dict[str, FuturesPosition] = {}
        self.trade_history: list[TradeResult] = []
        self.mark_history: list[FuturesMarkResult] = []
        self.roll_history: list[dict] = []
        self.margin_state = "OK"

    @property
    def margin_locked(self) -> float:
        return sum(p.margin_locked for p in self.positions.values())

    @property
    def available_cash_usd(self) -> float:
        return self.cash_usd - self.margin_locked

    @property
    def nav(self) -> float:
        # Futures variation PnL is already settled into cash.
        return self.cash_usd

    def process_order(self, order: TradeOrder, timestamp: str) -> TradeResult:
        if order.market != Market.FUTURES or order.asset_type != "futures":
            return self._reject(order, "not a futures order")
        if order.symbol not in self._config.futures.allowed_symbols:
            return self._reject(order, "futures symbol not allowed")

        action = (order.action or "").upper()
        if order.side == OrderSide.SELL or order.futures_side == "flat" or action == "CLOSE":
            return self._close(order, timestamp)
        return self._open_or_increase(order, timestamp)

    def mark_to_market(self, timestamp: str) -> list[FuturesMarkResult]:
        marks: list[FuturesMarkResult] = []
        for key, pos in list(self.positions.items()):
            bar = self._data.get_last_completed_futures_bar(pos.continuous_symbol, pos.contract_ticker, timestamp)
            if bar is None:
                continue
            previous = pos.previous_mark_price
            current = bar.close
            pnl_delta = pos.side_sign * pos.contracts * pos.multiplier * (current - previous)
            self.cash_usd += pnl_delta
            pos.cumulative_variation_pnl += pnl_delta
            pos.previous_mark_price = current
            pos.current_price = current
            pos.updated_at = timestamp
            state = self._compute_margin_state()
            mark = FuturesMarkResult(
                timestamp=timestamp,
                continuous_symbol=pos.continuous_symbol,
                contract_ticker=pos.contract_ticker,
                previous_mark_price=previous,
                current_price=current,
                pnl_delta=pnl_delta,
                cumulative_variation_pnl=pos.cumulative_variation_pnl,
                margin_locked=pos.margin_locked,
                margin_state=state,
            )
            marks.append(mark)
            self.mark_history.append(mark)
            self._roll_if_needed(pos, timestamp)
        self.margin_state = self._compute_margin_state()
        if self.margin_state == "BREACH":
            self.force_liquidate(timestamp)
        return marks

    def force_liquidate(self, timestamp: str) -> list[TradeResult]:
        results: list[TradeResult] = []
        ranked = sorted(
            self.positions.values(),
            key=lambda p: (p.margin_locked, -p.cumulative_variation_pnl, p.notional),
            reverse=True,
        )
        for pos in ranked:
            order = TradeOrder(
                symbol=pos.continuous_symbol,
                market=Market.FUTURES,
                side=OrderSide.SELL,
                quantity=pos.contracts,
                reason="forced liquidation: margin breach",
                asset_type="futures",
                action="CLOSE",
                futures_side="flat",
            )
            result = self._close(order, timestamp, forced=True)
            results.append(result)
            if self._compute_margin_state() != "BREACH":
                break
        self.margin_state = self._compute_margin_state()
        return results


    def _roll_if_needed(self, pos: FuturesPosition, timestamp: str) -> None:
        """Roll a held contract when the point-in-time resolver has moved on.

        The old contract is already marked to timestamp before this method runs.
        Roll PnL is therefore only the executable close/open prices and costs,
        not the continuous-contract price gap itself.
        """
        resolved = self._resolver.resolve(pos.continuous_symbol, timestamp)
        if not resolved.contract_ticker or resolved.contract_ticker == pos.contract_ticker:
            return
        old_contract = pos.contract_ticker
        old_contracts = pos.contracts
        old_price = pos.current_price
        old_notional = pos.notional
        old_cumulative_pnl = pos.cumulative_variation_pnl

        close_order = TradeOrder(
            symbol=pos.continuous_symbol,
            market=Market.FUTURES,
            side=OrderSide.SELL,
            quantity=pos.contracts,
            reason="roll old futures contract",
            asset_type="futures",
            action="CLOSE",
            futures_side="flat",
        )
        close_result = self._close(close_order, timestamp, roll=True)
        if not close_result.success:
            return

        target_pct = old_notional / self.nav if self.nav > 0 else 0.0
        open_order = TradeOrder(
            symbol=pos.continuous_symbol,
            market=Market.FUTURES,
            side=OrderSide.BUY,
            reason="roll into resolved futures contract",
            asset_type="futures",
            action="OPEN_OR_INCREASE",
            futures_side=pos.side,
            target_notional_pct_nav=target_pct,
            max_margin_pct_nav=self._config.futures.max_margin_pct_nav,
            risk_budget_pct_nav=self._config.futures.max_risk_budget_pct_nav,
        )
        open_result = self._open_or_increase(open_order, timestamp, roll=True)
        if not open_result.success:
            self.roll_history.append({
                "timestamp": timestamp,
                "continuous_symbol": pos.continuous_symbol,
                "old_contract": old_contract,
                "new_contract": resolved.contract_ticker,
                "old_contracts": old_contracts,
                "new_contracts": 0,
                "old_close_price": close_result.price,
                "new_open_price": 0.0,
                "roll_gap": 0.0,
                "roll_cost": close_result.fees,
                "selection_method": resolved.selection_method,
                "status": "closed_not_reopened",
                "reject_reason": open_result.error,
            })
            return

        new_contract = open_result.metadata.get("actual_contract", resolved.contract_ticker)
        new_open_price = open_result.price
        self.roll_history.append({
            "timestamp": timestamp,
            "continuous_symbol": pos.continuous_symbol,
            "old_contract": old_contract,
            "new_contract": new_contract,
            "old_contracts": old_contracts,
            "new_contracts": open_result.order.quantity,
            "old_close_price": close_result.price,
            "new_open_price": new_open_price,
            "roll_gap": new_open_price - close_result.price,
            "roll_cost": close_result.fees + open_result.fees,
            "selection_method": resolved.selection_method,
            "status": "rolled",
            "old_cumulative_variation_pnl": old_cumulative_pnl,
        })

    def _open_or_increase(self, order: TradeOrder, timestamp: str, roll: bool = False) -> TradeResult:
        if order.futures_side == "short" and not self._config.futures.allow_short:
            return self._reject(order, "short futures disabled")

        resolved = self._resolver.resolve(order.symbol, timestamp)
        if not resolved.contract_ticker or resolved.price is None or not resolved.notional_per_contract:
            return self._reject(order, "no active futures contract")

        exec_bar = self._data.get_next_executable_futures_bar(order.symbol, resolved.contract_ticker, timestamp)
        if exec_bar is None:
            return self._reject(order, "no next executable futures bar")

        base_price = exec_bar.open if self._config.futures.execution_price_mode == "next_bar_open" else exec_bar.close
        slip = self._config.futures.slippage_bps / 10_000
        side = order.futures_side or "long"
        fill_multiplier = 1 - slip if side == "short" else 1 + slip
        fill_price = self._round_to_tick(
            base_price * fill_multiplier,
            resolved.tick_size,
        )
        notional_per_contract = fill_price * resolved.multiplier
        target_notional_pct = order.target_notional_pct_nav
        if target_notional_pct is None or target_notional_pct <= 0:
            return self._reject(order, "missing target_notional_pct_nav")

        target_notional_usd = self.nav * target_notional_pct
        target_contracts = math.floor(abs(target_notional_usd) / notional_per_contract)
        if target_contracts < 1 and roll:
            target_contracts = 1
        if target_contracts < 1:
            return self._reject(order, "target_notional_too_small_for_one_contract")
        if target_contracts > self._config.futures.max_contracts_per_symbol:
            return self._reject(order, "max_contracts_per_symbol_exceeded")

        required_margin = target_contracts * resolved.initial_margin
        max_margin_pct = order.max_margin_pct_nav or self._config.futures.max_margin_pct_nav
        if required_margin > self.nav * max_margin_pct:
            return self._reject(order, "one_contract_exceeds_notional_or_margin_limit")
        if notional_per_contract * target_contracts > self.nav * self._config.futures.max_abs_notional_pct_nav:
            return self._reject(order, "one_contract_exceeds_notional_or_margin_limit")
        if required_margin + self.margin_locked > self.nav * self._config.futures.max_total_margin_pct_nav:
            return self._reject(order, "total_futures_margin_cap_exceeded")

        risk_budget_pct = order.risk_budget_pct_nav or self._config.futures.max_risk_budget_pct_nav
        adverse_loss = target_contracts * resolved.multiplier * self._estimate_atr_price_move(order.symbol, resolved.contract_ticker, timestamp, fill_price)
        if adverse_loss > self.nav * risk_budget_pct:
            return self._reject(order, "risk_budget_exceeded")

        commission = target_contracts * self._config.futures.commission_per_contract
        if self.available_cash_usd < required_margin + commission:
            return self._reject(order, "insufficient futures available cash")

        key = f"FUTURES:{order.symbol}"
        if key in self.positions:
            return self._reject(order, "futures increase not implemented for existing position")

        self.cash_usd -= commission
        pos = FuturesPosition(
            continuous_symbol=order.symbol,
            contract_ticker=resolved.contract_ticker,
            side=side,
            contracts=target_contracts,
            avg_entry_price=fill_price,
            previous_mark_price=fill_price,
            current_price=fill_price,
            multiplier=resolved.multiplier,
            initial_margin_per_contract=resolved.initial_margin,
            maintenance_margin_per_contract=resolved.maintenance_margin,
            margin_locked=required_margin,
            opened_at=timestamp,
            updated_at=timestamp,
        )
        self.positions[key] = pos
        result = TradeResult(
            order=TradeOrder(
                symbol=order.symbol,
                market=Market.FUTURES,
                side=OrderSide.BUY,
                quantity=target_contracts,
                reason=order.reason,
                asset_type="futures",
                action="OPEN_OR_INCREASE",
                futures_side=pos.side,
                target_notional_pct_nav=target_notional_pct,
                max_margin_pct_nav=max_margin_pct,
                risk_budget_pct_nav=risk_budget_pct,
                unit_hint=order.unit_hint,
                risk_trigger=order.risk_trigger,
            ),
            success=True,
            price=fill_price,
            cost=notional_per_contract * target_contracts,
            fees=commission,
            metadata={
                "actual_contract": resolved.contract_ticker,
                "notional_per_contract": notional_per_contract,
                "margin_locked": required_margin,
                "roll_status": resolved.roll_status,
                "selection_method": resolved.selection_method,
                "execution_bar_timestamp": exec_bar.timestamp,
                "roll_trade": roll,
            },
        )
        self.trade_history.append(result)
        self.margin_state = self._compute_margin_state()
        return result

    def _close(self, order: TradeOrder, timestamp: str, forced: bool = False, roll: bool = False) -> TradeResult:
        key = f"FUTURES:{order.symbol}"
        pos = self.positions.get(key)
        if pos is None:
            return self._reject(order, "no futures position to close")
        exec_bar = self._data.get_next_executable_futures_bar(order.symbol, pos.contract_ticker, timestamp)
        if exec_bar is None:
            return self._reject(order, "no next executable futures bar")
        base_price = exec_bar.open if self._config.futures.execution_price_mode == "next_bar_open" else exec_bar.close
        slip = self._config.futures.slippage_bps / 10_000
        fill_multiplier = 1 + slip if pos.side == "short" else 1 - slip
        fill_price = self._round_to_tick(
            base_price * fill_multiplier,
            self._config.futures.gc_tick_size,
        )
        pnl_delta = pos.side_sign * pos.contracts * pos.multiplier * (fill_price - pos.previous_mark_price)
        commission = pos.contracts * self._config.futures.commission_per_contract
        self.cash_usd += pnl_delta
        self.cash_usd -= commission
        pos.cumulative_variation_pnl += pnl_delta
        released_margin = pos.margin_locked
        del self.positions[key]
        result = TradeResult(
            order=TradeOrder(
                symbol=order.symbol,
                market=Market.FUTURES,
                side=OrderSide.SELL,
                quantity=pos.contracts,
                reason=order.reason,
                asset_type="futures",
                action="CLOSE",
                futures_side="flat",
            ),
            success=True,
            price=fill_price,
            cost=fill_price * pos.multiplier * pos.contracts,
            fees=commission,
            metadata={
                "actual_contract": pos.contract_ticker,
                "released_margin": released_margin,
                "pnl_delta": pnl_delta,
                "forced_liquidation": forced,
                "roll_trade": roll,
                "execution_bar_timestamp": exec_bar.timestamp,
            },
        )
        self.trade_history.append(result)
        self.margin_state = self._compute_margin_state()
        return result

    def _compute_margin_state(self) -> str:
        if not self.positions:
            return "OK"
        initial_required = sum(p.contracts * p.initial_margin_per_contract for p in self.positions.values())
        maintenance_required = sum(p.contracts * p.maintenance_margin_per_contract for p in self.positions.values())
        if self.cash_usd < maintenance_required:
            return "BREACH"
        if self.cash_usd < initial_required:
            return "WARNING"
        return "OK"

    def _estimate_atr_price_move(self, symbol: str, contract: str, timestamp: str, fallback_price: float) -> float:
        bars = self._data.load_futures_bars(symbol, contract, "2025-10-01", timestamp)
        relevant = [b for b in bars if b.timestamp <= timestamp]
        if len(relevant) < 15:
            return fallback_price * 0.01
        trs = []
        for prev, cur in zip(relevant[-15:-1], relevant[-14:]):
            trs.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
        return sum(trs) / len(trs) if trs else fallback_price * 0.01

    @staticmethod
    def _round_to_tick(price: float, tick_size: float) -> float:
        if tick_size <= 0:
            return price
        return round(price / tick_size) * tick_size

    def _reject(self, order: TradeOrder, reason: str) -> TradeResult:
        result = TradeResult(order=order, success=False, error=reason)
        self.trade_history.append(result)
        return result
