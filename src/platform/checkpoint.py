"""Complete, versioned runtime checkpoints for deterministic resume."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import typing
import zlib
from collections import defaultdict
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

import msgpack

from src.core import types as core_types
from src.portfolio.portfolio import FxLog, TargetConversionResult


STATE_SCHEMA_VERSION = 1


_COMPONENT_FIELDS = {
    "portfolio": (
        "_cash", "_positions", "_futures_positions", "_futures_margin_locked",
        "_futures_margin_state", "_futures_pnl_delta", "_reserved_usd",
        "_trade_history", "_fx_log",
    ),
    "constraints": (
        "_last_buy", "_daily_trades", "_sells_this_decision",
        "_last_decision_ts", "_tail_guard_active", "_tail_guard_markets",
    ),
    "settlement": ("_buy_history",),
    "futures_account": (
        "cash_usd", "positions", "trade_history", "roll_history", "margin_state",
    ),
    "scheduler": ("_last_full_decision", "_last_market_decision", "_last_focused"),
    "event_detector": ("_prev_regime", "_prev_open_state"),
    "nav_engine": ("_fx_rates",),
    "runner": (
        "_last_light_decision", "_risk_mode", "_stop_loss_buy_pause_until",
        "_stop_loss_recent_by_market", "_pending_daily_summary_injection",
        "_logged_futures_roll_count", "_logged_futures_trade_count",
    ),
}


def capture_runtime_state(runner: Any, loop_state: dict[str, Any]) -> dict[str, Any]:
    components = {
        "portfolio": runner.portfolio,
        "constraints": runner.constraints,
        "settlement": runner.settlement,
        "futures_account": runner.futures_account,
        "scheduler": runner.scheduler,
        "event_detector": runner.event_detector,
        "nav_engine": runner.nav_engine,
        "runner": runner,
    }
    state: dict[str, Any] = {
        "schema_version": STATE_SCHEMA_VERSION,
        "loop": loop_state,
        "components": {},
        "memory": dict(runner.memory.__dict__),
    }
    for name, component in components.items():
        state["components"][name] = {
            field: getattr(component, field) for field in _COMPONENT_FIELDS[name]
        }
    return state


def restore_runtime_state(runner: Any, state: dict[str, Any]) -> dict[str, Any]:
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema: {state.get('schema_version')}"
        )
    components = {
        "portfolio": runner.portfolio,
        "constraints": runner.constraints,
        "settlement": runner.settlement,
        "futures_account": runner.futures_account,
        "scheduler": runner.scheduler,
        "event_detector": runner.event_detector,
        "nav_engine": runner.nav_engine,
        "runner": runner,
    }
    for name, values in state["components"].items():
        component = components[name]
        for field, value in values.items():
            setattr(component, field, value)
    runner.memory.__dict__.clear()
    runner.memory.__dict__.update(state["memory"])
    _normalize_enums(runner)
    return state["loop"]


def _normalize_enums(runner: Any) -> None:
    """Convert string fields back to enums after checkpoint restore.

    Old checkpoints may store enum values as plain strings inside dataclass fields.
    Uses type hints to automatically detect and convert all enum fields.
    """
    import dataclasses
    from src.core.types import (
        Market, RiskMode, OrderSide, TriggerType, DecisionType,
    )

    def _fix_obj(obj: Any) -> None:
        """Recursively fix enum fields on a dataclass or list of objects."""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            hints = typing.get_type_hints(type(obj))
            for field in dataclasses.fields(obj):
                val = getattr(obj, field.name)
                if val is None:
                    continue
                hint = hints.get(field.name)
                # Direct enum field
                if isinstance(hint, type) and issubclass(hint, Enum) and isinstance(val, str):
                    try:
                        setattr(obj, field.name, hint(val))
                    except ValueError:
                        pass
                # Nested dataclass
                elif dataclasses.is_dataclass(val):
                    _fix_obj(val)
                # List of objects
                elif isinstance(val, list):
                    for item in val:
                        _fix_obj(item)
                # Dict
                elif isinstance(val, dict):
                    for v in val.values():
                        _fix_obj(v)
        elif isinstance(obj, list):
            for item in obj:
                _fix_obj(item)
        elif isinstance(obj, dict):
            for v in obj.values():
                _fix_obj(v)

    # Fix portfolio positions
    for pos in runner.portfolio._positions.values():
        _fix_obj(pos)
    # Fix portfolio trade history
    for result in runner.portfolio._trade_history:
        _fix_obj(result)
    # Fix futures account
    for result in runner.futures_account.trade_history:
        _fix_obj(result)
    # Fix memory plans (ActivePlan → PlanTrigger)
    for plan in runner.memory._plans.values():
        _fix_obj(plan)
    # Fix risk mode on runner
    if isinstance(runner._risk_mode, str):
        try:
            runner._risk_mode = RiskMode(runner._risk_mode)
        except ValueError:
            pass
    # Fix market_exposure keys (dict keys can't be dataclass fields)
    if hasattr(runner.portfolio, '_market_exposure') and isinstance(runner.portfolio._market_exposure, dict):
        runner.portfolio._market_exposure = {
            (Market(k) if isinstance(k, str) else k): v
            for k, v in runner.portfolio._market_exposure.items()
        }


_DATACLASS_TYPES = {
    value.__name__: value
    for value in vars(core_types).values()
    if inspect.isclass(value) and is_dataclass(value)
}
_DATACLASS_TYPES.update({
    FxLog.__name__: FxLog,
    TargetConversionResult.__name__: TargetConversionResult,
})
_ENUM_TYPES = {
    value.__name__: value
    for value in vars(core_types).values()
    if inspect.isclass(value) and issubclass(value, Enum)
}


def _to_wire(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, Enum):
        return {"__enum__": value.__class__.__name__, "value": value.value}
    if is_dataclass(value) and not isinstance(value, type):
        name = value.__class__.__name__
        if name not in _DATACLASS_TYPES:
            raise TypeError(f"Unregistered checkpoint dataclass: {name}")
        return {
            "__dataclass__": name,
            "fields": {
                field.name: _to_wire(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, defaultdict):
        if value.default_factory not in (None, list, dict):
            raise TypeError("Unsupported defaultdict factory in checkpoint")
        factory = value.default_factory.__name__ if value.default_factory else None
        return {"__defaultdict__": factory, "items": _to_wire(dict(value))}
    if isinstance(value, dict):
        return {
            "__dict__": [
                [_to_wire(key), _to_wire(item)] for key, item in value.items()
            ]
        }
    if isinstance(value, tuple):
        return {"__tuple__": [_to_wire(item) for item in value]}
    if isinstance(value, list):
        return [_to_wire(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return {"__set__": [_to_wire(item) for item in value]}
    raise TypeError(f"Unsupported checkpoint type: {type(value).__name__}")


def _from_wire(value: Any) -> Any:
    if isinstance(value, list):
        return [_from_wire(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "__enum__" in value:
        enum_type = _ENUM_TYPES.get(value["__enum__"])
        if enum_type is None:
            raise ValueError(f"Unknown checkpoint enum: {value['__enum__']}")
        return enum_type(value["value"])
    if "__dataclass__" in value:
        data_type = _DATACLASS_TYPES.get(value["__dataclass__"])
        if data_type is None:
            raise ValueError(
                f"Unknown checkpoint dataclass: {value['__dataclass__']}"
            )
        kwargs = {
            key: _from_wire(item) for key, item in value["fields"].items()
        }
        return data_type(**kwargs)
    if "__defaultdict__" in value:
        factories = {None: None, "list": list, "dict": dict}
        factory_name = value["__defaultdict__"]
        if factory_name not in factories:
            raise ValueError(f"Unknown defaultdict factory: {factory_name}")
        return defaultdict(
            factories[factory_name], _from_wire(value["items"]),
        )
    if "__dict__" in value:
        return {
            _from_wire(key): _from_wire(item)
            for key, item in value["__dict__"]
        }
    if "__tuple__" in value:
        return tuple(_from_wire(item) for item in value["__tuple__"])
    if "__set__" in value:
        return set(_from_wire(item) for item in value["__set__"])
    raise ValueError("Unknown checkpoint wire object")


def encode_checkpoint(state: dict[str, Any]) -> tuple[bytes, str]:
    raw = msgpack.packb(_to_wire(state), use_bin_type=True)
    blob = zlib.compress(raw, level=6)
    return blob, hashlib.sha256(blob).hexdigest()


def decode_checkpoint(blob: bytes, expected_hash: str) -> dict[str, Any]:
    actual_hash = hashlib.sha256(blob).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("Checkpoint hash mismatch")
    wire = msgpack.unpackb(
        zlib.decompress(blob), raw=False, strict_map_key=False,
    )
    state = _from_wire(wire)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint payload must be a dictionary")
    return state