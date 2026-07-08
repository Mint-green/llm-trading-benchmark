"""
Phase 3 unit tests — verify tool schemas, execution, and function calling integration.
"""

import pytest
import json
from src.core.types import Market, PortfolioSnapshot, Position
from src.agent.tools import ToolSystem, TOOL_SCHEMAS


# ============================================================
# Test: Tool Schemas
# ============================================================

class TestToolSchemas:
    def test_schema_count(self):
        """Should have 8 tool schemas."""
        assert len(TOOL_SCHEMAS) == 8

    def test_schema_names(self):
        """Should have correct tool names."""
        names = {t["function"]["name"] for t in TOOL_SCHEMAS}
        expected = {
            "screen_universe", "query_asset", "query_position",
            "query_history", "query_market_overview", "query_fx",
            "query_futures_contract", "query_futures_family",
        }
        assert names == expected

    def test_screen_universe_schema(self):
        """screen_universe should have required fields."""
        schema = next(t for t in TOOL_SCHEMAS if t["function"]["name"] == "screen_universe")
        params = schema["function"]["parameters"]
        assert "market" in params["properties"]
        assert "bucket" in params["properties"]
        assert "limit" in params["properties"]
        assert "market" in params["required"]
        assert "bucket" in params["required"]
        assert "limit" in params["required"]

    def test_query_asset_schema(self):
        """query_asset should have required fields."""
        schema = next(t for t in TOOL_SCHEMAS if t["function"]["name"] == "query_asset")
        params = schema["function"]["parameters"]
        assert "symbol" in params["properties"]
        assert "fields" in params["properties"]
        assert "symbol" in params["required"]
        assert "fields" in params["required"]

    def test_query_position_schema(self):
        """query_position should have required fields."""
        schema = next(t for t in TOOL_SCHEMAS if t["function"]["name"] == "query_position")
        params = schema["function"]["parameters"]
        assert "symbol" in params["properties"]
        assert "symbol" in params["required"]

    def test_query_history_schema(self):
        """query_history should have required fields."""
        schema = next(t for t in TOOL_SCHEMAS if t["function"]["name"] == "query_history")
        params = schema["function"]["parameters"]
        assert "symbol" in params["properties"]
        assert "lookback_bars" in params["properties"]
        assert "bar_size" in params["properties"]
        assert "symbol" in params["required"]
        assert "lookback_bars" in params["required"]
        assert "bar_size" in params["required"]

    def test_query_market_overview_schema(self):
        """query_market_overview should have required fields."""
        schema = next(t for t in TOOL_SCHEMAS if t["function"]["name"] == "query_market_overview")
        params = schema["function"]["parameters"]
        assert "markets" in params["properties"]
        assert "markets" in params["required"]

    def test_query_fx_schema(self):
        """query_fx should have no required fields."""
        schema = next(t for t in TOOL_SCHEMAS if t["function"]["name"] == "query_fx")
        params = schema["function"]["parameters"]
        assert "required" not in params or len(params.get("required", [])) == 0

    def test_all_schemas_are_function_type(self):
        """All schemas should be function type."""
        for schema in TOOL_SCHEMAS:
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]


# ============================================================
# Test: Tool Execution
# ============================================================

class MockDataProvider:
    """Mock data provider for testing."""

    def get_universe_symbols(self, market):
        return ["AAPL.US", "MSFT.US"] if market == Market.US else ["0700.HK"]

    def load_bars(self, market, symbol, start, end):
        # Return empty for simplicity
        return []


class MockFeatureGenerator:
    """Mock feature generator for testing."""

    def compute(self, bars, timestamp):
        return None


class TestToolExecution:
    def setup_method(self):
        self.snapshot = PortfolioSnapshot(
            timestamp="2026-01-07 10:00",
            cash=50000.0,
            positions={},
            total_nav=100000.0,
            market_exposure={},
            fx_rates={"USD": 1.0, "HKD": 7.80, "CNY": 7.25},
        )

        self.tool_system = ToolSystem(
            MockDataProvider(),
            MockFeatureGenerator(),
            lambda: self.snapshot,
        )

    def test_query_fx(self):
        """query_fx should return FX rates."""
        result = self.tool_system.execute_tool("query_fx", {}, "2026-01-07 10:00")
        assert "USD/HKD" in result
        assert "USD/CNY" in result

    def test_query_fx_with_balances(self):
        """query_fx with include_cash_balances should show cash."""
        result = self.tool_system.execute_tool(
            "query_fx", {"include_cash_balances": True}, "2026-01-07 10:00",
        )
        assert "Cash" in result

    def test_query_market_overview(self):
        """query_market_overview should return market info."""
        result = self.tool_system.execute_tool(
            "query_market_overview", {"markets": ["US", "HK"]}, "2026-01-07 10:00",
        )
        assert "US" in result
        assert "HK" in result

    def test_query_position_not_found(self):
        """query_position for non-existent position should return error."""
        result = self.tool_system.execute_tool(
            "query_position", {"symbol": "AAPL.US"}, "2026-01-07 10:00",
        )
        assert "No position" in result

    def test_query_position_found(self):
        """query_position for existing position should return details."""
        self.snapshot.positions["US:AAPL"] = Position(
            symbol="AAPL", market=Market.US, quantity=100,
            avg_cost=150.0, current_price=155.0,
        )
        result = self.tool_system.execute_tool(
            "query_position", {"symbol": "AAPL"}, "2026-01-07 10:00",
        )
        assert "AAPL" in result
        assert "100" in result

    def test_screen_universe_empty(self):
        """screen_universe with no data should return empty results."""
        result = self.tool_system.execute_tool(
            "screen_universe",
            {"market": "US", "bucket": "trend_leaders", "limit": 10},
            "2026-01-07 10:00",
        )
        assert "0 results" in result or "SCREEN" in result

    def test_query_asset_no_data(self):
        """query_asset with no data should return error."""
        result = self.tool_system.execute_tool(
            "query_asset",
            {"symbol": "AAPL.US", "fields": ["quote"]},
            "2026-01-07 10:00",
        )
        assert "No data" in result or "Error" in result

    def test_query_history_no_data(self):
        """query_history with no data should return error."""
        result = self.tool_system.execute_tool(
            "query_history",
            {"symbol": "AAPL.US", "lookback_bars": 48, "bar_size": "5m"},
            "2026-01-07 10:00",
        )
        assert "No data" in result or "Error" in result

    def test_unknown_tool(self):
        """Unknown tool should return error message."""
        result = self.tool_system.execute_tool(
            "unknown_tool", {}, "2026-01-07 10:00",
        )
        assert "Unknown tool" in result

    def test_query_futures_reserved(self):
        """query_futures_contract should return reserved message."""
        result = self.tool_system.execute_tool(
            "query_futures_contract", {"continuous_symbol": "ES"}, "2026-01-07 10:00",
        )
        assert "not available" in result


# ============================================================
# Test: Tool Descriptions
# ============================================================

class TestToolDescriptions:
    def setup_method(self):
        self.tool_system = ToolSystem(
            MockDataProvider(),
            MockFeatureGenerator(),
            lambda: None,
        )

    def test_get_tool_descriptions(self):
        """get_tool_descriptions should return all 8 tools."""
        descriptions = self.tool_system.get_tool_descriptions()
        assert len(descriptions) == 8

    def test_descriptions_have_correct_format(self):
        """All descriptions should have correct format."""
        descriptions = self.tool_system.get_tool_descriptions()
        for desc in descriptions:
            assert "type" in desc
            assert desc["type"] == "function"
            assert "function" in desc
            func = desc["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
