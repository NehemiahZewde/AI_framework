"""
Tests for AI Framework MCP Server v0.4.0.

This test uses the official MCP Python SDK's in-memory Client.
It verifies the MCP tool without Docker, HTTP, or OpenWebUI.
"""

from __future__ import annotations

import pytest
from mcp import Client

from ai_framework_mcp_server_v0_4_catalog import mcp


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client():
    async with Client(mcp, raise_exceptions=True) as connected_client:
        yield connected_client


@pytest.mark.anyio
async def test_get_healthcare_dataset_catalog(client: Client) -> None:
    result = await client.call_tool(
        "get_healthcare_dataset_catalog",
        {},
    )

    assert result.is_error is False
    assert result.structured_content is not None

    catalog = result.structured_content

    assert catalog["dataset_count"] > 0
    assert isinstance(catalog["columns"], list)
    assert isinstance(catalog["datasets"], list)
    assert len(catalog["datasets"]) == catalog["dataset_count"]
