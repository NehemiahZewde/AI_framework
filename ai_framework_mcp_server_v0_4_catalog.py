"""
AI Framework MCP Server

Milestone: M4 - Clinical Dataset Catalog
Version: 0.4.0

Purpose
-------
Expose exactly one read-only MCP tool:

    get_healthcare_dataset_catalog

The tool delegates directly to the existing AI framework function:

    ai_framework.healthcare_datasets.get_healthcare_dataset_catalog()

Architecture
------------
OpenWebUI
    -> Native MCP (Streamable HTTP)
    -> MCPServer
    -> ai_framework.healthcare_datasets.get_healthcare_dataset_catalog()

Tested target
-------------
MCP Python SDK 2.0.x

OpenWebUI endpoint
------------------
When this server is running on the Windows host and OpenWebUI is running
inside Docker, configure OpenWebUI to use:

    http://host.docker.internal:8000/mcp
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

import ai_framework.healthcare_datasets as hd


SERVER_NAME = "AI Framework Clinical Dataset Catalog"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
SERVER_PATH = "/mcp"


class HealthcareDatasetCatalogResult(BaseModel):
    """Structured result returned by the clinical dataset catalog tool."""

    dataset_count: int = Field(
        ge=0,
        description="Number of healthcare datasets in the AI framework catalog.",
    )
    columns: list[str] = Field(
        description="Column names present in the framework dataset catalog.",
    )
    datasets: list[dict[str, Any]] = Field(
        description="Complete set of dataset catalog records returned by the AI framework.",
    )


mcp = MCPServer(SERVER_NAME)


@mcp.tool(
    title="Get Healthcare Dataset Catalog",
    annotations=ToolAnnotations(
        read_only_hint=True,
        open_world_hint=False,
    ),
)
def get_healthcare_dataset_catalog() -> HealthcareDatasetCatalogResult:
    """Return all healthcare/clinical datasets available in the AI framework.

    Use this tool when the user asks what clinical datasets, healthcare datasets,
    or example datasets are available in the AI framework.

    This tool only lists the framework catalog. It does not load an individual
    dataset, preprocess data, train models, or modify any files.
    """

    catalog_df = hd.get_healthcare_dataset_catalog()

    if not isinstance(catalog_df, pd.DataFrame):
        raise TypeError(
            "ai_framework.healthcare_datasets.get_healthcare_dataset_catalog() "
            f"returned {type(catalog_df).__name__}; expected pandas.DataFrame."
        )

    records = json.loads(
        catalog_df.to_json(
            orient="records",
            date_format="iso",
        )
    )

    if not isinstance(records, list):
        raise TypeError(
            "Dataset catalog serialization did not produce a list of records."
        )

    return HealthcareDatasetCatalogResult(
        dataset_count=len(catalog_df),
        columns=[str(column) for column in catalog_df.columns],
        datasets=records,
    )


TRANSPORT_SECURITY = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[
        "127.0.0.1:*",
        "localhost:*",
        "host.docker.internal:*",
    ],
    allowed_origins=[
        "http://127.0.0.1:*",
        "http://localhost:*",
        "http://host.docker.internal:*",
    ],
)


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host=SERVER_HOST,
        port=SERVER_PORT,
        streamable_http_path=SERVER_PATH,
        json_response=True,
        transport_security=TRANSPORT_SECURITY,
    )
