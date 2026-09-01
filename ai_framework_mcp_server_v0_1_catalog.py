"""
AI Framework MCP Server

Milestone: M1 - Healthcare Dataset Catalog Tool
Version: 0.1.0

Purpose
-------
Expose one read-only MCP tool that calls:

    ai_framework.healthcare_datasets.get_healthcare_dataset_catalog()

The server uses MCP Streamable HTTP so OpenWebUI can connect to it.

Usage
-----
1. Direct framework self-test:

       python ai_framework_mcp_server_v0_1_catalog.py --self-test

2. Start the MCP server:

       python ai_framework_mcp_server_v0_1_catalog.py

MCP endpoint
------------
Local machine:
    http://127.0.0.1:8000/mcp

OpenWebUI running in Docker:
    http://host.docker.internal:8000/mcp
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

import ai_framework.healthcare_datasets as hd


SERVER_NAME = "AI Framework Dataset Catalog"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
SERVER_PATH = "/mcp"


transport_security = TransportSecuritySettings(
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


mcp = FastMCP(
    name=SERVER_NAME,
    instructions=(
        "Provide read-only access to the curated healthcare dataset catalog "
        "from the user's AI framework."
    ),
    host=SERVER_HOST,
    port=SERVER_PORT,
    streamable_http_path=SERVER_PATH,
    json_response=True,
    stateless_http=True,
    transport_security=transport_security,
)


def _dataframe_to_json_records(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-safe records.

    Using pandas JSON serialization converts values such as NumPy scalars,
    timestamps, and missing values into forms that MCP can return safely.
    """

    records = json.loads(
        dataframe.to_json(
            orient="records",
            date_format="iso",
        )
    )

    if not isinstance(records, list):
        raise TypeError(
            "Expected DataFrame JSON serialization to produce a list."
        )

    return records


@mcp.tool(
    name="get_healthcare_dataset_catalog",
    title="Get Healthcare Dataset Catalog",
    description=(
        "Return the curated healthcare datasets available in the user's "
        "AI framework. Use this tool when the user asks which healthcare "
        "datasets are available."
    ),
)
def get_healthcare_dataset_catalog() -> dict[str, Any]:
    """Return the AI framework healthcare dataset catalog.

    Returns
    -------
    dict[str, Any]
        A JSON-safe response containing the catalog row count, column names,
        and one record for each available dataset.

    Raises
    ------
    RuntimeError
        If the AI framework cannot create the dataset catalog.
    TypeError
        If the framework returns an unexpected object instead of a DataFrame.
    """

    try:
        catalog_df = hd.get_healthcare_dataset_catalog()
    except Exception as exc:
        raise RuntimeError(
            "The AI framework could not create the healthcare dataset "
            f"catalog. Original error: {exc}"
        ) from exc

    if not isinstance(catalog_df, pd.DataFrame):
        raise TypeError(
            "hd.get_healthcare_dataset_catalog() returned "
            f"{type(catalog_df).__name__}; expected pandas.DataFrame."
        )

    return {
        "status": "success",
        "dataset_count": int(len(catalog_df)),
        "columns": [str(column) for column in catalog_df.columns],
        "datasets": _dataframe_to_json_records(catalog_df),
    }


def run_self_test() -> None:
    """Call the tool directly and print the returned catalog."""

    result = get_healthcare_dataset_catalog()
    print(json.dumps(result, indent=2, ensure_ascii=False))


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the AI Framework dataset catalog MCP server."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Call get_healthcare_dataset_catalog directly, print the result, "
            "and exit without starting the MCP server."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the self-test or start the MCP Streamable HTTP server."""

    args = parse_arguments()

    if args.self_test:
        run_self_test()
        return

    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
