"""
AI Framework OpenAPI Tool Server

Milestone: M3 - User-Level OpenWebUI Connection
Version: 0.3.0

Purpose
-------
Expose one read-only OpenAPI tool that calls:

    ai_framework.healthcare_datasets.get_healthcare_dataset_catalog()

This version adds CORS support so a user-level OpenWebUI tool connection
can call the server directly from the browser.

Usage
-----
1. Direct framework self-test:

       python ai_framework_openapi_server_v0_3_catalog.py --self-test

2. Start the OpenAPI server:

       python ai_framework_openapi_server_v0_3_catalog.py

OpenWebUI user-level connection
-------------------------------
API Base URL:

    http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
SERVER_VERSION = "0.3.0"


app = FastAPI(
    title="AI Framework Tool Server",
    description=(
        "Read-only OpenAPI tools for accessing selected capabilities "
        "from the user's AI framework."
    ),
    version=SERVER_VERSION,
)


# A user-level OpenWebUI tool connection is called directly by the browser.
# These origins allow the locally hosted OpenWebUI interface to access this
# local FastAPI server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _dataframe_to_json_records(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-safe records."""

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


def _build_healthcare_dataset_catalog_response() -> dict[str, Any]:
    """Call the AI framework and create a JSON-safe catalog response."""

    try:
        import ai_framework.healthcare_datasets as hd

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


@app.get(
    "/get_healthcare_dataset_catalog",
    operation_id="get_healthcare_dataset_catalog",
    summary="Get Healthcare Dataset Catalog",
    description=(
        "Return the curated healthcare datasets available in the user's "
        "AI framework. Use this tool when the user asks which healthcare "
        "datasets are available."
    ),
)
def get_healthcare_dataset_catalog() -> dict[str, Any]:
    """Return the AI framework healthcare dataset catalog."""

    try:
        return _build_healthcare_dataset_catalog_response()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.get(
    "/health",
    include_in_schema=False,
)
def health_check() -> dict[str, str]:
    """Return a lightweight server health response."""

    return {
        "status": "ok",
        "server": "AI Framework OpenAPI Tool Server",
        "version": SERVER_VERSION,
    }


def run_self_test() -> None:
    """Call the framework directly and print the returned catalog."""

    result = _build_healthcare_dataset_catalog_response()
    print(json.dumps(result, indent=2, ensure_ascii=False))


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the AI Framework OpenAPI tool server."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Call get_healthcare_dataset_catalog directly, print the result, "
            "and exit without starting the OpenAPI server."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the self-test or start the OpenAPI server."""

    args = parse_arguments()

    if args.self_test:
        run_self_test()
        return

    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
    )


if __name__ == "__main__":
    main()
