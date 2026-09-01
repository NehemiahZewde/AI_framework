"""Per-chat, in-memory pandas workspace reserved for future ML operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd


@dataclass
class TabularWorkspace:
    """One locally loaded table and compact metadata for a Chainlit chat."""

    original_file_name: str
    file_extension: str
    content_type: str | None
    dataframe: pd.DataFrame
    row_count: int
    column_count: int
    column_names: list[str]
    sheet_names: list[str] = field(default_factory=list)
    active_sheet_name: str | None = None
    loaded_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    local_load_status: str = "loaded"
    local_load_error: str | None = None

    def summary(self) -> dict[str, object]:
        """Expose local workspace facts without returning the DataFrame itself."""

        return {
            "local_table_loaded": True,
            "original_filename": self.original_file_name,
            "file_type": self.file_extension,
            "active_worksheet": self.active_sheet_name,
            "available_worksheet_names": self.sheet_names,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "column_names": self.column_names,
            "local_load_status": self.local_load_status,
            "local_load_error": self.local_load_error,
        }


@dataclass
class MLAgentContext:
    """Runtime-only dependencies for local workspace and setup tools."""

    tabular_workspace: TabularWorkspace | None = None
    ml_project_state: Any | None = None


class TabularWorkspaceManager:
    """Keep one active local pandas table for each Chainlit chat session."""

    def __init__(self) -> None:
        self._workspaces: dict[str, TabularWorkspace] = {}

    def get(self, chainlit_session_id: str) -> TabularWorkspace | None:
        return self._workspaces.get(chainlit_session_id)

    def set(self, chainlit_session_id: str, workspace: TabularWorkspace) -> None:
        self._workspaces[chainlit_session_id] = workspace

    def remove_session(self, chainlit_session_id: str) -> None:
        self._workspaces.pop(chainlit_session_id, None)


tabular_workspaces = TabularWorkspaceManager()
