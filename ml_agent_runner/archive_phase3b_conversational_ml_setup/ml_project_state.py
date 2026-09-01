"""Per-chat in-memory state for controlled standardized dataset setup."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from tabular_workspace import TabularWorkspace


@dataclass
class MLProjectState:
    """The local-only state required for the first binary setup workflow."""

    source_file_name: str | None = None
    df: pd.DataFrame | None = None
    X: pd.DataFrame | None = None
    y: pd.Series | None = None
    feature_names: list[str] | None = None
    metadata: dict[str, Any] | None = None
    target_col: str | None = None
    target_values: list[Any] | None = None
    task_type: str | None = None
    positive_class_value: Any | None = None
    negative_class_value: Any | None = None
    target_mapping: dict[Any, float] | None = None
    setup_status: str | None = None
    setup_error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_workspace(cls, workspace: TabularWorkspace) -> "MLProjectState":
        """Start a new project state from a just-loaded local table."""

        return cls(
            source_file_name=workspace.original_file_name,
            df=workspace.dataframe.copy(deep=True),
        )

    def select_target(self, target_col: str, target_values: list[Any]) -> bool:
        """Store a target and clear every downstream setup decision when it changes."""

        target_changed = self.target_col != target_col
        self.target_col = target_col
        self.target_values = list(target_values)
        self.setup_error = None
        if target_changed:
            self.task_type = None
            self._clear_from_positive_class()
        self.updated_at = datetime.now(UTC)
        return target_changed

    def select_positive_class(self, positive_value: Any, negative_value: Any) -> None:
        """Store a user-selected positive class and invalidate the derived setup."""

        self.positive_class_value = positive_value
        self.negative_class_value = negative_value
        self.task_type = "binary_classification"
        self.target_mapping = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.metadata = None
        self.setup_status = "positive_class_selected"
        self.setup_error = None
        self.updated_at = datetime.now(UTC)

    def apply_setup(self, setup: dict[str, Any]) -> None:
        """Apply a validated setup returned by the pure dataset helper."""

        self.df = setup["df"]
        self.X = setup["X"]
        self.y = setup["y"]
        self.feature_names = setup["feature_names"]
        self.metadata = setup["metadata"]
        self.target_col = setup["target_col"]
        self.positive_class_value = setup["positive_class_value"]
        self.negative_class_value = setup["negative_class_value"]
        self.target_mapping = setup["target_mapping"]
        self.task_type = "binary_classification"
        self.setup_status = "completed"
        self.setup_error = None
        self.updated_at = datetime.now(UTC)

    def record_error(self, message: str) -> None:
        """Keep a non-sensitive local setup error available for tool inspection."""

        self.setup_error = message
        self.setup_status = "error"
        self.updated_at = datetime.now(UTC)

    def _clear_from_positive_class(self) -> None:
        self.positive_class_value = None
        self.negative_class_value = None
        self.target_mapping = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.metadata = None
        self.setup_status = "target_selected"


class MLProjectStateManager:
    """Keep one non-persistent ML project state for each Chainlit chat."""

    def __init__(self) -> None:
        self._states: dict[str, MLProjectState] = {}

    def get_or_create(
        self,
        chainlit_session_id: str,
        workspace: TabularWorkspace | None,
    ) -> MLProjectState:
        state = self._states.get(chainlit_session_id)
        if state is None:
            state = (
                MLProjectState.from_workspace(workspace)
                if workspace is not None
                else MLProjectState()
            )
            self._states[chainlit_session_id] = state
        return state

    def reset_for_workspace(
        self,
        chainlit_session_id: str,
        workspace: TabularWorkspace,
    ) -> MLProjectState:
        """Replace the project state when a new valid local table replaces the old one."""

        state = MLProjectState.from_workspace(workspace)
        self._states[chainlit_session_id] = state
        return state

    def remove_session(self, chainlit_session_id: str) -> None:
        self._states.pop(chainlit_session_id, None)


ml_project_states = MLProjectStateManager()
