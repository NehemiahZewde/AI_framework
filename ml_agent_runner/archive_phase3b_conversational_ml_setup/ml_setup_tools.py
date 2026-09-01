"""Controlled Agents SDK tools for the first local binary dataset setup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents import RunContextWrapper, function_tool

from dataset_setup import (
    build_standardized_dataset_setup,
    display_target_value,
    get_unique_non_null_target_values,
    match_target_value,
)
from ml_project_state import MLProjectState
from tabular_workspace import MLAgentContext


@function_tool
def get_current_ml_setup(
    context: RunContextWrapper[MLAgentContext],
) -> dict[str, object]:
    """Return a compact, non-tabular summary of this chat's local ML setup state."""

    workspace, state, error = _get_workspace_and_state(context)
    if error:
        return {"local_table_loaded": False, "message": error}

    return _state_summary(workspace, state)


@function_tool
def set_target_column(
    context: RunContextWrapper[MLAgentContext],
    target_col: str,
) -> dict[str, object]:
    """Validate and store a user-chosen target column; never infer one automatically."""

    workspace, state, error = _get_workspace_and_state(context)
    if error:
        return {"ok": False, "message": error}
    if not isinstance(target_col, str) or not target_col.strip():
        return {"ok": False, "message": "Provide a non-empty target column name."}
    if state.df is None or target_col not in state.df.columns:
        return {
            "ok": False,
            "message": f"Column {target_col!r} is not present in the active local table.",
            "available_columns": workspace.column_names,
        }

    target_values = get_unique_non_null_target_values(state.df, target_col)
    target_changed = state.select_target(target_col, target_values)
    binary_compatible = len(target_values) == 2
    return {
        "ok": True,
        "target_col": target_col,
        "target_values": _display_values(target_values),
        "non_null_unique_value_count": len(target_values),
        "binary_classification_compatible": binary_compatible,
        "downstream_setup_was_reset": target_changed,
        "next_step": (
            "Ask the user which displayed target value should be the positive class."
            if binary_compatible
            else "This target is not eligible for the current binary-only setup because it does not have exactly two non-null values."
        ),
    }


@function_tool
def set_positive_class(
    context: RunContextWrapper[MLAgentContext],
    positive_class_value: str,
) -> dict[str, object]:
    """Resolve a user-provided string and store the selected positive class."""

    _, state, error = _get_workspace_and_state(context)
    if error:
        return {"ok": False, "message": error}
    if state.target_col is None or state.target_values is None:
        return {
            "ok": False,
            "message": "Choose a target column before choosing a positive class.",
        }
    if len(state.target_values) != 2:
        return {
            "ok": False,
            "message": "The current binary-only setup requires exactly two non-null target values.",
            "target_values": _display_values(state.target_values),
        }

    try:
        positive_value = match_target_value(positive_class_value, state.target_values)
    except ValueError as exc:
        return {
            "ok": False,
            "message": str(exc),
            "target_values": _display_values(state.target_values),
        }

    negative_values = [
        value for value in state.target_values if not _values_equal(value, positive_value)
    ]
    if len(negative_values) != 1:
        return {
            "ok": False,
            "message": "The negative class could not be determined unambiguously.",
        }

    state.select_positive_class(positive_value, negative_values[0])
    return {
        "ok": True,
        "target_col": state.target_col,
        "positive_class_value": display_target_value(positive_value),
        "negative_class_value": display_target_value(negative_values[0]),
        "task_type": state.task_type,
        "next_step": "The standardized binary dataset setup can now be built.",
    }


@function_tool
def build_current_standardized_dataset_setup(
    context: RunContextWrapper[MLAgentContext],
) -> dict[str, object]:
    """Build df, X, y, feature names, metadata, and a float target mapping locally."""

    workspace, state, error = _get_workspace_and_state(context)
    if error:
        return {"ok": False, "message": error}
    if state.df is None:
        return {"ok": False, "message": "No local pandas table is available for setup."}
    if state.target_col is None:
        return {"ok": False, "message": "Choose a target column before building setup."}
    if state.target_values is None or len(state.target_values) != 2:
        return {
            "ok": False,
            "message": "The current binary-only setup requires exactly two non-null target values.",
        }
    if state.positive_class_value is None:
        return {"ok": False, "message": "Choose the positive class before building setup."}

    file_name = workspace.original_file_name
    dataset_name = Path(file_name).stem or "uploaded_dataset"
    try:
        setup = build_standardized_dataset_setup(
            df=state.df,
            target_col=state.target_col,
            positive_class_value=state.positive_class_value,
            dataset_name=dataset_name,
            display_name=file_name,
            source="user_uploaded_file",
            task_type="binary_classification",
        )
    except ValueError as exc:
        state.record_error(str(exc))
        return {"ok": False, "message": str(exc)}

    state.apply_setup(setup)
    return {"ok": True, **_state_summary(workspace, state)}


def _get_workspace_and_state(
    context: RunContextWrapper[MLAgentContext],
) -> tuple[Any, MLProjectState | None, str | None]:
    workspace = context.context.tabular_workspace
    state = context.context.ml_project_state
    if workspace is None:
        return None, None, "Attach a supported tabular file before starting local ML setup."
    if state is None:
        return workspace, None, "Local ML setup state is unavailable for this chat."
    if state.df is None:
        return workspace, state, "The active local table is unavailable in ML setup state."
    return workspace, state, None


def _state_summary(workspace: Any, state: MLProjectState) -> dict[str, object]:
    df_shape = list(state.df.shape) if state.df is not None else None
    x_shape = list(state.X.shape) if state.X is not None else None
    y_length = len(state.y) if state.y is not None else None
    return {
        "local_table_loaded": True,
        "source_file_name": state.source_file_name or workspace.original_file_name,
        "target_col": state.target_col,
        "target_values": _display_values(state.target_values or []),
        "task_type": state.task_type,
        "positive_class_value": _display_optional_value(state.positive_class_value),
        "negative_class_value": _display_optional_value(state.negative_class_value),
        "target_mapping": _mapping_entries(state.target_mapping),
        "df_shape": df_shape,
        "X_shape": x_shape,
        "y_length": y_length,
        "feature_names": state.feature_names,
        "metadata": state.metadata,
        "setup_status": state.setup_status,
        "setup_error": state.setup_error,
    }


def _display_values(values: list[Any]) -> list[str]:
    return [display_target_value(value) for value in values]


def _display_optional_value(value: Any) -> str | None:
    return display_target_value(value) if value is not None else None


def _mapping_entries(mapping: dict[Any, float] | None) -> list[dict[str, object]] | None:
    if mapping is None:
        return None
    return [
        {"original_value": display_target_value(original_value), "encoded_value": encoded_value}
        for original_value, encoded_value in mapping.items()
    ]


def _values_equal(left: Any, right: Any) -> bool:
    try:
        comparison = left == right
    except Exception:
        return False
    try:
        return bool(comparison)
    except (TypeError, ValueError):
        return False
