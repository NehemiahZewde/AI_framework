"""Controlled Agents SDK tools for interruptible ML-preparation decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents import RunContextWrapper, function_tool

from dataset_setup import (
    build_standardized_dataset_setup,
    display_target_value,
    get_unique_non_null_target_values,
    inspect_target_candidates as inspect_target_candidate_summaries,
    match_target_value,
)
from ml_project_state import MLProjectState, STANDARDIZED_DATASET_WORKFLOW
from tabular_workspace import MLAgentContext


INITIAL_SETUP_DISPLAY_NAME = "Initial dataset setup"
INITIAL_SETUP_STEPS = (
    "Step 1 of 4 — Target column",
    "Step 2 of 4 — Task type",
    "Step 3 of 4 — Positive class",
    "Step 4 of 4 — Review and create the initial dataset setup",
)
INITIAL_SETUP_OBJECTS = (
    "df",
    "X",
    "y",
    "feature_names",
    "metadata",
    "target_mapping",
)
DEFERRED_ML_STAGES = (
    "train/validation split",
    "preprocessing",
    "feature selection",
    "model training",
)


@function_tool
def start_ml_preparation(
    context: RunContextWrapper[MLAgentContext],
) -> dict[str, object]:
    """Start or resume optional ML preparation without selecting any user decisions."""

    workspace, state, error = _get_workspace_and_state(context)
    if error:
        return {"ok": False, "message": error}

    candidates = _inspect_target_candidates(state)
    suggestion = _suggest_target_column(state, candidates)
    state.start_workflow(suggestion["column"] if suggestion else None)
    status = _workflow_status(workspace, state)
    status.update(
        {
            "ok": True,
            "workflow_display_name": INITIAL_SETUP_DISPLAY_NAME,
            "introduction": (
                "We'll begin with the initial dataset setup. First explain the four-step "
                "roadmap, what will be created, and which later ML stages will not run."
            ),
            "workflow_requirements": list(INITIAL_SETUP_STEPS),
            "objects_created_after_confirmation": list(INITIAL_SETUP_OBJECTS),
            "not_run_in_this_stage": list(DEFERRED_ML_STAGES),
            "response_order": "Present the roadmap first, then begin Step 1 of 4 — Target column.",
            "target_suggestion": suggestion,
            "target_candidates": candidates,
        }
    )
    return status


@function_tool
def inspect_target_candidates(
    context: RunContextWrapper[MLAgentContext],
) -> dict[str, object]:
    """Return compact plausible target-column candidates for the active dataset."""

    workspace, state, error = _get_workspace_and_state(context)
    if error:
        return {"ok": False, "message": error}

    candidates = _inspect_target_candidates(state)
    return {
        "ok": True,
        "active_dataset_filename": workspace.original_file_name,
        "target_candidates": candidates,
        "current_target_proposal": state.workflow.target_proposal,
        "message": (
            "Candidates are suggestions only. Confirm a target explicitly before changing ML setup state."
        ),
    }


@function_tool
def get_ml_preparation_status(
    context: RunContextWrapper[MLAgentContext],
) -> dict[str, object]:
    """Return compact workflow status without returning the active dataset rows."""

    workspace, state, error = _get_workspace_and_state(context)
    if error:
        return {"ml_preparation_started": False, "message": error}
    state.refresh_workflow_progress()
    return _workflow_status(workspace, state)


@function_tool
def advance_ml_preparation(
    context: RunContextWrapper[MLAgentContext],
) -> dict[str, object]:
    """Return the next required ML-preparation decision without inventing one."""

    workspace, state, error = _get_workspace_and_state(context)
    if error:
        return {"ok": False, "message": error}
    if state.workflow.active_workflow != STANDARDIZED_DATASET_WORKFLOW:
        return {
            "ok": True,
            "ml_preparation_started": False,
            "message": (
                "ML preparation has not started. Ask whether the user would like "
                "guidance for the active dataset."
            ),
        }

    state.refresh_workflow_progress()
    status = _workflow_status(workspace, state)
    if state.workflow.current_step == "confirm_target":
        status["guidance"] = (
            "Step 1 of 4 — Target column. Ask the user to confirm the suggested target "
            "or name another column. If resuming, restate this step before the question."
        )
    elif state.workflow.current_step == "confirm_task_type":
        status["guidance"] = (
            "Step 2 of 4 — Task type. List the two target values and ask the user to "
            "confirm binary classification."
        )
    elif state.workflow.current_step == "confirm_positive_class":
        status["guidance"] = (
            "Step 3 of 4 — Positive class. List the two target values and ask which "
            "represents the positive outcome. Explain positive -> 1.0 and other -> 0.0."
        )
    elif state.workflow.current_step == "review_setup":
        status["guidance"] = (
            "Step 4 of 4 — Review and create the initial dataset setup. Present the "
            "review and explain exactly which six session objects will be created."
        )
        status["review"] = _setup_review(state)
    elif state.workflow.current_step == "build_setup":
        status["guidance"] = "The user confirmed the review; create the initial dataset setup now."
        status["review"] = _setup_review(state)
    else:
        status["guidance"] = (
            "Initial dataset setup complete. No train/validation split or preprocessing has run."
        )
    return {"ok": True, **status}


@function_tool
def cancel_ml_preparation(
    context: RunContextWrapper[MLAgentContext],
) -> dict[str, object]:
    """Cancel ML preparation while preserving the active uploaded dataset and general chat."""

    workspace, state, error = _get_workspace_and_state(context)
    if error:
        return {"ok": False, "message": error}

    cleared = state.cancel_workflow()
    return {
        "ok": True,
        "active_dataset_filename": workspace.original_file_name,
        "workflow_status": state.workflow.workflow_status,
        "cleared_ml_state": cleared or ["no prior ML decisions"],
        "message": "ML preparation was cancelled. The uploaded dataset and normal conversation remain available.",
    }


@function_tool
def get_current_ml_setup(
    context: RunContextWrapper[MLAgentContext],
) -> dict[str, object]:
    """Return a compact, non-tabular summary of this chat's ML setup state."""

    workspace, state, error = _get_workspace_and_state(context)
    if error:
        return {"active_dataset_available": False, "message": error}

    return {**_state_summary(workspace, state), **_workflow_status(workspace, state)}


@function_tool
def get_standardized_dataset_setup(
    context: RunContextWrapper[MLAgentContext],
) -> dict[str, object]:
    """Return a compact summary of the initial dataset setup without table contents."""

    workspace, state, error = _get_workspace_and_state(context)
    if error:
        return {"ok": False, "message": error}

    return {
        "ok": state.setup_status == "completed",
        "message": (
            "Initial dataset setup complete."
            if state.setup_status == "completed"
            else "The initial dataset setup has not been created yet."
        ),
        **_state_summary(workspace, state),
        **_workflow_status(workspace, state),
    }


@function_tool
def set_target_column(
    context: RunContextWrapper[MLAgentContext],
    target_col: str,
) -> dict[str, object]:
    """Validate and store a user-confirmed target column; never infer one automatically."""

    workspace, state, error = _get_workspace_and_state(context)
    if error:
        return {"ok": False, "message": error}
    if not isinstance(target_col, str) or not target_col.strip():
        return {"ok": False, "message": "Provide a non-empty target column name."}
    if state.df is None or target_col not in state.df.columns:
        return {
            "ok": False,
            "message": f"Column {target_col!r} is not present in the active dataset.",
            "available_columns": workspace.column_names,
        }

    target_values = get_unique_non_null_target_values(state.df, target_col)
    target_changed = state.select_target(target_col, target_values)
    binary_compatible = len(target_values) == 2
    response: dict[str, object] = {
        "ok": True,
        "target_col": target_col,
        "target_values": _display_values(target_values),
        "non_null_unique_value_count": len(target_values),
        "binary_classification_compatible": binary_compatible,
        "downstream_setup_was_reset": target_changed,
    }
    if state.workflow.active_workflow == STANDARDIZED_DATASET_WORKFLOW:
        response.update(_workflow_status(workspace, state))
    response["next_step"] = (
        "Step 2 of 4 — Task type. List the two values and ask the user to confirm binary classification."
        if binary_compatible
        else "This target cannot continue through the current binary-only preparation workflow because it does not have exactly two non-null values."
    )
    return response


@function_tool
def set_task_type(
    context: RunContextWrapper[MLAgentContext],
    task_type: str,
) -> dict[str, object]:
    """Confirm binary classification for the selected target; reject unsupported tasks."""

    workspace, state, error = _get_workspace_and_state(context)
    if error:
        return {"ok": False, "message": error}
    if state.workflow.active_workflow != STANDARDIZED_DATASET_WORKFLOW:
        return {"ok": False, "message": "Start ML preparation before confirming the task type."}
    if state.target_col is None or not state.workflow.target_confirmed:
        return {"ok": False, "message": "Confirm a target column before confirming the task type."}
    if state.target_values is None or len(state.target_values) != 2:
        return {
            "ok": False,
            "message": "Binary classification requires exactly two non-null target values.",
            "target_values": _display_values(state.target_values or []),
        }
    if _normalize_task_type(task_type) != "binary_classification":
        return {
            "ok": False,
            "message": "Only binary classification is supported in this ML-preparation phase.",
        }

    state.confirm_task_type("binary_classification")
    return {
        "ok": True,
        "task_type": "binary_classification",
        **_workflow_status(workspace, state),
        "next_step": (
            "Step 3 of 4 — Positive class. Ask which displayed value represents the "
            "positive outcome and explain the 1.0/0.0 encoding."
        ),
    }


@function_tool
def set_positive_class(
    context: RunContextWrapper[MLAgentContext],
    positive_class_value: str,
) -> dict[str, object]:
    """Resolve a user-provided string and store the selected positive class."""

    workspace, state, error = _get_workspace_and_state(context)
    if error:
        return {"ok": False, "message": error}
    if state.workflow.active_workflow != STANDARDIZED_DATASET_WORKFLOW:
        return {
            "ok": False,
            "message": "Start ML preparation before choosing the positive class.",
            **_workflow_status(workspace, state),
        }
    if state.target_col is None or state.target_values is None:
        return {
            "ok": False,
            "message": "Confirm a target column before choosing a positive class.",
            **_workflow_status(workspace, state),
        }
    if not state.workflow.task_type_confirmed or state.task_type != "binary_classification":
        return {
            "ok": False,
            "message": "Confirm binary classification before choosing the positive class.",
            **_workflow_status(workspace, state),
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
        **_workflow_status(workspace, state),
        "review": _setup_review(state),
        "next_step": (
            "Step 4 of 4 — Review and create the initial dataset setup. Present the "
            "review and explain exactly what will be created before asking for confirmation."
        ),
    }


@function_tool
def confirm_final_standardized_dataset_setup(
    context: RunContextWrapper[MLAgentContext],
) -> dict[str, object]:
    """Confirm, build, and store the reviewed initial dataset setup atomically."""

    workspace, state, error = _get_workspace_and_state(context)
    if error:
        return {"ok": False, "message": error}
    if state.workflow.active_workflow != STANDARDIZED_DATASET_WORKFLOW:
        return {"ok": False, "message": "Start ML preparation before confirming the final setup."}
    if not state.workflow.positive_class_confirmed:
        return {"ok": False, "message": "Choose the positive class before confirming the final setup."}
    if state.setup_status == "completed":
        return {
            "ok": True,
            "message": "The initial dataset setup is already complete.",
            **_state_summary(workspace, state),
            **_workflow_status(workspace, state),
        }

    state.confirm_final_setup()
    return _build_setup_response(workspace, state)


@function_tool
def build_current_standardized_dataset_setup(
    context: RunContextWrapper[MLAgentContext],
) -> dict[str, object]:
    """Build df, X, y, feature names, metadata, and float target mapping locally."""

    workspace, state, error = _get_workspace_and_state(context)
    if error:
        return {"ok": False, "message": error}
    if state.workflow.active_workflow != STANDARDIZED_DATASET_WORKFLOW:
        return {"ok": False, "message": "Start ML preparation before creating the initial dataset setup."}
    if not state.workflow.final_setup_confirmed:
        return {
            "ok": False,
            "message": "Present the final review and obtain explicit confirmation before building setup.",
        }
    if state.df is None or state.target_col is None:
        return {"ok": False, "message": "Confirm a target column before building setup."}
    if state.target_values is None or len(state.target_values) != 2:
        return {
            "ok": False,
            "message": "The current binary-only setup requires exactly two non-null target values.",
        }
    if state.task_type != "binary_classification" or not state.workflow.task_type_confirmed:
        return {"ok": False, "message": "Confirm binary classification before building setup."}
    if state.positive_class_value is None or not state.workflow.positive_class_confirmed:
        return {"ok": False, "message": "Choose the positive class before building setup."}

    return _build_setup_response(workspace, state)


def _build_setup_response(workspace: Any, state: MLProjectState) -> dict[str, object]:
    """Build the setup after all confirmations and return its compact summary."""

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
    return {
        "ok": True,
        "completion_heading": "Initial dataset setup complete.",
        "created_and_stored_for_this_session": list(INITIAL_SETUP_OBJECTS),
        "not_run_yet": list(DEFERRED_ML_STAGES),
        "next_stage_started": False,
        "next_stage_message": (
            "The next data-preparation stage will not begin until the user requests it."
        ),
        **_state_summary(workspace, state),
        **_workflow_status(workspace, state),
    }


def _get_workspace_and_state(
    context: RunContextWrapper[MLAgentContext],
) -> tuple[Any, MLProjectState | None, str | None]:
    workspace = context.context.tabular_workspace
    state = context.context.ml_project_state
    if workspace is None:
        return None, None, "Attach a supported tabular file before starting ML preparation."
    if state is None:
        return workspace, None, "ML preparation state is unavailable for this chat."
    if state.df is None:
        return workspace, state, "The active dataset is unavailable for ML preparation."
    return workspace, state, None


def _workflow_status(workspace: Any, state: MLProjectState) -> dict[str, object]:
    workflow = state.workflow
    return {
        "ml_preparation_started": workflow.active_workflow == STANDARDIZED_DATASET_WORKFLOW,
        "active_dataset_filename": state.source_file_name or workspace.original_file_name,
        "active_workflow": workflow.active_workflow,
        "workflow_status": workflow.workflow_status,
        "current_step": workflow.current_step,
        "user_facing_step": _user_facing_step_label(workflow.current_step),
        "pending_decision": workflow.pending_decision,
        "completed_steps": workflow.completed_steps,
        "target_suggestion": workflow.target_proposal,
        "target_col": state.target_col,
        "target_values": _display_values(state.target_values or []),
        "task_type": state.task_type,
        "target_confirmed": workflow.target_confirmed,
        "task_type_confirmed": workflow.task_type_confirmed,
        "positive_class_confirmed": workflow.positive_class_confirmed,
        "final_setup_confirmed": workflow.final_setup_confirmed,
        "positive_class_value": _display_optional_value(state.positive_class_value),
        "negative_class_value": _display_optional_value(state.negative_class_value),
        "target_mapping": _mapping_entries(state.target_mapping),
        "standardized_setup_built": state.setup_status == "completed",
    }


def _state_summary(workspace: Any, state: MLProjectState) -> dict[str, object]:
    df_shape = list(state.df.shape) if state.df is not None else None
    x_shape = list(state.X.shape) if state.X is not None else None
    y_length = len(state.y) if state.y is not None else None
    return {
        "active_dataset_available": True,
        "source_file_name": state.source_file_name or workspace.original_file_name,
        "df_shape": df_shape,
        "X_shape": x_shape,
        "y_length": y_length,
        "feature_names": state.feature_names,
        "metadata": state.metadata,
        "setup_status": state.setup_status,
        "setup_error": state.setup_error,
        "prepare_bundles_status": state.prepare_bundles.status,
        "prepare_bundles_complete": state.prepare_bundles.complete,
        "train_bundle_available": state.train_bundle is not None,
        "validation_bundle_available": state.validation_bundle is not None,
        "prep_meta_available": state.prep_meta is not None,
    }


def _setup_review(state: MLProjectState) -> dict[str, object]:
    feature_count = len(state.df.columns) - 1 if state.df is not None else None
    return {
        "step_label": INITIAL_SETUP_STEPS[3],
        "dataset": state.source_file_name,
        "target": state.target_col,
        "task_type": "Binary classification",
        "target_values": _display_values(state.target_values or []),
        "negative_class": {
            "original_value": _display_optional_value(state.negative_class_value),
            "encoded_value": 0.0,
        },
        "positive_class": {
            "original_value": _display_optional_value(state.positive_class_value),
            "encoded_value": 1.0,
        },
        "feature_count": feature_count,
        "will_create": list(INITIAL_SETUP_OBJECTS),
        "will_not_run": list(DEFERRED_ML_STAGES),
        "confirmation_prompt": (
            "Should I now create and store the initial dataset setup for this session?"
        ),
    }


def _user_facing_step_label(current_step: str | None) -> str | None:
    return {
        "confirm_target": INITIAL_SETUP_STEPS[0],
        "confirm_task_type": INITIAL_SETUP_STEPS[1],
        "confirm_positive_class": INITIAL_SETUP_STEPS[2],
        "review_setup": INITIAL_SETUP_STEPS[3],
        "build_setup": INITIAL_SETUP_STEPS[3],
        "completed": "Initial dataset setup complete",
    }.get(current_step)


def _suggest_target_column(
    state: MLProjectState,
    candidates: list[dict[str, object]] | None = None,
) -> dict[str, str] | None:
    candidates = candidates if candidates is not None else _inspect_target_candidates(state)
    if not candidates:
        return None
    top_candidate = candidates[0]
    return {
        "column": str(top_candidate["column_name"]),
        "reason": str(top_candidate["reason"]),
    }


def _inspect_target_candidates(state: MLProjectState) -> list[dict[str, object]]:
    if state.df is None:
        return []
    return inspect_target_candidate_summaries(state.df)


def _normalize_task_type(task_type: str) -> str | None:
    normalized = task_type.strip().casefold().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    if normalized in {"binary classification", "binary classifier", "binary"}:
        return "binary_classification"
    return None


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
