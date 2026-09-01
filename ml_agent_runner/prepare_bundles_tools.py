"""Deterministic Agents SDK tools for raw train/validation bundle preparation."""

from __future__ import annotations

from contextlib import redirect_stdout
from datetime import UTC, datetime
import io
from typing import Any

from agents import RunContextWrapper, function_tool

from dataset_setup import get_unique_non_null_target_values, match_target_value
from prepare_bundles_workflow import (
    DEFAULT_PROGRESS_KWARGS,
    DEFAULT_SETTING_SOURCES,
    DEFAULT_SHOW_PROGRESS,
    DEFAULT_SPLIT_KWARGS,
    DEFAULT_TARGET_NAME,
    EXTERNAL_VALIDATION_MODE,
    INTERNAL_VALIDATION_MODE,
    NO_VALIDATION_MODE,
    build_resolved_config,
    compact_prepare_bundles_status,
    fingerprint_step_1_config,
    normalize_validation_mode,
    record_step_1_review,
    validate_internal_configuration,
    validate_progress_configuration,
    validate_target_name,
)
from tabular_workspace import MLAgentContext


@function_tool
def start_prepare_bundles_stage(
    context: RunContextWrapper[MLAgentContext],
) -> dict[str, object]:
    """Start or resume raw train/final-validation bundle preparation."""

    state, error = _get_project_state(context)
    if error:
        return {"ok": False, "workflow_stage": "prepare_bundles", "message": error}
    if state.setup_status != "completed":
        return _error_response(
            state,
            "Complete the initial dataset setup before preparing raw bundles.",
        )
    if state.prepare_bundles.complete:
        return compact_prepare_bundles_status(state)
    if state.prepare_bundles.status in {"not_started", "failed"}:
        state.prepare_bundles.reset_configuration()
        _apply_validation_mode_transition(state, INTERNAL_VALIDATION_MODE)
    return compact_prepare_bundles_status(state)


@function_tool
def set_prepare_bundles_validation_mode(
    context: RunContextWrapper[MLAgentContext],
    validation_mode: str,
) -> dict[str, object]:
    """Persist the user's internal-split or external-validation choice; never acknowledge it without this tool."""

    state, error = _get_project_state(context)
    if error:
        return {"ok": False, "workflow_stage": "prepare_bundles", "message": error}
    if state.setup_status != "completed":
        return _error_response(state, "Complete the initial dataset setup first.")
    if state.prepare_bundles.status == "not_started":
        return _error_response(state, "Start the raw-bundle preparation stage first.")

    resolved_mode = normalize_validation_mode(validation_mode)
    if resolved_mode is None:
        return _error_response(
            state,
            "Choose an internal split, a separate validation dataset, or no separate final-validation set.",
        )

    workflow = state.prepare_bundles
    validation_mode_before = workflow.validation_mode
    status_before = workflow.status
    try:
        _apply_validation_mode_transition(state, resolved_mode)
    except Exception as exc:
        _log_prepare_state_transition(
            context,
            state,
            tool_name="set_prepare_bundles_validation_mode",
            validation_mode_before=validation_mode_before,
            status_before=status_before,
        )
        return _error_response(
            state,
            f"The validation mode could not be saved: {exc}",
            prior_configuration_preserved=True,
        )

    _log_prepare_state_transition(
        context,
        state,
        tool_name="set_prepare_bundles_validation_mode",
        validation_mode_before=validation_mode_before,
        status_before=status_before,
    )
    return compact_prepare_bundles_status(state)


@function_tool
def configure_internal_prepare_bundles(
    context: RunContextWrapper[MLAgentContext],
    target_name: str,
    validation_size: float,
    random_state: int,
    random_state_is_none: bool,
    stratify: bool,
    progress_enabled: bool,
    show_output_shapes: bool,
    return_progress_log: bool,
    show_progress: bool,
) -> dict[str, object]:
    """Apply defaults or updates as one fully resolved internal-split configuration."""

    state, error = _get_project_state(context)
    if error:
        return {"ok": False, "workflow_stage": "prepare_bundles", "message": error}
    response = _apply_internal_configuration(
        context,
        state,
        tool_name="configure_internal_prepare_bundles",
        target_name=target_name,
        validation_size=validation_size,
        random_state=None if random_state_is_none else random_state,
        stratify=stratify,
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
        return_progress_log=return_progress_log,
        show_progress=show_progress,
        source_updates={
            "target_name": "User selected",
            "validation_mode": "User selected",
            "validation_size": "User selected",
            "random_state": "User selected",
            "stratify": "User selected",
            "progress_enabled": "User selected",
            "show_output_shapes": "User selected",
            "return_progress_log": "User selected",
            "show_progress": "User selected",
        },
    )
    return response


@function_tool
def update_internal_prepare_bundles(
    context: RunContextWrapper[MLAgentContext],
    validation_size: str,
    random_state: str,
    stratify: str,
    target_name: str,
    progress_enabled: str,
    show_output_shapes: str,
    return_progress_log: str,
    show_progress: str,
) -> dict[str, object]:
    """Apply only supplied internal-setting changes; empty strings retain current values."""

    state, error = _get_project_state(context)
    if error:
        return {"ok": False, "workflow_stage": "prepare_bundles", "message": error}
    workflow = state.prepare_bundles
    current_split = dict(workflow.split_kwargs or DEFAULT_SPLIT_KWARGS)
    current_progress = dict(workflow.progress_kwargs or DEFAULT_PROGRESS_KWARGS)
    try:
        resolved_validation_size = _updated_float(
            validation_size,
            current=float(current_split["validation_size"]),
            setting_name="validation_size",
        )
        resolved_random_state = _updated_random_state(
            random_state,
            current=current_split.get("random_state"),
        )
        resolved_stratify = _updated_bool(
            stratify,
            current=bool(current_split.get("stratify", True)),
            setting_name="stratify",
        )
        resolved_target_name = target_name.strip() or workflow.target_name
        resolved_progress_enabled = _updated_bool(
            progress_enabled,
            current=bool(current_progress.get("enabled", True)),
            setting_name="progress_enabled",
        )
        resolved_show_output_shapes = _updated_bool(
            show_output_shapes,
            current=bool(current_progress.get("show_output_shapes", True)),
            setting_name="show_output_shapes",
        )
        resolved_return_progress_log = _updated_bool(
            return_progress_log,
            current=bool(current_progress.get("return_progress_log", True)),
            setting_name="return_progress_log",
        )
        resolved_show_progress = _updated_bool(
            show_progress,
            current=workflow.show_progress,
            setting_name="show_progress",
        )
    except ValueError as exc:
        return _error_response(state, str(exc), prior_configuration_preserved=True)

    response = _apply_internal_configuration(
        context,
        state,
        tool_name="update_internal_prepare_bundles",
        target_name=resolved_target_name,
        validation_size=resolved_validation_size,
        random_state=resolved_random_state,
        stratify=resolved_stratify,
        progress_enabled=resolved_progress_enabled,
        show_output_shapes=resolved_show_output_shapes,
        return_progress_log=resolved_return_progress_log,
        show_progress=resolved_show_progress,
        source_updates=_internal_source_updates(
            workflow,
            validation_size=validation_size,
            random_state=random_state,
            stratify=stratify,
            target_name=target_name,
            progress_enabled=progress_enabled,
            show_output_shapes=show_output_shapes,
            return_progress_log=return_progress_log,
            show_progress=show_progress,
        ),
    )
    if response.get("ok") is not False and any(
        value.strip()
        for value in (
            target_name,
            progress_enabled,
            show_output_shapes,
            return_progress_log,
            show_progress,
        )
    ):
        response["show_advanced_settings"] = True
    return response


@function_tool
def configure_external_prepare_bundles(
    context: RunContextWrapper[MLAgentContext],
    external_target_col: str,
    target_name: str,
    progress_enabled: bool,
    show_output_shapes: bool,
    return_progress_log: bool,
    show_progress: bool,
) -> dict[str, object]:
    """Resolve uploaded external validation X/y and grouped progress settings."""

    state, error = _get_project_state(context)
    if error:
        return {"ok": False, "workflow_stage": "prepare_bundles", "message": error}
    normalized_mode = normalize_validation_mode(
        state.prepare_bundles.validation_mode or ""
    )
    if normalized_mode != EXTERNAL_VALIDATION_MODE:
        return _error_response(state, "Select external validation before configuring it.")
    state.prepare_bundles.validation_mode = EXTERNAL_VALIDATION_MODE
    external_df = state.external_validation_df
    if external_df is None:
        state.prepare_bundles.status = "awaiting_external_data"
        return _error_response(state, "Attach the external validation dataset first.")
    if not isinstance(external_target_col, str) or external_target_col not in external_df.columns:
        return _error_response(
            state,
            f"External target column {external_target_col!r} is not present in the uploaded validation dataset.",
        )
    if state.X is None or state.feature_names is None or state.target_mapping is None:
        return _error_response(state, "The initial dataset setup objects are unavailable.")

    external_X = external_df.drop(columns=[external_target_col]).copy()
    external_y = external_df[external_target_col].copy()
    try:
        resolved_target_name = validate_target_name(
            target_name,
            [*state.feature_names, *list(external_X.columns)],
        )
        progress_kwargs, resolved_show_progress = validate_progress_configuration(
            enabled=progress_enabled,
            show_output_shapes=show_output_shapes,
            return_progress_log=return_progress_log,
            show_progress=show_progress,
        )
        _validate_external_target_values(external_y, state.target_mapping)
        common_features = [name for name in state.feature_names if name in external_X.columns]
        if not common_features:
            raise ValueError(
                "The external validation dataset has no feature columns in common with the training dataset."
            )
    except ValueError as exc:
        return _error_response(state, str(exc), prior_configuration_preserved=True)

    workflow = state.prepare_bundles
    workflow.target_name = resolved_target_name
    workflow.split_kwargs = None
    workflow.validation_kwargs = {
        "X": external_X,
        "y": external_y,
        "feature_names": list(external_X.columns),
    }
    workflow.progress_kwargs = progress_kwargs
    workflow.show_progress = resolved_show_progress
    workflow.external_target_col = external_target_col
    workflow.setting_sources.update(
        {
            "target_name": (
                "User selected"
                if resolved_target_name != DEFAULT_TARGET_NAME
                else "Framework default"
            ),
            "progress_enabled": _source_for_resolved_default(
                progress_enabled, DEFAULT_PROGRESS_KWARGS["enabled"]
            ),
            "show_output_shapes": _source_for_resolved_default(
                show_output_shapes, DEFAULT_PROGRESS_KWARGS["show_output_shapes"]
            ),
            "return_progress_log": _source_for_resolved_default(
                return_progress_log, DEFAULT_PROGRESS_KWARGS["return_progress_log"]
            ),
            "show_progress": _source_for_resolved_default(
                show_progress, DEFAULT_SHOW_PROGRESS
            ),
        }
    )
    workflow.configuration_confirmed = True
    workflow.status = "awaiting_final_confirmation"
    workflow.last_error = None
    record_step_1_review(state)
    return compact_prepare_bundles_status(state)


@function_tool
def get_prepare_bundles_status(
    context: RunContextWrapper[MLAgentContext],
) -> dict[str, object]:
    """Return compact raw-bundle workflow, configuration, and result status."""

    state, error = _get_project_state(context)
    if error:
        return {"ok": False, "workflow_stage": "prepare_bundles", "message": error}
    return compact_prepare_bundles_status(state)


@function_tool
def show_prepare_bundles_advanced_settings(
    context: RunContextWrapper[MLAgentContext],
) -> dict[str, object]:
    """Show detailed operational settings without changing workflow state."""

    state, error = _get_project_state(context)
    if error:
        return {"ok": False, "workflow_stage": "prepare_bundles", "message": error}
    if state.prepare_bundles.status == "not_started":
        return _error_response(
            state,
            "Start training and validation preparation before viewing its advanced settings.",
        )
    response = compact_prepare_bundles_status(state)
    response["show_advanced_settings"] = True
    return response


@function_tool
def inspect_prepare_bundles_function_call(
    context: RunContextWrapper[MLAgentContext],
) -> dict[str, object]:
    """Return the exact resolved framework call only when the user explicitly asks."""

    state, error = _get_project_state(context)
    if error:
        return {"ok": False, "workflow_stage": "prepare_bundles", "message": error}
    status = compact_prepare_bundles_status(state)
    resolved_call = status.get("resolved_function_call")
    if resolved_call is None:
        return {
            "ok": False,
            "workflow_stage": "prepare_bundles",
            "message": "Resolve the training and validation settings before showing the function call.",
        }
    return {
        "ok": True,
        "workflow_stage": "prepare_bundles",
        "resolved_function_call": resolved_call,
    }


@function_tool
def show_step_1_execution_log(
    context: RunContextWrapper[MLAgentContext],
) -> dict[str, object]:
    """Return the saved Step 1 framework log without executing Step 1 again."""

    state, error = _get_project_state(context)
    if error:
        return {"ok": False, "workflow_stage": "prepare_bundles", "message": error}
    execution_log = state.prepare_bundles.step_1_execution_log
    if not execution_log:
        return _error_response(state, "No Step 1 execution log is available yet.")
    response = compact_prepare_bundles_status(state)
    response.update(
        {
            "show_saved_execution_log": True,
            "step_1_execution_log": execution_log,
        }
    )
    return response


@function_tool
def inspect_step_1_results(
    context: RunContextWrapper[MLAgentContext],
) -> dict[str, object]:
    """Return compact structured Step 1 facts without full objects or log parsing."""

    state, error = _get_project_state(context)
    if error:
        return {"ok": False, "workflow_stage": "prepare_bundles", "message": error}
    return compact_prepare_bundles_status(state)


@function_tool
def run_prepare_train_validation_bundles(
    context: RunContextWrapper[MLAgentContext],
    allow_rerun: bool,
) -> dict[str, object]:
    """Run the real framework function after explicit final user confirmation."""

    state, error = _get_project_state(context)
    if error:
        return {"ok": False, "workflow_stage": "prepare_bundles", "message": error}
    workflow = state.prepare_bundles
    existing_results = state.train_bundle is not None and state.prep_meta is not None
    if workflow.status == "running":
        response = _error_response(state, "Step 1 is already running.")
        response["duplicate_execution"] = True
        return response
    if workflow.status == "failed" and not allow_rerun:
        response = _error_response(
            state,
            "The previous Step 1 attempt failed. Explicitly request a rerun to try again.",
            prior_configuration_preserved=existing_results,
        )
        response["duplicate_execution"] = True
        return response
    if existing_results and not allow_rerun:
        response = _error_response(
            state,
            "Step 1 has already completed. Explicitly request a rerun before replacing its outputs.",
            prior_configuration_preserved=True,
        )
        response["duplicate_execution"] = True
        return response
    if workflow.status not in {"awaiting_final_confirmation", "complete", "failed"}:
        return _error_response(
            state,
            "Resolve and review the complete Step 1 configuration before running it.",
        )
    reviewed_config = workflow.reviewed_prepare_bundles_config
    reviewed_fingerprint = workflow.step_1_review_fingerprint
    if (
        not workflow.configuration_confirmed
        or reviewed_config is None
        or reviewed_fingerprint is None
        or workflow.step_1_review_status not in {"awaiting_confirmation", "executed"}
    ):
        return _error_response(state, "The Step 1 configuration has not been reviewed.")
    if any(value is None for value in (state.X, state.y, state.feature_names, state.metadata, state.target_mapping)):
        return _error_response(state, "The initial dataset setup is incomplete.")
    if set(state.target_mapping.values()) != {0.0, 1.0} or not all(
        isinstance(value, float) for value in state.target_mapping.values()
    ):
        return _error_response(
            state,
            "The target mapping must use float encodings 0.0 and 1.0 before bundle preparation.",
        )

    current_config = build_resolved_config(state)
    current_fingerprint = fingerprint_step_1_config(current_config)
    if current_fingerprint != reviewed_fingerprint:
        workflow.status = "awaiting_final_confirmation"
        workflow.configuration_confirmed = True
        record_step_1_review(state)
        response = compact_prepare_bundles_status(state)
        response["execution_blocked_for_updated_review"] = True
        return response

    workflow.status = "running"
    workflow.last_error = None
    workflow.step_1_execution_error = None
    stdout_buffer = io.StringIO()
    try:
        import ai_framework.ml_data_preprocessing as mdp

        with redirect_stdout(stdout_buffer):
            train_bundle, validation_bundle, prep_meta = mdp.prepare_train_validation_bundles(
                X=state.X,
                y=state.y,
                feature_names=state.feature_names,
                dataset_metadata=state.metadata,
                target_name=reviewed_config["target_name"],
                target_mapping=reviewed_config["target_mapping"],
                split_kwargs=reviewed_config["split_kwargs"],
                validation_kwargs=reviewed_config["validation_kwargs"],
                progress_kwargs=reviewed_config["progress_kwargs"],
                show_progress=reviewed_config["show_progress"],
            )
    except Exception as exc:
        execution_log = _select_framework_execution_log(
            stdout_buffer.getvalue(),
            None,
        )
        workflow.last_error = str(exc)
        workflow.step_1_execution_error = str(exc)
        workflow.step_1_execution_log = execution_log
        workflow.step_1_review_status = "awaiting_confirmation"
        workflow.status = "failed"
        response = compact_prepare_bundles_status(state)
        response.update(
            {
                "ok": False,
                "execution_failed": True,
                "message": "Step 1 did not complete.",
                "prior_configuration_preserved": existing_results,
                "step_1_execution_log": execution_log,
            }
        )
        return response

    execution_log = _select_framework_execution_log(
        stdout_buffer.getvalue(),
        prep_meta.get("progress_log"),
    )
    state.train_bundle = train_bundle
    state.validation_bundle = validation_bundle
    state.prep_meta = prep_meta
    workflow.status = "complete"
    workflow.last_error = None
    workflow.step_1_execution_error = None
    workflow.step_1_execution_log = execution_log
    workflow.step_1_executed_at = datetime.now(UTC)
    workflow.step_1_executed_review_version = workflow.step_1_review_version
    workflow.step_1_executed_review_fingerprint = reviewed_fingerprint
    workflow.executed_prepare_bundles_config = dict(reviewed_config)
    workflow.step_1_review_status = "executed"
    workflow.run_count += 1
    workflow.successful_prepare_bundles_config = dict(reviewed_config)
    response = compact_prepare_bundles_status(state)
    response["step_1_execution_log"] = execution_log
    return response


def _apply_internal_configuration(
    context: RunContextWrapper[MLAgentContext],
    state: Any,
    *,
    tool_name: str,
    target_name: str,
    validation_size: float,
    random_state: int | None,
    stratify: bool,
    progress_enabled: bool,
    show_output_shapes: bool,
    return_progress_log: bool,
    show_progress: bool,
    source_updates: dict[str, str],
) -> dict[str, object]:
    workflow = state.prepare_bundles
    validation_mode_before = workflow.validation_mode
    status_before = workflow.status
    if normalize_validation_mode(workflow.validation_mode or "") != INTERNAL_VALIDATION_MODE:
        _log_prepare_state_transition(
            context,
            state,
            tool_name=tool_name,
            validation_mode_before=validation_mode_before,
            status_before=status_before,
        )
        return _error_response(
            state,
            "Select an internal validation split before changing split settings.",
        )
    if state.X is None or state.y is None or state.feature_names is None:
        return _error_response(state, "The confirmed dataset is unavailable.")

    try:
        resolved_target_name, split_kwargs = validate_internal_configuration(
            X=state.X,
            y=state.y,
            feature_names=state.feature_names,
            target_name=target_name,
            validation_size=validation_size,
            random_state=random_state,
            stratify=stratify,
        )
        progress_kwargs, resolved_show_progress = validate_progress_configuration(
            enabled=progress_enabled,
            show_output_shapes=show_output_shapes,
            return_progress_log=return_progress_log,
            show_progress=show_progress,
        )
    except ValueError as exc:
        return _error_response(state, str(exc), prior_configuration_preserved=True)

    workflow.validation_mode = INTERNAL_VALIDATION_MODE
    workflow.target_name = resolved_target_name
    workflow.split_kwargs = split_kwargs
    workflow.validation_kwargs = None
    workflow.progress_kwargs = progress_kwargs
    workflow.show_progress = resolved_show_progress
    workflow.setting_sources.update(source_updates)
    workflow.configuration_confirmed = True
    workflow.status = "awaiting_final_confirmation"
    workflow.last_error = None
    record_step_1_review(state)
    _log_prepare_state_transition(
        context,
        state,
        tool_name=tool_name,
        validation_mode_before=validation_mode_before,
        status_before=status_before,
    )
    return compact_prepare_bundles_status(state)


def _updated_float(value: str, *, current: float, setting_name: str) -> float:
    cleaned = value.strip()
    if not cleaned:
        return current
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1].strip()
    try:
        resolved = float(cleaned)
    except ValueError as exc:
        raise ValueError(f"{setting_name} must be numeric.") from exc
    return resolved / 100.0 if is_percent else resolved


def _updated_random_state(value: str, *, current: Any) -> int | None:
    cleaned = value.strip().casefold()
    if not cleaned:
        return current
    if cleaned in {"none", "no seed", "random"}:
        return None
    try:
        return int(cleaned)
    except ValueError as exc:
        raise ValueError("random_state must be an integer or None.") from exc


def _updated_bool(value: str, *, current: bool, setting_name: str) -> bool:
    cleaned = value.strip().casefold()
    if not cleaned:
        return current
    if cleaned in {"true", "yes", "enabled", "enable", "on", "1"}:
        return True
    if cleaned in {"false", "no", "disabled", "disable", "off", "0"}:
        return False
    raise ValueError(f"{setting_name} must be enabled or disabled.")


def _internal_source_updates(
    workflow: Any,
    *,
    validation_size: str,
    random_state: str,
    stratify: str,
    target_name: str,
    progress_enabled: str,
    show_output_shapes: str,
    return_progress_log: str,
    show_progress: str,
) -> dict[str, str]:
    current = workflow.setting_sources
    updates = {
        "validation_mode": (
            current.get("validation_mode")
            if current.get("validation_mode") == "User selected"
            else "Agent recommendation accepted by user"
        ),
        "target_name": (
            "User selected"
            if target_name.strip()
            else current.get("target_name", "Framework default")
        ),
    }
    primary_values = {
        "validation_size": validation_size,
        "random_state": random_state,
        "stratify": stratify,
    }
    for name, value in primary_values.items():
        if value.strip():
            updates[name] = "User selected"
        elif current.get(name) == "User selected":
            updates[name] = "User selected"
        else:
            updates[name] = "Agent recommendation accepted by user"
    advanced_values = {
        "progress_enabled": progress_enabled,
        "show_output_shapes": show_output_shapes,
        "return_progress_log": return_progress_log,
        "show_progress": show_progress,
    }
    for name, value in advanced_values.items():
        updates[name] = (
            "User selected"
            if value.strip()
            else current.get(name, "Framework default")
        )
    return updates


def _source_for_resolved_default(value: Any, default: Any) -> str:
    return "Framework default" if value == default else "User selected"


def _select_framework_execution_log(
    captured_stdout: str,
    progress_log: Any,
) -> str:
    captured = captured_stdout.strip()
    if captured:
        return captured
    return _format_structured_progress_log(progress_log)


def _format_structured_progress_log(progress_log: Any) -> str:
    if not isinstance(progress_log, list):
        return ""
    lines: list[str] = []
    labels = {"ok": "OK", "skipped": "SKIP", "fail": "FAIL"}
    for entry in progress_log:
        if not isinstance(entry, dict):
            continue
        label = labels.get(str(entry.get("status")), str(entry.get("status", "INFO")).upper())
        line = f"[{label}] {entry.get('step', 'Unnamed step')}"
        detail = entry.get("detail")
        if detail:
            line += f" -> {detail}"
        lines.append(line)
    return "\n".join(lines)


def _get_project_state(
    context: RunContextWrapper[MLAgentContext],
) -> tuple[Any | None, str | None]:
    state = context.context.ml_project_state
    if state is None:
        return None, "ML project state is unavailable for this chat."
    return state, None


def _apply_validation_mode_transition(state: Any, resolved_mode: str) -> None:
    """Apply one validation-mode transition and roll back if postconditions fail."""

    workflow = state.prepare_bundles
    workflow_snapshot = dict(workflow.__dict__)
    external_validation_snapshot = state.external_validation_df
    try:
        workflow.validation_mode = resolved_mode
        workflow.target_name = DEFAULT_TARGET_NAME
        workflow.progress_kwargs = dict(DEFAULT_PROGRESS_KWARGS)
        workflow.show_progress = DEFAULT_SHOW_PROGRESS
        workflow.resolved_prepare_bundles_config = None
        workflow.reviewed_prepare_bundles_config = None
        workflow.step_1_review_status = "not_reviewed"
        workflow.step_1_review_fingerprint = None
        workflow.setting_sources = dict(DEFAULT_SETTING_SOURCES)
        workflow.configuration_confirmed = False
        workflow.last_error = None

        if resolved_mode == INTERNAL_VALIDATION_MODE:
            state.external_validation_df = None
            workflow.external_validation_file_name = None
            workflow.external_target_col = None
            workflow.split_kwargs = dict(DEFAULT_SPLIT_KWARGS)
            workflow.validation_kwargs = None
            workflow.status = "awaiting_configuration"
            if not (
                workflow.validation_mode == "internal"
                and workflow.validation_kwargs is None
                and workflow.status == "awaiting_configuration"
            ):
                raise RuntimeError("internal validation postconditions were not satisfied")
        elif resolved_mode == EXTERNAL_VALIDATION_MODE:
            workflow.setting_sources["validation_mode"] = "User selected"
            workflow.split_kwargs = None
            workflow.validation_kwargs = None
            workflow.external_target_col = None
            workflow.status = (
                "awaiting_configuration"
                if state.external_validation_df is not None
                else "awaiting_external_data"
            )
        elif resolved_mode == NO_VALIDATION_MODE:
            workflow.setting_sources.update(
                {
                    "validation_mode": "User selected",
                    "validation_size": "Derived",
                    "random_state": "Framework default",
                    "stratify": "Derived",
                }
            )
            state.external_validation_df = None
            workflow.external_validation_file_name = None
            workflow.external_target_col = None
            workflow.split_kwargs = {
                "validation_size": 0.0,
                "random_state": DEFAULT_SPLIT_KWARGS["random_state"],
                "stratify": False,
            }
            workflow.validation_kwargs = None
            workflow.configuration_confirmed = True
            workflow.status = "awaiting_final_confirmation"
            record_step_1_review(state)
        else:
            raise ValueError(f"Unsupported validation mode {resolved_mode!r}.")
    except Exception:
        workflow.__dict__.clear()
        workflow.__dict__.update(workflow_snapshot)
        state.external_validation_df = external_validation_snapshot
        raise


def _log_prepare_state_transition(
    context: RunContextWrapper[MLAgentContext],
    state: Any,
    *,
    tool_name: str,
    validation_mode_before: Any,
    status_before: Any,
) -> None:
    """Log non-sensitive state identity and transition diagnostics."""

    workflow = state.prepare_bundles
    session_id = context.context.session_id or "unknown"
    print(
        "[ML_STATE] "
        f"session_id={session_id} "
        f"state_object_id={id(state)} "
        f"tool_name={tool_name} "
        f"validation_mode_before={_log_value(validation_mode_before)} "
        f"validation_mode_after={_log_value(workflow.validation_mode)} "
        f"status_before={_log_value(status_before)} "
        f"status_after={_log_value(workflow.status)}"
    )


def _log_value(value: Any) -> str:
    return "None" if value is None else str(value).replace(" ", "_")


def _error_response(
    state: Any,
    message: str,
    *,
    prior_configuration_preserved: bool = False,
) -> dict[str, object]:
    response = compact_prepare_bundles_status(state)
    response.update(
        {
            "ok": False,
            "message": message,
            "prior_configuration_preserved": prior_configuration_preserved,
        }
    )
    return response


def _validate_external_target_values(y: Any, target_mapping: dict[Any, float]) -> None:
    if bool(y.isna().any()):
        raise ValueError("The external validation target contains missing values.")
    target_values = get_unique_non_null_target_values(y.to_frame(name="target"), "target")
    for value in target_values:
        try:
            match_target_value(value, list(target_mapping.keys()))
        except ValueError as exc:
            raise ValueError(
                f"External target value {value!r} is not present in the confirmed target mapping."
            ) from exc

