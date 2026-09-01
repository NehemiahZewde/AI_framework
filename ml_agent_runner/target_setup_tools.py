"""Deterministic tools for the condensed prediction-target conversation."""

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
from tabular_workspace import MLAgentContext
from target_setup_workflow import (
    EVIDENCE_SOURCES,
    PredictionTargetWorkflowState,
    compact_prediction_target_status,
    create_prediction_target_proposal,
)


@function_tool
def start_prediction_target_setup(
    context: RunContextWrapper[MLAgentContext],
) -> dict[str, object]:
    """Propose one evidence-aware prediction-target setup or ask about genuine ambiguity."""

    state, error = _get_state(context)
    if error:
        return _plain_error(error)
    if state.setup_status == "completed" and state.target_mapping is not None:
        _hydrate_completed_target_status(state)
        return compact_prediction_target_status(state)
    try:
        create_prediction_target_proposal(state)
    except ValueError as exc:
        return _error_response(state, str(exc))
    return compact_prediction_target_status(state)


@function_tool
def revise_prediction_target_proposal(
    context: RunContextWrapper[MLAgentContext],
    target_col: str,
    positive_class_value: str,
    negative_class_description: str,
    positive_class_description: str,
    evidence_source: str,
    evidence_reason: str,
) -> dict[str, object]:
    """Apply a target choice or documented evidence and return one revised proposal."""

    state, error = _get_state(context)
    if error:
        return _plain_error(error)
    try:
        create_prediction_target_proposal(
            state,
            selected_target_col=target_col,
            target_selected_by_user=True,
            explicit_positive_value=positive_class_value,
            negative_class_description=negative_class_description,
            positive_class_description=positive_class_description,
            evidence_source=_normalize_source(evidence_source),
            evidence_reason=evidence_reason,
        )
    except ValueError as exc:
        return _error_response(state, str(exc))
    return compact_prediction_target_status(state)


@function_tool
def confirm_prediction_target_setup(
    context: RunContextWrapper[MLAgentContext],
    target_col: str,
    positive_class_value: str,
    negative_class_description: str,
    positive_class_description: str,
    class_description_source: str,
) -> dict[str, object]:
    """Confirm or correct the combined proposal and automatically create dataset objects."""

    state, error = _get_state(context)
    if error:
        return _plain_error(error)
    if state.df is None:
        return _error_response(state, "Attach a tabular dataset first.")
    if target_col not in state.df.columns:
        return _error_response(
            state,
            f"Column {target_col!r} is not present in the active dataset.",
        )

    target_values = get_unique_non_null_target_values(state.df, target_col)
    if len(target_values) != 2:
        create_prediction_target_proposal(
            state,
            selected_target_col=target_col,
            target_selected_by_user=True,
        )
        return compact_prediction_target_status(state)
    if not positive_class_value.strip():
        return _error_response(
            state,
            "Specify which of the two target values should be the positive outcome.",
        )
    try:
        positive_value = match_target_value(positive_class_value, target_values)
    except ValueError as exc:
        return _error_response(state, str(exc))
    negative_value = next(
        value for value in target_values if not _values_equal(value, positive_value)
    )

    proposal = state.target_setup
    proposal_changed = (
        proposal.proposed_target_column != target_col
        or proposal.proposed_positive_class is None
        or not _values_equal(proposal.proposed_positive_class, positive_value)
    )
    descriptions = dict(proposal.class_descriptions)
    if negative_class_description.strip():
        if descriptions.get(negative_value) != negative_class_description.strip():
            proposal_changed = True
        descriptions[negative_value] = negative_class_description.strip()
    if positive_class_description.strip():
        if descriptions.get(positive_value) != positive_class_description.strip():
            proposal_changed = True
        descriptions[positive_value] = positive_class_description.strip()

    source = _normalize_source(class_description_source)
    if proposal_changed:
        source = "user_statement"
    elif source == "unknown":
        source = proposal.class_description_source

    file_name = state.source_file_name or "uploaded_dataset.csv"
    try:
        setup = build_standardized_dataset_setup(
            df=state.df,
            target_col=target_col,
            positive_class_value=positive_value,
            dataset_name=Path(file_name).stem or "uploaded_dataset",
            display_name=file_name,
            source="user_uploaded_file",
            task_type="binary_classification",
        )
    except ValueError as exc:
        return _error_response(state, str(exc))

    setup["metadata"]["target_setup_evidence"] = {
        "target_candidate_reason": (
            "The user selected this column as the outcome to predict."
            if proposal.proposed_target_column != target_col
            else proposal.target_candidate_reason
        ),
        "target_candidate_source": (
            "user_statement"
            if proposal.proposed_target_column != target_col
            else proposal.target_candidate_source
        ),
        "class_descriptions": dict(descriptions),
        "class_description_source": source,
        "positive_class_reason": (
            "The user confirmed or corrected the positive outcome."
            if proposal_changed
            else proposal.positive_class_reason
        ),
        "positive_class_source": (
            "user_statement" if proposal_changed else proposal.positive_class_source
        ),
    }
    state.apply_setup(setup)
    state.target_setup = PredictionTargetWorkflowState(
        status="complete",
        proposed_target_column=target_col,
        target_candidate_reason=setup["metadata"]["target_setup_evidence"][
            "target_candidate_reason"
        ],
        target_candidate_confidence="high",
        target_candidate_source=setup["metadata"]["target_setup_evidence"][
            "target_candidate_source"
        ],
        target_values=list(target_values),
        class_descriptions=descriptions,
        class_description_source=source,
        proposed_positive_class=positive_value,
        positive_class_reason=setup["metadata"]["target_setup_evidence"][
            "positive_class_reason"
        ],
        positive_class_confidence="high",
        positive_class_source=setup["metadata"]["target_setup_evidence"][
            "positive_class_source"
        ],
        candidate_columns=list(proposal.candidate_columns),
    )
    return compact_prediction_target_status(state)


@function_tool
def get_prediction_target_status(
    context: RunContextWrapper[MLAgentContext],
) -> dict[str, object]:
    """Return the pending proposal, its evidence source, or compact completion status."""

    state, error = _get_state(context)
    if error:
        return _plain_error(error)
    if state.setup_status == "completed" and state.target_setup.status != "complete":
        _hydrate_completed_target_status(state)
    return compact_prediction_target_status(state)


def _get_state(
    context: RunContextWrapper[MLAgentContext],
) -> tuple[Any | None, str | None]:
    state = context.context.ml_project_state
    if state is None:
        return None, "Prediction-target state is unavailable for this chat."
    return state, None


def _error_response(state: Any, message: str) -> dict[str, object]:
    state.target_setup.last_error = message
    response = compact_prediction_target_status(state)
    response.update({"ok": False, "message": message})
    return response


def _plain_error(message: str) -> dict[str, object]:
    return {
        "ok": False,
        "workflow_stage": "prediction_target",
        "target_setup_status": "error",
        "message": message,
    }


def _normalize_source(source: str) -> str:
    normalized = source.strip().casefold().replace(" ", "_")
    return normalized if normalized in EVIDENCE_SOURCES else "unknown"


def _hydrate_completed_target_status(state: Any) -> None:
    if state.target_col is None or state.target_mapping is None:
        return
    positive_values = [
        value for value, encoded in state.target_mapping.items() if encoded == 1.0
    ]
    if len(positive_values) != 1:
        return
    evidence = (state.metadata or {}).get("target_setup_evidence", {})
    state.target_setup = PredictionTargetWorkflowState(
        status="complete",
        proposed_target_column=state.target_col,
        target_candidate_reason=evidence.get("target_candidate_reason"),
        target_candidate_confidence="high",
        target_candidate_source=evidence.get("target_candidate_source", "unknown"),
        target_values=list(state.target_values or state.target_mapping.keys()),
        class_descriptions=dict(evidence.get("class_descriptions") or {}),
        class_description_source=evidence.get("class_description_source", "unknown"),
        proposed_positive_class=positive_values[0],
        positive_class_reason=evidence.get("positive_class_reason"),
        positive_class_confidence="high",
        positive_class_source=evidence.get("positive_class_source", "unknown"),
    )


def _values_equal(left: Any, right: Any) -> bool:
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False
