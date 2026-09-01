"""Per-chat in-memory state for standardized dataset setup and workflow progress."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from prepare_bundles_workflow import (
    EXTERNAL_VALIDATION_MODE,
    PrepareBundlesWorkflowState,
)
from tabular_workspace import TabularWorkspace
from target_setup_workflow import PredictionTargetWorkflowState


STANDARDIZED_DATASET_WORKFLOW = "standardized_dataset_setup"


@dataclass
class MLWorkflowState:
    """Structured progress for the optional, interruptible ML-preparation flow."""

    active_workflow: str | None = None
    workflow_status: str = "not_started"
    current_step: str | None = None
    pending_decision: str | None = None
    completed_steps: list[str] = field(default_factory=list)
    target_proposal: str | None = None
    target_confirmed: bool = False
    task_type_confirmed: bool = False
    positive_class_confirmed: bool = False
    final_setup_confirmed: bool = False
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class MLProjectState:
    """The local-only dataset objects and workflow decisions for one chat."""

    source_file_name: str | None = None
    source_metadata: dict[str, Any] | None = None
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
    workflow: MLWorkflowState = field(default_factory=MLWorkflowState)
    target_setup: PredictionTargetWorkflowState = field(
        default_factory=PredictionTargetWorkflowState
    )
    prepare_bundles: PrepareBundlesWorkflowState = field(
        default_factory=PrepareBundlesWorkflowState
    )
    external_validation_df: pd.DataFrame | None = None
    train_bundle: dict[str, Any] | None = None
    validation_bundle: dict[str, Any] | None = None
    prep_meta: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_workspace(cls, workspace: TabularWorkspace) -> "MLProjectState":
        """Start a new project state from a just-loaded active dataset."""

        return cls(
            source_file_name=workspace.original_file_name,
            source_metadata=(
                dict(workspace.dataset_metadata)
                if workspace.dataset_metadata is not None
                else None
            ),
            df=workspace.dataframe.copy(deep=True),
        )

    def start_workflow(self, target_proposal: str | None) -> None:
        """Start or resume the structured setup workflow without choosing values."""

        if self.workflow.workflow_status == "cancelled":
            self._clear_configuration()
            self.workflow = MLWorkflowState()

        self.workflow.active_workflow = STANDARDIZED_DATASET_WORKFLOW
        if self.target_col is not None:
            self.workflow.target_confirmed = True
        if self.task_type == "binary_classification":
            self.workflow.task_type_confirmed = True
        if self.positive_class_value is not None:
            self.workflow.positive_class_confirmed = True
        if self.setup_status == "completed":
            self.workflow.final_setup_confirmed = True
        if self.target_col is None:
            self.workflow.target_proposal = target_proposal

        self._refresh_workflow_progress()

    def select_target(self, target_col: str, target_values: list[Any]) -> bool:
        """Store a confirmed target and invalidate every target-dependent decision."""

        target_changed = self.target_col != target_col
        self.target_col = target_col
        self.target_values = list(target_values)
        self.setup_error = None
        if target_changed:
            self.task_type = None
            self._clear_from_positive_class()

        if self.workflow.active_workflow == STANDARDIZED_DATASET_WORKFLOW:
            self.workflow.target_proposal = target_col
            self.workflow.target_confirmed = True
            if target_changed:
                self.workflow.task_type_confirmed = False
                self.workflow.positive_class_confirmed = False
                self.workflow.final_setup_confirmed = False
            self._refresh_workflow_progress()
        else:
            self._touch()
        return target_changed

    def confirm_task_type(self, task_type: str) -> None:
        """Store a confirmed task type and invalidate class-dependent setup objects."""

        task_changed = self.task_type != task_type
        self.task_type = task_type
        self.setup_error = None
        if task_changed:
            self._clear_from_positive_class()

        if self.workflow.active_workflow == STANDARDIZED_DATASET_WORKFLOW:
            self.workflow.task_type_confirmed = True
            if task_changed:
                self.workflow.positive_class_confirmed = False
                self.workflow.final_setup_confirmed = False
            self._refresh_workflow_progress()
        else:
            self._touch()

    def select_positive_class(self, positive_value: Any, negative_value: Any) -> None:
        """Store a confirmed positive class and invalidate the derived setup."""

        self.positive_class_value = positive_value
        self.negative_class_value = negative_value
        self.target_mapping = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.metadata = None
        self.setup_status = "positive_class_selected"
        self.setup_error = None
        self._clear_prepare_bundles_state()

        if self.workflow.active_workflow == STANDARDIZED_DATASET_WORKFLOW:
            self.workflow.positive_class_confirmed = True
            self.workflow.final_setup_confirmed = False
            self._refresh_workflow_progress()
        else:
            self._touch()

    def confirm_final_setup(self) -> None:
        """Record a human confirmation before the standardized setup is built."""

        self.workflow.final_setup_confirmed = True
        self._refresh_workflow_progress()

    def apply_setup(self, setup: dict[str, Any]) -> None:
        """Apply a validated standardized dataset setup dictionary."""

        self._clear_prepare_bundles_state()
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

        self.workflow.active_workflow = STANDARDIZED_DATASET_WORKFLOW
        self.workflow.target_confirmed = True
        self.workflow.task_type_confirmed = True
        self.workflow.positive_class_confirmed = True
        self.workflow.final_setup_confirmed = True
        self._refresh_workflow_progress()

    def accepts_external_validation_upload(self) -> bool:
        """Return whether the next supported tabular upload is external validation."""

        return (
            self.setup_status == "completed"
            and self.prepare_bundles.validation_mode == EXTERNAL_VALIDATION_MODE
            and self.prepare_bundles.status
            in {
                "awaiting_external_data",
                "awaiting_configuration",
            }
        )

    def attach_external_validation_workspace(self, workspace: TabularWorkspace) -> None:
        """Keep an uploaded validation table separate from the active training table."""

        self.external_validation_df = workspace.dataframe.copy(deep=True)
        self.prepare_bundles.external_validation_file_name = workspace.original_file_name
        self.prepare_bundles.external_target_col = None
        self.prepare_bundles.validation_kwargs = None
        self.prepare_bundles.resolved_prepare_bundles_config = None
        self.prepare_bundles.reviewed_prepare_bundles_config = None
        self.prepare_bundles.step_1_review_status = "not_reviewed"
        self.prepare_bundles.step_1_review_fingerprint = None
        self.prepare_bundles.configuration_confirmed = False
        self.prepare_bundles.status = "awaiting_configuration"
        self.prepare_bundles.last_error = None
        self._touch()

    def cancel_workflow(self) -> list[str]:
        """Cancel ML preparation while retaining the active uploaded dataset."""

        cleared = [
            name
            for name, value in (
                ("target column", self.target_col),
                ("task type", self.task_type),
                ("positive and negative classes", self.positive_class_value),
                ("target mapping", self.target_mapping),
                ("standardized setup", self.X),
            )
            if value is not None
        ]
        self._clear_configuration()
        self.workflow = MLWorkflowState(
            workflow_status="cancelled",
            last_updated=datetime.now(UTC),
        )
        self._touch()
        return cleared

    def record_error(self, message: str) -> None:
        """Keep a non-sensitive local setup error available for status tools."""

        self.setup_error = message
        self.setup_status = "error"
        if self.workflow.active_workflow == STANDARDIZED_DATASET_WORKFLOW:
            self.workflow.workflow_status = "error"
        self._touch()

    def refresh_workflow_progress(self) -> None:
        """Refresh the current workflow step without selecting or confirming anything."""

        self._refresh_workflow_progress()

    def _refresh_workflow_progress(self) -> None:
        workflow = self.workflow
        if workflow.active_workflow != STANDARDIZED_DATASET_WORKFLOW:
            self._touch()
            return

        completed_steps: list[str] = []
        if workflow.target_confirmed:
            completed_steps.append("confirm_target")
        if workflow.task_type_confirmed:
            completed_steps.append("confirm_task_type")
        if workflow.positive_class_confirmed:
            completed_steps.append("confirm_positive_class")
        if workflow.final_setup_confirmed:
            completed_steps.append("review_setup")

        if self.setup_status == "completed":
            workflow.workflow_status = "completed"
            workflow.current_step = "completed"
            workflow.pending_decision = None
            workflow.completed_steps = [*completed_steps, "build_setup"]
        elif not workflow.target_confirmed:
            workflow.workflow_status = "waiting_for_user"
            workflow.current_step = "confirm_target"
            workflow.pending_decision = "Step 1 of 4 — Target column: confirm the target column."
            workflow.completed_steps = completed_steps
        elif not workflow.task_type_confirmed:
            workflow.workflow_status = "waiting_for_user"
            workflow.current_step = "confirm_task_type"
            workflow.pending_decision = (
                "Step 2 of 4 — Task type: confirm whether this is binary classification."
            )
            workflow.completed_steps = completed_steps
        elif not workflow.positive_class_confirmed:
            workflow.workflow_status = "waiting_for_user"
            workflow.current_step = "confirm_positive_class"
            workflow.pending_decision = (
                "Step 3 of 4 — Positive class: choose the positive target value."
            )
            workflow.completed_steps = completed_steps
        elif not workflow.final_setup_confirmed:
            workflow.workflow_status = "waiting_for_user"
            workflow.current_step = "review_setup"
            workflow.pending_decision = (
                "Step 4 of 4 — Review and create the initial dataset setup: confirm creation."
            )
            workflow.completed_steps = completed_steps
        else:
            workflow.workflow_status = "ready_to_build"
            workflow.current_step = "build_setup"
            workflow.pending_decision = (
                "Step 4 of 4 — Review and create the initial dataset setup: create and store it."
            )
            workflow.completed_steps = completed_steps

        self._touch()

    def _clear_configuration(self) -> None:
        self.target_col = None
        self.target_values = None
        self.task_type = None
        self._clear_from_positive_class()
        self.setup_error = None
        self.target_setup = PredictionTargetWorkflowState()
        self._clear_prepare_bundles_state()

    def _clear_from_positive_class(self) -> None:
        self.positive_class_value = None
        self.negative_class_value = None
        self.target_mapping = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.metadata = None
        self.setup_status = None
        self._clear_prepare_bundles_state()

    def _clear_prepare_bundles_state(self) -> None:
        self.prepare_bundles = PrepareBundlesWorkflowState()
        self.external_validation_df = None
        self.train_bundle = None
        self.validation_bundle = None
        self.prep_meta = None

    def _touch(self) -> None:
        now = datetime.now(UTC)
        self.updated_at = now
        self.workflow.last_updated = now


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
        """Replace all project/workflow state when a new active dataset is loaded."""

        state = MLProjectState.from_workspace(workspace)
        self._states[chainlit_session_id] = state
        return state

    def remove_session(self, chainlit_session_id: str) -> None:
        self._states.pop(chainlit_session_id, None)


ml_project_states = MLProjectStateManager()
