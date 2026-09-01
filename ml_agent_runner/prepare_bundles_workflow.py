"""State, validation, review, and presentation for raw bundle preparation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
from numbers import Integral, Real
from typing import Any

from sklearn.model_selection import train_test_split

from dataset_setup import display_target_value


PREPARE_BUNDLES_STAGE_NAME = "Prepare the training and validation data"
INTERNAL_VALIDATION_MODE = "internal"
EXTERNAL_VALIDATION_MODE = "external"
NO_VALIDATION_MODE = "none"
PREPARE_BUNDLES_DIRECT_TOOL_NAMES = frozenset(
    {
        "start_prepare_bundles_stage",
        "set_prepare_bundles_validation_mode",
        "configure_internal_prepare_bundles",
        "update_internal_prepare_bundles",
        "configure_external_prepare_bundles",
        "get_prepare_bundles_status",
        "show_prepare_bundles_advanced_settings",
        "show_step_1_execution_log",
        "run_prepare_train_validation_bundles",
    }
)

DEFAULT_TARGET_NAME = "target"
DEFAULT_SPLIT_KWARGS: dict[str, Any] = {
    "validation_size": 0.20,
    "random_state": 42,
    "stratify": True,
}
DEFAULT_PROGRESS_KWARGS: dict[str, bool] = {
    "enabled": True,
    "show_output_shapes": True,
    "return_progress_log": True,
}
DEFAULT_SHOW_PROGRESS = True
DEFAULT_SETTING_SOURCES: dict[str, str] = {
    "target_name": "Framework default",
    "validation_mode": "Agent recommendation",
    "validation_size": "Agent recommendation",
    "random_state": "Agent recommendation",
    "stratify": "Agent recommendation",
    "progress_enabled": "Framework default",
    "show_output_shapes": "Framework default",
    "return_progress_log": "Framework default",
    "show_progress": "Framework default",
}


@dataclass
class PrepareBundlesWorkflowState:
    """In-memory configuration and progress for one raw-bundle stage."""

    status: str = "not_started"
    target_name: str = DEFAULT_TARGET_NAME
    validation_mode: str | None = None
    split_kwargs: dict[str, Any] | None = None
    validation_kwargs: dict[str, Any] | None = None
    progress_kwargs: dict[str, bool] = field(
        default_factory=lambda: dict(DEFAULT_PROGRESS_KWARGS)
    )
    show_progress: bool = DEFAULT_SHOW_PROGRESS
    resolved_prepare_bundles_config: dict[str, Any] | None = None
    reviewed_prepare_bundles_config: dict[str, Any] | None = None
    step_1_review_status: str = "not_reviewed"
    step_1_review_version: int = 0
    step_1_review_fingerprint: str | None = None
    setting_sources: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_SETTING_SOURCES)
    )
    step_1_execution_log: str | None = None
    step_1_execution_error: str | None = None
    step_1_executed_at: datetime | None = None
    step_1_executed_review_version: int | None = None
    step_1_executed_review_fingerprint: str | None = None
    executed_prepare_bundles_config: dict[str, Any] | None = None
    successful_prepare_bundles_config: dict[str, Any] | None = None
    configuration_confirmed: bool = False
    external_validation_file_name: str | None = None
    external_target_col: str | None = None
    last_error: str | None = None
    run_count: int = 0

    @property
    def complete(self) -> bool:
        return self.status == "complete"

    def reset_configuration(self, *, preserve_run_count: bool = True) -> None:
        run_count = self.run_count if preserve_run_count else 0
        self.status = "not_started"
        self.target_name = DEFAULT_TARGET_NAME
        self.validation_mode = None
        self.split_kwargs = None
        self.validation_kwargs = None
        self.progress_kwargs = dict(DEFAULT_PROGRESS_KWARGS)
        self.show_progress = DEFAULT_SHOW_PROGRESS
        self.resolved_prepare_bundles_config = None
        self.reviewed_prepare_bundles_config = None
        self.step_1_review_status = "not_reviewed"
        self.step_1_review_version = 0
        self.step_1_review_fingerprint = None
        self.setting_sources = dict(DEFAULT_SETTING_SOURCES)
        self.step_1_execution_log = None
        self.step_1_execution_error = None
        self.step_1_executed_at = None
        self.step_1_executed_review_version = None
        self.step_1_executed_review_fingerprint = None
        self.executed_prepare_bundles_config = None
        self.successful_prepare_bundles_config = None
        self.configuration_confirmed = False
        self.external_validation_file_name = None
        self.external_target_col = None
        self.last_error = None
        self.run_count = run_count


def normalize_validation_mode(value: str) -> str | None:
    normalized = " ".join(value.strip().casefold().replace("_", " ").split())
    if normalized in {"internal", "internal split", "split", "train validation split"}:
        return INTERNAL_VALIDATION_MODE
    if normalized in {
        "external",
        "external validation",
        "provided validation",
        "separate validation dataset",
    }:
        return EXTERNAL_VALIDATION_MODE
    if normalized in {
        "none",
        "no validation",
        "no validation data",
        "without validation",
        "continue without validation",
        "continue without validation data",
        "training only",
        "train only",
        "no separate final validation set",
        "no separate final-validation set",
        "use all current data",
        "use all current data without a separate final validation set",
        "use all current data without a separate final-validation set",
    }:
        return NO_VALIDATION_MODE
    return None


def validate_target_name(target_name: str, feature_names: Sequence[str]) -> str:
    if not isinstance(target_name, str) or not target_name.strip():
        raise ValueError("The internal target name must be a non-empty string.")
    resolved = target_name.strip()
    if resolved in feature_names:
        raise ValueError(
            f"Internal target name {resolved!r} conflicts with an existing feature column."
        )
    return resolved


def validate_internal_configuration(
    *,
    X: Any,
    y: Any,
    feature_names: Sequence[str],
    target_name: str,
    validation_size: float,
    random_state: int | None,
    stratify: bool,
) -> tuple[str, dict[str, Any]]:
    """Validate a complete internal-split configuration without changing state."""

    resolved_target_name = validate_target_name(target_name, feature_names)
    if isinstance(validation_size, bool) or not isinstance(validation_size, Real):
        raise ValueError("validation_size must be numeric.")
    resolved_size = float(validation_size)
    if not 0.0 < resolved_size < 1.0:
        raise ValueError("validation_size must be greater than 0 and less than 1.")
    if random_state is not None and (
        isinstance(random_state, bool) or not isinstance(random_state, Integral)
    ):
        raise ValueError("random_state must be an integer or None.")
    if not isinstance(stratify, bool):
        raise ValueError("stratify must be a boolean.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows before splitting.")

    split_config = {
        "validation_size": resolved_size,
        "random_state": int(random_state) if random_state is not None else None,
        "stratify": stratify,
    }
    try:
        train_test_split(
            range(len(y)),
            test_size=resolved_size,
            random_state=split_config["random_state"],
            stratify=list(y) if stratify else None,
        )
    except ValueError as exc:
        prefix = "The requested stratified split is not possible" if stratify else "The requested split is not possible"
        raise ValueError(f"{prefix}: {exc}") from exc

    return resolved_target_name, split_config


def validate_progress_configuration(
    *,
    enabled: bool,
    show_output_shapes: bool,
    return_progress_log: bool,
    show_progress: bool,
) -> tuple[dict[str, bool], bool]:
    values = {
        "enabled": enabled,
        "show_output_shapes": show_output_shapes,
        "return_progress_log": return_progress_log,
        "show_progress": show_progress,
    }
    invalid = [name for name, value in values.items() if not isinstance(value, bool)]
    if invalid:
        raise ValueError(f"These progress settings must be boolean: {', '.join(invalid)}.")
    return (
        {
            "enabled": enabled,
            "show_output_shapes": show_output_shapes,
            "return_progress_log": return_progress_log,
        },
        show_progress,
    )


def build_resolved_config(project_state: Any) -> dict[str, Any]:
    workflow = project_state.prepare_bundles
    return {
        "validation_mode": workflow.validation_mode,
        "target_name": workflow.target_name,
        "target_mapping": dict(project_state.target_mapping or {}),
        "split_kwargs": (
            dict(workflow.split_kwargs) if workflow.split_kwargs is not None else None
        ),
        "validation_kwargs": workflow.validation_kwargs,
        "progress_kwargs": dict(workflow.progress_kwargs),
        "show_progress": workflow.show_progress,
        "external_validation_file_name": workflow.external_validation_file_name,
        "external_target_col": workflow.external_target_col,
        "X_identity": _object_identity(project_state.X),
        "y_identity": _object_identity(project_state.y),
        "feature_names_snapshot": tuple(project_state.feature_names or ()),
        "metadata_identity": _object_identity(project_state.metadata),
        "X_shape": tuple(project_state.X.shape) if project_state.X is not None else None,
        "y_length": len(project_state.y) if project_state.y is not None else None,
        "feature_count": len(project_state.feature_names or []),
        "metadata_available": project_state.metadata is not None,
    }


def record_step_1_review(project_state: Any) -> None:
    """Store the exact structured configuration represented by the audit view."""

    workflow = project_state.prepare_bundles
    resolved = build_resolved_config(project_state)
    workflow.resolved_prepare_bundles_config = resolved
    workflow.reviewed_prepare_bundles_config = dict(resolved)
    workflow.step_1_review_version += 1
    workflow.step_1_review_fingerprint = fingerprint_step_1_config(resolved)
    workflow.step_1_review_status = "awaiting_confirmation"


def fingerprint_step_1_config(config: Mapping[str, Any]) -> str:
    """Return a compact identity for execution-relevant structured settings."""

    validation_kwargs = config.get("validation_kwargs")
    if isinstance(validation_kwargs, Mapping):
        validation_identity = (
            _object_identity(validation_kwargs.get("X")),
            _object_identity(validation_kwargs.get("y")),
            tuple(validation_kwargs.get("feature_names") or ()),
        )
    else:
        validation_identity = None
    payload = (
        config.get("validation_mode"),
        config.get("target_name"),
        _stable_mapping_items(config.get("target_mapping")),
        _stable_mapping_items(config.get("split_kwargs")),
        validation_identity,
        _stable_mapping_items(config.get("progress_kwargs")),
        config.get("show_progress"),
        config.get("external_validation_file_name"),
        config.get("external_target_col"),
        config.get("X_identity"),
        config.get("y_identity"),
        config.get("feature_names_snapshot"),
        config.get("metadata_identity"),
        config.get("X_shape"),
        config.get("y_length"),
        config.get("feature_count"),
        config.get("metadata_available"),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def compact_prepare_bundles_status(project_state: Any) -> dict[str, Any]:
    """Return model-safe status without full DataFrames or bundle contents."""

    workflow = project_state.prepare_bundles
    data: dict[str, Any] = {
        "workflow_stage": "prepare_bundles",
        "stage_name": PREPARE_BUNDLES_STAGE_NAME,
        "ok": workflow.status != "failed",
        "initial_dataset_setup_complete": project_state.setup_status == "completed",
        "prepare_bundles_status": workflow.status,
        "prepare_bundles_complete": workflow.complete,
        "validation_mode": workflow.validation_mode,
        "target_name": workflow.target_name,
        "split_kwargs": (
            dict(workflow.split_kwargs) if workflow.split_kwargs is not None else None
        ),
        "validation_kwargs_summary": _validation_kwargs_summary(project_state),
        "progress_kwargs": dict(workflow.progress_kwargs),
        "show_progress": workflow.show_progress,
        "configuration_confirmed": workflow.configuration_confirmed,
        "step_1_review_status": workflow.step_1_review_status,
        "step_1_review_version": workflow.step_1_review_version,
        "step_1_review_fingerprint": workflow.step_1_review_fingerprint,
        "step_1_status": workflow.status,
        "step_1_execution_log_available": bool(workflow.step_1_execution_log),
        "step_1_execution_error": workflow.step_1_execution_error,
        "step_1_executed_at": (
            workflow.step_1_executed_at.isoformat()
            if workflow.step_1_executed_at is not None
            else None
        ),
        "step_1_executed_review_version": workflow.step_1_executed_review_version,
        "step_1_executed_review_fingerprint": workflow.step_1_executed_review_fingerprint,
        "setting_sources": dict(workflow.setting_sources),
        "external_validation_file_name": workflow.external_validation_file_name,
        "external_target_col": workflow.external_target_col,
        "last_error": workflow.last_error,
        "run_count": workflow.run_count,
        "X_shape": list(project_state.X.shape) if project_state.X is not None else None,
        "y_length": len(project_state.y) if project_state.y is not None else None,
        "feature_count": len(project_state.feature_names or []),
        "dataset_name": (project_state.metadata or {}).get("dataset_name"),
        "metadata_available": project_state.metadata is not None,
        "source_metadata_available": project_state.source_metadata is not None,
        "original_target_col": project_state.target_col,
        "target_mapping": _mapping_entries(project_state.target_mapping),
        "resolved_function_call": render_prepare_bundles_call(project_state),
    }
    if project_state.train_bundle is not None and project_state.prep_meta is not None:
        data.update(summarize_bundle_results(project_state))
    return data


def render_prepare_bundles_call(project_state: Any) -> str | None:
    workflow = project_state.prepare_bundles
    config = workflow.reviewed_prepare_bundles_config
    if config is None:
        return None

    mapping = config.get("target_mapping") or {}
    mapping_lines = [
        f"            {_python_repr(key)}: {_python_repr(value)},"
        for key, value in mapping.items()
    ]
    split_kwargs = config.get("split_kwargs")
    progress_kwargs = config.get("progress_kwargs")
    split_lines = _dict_call_lines(split_kwargs, indent=12)
    progress_lines = _dict_call_lines(progress_kwargs, indent=12)
    if config.get("validation_mode") == EXTERNAL_VALIDATION_MODE:
        validation_display = (
            "{\n"
            "            \"X\": external_validation_X,\n"
            "            \"y\": external_validation_y,\n"
            "            \"feature_names\": external_validation_feature_names,\n"
            "        }"
        )
    else:
        validation_display = "None"

    return "\n".join(
        [
            "train_bundle, validation_bundle, prep_meta = (",
            "    mdp.prepare_train_validation_bundles(",
            "        X=X,",
            "        y=y,",
            "        feature_names=feature_names,",
            "        dataset_metadata=metadata,",
            f"        target_name={_python_repr(config.get('target_name'))},",
            "        target_mapping={",
            *mapping_lines,
            "        },",
            "        split_kwargs=" + ("{\n" if split_kwargs is not None else "None,"),
            *split_lines,
            "        }," if split_kwargs is not None else "",
            f"        validation_kwargs={validation_display},",
            "        progress_kwargs={",
            *progress_lines,
            "        },",
            f"        show_progress={config.get('show_progress')!r},",
            "    )",
            ")",
        ]
    ).replace("\n\n", "\n")


def summarize_bundle_results(project_state: Any) -> dict[str, Any]:
    train_bundle = project_state.train_bundle or {}
    validation_bundle = project_state.validation_bundle
    prep_meta = project_state.prep_meta or {}
    train_y = train_bundle.get("y", [])
    validation_y = validation_bundle.get("y", []) if validation_bundle else []
    split_meta = prep_meta.get("split_metadata", {})
    return {
        "training_row_count": len(train_y),
        "validation_row_count": len(validation_y),
        "raw_feature_count": len(train_bundle.get("feature_names", [])),
        "result_target_name": train_bundle.get("target_name"),
        "result_validation_mode": prep_meta.get("validation_mode"),
        "result_random_state": split_meta.get("random_state"),
        "result_stratify": split_meta.get("stratify"),
        "training_class_counts": _display_count_mapping(
            split_meta.get("train_class_counts")
        ),
        "validation_class_counts": _display_count_mapping(
            split_meta.get("validation_class_counts")
        ),
        "train_bundle_keys": sorted(train_bundle.keys()),
        "validation_bundle_keys": (
            sorted(validation_bundle.keys()) if validation_bundle is not None else []
        ),
        "prep_meta_keys": sorted(prep_meta.keys()),
        "progress_log_stored": "progress_log" in prep_meta,
        "features_are_raw": bool(train_bundle.get("is_raw_split")),
        "features_are_preprocessed": bool(train_bundle.get("is_preprocessed")),
    }


def render_prepare_bundles_output(tool_name: str, data: Mapping[str, Any]) -> str:
    """Render complete Markdown for each deterministic prepare-bundles tool."""

    if tool_name == "run_prepare_train_validation_bundles":
        return _render_step_1_execution(data)
    if tool_name == "show_step_1_execution_log":
        if data.get("ok") is False:
            return _render_error(data)
        return _render_saved_step_1_log(data)
    if data.get("ok") is False:
        return _render_error(data)
    if tool_name == "start_prepare_bundles_stage":
        return _render_start(data)
    if tool_name == "set_prepare_bundles_validation_mode":
        return _render_validation_mode(data)
    if tool_name in {
        "configure_internal_prepare_bundles",
        "update_internal_prepare_bundles",
        "configure_external_prepare_bundles",
    }:
        return _render_resolved_review(data)
    if tool_name == "show_prepare_bundles_advanced_settings":
        return _render_advanced_settings(data)
    return _render_status(data)


def _render_start(data: Mapping[str, Any]) -> str:
    if data.get("prepare_bundles_complete"):
        return _render_completion(data)
    split_kwargs = data.get("split_kwargs") or DEFAULT_SPLIT_KWARGS
    return "\n".join(
        [
            "## Prepare the training and validation data",
            "",
            "This stage determines whether a separate final-validation dataset is created.",
            "",
            "Cross-validation folds, including stratified or nested cross-validation, will be configured later during model development.",
            "",
            "### Primary settings",
            "",
            "| Setting | ML parameter | Recommended value | What it controls |",
            "|---|---|---|---|",
            "| Validation approach | `validation_mode` | **Internal split (Recommended)** | Determines how final validation data are obtained |",
            f"| Final-validation proportion | `validation_size` | **{_format_percent(split_kwargs.get('validation_size'))}** | Keeps part of the current data separate from training |",
            f"| Random seed | `random_state` | **{split_kwargs.get('random_state')}** | Reproduces the same split |",
            f"| Preserve outcome proportions | `stratify` | **{_enabled_text(split_kwargs.get('stratify'))}** | Maintains similar positive and negative outcome proportions in both groups |",
            "",
            "### Advanced settings",
            "",
            _advanced_summary(data),
            "",
            "Say `Show the advanced settings` to see their exact values.",
            "",
            "Supported validation approaches:",
            "",
            "- **Internal final-validation split:** Reserve part of the current dataset for a separate final evaluation.",
            "- **Separate external validation dataset:** Use the current dataset for model development and evaluate later on another compatible dataset.",
            "- **No separate final-validation set:** Keep all current observations available for model development. Evaluation can be configured later using cross-validation, nested cross-validation, bootstrapping, or another supported resampling strategy.",
            "",
            "Examples:",
            "",
            "- `Use the recommended settings.`",
            "- `Use 25% for final validation.`",
            "- `Use random seed 100.`",
            "- `Do not stratify.`",
            "- `I have a separate validation dataset.`",
            "- `Use all current data without a separate final-validation set.`",
        ]
    )


def _render_validation_mode(data: Mapping[str, Any]) -> str:
    if data.get("validation_mode") == NO_VALIDATION_MODE:
        return _render_resolved_review(data)
    if data.get("validation_mode") == EXTERNAL_VALIDATION_MODE:
        if data.get("external_validation_file_name"):
            return "\n".join(
                [
                    "## Prepare the training and validation data",
                    "",
                    f"Loaded the separate validation dataset `{data.get('external_validation_file_name')}`.",
                    "",
                    "Which column contains its prediction outcome?",
                ]
            )
        return "\n".join(
            [
                "## Prepare the training and validation data",
                "",
                "Please attach the separate validation dataset.",
                "",
                "It must contain compatible input features and a prediction-outcome column. Your current training data and pending settings will remain unchanged while you upload it.",
            ]
        )
    return _render_start(data)


def _render_resolved_review(data: Mapping[str, Any]) -> str:
    resolved_call = data.get("resolved_function_call")
    lines = [
        "## Step 1 — Review raw data preparation",
        "",
        "This step organizes the raw feature and outcome data, applies the confirmed target mapping, and prepares the training and final-validation data.",
        "",
        "It does not clean features, handle missing values, encode categories, scale features, select features, or train models.",
        "",
        "| Area | Input or decision | Resolved value | Source |",
        "|---|---|---|---|",
        *_step_1_review_rows(data),
        "",
        "### Exact Step 1 function call",
        "",
        "```python",
        str(resolved_call or "# The Step 1 call is not resolved yet."),
        "```",
        "",
        "**Run Step 1 using this configuration?**",
    ]
    return "\n".join(lines)


def _render_step_1_execution(data: Mapping[str, Any]) -> str:
    if data.get("execution_blocked_for_updated_review"):
        return _render_resolved_review(data)
    if data.get("duplicate_execution"):
        if data.get("prepare_bundles_status") == "running":
            return "Step 1 is already running."
        if data.get("prepare_bundles_status") == "failed":
            return "The previous Step 1 attempt failed. Reply `Rerun Step 1` to try the reviewed configuration again."
        return "Step 1 has already completed. Reply `Rerun Step 1` if you want to replace the previous outputs with a new run."

    log_block = _execution_log_block(data.get("step_1_execution_log"))
    if data.get("execution_failed") or data.get("ok") is False:
        if data.get("prior_configuration_preserved"):
            statement = "Step 1 did not complete. The last valid outputs were preserved."
        else:
            statement = "Step 1 did not complete. No new outputs were stored."
        return f"{log_block}\n\n{statement}"
    return (
        f"{log_block}\n\n"
        "Step 1 completed successfully. The returned features remain raw; no preprocessing has run."
    )


def _render_saved_step_1_log(data: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "### Step 1 execution log",
            "",
            _execution_log_block(data.get("step_1_execution_log")),
        ]
    )


def _execution_log_block(value: Any) -> str:
    log_text = str(value or "No framework progress output was produced.")
    return f"```text\n{log_text}\n```"


def _render_completion(data: Mapping[str, Any]) -> str:
    mode = data.get("result_validation_mode")
    if mode == "train_only":
        return _render_no_validation_completion(data)
    stratified = mode == "internal_split" and data.get("result_stratify") is True
    approach = {
        "internal_split": "Internal stratified split" if stratified else "Internal split",
        "provided_validation": "Separate validation dataset",
    }.get(mode, "Configured validation approach")
    random_seed = (
        data.get("result_random_state") if mode == "internal_split" else None
    )
    lines = [
        "## Training and validation data prepared",
        "",
        "| Result | Value |",
        "|---|---:|",
        f"| Training rows | {data.get('training_row_count')} |",
        f"| Final-validation rows | {data.get('validation_row_count')} |",
        f"| Input features | {data.get('raw_feature_count')} |",
        f"| Validation approach | {approach} |",
        f"| Random seed | {random_seed if random_seed is not None else 'Not applicable'} |",
        "",
    ]
    if stratified:
        lines.extend(
            [
                "### Outcome distribution",
                "",
                "| Dataset | Negative outcomes | Positive outcomes |",
                "|---|---:|---:|",
                f"| Training | {_class_count(data.get('training_class_counts'), 0.0)} | {_class_count(data.get('training_class_counts'), 1.0)} |",
                f"| Final validation | {_class_count(data.get('validation_class_counts'), 0.0)} | {_class_count(data.get('validation_class_counts'), 1.0)} |",
                "",
                "The proportion of positive and negative outcomes was preserved in both groups.",
                "",
            ]
        )
    lines.extend(
        [
            "The input features are still in their original form. No missing-value handling, categorical encoding, scaling, feature selection, or model training has run yet.",
            "",
            "The next stage will prepare the input features for modeling.",
        ]
    )
    return "\n".join(lines)


def _render_no_validation_completion(data: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "## Training data prepared",
            "",
            "| Result | Value |",
            "|---|---:|",
            f"| Training rows | {data.get('training_row_count')} |",
            "| Separate final-validation rows | None |",
            f"| Input features | {data.get('raw_feature_count')} |",
            "| Validation approach | No separate final-validation set |",
            "| Later evaluation strategy | To be configured |",
            "",
            "All current observations remain available for model development.",
            "",
            "No independent held-out evaluation set was created from this dataset. Cross-validation, nested cross-validation, bootstrapping, or another supported evaluation strategy can be configured later.",
            "",
            "The input features are still in their original form. No missing-value handling, categorical encoding, scaling, feature selection, or model training has run yet.",
            "",
            "The next stage will prepare the input features for modeling.",
        ]
    )


def _render_advanced_settings(data: Mapping[str, Any]) -> str:
    lines = ["## Advanced settings", "", *_advanced_settings_table(data), ""]
    if data.get("prepare_bundles_status") == "awaiting_final_confirmation":
        lines.append("The selected preparation settings are still awaiting your final confirmation.")
    else:
        lines.append("The training and validation configuration is still awaiting your decision.")
    return "\n".join(lines)


def _render_status(data: Mapping[str, Any]) -> str:
    status = data.get("prepare_bundles_status")
    if status == "complete":
        return _render_completion(data)
    if status == "awaiting_external_data":
        return _render_validation_mode(data)
    if status == "awaiting_configuration":
        return _render_validation_mode(data)
    if status == "awaiting_final_confirmation":
        return _render_resolved_review(data)
    return "\n".join(
        [
            "## Prepare the training and validation data",
            "",
            "The configuration is waiting for your next decision.",
        ]
    )


def _render_error(data: Mapping[str, Any]) -> str:
    lines = [
        "## Prepare the training and validation data",
        "",
        str(data.get("message") or data.get("last_error") or "The requested change could not be applied."),
    ]
    if data.get("prior_configuration_preserved"):
        lines.append("Your previous valid settings were preserved.")
    return "\n".join(lines)


def _step_1_review_rows(data: Mapping[str, Any]) -> list[str]:
    sources = data.get("setting_sources") or {}
    X_shape = data.get("X_shape") or [0, 0]
    rows = X_shape[0] if len(X_shape) > 0 else 0
    features = data.get("feature_count") or 0
    metadata_available = bool(data.get("metadata_available"))
    metadata_source = (
        "Dataset metadata"
        if data.get("source_metadata_available")
        else "Uploaded dataset"
    )
    review_rows = [
        _review_row("Dataset", "Feature data", f"`X` — {rows} rows × {features} features", "Uploaded dataset"),
        _review_row("Dataset", "Outcome data", f"`y` — {data.get('y_length')} values", "Uploaded dataset"),
        _review_row("Dataset", "Feature names", f"{features} feature names", "Uploaded dataset"),
        _review_row("Dataset", "Dataset metadata", "Available" if metadata_available else "Not available", metadata_source),
        _review_row("Target", "Internal target name", f"`{data.get('target_name')}`", _setting_source(sources, "target_name")),
        _review_row("Target", "Negative-class encoding", _target_encoding(data.get("target_mapping"), 0.0), "User confirmed"),
        _review_row("Target", "Positive-class encoding", _target_encoding(data.get("target_mapping"), 1.0), "User confirmed"),
    ]

    validation_mode = data.get("validation_mode")
    split_kwargs = data.get("split_kwargs") or {}
    if validation_mode == INTERNAL_VALIDATION_MODE:
        validation_size = float(split_kwargs.get("validation_size", 0.0))
        review_rows.extend(
            [
                _review_row("Validation", "Validation approach", "Internal split", _setting_source(sources, "validation_mode")),
                _review_row("Validation", "Training proportion", _format_percent(1.0 - validation_size), "Derived"),
                _review_row("Validation", "Final-validation proportion", _format_percent(validation_size), _setting_source(sources, "validation_size")),
                _review_row("Validation", "Random state", str(split_kwargs.get("random_state")), _setting_source(sources, "random_state")),
                _review_row("Validation", "Stratification", _enabled_text(split_kwargs.get("stratify")), _setting_source(sources, "stratify")),
                _review_row("Validation", "External validation data", "None", "Derived"),
            ]
        )
    elif validation_mode == EXTERNAL_VALIDATION_MODE:
        summary = data.get("validation_kwargs_summary") or {}
        shape = summary.get("X_shape") or [0, 0]
        external_value = (
            f"`{data.get('external_validation_file_name')}` — "
            f"{shape[0]} rows × {shape[1]} features"
        )
        review_rows.extend(
            [
                _review_row("Validation", "Validation approach", "Separate external validation dataset", _setting_source(sources, "validation_mode")),
                _review_row("Validation", "Training proportion", "100% of current dataset", "Derived"),
                _review_row("Validation", "Final-validation data", "Separate uploaded dataset", "User selected"),
                _review_row("Validation", "Random state", "Not applicable", "Derived"),
                _review_row("Validation", "Stratification", "Not applicable", "Derived"),
                _review_row("Validation", "External validation data", external_value, "Uploaded dataset"),
                _review_row("Validation", "External outcome column", f"`{data.get('external_target_col')}`", "User selected"),
            ]
        )
    else:
        review_rows.extend(
            [
                _review_row("Validation", "Validation approach", "No separate final-validation set", _setting_source(sources, "validation_mode")),
                _review_row("Validation", "Data available for model development", "100%", "Derived"),
                _review_row("Validation", "Separate held-out validation data", "None", "Derived"),
                _review_row("Validation", "Random state", f"{split_kwargs.get('random_state')} (not used without a split)", _setting_source(sources, "random_state")),
                _review_row("Validation", "Stratification", "Disabled", "Derived"),
                _review_row("Validation", "Later evaluation strategy", "To be configured during model development", "Derived"),
            ]
        )

    progress_kwargs = data.get("progress_kwargs") or {}
    review_rows.extend(
        [
            _review_row("Reporting", "Progress messages", _enabled_text(progress_kwargs.get("enabled")), _setting_source(sources, "progress_enabled")),
            _review_row("Reporting", "Show output shapes", _enabled_text(progress_kwargs.get("show_output_shapes")), _setting_source(sources, "show_output_shapes")),
            _review_row("Reporting", "Store progress log", _enabled_text(progress_kwargs.get("return_progress_log")), _setting_source(sources, "return_progress_log")),
            _review_row("Reporting", "Backward-compatible progress", _enabled_text(data.get("show_progress")), _setting_source(sources, "show_progress")),
        ]
    )
    return review_rows


def _review_row(area: str, input_name: str, value: str, source: str) -> str:
    return "| " + " | ".join(
        _table_cell(item) for item in (area, input_name, value, source)
    ) + " |"


def _table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _stable_mapping_items(value: Any) -> tuple[tuple[str, str], ...] | None:
    if not isinstance(value, Mapping):
        return None
    return tuple(sorted((repr(key), repr(item)) for key, item in value.items()))


def _object_identity(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    shape = tuple(value.shape) if hasattr(value, "shape") else None
    columns = tuple(value.columns) if hasattr(value, "columns") else None
    try:
        length = len(value)
    except TypeError:
        length = None
    return (id(value), shape, columns, length)


def _setting_source(sources: Mapping[str, Any], name: str) -> str:
    return str(sources.get(name) or "Source not recorded")


def _target_encoding(entries: Any, encoded_value: float) -> str:
    if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            if entry.get("encoded_value") == encoded_value:
                return f"`{entry.get('original_value')} → {encoded_value:.1f}`"
    return "Not available"


def _advanced_summary(data: Mapping[str, Any]) -> str:
    progress_kwargs = data.get("progress_kwargs") or DEFAULT_PROGRESS_KWARGS
    settings = [
        ("Progress messages", progress_kwargs.get("enabled")),
        ("output-dimension reporting", progress_kwargs.get("show_output_shapes")),
        ("progress-history storage", progress_kwargs.get("return_progress_log")),
    ]
    if all(value is True for _, value in settings):
        return "Progress messages, output-dimension reporting, and progress-history storage are enabled."
    return "; ".join(
        f"{name} {_enabled_text(value).casefold()}" for name, value in settings
    ) + "."


def _advanced_settings_table(data: Mapping[str, Any]) -> list[str]:
    progress_kwargs = data.get("progress_kwargs") or DEFAULT_PROGRESS_KWARGS
    return [
        "| Setting | ML parameter | Selected value |",
        "|---|---|---|",
        f"| Progress messages | `enabled` | {_enabled_text(progress_kwargs.get('enabled'))} |",
        f"| Show resulting dimensions | `show_output_shapes` | {_enabled_text(progress_kwargs.get('show_output_shapes'))} |",
        f"| Store progress history | `return_progress_log` | {_enabled_text(progress_kwargs.get('return_progress_log'))} |",
        f"| Internal outcome name | `target_name` | `{data.get('target_name')}` |",
    ]


def _class_count(entries: Any, encoded_value: float) -> int | str:
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return "Unavailable"
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        try:
            value = float(entry.get("value"))
        except (TypeError, ValueError):
            continue
        if value == encoded_value:
            return int(entry.get("count", 0))
    return 0


def _validation_kwargs_summary(project_state: Any) -> dict[str, Any] | None:
    workflow = project_state.prepare_bundles
    if workflow.validation_kwargs is None:
        return None
    X = workflow.validation_kwargs.get("X")
    y = workflow.validation_kwargs.get("y")
    return {
        "file_name": workflow.external_validation_file_name,
        "target_col": workflow.external_target_col,
        "X_shape": list(X.shape) if hasattr(X, "shape") else None,
        "y_length": len(y) if y is not None else None,
        "feature_count": len(workflow.validation_kwargs.get("feature_names") or []),
    }


def _format_percent(value: Any) -> str:
    try:
        percentage = float(value) * 100.0
    except (TypeError, ValueError):
        return "Unknown"
    return f"{percentage:g}%"


def _enabled_text(value: Any) -> str:
    return "Enabled" if value is True else "Disabled"


def _mapping_entries(mapping: Mapping[Any, float] | None) -> list[dict[str, Any]]:
    return [
        {
            "original_value": display_target_value(key),
            "encoded_value": float(value),
        }
        for key, value in (mapping or {}).items()
    ]


def _display_count_mapping(mapping: Mapping[Any, Any] | None) -> list[dict[str, Any]]:
    return [
        {"value": display_target_value(key), "count": int(value)}
        for key, value in (mapping or {}).items()
    ]


def _dict_call_lines(mapping: Mapping[str, Any] | None, *, indent: int) -> list[str]:
    if mapping is None:
        return []
    spaces = " " * indent
    return [f"{spaces}{key!r}: {_python_repr(value)}," for key, value in mapping.items()]


def _python_repr(value: Any) -> str:
    if isinstance(value, Real) and not isinstance(value, (bool, Integral)):
        return repr(float(value))
    return repr(value)


def _mapping_text(entries: Any) -> str:
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return "None"
    parts = []
    for entry in entries:
        if isinstance(entry, Mapping):
            parts.append(
                f"`{entry.get('original_value')}` -> `{entry.get('encoded_value')}`"
            )
    return ", ".join(parts) or "None"


def _count_text(entries: Any) -> str:
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return "None"
    parts = []
    for entry in entries:
        if isinstance(entry, Mapping):
            parts.append(f"`{entry.get('value')}`: {entry.get('count')}")
    return ", ".join(parts) or "None"


def _validation_summary_text(summary: Any) -> str:
    if not isinstance(summary, Mapping):
        return "`None`"
    return (
        f"external file `{summary.get('file_name')}`, target `{summary.get('target_col')}`, "
        f"X shape {_shape_text(summary.get('X_shape'))}"
    )


def _shape_text(shape: Any) -> str:
    if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)) and len(shape) == 2:
        return f"{shape[0]} x {shape[1]}"
    return "unknown"


def _code_list(values: Any) -> str:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return "None"
    return ", ".join(f"`{value}`" for value in values) or "None"
