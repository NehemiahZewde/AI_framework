# MILESTONE 6B - DOCUMENTATION AND READABILITY POLISH - VERSION 2
# Raw train/validation organization, validation, splitting, and bundle creation.
# Behavior-preserving documentation and inline-comment pass over the tested Milestone 6A pipeline.
# Public behavior, progress output, bundle keys, metadata, and error messages remain unchanged.

"""Train/validation preparation utilities for the AI framework.

This module owns raw dataset organization before feature preprocessing:
- resolving and validating feature and target inputs;
- validating Version 1 task, target, and split contracts;
- creating internal train/final-validation splits;
- aligning provided external validation data;
- encoding target labels;
- constructing raw train and validation bundles.

The public preparation function is intentionally organized as a pipeline:
preflight validation first, followed by focused execution steps. Feature
imputation, feature encoding, scaling, cleaning, and sanitization remain in
``ml_data_preprocessing.py``.
"""

from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = [
    "prepare_dataset",
    "prepare_train_validation_bundles",
    "encode_target_labels",
    "make_pipeline_progress_helpers",
]

# =============================================================================
# 1. Progress reporting utilities
# =============================================================================

def make_pipeline_progress_helpers(
    *,
    progress_enabled: bool = True,
    show_output_shapes: bool = True,
) -> Tuple[
    List[Dict[str, Any]],
    Callable[[Any], str],
    Callable[[str], None],
    Callable[[str, Any], None],
    Callable[[str, str], None],
    Callable[[str, Exception], None],
]:
    """Create the progress functions shared by pipeline-style workflows.

    Parameters
    ----------
    progress_enabled : bool, default=True
        Whether step-start, success, skip, and failure messages should be printed.
    show_output_shapes : bool, default=True
        Whether successful step messages should include a compact description of
        the returned object, such as a DataFrame shape or bundle shape.

    Returns
    -------
    tuple
        A six-item tuple containing the mutable progress log and five helper
        functions: object description, step start, step success, step skip, and
        step failure.

    Notes
    -----
    The returned helper functions all close over the same ``progress_log`` list.
    This allows the public preparation pipeline to print progress and retain the
    same information in ``prep_meta`` without duplicating reporting logic.
    """

    # Initialize progress log.
    progress_log: List[Dict[str, Any]] = []


    def describe_object(obj: Any) -> str:
        """
        Return a compact string describing a pipeline output object.
        """
        # Respect the caller's choice to suppress shape and object summaries.
        if not show_output_shapes:
            return ""

        # Handle each common pipeline return type with a concise description.
        if obj is None:
            return "None"

        if isinstance(obj, pd.DataFrame):
            return f"DataFrame shape={obj.shape}"

        # Bundles and metadata dictionaries need more informative summaries
        # than the generic Python type name.
        if isinstance(obj, dict):
            if "X_raw" in obj and "X_scaled" in obj:
                return (
                    f"bundle X_raw={tuple(obj['X_raw'].shape)}, "
                    f"X_scaled={tuple(obj['X_scaled'].shape)}"
                )

            if "X_raw" in obj:
                return f"bundle X_raw={tuple(obj['X_raw'].shape)}"

            return f"dict keys={list(obj.keys())}"

        if isinstance(obj, np.ndarray):
            return f"ndarray shape={obj.shape}"

        if isinstance(obj, (list, tuple)):
            return f"{type(obj).__name__} len={len(obj)}"

        if isinstance(obj, str):
            return obj

        # Fall back to the object's class name for uncommon return types.
        return type(obj).__name__

    def start_step(name: str) -> None:
        """
        Print the start of a pipeline step.
        """
        # Printing is optional, but logging behavior remains available
        # through the other callbacks.
        if progress_enabled:
            print(f">> {name}")

    def ok_step(name: str, obj: Any = None) -> None:
        """
        Record and print a successful pipeline step.
        """
        # Convert the returned object into the compact detail shown to the user.
        desc = describe_object(obj)

        # Build the same stable success message used by the existing notebooks.
        message = f"[OK] {name}"
        if desc:
            message += f" -> {desc}"

        # Store the machine-readable version of the displayed progress event.
        progress_log.append(
            {
                "step": name,
                "status": "ok",
                "detail": desc,
            }
        )

        if progress_enabled:
            print(message)

    def skip_step(name: str, reason: str) -> None:
        """
        Record and print a skipped pipeline step.
        """
        # Record why the optional step did not run so the audit trail is complete.
        progress_log.append(
            {
                "step": name,
                "status": "skipped",
                "detail": reason,
            }
        )

        if progress_enabled:
            print(f"[SKIP] {name} -> {reason}")

    def fail_step(name: str, err: Exception) -> None:
        """
        Record and print a failed pipeline step.
        """
        # Preserve the original exception text in the progress log before
        # the exception is re-raised by the pipeline.
        progress_log.append(
            {
                "step": name,
                "status": "fail",
                "detail": str(err),
            }
        )

        if progress_enabled:
            print(f"[FAIL] {name} -> {err}")

    return (
        progress_log,
        describe_object,
        start_step,
        ok_step,
        skip_step,
        fail_step,
    )

def _run_progress_step(
    step_name: str,
    operation: Callable[[], Any],
    *,
    start_step: Callable[[str], None],
    ok_step: Callable[[str, Any], None],
    fail_step: Callable[[str, Exception], None],
    display_value: Optional[Callable[[Any], Any]] = None,
) -> Any:
    """Execute one pipeline operation using the standard progress lifecycle.

    Parameters
    ----------
    step_name : str
        Human-readable step name shown in the progress output and stored in the
        progress log.
    operation : Callable[[], Any]
        Zero-argument callable that performs the actual pipeline work.
    start_step : Callable[[str], None]
        Progress callback used before the operation starts.
    ok_step : Callable[[str, Any], None]
        Progress callback used after successful completion.
    fail_step : Callable[[str, Exception], None]
        Progress callback used when the operation raises an exception.
    display_value : Callable[[Any], Any], optional
        Optional adapter that extracts a compact value to report while preserving
        the operation's full return value.

    Returns
    -------
    Any
        The unmodified result returned by ``operation``.

    Raises
    ------
    Exception
        Re-raises any exception produced by ``operation`` after recording the
        failed progress step.

    Notes
    -----
    Centralizing this start/try/success/failure pattern keeps the public pipeline
    short while preserving the established visible progress output.
    """
    # Announce the operation before any user code is executed.
    start_step(step_name)

    try:
        # Run the focused pipeline helper and keep its complete return value.
        result = operation()
        # Report either the full result or a smaller display-oriented summary.
        reported_value = display_value(result) if display_value else result
        ok_step(step_name, reported_value)
        return result
    except Exception as err:
        # Record the failed step and then preserve the original traceback.
        fail_step(step_name, err)
        raise

# =============================================================================
# 2. Configuration validation and normalization
# =============================================================================

def _normalize_validation_feature_policy(policy: Any) -> str:
    """Normalize and validate the provided-validation feature policy.

    Parameters
    ----------
    policy : Any
        Requested policy value. Supported string values are ``"strict"`` and
        ``"intersection"``; matching is case-insensitive and surrounding
        whitespace is ignored.

    Returns
    -------
    str
        The normalized lowercase policy.

    Raises
    ------
    TypeError
        If ``policy`` is not a string.
    ValueError
        If the normalized policy is unsupported.
    """
    # Reject non-string policies before attempting string normalization.
    if not isinstance(policy, str):
        raise TypeError(
            "validation_feature_policy must be a string with value "
            "'strict' or 'intersection'."
        )

    # Normalize cosmetic differences so callers receive one stable policy value.
    # Normalize harmless capitalization and surrounding whitespace.
    normalized_policy = policy.strip().lower()

    # Keep the supported choices explicit so unknown values cannot silently
    # fall back to one of the alignment modes.
    supported_policies = {"strict", "intersection"}

    if normalized_policy not in supported_policies:
        raise ValueError(
            "Unknown validation_feature_policy="
            f"{policy!r}. Supported values are: "
            f"{sorted(supported_policies)}"
        )

    return normalized_policy

def _validate_version1_declared_task(
    dataset_metadata: Optional[Mapping[str, Any]],
) -> None:
    """Validate the task declared in dataset metadata for Version 1.

    Parameters
    ----------
    dataset_metadata : Mapping[str, Any], optional
        Dataset metadata that may contain an ``ml_task`` field.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If an explicit task is present and is not binary classification.

    Notes
    -----
    An absent metadata dictionary or absent ``ml_task`` value is allowed. The
    actual target values are validated separately by the target-contract helpers.
    """
    # An absent metadata dictionary cannot declare an unsupported task.
    if not dataset_metadata:
        return

    # Read only the task field; unrelated dataset metadata is preserved.
    declared_task = dataset_metadata.get("ml_task")
    if declared_task is None:
        return

    # Normalize common spelling formats such as "binary classification"
    # and "binary-classification" to the same internal representation.
    normalized_task = str(declared_task).strip().lower().replace("-", "_").replace(" ", "_")
    supported_tasks = {"binary", "binary_classification"}

    if normalized_task not in supported_tasks:
        raise ValueError(
            "Version 1 supports binary classification only. "
            f"Received dataset_metadata['ml_task']={declared_task!r}. "
            "Regression and multiclass support are deferred to Version 2."
        )

def _validate_prepare_split_configuration(split_config: Dict[str, Any]) -> None:
    """Validate and normalize the internal split configuration in place.

    Parameters
    ----------
    split_config : dict[str, Any]
        Mutable split configuration containing ``validation_size``,
        ``random_state``, and ``stratify``.

    Returns
    -------
    None
        The supplied dictionary is updated in place with a real Boolean
        ``stratify`` value and a floating-point ``validation_size`` when present.

    Raises
    ------
    TypeError
        If ``stratify`` is not Boolean or ``validation_size`` is not a numeric
        fraction or ``None``.
    ValueError
        If ``validation_size`` is outside ``0.0 <= validation_size < 1.0``.

    Notes
    -----
    This framework-level validation runs before scikit-learn so callers receive
    stable, actionable configuration errors.
    """
    # Validate the Boolean exactly; strings such as "False" are truthy in Python
    # and would otherwise change the requested statistical behavior.
    stratify = split_config.get("stratify", True)
    if not isinstance(stratify, (bool, np.bool_)):
        raise TypeError(
            "split_kwargs['stratify'] must be a bool. "
            f"Received {type(stratify).__name__}: {stratify!r}."
        )
    # Store a native bool so downstream metadata and sklearn calls are consistent.
    split_config["stratify"] = bool(stratify)

    # A value of None or 0.0 intentionally represents train-only mode.
    validation_size = split_config.get("validation_size", 0.2)
    if validation_size is None:
        return

    if isinstance(validation_size, (bool, np.bool_)):
        raise TypeError(
            "validation_size must be a numeric fraction in the range "
            "0.0 <= validation_size < 1.0, or None."
        )

    # Normalize numeric inputs such as NumPy scalars to a standard float.
    try:
        validation_size = float(validation_size)
    except (TypeError, ValueError) as err:
        raise TypeError(
            "validation_size must be a numeric fraction in the range "
            "0.0 <= validation_size < 1.0, or None."
        ) from err

    if validation_size < 0.0:
        raise ValueError(
            "validation_size must be greater than or equal to 0.0."
        )
    if validation_size >= 1.0:
        raise ValueError(
            "validation_size must be less than 1.0 so that training rows remain."
        )

    # Write the normalized value back into the configuration used later.
    split_config["validation_size"] = validation_size


def _resolve_progress_configuration(
    progress_kwargs: Optional[Dict[str, Any]],
    *,
    show_progress: bool,
) -> Dict[str, Any]:
    """Merge caller progress options with the stable pipeline defaults.

    Parameters
    ----------
    progress_kwargs : dict[str, Any], optional
        Caller overrides for ``enabled``, ``show_output_shapes``, and
        ``return_progress_log``.
    show_progress : bool
        Backward-compatible default for the ``enabled`` option.

    Returns
    -------
    dict[str, Any]
        Complete progress configuration used by the public pipeline.
    """
    # Start from the stable progress behavior used by existing notebooks.
    progress_defaults: Dict[str, Any] = {
        "enabled": show_progress,
        "show_output_shapes": True,
        "return_progress_log": True,
    }

    # Caller-provided settings override only the corresponding defaults.
    return {
        **progress_defaults,
        **dict(progress_kwargs or {}),
    }

def _resolve_and_validate_prepare_configuration(
    *,
    dataset_metadata: Optional[Mapping[str, Any]],
    split_kwargs: Optional[Dict[str, Any]],
    validation_kwargs: Optional[Dict[str, Any]],
    validation_feature_policy: Any,
) -> Dict[str, Any]:
    """Resolve and validate all caller-controlled preparation settings.

    Parameters
    ----------
    dataset_metadata : Mapping[str, Any], optional
        Dataset metadata used to enforce the Version 1 task boundary.
    split_kwargs : dict[str, Any], optional
        Overrides for the supported internal split settings.
    validation_kwargs : dict[str, Any], optional
        Optional provided-validation inputs. When either ``X`` or ``y`` is
        supplied, both are required.
    validation_feature_policy : Any
        Requested provided-validation feature policy.

    Returns
    -------
    dict[str, Any]
        Normalized split settings, validation settings, a Boolean indicating
        whether provided validation is active, and the normalized feature policy.

    Raises
    ------
    ValueError
        If split keys are unknown, provided validation is incomplete, the task is
        unsupported, or a configuration value is outside its accepted contract.
    TypeError
        If a typed configuration value has the wrong type.

    Notes
    -----
    This helper performs configuration-only preflight checks. It does not inspect
    or modify feature and target data.
    """
    # Define the only supported internal-split options and their defaults.
    split_defaults: Dict[str, Any] = {
        "validation_size": 0.2,
        "random_state": 42,
        "stratify": True,
    }

    # Compare caller keys against the explicit contract before merging values.
    supported_split_keys = set(split_defaults)
    provided_split_kwargs = dict(split_kwargs or {})
    unknown_split_keys = sorted(
        set(provided_split_kwargs) - supported_split_keys
    )

    if unknown_split_keys:
        raise ValueError(
            "Unknown or unsupported split_kwargs keys: "
            f"{unknown_split_keys}. Supported keys are: "
            f"{sorted(supported_split_keys)}"
        )

    # Caller values override defaults only after unsupported keys are rejected.
    # Merge validated overrides into an independent configuration dictionary.
    split_config: Dict[str, Any] = {
        **split_defaults,
        **provided_split_kwargs,
    }

    # Validate the task boundary and normalize split values before any data work.
    _validate_version1_declared_task(dataset_metadata)
    _validate_prepare_split_configuration(split_config)

    normalized_validation_feature_policy = (
        _normalize_validation_feature_policy(validation_feature_policy)
    )

    # Copy the external-validation settings so caller-owned dictionaries are not modified.
    validation_config: Dict[str, Any] = dict(validation_kwargs or {})
    # Enter provided-validation mode when either X or y is present, then
    # require the matching partner below.
    provided_validation = (
        validation_config.get("X", None) is not None
        or validation_config.get("y", None) is not None
    )

    if provided_validation:
        if validation_config.get("X", None) is None:
            raise ValueError(
                "validation_kwargs includes a validation target but no validation X. "
                "Provide validation_kwargs['X']."
            )

        if validation_config.get("y", None) is None:
            raise ValueError(
                "validation_kwargs includes validation X but no validation target. "
                "Provide validation_kwargs['y']."
            )

    # Return one trusted configuration object for the later preflight steps.
    return {
        "split_config": split_config,
        "validation_config": validation_config,
        "provided_validation": provided_validation,
        "validation_feature_policy": normalized_validation_feature_policy,
    }

# =============================================================================
# 3. Target validation and encoding
# =============================================================================

def encode_target_labels(
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    mapping: Optional[Mapping[Any, Any]] = None,
    inplace: bool = False,
    return_metadata: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """Encode one target column using an explicit class mapping.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the target column.
    target_col : str, default="target"
        Name of the target column to encode.
    mapping : Mapping[Any, Any], optional
        Mapping from observed source labels to encoded values. When omitted, the
        legacy ``tested_negative``/``tested_positive`` mapping is used.
    inplace : bool, default=False
        Whether to modify ``df`` directly. The default returns an independent copy.
    return_metadata : bool, default=False
        Whether to return target-encoding metadata together with the DataFrame.

    Returns
    -------
    pandas.DataFrame or tuple[pandas.DataFrame, dict[str, Any]]
        Encoded DataFrame, optionally paired with the mapping, class counts,
        resulting dtype, and in-place flag.

    Raises
    ------
    TypeError
        If ``df`` is not a pandas DataFrame.
    KeyError
        If ``target_col`` is absent.
    ValueError
        If an observed non-missing label is not represented in ``mapping``.

    Notes
    -----
    This helper encodes supervised outcomes only. Feature encoding belongs in
    ``ml_data_preprocessing.py``.
    """

    # Ensure the input is a pandas DataFrame.
    # Target encoding operates on named DataFrame columns, so reject other inputs.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    # Use the default binary class mapping if no custom mapping is provided.
    # Use the historical default only when the caller does not provide a mapping.
    if mapping is None:
        mapping = {
            "tested_negative": 0,
            "tested_positive": 1,
        }

    # Check that the requested target column exists in the DataFrame.
    if target_col not in df.columns:
        raise KeyError(f"target_col='{target_col}' was not found in the DataFrame.")

    # Either modify the original DataFrame or create a copy, depending on inplace.
    # Copy by default so target encoding cannot mutate caller-owned data.
    encoded_df = df if inplace else df.copy()

    # Store the original target values before encoding for validation and metadata.
    original_target = encoded_df[target_col]

    # Find all unique non-missing labels in the target column.
    observed_labels = set(original_target.dropna().unique())

    # Validate coverage before mapping so an unknown clinical label cannot be
    # silently converted to a missing value by pandas.Series.map().
    # Find labels that appear in the data but are missing from the mapping.
    # Mapping unknown labels would create missing values, so detect them explicitly first.
    unknown_labels = observed_labels - set(mapping.keys())

    # Raise an error if any target labels cannot be encoded.
    if unknown_labels:
        raise ValueError(
            f"Found target labels not present in mapping: {sorted(unknown_labels)}"
        )

    # Count the original class labels before encoding.
    class_counts_before = original_target.value_counts(dropna=False).to_dict()

    # Replace original labels with encoded integer values.
    # Apply the approved mapping only after every observed label is known to be covered.
    encoded_df[target_col] = original_target.map(mapping)

    # Count the encoded class labels after encoding.
    class_counts_after = encoded_df[target_col].value_counts(dropna=False).to_dict()

    # Build metadata so the encoding can be inspected later if desired.
    metadata: Dict[str, Any] = {
        "target_col": target_col,
        "mapping": mapping,
        "class_counts_before": class_counts_before,
        "class_counts_after": class_counts_after,
        "encoded_dtype": str(encoded_df[target_col].dtype),
        "inplace": inplace,
    }

    # Return both the encoded DataFrame and metadata if requested.
    if return_metadata:
        return encoded_df, metadata

    # Otherwise, return only the encoded DataFrame.
    return encoded_df

def _raise_for_missing_target_values(
    y_values: np.ndarray,
    *,
    input_name: str,
) -> None:
    """Reject missing supervised target values before preparation begins.

    Parameters
    ----------
    y_values : numpy.ndarray
        Resolved one-dimensional target values.
    input_name : str
        Human-readable input name included in the error message.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If one or more target values are missing. The message reports their
        positional locations so the source data can be corrected explicitly.
    """
    # Build a positional mask so the error can identify exactly where targets are missing.
    missing_mask = pd.isna(y_values)
    missing_count = int(np.asarray(missing_mask).sum())
    if missing_count:
        missing_positions = np.flatnonzero(np.asarray(missing_mask)).tolist()
        raise ValueError(
            f"{input_name} contains {missing_count} missing target value(s) "
            f"at position(s) {missing_positions}. Missing target values must "
            "be handled before preparing supervised bundles."
        )

def _observed_target_labels(y_values: np.ndarray) -> List[Any]:
    """Return unique non-missing target labels in first-occurrence order.

    Parameters
    ----------
    y_values : numpy.ndarray
        Resolved target vector.

    Returns
    -------
    list[Any]
        Stable list of observed labels without sorting or type coercion.
    """
    # pandas preserves first-occurrence order while removing missing values and duplicates.
    return pd.Series(y_values).dropna().drop_duplicates().tolist()

def _validate_labels_present_in_mapping(
    y_values: np.ndarray,
    *,
    target_mapping: Optional[Mapping[Any, Any]],
) -> None:
    """Confirm that every observed label is represented in the target mapping.

    Parameters
    ----------
    y_values : numpy.ndarray
        Resolved target vector.
    target_mapping : Mapping[Any, Any], optional
        Explicit mapping used by target encoding.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If an observed label is absent from the explicit mapping.
    """
    # Without an explicit mapping there are no mapping keys to validate.
    if target_mapping is None:
        return

    # Compare every observed source label against the approved mapping keys.
    observed_labels = _observed_target_labels(y_values)
    unknown_labels = [label for label in observed_labels if label not in target_mapping]
    if unknown_labels:
        raise ValueError(
            f"Found target labels not present in mapping: {unknown_labels}"
        )

def _validate_version1_training_target_contract(
    y_values: np.ndarray,
    *,
    target_mapping: Optional[Mapping[Any, Any]],
    input_name: str,
) -> None:
    """Enforce the complete Version 1 training-target contract.

    Parameters
    ----------
    y_values : numpy.ndarray
        Resolved training target vector before encoding.
    target_mapping : Mapping[Any, Any], optional
        Explicit source-label mapping. When omitted, the observed targets must
        already represent ``0.0`` and ``1.0``.
    input_name : str
        Human-readable input name used in errors.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If targets are missing, labels are absent from the mapping, the source data
        does not contain exactly two observed classes, or the encoded contract is
        not exactly ``0.0`` and ``1.0``.

    Notes
    -----
    The source data itself must be genuinely binary. Version 1 does not permit a
    mapping that collapses three or more observed source classes into two outputs.
    """
    # Missing outcomes are rejected before class and mapping checks.
    _raise_for_missing_target_values(y_values, input_name=input_name)
    _validate_labels_present_in_mapping(
        y_values,
        target_mapping=target_mapping,
    )

    # Version 1 requires a genuinely binary source target, not a mapping
    # that collapses three or more observed classes into two outputs.
    observed_labels = _observed_target_labels(y_values)
    if len(observed_labels) != 2:
        raise ValueError(
            "Version 1 binary classification requires exactly two observed "
            f"target classes in {input_name}. Observed {len(observed_labels)}: "
            f"{observed_labels}. Both binary classes must be present."
        )

    if target_mapping is not None:
        # Validate the complete mapping range, not only the labels observed
        # in the current training sample.
        try:
            mapping_values = {float(value) for value in target_mapping.values()}
        except (TypeError, ValueError) as err:
            raise ValueError(
                "Version 1 target mapping values must be exactly 0.0 and 1.0."
            ) from err

        if mapping_values != {0.0, 1.0}:
            raise ValueError(
                "Version 1 target mapping values must be exactly 0.0 and 1.0. "
                f"Received values: {sorted(mapping_values)}"
            )

        # Confirm that the two observed source labels reach both binary outputs.
        encoded_values = {
            float(target_mapping[label])
            for label in observed_labels
        }
    else:
        # When no mapping is supplied, the source target must already use
        # the framework's 0.0/1.0 convention.
        try:
            encoded_values = {float(label) for label in observed_labels}
        except (TypeError, ValueError) as err:
            raise ValueError(
                "Version 1 targets must already be encoded as 0.0 and 1.0 when "
                "target_mapping is not provided."
            ) from err

    if encoded_values != {0.0, 1.0}:
        raise ValueError(
            "Version 1 requires both binary classes to be present and encoded "
            "as exactly 0.0 and 1.0. "
            f"Observed encoded values: {sorted(encoded_values)}"
        )

def _validate_version1_validation_target_contract(
    y_values: np.ndarray,
    *,
    target_mapping: Optional[Mapping[Any, Any]],
    input_name: str,
) -> None:
    """Validate provided-validation targets against the training contract.

    Parameters
    ----------
    y_values : numpy.ndarray
        Resolved provided-validation target vector before encoding.
    target_mapping : Mapping[Any, Any], optional
        Mapping established for the training target.
    input_name : str
        Human-readable input name used in errors.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If targets are missing, an observed label is absent from the mapping, or
        unmapped targets contain values outside ``0.0`` and ``1.0``.

    Notes
    -----
    Unlike training data, provided validation is not required to contain both
    classes. A legitimate external cohort may contain only one class, but every
    observed value must still follow the training target definition.
    """
    _raise_for_missing_target_values(y_values, input_name=input_name)
    _validate_labels_present_in_mapping(
        y_values,
        target_mapping=target_mapping,
    )

    # External validation may contain one or both binary classes; it only
    # needs to remain within the training target contract.
    if target_mapping is None:
        observed_labels = _observed_target_labels(y_values)
        try:
            observed_values = {float(label) for label in observed_labels}
        except (TypeError, ValueError) as err:
            raise ValueError(
                "Provided validation targets must use 0.0 and 1.0 when "
                "target_mapping is not provided."
            ) from err

        if not observed_values.issubset({0.0, 1.0}):
            raise ValueError(
                "Provided validation targets must use only 0.0 and 1.0. "
                f"Observed values: {sorted(observed_values)}"
            )

def _validate_stratification_feasibility(
    y_values: Union[pd.Series, np.ndarray, Sequence[Any]],
    *,
    validation_size: float,
) -> None:
    """Check whether a binary stratified split can preserve both classes.

    Parameters
    ----------
    y_values : pandas.Series, numpy.ndarray, or sequence
        Target values that will be passed to scikit-learn for stratification.
    validation_size : float
        Fraction of rows assigned to validation.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If either class has fewer than two observations or either resulting
        partition is too small to contain every class.

    Notes
    -----
    The calculation mirrors scikit-learn's effective validation-row rounding so a
    clear framework error is raised before ``train_test_split`` is called.
    """
    # Count the encoded classes that sklearn will actually stratify.
    y_series = pd.Series(y_values)
    class_counts = y_series.value_counts(dropna=False).to_dict()

    if len(class_counts) != 2 or min(class_counts.values()) < 2:
        raise ValueError(
            "Insufficient class counts for stratification: each binary class "
            f"must contain at least 2 observations. Counts: {class_counts}"
        )

    # Reproduce sklearn's effective validation row count so an impossible
    # split is rejected with a framework-level explanation first.
    n_samples = int(len(y_series))
    n_validation = int(np.ceil(n_samples * validation_size))
    n_training = n_samples - n_validation
    n_classes = len(class_counts)

    if n_validation < n_classes or n_training < n_classes:
        raise ValueError(
            "Insufficient split size for stratification: both training and "
            "validation partitions must have room for every binary class. "
            f"n_training={n_training}, n_validation={n_validation}, "
            f"n_classes={n_classes}."
        )


# =============================================================================
# 4. Feature and target input resolution
# =============================================================================

def _validate_pandas_feature_target_index_alignment(
    X: Any,
    y: Any,
    *,
    input_name: str,
) -> None:
    """Validate patient-level row alignment for pandas feature and target inputs.

    Parameters
    ----------
    X : Any
        Candidate feature input.
    y : Any
        Candidate target input.
    input_name : str
        Human-readable dataset name included in errors.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``X`` is a DataFrame, ``y`` is a Series or DataFrame, their lengths are
        equal, but their indices differ in values or order.

    Notes
    -----
    The comparison occurs before either object is reset or converted to NumPy.
    Equal row counts alone do not prove that each patient is paired with the right
    outcome.
    """
    # Index alignment is meaningful only when X carries a pandas index.
    if not isinstance(X, pd.DataFrame):
        return

    # NumPy and generic sequence targets have no independent pandas index.
    if not isinstance(y, (pd.Series, pd.DataFrame)):
        return

    # Let the existing row-count validation produce its specific error first.
    if len(X) != len(y):
        return

    # Compare both index values and order before either object is reset.
    # Compare both labels and order before either input is reset to RangeIndex.
    if not X.index.equals(y.index):
        raise ValueError(
            f"{input_name}: pandas feature and target indices must match exactly "
            "and be in the same order to preserve row alignment."
        )

def _resolve_feature_dataframe(
    X: Union[pd.DataFrame, np.ndarray],
    *,
    feature_names: Optional[Sequence[str]] = None,
    fallback_feature_names: Optional[Sequence[str]] = None,
    input_name: str = "X",
) -> pd.DataFrame:
    """Resolve a feature input into an independent DataFrame with explicit names.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Two-dimensional feature matrix.
    feature_names : sequence[str], optional
        Explicit names that override DataFrame columns when supplied.
    fallback_feature_names : sequence[str], optional
        Names used only for unnamed array inputs, typically provided validation.
    input_name : str, default="X"
        Human-readable name used in validation messages.

    Returns
    -------
    pandas.DataFrame
        Copied DataFrame with string-normalized, unique column names and a fresh
        zero-based index.

    Raises
    ------
    ValueError
        If the input is not two-dimensional, names are unavailable, the number of
        names does not match the matrix width, or duplicate names remain after
        string normalization.

    Notes
    -----
    Duplicate detection occurs after converting names to strings because raw names
    such as ``1`` and ``"1"`` would otherwise produce the same downstream feature
    identifier.
    """

    # Convert X to DataFrame while preserving DataFrame dtypes when possible.
    # Preserve DataFrame dtypes and values by copying the caller's table.
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
        n_cols = X_df.shape[1]

    else:
        # Convert array-like inputs once, then enforce the required 2D shape.
        X_values = np.asarray(X)

        if X_values.ndim != 2:
            raise ValueError(
                f"{input_name} must be 2D, but got array with shape {X_values.shape}."
            )

        n_cols = X_values.shape[1]
        X_df = pd.DataFrame(X_values)

    # Resolve column names.
    # Explicit names take priority because callers may intentionally override
    # DataFrame column labels with their standardized feature names.
    if feature_names is not None:
        resolved_feature_names = [str(name) for name in feature_names]

        if len(resolved_feature_names) != n_cols:
            raise ValueError(
                f"{input_name}: number of feature_names does not match number of columns. "
                f"{input_name} has {n_cols} columns, but feature_names has "
                f"{len(resolved_feature_names)} entries."
            )

    elif isinstance(X, pd.DataFrame):
        # Otherwise retain the DataFrame's own column identity.
        resolved_feature_names = [str(name) for name in X.columns]

    elif fallback_feature_names is not None:
        # Array-based validation data may safely inherit the already-resolved
        # training feature names when the widths match.
        resolved_feature_names = [str(name) for name in fallback_feature_names]

        if len(resolved_feature_names) != n_cols:
            raise ValueError(
                f"{input_name}: fallback_feature_names does not match number of columns. "
                f"{input_name} has {n_cols} columns, but fallback_feature_names has "
                f"{len(resolved_feature_names)} entries."
            )

    else:
        raise ValueError(
            f"{input_name}: feature_names must be provided when {input_name} "
            "is not a pandas DataFrame."
        )

    # Duplicate detection must follow string normalization: names such as 1 and
    # "1" are distinct Python objects but become the same downstream feature key.
    # Check duplicates after string conversion because values such as 1 and
    # "1" would otherwise appear distinct but create the same bundle key.
    duplicate_feature_names = (
        pd.Index(resolved_feature_names)[
            pd.Index(resolved_feature_names).duplicated(keep=False)
        ]
        .unique()
        .tolist()
    )

    if duplicate_feature_names:
        raise ValueError(
            f"{input_name}: duplicate feature names are not allowed after "
            "converting feature names to strings: "
            f"{duplicate_feature_names}"
        )

    # Assign the validated names and remove the source index only after any
    # pandas X/y alignment check has already passed.
    X_df.columns = resolved_feature_names
    X_df = X_df.reset_index(drop=True)

    return X_df

def _resolve_target_vector(
    y: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[Any]],
    *,
    input_name: str = "y",
) -> np.ndarray:
    """Resolve supported target inputs into a one-dimensional NumPy vector.

    Parameters
    ----------
    y : pandas.Series, pandas.DataFrame, numpy.ndarray, or sequence
        Target input. A two-dimensional input is accepted only when it contains
        exactly one column.
    input_name : str, default="y"
        Human-readable name used in validation messages.

    Returns
    -------
    numpy.ndarray
        One-dimensional target values in their original row order.

    Raises
    ------
    ValueError
        If the input represents multiple target columns or otherwise cannot be
        interpreted as one supervised outcome vector.
    """

    # Accept a one-column DataFrame while rejecting true multi-output targets.
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(
                f"{input_name} must be 1D or a single-column DataFrame, "
                f"but got shape {y.shape}."
            )

        y_values = y.iloc[:, 0].to_numpy()

    elif isinstance(y, pd.Series):
        # Convert the Series without changing the value order established by its index.
        y_values = y.to_numpy()

    else:
        # Normalize other array-like targets to NumPy for consistent shape checks.
        y_values = np.asarray(y)

    # Flatten only the unambiguous (n, 1) case; never flatten multi-output data.
    if y_values.ndim == 2 and y_values.shape[1] == 1:
        y_values = y_values[:, 0]
    elif y_values.ndim != 1:
        raise ValueError(
            f"{input_name} must be a 1D target vector or contain exactly one "
            f"target column. Received shape {y_values.shape}; multi-output "
            "targets are not supported."
        )

    return y_values


# =============================================================================
# 5. Provided-validation feature alignment
# =============================================================================

def _align_train_validation_raw_features(
    X_train_df: pd.DataFrame,
    X_validation_df: pd.DataFrame,
    *,
    policy: str = "strict",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Align training and provided-validation features under an explicit policy.

    Parameters
    ----------
    X_train_df : pandas.DataFrame
        Resolved training feature DataFrame.
    X_validation_df : pandas.DataFrame
        Resolved provided-validation feature DataFrame.
    policy : {"strict", "intersection"}, default="strict"
        ``strict`` requires identical feature-name sets and only reorders validation
        columns. ``intersection`` retains shared features in training-column order
        and explicitly drops non-shared columns.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, dict[str, Any]]
        Aligned copies of the training and validation DataFrames plus detailed
        feature-alignment metadata.

    Raises
    ------
    ValueError
        If duplicate columns make alignment ambiguous, strict sets differ, or the
        intersection policy finds no shared features.
    TypeError
        If the policy is not a string.

    Notes
    -----
    Training-column order is always authoritative because downstream model columns
    must represent the same variables in the same positions for both datasets.
    """
    # Normalize the policy again at this boundary so the helper is safe when
    # called independently of the public preflight pipeline.
    normalized_policy = _normalize_validation_feature_policy(policy)

    # Capture original column order because training order defines model input order.
    train_cols = list(X_train_df.columns)
    validation_cols = list(X_validation_df.columns)

    train_duplicate_cols = (
        pd.Index(train_cols)[pd.Index(train_cols).duplicated(keep=False)]
        .unique()
        .tolist()
    )
    validation_duplicate_cols = (
        pd.Index(validation_cols)[pd.Index(validation_cols).duplicated(keep=False)]
        .unique()
        .tolist()
    )

    if train_duplicate_cols:
        raise ValueError(
            "Training features contain duplicate column names, which makes "
            f"provided validation alignment ambiguous: {train_duplicate_cols}"
        )
    if validation_duplicate_cols:
        raise ValueError(
            "Validation features contain duplicate column names, which makes "
            f"provided validation alignment ambiguous: {validation_duplicate_cols}"
        )

    # Use sets for membership checks while preserving ordered lists for output.
    train_col_set = set(train_cols)
    validation_col_set = set(validation_cols)
    common_cols = [col for col in train_cols if col in validation_col_set]
    train_only_cols = [col for col in train_cols if col not in validation_col_set]
    validation_only_cols = [col for col in validation_cols if col not in train_col_set]
    exact_feature_set_match = not train_only_cols and not validation_only_cols
    validation_reordered = validation_cols != train_cols

    # Training order is authoritative because downstream model columns must map
    # to the same variables at the same positions in validation data.
    if normalized_policy == "strict":
        # Strict mode permits reordering only; no feature may be added or removed.
        if not exact_feature_set_match:
            raise ValueError(
                "Strict validation feature policy requires identical feature-name "
                "sets in training and validation. "
                f"Missing from validation: {train_only_cols}. "
                f"Validation-only features: {validation_only_cols}."
            )

        # Strict mode preserves every feature and changes only validation order.
        # Preserve training order in both matrices so every model column has
        # the same meaning during fitting and external validation.
        X_train_aligned = X_train_df.loc[:, train_cols].copy()
        X_validation_aligned = X_validation_df.loc[:, train_cols].copy()
        retained_cols = train_cols
        train_only_dropped: List[str] = []
        validation_only_dropped: List[str] = []

    else:
        # Intersection mode is an explicit opt-in that intentionally removes
        # non-shared features from both datasets.
        if not common_cols:
            raise ValueError(
                "No overlapping raw feature columns were found between training "
                "and provided validation data under validation_feature_policy="
                "'intersection'."
            )

        # Intersection is an explicit opt-in because it intentionally drops
        # non-shared columns from both datasets.
        X_train_aligned = X_train_df.loc[:, common_cols].copy()
        X_validation_aligned = X_validation_df.loc[:, common_cols].copy()
        retained_cols = common_cols
        train_only_dropped = train_only_cols
        validation_only_dropped = validation_only_cols

    # Record both retained and excluded features so alignment is auditable.
    alignment_meta: Dict[str, Any] = {
        "enabled": True,
        "requested_policy": normalized_policy,
        "effective_policy": normalized_policy,
        "mode": normalized_policy,
        "exact_feature_set_match": exact_feature_set_match,
        "n_train_features_before": len(train_cols),
        "n_validation_features_before": len(validation_cols),
        "n_common_features": len(common_cols),
        "common_features": common_cols,
        "n_features_retained": len(retained_cols),
        "features_retained": retained_cols,
        "training_features_missing_from_validation": train_only_cols,
        "validation_only_features": validation_only_cols,
        "n_train_only_features_dropped": len(train_only_dropped),
        "train_only_features_dropped": train_only_dropped,
        "n_validation_only_features_dropped": len(validation_only_dropped),
        "validation_only_features_dropped": validation_only_dropped,
        "validation_reordered_to_match_train": validation_reordered,
    }

    return X_train_aligned, X_validation_aligned, alignment_meta

# =============================================================================
# 6. Coordinated preflight validation
# =============================================================================

def _resolve_and_validate_training_inputs(
    *,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[Any]],
    feature_names: Optional[Sequence[str]],
    target_name: str,
    target_mapping: Optional[Mapping[Any, Any]],
    split_config: Dict[str, Any],
    provided_validation: bool,
) -> Dict[str, Any]:
    """Resolve training inputs and complete data-level preflight validation.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Raw training features.
    y : pandas.Series, pandas.DataFrame, numpy.ndarray, or sequence
        Raw training target.
    feature_names : sequence[str], optional
        Explicit feature names.
    target_name : str
        Internal target-column name.
    target_mapping : Mapping[Any, Any], optional
        Version 1 target mapping.
    split_config : dict[str, Any]
        Validated internal split configuration.
    provided_validation : bool
        Whether a separate validation dataset will bypass internal splitting.

    Returns
    -------
    dict[str, Any]
        Resolved training DataFrame, target vector, feature names, and input
        metadata.

    Raises
    ------
    ValueError
        If row alignment, shape, names, target values, task contract, or
        stratification feasibility is invalid.

    Notes
    -----
    Stratification feasibility is checked here during preflight, using an encoded
    preview when a mapping is supplied, so execution does not discover a late
    split error.
    """
    # Validate patient-level pandas alignment before either index is reset.
    _validate_pandas_feature_target_index_alignment(
        X,
        y,
        input_name="training data",
    )

    # Resolve the feature matrix and its final raw feature names together.
    X_train_df = _resolve_feature_dataframe(
        X,
        feature_names=feature_names,
        input_name="X",
    )

    # Prevent the temporary target column from overwriting a real feature.
    if target_name in X_train_df.columns:
        raise ValueError(
            f"target_name={target_name!r} conflicts with an existing "
            "training feature column."
        )

    # Convert the target to one dimension only after pandas alignment is confirmed.
    y_train_values = _resolve_target_vector(
        y,
        input_name="y",
    )

    if len(y_train_values) != len(X_train_df):
        raise ValueError(
            f"X and y have different numbers of rows: "
            f"X has {len(X_train_df)} rows, y has {len(y_train_values)} rows."
        )

    # Enforce the binary source-label and 0.0/1.0 output contract up front.
    _validate_version1_training_target_contract(
        y_train_values,
        target_mapping=target_mapping,
        input_name="y",
    )

    # Validate stratification feasibility during preflight so the execution
    # pipeline can create the split without discovering a late configuration
    # problem. Use the encoded preview to preserve the established class-count
    # error messages.
    validation_size = split_config.get("validation_size", 0.2)
    stratify_enabled = split_config.get("stratify", True)
    has_internal_validation = (
        not provided_validation
        and validation_size is not None
        and validation_size > 0.0
    )

    if has_internal_validation and stratify_enabled:
        if target_mapping is not None:
            stratification_values = (
                pd.Series(y_train_values)
                .map(target_mapping)
                .to_numpy()
            )
        else:
            stratification_values = y_train_values

        _validate_stratification_feasibility(
            stratification_values,
            validation_size=float(validation_size),
        )

    # Preserve the final training feature order for validation alignment and metadata.
    resolved_feature_names = list(X_train_df.columns)
    input_meta: Dict[str, Any] = {
        "x_input_type": type(X).__name__,
        "y_input_type": type(y).__name__,
        "n_train_rows_input": int(X_train_df.shape[0]),
        "n_train_features_input": int(X_train_df.shape[1]),
        "feature_names": resolved_feature_names,
        "target_name": target_name,
    }

    return {
        "X_train_df": X_train_df,
        "y_train_values": y_train_values,
        "resolved_feature_names": resolved_feature_names,
        "input_meta": input_meta,
    }

def _resolve_and_validate_provided_validation_inputs(
    *,
    validation_config: Dict[str, Any],
    resolved_feature_names: Sequence[str],
    target_name: str,
    target_mapping: Optional[Mapping[Any, Any]],
) -> Dict[str, Any]:
    """Resolve and validate a caller-supplied external validation dataset.

    Parameters
    ----------
    validation_config : dict[str, Any]
        Validated dictionary containing validation ``X`` and ``y`` and optional
        feature names.
    resolved_feature_names : sequence[str]
        Training feature names available as a fallback for unnamed validation
        arrays.
    target_name : str
        Internal target-column name.
    target_mapping : Mapping[Any, Any], optional
        Mapping established for training outcomes.

    Returns
    -------
    dict[str, Any]
        Resolved validation DataFrame, target vector, and input metadata.

    Raises
    ------
    ValueError
        If validation rows, names, target shape, target labels, or pandas indices
        violate the established training contract.
    """
    # Apply the same row-alignment protection to the supplied validation data.
    _validate_pandas_feature_target_index_alignment(
        validation_config["X"],
        validation_config["y"],
        input_name="provided validation data",
    )

    # Resolve explicit validation names first, or inherit training names for
    # an unnamed array with the same width.
    X_validation_df = _resolve_feature_dataframe(
        validation_config["X"],
        feature_names=validation_config.get("feature_names", None),
        fallback_feature_names=resolved_feature_names,
        input_name="validation_kwargs['X']",
    )

    if target_name in X_validation_df.columns:
        raise ValueError(
            f"target_name={target_name!r} conflicts with an existing "
            "validation feature column."
        )

    y_validation_values = _resolve_target_vector(
        validation_config["y"],
        input_name="validation_kwargs['y']",
    )

    if len(y_validation_values) != len(X_validation_df):
        raise ValueError(
            "validation_kwargs['X'] and validation_kwargs['y'] have "
            f"different numbers of rows: X has {len(X_validation_df)} rows, "
            f"y has {len(y_validation_values)} rows."
        )

    # Validation labels must stay inside the training target contract, but
    # the external sample is allowed to contain only one observed class.
    _validate_version1_validation_target_contract(
        y_validation_values,
        target_mapping=target_mapping,
        input_name="validation_kwargs['y']",
    )

    validation_input_meta: Dict[str, Any] = {
        "x_input_type": type(validation_config["X"]).__name__,
        "y_input_type": type(validation_config["y"]).__name__,
        "n_validation_rows_input": int(X_validation_df.shape[0]),
        "n_validation_features_input": int(X_validation_df.shape[1]),
        "validation_feature_names": list(X_validation_df.columns),
    }

    return {
        "X_validation_df": X_validation_df,
        "y_validation_values": y_validation_values,
        "validation_input_meta": validation_input_meta,
    }

def _align_preflight_validation_features(
    *,
    X_train_df: pd.DataFrame,
    X_validation_df: pd.DataFrame,
    validation_feature_policy: str,
) -> Dict[str, Any]:
    """Apply the selected feature policy to preflight-validated DataFrames.

    Parameters
    ----------
    X_train_df : pandas.DataFrame
        Trusted training feature DataFrame.
    X_validation_df : pandas.DataFrame
        Trusted provided-validation feature DataFrame.
    validation_feature_policy : str
        Normalized ``strict`` or ``intersection`` policy.

    Returns
    -------
    dict[str, Any]
        Aligned DataFrames, feature-alignment metadata, and the compact text used by
        the visible progress report.
    """
    # Delegate the actual strict/intersection logic to the focused alignment helper.
    (
        X_train_aligned,
        X_validation_aligned,
        validation_alignment_meta,
    ) = _align_train_validation_raw_features(
        X_train_df,
        X_validation_df,
        policy=validation_feature_policy,
    )

    # Build the compact text used by the existing progress output.
    alignment_detail = (
        f"policy={validation_alignment_meta['effective_policy']}; "
        f"train features {validation_alignment_meta['n_train_features_before']} -> "
        f"{validation_alignment_meta['n_features_retained']}; "
        f"validation features {validation_alignment_meta['n_validation_features_before']} -> "
        f"{validation_alignment_meta['n_features_retained']}; "
        f"train-only dropped="
        f"{validation_alignment_meta['n_train_only_features_dropped']}; "
        f"validation-only dropped="
        f"{validation_alignment_meta['n_validation_only_features_dropped']}"
    )

    return {
        "X_train_df": X_train_aligned,
        "X_validation_df": X_validation_aligned,
        "validation_alignment_meta": validation_alignment_meta,
        "alignment_detail": alignment_detail,
    }

def _run_preflight_validation(
    *,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[Any]],
    feature_names: Optional[Sequence[str]],
    target_name: str,
    target_mapping: Optional[Dict[Any, float]],
    dataset_metadata: Optional[Dict[str, Any]],
    split_kwargs: Optional[Dict[str, Any]],
    validation_kwargs: Optional[Dict[str, Any]],
    validation_feature_policy: Any,
    start_step: Callable[[str], None],
    ok_step: Callable[[str, Any], None],
    skip_step: Callable[[str, str], None],
    fail_step: Callable[[str, Exception], None],
) -> Dict[str, Any]:
    """Run all configuration and input validation before pipeline execution.

    Parameters
    ----------
    X, y
        Raw training feature and target inputs.
    feature_names : sequence[str], optional
        Explicit training feature names.
    target_name : str
        Internal target-column name.
    target_mapping : dict[Any, float], optional
        Explicit Version 1 target mapping.
    dataset_metadata : dict[str, Any], optional
        Dataset metadata used by task validation and returned metadata.
    split_kwargs : dict[str, Any], optional
        Internal split overrides.
    validation_kwargs : dict[str, Any], optional
        Optional external-validation inputs.
    validation_feature_policy : Any
        Requested external feature policy.
    start_step, ok_step, skip_step, fail_step : Callable
        Progress callbacks used to preserve the established detailed output.

    Returns
    -------
    dict[str, Any]
        Trusted and normalized configuration, training data, optional validation
        data, input metadata, and feature-alignment metadata.

    Raises
    ------
    TypeError
        If a typed configuration value is invalid.
    ValueError
        If any configuration, training input, validation input, target contract, or
        feature-alignment check fails.

    Notes
    -----
    This is the single preflight entry point called by the public pipeline. It
    coordinates smaller helpers rather than embedding every check in one long
    function, while retaining the same individual progress steps.
    """
    # Validate all caller-controlled configuration before touching the data.
    configuration = _resolve_and_validate_prepare_configuration(
        dataset_metadata=dataset_metadata,
        split_kwargs=split_kwargs,
        validation_kwargs=validation_kwargs,
        validation_feature_policy=validation_feature_policy,
    )

    # Resolve and fully validate training inputs as the first visible preflight step.
    training_inputs = _run_progress_step(
        "Resolve training inputs",
        lambda: _resolve_and_validate_training_inputs(
            X=X,
            y=y,
            feature_names=feature_names,
            target_name=target_name,
            target_mapping=target_mapping,
            split_config=configuration["split_config"],
            provided_validation=configuration["provided_validation"],
        ),
        start_step=start_step,
        ok_step=ok_step,
        fail_step=fail_step,
        display_value=lambda result: result["X_train_df"],
    )

    # Resolve external validation only when the paired X/y inputs were supplied.
    if configuration["provided_validation"]:
        validation_inputs = _run_progress_step(
            "Resolve provided validation inputs",
            lambda: _resolve_and_validate_provided_validation_inputs(
                validation_config=configuration["validation_config"],
                resolved_feature_names=training_inputs["resolved_feature_names"],
                target_name=target_name,
                target_mapping=target_mapping,
            ),
            start_step=start_step,
            ok_step=ok_step,
            fail_step=fail_step,
            display_value=lambda result: result["X_validation_df"],
        )
    else:
        # Keep a uniform dictionary shape so downstream code does not need
        # separate missing-key logic for internal-split mode.
        validation_inputs = {
            "X_validation_df": None,
            "y_validation_values": None,
            "validation_input_meta": None,
        }
        skip_step(
            "Resolve provided validation inputs",
            "validation_kwargs with X and y was not provided",
        )

    # Feature alignment is the final preflight operation for external validation.
    if configuration["provided_validation"]:
        aligned_inputs = _run_progress_step(
            "Align provided validation features",
            lambda: _align_preflight_validation_features(
                X_train_df=training_inputs["X_train_df"],
                X_validation_df=validation_inputs["X_validation_df"],
                validation_feature_policy=(
                    configuration["validation_feature_policy"]
                ),
            ),
            start_step=start_step,
            ok_step=ok_step,
            fail_step=fail_step,
            display_value=lambda result: result["alignment_detail"],
        )

        X_train_df = aligned_inputs["X_train_df"]
        X_validation_df = aligned_inputs["X_validation_df"]
        validation_alignment_meta = aligned_inputs[
            "validation_alignment_meta"
        ]
        resolved_feature_names = list(X_train_df.columns)
    else:
        X_train_df = training_inputs["X_train_df"]
        X_validation_df = None
        validation_alignment_meta = None
        resolved_feature_names = training_inputs["resolved_feature_names"]
        skip_step(
            "Align provided validation features",
            "provided validation data was not supplied",
        )

    # Return normalized, validated objects only; the execution phase can now
    # operate without rediscovering input problems.
    return {
        **configuration,
        "X_train_df": X_train_df,
        "y_train_values": training_inputs["y_train_values"],
        "resolved_feature_names": resolved_feature_names,
        "input_meta": training_inputs["input_meta"],
        "X_validation_df": X_validation_df,
        "y_validation_values": validation_inputs["y_validation_values"],
        "validation_input_meta": validation_inputs["validation_input_meta"],
        "validation_alignment_meta": validation_alignment_meta,
    }

# =============================================================================
# 7. Preparation execution helpers
# =============================================================================

def _build_raw_modeling_dataframes(
    *,
    preflight: Dict[str, Any],
    target_name: str,
) -> Dict[str, Any]:
    """Build feature-plus-target DataFrames from trusted preflight inputs.

    Parameters
    ----------
    preflight : dict[str, Any]
        Successful preflight result containing normalized training and optional
        validation inputs.
    target_name : str
        Name assigned to the temporary target column.

    Returns
    -------
    dict[str, Any]
        Training and optional validation modeling DataFrames plus their shapes.

    Notes
    -----
    Fresh copies and reset indices prevent later pipeline operations from mutating
    caller-owned objects or carrying ambiguous source indices into the split.
    """
    # Combine trusted raw features and targets only after preflight succeeds.
    train_full_df = preflight["X_train_df"].reset_index(drop=True).copy()
    train_full_df[target_name] = preflight["y_train_values"]

    if preflight["provided_validation"]:
        # Supplied validation bypasses internal splitting and receives its own
        # feature-plus-target modeling DataFrame.
        validation_full_df = (
            preflight["X_validation_df"].reset_index(drop=True).copy()
        )
        validation_full_df[target_name] = preflight["y_validation_values"]
    else:
        validation_full_df = None

    # Store compact shapes for progress reporting and later audit metadata.
    modeling_df_meta: Dict[str, Any] = {
        "train_full_df_shape": tuple(train_full_df.shape),
        "validation_full_df_shape": (
            tuple(validation_full_df.shape)
            if validation_full_df is not None
            else None
        ),
    }

    return {
        "train_full_df": train_full_df,
        "validation_full_df": validation_full_df,
        "modeling_df_meta": modeling_df_meta,
    }

def _encode_modeling_targets(
    *,
    modeling_data: Dict[str, Any],
    target_name: str,
    target_mapping: Optional[Dict[Any, float]],
) -> Dict[str, Any]:
    """Encode training and optional validation targets consistently.

    Parameters
    ----------
    modeling_data : dict[str, Any]
        Feature-plus-target DataFrames created from preflight inputs.
    target_name : str
        Target column to encode.
    target_mapping : dict[Any, float], optional
        Explicit mapping applied identically to training and provided validation.

    Returns
    -------
    dict[str, Any]
        DataFrames containing encoded targets and target-encoding metadata for each
        available dataset.

    Notes
    -----
    When no mapping is supplied, the already-valid target values are retained and
    metadata records that no encoding was performed.
    """
    # Work with the DataFrames created by the immediately preceding pipeline step.
    train_full_df = modeling_data["train_full_df"]
    validation_full_df = modeling_data["validation_full_df"]

    if target_mapping is not None:
        # Apply one approved mapping consistently to training and supplied validation.
        train_full_df, train_target_meta = encode_target_labels(
            df=train_full_df,
            target_col=target_name,
            mapping=target_mapping,
            inplace=False,
            return_metadata=True,
        )

        if validation_full_df is not None:
            validation_full_df, validation_target_meta = encode_target_labels(
                df=validation_full_df,
                target_col=target_name,
                mapping=target_mapping,
                inplace=False,
                return_metadata=True,
            )
        else:
            validation_target_meta = None
    else:
        # Already encoded targets are retained unchanged while still producing
        # the metadata expected by downstream code.
        train_target_meta = {
            "target_col": target_name,
            "mapping": None,
            "encoded": False,
            "class_counts": train_full_df[target_name]
            .value_counts(dropna=False)
            .to_dict(),
        }

        if validation_full_df is not None:
            validation_target_meta = {
                "target_col": target_name,
                "mapping": None,
                "encoded": False,
                "class_counts": validation_full_df[target_name]
                .value_counts(dropna=False)
                .to_dict(),
            }
        else:
            validation_target_meta = None

    return {
        "train_full_df": train_full_df,
        "validation_full_df": validation_full_df,
        "train_target_meta": train_target_meta,
        "validation_target_meta": validation_target_meta,
    }

def _create_train_validation_dataframes(
    *,
    encoded_data: Dict[str, Any],
    preflight: Dict[str, Any],
    target_name: str,
) -> Dict[str, Any]:
    """Create provided-validation, internal-split, or train-only DataFrames.

    Parameters
    ----------
    encoded_data : dict[str, Any]
        Modeling DataFrames after target handling.
    preflight : dict[str, Any]
        Trusted configuration indicating the requested validation mode.
    target_name : str
        Target column used for stratification and class-count metadata.

    Returns
    -------
    dict[str, Any]
        Final train and optional validation DataFrames plus split metadata.

    Notes
    -----
    A supplied validation dataset bypasses ``train_test_split`` completely.
    Internal splitting occurs only when no supplied validation data exists and
    ``validation_size`` is greater than zero.
    """
    train_full_df = encoded_data["train_full_df"]
    validation_full_df = encoded_data["validation_full_df"]
    split_config = preflight["split_config"]

    if preflight["provided_validation"]:
        # Supplied validation is already independent data, so it bypasses the
        # internal random split entirely.
        train_df = train_full_df.reset_index(drop=True)
        validation_df = validation_full_df.reset_index(drop=True)
        validation_mode = "provided_validation"

        split_meta: Dict[str, Any] = {
            "validation_mode": validation_mode,
            "has_validation": True,
            "internal_split_used": False,
            "provided_validation_used": True,
            "split_kwargs": split_config,
            "train_shape": tuple(train_df.shape),
            "validation_shape": tuple(validation_df.shape),
            "train_class_counts": train_df[target_name]
            .value_counts(dropna=False)
            .to_dict(),
            "validation_class_counts": validation_df[target_name]
            .value_counts(dropna=False)
            .to_dict(),
        }
    else:
        # Internal-split and train-only behavior are controlled by validation_size.
        validation_size = split_config.get("validation_size", 0.2)
        has_validation = (
            validation_size is not None
            and validation_size > 0.0
        )

        if has_validation:
            # Pass encoded targets to sklearn only when stratification was requested.
            stratify_values = (
                train_full_df[target_name]
                if split_config.get("stratify", True)
                else None
            )

            # Preflight has already confirmed that this split is statistically feasible.
            train_df, validation_df = train_test_split(
                train_full_df,
                test_size=float(validation_size),
                random_state=split_config.get("random_state", 42),
                stratify=stratify_values,
            )

            train_df = train_df.reset_index(drop=True)
            validation_df = validation_df.reset_index(drop=True)
            validation_mode = "internal_split"
        else:
            # Train-only mode retains every row and returns no validation bundle.
            train_df = train_full_df.reset_index(drop=True)
            validation_df = None
            validation_mode = "train_only"

        split_meta = {
            "validation_mode": validation_mode,
            "has_validation": validation_df is not None,
            "internal_split_used": validation_mode == "internal_split",
            "provided_validation_used": False,
            "validation_size": validation_size,
            "random_state": split_config.get("random_state", 42),
            "stratify": split_config.get("stratify", True),
            "train_shape": tuple(train_df.shape),
            "validation_shape": (
                tuple(validation_df.shape)
                if validation_df is not None
                else None
            ),
            "train_class_counts": train_df[target_name]
            .value_counts(dropna=False)
            .to_dict(),
            "validation_class_counts": (
                validation_df[target_name].value_counts(dropna=False).to_dict()
                if validation_df is not None
                else None
            ),
        }

    return {
        "train_df": train_df,
        "validation_df": validation_df,
        "split_meta": split_meta,
    }

def _separate_features_and_target(
    *,
    split_data: Dict[str, Any],
    target_name: str,
) -> Dict[str, Any]:
    """Separate target columns from the final raw modeling DataFrames.

    Parameters
    ----------
    split_data : dict[str, Any]
        Train and optional validation DataFrames after validation-mode selection.
    target_name : str
        Column to remove from the feature matrices.

    Returns
    -------
    dict[str, Any]
        Raw feature DataFrames, target Series objects, and the authoritative final
        feature-name order.
    """
    train_df = split_data["train_df"]
    validation_df = split_data["validation_df"]

    # Remove the temporary target column while preserving row order in each split.
    X_train_final_df = train_df.drop(columns=[target_name]).reset_index(drop=True)
    y_train_final = train_df[target_name].reset_index(drop=True)

    if validation_df is not None:
        # Apply the identical separation rule to the validation partition.
        X_validation_final_df = (
            validation_df.drop(columns=[target_name]).reset_index(drop=True)
        )
        y_validation_final = validation_df[target_name].reset_index(drop=True)
    else:
        X_validation_final_df = None
        y_validation_final = None

    # Training columns define the canonical raw feature order for both bundles.
    raw_feature_names = list(X_train_final_df.columns)

    return {
        "X_train_final_df": X_train_final_df,
        "y_train_final": y_train_final,
        "X_validation_final_df": X_validation_final_df,
        "y_validation_final": y_validation_final,
        "raw_feature_names": raw_feature_names,
    }

def _build_raw_bundles_and_metadata(
    *,
    preflight: Dict[str, Any],
    modeling_data: Dict[str, Any],
    encoded_data: Dict[str, Any],
    split_data: Dict[str, Any],
    separated_data: Dict[str, Any],
    target_name: str,
    target_mapping: Optional[Dict[Any, float]],
    dataset_metadata: Optional[Dict[str, Any]],
    progress_config: Dict[str, Any],
    progress_log: List[Dict[str, Any]],
    return_progress_log: bool,
    return_dataframes: bool,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]]:
    """Construct final raw bundles and complete preparation metadata.

    Parameters
    ----------
    preflight, modeling_data, encoded_data, split_data, separated_data : dict[str, Any]
        Outputs from the preceding preparation stages.
    target_name : str
        Target identifier stored in bundles and metadata.
    target_mapping : dict[Any, float], optional
        Mapping recorded for auditability.
    dataset_metadata : dict[str, Any], optional
        Caller dataset metadata. Independent deep copies are stored in each returned
        object.
    progress_config : dict[str, Any]
        Effective progress settings.
    progress_log : list[dict[str, Any]]
        Shared mutable progress log.
    return_progress_log : bool
        Whether to expose the progress log in ``prep_meta``.
    return_dataframes : bool
        Whether to expose intermediate audit DataFrames in ``prep_meta``.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any] or None, dict[str, Any]]
        Raw training bundle, optional raw validation bundle, and preparation
        metadata.

    Notes
    -----
    Dataset metadata is deep-copied separately for train, validation, and preparation
    metadata so later mutation of one returned object cannot silently change the
    others. The progress log intentionally retains its shared list reference so the
    final successful bundle step appears in ``prep_meta`` as in the tested contract.
    """
    train_df = split_data["train_df"]
    validation_df = split_data["validation_df"]
    split_meta = split_data["split_meta"]

    X_train_final_df = separated_data["X_train_final_df"]
    y_train_final = separated_data["y_train_final"]
    X_validation_final_df = separated_data["X_validation_final_df"]
    y_validation_final = separated_data["y_validation_final"]
    raw_feature_names = separated_data["raw_feature_names"]

    train_target_meta = encoded_data["train_target_meta"]
    validation_target_meta = encoded_data["validation_target_meta"]

    # Extend target metadata with the class counts observed in the final split.
    train_target_meta_out = {
        **train_target_meta,
        "class_counts_split": y_train_final.value_counts(dropna=False).to_dict(),
    }

    if validation_df is not None:
        # Internal validation is created after target encoding, so it inherits
        # the training target contract and adds its own split-specific counts.
        validation_target_base_meta = (
            validation_target_meta
            if validation_target_meta is not None
            else train_target_meta
        )

        validation_target_meta_out = {
            **validation_target_base_meta,
            "class_counts_split": y_validation_final.value_counts(
                dropna=False
            ).to_dict(),
        }
    else:
        validation_target_meta_out = None

    validation_alignment_meta = preflight["validation_alignment_meta"]
    validation_feature_policy = preflight["validation_feature_policy"]

    # Deep-copy dataset metadata independently into each returned object so a
    # later mutation in one bundle cannot silently alter the others.
    # Build a self-contained raw training bundle for later preprocessing stages.
    train_bundle: Dict[str, Any] = {
        "X_raw": X_train_final_df.copy(),
        "y": y_train_final.to_numpy(),
        "feature_names": raw_feature_names,
        "feature_name_to_idx": {
            name: i for i, name in enumerate(raw_feature_names)
        },
        "target_name": target_name,
        "split": "train",
        # Deep-copy nested metadata so later bundle edits cannot mutate the
        # caller input or the separately returned validation/preparation metadata.
        "dataset_metadata": deepcopy(dataset_metadata),
        "target_metadata": train_target_meta_out,
        "feature_encoding_metadata": None,
        "feature_name_sanitization": None,
        "validation_feature_alignment": validation_alignment_meta,
        "validation_feature_policy": validation_feature_policy,
        "is_raw_split": True,
        "is_encoded": False,
        "is_preprocessed": False,
    }

    if validation_df is not None:
        # Mirror the training contract for whichever validation mode was used.
        validation_bundle: Optional[Dict[str, Any]] = {
            "X_raw": X_validation_final_df.copy(),
            "y": y_validation_final.to_numpy(),
            "feature_names": raw_feature_names,
            "feature_name_to_idx": {
                name: i for i, name in enumerate(raw_feature_names)
            },
            "target_name": target_name,
            "split": "validation",
            "dataset_metadata": deepcopy(dataset_metadata),
            "target_metadata": validation_target_meta_out,
            "feature_encoding_metadata": None,
            "validation_feature_encoding_metadata": None,
            "feature_name_sanitization": None,
            "validation_feature_alignment": validation_alignment_meta,
            "validation_feature_policy": validation_feature_policy,
            "is_raw_split": True,
            "is_encoded": False,
            "is_preprocessed": False,
        }
    else:
        validation_bundle = None

    # Collect the complete audit trail separately from either modeling bundle.
    prep_meta: Dict[str, Any] = {
        "target_name": target_name,
        "target_mapping": target_mapping,
        "split_kwargs": preflight["split_config"],
        "validation_kwargs_used": preflight["provided_validation"],
        "validation_mode": split_meta["validation_mode"],
        "has_validation": validation_bundle is not None,
        "progress_kwargs": progress_config,
        "dataset_metadata": deepcopy(dataset_metadata),
        "input_metadata": preflight["input_meta"],
        "validation_input_metadata": preflight["validation_input_meta"],
        "validation_feature_alignment": validation_alignment_meta,
        "validation_feature_policy": validation_feature_policy,
        "modeling_dataframe_metadata": modeling_data["modeling_df_meta"],
        "split_metadata": split_meta,
        "train_shape_raw": tuple(X_train_final_df.shape),
        "validation_shape_raw": (
            tuple(X_validation_final_df.shape)
            if X_validation_final_df is not None
            else None
        ),
        "feature_names": raw_feature_names,
        "n_features": len(raw_feature_names),
        "train_target_metadata": train_target_meta_out,
        "validation_target_metadata": validation_target_meta_out,
        "feature_encoding_metadata": None,
        "validation_feature_encoding_metadata": None,
        "feature_name_sanitization": None,
        "is_raw_split": True,
        "is_encoded": False,
        "is_preprocessed": False,
    }

    if return_dataframes:
        # Expose intermediate frames only when the caller explicitly requests them.
        prep_meta["train_full_df"] = encoded_data["train_full_df"]
        prep_meta["validation_full_df"] = encoded_data["validation_full_df"]
        prep_meta["train_df"] = train_df
        prep_meta["validation_df"] = validation_df
        prep_meta["X_train_df"] = X_train_final_df
        prep_meta["X_validation_df"] = X_validation_final_df

    if return_progress_log:
        # The same mutable list is retained so the final successful bundle step,
        # recorded immediately after this helper returns, is included as before.
        prep_meta["progress_log"] = progress_log

    return train_bundle, validation_bundle, prep_meta

# =============================================================================
# 8. Public train/validation preparation pipeline
# =============================================================================


def prepare_train_validation_bundles(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[Any]],
    *,
    feature_names: Optional[Sequence[str]] = None,
    target_name: str = "target",
    target_mapping: Optional[Dict[Any, float]] = None,
    dataset_metadata: Optional[Dict[str, Any]] = None,
    split_kwargs: Optional[Dict[str, Any]] = None,
    validation_kwargs: Optional[Dict[str, Any]] = None,
    validation_feature_policy: Literal["strict", "intersection"] = "strict",
    progress_kwargs: Optional[Dict[str, Any]] = None,
    show_progress: bool = True,
    return_dataframes: bool = False,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]]:
    """Prepare raw training and optional validation bundles from feature and target data.

    The public function is an orchestration pipeline. It first runs coordinated
    preflight validation, then executes focused steps for DataFrame construction,
    target encoding, validation-mode selection, feature/target separation, and
    bundle construction. Feature cleaning and transformation are intentionally
    performed later by ``preprocess_train_validation_bundles`` in
    ``ml_data_preprocessing.py``.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Raw training feature matrix.
    y : pandas.Series, pandas.DataFrame, numpy.ndarray, or sequence
        Training target values.
    feature_names : sequence[str], optional
        Explicit feature names. When supplied, they override DataFrame columns and
        are required to match the feature width.
    target_name : str, default="target"
        Internal target-column name used during preparation.
    target_mapping : dict[Any, float], optional
        Explicit source-label mapping. Version 1 requires the resulting values to be
        exactly ``0.0`` and ``1.0``.
    dataset_metadata : dict[str, Any], optional
        Dataset-level metadata copied into returned bundles and preparation
        metadata.
    split_kwargs : dict[str, Any], optional
        Internal split settings. Supported keys are ``validation_size``,
        ``random_state``, and ``stratify``.
    validation_kwargs : dict[str, Any], optional
        Optional external validation data with keys ``X``, ``y``, and optional
        ``feature_names``. Providing validation data bypasses the internal split.
    validation_feature_policy : {"strict", "intersection"}, default="strict"
        Feature policy used only for provided validation. Strict mode requires the
        same normalized feature-name set and reorders validation to training order.
        Intersection mode must be requested explicitly and retains shared features.
    progress_kwargs : dict[str, Any], optional
        Overrides for progress printing, compact output descriptions, and returning
        the progress log.
    show_progress : bool, default=True
        Backward-compatible default controlling visible progress output.
    return_dataframes : bool, default=False
        Whether intermediate DataFrames should be retained in ``prep_meta`` for
        auditing.

    Returns
    -------
    train_bundle : dict[str, Any]
        Raw training features, encoded target, feature maps, metadata, and state
        flags.
    validation_bundle : dict[str, Any] or None
        Raw validation bundle for internal or provided validation, or ``None`` in
        train-only mode.
    prep_meta : dict[str, Any]
        Complete preparation, split, input, target, feature-alignment, and optional
        progress/audit metadata.

    Raises
    ------
    TypeError
        If a required typed setting is invalid or an input has an unsupported type.
    ValueError
        If configuration, row alignment, dimensions, feature names, target values,
        Version 1 contracts, stratification, or provided-validation alignment is
        invalid.

    Notes
    -----
    This function does not impute, encode feature columns, scale, clean, drop
    features, cap outliers, or sanitize final feature names. Those operations remain
    separate so train-fitted transformations can be transferred safely to
    validation data.
    """
    # Resolve progress settings before creating the callbacks used by every step.
    progress_config = _resolve_progress_configuration(
        progress_kwargs,
        show_progress=show_progress,
    )

    # Read the normalized progress choices once for the full pipeline run.
    progress_enabled = bool(progress_config.get("enabled", show_progress))
    show_output_shapes = bool(
        progress_config.get("show_output_shapes", True)
    )
    return_progress_log = bool(
        progress_config.get("return_progress_log", True)
    )

    # Create one shared progress log and callback set for preflight and execution.
    (
        progress_log,
        _describe_object,
        _start_step,
        _ok_step,
        _skip_step,
        _fail_step,
    ) = make_pipeline_progress_helpers(
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
    )

    # Preserve the established human-readable pipeline heading.
    if progress_enabled:
        print("Prepare raw train/validation bundles")
        print("------------------------------------")

    # ------------------------------------------------------------------
    # Phase 1: Validate and normalize every caller-controlled input.
    # ------------------------------------------------------------------
    preflight = _run_preflight_validation(
        X=X,
        y=y,
        feature_names=feature_names,
        target_name=target_name,
        target_mapping=target_mapping,
        dataset_metadata=dataset_metadata,
        split_kwargs=split_kwargs,
        validation_kwargs=validation_kwargs,
        validation_feature_policy=validation_feature_policy,
        start_step=_start_step,
        ok_step=_ok_step,
        skip_step=_skip_step,
        fail_step=_fail_step,
    )

    # ------------------------------------------------------------------
    # Phase 2: Execute the preparation pipeline using trusted inputs.
    # ------------------------------------------------------------------
    modeling_data = _run_progress_step(
        "Build raw modeling dataframe(s)",
        lambda: _build_raw_modeling_dataframes(
            preflight=preflight,
            target_name=target_name,
        ),
        start_step=_start_step,
        ok_step=_ok_step,
        fail_step=_fail_step,
        display_value=lambda result: result["modeling_df_meta"],
    )

    encoded_data = _run_progress_step(
        "Encode target labels",
        lambda: _encode_modeling_targets(
            modeling_data=modeling_data,
            target_name=target_name,
            target_mapping=target_mapping,
        ),
        start_step=_start_step,
        ok_step=_ok_step,
        fail_step=_fail_step,
        display_value=lambda result: result["train_target_meta"],
    )

    split_data = _run_progress_step(
        "Create train/validation dataframes",
        lambda: _create_train_validation_dataframes(
            encoded_data=encoded_data,
            preflight=preflight,
            target_name=target_name,
        ),
        start_step=_start_step,
        ok_step=_ok_step,
        fail_step=_fail_step,
        display_value=lambda result: result["split_meta"],
    )

    separated_data = _run_progress_step(
        "Separate features and target",
        lambda: _separate_features_and_target(
            split_data=split_data,
            target_name=target_name,
        ),
        start_step=_start_step,
        ok_step=_ok_step,
        fail_step=_fail_step,
        display_value=lambda result: result["X_train_final_df"],
    )

    # The final helper constructs both bundles and the preparation metadata.
    bundles = _run_progress_step(
        "Build raw bundles and metadata",
        lambda: _build_raw_bundles_and_metadata(
            preflight=preflight,
            modeling_data=modeling_data,
            encoded_data=encoded_data,
            split_data=split_data,
            separated_data=separated_data,
            target_name=target_name,
            target_mapping=target_mapping,
            dataset_metadata=dataset_metadata,
            progress_config=progress_config,
            progress_log=progress_log,
            return_progress_log=return_progress_log,
            return_dataframes=return_dataframes,
        ),
        start_step=_start_step,
        ok_step=_ok_step,
        fail_step=_fail_step,
        display_value=lambda result: result[0],
    )

    # Close the visible report only after every preparation step succeeds.
    if progress_enabled:
        print("------------------------------------")
        print("[OK] Pipeline complete")

    # Return the stable three-item contract: train bundle, optional validation
    # bundle, and preparation metadata.
    return bundles

# =============================================================================
# 9. Legacy compatibility helper
# =============================================================================

def prepare_dataset(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence[Any]],
    feature_names: Sequence[str],
    target_name: str = "target",
    validation_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare a legacy train/validation DataFrame split.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix.
    y : pandas.Series, pandas.DataFrame, numpy.ndarray, or sequence
        Target values aligned row-for-row with ``X``.
    feature_names : sequence[str]
        Names assigned to feature columns.
    target_name : str, default="target"
        Name assigned to the appended target column.
    validation_size : float, default=0.2
        Fraction of rows placed in validation.
    random_state : int, default=42
        Reproducibility seed passed to scikit-learn.
    stratify : bool, default=True
        Whether to preserve the target distribution across both partitions.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Reset-index training and validation DataFrames containing both features and
        the target column.

    Notes
    -----
    This helper is retained for backward compatibility. New raw-bundle workflows
    should use ``prepare_train_validation_bundles`` because it enforces the tested
    Version 1 validation and metadata contracts.
    """

    # Create a DataFrame from the feature matrix using the provided column names.
    df = pd.DataFrame(X, columns=feature_names)

    # Add the target values as a new column in the DataFrame.
    df[target_name] = y

    # Use the target column for stratification if requested.
    stratify_values = df[target_name] if stratify else None

    # Split the full dataset into training and validation DataFrames.
    train_df, validation_df = train_test_split(
        df,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify_values,
    )

    # Reset the training DataFrame index after splitting.
    train_df = train_df.reset_index(drop=True)

    # Reset the validation DataFrame index after splitting.
    validation_df = validation_df.reset_index(drop=True)

    # Return both prepared datasets.
    return train_df, validation_df