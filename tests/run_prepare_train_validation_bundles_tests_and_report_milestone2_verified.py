# MILESTONES 3 AND 4 - VERSION 1 TEST RUNNER
# Contract: 42 tests total = 42 normal tests + 0 documented known gaps.
# Default provided-validation feature policy: strict exact feature-name matching.

"""Human-auditable Milestones 3 and 4 tests for ``prepare_train_validation_bundles``.

This is a self-contained test and reporting file. It contains:

1. The complete pytest test suite.
2. A short docstring on every test function explaining its purpose.
3. Structured descriptions of what each test uses and expects.
4. A built-in report runner that records the actual output of every test.

Recommended project location
----------------------------
<project_root>/tests/test_prepare_train_validation_bundles.py

Run from the project root
-------------------------
python tests/test_prepare_train_validation_bundles.py

Generated report
----------------
tests/test_reports/prepare_train_validation_bundles_detailed_report.txt

Version 1 scope: binary classification only. Regression and multiclass support are deferred to Version 2.
Provided validation defaults to strict, exact feature-name matching.

The report shows, for every test:
- what is being tested;
- why the test matters;
- the controlled test inputs;
- the expected behavior;
- the actual observed output;
- PASS, KNOWN GAP, FAIL, ERROR, or UNEXPECTED PASS status;
- a plain-language interpretation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

# Allow this file to be run directly from the tests folder while still importing
# the sibling ai_framework package from the project root.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import ai_framework.ml_data_preprocessing as mdp


# =============================================================================
# Human-readable test catalog
# =============================================================================

TEST_GROUPS: dict[str, str] = {
    "A. Core internal-split behavior": (
        "Checks the normal internal train/final-validation split used by the "
        "current Coimbra notebook workflow."
    ),
    "B. Provided external-validation behavior": (
        "Checks the branch where an external validation dataset is supplied "
        "instead of creating an internal split."
    ),
    "C. Input formats and error handling": (
        "Checks supported input forms and verifies that malformed inputs are "
        "rejected rather than silently converted."
    ),
    "D. Bundle and metadata integrity": (
        "Checks that returned bundles, intermediate DataFrames, feature maps, "
        "and metadata agree with the actual data and do not share unsafe state."
    ),
    "E. Version 1 binary-classification contract": (
        "Defines the explicit Version 1 task boundary: one binary target mapped "
        "to 0.0 and 1.0. Regression and multiclass tasks are deferred to Version 2."
    ),
    "F. Safety and configuration contracts": (
        "Checks permanent safety and configuration safeguards enforced by the production function."
    ),
}


def _case(
    *,
    number: int,
    title: str,
    group: str,
    purpose: str,
    why_it_matters: str,
    inputs: dict[str, Any],
    expected: dict[str, Any],
    pass_interpretation: str,
    gap_interpretation: str | None = None,
) -> dict[str, Any]:
    """Build one structured human-readable test definition."""
    return {
        "number": number,
        "title": title,
        "group": group,
        "purpose": purpose,
        "why_it_matters": why_it_matters,
        "inputs": inputs,
        "expected": expected,
        "pass_interpretation": pass_interpretation,
        "gap_interpretation": gap_interpretation,
    }


TEST_DEFINITIONS: dict[str, dict[str, Any]] = {
    "test_internal_stratified_split_builds_expected_raw_bundles": _case(
        number=1,
        title="Internal stratified split builds the expected raw bundles",
        group="A. Core internal-split behavior",
        purpose=(
            "Verify that a 20% stratified validation split produces correctly "
            "sized raw train and validation bundles with float target labels."
        ),
        why_it_matters=(
            "The train/final-validation split is the foundation for all later "
            "model evaluation. Incorrect sizes, labels, or flags would propagate "
            "through the entire framework."
        ),
        inputs={
            "rows": 20,
            "features": ["patient_id", "age", "biomarker"],
            "original_labels": [1, 2],
            "target_mapping": {1: 0.0, 2: 1.0},
            "validation_size": 0.20,
            "random_state": 42,
            "stratify": True,
        },
        expected={
            "train_shape": [16, 3],
            "validation_shape": [4, 3],
            "train_class_counts": {0.0: 8, 1.0: 8},
            "validation_class_counts": {0.0: 2, 1.0: 2},
            "target_dtype": "floating",
            "bundles_remain_raw": True,
        },
        pass_interpretation=(
            "The standard internal-split pathway produced correctly sized, "
            "stratified raw bundles with the intended float-label contract."
        ),
    ),
    "test_internal_split_is_reproducible": _case(
        number=2,
        title="The internal split is reproducible with a fixed random seed",
        group="A. Core internal-split behavior",
        purpose=(
            "Run the same split twice and confirm that the train rows, validation "
            "rows, and targets are identical."
        ),
        why_it_matters=(
            "Reproducibility is required for debugging, auditing, comparing model "
            "changes, and interpreting later performance differences."
        ),
        inputs={
            "rows": 20,
            "validation_size": 0.20,
            "random_state": 42,
            "stratify": True,
            "number_of_runs": 2,
        },
        expected={
            "train_features_identical_across_runs": True,
            "validation_features_identical_across_runs": True,
            "train_targets_identical_across_runs": True,
            "validation_targets_identical_across_runs": True,
        },
        pass_interpretation=(
            "Using the same random seed reproduced the exact same split and labels."
        ),
    ),
    "test_split_preserves_feature_target_row_alignment": _case(
        number=3,
        title="Feature rows remain aligned with their target labels after splitting",
        group="A. Core internal-split behavior",
        purpose=(
            "Verify every patient_id in both returned bundles still has the target "
            "label associated with that patient before the split."
        ),
        why_it_matters=(
            "A row-label misalignment would train the model using outcomes from the "
            "wrong patients and could invalidate the full analysis while still "
            "producing apparently reasonable output."
        ),
        inputs={
            "unique_row_identifier": "patient_id",
            "rows": 20,
            "target_mapping": {1: 0.0, 2: 1.0},
        },
        expected={
            "mismatched_patient_target_pairs": 0,
        },
        pass_interpretation=(
            "All split rows retained the correct patient-to-target relationship."
        ),
    ),
    "test_train_and_validation_are_disjoint_and_exhaustive": _case(
        number=4,
        title="Train and validation rows are disjoint and include every input row",
        group="A. Core internal-split behavior",
        purpose=(
            "Confirm no patient appears in both splits and no patient is lost or "
            "duplicated during splitting."
        ),
        why_it_matters=(
            "Overlap causes data leakage, while missing or duplicated patients alter "
            "the intended sample and class distribution."
        ),
        inputs={
            "input_patient_ids": "100 through 119",
            "rows": 20,
        },
        expected={
            "train_validation_overlap_count": 0,
            "combined_unique_patient_count": 20,
            "missing_patient_count": 0,
        },
        pass_interpretation=(
            "The split created two non-overlapping sets whose union exactly matched "
            "the original participants."
        ),
    ),
    "test_inputs_are_not_modified": _case(
        number=5,
        title="The function does not modify caller-owned inputs",
        group="A. Core internal-split behavior",
        purpose=(
            "Compare X, y, and nested dataset metadata before and after the function call."
        ),
        why_it_matters=(
            "Unexpected in-place mutation can change later notebook cells, create "
            "hard-to-reproduce state, and make repeated analyses inconsistent."
        ),
        inputs={
            "X_type": "pandas DataFrame",
            "y_type": "pandas Series",
            "metadata_contains_nested_dictionary": True,
        },
        expected={
            "X_unchanged": True,
            "y_unchanged": True,
            "dataset_metadata_unchanged": True,
        },
        pass_interpretation=(
            "The preparation function left all caller-owned inputs unchanged."
        ),
    ),
    "test_train_only_mode_with_zero_validation_size": _case(
        number=6,
        title="A zero validation size intentionally creates train-only mode",
        group="A. Core internal-split behavior",
        purpose=(
            "Confirm validation_size=0.0 keeps all rows in training and returns no "
            "validation bundle."
        ),
        why_it_matters=(
            "Train-only operation should be explicit and should not accidentally "
            "discard rows or create an empty validation object."
        ),
        inputs={
            "rows": 20,
            "validation_size": 0.0,
        },
        expected={
            "train_rows": 20,
            "validation_bundle": None,
            "validation_mode": "train_only",
        },
        pass_interpretation=(
            "The explicit train-only configuration retained every row and created no "
            "validation bundle."
        ),
    ),
    "test_provided_validation_uses_intersection_and_training_column_order": _case(
        number=7,
        title="Provided validation uses shared features in training-column order",
        group="B. Provided external-validation behavior",
        purpose=(
            "Verify that supplied validation data bypasses the internal split, keeps "
            "only shared raw columns, and reorders them to the training order."
        ),
        why_it_matters=(
            "External validation must use an explicitly aligned feature contract. "
            "Different feature order would apply values to the wrong model columns."
        ),
        inputs={
            "training_columns": ["age", "bmi", "train_only"],
            "validation_columns": ["validation_only", "bmi", "age"],
            "provided_validation_rows": 2,
            "split_kwargs_are_present_but_should_be_ignored": True,
        },
        expected={
            "shared_columns": ["age", "bmi"],
            "train_only_columns_dropped": ["train_only"],
            "validation_only_columns_dropped": ["validation_only"],
            "internal_split_used": False,
            "validation_mode": "provided_validation",
        },
        pass_interpretation=(
            "The external validation branch aligned shared columns correctly and did "
            "not create an additional internal split."
        ),
    ),
    "test_provided_validation_requires_both_X_and_y": _case(
        number=8,
        title="Provided validation requires both features and targets",
        group="B. Provided external-validation behavior",
        purpose=(
            "Check that supplying only validation X or only validation y raises an "
            "informative framework-level error."
        ),
        why_it_matters=(
            "A supervised validation bundle cannot be evaluated without both inputs. "
            "Failing early prevents partially defined validation state."
        ),
        inputs={
            "scenario_1": "validation y supplied without validation X",
            "scenario_2": "validation X supplied without validation y",
        },
        expected={
            "scenario_1_error": "validation target but no validation X",
            "scenario_2_error": "validation X but no validation target",
        },
        pass_interpretation=(
            "Both incomplete validation configurations were rejected with clear errors."
        ),
    ),
    "test_provided_validation_with_no_shared_features_raises": _case(
        number=9,
        title="Provided validation with no shared raw features is rejected",
        group="B. Provided external-validation behavior",
        purpose=(
            "Confirm alignment fails when training and validation have no overlapping "
            "feature names."
        ),
        why_it_matters=(
            "A model cannot be validated when there is no common feature space. "
            "Proceeding would create meaningless or empty inputs."
        ),
        inputs={
            "training_columns": ["a", "b"],
            "validation_columns": ["c", "d"],
        },
        expected={
            "error_type": "ValueError",
            "error_message_contains": "No overlapping raw feature columns",
        },
        pass_interpretation=(
            "The function correctly blocked validation when no common feature space existed."
        ),
    ),
    "test_unknown_target_label_raises": _case(
        number=10,
        title="A target label missing from the mapping is rejected",
        group="C. Input formats and error handling",
        purpose=(
            "Insert label 3 while the mapping only defines labels 1 and 2 and confirm "
            "the function raises an error."
        ),
        why_it_matters=(
            "Silently converting an unknown clinical outcome to missing or an unintended "
            "class would corrupt the target definition."
        ),
        inputs={
            "observed_labels": [1, 2, 3],
            "mapping_keys": [1, 2],
        },
        expected={
            "error_type": "ValueError",
            "error_message_contains": "target labels not present in mapping",
        },
        pass_interpretation=(
            "The target encoder refused to process a class that was not explicitly mapped."
        ),
    ),
    "test_X_y_row_count_mismatch_raises": _case(
        number=11,
        title="Different X and y row counts are rejected",
        group="C. Input formats and error handling",
        purpose=(
            "Provide 20 feature rows and 19 target values and verify an early error."
        ),
        why_it_matters=(
            "Every patient row requires exactly one target. A count mismatch makes row-level "
            "alignment undefined."
        ),
        inputs={
            "X_rows": 20,
            "y_values": 19,
        },
        expected={
            "error_type": "ValueError",
            "error_message_contains": "different numbers of rows",
        },
        pass_interpretation=(
            "The function detected and rejected the unequal feature and target lengths."
        ),
    ),
    "test_numpy_X_requires_feature_names": _case(
        number=12,
        title="NumPy feature matrices require explicit feature names",
        group="C. Input formats and error handling",
        purpose=(
            "Call the function with a NumPy X array and no feature_names argument."
        ),
        why_it_matters=(
            "Unnamed array columns cannot be audited, aligned to validation data, or "
            "reliably interpreted downstream."
        ),
        inputs={
            "X_type": "NumPy array",
            "feature_names": None,
        },
        expected={
            "error_type": "ValueError",
            "error_message_contains": "feature_names must be provided",
        },
        pass_interpretation=(
            "The function required explicit names before accepting an array feature matrix."
        ),
    ),
    "test_numpy_X_uses_explicit_feature_names": _case(
        number=13,
        title="NumPy feature matrices use the supplied feature names",
        group="C. Input formats and error handling",
        purpose=(
            "Supply an array with three explicit feature names and confirm those names "
            "appear in the returned DataFrames and bundle metadata."
        ),
        why_it_matters=(
            "Array inputs are only safe when the exact column identity and order are retained."
        ),
        inputs={
            "X_type": "NumPy array",
            "feature_names": ["patient_id", "age", "biomarker"],
        },
        expected={
            "returned_feature_names": ["patient_id", "age", "biomarker"],
            "returned_dataframe_column_order_matches": True,
        },
        pass_interpretation=(
            "The explicit array column names were preserved in the returned raw bundle."
        ),
    ),
    "test_single_column_target_dataframe_is_supported": _case(
        number=14,
        title="A single-column target DataFrame is supported",
        group="C. Input formats and error handling",
        purpose=(
            "Pass y as a one-column DataFrame and confirm it is resolved into a valid "
            "one-dimensional target."
        ),
        why_it_matters=(
            "Clinical loaders may return targets as Series or one-column DataFrames; both "
            "forms should work when they represent one outcome."
        ),
        inputs={
            "y_type": "pandas DataFrame",
            "y_shape": [20, 1],
        },
        expected={
            "train_target_length": 16,
            "validation_target_length": 4,
        },
        pass_interpretation=(
            "The one-column target table was safely converted to a one-dimensional target."
        ),
    ),
    "test_multi_column_target_dataframe_raises": _case(
        number=15,
        title="A multi-column target DataFrame is rejected",
        group="C. Input formats and error handling",
        purpose=(
            "Pass two target columns and verify the function refuses to guess how they "
            "should be combined."
        ),
        why_it_matters=(
            "This workflow currently supports a single supervised target. Flattening or "
            "selecting one column silently would change the modeling task."
        ),
        inputs={
            "y_type": "pandas DataFrame",
            "y_shape": [20, 2],
        },
        expected={
            "error_type": "ValueError",
            "error_message_contains": "single-column DataFrame",
        },
        pass_interpretation=(
            "The function correctly rejected a genuinely multi-output target table."
        ),
    ),
    "test_rejects_unsupported_regression_task_in_version_1": _case(
        number=16,
        title="Version 1 rejects regression tasks explicitly",
        group="E. Version 1 binary-classification contract",
        purpose=(
            "Provide dataset metadata declaring ml_task='regression' and require "
            "the preparation function to stop before splitting."
        ),
        why_it_matters=(
            "A successful data split could falsely imply that the full framework "
            "supports regression. Version 1 should state its binary-classification "
            "boundary clearly and defer regression to Version 2."
        ),
        inputs={
            "rows": 30,
            "ml_task": "regression",
            "stratify": False,
        },
        expected={
            "error_type": "ValueError",
            "error_message_contains": "Version 1 supports binary classification only",
        },
        pass_interpretation=(
            "The framework now rejects regression immediately and communicates the Version 1 scope."
        ),
        gap_interpretation=(
            "Known gap: the current function accepts regression data when stratification is disabled, "
            "even though the rest of Version 1 is classification-specific."
        ),
    ),
    "test_feature_name_to_idx_matches_dataframe_column_positions": _case(
        number=17,
        title="feature_name_to_idx matches the actual DataFrame positions",
        group="D. Bundle and metadata integrity",
        purpose=(
            "Rebuild the expected name-to-position dictionary from X_raw and compare it "
            "with both returned bundles."
        ),
        why_it_matters=(
            "Downstream code may use this mapping to retrieve feature columns. Any mismatch "
            "would return the wrong values."
        ),
        inputs={
            "features": ["patient_id", "age", "biomarker"],
        },
        expected={
            "feature_name_to_idx": {"patient_id": 0, "age": 1, "biomarker": 2},
        },
        pass_interpretation=(
            "The bundle feature-index mappings matched the actual train and validation columns."
        ),
    ),
    "test_return_dataframes_exposes_consistent_intermediate_outputs": _case(
        number=18,
        title="Returned intermediate DataFrames agree with the final bundles",
        group="D. Bundle and metadata integrity",
        purpose=(
            "Enable return_dataframes and compare the stored intermediate train and validation "
            "objects with X_raw and y in the final bundles."
        ),
        why_it_matters=(
            "Audit DataFrames are useful only if they represent the same rows and targets that "
            "are actually passed downstream."
        ),
        inputs={
            "return_dataframes": True,
            "validation_size": 0.20,
        },
        expected={
            "required_intermediate_keys_present": True,
            "X_train_df_equals_train_bundle_X_raw": True,
            "X_validation_df_equals_validation_bundle_X_raw": True,
            "stored_targets_equal_bundle_targets": True,
        },
        pass_interpretation=(
            "All optional audit DataFrames matched the data stored in the returned bundles."
        ),
    ),
    "test_progress_log_records_completed_and_skipped_steps": _case(
        number=19,
        title="The progress log records completed and skipped preparation steps",
        group="D. Bundle and metadata integrity",
        purpose=(
            "Inspect prep_meta['progress_log'] after an internal split and confirm the expected "
            "steps are marked ok or skipped."
        ),
        why_it_matters=(
            "The progress log is part of the audit trail. It should accurately describe what the "
            "pipeline did and did not execute."
        ),
        inputs={
            "provided_validation": False,
            "return_progress_log": True,
        },
        expected={
            "resolve_training_inputs": "ok",
            "resolve_provided_validation_inputs": "skipped",
            "align_provided_validation_features": "skipped",
            "encode_target_labels": "ok",
            "create_train_validation_dataframes": "ok",
            "build_raw_bundles_and_metadata": "ok",
        },
        pass_interpretation=(
            "The preparation audit log accurately distinguished executed and skipped steps."
        ),
    ),
    "test_rejects_target_name_collision_with_feature_column": _case(
        number=20,
        title="Reject a target_name that already exists as a feature column",
        group="F. Safety and configuration contracts",
        purpose=(
            "Use a feature named 'target' while target_name='target' and require an early error."
        ),
        why_it_matters=(
            "The current assignment can overwrite the original feature silently, changing the "
            "modeling data without warning."
        ),
        inputs={
            "feature_columns": ["target", "feature"],
            "target_name": "target",
        },
        expected={
            "error_type": "ValueError",
            "error_message_contains": "target_name conflict or already exists",
        },
        pass_interpretation=(
            "The production function now protects existing feature columns from target overwrite."
        ),
        gap_interpretation=(
            "Known gap: the current function overwrites an existing feature named 'target' instead "
            "of rejecting the collision."
        ),
    ),
    "test_rejects_duplicate_feature_names": _case(
        number=21,
        title="Reject duplicate feature names before building a bundle",
        group="F. Safety and configuration contracts",
        purpose=(
            "Supply three array columns named ['a', 'a', 'b'] and require an early error."
        ),
        why_it_matters=(
            "Duplicate names make feature_name_to_idx incomplete because a dictionary cannot "
            "represent two distinct columns under the same key."
        ),
        inputs={
            "X_shape": [20, 3],
            "feature_names": ["a", "a", "b"],
        },
        expected={
            "error_type": "ValueError",
            "error_message_contains": "duplicate feature or duplicate column",
        },
        pass_interpretation=(
            "The function now prevents internally inconsistent feature mappings."
        ),
        gap_interpretation=(
            "Known gap: duplicate feature names currently produce a bundle whose column count and "
            "feature_name_to_idx entry count disagree."
        ),
    ),
    "test_rejects_true_two_dimensional_target_array": _case(
        number=22,
        title="Reject a true two-dimensional multi-output target array",
        group="F. Safety and configuration contracts",
        purpose=(
            "Pass y with shape (10, 2) and require the function to reject it rather than flatten it."
        ),
        why_it_matters=(
            "Flattening a multi-output target can accidentally create a one-dimensional vector of "
            "the right length while changing the scientific meaning of the task."
        ),
        inputs={
            "X_rows": 20,
            "y_shape": [10, 2],
        },
        expected={
            "error_type": "ValueError",
            "error_message_contains": "target must be 1D, one column, or not multi-output",
        },
        pass_interpretation=(
            "The function now distinguishes a single target from a multi-output target array."
        ),
        gap_interpretation=(
            "Known gap: the current helper applies np.ravel and silently converts this multi-output "
            "array into a length-20 target."
        ),
    ),
    "test_rejects_negative_validation_size": _case(
        number=23,
        title="Reject a negative validation size",
        group="F. Safety and configuration contracts",
        purpose=(
            "Set validation_size=-0.20 and require a configuration error."
        ),
        why_it_matters=(
            "A negative fraction is invalid. Treating it as train-only can hide a notebook typo "
            "and remove the intended final-validation set."
        ),
        inputs={
            "validation_size": -0.20,
        },
        expected={
            "error_type": "ValueError",
            "valid_range": "0.0 <= validation_size < 1.0, or None by policy",
        },
        pass_interpretation=(
            "The split configuration now rejects impossible negative fractions."
        ),
        gap_interpretation=(
            "Known gap: a negative validation size is currently interpreted as train-only mode."
        ),
    ),
    "test_rejects_validation_size_equal_to_one_with_framework_message": _case(
        number=24,
        title="Reject validation_size=1.0 with a clear framework message",
        group="F. Safety and configuration contracts",
        purpose=(
            "Set validation_size=1.0 and require the framework to validate it before sklearn runs."
        ),
        why_it_matters=(
            "A full validation fraction leaves no training data. The user should receive a direct "
            "configuration explanation rather than a lower-level library error."
        ),
        inputs={
            "validation_size": 1.0,
        },
        expected={
            "error_type": "ValueError",
            "error_message_contains": "validation_size must be less than 1",
        },
        pass_interpretation=(
            "The framework now validates the split range before delegating to sklearn."
        ),
        gap_interpretation=(
            "Known gap: the current function relies on sklearn to reject validation_size=1.0."
        ),
    ),
    "test_rejects_unknown_split_configuration_key": _case(
        number=25,
        title="Reject an unknown split configuration key",
        group="F. Safety and configuration contracts",
        purpose=(
            "Use the misspelled key validation_sze and require an explicit configuration error."
        ),
        why_it_matters=(
            "Silently ignoring a typo causes the default validation fraction to be used while the "
            "notebook appears to request something else."
        ),
        inputs={
            "split_kwargs": {
                "validation_sze": 0.40,
                "random_state": 42,
                "stratify": True,
            },
        },
        expected={
            "error_type": "ValueError",
            "error_message_contains": "unknown or unsupported split key",
        },
        pass_interpretation=(
            "The framework now prevents silent fallback when a split option is misspelled."
        ),
        gap_interpretation=(
            "Known gap: unknown split_kwargs keys are currently ignored and defaults are used."
        ),
    ),
    "test_rejects_misaligned_pandas_indices": _case(
        number=26,
        title="Reject mismatched pandas X and y indices",
        group="F. Safety and configuration contracts",
        purpose=(
            "Provide X and y with equal lengths but different index order and require an error."
        ),
        why_it_matters=(
            "For patient-level pandas data, equal length does not guarantee that each target is "
            "paired with the correct patient row."
        ),
        inputs={
            "X_index": [100, 101, 102, 103],
            "y_index": [103, 100, 101, 102],
        },
        expected={
            "error_type": "ValueError",
            "error_message_contains": "indices must align or match",
        },
        pass_interpretation=(
            "The function now protects patient-level alignment when pandas indices disagree."
        ),
        gap_interpretation=(
            "Known gap: the current function discards both indices and pairs X and y by position."
        ),
    ),
    "test_rejects_missing_target_values_early": _case(
        number=27,
        title="Reject missing target values before splitting",
        group="F. Safety and configuration contracts",
        purpose=(
            "Insert one NaN target and require a clear framework-level error before sklearn is called."
        ),
        why_it_matters=(
            "Supervised outcomes must be defined. Missing targets should be handled by an explicit "
            "loader policy or rejected with an auditable message."
        ),
        inputs={
            "rows": 20,
            "missing_target_count": 1,
        },
        expected={
            "error_type": "ValueError",
            "error_message_contains": "missing target or target NaN",
        },
        pass_interpretation=(
            "The framework now catches undefined outcomes before attempting a split."
        ),
        gap_interpretation=(
            "Known gap: missing targets are currently rejected later by sklearn rather than by a "
            "clear framework validation step."
        ),
    ),
    "test_rejects_multiclass_target_in_version_1": _case(
        number=28,
        title="Version 1 rejects targets with more than two original classes",
        group="E. Version 1 binary-classification contract",
        purpose=(
            "Provide three observed target labels and require rejection even when "
            "a mapping collapses them into two numeric outputs."
        ),
        why_it_matters=(
            "Version 1 is designed for a genuinely binary clinical outcome. Collapsing "
            "three source classes through a mapping can hide a different scientific task."
        ),
        inputs={
            "observed_target_labels": [1, 2, 3],
            "target_mapping": {1: 0.0, 2: 1.0, 3: 1.0},
        },
        expected={
            "error_type": "ValueError",
            "error_message_contains": "exactly two observed target classes",
        },
        pass_interpretation=(
            "The framework now enforces a genuinely binary source target for Version 1."
        ),
        gap_interpretation=(
            "Known gap: the current function accepts three original classes when the mapping "
            "collapses them into two encoded values."
        ),
    ),
    "test_feature_name_length_mismatch_raises": _case(
        number=29,
        title="Feature-name count must match the number of feature columns",
        group="C. Input formats and error handling",
        purpose=(
            "Provide a three-column NumPy matrix with only two feature names and require an error."
        ),
        why_it_matters=(
            "Every feature column must have exactly one auditable identity. A count mismatch makes "
            "column interpretation and validation alignment unreliable."
        ),
        inputs={"X_shape": [20, 3], "feature_names_count": 2},
        expected={"error_type": "ValueError", "error_message_contains": "feature_names does not match"},
        pass_interpretation=(
            "The function correctly rejected a feature-name list that did not match the matrix width."
        ),
    ),
    "test_rejects_duplicate_feature_names_after_string_conversion": _case(
        number=30,
        title="Reject duplicate feature names created by string conversion",
        group="F. Safety and configuration contracts",
        purpose=(
            "Use feature names 1 and '1', which become identical after normalization to strings, "
            "and require an early duplicate-name error."
        ),
        why_it_matters=(
            "Duplicate checks must occur after standardization. Otherwise apparently different raw "
            "names can create the same bundle key and an inconsistent feature map."
        ),
        inputs={"X_shape": [20, 2], "feature_names": [1, "1"]},
        expected={"error_type": "ValueError", "error_message_contains": "duplicate feature"},
        pass_interpretation=(
            "The function now detects collisions introduced while standardizing feature names."
        ),
        gap_interpretation=(
            "Known gap: names that become identical after str(...) are currently accepted."
        ),
    ),
    "test_matching_nondefault_pandas_indices_are_supported": _case(
        number=31,
        title="Matching non-default pandas indices preserve row-target alignment",
        group="C. Input formats and error handling",
        purpose=(
            "Provide X and y with the same non-default patient index and verify the resulting "
            "patient-target pairs remain correct."
        ),
        why_it_matters=(
            "Clinical tables often retain patient-specific indices. Matching indices should be accepted "
            "while still preserving the correct row-to-outcome relationship."
        ),
        inputs={"shared_index": [501, 503, 507, 509, 511, 513, 517, 519]},
        expected={"mismatched_patient_target_pairs": 0, "function_succeeds": True},
        pass_interpretation=(
            "Matching non-default pandas indices were accepted and row-target alignment was preserved."
        ),
    ),
    "test_provided_validation_X_y_row_count_mismatch_raises": _case(
        number=32,
        title="Provided validation X and y must have the same number of rows",
        group="B. Provided external-validation behavior",
        purpose=(
            "Provide three validation feature rows and two validation targets and require an early error."
        ),
        why_it_matters=(
            "Every external-validation participant requires exactly one outcome. Unequal lengths make "
            "validation row alignment undefined."
        ),
        inputs={"validation_X_rows": 3, "validation_y_values": 2},
        expected={"error_type": "ValueError", "error_message_contains": "different numbers of rows"},
        pass_interpretation=(
            "The function rejected the externally supplied validation data before building a bundle."
        ),
    ),
    "test_provided_validation_uses_same_target_mapping_as_training": _case(
        number=33,
        title="Provided validation targets use the same mapping as training targets",
        group="B. Provided external-validation behavior",
        purpose=(
            "Supply original labels 1 and 2 in both datasets and verify both bundles use "
            "the same 1->0.0 and 2->1.0 mapping."
        ),
        why_it_matters=(
            "Training and external validation must share one target definition. Different encoding "
            "would reverse or distort performance interpretation."
        ),
        inputs={"target_mapping": {1: 0.0, 2: 1.0}, "provided_validation": True},
        expected={"train_target_values": [0.0, 1.0], "validation_target_values": [0.0, 1.0]},
        pass_interpretation=(
            "Training and validation targets followed the same explicit binary mapping."
        ),
    ),
    "test_unknown_provided_validation_target_label_raises": _case(
        number=34,
        title="Unknown target labels in provided validation data are rejected",
        group="B. Provided external-validation behavior",
        purpose=(
            "Include label 3 only in the external validation target while the mapping defines 1 and 2."
        ),
        why_it_matters=(
            "An unseen validation outcome must not become missing or be silently reinterpreted, because "
            "that would corrupt external performance evaluation."
        ),
        inputs={"training_labels": [1, 2], "validation_labels": [1, 3], "mapping_keys": [1, 2]},
        expected={"error_type": "ValueError", "error_message_contains": "target labels not present in mapping"},
        pass_interpretation=(
            "The function rejected a validation class that was absent from the approved mapping."
        ),
    ),
    "test_rejects_nonboolean_stratify_setting": _case(
        number=35,
        title="The stratify setting must be a real Boolean",
        group="F. Safety and configuration contracts",
        purpose=(
            "Pass the string 'False' instead of the Boolean False and require a configuration error."
        ),
        why_it_matters=(
            "Python treats non-empty strings as True. Without type validation, a notebook can request "
            "stratify='False' but still perform stratification."
        ),
        inputs={"stratify": "False", "input_type": "str"},
        expected={"error_type": "TypeError", "error_message_contains": "stratify must be a bool"},
        pass_interpretation=(
            "The framework now prevents truthiness conversion from changing the intended split behavior."
        ),
        gap_interpretation=(
            "Known gap: the current function converts non-empty strings to True through bool(...)."
        ),
    ),
    "test_too_small_class_for_stratification_raises_clear_framework_error": _case(
        number=36,
        title="Insufficient binary class counts produce a clear stratification error",
        group="F. Safety and configuration contracts",
        purpose=(
            "Use one positive patient and five negative patients with stratify=True and require an "
            "early framework explanation."
        ),
        why_it_matters=(
            "A class represented by one patient cannot be distributed safely across train and validation. "
            "The framework should explain the class-count limitation before sklearn fails."
        ),
        inputs={"class_counts": {"negative": 5, "positive": 1}, "stratify": True},
        expected={"error_type": "ValueError", "error_message_contains": "insufficient class counts for stratification"},
        pass_interpretation=(
            "The framework now validates class counts and gives an actionable split message."
        ),
        gap_interpretation=(
            "Known gap: the current function relies on sklearn's lower-level least-populated-class error."
        ),
    ),
    "test_rejects_target_mapping_outputs_other_than_zero_and_one": _case(
        number=37,
        title="Version 1 target mapping must encode classes as exactly 0.0 and 1.0",
        group="E. Version 1 binary-classification contract",
        purpose=(
            "Map the positive source class to 2.0 and require rejection."
        ),
        why_it_matters=(
            "Downstream Version 1 metrics, probability interpretation, calibration, and threshold logic "
            "assume a negative class of 0.0 and a positive class of 1.0."
        ),
        inputs={"target_mapping": {1: 0.0, 2: 2.0}},
        expected={"error_type": "ValueError", "error_message_contains": "mapping values must be exactly 0.0 and 1.0"},
        pass_interpretation=(
            "The function now enforces the shared Version 1 binary target convention."
        ),
        gap_interpretation=(
            "Known gap: the current function accepts arbitrary numeric mapping outputs."
        ),
    ),
    "test_internal_stratified_split_requires_both_binary_classes": _case(
        number=38,
        title="Internal stratified splitting requires both mapped binary classes",
        group="E. Version 1 binary-classification contract",
        purpose=(
            "Provide only the negative source label while the mapping defines both classes and "
            "require an early error."
        ),
        why_it_matters=(
            "A one-class dataset cannot train or validate a meaningful binary classifier, even if "
            "the mapping dictionary lists a second class that is absent from the data."
        ),
        inputs={"observed_source_labels": [1], "mapping": {1: 0.0, 2: 1.0}, "stratify": True},
        expected={"error_type": "ValueError", "error_message_contains": "both binary classes must be present"},
        pass_interpretation=(
            "The framework now blocks a nominally binary task that contains only one observed class."
        ),
        gap_interpretation=(
            "Known gap: the current function can split a one-class target and return apparently valid bundles."
        ),
    ),
    "test_returned_dataset_metadata_objects_are_independent": _case(
        number=39,
        title="Returned dataset metadata objects should not share mutable references",
        group="D. Bundle and metadata integrity",
        purpose=(
            "Mutate nested metadata in the returned train bundle and verify the original input, "
            "validation bundle, and prep metadata remain unchanged."
        ),
        why_it_matters=(
            "Shared mutable dictionaries allow one notebook step to change metadata everywhere, creating "
            "hidden state and making audit records unreliable."
        ),
        inputs={"metadata_contains_nested_dictionary": True, "mutation_target": "train_bundle"},
        expected={
            "original_metadata_unchanged": True,
            "validation_metadata_unchanged": True,
            "prep_metadata_unchanged": True,
        },
        pass_interpretation=(
            "Each returned object now owns an independent metadata copy."
        ),
        gap_interpretation=(
            "Known gap: the same dataset_metadata object is currently stored by reference in multiple outputs."
        ),
    ),
    "test_strict_validation_policy_is_default_and_reorders_columns": _case(
        number=40,
        title="Strict validation policy is the default and reorders columns safely",
        group="B. Provided external-validation behavior",
        purpose=(
            "Provide the same feature-name set in a different validation order and verify the "
            "default strict policy preserves every feature while reordering validation columns."
        ),
        why_it_matters=(
            "External validation must use the exact training feature space. Column order can be "
            "corrected safely, but no feature should be silently removed."
        ),
        inputs={
            "training_columns": ["age", "bmi", "biomarker"],
            "validation_columns": ["biomarker", "age", "bmi"],
            "validation_feature_policy": "strict (default)",
        },
        expected={
            "train_columns": ["age", "bmi", "biomarker"],
            "validation_columns_after_alignment": ["age", "bmi", "biomarker"],
            "features_dropped": 0,
            "validation_reordered": True,
        },
        pass_interpretation=(
            "The default strict policy preserved the complete feature set and corrected only column order."
        ),
    ),
    "test_strict_validation_policy_rejects_missing_and_extra_features": _case(
        number=41,
        title="Strict validation policy rejects missing and extra feature names",
        group="B. Provided external-validation behavior",
        purpose=(
            "Use one training-only feature and one validation-only feature and require a clear "
            "strict-policy error listing both differences."
        ),
        why_it_matters=(
            "A model cannot be evaluated safely when validation substitutes or omits training features."
        ),
        inputs={
            "training_columns": ["age", "bmi", "biomarker"],
            "validation_columns": ["age", "bmi", "site"],
            "validation_feature_policy": "strict",
        },
        expected={
            "error_type": "ValueError",
            "missing_from_validation": ["biomarker"],
            "validation_only_features": ["site"],
        },
        pass_interpretation=(
            "Strict alignment blocked a scientifically incompatible validation feature space."
        ),
    ),
    "test_rejects_unknown_validation_feature_policy": _case(
        number=42,
        title="Unknown validation feature policies are rejected",
        group="F. Safety and configuration contracts",
        purpose=(
            "Pass an unsupported validation_feature_policy value and require a direct configuration error."
        ),
        why_it_matters=(
            "A misspelled policy must not silently fall back to strict or intersection behavior."
        ),
        inputs={"validation_feature_policy": "automatic"},
        expected={
            "error_type": "ValueError",
            "supported_values": ["intersection", "strict"],
        },
        pass_interpretation=(
            "The framework rejected an unknown alignment policy before processing data."
        ),
    ),

}


# =============================================================================
# Reporting helpers used by every test
# =============================================================================


def _json_safe(value: Any) -> Any:
    """Convert pandas/NumPy values into JSON-safe plain Python objects."""
    if isinstance(value, dict):
        return {str(_json_safe(k)): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, pd.Index)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, pd.Series):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return {
            "shape": list(value.shape),
            "columns": [str(c) for c in value.columns],
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, float) and np.isnan(value):
        return "NaN"
    return value


def _record_json(record_property: Callable[[str, object], None], key: str, value: Any) -> None:
    """Store structured test information in a pytest report property."""
    record_property(key, json.dumps(_json_safe(value), sort_keys=False))


@pytest.fixture(autouse=True)
def _attach_human_readable_metadata(request, record_property):
    """Attach the static test-plan definition to every pytest result."""
    definition = TEST_DEFINITIONS.get(request.node.name)
    if definition is None:
        return

    for key in (
        "number",
        "title",
        "group",
        "purpose",
        "why_it_matters",
        "inputs",
        "expected",
        "pass_interpretation",
        "gap_interpretation",
    ):
        _record_json(record_property, key, definition.get(key))



def _record_actual(record_property, **actual: Any) -> None:
    """Record the concrete observed output before assertions are evaluated."""
    _record_json(record_property, "actual", actual)



def _exception_summary(error: BaseException | None) -> dict[str, Any]:
    """Return a compact human-readable description of an observed exception."""
    if error is None:
        return {
            "exception_raised": False,
            "exception_type": None,
            "exception_message": None,
        }
    return {
        "exception_raised": True,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
    }



def _call_and_capture(call: Callable[[], Any]) -> tuple[Any, BaseException | None]:
    """Execute a callable and return either its result or the raised exception."""
    try:
        return call(), None
    except BaseException as error:  # test helper: intentionally capture for reporting
        return None, error



def _assert_expected_error(
    *,
    error: BaseException | None,
    error_type: type[BaseException],
    message_pattern: str,
) -> None:
    """Assert an expected exception after its actual details have been recorded."""
    assert error is not None, "Expected an exception, but the function returned successfully."
    assert isinstance(error, error_type), (
        f"Expected {error_type.__name__}, but received {type(error).__name__}: {error}"
    )
    assert re.search(message_pattern, str(error), flags=re.IGNORECASE), (
        f"Exception message did not match /{message_pattern}/. Actual message: {error}"
    )


# =============================================================================
# Shared fixtures and execution helper
# =============================================================================


@pytest.fixture
def binary_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return deterministic binary data with a unique row identifier."""
    X = pd.DataFrame(
        {
            "patient_id": np.arange(100, 120),
            "age": np.arange(40, 60),
            "biomarker": np.linspace(0.10, 2.00, 20),
        }
    )
    y = pd.Series([1, 2] * 10, name="diagnosis")
    return X, y



def _prepare(X, y, **kwargs):
    """Call the production function with quiet but retained progress logging."""
    progress_kwargs = kwargs.pop(
        "progress_kwargs",
        {
            "enabled": False,
            "show_output_shapes": False,
            "return_progress_log": True,
        },
    )

    return mdp.prepare_train_validation_bundles(
        X=X,
        y=y,
        progress_kwargs=progress_kwargs,
        show_progress=False,
        **kwargs,
    )


# =============================================================================
# Baseline behavior tests
# =============================================================================



def test_internal_stratified_split_builds_expected_raw_bundles(
    binary_classification_data,
    record_property,
):
    """
    Verify that a 20% stratified validation split produces correctly sized raw train and validation bundles with float target labels.
    """
    X, y = binary_classification_data

    train_bundle, validation_bundle, prep_meta = _prepare(
        X,
        y,
        feature_names=list(X.columns),
        target_name="target",
        target_mapping={1: 0.0, 2: 1.0},
        dataset_metadata={"ml_task": "binary_classification"},
        split_kwargs={
            "validation_size": 0.20,
            "random_state": 42,
            "stratify": True,
        },
    )

    actual = {
        "validation_bundle_created": validation_bundle is not None,
        "train_shape": list(train_bundle["X_raw"].shape),
        "validation_shape": list(validation_bundle["X_raw"].shape),
        "train_feature_names": train_bundle["feature_names"],
        "validation_feature_names": validation_bundle["feature_names"],
        "train_target_values": sorted(np.unique(train_bundle["y"]).tolist()),
        "validation_target_values": sorted(np.unique(validation_bundle["y"]).tolist()),
        "train_target_dtype": str(train_bundle["y"].dtype),
        "validation_target_dtype": str(validation_bundle["y"].dtype),
        "train_class_counts": train_bundle["target_metadata"]["class_counts_split"],
        "validation_class_counts": validation_bundle["target_metadata"]["class_counts_split"],
        "validation_mode": prep_meta["validation_mode"],
        "train_flags": {
            "is_raw_split": train_bundle["is_raw_split"],
            "is_encoded": train_bundle["is_encoded"],
            "is_preprocessed": train_bundle["is_preprocessed"],
        },
    }
    _record_actual(record_property, **actual)

    assert validation_bundle is not None
    assert train_bundle["X_raw"].shape == (16, 3)
    assert validation_bundle["X_raw"].shape == (4, 3)
    assert isinstance(train_bundle["X_raw"], pd.DataFrame)
    assert isinstance(validation_bundle["X_raw"], pd.DataFrame)
    assert train_bundle["feature_names"] == list(X.columns)
    assert validation_bundle["feature_names"] == list(X.columns)
    assert set(np.unique(train_bundle["y"])) == {0.0, 1.0}
    assert set(np.unique(validation_bundle["y"])) == {0.0, 1.0}
    assert np.issubdtype(train_bundle["y"].dtype, np.floating)
    assert np.issubdtype(validation_bundle["y"].dtype, np.floating)
    assert train_bundle["split"] == "train"
    assert validation_bundle["split"] == "validation"
    assert train_bundle["is_raw_split"] is True
    assert train_bundle["is_encoded"] is False
    assert train_bundle["is_preprocessed"] is False
    assert prep_meta["validation_mode"] == "internal_split"
    assert prep_meta["has_validation"] is True
    assert prep_meta["train_shape_raw"] == (16, 3)
    assert prep_meta["validation_shape_raw"] == (4, 3)
    assert train_bundle["target_metadata"]["class_counts_split"] == {0.0: 8, 1.0: 8}
    assert validation_bundle["target_metadata"]["class_counts_split"] == {0.0: 2, 1.0: 2}



def test_internal_split_is_reproducible(binary_classification_data, record_property):
    """
    Run the same split twice and confirm that the train rows, validation rows, and targets are identical.
    """
    X, y = binary_classification_data
    common_kwargs = {
        "feature_names": list(X.columns),
        "target_mapping": {1: 0.0, 2: 1.0},
        "split_kwargs": {
            "validation_size": 0.20,
            "random_state": 42,
            "stratify": True,
        },
    }

    train_1, validation_1, _ = _prepare(X, y, **common_kwargs)
    train_2, validation_2, _ = _prepare(X, y, **common_kwargs)

    actual = {
        "run_1_train_patient_ids": train_1["X_raw"]["patient_id"].tolist(),
        "run_2_train_patient_ids": train_2["X_raw"]["patient_id"].tolist(),
        "run_1_validation_patient_ids": validation_1["X_raw"]["patient_id"].tolist(),
        "run_2_validation_patient_ids": validation_2["X_raw"]["patient_id"].tolist(),
        "train_rows_identical": train_1["X_raw"].equals(train_2["X_raw"]),
        "validation_rows_identical": validation_1["X_raw"].equals(validation_2["X_raw"]),
        "train_targets_identical": bool(np.array_equal(train_1["y"], train_2["y"])),
        "validation_targets_identical": bool(np.array_equal(validation_1["y"], validation_2["y"])),
    }
    _record_actual(record_property, **actual)

    assert_frame_equal(train_1["X_raw"], train_2["X_raw"])
    assert_frame_equal(validation_1["X_raw"], validation_2["X_raw"])
    np.testing.assert_array_equal(train_1["y"], train_2["y"])
    np.testing.assert_array_equal(validation_1["y"], validation_2["y"])



def test_split_preserves_feature_target_row_alignment(
    binary_classification_data,
    record_property,
):
    """
    Verify every patient_id in both returned bundles still has the target label associated with that patient before the split.
    """
    X, y = binary_classification_data
    expected_target_by_patient = {
        int(patient_id): float(mapped_target)
        for patient_id, mapped_target in zip(X["patient_id"], y.map({1: 0.0, 2: 1.0}))
    }

    train_bundle, validation_bundle, _ = _prepare(
        X,
        y,
        target_mapping={1: 0.0, 2: 1.0},
        split_kwargs={"validation_size": 0.20, "random_state": 42, "stratify": True},
    )

    mismatches: list[dict[str, Any]] = []
    checked_pairs = 0
    for split_name, bundle in (("train", train_bundle), ("validation", validation_bundle)):
        for patient_id, target in zip(bundle["X_raw"]["patient_id"], bundle["y"]):
            checked_pairs += 1
            expected_target = expected_target_by_patient[int(patient_id)]
            if float(target) != expected_target:
                mismatches.append(
                    {
                        "split": split_name,
                        "patient_id": int(patient_id),
                        "expected_target": expected_target,
                        "actual_target": float(target),
                    }
                )

    _record_actual(
        record_property,
        checked_patient_target_pairs=checked_pairs,
        mismatch_count=len(mismatches),
        mismatches=mismatches,
    )

    assert mismatches == []



def test_train_and_validation_are_disjoint_and_exhaustive(
    binary_classification_data,
    record_property,
):
    """
    Confirm no patient appears in both splits and no patient is lost or duplicated during splitting.
    """
    X, y = binary_classification_data
    train_bundle, validation_bundle, _ = _prepare(
        X,
        y,
        target_mapping={1: 0.0, 2: 1.0},
        split_kwargs={"validation_size": 0.20, "random_state": 42, "stratify": True},
    )

    train_ids = set(train_bundle["X_raw"]["patient_id"])
    validation_ids = set(validation_bundle["X_raw"]["patient_id"])
    original_ids = set(X["patient_id"])
    overlap = train_ids & validation_ids
    combined = train_ids | validation_ids
    missing = original_ids - combined
    extra = combined - original_ids

    _record_actual(
        record_property,
        train_unique_patient_count=len(train_ids),
        validation_unique_patient_count=len(validation_ids),
        overlap_count=len(overlap),
        overlapping_patient_ids=sorted(overlap),
        combined_unique_patient_count=len(combined),
        missing_patient_ids=sorted(missing),
        unexpected_patient_ids=sorted(extra),
    )

    assert train_ids.isdisjoint(validation_ids)
    assert combined == original_ids



def test_inputs_are_not_modified(binary_classification_data, record_property):
    """
    Compare X, y, and nested dataset metadata before and after the function call.
    """
    X, y = binary_classification_data
    X_before = X.copy(deep=True)
    y_before = y.copy(deep=True)
    metadata = {"ml_task": "binary_classification", "nested": {"a": 1}}
    metadata_before = deepcopy(metadata)

    _prepare(
        X,
        y,
        feature_names=list(X.columns),
        target_mapping={1: 0.0, 2: 1.0},
        dataset_metadata=metadata,
        split_kwargs={"validation_size": 0.20, "random_state": 42, "stratify": True},
    )

    actual = {
        "X_unchanged": X.equals(X_before),
        "y_unchanged": y.equals(y_before),
        "dataset_metadata_unchanged": metadata == metadata_before,
        "metadata_after_call": metadata,
    }
    _record_actual(record_property, **actual)

    assert_frame_equal(X, X_before)
    assert_series_equal(y, y_before)
    assert metadata == metadata_before



def test_train_only_mode_with_zero_validation_size(
    binary_classification_data,
    record_property,
):
    """
    Confirm validation_size=0.0 keeps all rows in training and returns no validation bundle.
    """
    X, y = binary_classification_data
    train_bundle, validation_bundle, prep_meta = _prepare(
        X,
        y,
        target_mapping={1: 0.0, 2: 1.0},
        split_kwargs={"validation_size": 0.0, "random_state": 42, "stratify": True},
    )

    _record_actual(
        record_property,
        train_shape=list(train_bundle["X_raw"].shape),
        train_target_length=len(train_bundle["y"]),
        validation_bundle_created=validation_bundle is not None,
        validation_mode=prep_meta["validation_mode"],
        has_validation=prep_meta["has_validation"],
    )

    assert validation_bundle is None
    assert train_bundle["X_raw"].shape == X.shape
    assert len(train_bundle["y"]) == len(y)
    assert prep_meta["validation_mode"] == "train_only"
    assert prep_meta["has_validation"] is False



def test_provided_validation_uses_intersection_and_training_column_order(record_property):
    """
    Verify that supplied validation data bypasses the internal split, keeps only shared raw columns, and reorders them to the training order.
    """
    X_train = pd.DataFrame(
        {
            "age": [40, 45, 50, 55, 60, 65],
            "bmi": [21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
            "train_only": [1, 2, 3, 4, 5, 6],
        }
    )
    y_train = pd.Series([1, 2, 1, 2, 1, 2])
    X_validation = pd.DataFrame(
        {
            "validation_only": [900, 901],
            "bmi": [27.0, 28.0],
            "age": [70, 75],
        }
    )
    y_validation = pd.Series([2, 1])

    train_bundle, validation_bundle, prep_meta = _prepare(
        X_train,
        y_train,
        target_mapping={1: 0.0, 2: 1.0},
        validation_kwargs={"X": X_validation, "y": y_validation},
        validation_feature_policy="intersection",
        split_kwargs={"validation_size": 0.50, "random_state": 999, "stratify": True},
    )

    alignment = prep_meta["validation_feature_alignment"]
    _record_actual(
        record_property,
        train_columns_after_alignment=list(train_bundle["X_raw"].columns),
        validation_columns_after_alignment=list(validation_bundle["X_raw"].columns),
        train_shape=list(train_bundle["X_raw"].shape),
        validation_shape=list(validation_bundle["X_raw"].shape),
        validation_age_values=validation_bundle["X_raw"]["age"].tolist(),
        validation_bmi_values=validation_bundle["X_raw"]["bmi"].tolist(),
        alignment_metadata=alignment,
        validation_mode=prep_meta["validation_mode"],
        internal_split_used=prep_meta["split_metadata"]["internal_split_used"],
    )

    assert validation_bundle is not None
    assert train_bundle["feature_names"] == ["age", "bmi"]
    assert validation_bundle["feature_names"] == ["age", "bmi"]
    assert list(train_bundle["X_raw"].columns) == ["age", "bmi"]
    assert list(validation_bundle["X_raw"].columns) == ["age", "bmi"]
    assert train_bundle["X_raw"].shape == (6, 2)
    assert validation_bundle["X_raw"].shape == (2, 2)
    assert validation_bundle["X_raw"]["age"].tolist() == [70, 75]
    assert validation_bundle["X_raw"]["bmi"].tolist() == [27.0, 28.0]
    assert alignment["common_features"] == ["age", "bmi"]
    assert alignment["train_only_features_dropped"] == ["train_only"]
    assert alignment["validation_only_features_dropped"] == ["validation_only"]
    assert prep_meta["validation_mode"] == "provided_validation"
    assert prep_meta["split_metadata"]["internal_split_used"] is False



def test_provided_validation_requires_both_X_and_y(
    binary_classification_data,
    record_property,
):
    """
    Check that supplying only validation X or only validation y raises an informative framework-level error.
    """
    X, y = binary_classification_data

    _, error_missing_X = _call_and_capture(
        lambda: _prepare(X, y, validation_kwargs={"y": pd.Series([1, 2])})
    )
    _, error_missing_y = _call_and_capture(
        lambda: _prepare(X, y, validation_kwargs={"X": X.iloc[:2].copy()})
    )

    _record_actual(
        record_property,
        missing_validation_X=_exception_summary(error_missing_X),
        missing_validation_y=_exception_summary(error_missing_y),
    )

    _assert_expected_error(
        error=error_missing_X,
        error_type=ValueError,
        message_pattern=r"validation target but no validation X",
    )
    _assert_expected_error(
        error=error_missing_y,
        error_type=ValueError,
        message_pattern=r"validation X but no validation target",
    )



def test_provided_validation_with_no_shared_features_raises(record_property):
    """
    Confirm alignment fails when training and validation have no overlapping feature names.
    """
    X_train = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    y_train = pd.Series([1, 2, 1, 2])
    X_validation = pd.DataFrame({"c": [9, 10], "d": [11, 12]})
    y_validation = pd.Series([1, 2])

    _, error = _call_and_capture(
        lambda: _prepare(
            X_train,
            y_train,
            target_mapping={1: 0.0, 2: 1.0},
            validation_kwargs={"X": X_validation, "y": y_validation},
            validation_feature_policy="intersection",
        )
    )
    _record_actual(record_property, **_exception_summary(error))
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"No overlapping raw feature columns",
    )



def test_unknown_target_label_raises(binary_classification_data, record_property):
    """
    Insert label 3 while the mapping only defines labels 1 and 2 and confirm the function raises an error.
    """
    X, y = binary_classification_data
    y_with_unknown = y.copy()
    y_with_unknown.iloc[0] = 3

    _, error = _call_and_capture(
        lambda: _prepare(X, y_with_unknown, target_mapping={1: 0.0, 2: 1.0})
    )
    _record_actual(
        record_property,
        observed_labels=sorted(y_with_unknown.unique().tolist()),
        mapping_keys=[1, 2],
        **_exception_summary(error),
    )
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"target labels not present in mapping",
    )



def test_X_y_row_count_mismatch_raises(binary_classification_data, record_property):
    """
    Provide 20 feature rows and 19 target values and verify an early error.
    """
    X, y = binary_classification_data
    _, error = _call_and_capture(lambda: _prepare(X, y.iloc[:-1]))
    _record_actual(
        record_property,
        X_rows=len(X),
        y_values=len(y.iloc[:-1]),
        **_exception_summary(error),
    )
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"different numbers of rows",
    )



def test_numpy_X_requires_feature_names(binary_classification_data, record_property):
    """
    Call the function with a NumPy X array and no feature_names argument.
    """
    X, y = binary_classification_data
    _, error = _call_and_capture(lambda: _prepare(X.to_numpy(), y))
    _record_actual(
        record_property,
        X_shape=list(X.to_numpy().shape),
        feature_names_supplied=False,
        **_exception_summary(error),
    )
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"feature_names must be provided",
    )



def test_numpy_X_uses_explicit_feature_names(binary_classification_data, record_property):
    """
    Supply an array with three explicit feature names and confirm those names appear in the returned DataFrames and bundle metadata.
    """
    X, y = binary_classification_data
    train_bundle, validation_bundle, _ = _prepare(
        X.to_numpy(),
        y.to_numpy(),
        feature_names=list(X.columns),
        target_mapping={1: 0.0, 2: 1.0},
        split_kwargs={"validation_size": 0.20, "random_state": 42, "stratify": True},
    )

    _record_actual(
        record_property,
        requested_feature_names=list(X.columns),
        train_feature_names=train_bundle["feature_names"],
        train_dataframe_columns=list(train_bundle["X_raw"].columns),
        validation_dataframe_columns=list(validation_bundle["X_raw"].columns),
    )

    assert validation_bundle is not None
    assert train_bundle["feature_names"] == list(X.columns)
    assert list(train_bundle["X_raw"].columns) == list(X.columns)



def test_single_column_target_dataframe_is_supported(
    binary_classification_data,
    record_property,
):
    """
    Pass y as a one-column DataFrame and confirm it is resolved into a valid one-dimensional target.
    """
    X, y = binary_classification_data
    y_df = y.to_frame(name="diagnosis")
    train_bundle, validation_bundle, _ = _prepare(
        X,
        y_df,
        target_mapping={1: 0.0, 2: 1.0},
        split_kwargs={"validation_size": 0.20, "random_state": 42, "stratify": True},
    )

    _record_actual(
        record_property,
        input_y_shape=list(y_df.shape),
        train_target_shape=list(train_bundle["y"].shape),
        validation_target_shape=list(validation_bundle["y"].shape),
        train_target_length=len(train_bundle["y"]),
        validation_target_length=len(validation_bundle["y"]),
    )

    assert validation_bundle is not None
    assert len(train_bundle["y"]) == 16
    assert len(validation_bundle["y"]) == 4



def test_multi_column_target_dataframe_raises(binary_classification_data, record_property):
    """
    Pass two target columns and verify the function refuses to guess how they should be combined.
    """
    X, y = binary_classification_data
    y_df = pd.DataFrame({"target_a": y, "target_b": y})
    _, error = _call_and_capture(lambda: _prepare(X, y_df))
    _record_actual(
        record_property,
        input_y_shape=list(y_df.shape),
        input_y_columns=list(y_df.columns),
        **_exception_summary(error),
    )
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"single-column DataFrame",
    )




def test_rejects_unsupported_regression_task_in_version_1(record_property):
    """Require Version 1 to reject an explicitly declared regression task before splitting."""
    X = pd.DataFrame(
        {
            "patient_id": np.arange(30),
            "feature": np.linspace(-1.0, 1.0, 30),
        }
    )
    y = pd.Series(np.linspace(0.5, 8.0, 30), name="outcome")

    result, error = _call_and_capture(
        lambda: _prepare(
            X,
            y,
            dataset_metadata={"ml_task": "regression"},
            split_kwargs={"validation_size": 0.20, "random_state": 42, "stratify": False},
        )
    )

    if result is not None:
        train_bundle, validation_bundle, prep_meta = result
        result_summary = {
            "function_returned_successfully": True,
            "train_shape": list(train_bundle["X_raw"].shape),
            "validation_shape": (
                list(validation_bundle["X_raw"].shape)
                if validation_bundle is not None
                else None
            ),
            "validation_mode": prep_meta["validation_mode"],
        }
    else:
        result_summary = {"function_returned_successfully": False}

    _record_actual(record_property, **result_summary, **_exception_summary(error))
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"Version 1.*binary classification only|binary classification.*Version 1",
    )


def test_feature_name_to_idx_matches_dataframe_column_positions(
    binary_classification_data,
    record_property,
):
    """
    Rebuild the expected name-to-position dictionary from X_raw and compare it with both returned bundles.
    """
    X, y = binary_classification_data
    train_bundle, validation_bundle, _ = _prepare(
        X,
        y,
        target_mapping={1: 0.0, 2: 1.0},
        split_kwargs={"validation_size": 0.20, "random_state": 42, "stratify": True},
    )

    actual_maps: dict[str, Any] = {}
    for split_name, bundle in (("train", train_bundle), ("validation", validation_bundle)):
        expected_map = {name: index for index, name in enumerate(bundle["X_raw"].columns)}
        actual_maps[split_name] = {
            "dataframe_columns": list(bundle["X_raw"].columns),
            "expected_mapping": expected_map,
            "actual_mapping": bundle["feature_name_to_idx"],
            "mapping_matches": bundle["feature_name_to_idx"] == expected_map,
        }
        assert bundle["feature_name_to_idx"] == expected_map

    _record_actual(record_property, bundles=actual_maps)



def test_return_dataframes_exposes_consistent_intermediate_outputs(
    binary_classification_data,
    record_property,
):
    """
    Enable return_dataframes and compare the stored intermediate train and validation objects with X_raw and y in the final bundles.
    """
    X, y = binary_classification_data
    train_bundle, validation_bundle, prep_meta = _prepare(
        X,
        y,
        target_mapping={1: 0.0, 2: 1.0},
        split_kwargs={"validation_size": 0.20, "random_state": 42, "stratify": True},
        return_dataframes=True,
    )

    expected_keys = {
        "train_full_df",
        "validation_full_df",
        "train_df",
        "validation_df",
        "X_train_df",
        "X_validation_df",
    }
    actual = {
        "required_keys": sorted(expected_keys),
        "keys_present": sorted(expected_keys.intersection(prep_meta.keys())),
        "missing_keys": sorted(expected_keys.difference(prep_meta.keys())),
        "X_train_matches_bundle": prep_meta["X_train_df"].equals(train_bundle["X_raw"]),
        "X_validation_matches_bundle": prep_meta["X_validation_df"].equals(validation_bundle["X_raw"]),
        "train_targets_match_bundle": bool(
            np.array_equal(prep_meta["train_df"]["target"].to_numpy(), train_bundle["y"])
        ),
        "validation_targets_match_bundle": bool(
            np.array_equal(
                prep_meta["validation_df"]["target"].to_numpy(), validation_bundle["y"]
            )
        ),
    }
    _record_actual(record_property, **actual)

    assert validation_bundle is not None
    assert expected_keys.issubset(prep_meta)
    assert_frame_equal(prep_meta["X_train_df"], train_bundle["X_raw"])
    assert_frame_equal(prep_meta["X_validation_df"], validation_bundle["X_raw"])
    np.testing.assert_array_equal(prep_meta["train_df"]["target"].to_numpy(), train_bundle["y"])
    np.testing.assert_array_equal(
        prep_meta["validation_df"]["target"].to_numpy(), validation_bundle["y"]
    )



def test_progress_log_records_completed_and_skipped_steps(
    binary_classification_data,
    record_property,
):
    """
    Inspect prep_meta['progress_log'] after an internal split and confirm the expected steps are marked ok or skipped.
    """
    X, y = binary_classification_data
    _, _, prep_meta = _prepare(
        X,
        y,
        target_mapping={1: 0.0, 2: 1.0},
        split_kwargs={"validation_size": 0.20, "random_state": 42, "stratify": True},
        progress_kwargs={
            "enabled": False,
            "show_output_shapes": True,
            "return_progress_log": True,
        },
    )

    progress_log = prep_meta["progress_log"]
    statuses_by_step = {item["step"]: item["status"] for item in progress_log}
    _record_actual(
        record_property,
        progress_log=progress_log,
        statuses_by_step=statuses_by_step,
    )

    assert statuses_by_step["Resolve training inputs"] == "ok"
    assert statuses_by_step["Resolve provided validation inputs"] == "skipped"
    assert statuses_by_step["Align provided validation features"] == "skipped"
    assert statuses_by_step["Encode target labels"] == "ok"
    assert statuses_by_step["Create train/validation dataframes"] == "ok"
    assert statuses_by_step["Build raw bundles and metadata"] == "ok"


# =============================================================================
# Safety and configuration contract tests
# =============================================================================


def test_rejects_target_name_collision_with_feature_column(record_property):
    """
    Use a feature named 'target' while target_name='target' and require an early error.
    """
    X = pd.DataFrame(
        {
            "target": np.arange(10),
            "feature": np.linspace(0.0, 1.0, 10),
        }
    )
    y = pd.Series([1, 2] * 5)

    result, error = _call_and_capture(
        lambda: _prepare(
            X,
            y,
            target_name="target",
            target_mapping={1: 0.0, 2: 1.0},
        )
    )

    if result is not None:
        train_bundle, validation_bundle, prep_meta = result
        result_summary = {
            "function_returned_successfully": True,
            "train_columns_after_call": list(train_bundle["X_raw"].columns),
            "validation_columns_after_call": (
                list(validation_bundle["X_raw"].columns)
                if validation_bundle is not None
                else None
            ),
            "original_feature_named_target_still_present": "target" in train_bundle["X_raw"].columns,
            "validation_mode": prep_meta["validation_mode"],
        }
    else:
        result_summary = {"function_returned_successfully": False}

    _record_actual(record_property, **result_summary, **_exception_summary(error))
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"target_name.*conflict|already exists",
    )


def test_rejects_duplicate_feature_names(record_property):
    """
    Supply three array columns named ['a', 'a', 'b'] and require an early error.
    """
    X = np.arange(60).reshape(20, 3)
    y = np.array([1, 2] * 10)

    result, error = _call_and_capture(
        lambda: _prepare(
            X,
            y,
            feature_names=["a", "a", "b"],
            target_mapping={1: 0.0, 2: 1.0},
        )
    )

    if result is not None:
        train_bundle, validation_bundle, _ = result
        result_summary = {
            "function_returned_successfully": True,
            "train_X_raw_column_count": train_bundle["X_raw"].shape[1],
            "train_feature_names": train_bundle["feature_names"],
            "train_feature_names_count": len(train_bundle["feature_names"]),
            "train_feature_name_to_idx": train_bundle["feature_name_to_idx"],
            "train_feature_name_to_idx_count": len(train_bundle["feature_name_to_idx"]),
            "validation_bundle_created": validation_bundle is not None,
        }
    else:
        result_summary = {"function_returned_successfully": False}

    _record_actual(record_property, **result_summary, **_exception_summary(error))
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"duplicate feature|duplicate column",
    )


def test_rejects_true_two_dimensional_target_array(record_property):
    """
    Pass y with shape (10, 2) and require the function to reject it rather than flatten it.
    """
    X = pd.DataFrame({"feature": np.arange(20)})
    y = np.array([[0, 1]] * 10)

    result, error = _call_and_capture(
        lambda: _prepare(X, y, split_kwargs={"validation_size": 0.0})
    )

    if result is not None:
        train_bundle, validation_bundle, _ = result
        result_summary = {
            "function_returned_successfully": True,
            "input_y_shape": list(y.shape),
            "returned_train_target_shape": list(train_bundle["y"].shape),
            "returned_train_target_values_preview": train_bundle["y"][:10].tolist(),
            "validation_bundle_created": validation_bundle is not None,
        }
    else:
        result_summary = {
            "function_returned_successfully": False,
            "input_y_shape": list(y.shape),
        }

    _record_actual(record_property, **result_summary, **_exception_summary(error))
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"target.*1D|multi-output|one column",
    )


def test_rejects_negative_validation_size(binary_classification_data, record_property):
    """
    Set validation_size=-0.20 and require a configuration error.
    """
    X, y = binary_classification_data
    result, error = _call_and_capture(
        lambda: _prepare(
            X,
            y,
            split_kwargs={"validation_size": -0.20, "random_state": 42, "stratify": True},
        )
    )

    if result is not None:
        train_bundle, validation_bundle, prep_meta = result
        result_summary = {
            "function_returned_successfully": True,
            "train_rows": train_bundle["X_raw"].shape[0],
            "validation_bundle_created": validation_bundle is not None,
            "validation_mode": prep_meta["validation_mode"],
            "recorded_validation_size": prep_meta["split_metadata"].get("validation_size"),
        }
    else:
        result_summary = {"function_returned_successfully": False}

    _record_actual(record_property, **result_summary, **_exception_summary(error))
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"validation_size.*between|validation_size.*0",
    )



def test_rejects_validation_size_equal_to_one_with_framework_message(
    binary_classification_data,
    record_property,
):
    """
    Set validation_size=1.0 and require the framework to validate it before sklearn runs.
    """
    X, y = binary_classification_data
    _, error = _call_and_capture(
        lambda: _prepare(
            X,
            y,
            split_kwargs={"validation_size": 1.0, "random_state": 42, "stratify": True},
        )
    )

    _record_actual(record_property, **_exception_summary(error))
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"validation_size.*less than 1",
    )


def test_rejects_unknown_split_configuration_key(
    binary_classification_data,
    record_property,
):
    """
    Use the misspelled key validation_sze and require an explicit configuration error.
    """
    X, y = binary_classification_data
    result, error = _call_and_capture(
        lambda: _prepare(
            X,
            y,
            split_kwargs={
                "validation_sze": 0.40,
                "random_state": 42,
                "stratify": True,
            },
        )
    )

    if result is not None:
        train_bundle, validation_bundle, prep_meta = result
        result_summary = {
            "function_returned_successfully": True,
            "train_rows": train_bundle["X_raw"].shape[0],
            "validation_rows": (
                validation_bundle["X_raw"].shape[0]
                if validation_bundle is not None
                else None
            ),
            "split_kwargs_recorded": prep_meta["split_kwargs"],
            "effective_validation_size": prep_meta["split_metadata"].get("validation_size"),
        }
    else:
        result_summary = {"function_returned_successfully": False}

    _record_actual(record_property, **result_summary, **_exception_summary(error))
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"unknown.*split_kwargs|unsupported.*split",
    )


def test_rejects_misaligned_pandas_indices(record_property):
    """
    Provide X and y with equal lengths but different index order and require an error.
    """
    X = pd.DataFrame(
        {"patient_id": [10, 11, 12, 13]},
        index=[100, 101, 102, 103],
    )
    y = pd.Series([1, 2, 1, 2], index=[103, 100, 101, 102])

    result, error = _call_and_capture(
        lambda: _prepare(
            X,
            y,
            target_mapping={1: 0.0, 2: 1.0},
            split_kwargs={"validation_size": 0.0},
        )
    )

    if result is not None:
        train_bundle, _, _ = result
        result_summary = {
            "function_returned_successfully": True,
            "original_X_index": X.index.tolist(),
            "original_y_index": y.index.tolist(),
            "returned_X_index": train_bundle["X_raw"].index.tolist(),
            "returned_patient_ids": train_bundle["X_raw"]["patient_id"].tolist(),
            "returned_targets": train_bundle["y"].tolist(),
        }
    else:
        result_summary = {
            "function_returned_successfully": False,
            "original_X_index": X.index.tolist(),
            "original_y_index": y.index.tolist(),
        }

    _record_actual(record_property, **result_summary, **_exception_summary(error))
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"index.*align|indices.*match",
    )



def test_rejects_missing_target_values_early(
    binary_classification_data,
    record_property,
):
    """
    Insert one NaN target and require a clear framework-level error before sklearn is called.
    """
    X, y = binary_classification_data
    y = y.astype(float)
    y.iloc[3] = np.nan

    _, error = _call_and_capture(
        lambda: _prepare(X, y, target_mapping={1.0: 0.0, 2.0: 1.0})
    )

    _record_actual(
        record_property,
        missing_target_count=int(y.isna().sum()),
        missing_target_positions=np.flatnonzero(y.isna().to_numpy()).tolist(),
        **_exception_summary(error),
    )
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"missing target|target.*NaN",
    )



def test_rejects_multiclass_target_in_version_1(record_property):
    """Require Version 1 to reject three observed source target classes before mapping and splitting."""
    X = pd.DataFrame({"feature": np.arange(18)})
    y = pd.Series([1, 2, 3] * 6)

    result, error = _call_and_capture(
        lambda: _prepare(
            X,
            y,
            target_mapping={1: 0.0, 2: 1.0, 3: 1.0},
            split_kwargs={"validation_size": 0.33, "random_state": 42, "stratify": True},
        )
    )

    if result is not None:
        train_bundle, validation_bundle, _ = result
        result_summary = {
            "function_returned_successfully": True,
            "source_target_values": sorted(y.unique().tolist()),
            "train_encoded_target_values": sorted(np.unique(train_bundle["y"]).tolist()),
            "validation_encoded_target_values": (
                sorted(np.unique(validation_bundle["y"]).tolist())
                if validation_bundle is not None
                else None
            ),
        }
    else:
        result_summary = {"function_returned_successfully": False}

    _record_actual(record_property, **result_summary, **_exception_summary(error))
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"exactly two.*target classes|binary.*two.*classes",
    )


def test_feature_name_length_mismatch_raises(record_property):
    """Verify that the feature-name list length must equal the NumPy matrix column count."""
    X = np.arange(60).reshape(20, 3)
    y = np.array([1, 2] * 10)
    _, error = _call_and_capture(
        lambda: _prepare(
            X,
            y,
            feature_names=["a", "b"],
            target_mapping={1: 0.0, 2: 1.0},
        )
    )
    _record_actual(
        record_property,
        X_shape=list(X.shape),
        feature_names=["a", "b"],
        **_exception_summary(error),
    )
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"feature_names.*does not match|3 columns.*2 entries",
    )


def test_rejects_duplicate_feature_names_after_string_conversion(record_property):
    """Require duplicate detection after feature names are normalized to strings."""
    X = np.arange(40).reshape(20, 2)
    y = np.array([1, 2] * 10)
    result, error = _call_and_capture(
        lambda: _prepare(
            X,
            y,
            feature_names=[1, "1"],
            target_mapping={1: 0.0, 2: 1.0},
        )
    )

    if result is not None:
        train_bundle, validation_bundle, _ = result
        result_summary = {
            "function_returned_successfully": True,
            "normalized_train_feature_names": train_bundle["feature_names"],
            "feature_name_to_idx": train_bundle["feature_name_to_idx"],
            "validation_bundle_created": validation_bundle is not None,
        }
    else:
        result_summary = {"function_returned_successfully": False}

    _record_actual(record_property, **result_summary, **_exception_summary(error))
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"duplicate feature|duplicate column",
    )


def test_matching_nondefault_pandas_indices_are_supported(record_property):
    """Verify matching non-default pandas indices preserve patient-to-target alignment."""
    shared_index = [501, 503, 507, 509, 511, 513, 517, 519]
    X = pd.DataFrame(
        {
            "patient_id": [101, 102, 103, 104, 105, 106, 107, 108],
            "feature": np.linspace(0.1, 0.8, 8),
        },
        index=shared_index,
    )
    y = pd.Series([1, 2, 1, 2, 1, 2, 1, 2], index=shared_index)
    expected_by_patient = {
        int(patient_id): float(target)
        for patient_id, target in zip(X["patient_id"], y.map({1: 0.0, 2: 1.0}))
    }

    train_bundle, validation_bundle, _ = _prepare(
        X,
        y,
        target_mapping={1: 0.0, 2: 1.0},
        split_kwargs={"validation_size": 0.25, "random_state": 42, "stratify": True},
    )

    mismatches = []
    for split_name, bundle in (("train", train_bundle), ("validation", validation_bundle)):
        for patient_id, observed_target in zip(bundle["X_raw"]["patient_id"], bundle["y"]):
            expected_target = expected_by_patient[int(patient_id)]
            if float(observed_target) != expected_target:
                mismatches.append(
                    {
                        "split": split_name,
                        "patient_id": int(patient_id),
                        "expected": expected_target,
                        "actual": float(observed_target),
                    }
                )

    _record_actual(
        record_property,
        input_X_index=X.index.tolist(),
        input_y_index=y.index.tolist(),
        output_indices_are_reset={
            "train": train_bundle["X_raw"].index.tolist(),
            "validation": validation_bundle["X_raw"].index.tolist(),
        },
        mismatch_count=len(mismatches),
        mismatches=mismatches,
    )
    assert mismatches == []


def test_provided_validation_X_y_row_count_mismatch_raises(record_property):
    """Verify externally provided validation features and targets must have equal row counts."""
    X_train = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    y_train = pd.Series([1, 2, 1, 2])
    X_validation = pd.DataFrame({"a": [9, 10, 11], "b": [12, 13, 14]})
    y_validation = pd.Series([1, 2])

    _, error = _call_and_capture(
        lambda: _prepare(
            X_train,
            y_train,
            target_mapping={1: 0.0, 2: 1.0},
            validation_kwargs={"X": X_validation, "y": y_validation},
        )
    )
    _record_actual(
        record_property,
        validation_X_rows=len(X_validation),
        validation_y_values=len(y_validation),
        **_exception_summary(error),
    )
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"different numbers of rows",
    )


def test_provided_validation_uses_same_target_mapping_as_training(record_property):
    """Verify the same explicit binary mapping is applied to train and provided validation targets."""
    X_train = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [10, 20, 30, 40, 50, 60]})
    y_train = pd.Series([1, 2, 1, 2, 1, 2])
    X_validation = pd.DataFrame({"a": [7, 8, 9, 10], "b": [70, 80, 90, 100]})
    y_validation = pd.Series([2, 1, 2, 1])

    train_bundle, validation_bundle, _ = _prepare(
        X_train,
        y_train,
        target_mapping={1: 0.0, 2: 1.0},
        validation_kwargs={"X": X_validation, "y": y_validation},
    )

    _record_actual(
        record_property,
        train_target_values=sorted(np.unique(train_bundle["y"]).tolist()),
        validation_target_values=sorted(np.unique(validation_bundle["y"]).tolist()),
        train_mapping=train_bundle["target_metadata"]["mapping"],
        validation_mapping=validation_bundle["target_metadata"]["mapping"],
        validation_targets_in_input_order=validation_bundle["y"].tolist(),
    )

    assert set(np.unique(train_bundle["y"])) == {0.0, 1.0}
    assert set(np.unique(validation_bundle["y"])) == {0.0, 1.0}
    assert validation_bundle["y"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert train_bundle["target_metadata"]["mapping"] == {1: 0.0, 2: 1.0}
    assert validation_bundle["target_metadata"]["mapping"] == {1: 0.0, 2: 1.0}


def test_unknown_provided_validation_target_label_raises(record_property):
    """Verify an external-validation target label absent from the mapping is rejected."""
    X_train = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    y_train = pd.Series([1, 2, 1, 2])
    X_validation = pd.DataFrame({"a": [9, 10], "b": [11, 12]})
    y_validation = pd.Series([1, 3])

    _, error = _call_and_capture(
        lambda: _prepare(
            X_train,
            y_train,
            target_mapping={1: 0.0, 2: 1.0},
            validation_kwargs={"X": X_validation, "y": y_validation},
        )
    )
    _record_actual(
        record_property,
        training_labels=sorted(y_train.unique().tolist()),
        validation_labels=sorted(y_validation.unique().tolist()),
        mapping_keys=[1, 2],
        **_exception_summary(error),
    )
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"target labels not present in mapping",
    )



def test_rejects_nonboolean_stratify_setting(binary_classification_data, record_property):
    """Require split_kwargs['stratify'] to be a Boolean rather than a truthy string."""
    X, y = binary_classification_data
    result, error = _call_and_capture(
        lambda: _prepare(
            X,
            y,
            target_mapping={1: 0.0, 2: 1.0},
            split_kwargs={"validation_size": 0.20, "random_state": 42, "stratify": "False"},
        )
    )

    if result is not None:
        train_bundle, validation_bundle, prep_meta = result
        result_summary = {
            "function_returned_successfully": True,
            "input_stratify_value": "False",
            "input_stratify_type": "str",
            "effective_recorded_stratify": prep_meta["split_metadata"]["stratify"],
            "train_rows": len(train_bundle["y"]),
            "validation_rows": len(validation_bundle["y"]) if validation_bundle else None,
        }
    else:
        result_summary = {"function_returned_successfully": False}

    _record_actual(record_property, **result_summary, **_exception_summary(error))
    _assert_expected_error(
        error=error,
        error_type=TypeError,
        message_pattern=r"stratify.*bool|Boolean.*stratify",
    )



def test_too_small_class_for_stratification_raises_clear_framework_error(record_property):
    """Require a clear framework error when one binary class has only one observation."""
    X = pd.DataFrame({"patient_id": np.arange(6), "feature": np.linspace(0.0, 1.0, 6)})
    y = pd.Series([1, 1, 1, 1, 1, 2])

    _, error = _call_and_capture(
        lambda: _prepare(
            X,
            y,
            target_mapping={1: 0.0, 2: 1.0},
            split_kwargs={"validation_size": 0.33, "random_state": 42, "stratify": True},
        )
    )
    _record_actual(
        record_property,
        class_counts=y.value_counts().sort_index().to_dict(),
        **_exception_summary(error),
    )
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"insufficient class counts.*stratification|stratification.*at least 2",
    )



def test_rejects_target_mapping_outputs_other_than_zero_and_one(
    binary_classification_data,
    record_property,
):
    """Require Version 1 target mappings to produce exactly the encoded values 0.0 and 1.0."""
    X, y = binary_classification_data
    result, error = _call_and_capture(
        lambda: _prepare(
            X,
            y,
            target_mapping={1: 0.0, 2: 2.0},
            split_kwargs={"validation_size": 0.20, "random_state": 42, "stratify": True},
        )
    )

    if result is not None:
        train_bundle, validation_bundle, _ = result
        result_summary = {
            "function_returned_successfully": True,
            "train_encoded_values": sorted(np.unique(train_bundle["y"]).tolist()),
            "validation_encoded_values": sorted(np.unique(validation_bundle["y"]).tolist()),
            "mapping": train_bundle["target_metadata"]["mapping"],
        }
    else:
        result_summary = {"function_returned_successfully": False}

    _record_actual(record_property, **result_summary, **_exception_summary(error))
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"mapping values.*0\.0.*1\.0|exactly.*0.*1",
    )



def test_internal_stratified_split_requires_both_binary_classes(record_property):
    """Require both encoded binary classes to be observed before an internal stratified split."""
    X = pd.DataFrame({"patient_id": np.arange(10), "feature": np.linspace(0.0, 1.0, 10)})
    y = pd.Series([1] * 10)

    result, error = _call_and_capture(
        lambda: _prepare(
            X,
            y,
            target_mapping={1: 0.0, 2: 1.0},
            split_kwargs={"validation_size": 0.20, "random_state": 42, "stratify": True},
        )
    )

    if result is not None:
        train_bundle, validation_bundle, _ = result
        result_summary = {
            "function_returned_successfully": True,
            "train_target_values": sorted(np.unique(train_bundle["y"]).tolist()),
            "validation_target_values": sorted(np.unique(validation_bundle["y"]).tolist()),
        }
    else:
        result_summary = {"function_returned_successfully": False}

    _record_actual(record_property, **result_summary, **_exception_summary(error))
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"both binary classes.*present|exactly two.*observed classes",
    )


def test_returned_dataset_metadata_objects_are_independent(
    binary_classification_data,
    record_property,
):
    """Require independent deep copies of dataset metadata in every returned object."""
    X, y = binary_classification_data
    metadata = {
        "ml_task": "binary_classification",
        "nested": {"clinical_context": "original"},
    }
    original_before = deepcopy(metadata)

    train_bundle, validation_bundle, prep_meta = _prepare(
        X,
        y,
        target_mapping={1: 0.0, 2: 1.0},
        dataset_metadata=metadata,
        split_kwargs={"validation_size": 0.20, "random_state": 42, "stratify": True},
    )

    train_bundle["dataset_metadata"]["nested"]["clinical_context"] = "mutated_in_train"

    actual = {
        "original_metadata_after_train_mutation": metadata,
        "validation_metadata_after_train_mutation": validation_bundle["dataset_metadata"],
        "prep_metadata_after_train_mutation": prep_meta["dataset_metadata"],
        "train_metadata_is_original_object": train_bundle["dataset_metadata"] is metadata,
        "validation_metadata_is_original_object": validation_bundle["dataset_metadata"] is metadata,
        "prep_metadata_is_original_object": prep_meta["dataset_metadata"] is metadata,
    }
    _record_actual(record_property, **actual)

    assert metadata == original_before
    assert validation_bundle["dataset_metadata"] == original_before
    assert prep_meta["dataset_metadata"] == original_before
    assert train_bundle["dataset_metadata"] is not metadata
    assert validation_bundle["dataset_metadata"] is not metadata
    assert prep_meta["dataset_metadata"] is not metadata


def test_strict_validation_policy_is_default_and_reorders_columns(record_property):
    """Verify the default strict policy preserves identical feature sets and reorders validation."""
    X_train = pd.DataFrame(
        {
            "age": [40, 45, 50, 55],
            "bmi": [21.0, 22.0, 23.0, 24.0],
            "biomarker": [0.1, 0.2, 0.3, 0.4],
        }
    )
    y_train = pd.Series([1, 2, 1, 2])
    X_validation = pd.DataFrame(
        {
            "biomarker": [0.5, 0.6],
            "age": [60, 65],
            "bmi": [25.0, 26.0],
        }
    )
    y_validation = pd.Series([2, 1])

    train_bundle, validation_bundle, prep_meta = _prepare(
        X_train,
        y_train,
        target_mapping={1: 0.0, 2: 1.0},
        validation_kwargs={"X": X_validation, "y": y_validation},
    )

    alignment = prep_meta["validation_feature_alignment"]
    actual = {
        "train_columns": list(train_bundle["X_raw"].columns),
        "validation_columns": list(validation_bundle["X_raw"].columns),
        "requested_policy": alignment["requested_policy"],
        "effective_policy": alignment["effective_policy"],
        "exact_feature_set_match": alignment["exact_feature_set_match"],
        "train_features_dropped": alignment["n_train_only_features_dropped"],
        "validation_features_dropped": alignment["n_validation_only_features_dropped"],
        "validation_reordered": alignment["validation_reordered_to_match_train"],
        "prep_policy": prep_meta["validation_feature_policy"],
    }
    _record_actual(record_property, **actual)

    assert list(train_bundle["X_raw"].columns) == ["age", "bmi", "biomarker"]
    assert list(validation_bundle["X_raw"].columns) == ["age", "bmi", "biomarker"]
    assert alignment["requested_policy"] == "strict"
    assert alignment["effective_policy"] == "strict"
    assert alignment["exact_feature_set_match"] is True
    assert alignment["n_train_only_features_dropped"] == 0
    assert alignment["n_validation_only_features_dropped"] == 0
    assert alignment["validation_reordered_to_match_train"] is True
    assert prep_meta["validation_feature_policy"] == "strict"


def test_strict_validation_policy_rejects_missing_and_extra_features(record_property):
    """Require strict alignment to list training-missing and validation-only features."""
    X_train = pd.DataFrame(
        {
            "age": [40, 45, 50, 55],
            "bmi": [21.0, 22.0, 23.0, 24.0],
            "biomarker": [0.1, 0.2, 0.3, 0.4],
        }
    )
    y_train = pd.Series([1, 2, 1, 2])
    X_validation = pd.DataFrame(
        {
            "age": [60, 65],
            "bmi": [25.0, 26.0],
            "site": [1, 2],
        }
    )
    y_validation = pd.Series([2, 1])

    _, error = _call_and_capture(
        lambda: _prepare(
            X_train,
            y_train,
            target_mapping={1: 0.0, 2: 1.0},
            validation_kwargs={"X": X_validation, "y": y_validation},
            validation_feature_policy="strict",
        )
    )
    _record_actual(record_property, **_exception_summary(error))
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"strict.*identical feature-name sets|Missing from validation.*biomarker",
    )
    assert "biomarker" in str(error)
    assert "site" in str(error)


def test_rejects_unknown_validation_feature_policy(
    binary_classification_data,
    record_property,
):
    """Reject unsupported validation feature-policy names before processing data."""
    X, y = binary_classification_data
    _, error = _call_and_capture(
        lambda: _prepare(
            X,
            y,
            target_mapping={1: 0.0, 2: 1.0},
            validation_feature_policy="automatic",
        )
    )
    _record_actual(record_property, **_exception_summary(error))
    _assert_expected_error(
        error=error,
        error_type=ValueError,
        message_pattern=r"Unknown validation_feature_policy|Supported values.*strict.*intersection|Supported values.*intersection.*strict",
    )


# =============================================================================
# Built-in human-readable report runner
# =============================================================================


@dataclass
class _ExecutionRecord:
    """One collected pytest outcome and its report properties."""

    test_name: str
    nodeid: str
    status: str
    properties: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    traceback: str = ""


@dataclass(eq=False)
class _ResultCollector:
    """Collect pytest outcomes when this file is run as a normal Python script."""

    results: dict[str, _ExecutionRecord] = field(default_factory=dict)

    @staticmethod
    def _parse_property_value(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    @classmethod
    def _properties_from_report(cls, report: Any) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        for key, value in getattr(report, "user_properties", []):
            properties[str(key)] = cls._parse_property_value(value)
        return properties

    @staticmethod
    def _is_strict_xpass(report: Any) -> bool:
        text = str(getattr(report, "longrepr", ""))
        return bool(report.failed and "XPASS(strict)" in text)

    def pytest_runtest_logreport(self, report: Any) -> None:
        test_name = report.nodeid.split("::")[-1]

        if report.when != "call":
            if report.failed:
                self.results[test_name] = _ExecutionRecord(
                    test_name=test_name,
                    nodeid=report.nodeid,
                    status="ERROR",
                    properties=self._properties_from_report(report),
                    reason="The test could not complete setup or teardown.",
                    traceback=getattr(report, "longreprtext", str(report.longrepr)),
                )
            return

        properties = self._properties_from_report(report)
        was_xfail = getattr(report, "wasxfail", None)

        if report.skipped and was_xfail:
            status = "KNOWN GAP"
            reason = str(was_xfail)
        elif self._is_strict_xpass(report):
            status = "UNEXPECTED PASS"
            reason = (
                "A strict known-gap test passed. Confirm the safeguard is truly "
                "implemented, then remove its xfail marker."
            )
        elif report.passed and was_xfail:
            status = "UNEXPECTED PASS"
            reason = str(was_xfail)
        elif report.passed:
            status = "PASS"
            reason = ""
        elif report.failed:
            status = "FAIL"
            reason = "The observed behavior did not satisfy the asserted contract."
        elif report.skipped:
            status = "SKIPPED"
            reason = "The test was skipped."
        else:
            status = str(report.outcome).upper()
            reason = ""

        traceback = ""
        if status in {"FAIL", "ERROR", "UNEXPECTED PASS"}:
            traceback = getattr(report, "longreprtext", str(report.longrepr))

        self.results[test_name] = _ExecutionRecord(
            test_name=test_name,
            nodeid=report.nodeid,
            status=status,
            properties=properties,
            reason=reason,
            traceback=traceback,
        )


def _report_scalar(value: Any) -> str:
    """Format one scalar value for the text report."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def _append_report_value(
    lines: list[str],
    value: Any,
    *,
    indent: int = 0,
    label: str | None = None,
) -> None:
    """Append nested data using readable indentation."""
    prefix = " " * indent

    if label is not None:
        if isinstance(value, (dict, list, tuple)):
            lines.append(f"{prefix}- {label}:")
            _append_report_value(lines, value, indent=indent + 2)
        else:
            lines.append(f"{prefix}- {label}: {_report_scalar(value)}")
        return

    if isinstance(value, dict):
        if not value:
            lines.append(f"{prefix}(none)")
            return
        for key, nested_value in value.items():
            readable_key = str(key).replace("_", " ")
            _append_report_value(
                lines,
                nested_value,
                indent=indent,
                label=readable_key,
            )
        return

    if isinstance(value, (list, tuple)):
        if not value:
            lines.append(f"{prefix}(none)")
            return
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                lines.append(f"{prefix}-")
                _append_report_value(lines, item, indent=indent + 2)
            else:
                lines.append(f"{prefix}- {_report_scalar(item)}")
        return

    lines.append(f"{prefix}{_report_scalar(value)}")


def _append_report_section(lines: list[str], title: str, value: Any) -> None:
    """Add a named section to one test's report entry."""
    lines.append("")
    lines.append(title)
    lines.append("-" * len(title))
    _append_report_value(lines, value)


def _ordered_test_names() -> list[str]:
    """Return documented test names in their intended human review order."""
    return sorted(
        TEST_DEFINITIONS,
        key=lambda name: int(TEST_DEFINITIONS[name]["number"]),
    )


def _execution_interpretation(
    *,
    status: str,
    definition: dict[str, Any],
    execution: _ExecutionRecord | None,
) -> str:
    """Return a plain-language interpretation for one test outcome."""
    if status == "PASS":
        return str(definition.get("pass_interpretation") or "The test passed.")
    if status == "KNOWN GAP":
        return str(
            definition.get("gap_interpretation")
            or (execution.reason if execution else "")
            or "The documented safety gap remains present."
        )
    if status == "UNEXPECTED PASS":
        return (
            "This known-gap test now passes. Confirm the production safeguard and "
            "then convert this into a normal permanent test."
        )
    if status == "FAIL":
        return (
            "This was not an expected known gap. Review the expected result, actual "
            "result, and traceback before changing production code."
        )
    if status == "ERROR":
        return (
            "The test could not complete. Resolve the setup, import, fixture, or "
            "execution problem before interpreting the target behavior."
        )
    if status == "NOT RUN":
        return "No pytest result was collected for this documented test."
    return execution.reason if execution and execution.reason else "No interpretation available."


def _write_detailed_report(
    *,
    output_file: Path,
    collector: _ResultCollector,
    pytest_exit_code: int,
) -> None:
    """Write one comprehensive report containing the full result of every test."""
    statuses = {
        "PASS": 0,
        "KNOWN GAP": 0,
        "FAIL": 0,
        "ERROR": 0,
        "UNEXPECTED PASS": 0,
        "SKIPPED": 0,
        "NOT RUN": 0,
    }

    for test_name in TEST_DEFINITIONS:
        execution = collector.results.get(test_name)
        status = execution.status if execution is not None else "NOT RUN"
        statuses[status] = statuses.get(status, 0) + 1

    action_required = (
        statuses["FAIL"]
        + statuses["ERROR"]
        + statuses["UNEXPECTED PASS"]
        + statuses["NOT RUN"]
    )

    if action_required:
        overall_status = "ACTION REQUIRED"
        overall_interpretation = (
            "At least one unexpected result requires review. Known gaps are listed "
            "separately and are not counted as new failures."
        )
    elif statuses["KNOWN GAP"]:
        overall_status = "PASS WITH DOCUMENTED KNOWN GAPS"
        overall_interpretation = (
            "All current-behavior tests passed. The known-gap tests failed in the "
            "expected way and document safeguards that have not yet been implemented."
        )
    else:
        overall_status = "PASS"
        overall_interpretation = "Every documented test passed and no known gaps remain."

    lines: list[str] = [
        "PREPARE TRAIN/VALIDATION BUNDLES — DETAILED HUMAN-READABLE TEST REPORT",
        "=" * 76,
        f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Project root: {_PROJECT_ROOT}",
        f"Python test/report file: {_THIS_FILE}",
        f"Pytest exit code: {pytest_exit_code}",
        "",
        f"OVERALL STATUS: {overall_status}",
        overall_interpretation,
        "",
        "SUMMARY",
        "-------",
        f"PASS: {statuses['PASS']}",
        f"KNOWN GAP: {statuses['KNOWN GAP']}",
        f"UNEXPECTED FAILURE: {statuses['FAIL']}",
        f"EXECUTION ERROR: {statuses['ERROR']}",
        f"UNEXPECTED PASS: {statuses['UNEXPECTED PASS']}",
        f"SKIPPED: {statuses['SKIPPED']}",
        f"NOT RUN: {statuses['NOT RUN']}",
        "",
        "HOW TO REVIEW EACH TEST",
        "-----------------------",
        "Each test below explicitly shows:",
        "1. What was tested.",
        "2. Why it matters.",
        "3. The controlled inputs used by the test.",
        "4. The expected behavior.",
        "5. The actual output observed during this run.",
        "6. The test status and its plain-language interpretation.",
        "",
        "VERSION 1 SCOPE",
        "---------------",
        "This test contract covers binary classification only.",
        "Regression and multiclass support are deferred to Version 2.",
        "",
        "STATUS DEFINITIONS",
        "------------------",
        "PASS: Actual output matched the intended current behavior.",
        "KNOWN GAP: The test intentionally demonstrates an unfixed weakness.",
        "FAIL: A normal test produced an unexpected result.",
        "ERROR: The test could not complete.",
        "UNEXPECTED PASS: A known-gap test now passes and needs review.",
    ]

    current_group: str | None = None
    for test_name in _ordered_test_names():
        definition = TEST_DEFINITIONS[test_name]
        execution = collector.results.get(test_name)
        group = str(definition["group"])

        if group != current_group:
            lines.extend(["", "", group, "=" * len(group)])
            lines.append(TEST_GROUPS.get(group, ""))
            current_group = group

        heading = f"TEST {int(definition['number']):02d} — {definition['title']}"
        lines.extend(["", heading, "-" * len(heading)])
        lines.append(f"Python test function: {test_name}")

        if execution is None:
            status = "NOT RUN"
            actual = {"message": "No pytest call result was collected."}
            reason = ""
            traceback = ""
        else:
            status = execution.status
            actual = execution.properties.get(
                "actual",
                {"message": "The test did not record an actual-result payload."},
            )
            reason = execution.reason
            traceback = execution.traceback

        interpretation = _execution_interpretation(
            status=status,
            definition=definition,
            execution=execution,
        )

        _append_report_section(lines, "What this test checks", definition["purpose"])
        _append_report_section(lines, "Why this test matters", definition["why_it_matters"])
        _append_report_section(lines, "Controlled test inputs", definition["inputs"])
        _append_report_section(lines, "Expected behavior", definition["expected"])
        _append_report_section(lines, "Actual observed output", actual)
        _append_report_section(lines, "Status", status)
        _append_report_section(lines, "Interpretation", interpretation)

        if reason and status in {
            "KNOWN GAP",
            "FAIL",
            "ERROR",
            "UNEXPECTED PASS",
            "SKIPPED",
        }:
            _append_report_section(lines, "Pytest reason", reason)

        if traceback and status in {"FAIL", "ERROR", "UNEXPECTED PASS"}:
            _append_report_section(lines, "Technical traceback", traceback)

    lines.extend(
        [
            "",
            "",
            "COVERAGE REASSESSMENT QUESTIONS",
            "===============================",
            "1. Does every important branch of prepare_train_validation_bundles have a test?",
            "2. Are the controlled inputs representative of clinical tabular datasets?",
            "3. Are there additional failure modes or metadata contracts we should test?",
            "4. Do the expected behaviors match the intended statistical and workflow design?",
            "5. Should any current behavior be changed before it becomes a permanent contract?",
            "",
            "NEXT STEP",
            "=========",
        ]
    )

    if action_required:
        lines.append(
            "Review all FAIL, ERROR, UNEXPECTED PASS, and NOT RUN entries before "
            "changing production code."
        )
    elif statuses["KNOWN GAP"]:
        lines.append(
            f"Review all {len(TEST_DEFINITIONS)} test entries for completeness. After the coverage is approved, "
            "retain this report as the verified Milestones 3 and 4 contract before broader integration testing."
        )
    else:
        lines.append(
            "All documented checks are green. Proceed to broader integration testing or the next function."
        )

    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the tests in this file and generate one detailed human-readable report."""
    report_dir = _THIS_FILE.parent / "test_reports"
    report_file = report_dir / "prepare_train_validation_bundles_detailed_report.txt"
    report_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(_PROJECT_ROOT)
    collector = _ResultCollector()

    print("Running prepare_train_validation_bundles tests...")
    print(f"Test/report file: {_THIS_FILE}")
    print("")

    pytest_exit_code = pytest.main(
        [
            str(_THIS_FILE),
            "-q",
            "-rxX",
            "--disable-warnings",
        ],
        plugins=[collector],
    )

    _write_detailed_report(
        output_file=report_file,
        collector=collector,
        pytest_exit_code=int(pytest_exit_code),
    )

    unexpected_count = sum(
        1
        for result in collector.results.values()
        if result.status in {"FAIL", "ERROR", "UNEXPECTED PASS"}
    )
    missing_count = len(set(TEST_DEFINITIONS) - set(collector.results))
    if int(pytest_exit_code) not in {0, 1} or missing_count:
        unexpected_count += 1

    known_gap_count = sum(
        1 for result in collector.results.values() if result.status == "KNOWN GAP"
    )

    print("")
    print("=" * 76)
    print("Detailed human-readable report created:")
    print(report_file)
    if unexpected_count:
        print("Overall status: ACTION REQUIRED")
    elif known_gap_count:
        print("Overall status: PASS WITH DOCUMENTED KNOWN GAPS")
    else:
        print("Overall status: PASS")
    print("=" * 76)

    # Documented known gaps do not cause a nonzero script exit status.
    return 1 if unexpected_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
