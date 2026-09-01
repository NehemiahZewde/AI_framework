"""
qEEG analysis, aggregation, plotting, and Part 1 pipeline utilities.

Milestone: Part 1 qEEG pipeline refactor v2
Scientific spectral calculations are preserved; orchestration and metadata flow
are consolidated into standardized analysis/preparation layers while plotting
remains intentionally explicit and user-controlled.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer


# =============================================================================
# MODULE VERSION AND SHARED qEEG PIPELINE CONFIGURATION
# =============================================================================
# Milestone: Part 1 qEEG pipeline refactor v2
#
# Goal:
#   preprocessing outputs -> one Part 1 pipeline call -> prepared result object
#   -> one plot-data preparation call -> explicit user-controlled plot calls
#
# Scientific PSD/power/ratio calculations remain unchanged from the validated
# development implementation. This milestone primarily cleans orchestration,
# metadata propagation, grouping, and plotting preparation.


# Metadata automatically propagated into PSD and metric tables.
DEFAULT_QEEG_METADATA_FIELDS: tuple[str, ...] = (
    "source_recording_id",
    "subject_id",
    "label",
    "condition",
    "analysis_condition",
    "eye_state",
    "cohort",
    "visit",
    "timepoint",
    "dose",
    "qc_idx",
)

DEFAULT_QEEG_GROUP_COLUMNS: tuple[str, ...] = (
    "eye_state",
    "timepoint",
)

def get_default_qeeg_part1_config() -> dict[str, Any]:
    """Return a fresh default configuration for the Part 1 qEEG pipeline."""
    return {
        "labels": None,                         # None -> analyze every available label
        "condition_to_eye_state": {"EO": "EO", "EC": "EC"},
        "psd_range_hz": (0.5, 45.0),           # PSD frequency range
        "total_range_hz": (1.0, 45.0),         # Relative-power denominator
        "picks": "eeg",                         # qEEG uses EEG channels only
        "bands": {
            "delta": (1.0, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 45.0),
        },

        # None preserves the original behavior: calculate relative power for
        # every configured band. Study-specific configs may provide a subset.
        "relative_power_bands": None,

        "ratio_definitions": {
            "theta_beta": ("theta", "beta"),
            "alpha_theta": ("alpha", "theta"),
            "alpha_beta": ("alpha", "beta"),
            "delta_alpha": ("delta", "alpha"),
        },
        "ratio_summary_method": "ratio_of_means",
        "psd_kwargs": {},                       # Optional Welch overrides
        "group_columns": DEFAULT_QEEG_GROUP_COLUMNS,
        "metadata_fields": DEFAULT_QEEG_METADATA_FIELDS,
        "topomap_ddof": 1,
        "log_mode": "summary",
        "progress_every": 1,
    }


def get_default_qeeg_plot_config() -> dict[str, Any]:
    """Return a fresh default configuration for Part 1 qEEG plots."""
    return {
        "condition_column": "eye_state",
        "timepoint_column": "timepoint",
        "preferred_condition_order": ["EO", "EC"],
        "condition_alias": {
            "EO": "Eyes Open",
            "EC": "Eyes Closed",
        },
        "condition_palette": {
            "EO": "#355C8A",
            "EC": "#C96B32",
        },
        "band_order": ["delta", "theta", "alpha", "beta", "gamma"],
        "band_alias": {
            "delta": "Delta",
            "theta": "Theta",
            "alpha": "Alpha",
            "beta": "Beta",
            "gamma": "Gamma",
        },
        "ratio_order": ["theta_beta", "alpha_theta"],
        "ratio_alias": {
            "theta_beta": "Theta/Beta",
            "alpha_theta": "Alpha/Theta",
        },
        "psd": {
            "xlim": (0.5, 45.0),
            "xticks": np.arange(5, 46, 5),
            "linewidth": 2.8,
            "fill_alpha": 0.20,
            "figsize": (9, 5),
            "font_size": 12,
        },
        "bar": {
            "errorbar": "sd",
            "annotate": True,
            "annotation_mode": "mean_sd",
            "figsize": (9, 5),
            "font_size": 12,
        },
        "topomap": {
            "shared_scale": False,
            "sphere": "auto",
            "cmap": "viridis",
            "contours": 6,
            "sensors": "k.",
            "n_cols": 3,
            "font_size": 12,
            "colorbar_decimals": 1,
        },
    }


def _merge_nested_config(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
    *,
    replace_mapping_keys: Sequence[str] = ("bands",),
) -> dict[str, Any]:
    """
    Merge configuration overrides into a copied base mapping.

    Nested mappings are normally merged recursively. Selected configuration
    blocks such as 'bands' are replaced completely so study-specific band
    definitions do not retain obsolete default bands.
    """
    replace_keys = set(replace_mapping_keys)

    def _copy_value(value):
        if isinstance(value, Mapping):
            return {key: _copy_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return tuple(value)
        return value

    merged = {key: _copy_value(value) for key, value in base.items()}

    if overrides is None:
        return merged
    if not isinstance(overrides, Mapping):
        raise TypeError("config overrides must be a mapping or None.")

    for key, value in overrides.items():
        if key in replace_keys:
            merged[key] = _copy_value(value)
        elif key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_nested_config(
                merged[key], value, replace_mapping_keys=replace_mapping_keys
            )
        else:
            merged[key] = _copy_value(value)

    return merged
    
def _normalize_group_columns(
    group_columns: Sequence[str] | str,
) -> tuple[str, ...]:
    """Normalize one or more grouping columns to a validated tuple."""
    if isinstance(group_columns, str):
        normalized = (group_columns,)
    else:
        normalized = tuple(str(value) for value in group_columns)
    if not normalized:
        raise ValueError("At least one group column must be supplied.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("group_columns contains duplicate values.")
    return normalized


def _infer_physical_level_group_columns(
    data: pd.DataFrame,
    group_columns: Sequence[str] | str,
    *,
    physical_recording_col: str,
) -> list[str]:
    """
    Keep requested grouping columns that are constant within each physical recording.

    Annotation-derived condition columns vary within one physical recording and
    are therefore excluded automatically. Study-level fields such as timepoint,
    cohort, dose, or label are retained when they remain constant.
    """
    group_columns = _normalize_group_columns(group_columns)
    if physical_recording_col not in data.columns:
        return [column for column in group_columns if column in data.columns]

    source = data.loc[data[physical_recording_col].notna()].copy()
    if source.empty:
        return []

    physical_columns = []
    for column in group_columns:
        if column not in source.columns:
            continue
        n_unique = source.groupby(physical_recording_col, observed=True, dropna=False)[column].nunique(dropna=False)
        if (n_unique <= 1).all():
            physical_columns.append(column)

    return physical_columns



def _extract_qeeg_metadata(
    recording_id: str,
    result: Mapping[str, Any],
    metadata_fields: Sequence[str],
) -> dict[str, Any]:
    """Extract standardized identifying/study metadata from one qEEG result."""
    metadata_values: dict[str, Any] = {"recording_id": str(recording_id)}
    for field in metadata_fields:
        field = str(field)
        if field == "recording_id":
            continue
        metadata_values[field] = result.get(field)
    return metadata_values



# def build_preprocessing_qc_summary(
#     qc_records: Sequence[Mapping[str, Any]],
#     *,
#     group_columns: Sequence[str] | str = ("eye_state", "timepoint"),
#     group_values: Mapping[str, Any] | None = None,
#     verbose: bool = True,
# ) -> dict[str, pd.DataFrame]:
#     """
#     Build detailed and grouped preprocessing QC tables.

#     The detailed table contains one row per attempted recording. The grouped
#     metric table summarizes successful recordings within the requested groups,
#     such as eye state and timepoint.

#     Parameters
#     ----------
#     qc_records
#         Recording-level QC dictionaries returned by
#         ``build_label_epoch_arrays``.

#     group_columns
#         Columns used to group the metric summary. Defaults to
#         ``("eye_state", "timepoint")``.

#     group_values
#         Optional constant values added to every recording. This is useful for
#         test data that do not yet contain real study metadata.

#         Example::

#             {
#                 "eye_state": "EO",
#                 "timepoint": "Baseline",
#             }

#     verbose
#         Whether to print a brief description of the generated tables.

#     Returns
#     -------
#     dict[str, pd.DataFrame]
#         Dictionary containing:

#         - ``recording_qc_df``: one row per attempted recording.
#         - ``qc_metric_summary_df``: grouped descriptive QC statistics.

#     Notes
#     -----
#     The grouped metric summary uses successfully processed recordings only.
#     No pass, review, or exclusion thresholds are applied by this function.
#     """
#     if not qc_records:
#         raise ValueError("qc_records is empty.")

#     # Convert the recording-level QC dictionaries into a DataFrame.
#     qc_df = pd.DataFrame([dict(record) for record in qc_records])

#     required_columns = {
#         "recording_id",
#         "subject_id",
#         "label",
#         "processing_status",
#     }

#     missing_columns = required_columns - set(qc_df.columns)

#     if missing_columns:
#         raise KeyError(
#             "qc_records is missing required fields: "
#             f"{sorted(missing_columns)}"
#         )

#     # Add constant study metadata when it is not yet stored per recording.
#     if group_values is not None:
#         if not isinstance(group_values, Mapping):
#             raise TypeError("group_values must be a mapping.")

#         for column, value in group_values.items():
#             qc_df[str(column)] = value

#     # Accept either one grouping column or multiple grouping columns.
#     if isinstance(group_columns, str):
#         group_columns_used = (group_columns,)
#     else:
#         group_columns_used = tuple(group_columns)

#     if not group_columns_used:
#         raise ValueError("At least one group column must be supplied.")

#     missing_group_columns = [
#         column
#         for column in group_columns_used
#         if column not in qc_df.columns
#     ]

#     if missing_group_columns:
#         raise KeyError(
#             "The following group columns are missing from the QC data: "
#             f"{missing_group_columns}"
#         )

#     # Convert available QC measurements to numeric values.
#     numeric_columns = [
#         # Epoch QC
#         "n_epochs_attempted", "n_epochs_rejected", "n_epochs_retained", "epoch_retention_percent", "usable_clean_minutes",

#         # Bad-channel detector QC
#         "n_mad_bad_channels", "n_ransac_bad_channels", "n_bad_channels",

#         # ICA / EOG QC
#         "n_excluded_ics", "final_n_eog_channels", "n_eog_candidate_ics", "processing_seconds",
#     ]

#     for column in numeric_columns:
#         if column in qc_df.columns:
#             qc_df[column] = pd.to_numeric(qc_df[column], errors="coerce")


#     # ============================================================
#     # 1. Build the detailed recording-level QC table
#     # ============================================================
#     important_columns = [
#         "recording_id", "subject_id", "label", *group_columns_used,
#         "processing_status", "qc_flag",

#         # Epoch QC
#         "n_epochs_attempted", "n_epochs_rejected", "n_epochs_retained",
#         "epoch_retention_percent", "usable_clean_minutes",

#         # Bad-channel detector QC
#         # Keep detector-specific fields next to the final union so we can see whether
#         # an interpolated channel was flagged by MAD, RANSAC, or both.
#         "n_mad_bad_channels", "mad_bad_channels","n_ransac_bad_channels", "ransac_bad_channels","n_bad_channels", "bad_channels",

#         # Final auxiliary-channel information
#         "final_n_eog_channels", "final_eog_channels",

#         # ICA / ocular-artifact QC
#         "n_excluded_ics", "excluded_ics", "excluded_ic_labels","eog_available", "eog_channels", "eog_candidate_ics", "n_eog_candidate_ics",

#         "processing_seconds", "processing_error",
#     ]
#     # Remove duplicate names while preserving the requested order.
#     ordered_columns = list(
#         dict.fromkeys(
#             column
#             for column in important_columns
#             if column in qc_df.columns
#         )
#     )

#     remaining_columns = [
#         column
#         for column in qc_df.columns
#         if column not in ordered_columns
#     ]

#     recording_qc_df = qc_df[
#         ordered_columns + remaining_columns
#     ].copy()

#     # ============================================================
#     # 2. Build the grouped human-readable metric summary
#     # ============================================================
#     metric_info: dict[str, tuple[str, str]] = {
#         # Epoch QC
#         "n_epochs_attempted": (
#             "Attempted epochs",
#             "epochs",
#         ),
#         "n_epochs_rejected": (
#             "Rejected epochs",
#             "epochs",
#         ),
#         "n_epochs_retained": (
#             "Retained clean epochs",
#             "epochs",
#         ),
#         "epoch_retention_percent": (
#             "Epoch retention",
#             "%",
#         ),
#         "usable_clean_minutes": (
#             "Usable clean data",
#             "minutes",
#         ),

#         # Channel QC
#         "n_bad_channels": (
#             "Bad/interpolated channels",
#             "channels",
#         ),
#         "final_n_eog_channels": (
#             "Available EOG channels",
#             "channels",
#         ),

#         # ICA / ocular-artifact QC
#         "n_excluded_ics": (
#             "Excluded ICA components",
#             "components",
#         ),
#         "n_eog_candidate_ics": (
#             "EOG-supported ICA components",
#             "components",
#         ),

#         # Processing performance
#         "processing_seconds": (
#             "Processing time",
#             "seconds",
#         ),
#     }

#     # Failed recordings remain in recording_qc_df but are not used
#     # to calculate distributions of successful preprocessing outputs.
#     success_df = qc_df.loc[
#         qc_df["processing_status"] == "success"
#     ].copy()

#     metric_rows: list[dict[str, Any]] = []

#     grouped_success = success_df.groupby(
#         list(group_columns_used),
#         observed=True,
#         dropna=False,
#         sort=False,
#     )

#     for group_key, group_df in grouped_success:
#         if not isinstance(group_key, tuple):
#             group_key = (group_key,)

#         group_values_dict = dict(
#             zip(group_columns_used, group_key)
#         )

#         for metric_key, (metric_label, unit) in metric_info.items():
#             if metric_key not in group_df.columns:
#                 continue

#             values = group_df[metric_key].dropna()

#             metric_rows.append({
#                 **group_values_dict,
#                 "metric": metric_label,
#                 "unit": unit,
#                 "n": int(values.count()),
#                 "mean": (
#                     float(values.mean())
#                     if not values.empty
#                     else np.nan
#                 ),
#                 "sd": (
#                     float(values.std(ddof=1))
#                     if len(values) > 1
#                     else np.nan
#                 ),
#                 "median": (
#                     float(values.median())
#                     if not values.empty
#                     else np.nan
#                 ),
#                 "minimum": (
#                     float(values.min())
#                     if not values.empty
#                     else np.nan
#                 ),
#                 "maximum": (
#                     float(values.max())
#                     if not values.empty
#                     else np.nan
#                 ),
#             })

#     summary_columns = [
#         *group_columns_used,
#         "metric",
#         "unit",
#         "n",
#         "mean",
#         "sd",
#         "median",
#         "minimum",
#         "maximum",
#     ]

#     qc_metric_summary_df = pd.DataFrame(
#         metric_rows,
#         columns=summary_columns,
#     )

#     if verbose:
#         print(
#             f"Created recording_qc_df with "
#             f"{len(recording_qc_df)} recordings."
#         )
#         print(
#             f"Created qc_metric_summary_df grouped by "
#             f"{', '.join(group_columns_used)}."
#         )

#     return {
#         "recording_qc_df": recording_qc_df,
#         "qc_metric_summary_df": qc_metric_summary_df,
#     }


def build_preprocessing_qc_summary(
    qc_records: Sequence[Mapping[str, Any]], *,
    group_columns: Sequence[str] | str = ("eye_state", "timepoint"),
    group_values: Mapping[str, Any] | None = None, verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Build detailed and grouped preprocessing QC tables.
    The detailed table contains one row per attempted recording. The grouped
    metric table summarizes successful recordings within the requested groups,
    such as eye state and timepoint.
    Parameters
    ----------
    qc_records
        Recording-level QC dictionaries returned by
        ``build_label_epoch_arrays``.
    group_columns
        Columns used to group the metric summary. Defaults to
        ``("eye_state", "timepoint")``.
    group_values
        Optional constant values added to every recording. This is useful for
        test data that do not yet contain real study metadata.
        Example::
            {
                "eye_state": "EO",
                "timepoint": "Baseline",
            }
    verbose
        Whether to print a brief description of the generated tables.
    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing:
        - ``recording_qc_df``: one row per attempted recording.
        - ``qc_metric_summary_df``: grouped descriptive QC statistics.
    Notes
    -----
    The grouped metric summary uses successfully processed recordings only.
    No pass, review, or exclusion thresholds are applied by this function.
    """
    if not qc_records:
        raise ValueError("qc_records is empty.")

    # Convert the recording-level QC dictionaries into a DataFrame.
    qc_df = pd.DataFrame([dict(record) for record in qc_records])
    required_columns = {"recording_id", "subject_id", "label", "processing_status"}
    missing_columns = required_columns - set(qc_df.columns)
    if missing_columns:
        raise KeyError(f"qc_records is missing required fields: {sorted(missing_columns)}")

    # Add constant study metadata when it is not yet stored per recording.
    if group_values is not None:
        if not isinstance(group_values, Mapping):
            raise TypeError("group_values must be a mapping.")
        for column, value in group_values.items():
            qc_df[str(column)] = value

    # Accept either one grouping column or multiple grouping columns.
    group_columns_used = (group_columns,) if isinstance(group_columns, str) else tuple(group_columns)
    if not group_columns_used:
        raise ValueError("At least one group column must be supplied.")
    missing_group_columns = [column for column in group_columns_used if column not in qc_df.columns]
    if missing_group_columns:
        raise KeyError(f"The following group columns are missing from the QC data: {missing_group_columns}")

    # Convert available QC measurements to numeric values.
    numeric_columns = [
        # Epoch QC
        "n_epochs_attempted", "n_epochs_rejected", "n_epochs_retained", "epoch_retention_percent", "usable_clean_minutes",
        # Bad-channel detector QC
        "n_mad_bad_channels", "n_ransac_bad_channels", "n_bad_channels",
        # ICA / EOG QC
        "n_excluded_ics", "final_n_eog_channels", "n_eog_candidate_ics", "processing_seconds",
    ]
    for column in numeric_columns:
        if column in qc_df.columns:
            qc_df[column] = pd.to_numeric(qc_df[column], errors="coerce")

    # ============================================================
    # 1. Build the detailed recording-level QC table
    # ============================================================
    important_columns = [
        "recording_id", "subject_id", "label", *group_columns_used, "processing_status", "qc_flag",
        # Epoch QC
        "n_epochs_attempted", "n_epochs_rejected", "n_epochs_retained", "epoch_retention_percent", "usable_clean_minutes",
        # Bad-channel detector QC
        # Keep detector-specific fields next to the final union so we can see whether
        # an interpolated channel was flagged by MAD, RANSAC, or both.
        "n_mad_bad_channels", "mad_bad_channels", "n_ransac_bad_channels", "ransac_bad_channels", "n_bad_channels", "bad_channels",
        # Final auxiliary-channel information
        "final_n_eog_channels", "final_eog_channels",
        # ICA / ocular-artifact QC
        "n_excluded_ics", "excluded_ics", "excluded_ic_labels", "eog_available", "eog_channels", "eog_candidate_ics", "n_eog_candidate_ics",
        "processing_seconds", "processing_error",
    ]
    # Remove duplicate names while preserving the requested order.
    ordered_columns = list(dict.fromkeys(column for column in important_columns if column in qc_df.columns))
    remaining_columns = [column for column in qc_df.columns if column not in ordered_columns]
    recording_qc_df = qc_df[ordered_columns + remaining_columns].copy()

    # ============================================================
    # 2. Build the grouped human-readable metric summary
    # ============================================================
    metric_info: dict[str, tuple[str, str]] = {
        # Epoch QC
        "n_epochs_attempted": ("Attempted epochs", "epochs"),
        "n_epochs_rejected": ("Rejected epochs", "epochs"),
        "n_epochs_retained": ("Retained clean epochs", "epochs"),
        "epoch_retention_percent": ("Epoch retention", "%"),
        "usable_clean_minutes": ("Usable clean data", "minutes"),
        # Channel QC
        "n_bad_channels": ("Bad/interpolated channels", "channels"),
        "final_n_eog_channels": ("Available EOG channels", "channels"),
        # ICA / ocular-artifact QC
        "n_excluded_ics": ("Excluded ICA components", "components"),
        "n_eog_candidate_ics": ("EOG-supported ICA components", "components"),
        # Processing performance
        "processing_seconds": ("Processing time", "seconds"),
    }

    # Failed recordings remain in recording_qc_df but are not used
    # to calculate distributions of successful preprocessing outputs.
    success_df = qc_df.loc[qc_df["processing_status"] == "success"].copy()
    metric_rows: list[dict[str, Any]] = []
    grouped_success = success_df.groupby(list(group_columns_used), observed=True, dropna=False, sort=False)

    for group_key, group_df in grouped_success:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_values_dict = dict(zip(group_columns_used, group_key))

        for metric_key, (metric_label, unit) in metric_info.items():
            if metric_key not in group_df.columns:
                continue
            values = group_df[metric_key].dropna()
            metric_rows.append({
                **group_values_dict,
                "metric": metric_label, "unit": unit, "n": int(values.count()),
                "mean": float(values.mean()) if not values.empty else np.nan,
                "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "median": float(values.median()) if not values.empty else np.nan,
                "minimum": float(values.min()) if not values.empty else np.nan,
                "maximum": float(values.max()) if not values.empty else np.nan,
            })

    summary_columns = [
        *group_columns_used, "metric", "unit", "n",
        "mean", "sd", "median", "minimum", "maximum",
    ]
    qc_metric_summary_df = pd.DataFrame(metric_rows, columns=summary_columns)

    if verbose:
        print(f"Created recording_qc_df with {len(recording_qc_df)} recordings.")
        print(f"Created qc_metric_summary_df grouped by {', '.join(group_columns_used)}.")

    return {
        "recording_qc_df": recording_qc_df,
        "qc_metric_summary_df": qc_metric_summary_df,
    }

# =============================================================================
# COHORT QC REPORT-PREPARATION HELPERS
# =============================================================================
# These helpers do NOT perform EEG preprocessing or calculate new QC metrics.
# They only reorganize existing QC results into compact cohort-level tables.
# =============================================================================




# -----------------------------------------------------------------------------
# Prepare preprocessing / data-completeness QC for cohort-level reporting.
def prepare_cohort_preprocessing_qc(
    cohort_interim_results,
    recording_qc_df,
    *,
    n_physical_total=None,
    timepoint_order=None,
    condition_col="eye_state",
    condition_order=None,
    condition_alias=None,
    minimum_clean_minutes=4.0,
    physical_recording_col="source_recording_id",
    eog_col="eog_available",
):
    """
    Prepare compact preprocessing / completeness QC tables from existing results.

    No preprocessing or qEEG calculation occurs here.

    condition_col / condition_order are optional so the helper can also be used
    with datasets that do not contain annotation-derived EO/EC conditions.
    """

    # ------------------------------------------------------------------
    # Retrieve existing cohort and physical-recording QC results
    # ------------------------------------------------------------------
    if not isinstance(cohort_interim_results, dict) or "timepoint_summary_df" not in cohort_interim_results:
        raise KeyError("cohort_interim_results must contain 'timepoint_summary_df'.")

    summary_df = cohort_interim_results["timepoint_summary_df"].copy()
    physical_df = recording_qc_df.drop_duplicates(physical_recording_col).copy()

    if n_physical_total is None:
        n_physical_total = physical_df[physical_recording_col].nunique()

    condition_alias = {} if condition_alias is None else dict(condition_alias)

    # ------------------------------------------------------------------
    # Headline cohort metrics
    # ------------------------------------------------------------------
    n_physical_processed = physical_df[physical_recording_col].nunique()
    n_logical_total = int(summary_df["n_records"].sum())
    n_qeeg_available = int(summary_df["n_qeeg_available"].sum())
    n_meeting_duration = int(summary_df["n_meeting_clean_duration"].sum())

    if eog_col in physical_df.columns:
        n_eog_available = int(physical_df[eog_col].fillna(False).astype(bool).sum())
    else:
        n_eog_available = 0

    duration_label = f">={minimum_clean_minutes:g}-min clean EEG requirement"

    kpi_df = pd.DataFrame([
        {"Metric": "Physical recordings processed", "Value": f"{n_physical_processed}/{n_physical_total}"},
        {"Metric": "Condition-specific qEEG outputs available", "Value": f"{n_qeeg_available}/{n_logical_total}"},
        {"Metric": duration_label, "Value": f"{n_meeting_duration}/{n_logical_total} ({100*n_meeting_duration/n_logical_total:.1f}%)"},
        {"Metric": "Physical recordings with EOG available", "Value": f"{n_eog_available}/{n_physical_total} ({100*n_eog_available/n_physical_total:.1f}%)"},
    ])

    # ------------------------------------------------------------------
    # Determine reporting order
    # ------------------------------------------------------------------
    if timepoint_order is None:
        timepoint_order = summary_df["timepoint"].dropna().astype(str).drop_duplicates().tolist()

    has_conditions = condition_col is not None and condition_col in summary_df.columns

    if has_conditions and condition_order is None:
        condition_order = summary_df[condition_col].dropna().astype(str).drop_duplicates().tolist()

    # ------------------------------------------------------------------
    # Compact timepoint summary
    # ------------------------------------------------------------------
    rows = []

    for timepoint in timepoint_order:
        current = summary_df.loc[summary_df["timepoint"].astype(str) == str(timepoint)].copy()
        if current.empty:
            continue

        row = {
            "Timepoint": str(timepoint),
            "N": int(current["n_subjects"].max()) if "n_subjects" in current.columns else int(current["n_records"].sum()),
        }

        if has_conditions:
            for condition in condition_order:
                condition_df = current.loc[current[condition_col].astype(str) == str(condition)]
                label = condition_alias.get(condition, str(condition))

                if condition_df.empty:
                    row[f"{label} >= {minimum_clean_minutes:g} min"] = "—"
                    row[f"{label} mean clean min"] = "—"
                    continue

                condition_row = condition_df.iloc[0]
                n = int(condition_row["n_records"])
                n_meet = int(condition_row["n_meeting_clean_duration"])

                row[f"{label} >= {minimum_clean_minutes:g} min"] = f"{n_meet}/{n} ({100*n_meet/n:.0f}%)"
                row[f"{label} mean clean min"] = f"{condition_row['mean_clean_minutes']:.2f}"

        else:
            n = int(current["n_records"].sum())
            n_meet = int(current["n_meeting_clean_duration"].sum())
            weights = current["n_records"].to_numpy(dtype=float)
            mean_clean = np.average(current["mean_clean_minutes"], weights=weights) if weights.sum() else np.nan

            row[f">= {minimum_clean_minutes:g} min"] = f"{n_meet}/{n} ({100*n_meet/n:.0f}%)"
            row["Mean clean min"] = f"{mean_clean:.2f}"

        rows.append(row)

    return {
        "kpi_df": kpi_df,
        "timepoint_df": pd.DataFrame(rows),
        "physical_recording_df": physical_df,
    }



# =============================================================================
# BAD / INTERPOLATED CHANNEL RECURRENCE — REPORT SUMMARY
# =============================================================================
def build_bad_channel_recurrence_summary(
    bad_channel_recurrence_df,
    physical_qc_df,
    *,
    group_col=None,
    group_order=None,
    group_label=None,
    expected_timepoints=None,
    top_n=6,
    physical_recording_col=None,
):
    """
    Build complete and report-ready bad/interpolated-channel recurrence tables.

    The recurrence calculation itself is not repeated here. This helper only
    collapses the existing physical-recording recurrence table into a compact
    report summary.

    Parameters
    ----------
    bad_channel_recurrence_df
        Existing recurrence table from summarize_qc_completeness().

    physical_qc_df
        Physical-recording QC table used to determine the total number of
        physical recordings represented in the analysis.

    group_col
        Optional physical-recording grouping dimension carried by the recurrence
        table, such as "timepoint" or "label". If None, the helper summarizes
        overall recurrence without group coverage.

    group_order
        Optional preferred order of group values.

    group_label
        Optional plural report label for group coverage, such as "Timepoints"
        or "Groups".

    expected_timepoints
        Backward-compatible NeuShen argument. When supplied, it is equivalent to:
            group_col="timepoint"
            group_order=expected_timepoints
            group_label="Timepoints"

    top_n
        Number of channels included in the compact report table.

    physical_recording_col
        Physical-recording identifier. If None, "source_recording_id" is used
        when available, otherwise "recording_id".
    """
    if not isinstance(bad_channel_recurrence_df, pd.DataFrame):
        raise TypeError("bad_channel_recurrence_df must be a pandas DataFrame.")
    if not isinstance(physical_qc_df, pd.DataFrame) or physical_qc_df.empty:
        raise ValueError("physical_qc_df must be a non-empty pandas DataFrame.")

    top_n = int(top_n)
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")

    # ------------------------------------------------------------------
    # Backward compatibility for the current NeuShen notebook
    # ------------------------------------------------------------------
    if expected_timepoints is not None:
        if group_col is None:
            group_col = "timepoint"
        if group_order is None:
            group_order = list(expected_timepoints)
        if group_label is None:
            group_label = "Timepoints"

    # ------------------------------------------------------------------
    # Resolve the physical-recording identifier
    # ------------------------------------------------------------------
    if physical_recording_col is None:
        physical_recording_col = (
            "source_recording_id"
            if "source_recording_id" in physical_qc_df.columns
            else "recording_id"
        )
    if physical_recording_col not in physical_qc_df.columns:
        raise KeyError(f"physical_qc_df does not contain '{physical_recording_col}'.")

    n_physical_recordings = physical_qc_df[physical_recording_col].dropna().astype(str).nunique()
    if n_physical_recordings == 0:
        raise ValueError("No physical recordings were found.")

    # ------------------------------------------------------------------
    # Gracefully handle a cohort with no recurring bad channels
    # ------------------------------------------------------------------
    coverage_label = group_label or (
        "Timepoints" if group_col == "timepoint"
        else group_col.replace("_", " ").title() if group_col else None
    )
    report_columns = ["Channel", "Recordings affected", "Percent affected"]
    if group_col is not None and group_order is not None and len(list(group_order)) > 1:
        report_columns.append(f"{coverage_label} affected")

    if bad_channel_recurrence_df.empty:
        return {
            "full_df": pd.DataFrame(columns=["bad_channel", "n_recordings_affected", "n_recordings_total", "percent_recordings_affected"]),
            "report_df": pd.DataFrame(columns=report_columns),
        }

    required = {"bad_channel", "n_recordings_affected"}
    missing = required - set(bad_channel_recurrence_df.columns)
    if missing:
        raise KeyError(f"bad_channel_recurrence_df is missing required columns: {sorted(missing)}")

    data = bad_channel_recurrence_df.copy()
    data["n_recordings_affected"] = pd.to_numeric(data["n_recordings_affected"], errors="coerce").fillna(0)

    # ------------------------------------------------------------------
    # Optional group coverage
    # ------------------------------------------------------------------
    if group_col is not None:
        if group_col not in data.columns:
            raise KeyError(f"bad_channel_recurrence_df does not contain group_col='{group_col}'.")

        observed_groups = data[group_col].dropna().drop_duplicates().tolist()
        group_order = list(group_order) if group_order is not None else observed_groups
        group_label = group_label or ("Timepoints" if group_col == "timepoint" else group_col.replace("_", " ").title())

        def ordered_groups(values):
            observed = set(values.dropna().astype(str))
            preferred = [str(value) for value in group_order if str(value) in observed]
            extras = [str(value) for value in values.dropna().drop_duplicates() if str(value) not in preferred]
            return ", ".join(preferred + extras)

        full_df = (
            data.groupby("bad_channel", observed=True)
            .agg(
                n_recordings_affected=("n_recordings_affected", "sum"),
                n_groups_affected=(group_col, "nunique"),
                groups_affected=(group_col, ordered_groups),
            )
            .reset_index()
        )

        # Preserve the established NeuShen column names for compatibility.
        if group_col == "timepoint":
            full_df["n_timepoints_affected"] = full_df["n_groups_affected"]
            full_df["timepoints_affected"] = full_df["groups_affected"]
    else:
        full_df = (
            data.groupby("bad_channel", observed=True)
            .agg(n_recordings_affected=("n_recordings_affected", "sum"))
            .reset_index()
        )
        group_order = []

    # ------------------------------------------------------------------
    # Overall cohort recurrence
    # ------------------------------------------------------------------
    full_df["n_recordings_affected"] = full_df["n_recordings_affected"].astype(int)
    full_df["n_recordings_total"] = n_physical_recordings
    full_df["percent_recordings_affected"] = 100.0 * full_df["n_recordings_affected"] / n_physical_recordings

    sort_columns = ["n_recordings_affected"]
    ascending = [False]
    if "n_groups_affected" in full_df.columns:
        sort_columns.append("n_groups_affected")
        ascending.append(False)
    sort_columns.append("bad_channel")
    ascending.append(True)
    full_df = full_df.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Compact report-facing table
    # ------------------------------------------------------------------
    report_df = full_df.head(top_n).copy()
    report_df["Recordings affected"] = report_df["n_recordings_affected"].astype(str) + f"/{n_physical_recordings}"
    report_df["Percent affected"] = report_df["percent_recordings_affected"].map(lambda x: f"{x:.1f}%")
    report_df = report_df.rename(columns={"bad_channel": "Channel"})

    report_columns = ["Channel", "Recordings affected", "Percent affected"]
    if group_col is not None and len(group_order) > 1:
        coverage_column = f"{group_label} affected"
        report_df[coverage_column] = report_df["n_groups_affected"].astype(int).astype(str) + f"/{len(group_order)}"
        report_columns.append(coverage_column)

    report_df = report_df[report_columns].reset_index(drop=True)
    return {"full_df": full_df, "report_df": report_df}



# -----------------------------------------------------------------------------
# Prepare physical-recording ICA / EOG QC for cohort-level reporting.
def prepare_ocular_ica_qc(
    recording_qc_df,
    *,
    group_col="timepoint",
    group_order=None,
    physical_recording_col="source_recording_id",
    excluded_ics_col="excluded_ics",
    eog_candidate_ics_col="eog_candidate_ics",
    eog_available_col="eog_available",
):
    """
    Prepare compact ocular / ICA QC tables from existing recording-level QC.

    ICA is summarized once per physical recording. EOG-supported components
    remain independent QC evidence and do not redefine the ICA exclusion set.
    """

    # ------------------------------------------------------------------
    # Keep one row per physical recording
    # ------------------------------------------------------------------
    df = recording_qc_df.drop_duplicates(physical_recording_col).copy()

    def _as_list(value):
        if isinstance(value, (list, tuple, set, pd.Series)):
            return list(value)
        if hasattr(value, "tolist"):
            value = value.tolist()
            return value if isinstance(value, list) else [value]
        return []

    df["_excluded_ics"] = df[excluded_ics_col].apply(_as_list)
    df["_eog_candidate_ics"] = df[eog_candidate_ics_col].apply(_as_list)
    df["n_excluded_ics_report"] = df["_excluded_ics"].apply(len)
    df["n_eog_supported_ics_report"] = df["_eog_candidate_ics"].apply(len)
    df["n_eog_supported_excluded_ics"] = df.apply(
        lambda row: len(set(row["_excluded_ics"]) & set(row["_eog_candidate_ics"])), axis=1
    )
    df["has_eog_exclusion_overlap"] = df["n_eog_supported_excluded_ics"] > 0
    df[eog_available_col] = df[eog_available_col].fillna(False).astype(bool)

    # ------------------------------------------------------------------
    # Headline cohort metrics
    # ------------------------------------------------------------------
    n = df[physical_recording_col].nunique()
    n_eog = int(df[eog_available_col].sum())
    n_excluded = int((df["n_excluded_ics_report"] > 0).sum())
    n_eog_candidates = int((df["n_eog_supported_ics_report"] > 0).sum())
    n_overlap = int(df["has_eog_exclusion_overlap"].sum())

    kpi_df = pd.DataFrame([
        {"Metric": "Physical recordings reviewed", "Value": f"{n}/{n}"},
        {"Metric": "Physical recordings with EOG available", "Value": f"{n_eog}/{n} ({100*n_eog/n:.1f}%)"},
        {"Metric": "Recordings with excluded ICA components", "Value": f"{n_excluded}/{n} ({100*n_excluded/n:.1f}%)"},
        {"Metric": "Recordings with EOG-supported ICA candidates", "Value": f"{n_eog_candidates}/{n} ({100*n_eog_candidates/n:.1f}%)"},
        {"Metric": "Recordings with EOG / exclusion overlap", "Value": f"{n_overlap}/{n} ({100*n_overlap/n:.1f}%)"},
    ])

    # ------------------------------------------------------------------
    # Grouped cohort summary
    # ------------------------------------------------------------------
    if group_col is None or group_col not in df.columns:
        df["_report_group"] = "Overall"
        group_col_used = "_report_group"
        group_order = ["Overall"]
    else:
        group_col_used = group_col
        if group_order is None:
            group_order = df[group_col_used].dropna().astype(str).drop_duplicates().tolist()

    rows = []

    for group in group_order:
        current = df.loc[df[group_col_used].astype(str) == str(group)]
        if current.empty:
            continue

        n_group = current[physical_recording_col].nunique()
        excluded_sd = current["n_excluded_ics_report"].std()
        eog_sd = current["n_eog_supported_ics_report"].std()

        rows.append({
            "Timepoint" if group_col_used == "timepoint" else "Group": str(group),
            "N": n_group,
            "EOG Available": f"{current[eog_available_col].sum()}/{n_group} ({100*current[eog_available_col].mean():.0f}%)",
            "Excluded ICs": f"{current['n_excluded_ics_report'].mean():.2f} ± {0.0 if pd.isna(excluded_sd) else excluded_sd:.2f}",
            "EOG-Supported ICs": f"{current['n_eog_supported_ics_report'].mean():.2f} ± {0.0 if pd.isna(eog_sd) else eog_sd:.2f}",
            "EOG / Exclusion Overlap": f"{current['has_eog_exclusion_overlap'].sum()}/{n_group} ({100*current['has_eog_exclusion_overlap'].mean():.0f}%)",
        })

    return {
        "kpi_df": kpi_df,
        "group_df": pd.DataFrame(rows),
        "recording_df": df,
    }


# -----------------------------------------------------------------------------
# Prepare posterior-alpha physiological QC for cohort-level reporting.

# def prepare_posterior_alpha_qc(
#     posterior_alpha_qc,
#     *,
#     timepoint_order=None,
#     condition_col="eye_state",
#     condition_order=None,
#     condition_label="Eye State",
# ):
#     """
#     Prepare compact posterior-alpha QC from build_posterior_alpha_qc() output.

#     This function does not calculate posterior alpha or PSD. It only summarizes
#     the existing recording-level posterior-alpha QC results.
#     """

#     # ------------------------------------------------------------------
#     # Retrieve existing posterior-alpha recording-level results
#     # ------------------------------------------------------------------
#     if not isinstance(posterior_alpha_qc, dict) or "summary_df" not in posterior_alpha_qc:
#         raise KeyError("posterior_alpha_qc must contain 'summary_df'.")

#     df = posterior_alpha_qc["summary_df"].copy()
#     df["posterior_alpha_predominant"] = df["posterior_alpha_predominant"].fillna(False).astype(bool)

#     if timepoint_order is not None and "timepoint" in df.columns:
#         df["timepoint"] = pd.Categorical(df["timepoint"], categories=timepoint_order, ordered=True)

#     if condition_order is not None and condition_col in df.columns:
#         df[condition_col] = pd.Categorical(df[condition_col], categories=condition_order, ordered=True)

#     # ------------------------------------------------------------------
#     # Use only grouping variables actually present in the result table
#     # ------------------------------------------------------------------
#     group_cols = []
#     if "timepoint" in df.columns:
#         group_cols.append("timepoint")
#     if condition_col is not None and condition_col in df.columns:
#         group_cols.append(condition_col)

#     if not group_cols:
#         df["_report_group"] = "Overall"
#         group_cols = ["_report_group"]

#     # ------------------------------------------------------------------
#     # Aggregate existing physiological QC metrics
#     # ------------------------------------------------------------------
#     numeric_df = (
#         df.groupby(group_cols, observed=True)
#         .agg(
#             n_recordings=("recording_id", "nunique"),
#             mean_posterior_alpha_percent=("posterior_alpha_percent", "mean"),
#             sd_posterior_alpha_percent=("posterior_alpha_percent", "std"),
#             mean_scalp_alpha_percent=("scalp_alpha_percent", "mean"),
#             sd_scalp_alpha_percent=("scalp_alpha_percent", "std"),
#             mean_posterior_to_scalp_ratio=("posterior_to_scalp_alpha_ratio", "mean"),
#             sd_posterior_to_scalp_ratio=("posterior_to_scalp_alpha_ratio", "std"),
#             n_posterior_alpha_predominant=("posterior_alpha_predominant", "sum"),
#         )
#         .reset_index()
#     )

#     numeric_df["percent_posterior_alpha_predominant"] = (
#         100 * numeric_df["n_posterior_alpha_predominant"] / numeric_df["n_recordings"]
#     )

#     # ------------------------------------------------------------------
#     # Format compact report-facing table
#     # ------------------------------------------------------------------
#     rows = []

#     for _, row in numeric_df.iterrows():
#         report_row = {}

#         if "timepoint" in numeric_df.columns:
#             report_row["Timepoint"] = str(row["timepoint"])

#         if condition_col in numeric_df.columns:
#             report_row[condition_label] = str(row[condition_col])

#         report_row.update({
#             "N": int(row["n_recordings"]),
#             "Posterior Alpha (%)": f"{row['mean_posterior_alpha_percent']:.2f} ± {row['sd_posterior_alpha_percent']:.2f}",
#             "Scalp Alpha (%)": f"{row['mean_scalp_alpha_percent']:.2f} ± {row['sd_scalp_alpha_percent']:.2f}",
#             "Posterior/Scalp": f"{row['mean_posterior_to_scalp_ratio']:.2f} ± {row['sd_posterior_to_scalp_ratio']:.2f}",
#             "Posterior Predominant": f"{int(row['n_posterior_alpha_predominant'])}/{int(row['n_recordings'])} ({row['percent_posterior_alpha_predominant']:.0f}%)",
#         })

#         rows.append(report_row)

#     return {
#         "numeric_df": numeric_df,
#         "report_df": pd.DataFrame(rows),
#     }


def prepare_posterior_alpha_qc(
    posterior_alpha_qc: Mapping[str, Any],
    *,
    group_columns: Sequence[str] | str | None = None,
    group_order: Mapping[str, Sequence[Any]] | None = None,
    group_labels: Mapping[str, str] | None = None,

    # Backward-compatible NeuShen arguments.
    timepoint_order: Sequence[Any] | None = None,
    condition_col: str | None = "eye_state",
    condition_order: Sequence[Any] | None = None,
    condition_label: str = "Eye State",
) -> dict[str, pd.DataFrame]:
    """
    Prepare compact posterior-alpha QC for arbitrary study/condition grouping.

    New preferred interface:
        group_columns=("timepoint", "eye_state")
        group_order={"timepoint": expected_timepoints, "eye_state": eye_state_order}
        group_labels={"timepoint": "Timepoint", "eye_state": "Eye State"}

    ABC-CT example:
        group_columns=("label",)
        group_labels={"label": "Group"}

    Legacy timepoint/condition arguments remain supported until notebooks are updated.
    """
    if not isinstance(posterior_alpha_qc, Mapping):
        raise TypeError("posterior_alpha_qc must be a mapping.")

    source_df = posterior_alpha_qc.get("summary_df", posterior_alpha_qc.get("recording_df"))
    if not isinstance(source_df, pd.DataFrame) or source_df.empty:
        raise ValueError("posterior_alpha_qc must contain a non-empty 'summary_df' or 'recording_df'.")

    required = {
        "recording_id", "posterior_alpha_percent", "scalp_alpha_percent",
        "posterior_to_scalp_alpha_ratio", "posterior_alpha_predominant",
    }
    missing = required - set(source_df.columns)
    if missing:
        raise KeyError(f"Posterior-alpha QC table is missing required columns: {sorted(missing)}")

    df = source_df.copy()
    df["posterior_alpha_predominant"] = df["posterior_alpha_predominant"].fillna(False).astype(bool)

    # ------------------------------------------------------------------
    # Resolve generic grouping; preserve legacy calls during transition
    # ------------------------------------------------------------------
    if group_columns is not None:
        group_columns_used = list(_normalize_group_columns(group_columns))
    else:
        group_columns_used = []
        if "timepoint" in df.columns and df["timepoint"].notna().any():
            group_columns_used.append("timepoint")
        if condition_col and condition_col in df.columns and df[condition_col].notna().any():
            group_columns_used.append(condition_col)
        if not group_columns_used and "label" in df.columns and df["label"].notna().any():
            group_columns_used = ["label"]

    missing_groups = [column for column in group_columns_used if column not in df.columns]
    if missing_groups:
        raise KeyError(f"Posterior-alpha QC table is missing grouping columns: {missing_groups}")

    if not group_columns_used:
        df["_report_group"] = "Overall"
        group_columns_used = ["_report_group"]

    order_map = {str(column): list(values) for column, values in dict(group_order or {}).items()}
    if timepoint_order is not None and "timepoint" in group_columns_used:
        order_map.setdefault("timepoint", list(timepoint_order))
    if condition_order is not None and condition_col in group_columns_used:
        order_map.setdefault(condition_col, list(condition_order))

    label_map = {str(column): str(label) for column, label in dict(group_labels or {}).items()}
    if condition_col in group_columns_used:
        label_map.setdefault(condition_col, condition_label)

    sort_columns = []
    for column in group_columns_used:
        observed = df[column].dropna().drop_duplicates().tolist()
        preferred = order_map.get(column, [])
        order_used = preferred + [value for value in observed if value not in preferred]
        if order_used:
            df[column] = pd.Categorical(df[column], categories=order_used, ordered=True)
        sort_columns.append(column)

    numeric_df = (
        df.groupby(group_columns_used, observed=True, dropna=False, sort=False)
        .agg(
            n_recordings=("recording_id", "nunique"),
            mean_posterior_alpha_percent=("posterior_alpha_percent", "mean"),
            sd_posterior_alpha_percent=("posterior_alpha_percent", "std"),
            mean_scalp_alpha_percent=("scalp_alpha_percent", "mean"),
            sd_scalp_alpha_percent=("scalp_alpha_percent", "std"),
            mean_posterior_to_scalp_ratio=("posterior_to_scalp_alpha_ratio", "mean"),
            sd_posterior_to_scalp_ratio=("posterior_to_scalp_alpha_ratio", "std"),
            n_posterior_alpha_predominant=("posterior_alpha_predominant", "sum"),
        )
        .reset_index()
    )
    numeric_df["percent_posterior_alpha_predominant"] = (
        100.0 * numeric_df["n_posterior_alpha_predominant"] / numeric_df["n_recordings"]
    )

    def format_mean_sd(mean_value, sd_value):
        if not np.isfinite(mean_value):
            return "N/A"
        return f"{mean_value:.2f} ± {sd_value:.2f}" if np.isfinite(sd_value) else f"{mean_value:.2f}"

    rows = []
    for _, row in numeric_df.iterrows():
        report_row = {}
        for column in group_columns_used:
            if column == "_report_group":
                report_row["Group"] = str(row[column])
            else:
                report_row[label_map.get(column, column.replace("_", " ").title())] = str(row[column])

        report_row.update({
            "N": int(row["n_recordings"]),
            "Posterior Alpha (%)": format_mean_sd(row["mean_posterior_alpha_percent"], row["sd_posterior_alpha_percent"]),
            "Scalp Alpha (%)": format_mean_sd(row["mean_scalp_alpha_percent"], row["sd_scalp_alpha_percent"]),
            "Posterior/Scalp": format_mean_sd(row["mean_posterior_to_scalp_ratio"], row["sd_posterior_to_scalp_ratio"]),
            "Posterior Predominant": (
                f"{int(row['n_posterior_alpha_predominant'])}/{int(row['n_recordings'])} "
                f"({row['percent_posterior_alpha_predominant']:.0f}%)"
            ),
        })
        rows.append(report_row)

    return {"numeric_df": numeric_df, "report_df": pd.DataFrame(rows)}




# -----------------------------------------------------------------------------
# Prepare existing frontal high-frequency QC summaries across study groups.

# def prepare_high_frequency_qc(
#     frontal_hf_tables,
#     *,
#     group_col="timepoint",
#     group_order=None,
#     group_label="Timepoint",
# ):
#     """
#     Prepare a combined high-frequency QC slide table from existing summaries.

#     Reuses build_frontal_high_frequency_slide_summary(); no high-frequency
#     metrics are recalculated here.
#     """

#     # ------------------------------------------------------------------
#     # Determine reporting groups from the existing aggregate table
#     # ------------------------------------------------------------------
#     if group_col is None:
#         return build_frontal_high_frequency_slide_summary(frontal_hf_tables)

#     aggregate_df = frontal_hf_tables.get("aggregate_df", pd.DataFrame())

#     if group_col not in aggregate_df.columns:
#         raise KeyError(f"frontal_hf_tables['aggregate_df'] does not contain '{group_col}'.")

#     if group_order is None:
#         group_order = aggregate_df[group_col].dropna().astype(str).drop_duplicates().tolist()

#     # ------------------------------------------------------------------
#     # Reuse the existing slide-summary helper once per requested group
#     # ------------------------------------------------------------------
#     rows = []

#     for group in group_order:
#         current = build_frontal_high_frequency_slide_summary(
#             frontal_hf_tables,
#             group_filters={group_col: group},
#         )

#         if current is None or current.empty:
#             continue

#         current = current.copy()
#         current.insert(0, group_label, str(group))
#         rows.append(current)

#     return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def prepare_high_frequency_qc(
    frontal_hf_tables: Mapping[str, Any],
    *,
    group_columns: Sequence[str] | str | None = None,
    group_order: Mapping[str, Sequence[Any]] | Sequence[Any] | None = None,
    group_labels: Mapping[str, str] | None = None,
    bands: Sequence[str] = ("beta", "gamma"),

    # Backward-compatible arguments used by the current NeuShen notebook.
    group_col: str | None = None,
    group_label: str | None = None,
) -> pd.DataFrame:
    """
    Prepare the report-facing high-frequency QC table.

    Preferred interface uses group_columns/group_order/group_labels. The legacy
    group_col/group_label interface remains supported until notebooks are updated.
    """
    aggregate_df = frontal_hf_tables.get("aggregate_df", pd.DataFrame())
    if not isinstance(aggregate_df, pd.DataFrame) or aggregate_df.empty:
        raise ValueError("frontal_hf_tables['aggregate_df'] must be a non-empty DataFrame.")

    # ------------------------------------------------------------------
    # Translate the current notebook interface into generic grouping
    # ------------------------------------------------------------------
    if group_columns is None and group_col is not None:
        columns = [group_col]
        if "eye_state" in aggregate_df.columns and aggregate_df["eye_state"].notna().any():
            columns.append("eye_state")
        group_columns = tuple(columns)

        if isinstance(group_order, Mapping):
            order_map = {str(column): list(values) for column, values in group_order.items()}
        else:
            order_map = {group_col: list(group_order)} if group_order is not None else {}
        if "eye_state" in columns and "eye_state" not in order_map:
            order_map["eye_state"] = aggregate_df["eye_state"].dropna().drop_duplicates().tolist()

        label_map = dict(group_labels or {})
        if group_label is not None:
            label_map.setdefault(group_col, group_label)
        label_map.setdefault("eye_state", "Eye State")
    else:
        order_map = (
            {str(column): list(values) for column, values in group_order.items()}
            if isinstance(group_order, Mapping) else {}
        )
        label_map = dict(group_labels or {})

    return build_frontal_high_frequency_slide_summary(
        frontal_hf_tables,
        group_columns=group_columns,
        group_order=order_map,
        group_labels=label_map,
        bands=bands,
    )


# =============================================================================
# COHORT-LEVEL INTERIM RESULTS PREPARATION
# =============================================================================

def build_cohort_interim_results(
    qeeg_results: Mapping[str, Any],
    *,
    minimum_clean_minutes: float = 4.0,
    timepoint_order: Sequence[Any] | None = None,
    eye_state_order: Sequence[Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Build one coordinated cohort-level interim-results object from the
    completed Part 1 qEEG pipeline.

    This function does NOT recalculate preprocessing or qEEG endpoints.
    It organizes the existing Part 1 outputs so QC summaries and
    sponsor-facing cohort-level plots can be created consistently across
    all study timepoints.

    The function keeps every successfully calculated qEEG result.
    The clean-duration requirement is added as a flag rather than being
    used to silently remove recordings.

    Parameters
    ----------
    qeeg_results
        Result object returned by run_qeeg_part1_pipeline().

    minimum_clean_minutes
        Minimum usable clean EEG duration required for the intended analysis.

        For the current NeuShen study, the confirmed requirement is
        4.0 minutes per EO/EC condition.

    timepoint_order
        Optional preferred study-timepoint order.

        Example:
            [
                "PREDOSE",
                "H1",
                "H2",
                "H3",
                "H4",
                "H6",
                "H8",
                "H24",
            ]

        If None, timepoints are retained in their observed order.

    eye_state_order
        Optional preferred eye-state order.

        Example:
            ["EO", "EC"]

        If None, eye states are retained in their observed order.

    verbose
        Whether to print a concise cohort-level preparation summary.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:

        analysis_records_df
            One row per logical EO/EC recording.

            Preserves the detailed preprocessing/QC fields and adds:
                - qeeg_result_available
                - meets_clean_duration_requirement
                - timepoint_order_index
                - eye_state_order_index

        timepoint_summary_df
            One row per eye-state × timepoint containing:
                - number of available recordings
                - number of available subjects
                - qEEG availability
                - number meeting the clean-duration requirement
                - percentage meeting the requirement
                - clean-duration descriptive statistics

        psd_df
            Recording-level PSD values across ALL timepoints with
            QC/clean-duration flags attached.

        absolute_power_df
            Recording-level absolute band-power values across ALL
            timepoints with QC/clean-duration flags attached.

        relative_power_df
            Recording-level relative band-power values across ALL
            timepoints with QC/clean-duration flags attached.

        spectral_ratio_df
            Recording-level spectral-ratio values across ALL timepoints
            with QC/clean-duration flags attached.

        settings
            Main settings used to construct the interim-results object.

    Notes
    -----
    - No subject-level treatment trajectories are calculated here.
    - No change-from-baseline calculations are performed here.
    - No qEEG values are recalculated.
    - No recording is automatically excluded because it falls below the
      clean-duration requirement.
    - This is a preparation layer for blinded cohort-level interim summaries.
    """

    # ============================================================
    # 1. Validate the main Part 1 result object
    # ============================================================
    if not isinstance(qeeg_results, Mapping):
        raise TypeError(
            "qeeg_results must be the mapping returned by "
            "run_qeeg_part1_pipeline()."
        )

    minimum_clean_minutes = float(minimum_clean_minutes)

    if minimum_clean_minutes < 0:
        raise ValueError(
            "minimum_clean_minutes must be greater than or equal to zero."
        )

    required_sections = {
        "results_by_recording",
        "qc",
        "tables",
    }

    missing_sections = (
        required_sections
        - set(qeeg_results.keys())
    )

    if missing_sections:
        raise KeyError(
            "qeeg_results is missing required sections: "
            f"{sorted(missing_sections)}"
        )

    # ============================================================
    # 2. Retrieve the detailed preprocessing/QC table
    # ============================================================
    qc_section = qeeg_results["qc"]

    if not isinstance(qc_section, Mapping):
        raise TypeError(
            "qeeg_results['qc'] must be a mapping."
        )

    if "recording_qc_df" not in qc_section:
        raise KeyError(
            "qeeg_results['qc'] is missing 'recording_qc_df'."
        )

    analysis_records_df = (
        qc_section["recording_qc_df"]
        .copy()
    )

    if not isinstance(analysis_records_df, pd.DataFrame):
        raise TypeError(
            "qeeg_results['qc']['recording_qc_df'] "
            "must be a pandas DataFrame."
        )

    required_qc_columns = {
        "recording_id",
        "subject_id",
        "eye_state",
        "timepoint",
        "processing_status",
        "usable_clean_minutes",
    }

    missing_qc_columns = (
        required_qc_columns
        - set(analysis_records_df.columns)
    )

    if missing_qc_columns:
        raise KeyError(
            "recording_qc_df is missing required columns: "
            f"{sorted(missing_qc_columns)}"
        )

    # Recording ID is the common key connecting QC and qEEG tables.
    analysis_records_df["recording_id"] = (
        analysis_records_df["recording_id"]
        .astype(str)
    )

    # One QC row should correspond to one logical EO/EC recording.
    if analysis_records_df["recording_id"].duplicated().any():
        duplicated_ids = (
            analysis_records_df.loc[
                analysis_records_df["recording_id"].duplicated(
                    keep=False
                ),
                "recording_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "recording_qc_df contains duplicate recording IDs: "
            f"{duplicated_ids[:10]}"
        )

    # Ensure clean duration is numeric.
    analysis_records_df["usable_clean_minutes"] = (
        pd.to_numeric(
            analysis_records_df["usable_clean_minutes"],
            errors="coerce",
        )
    )

    # ============================================================
    # 3. Determine which logical recordings have qEEG results
    # ============================================================
    results_by_recording = (
        qeeg_results["results_by_recording"]
    )

    if not isinstance(results_by_recording, Mapping):
        raise TypeError(
            "qeeg_results['results_by_recording'] "
            "must be a mapping."
        )

    qeeg_recording_ids = {
        str(recording_id)
        for recording_id in results_by_recording.keys()
    }

    analysis_records_df["qeeg_result_available"] = (
        analysis_records_df["recording_id"]
        .isin(qeeg_recording_ids)
    )

    # ============================================================
    # 4. Add the clean-duration requirement flag
    # ============================================================
    #
    # IMPORTANT:
    # This flag documents whether the recording meets the intended
    # clean-duration requirement.
    #
    # The function does NOT automatically exclude recordings that
    # fall below the threshold.
    # ============================================================
    analysis_records_df[
        "meets_clean_duration_requirement"
    ] = (
        analysis_records_df["usable_clean_minutes"].notna()
        & (
            analysis_records_df["usable_clean_minutes"]
            >= minimum_clean_minutes
        )
    )

    # ============================================================
    # 5. Determine plotting/display order
    # ============================================================
    observed_timepoints = (
        analysis_records_df["timepoint"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    if timepoint_order is None:
        timepoint_order_used = list(
            observed_timepoints
        )
    else:
        timepoint_order_used = list(
            dict.fromkeys(timepoint_order)
        )

        # Preserve any unexpected/additional timepoints rather than
        # silently dropping them.
        timepoint_order_used.extend([
            value
            for value in observed_timepoints
            if value not in timepoint_order_used
        ])

    observed_eye_states = (
        analysis_records_df["eye_state"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    if eye_state_order is None:
        eye_state_order_used = list(
            observed_eye_states
        )
    else:
        eye_state_order_used = list(
            dict.fromkeys(eye_state_order)
        )

        # Preserve any additional condition values.
        eye_state_order_used.extend([
            value
            for value in observed_eye_states
            if value not in eye_state_order_used
        ])

    timepoint_to_index = {
        value: index
        for index, value in enumerate(
            timepoint_order_used
        )
    }

    eye_state_to_index = {
        value: index
        for index, value in enumerate(
            eye_state_order_used
        )
    }

    analysis_records_df["timepoint_order_index"] = (
        analysis_records_df["timepoint"]
        .map(timepoint_to_index)
    )

    analysis_records_df["eye_state_order_index"] = (
        analysis_records_df["eye_state"]
        .map(eye_state_to_index)
    )

    # ============================================================
    # 6. Build cohort-level QC / completeness summary
    # ============================================================
    #
    # Only rows with a defined eye state and timepoint are used in
    # the EO/EC × timepoint summary.
    #
    # All detailed rows remain preserved in analysis_records_df.
    # ============================================================
    summary_source_df = (
        analysis_records_df.loc[
            analysis_records_df["eye_state"].notna()
            & analysis_records_df["timepoint"].notna()
        ]
        .copy()
    )

    summary_rows: list[dict[str, Any]] = []

    grouped = summary_source_df.groupby(
        [
            "eye_state",
            "timepoint",
        ],
        observed=True,
        dropna=False,
        sort=False,
    )

    for (
        eye_state,
        timepoint,
    ), group_df in grouped:

        processing_success = (
            group_df["processing_status"]
            .astype(str)
            .str.lower()
            .eq("success")
        )

        qeeg_available = (
            group_df["qeeg_result_available"]
            .astype(bool)
        )

        meets_requirement = (
            group_df[
                "meets_clean_duration_requirement"
            ]
            .astype(bool)
        )

        # Clean-duration descriptive statistics are calculated
        # from logical recordings that reached the qEEG result stage.
        clean_minutes = (
            group_df.loc[
                qeeg_available,
                "usable_clean_minutes",
            ]
            .dropna()
        )

        n_qeeg_available = int(
            qeeg_available.sum()
        )

        n_meeting_requirement = int(
            (
                qeeg_available
                & meets_requirement
            ).sum()
        )

        percent_meeting_requirement = (
            100.0
            * n_meeting_requirement
            / n_qeeg_available
            if n_qeeg_available > 0
            else np.nan
        )

        subjects_with_qeeg = (
            group_df.loc[
                qeeg_available,
                "subject_id",
            ]
            .dropna()
            .nunique()
        )

        subjects_meeting_requirement = (
            group_df.loc[
                qeeg_available
                & meets_requirement,
                "subject_id",
            ]
            .dropna()
            .nunique()
        )

        summary_rows.append({
            "eye_state":
                eye_state,

            "timepoint":
                timepoint,

            "n_records":
                int(len(group_df)),

            "n_subjects":
                int(
                    group_df["subject_id"]
                    .dropna()
                    .nunique()
                ),

            "n_processing_success":
                int(processing_success.sum()),

            "n_qeeg_available":
                n_qeeg_available,

            "n_subjects_qeeg_available":
                int(subjects_with_qeeg),

            "n_meeting_clean_duration":
                n_meeting_requirement,

            "n_below_clean_duration":
                int(
                    n_qeeg_available
                    - n_meeting_requirement
                ),

            "n_subjects_meeting_clean_duration":
                int(subjects_meeting_requirement),

            "percent_meeting_clean_duration":
                float(percent_meeting_requirement),

            "mean_clean_minutes":
                (
                    float(clean_minutes.mean())
                    if not clean_minutes.empty
                    else np.nan
                ),

            "sd_clean_minutes":
                (
                    float(clean_minutes.std(ddof=1))
                    if len(clean_minutes) > 1
                    else np.nan
                ),

            "median_clean_minutes":
                (
                    float(clean_minutes.median())
                    if not clean_minutes.empty
                    else np.nan
                ),

            "minimum_clean_minutes":
                (
                    float(clean_minutes.min())
                    if not clean_minutes.empty
                    else np.nan
                ),

            "maximum_clean_minutes":
                (
                    float(clean_minutes.max())
                    if not clean_minutes.empty
                    else np.nan
                ),

            "clean_duration_requirement_minutes":
                float(minimum_clean_minutes),

            "timepoint_order_index":
                timepoint_to_index.get(
                    timepoint,
                    np.nan,
                ),

            "eye_state_order_index":
                eye_state_to_index.get(
                    eye_state,
                    np.nan,
                ),
        })

    timepoint_summary_df = pd.DataFrame(
        summary_rows
    )

    if not timepoint_summary_df.empty:
        timepoint_summary_df = (
            timepoint_summary_df
            .sort_values(
                [
                    "eye_state_order_index",
                    "timepoint_order_index",
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    # ============================================================
    # 7. Retrieve the existing qEEG result tables
    # ============================================================
    tables = qeeg_results["tables"]

    if not isinstance(tables, Mapping):
        raise TypeError(
            "qeeg_results['tables'] must be a mapping."
        )

    required_tables = {
        "combined_psd_df",
        "absolute_power_df",
        "relative_power_df",
        "spectral_ratio_df",
    }

    missing_tables = (
        required_tables
        - set(tables.keys())
    )

    if missing_tables:
        raise KeyError(
            "qeeg_results['tables'] is missing required tables: "
            f"{sorted(missing_tables)}"
        )

    # ============================================================
    # 8. Build the small QC lookup attached to every qEEG table
    # ============================================================
    #
    # Do NOT merge all QC columns into every frequency/band row.
    # That would unnecessarily duplicate large amounts of metadata.
    #
    # Only fields needed for cohort-level plotting and traceability
    # are attached here.
    # ============================================================
    qc_lookup_df = (
        analysis_records_df[[
            "recording_id",
            "processing_status",
            "usable_clean_minutes",
            "qeeg_result_available",
            "meets_clean_duration_requirement",
            "timepoint_order_index",
            "eye_state_order_index",
        ]]
        .copy()
    )

    # ============================================================
    # 9. Attach QC / ordering information to all qEEG tables
    # ============================================================
    prepared_tables: dict[str, pd.DataFrame] = {}

    source_table_lookup = {
        "psd_df":
            tables["combined_psd_df"],

        "absolute_power_df":
            tables["absolute_power_df"],

        "relative_power_df":
            tables["relative_power_df"],

        "spectral_ratio_df":
            tables["spectral_ratio_df"],
    }

    for (
        output_name,
        source_df,
    ) in source_table_lookup.items():

        if not isinstance(source_df, pd.DataFrame):
            raise TypeError(
                f"{output_name} must be a pandas DataFrame."
            )

        if "recording_id" not in source_df.columns:
            raise KeyError(
                f"{output_name} is missing 'recording_id'."
            )

        prepared_df = source_df.copy()

        prepared_df["recording_id"] = (
            prepared_df["recording_id"]
            .astype(str)
        )

        prepared_df = prepared_df.merge(
            qc_lookup_df,
            on="recording_id",
            how="left",
            validate="many_to_one",
        )

        # A qEEG row without a corresponding QC record would break
        # traceability and should therefore fail loudly.
        missing_qc_match = (
            prepared_df["usable_clean_minutes"]
            .isna()
            & ~prepared_df["recording_id"].isin(
                analysis_records_df.loc[
                    analysis_records_df[
                        "usable_clean_minutes"
                    ].isna(),
                    "recording_id",
                ]
            )
        )

        if missing_qc_match.any():
            missing_ids = (
                prepared_df.loc[
                    missing_qc_match,
                    "recording_id",
                ]
                .drop_duplicates()
                .tolist()
            )

            raise RuntimeError(
                "Some qEEG rows could not be matched to QC records: "
                f"{missing_ids[:10]}"
            )

        # Sort values into the study order while keeping the actual
        # timepoint and eye-state strings unchanged.
        sort_columns = [
            column
            for column in (
                "eye_state_order_index",
                "timepoint_order_index",
                "recording_id",
            )
            if column in prepared_df.columns
        ]

        if sort_columns:
            prepared_df = (
                prepared_df
                .sort_values(
                    sort_columns,
                    kind="stable",
                )
                .reset_index(drop=True)
            )

        prepared_tables[
            output_name
        ] = prepared_df

    # ============================================================
    # 10. Sort the master recording table
    # ============================================================
    analysis_records_df = (
        analysis_records_df
        .sort_values(
            [
                "eye_state_order_index",
                "timepoint_order_index",
                "subject_id",
                "recording_id",
            ],
            kind="stable",
            na_position="last",
        )
        .reset_index(drop=True)
    )

    # ============================================================
    # 11. Store settings for traceability
    # ============================================================
    settings = {
        "minimum_clean_minutes":
            float(minimum_clean_minutes),

        "timepoint_order":
            list(timepoint_order_used),

        "eye_state_order":
            list(eye_state_order_used),

        "n_analysis_records":
            int(len(analysis_records_df)),

        "n_qeeg_results":
            int(
                analysis_records_df[
                    "qeeg_result_available"
                ].sum()
            ),

        "n_meeting_clean_duration":
            int(
                analysis_records_df[
                    "meets_clean_duration_requirement"
                ].sum()
            ),
    }

    # ============================================================
    # 12. Concise reporting
    # ============================================================
    if verbose:
        print("\nCohort interim-results preparation")
        print("=" * 60)

        print(
            f"Logical analysis records: "
            f"{settings['n_analysis_records']}"
        )

        print(
            f"qEEG results available:   "
            f"{settings['n_qeeg_results']}"
        )

        print(
            f"Clean-duration target:    "
            f">= {minimum_clean_minutes:.2f} minutes"
        )

        print(
            f"Meeting duration target:  "
            f"{settings['n_meeting_clean_duration']}/"
            f"{settings['n_qeeg_results']}"
        )

        print(
            "Timepoints:               "
            + ", ".join(
                map(
                    str,
                    timepoint_order_used,
                )
            )
        )

        print(
            "Eye states:               "
            + ", ".join(
                map(
                    str,
                    eye_state_order_used,
                )
            )
        )

        print("=" * 60)

    # ============================================================
    # 13. Return one coordinated interim-results object
    # ============================================================
    return {
        "analysis_records_df":
            analysis_records_df,

        "timepoint_summary_df":
            timepoint_summary_df,

        "psd_df":
            prepared_tables["psd_df"],

        "absolute_power_df":
            prepared_tables[
                "absolute_power_df"
            ],

        "relative_power_df":
            prepared_tables[
                "relative_power_df"
            ],

        "spectral_ratio_df":
            prepared_tables[
                "spectral_ratio_df"
            ],

        "settings":
            settings,
    }

# =============================================================================
# PREPROCESSING QC — DATA COMPLETENESS / RECORDING-QUALITY SUMMARY
# =============================================================================

def summarize_qc_completeness(
    preprocessing_qc: Mapping[str, Any],
    *,
    minimum_clean_minutes: float | None = None,
    group_columns: Sequence[str] | str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build recording-level and aggregate EEG QC/completeness summaries.

    Summarizes processing success, clean EEG duration, epoch rejection/retention,
    bad/interpolated channels, ICA/EOG QC, QC notes, and bad-channel recurrence.

    minimum_clean_minutes is optional:
        NeuShen -> 4.0
        ABC-CT -> None

    If None, clean-duration values are summarized but no duration-based pass/fail
    criterion is created. This function never excludes recordings.
    """
    if not isinstance(preprocessing_qc, Mapping):
        raise TypeError("preprocessing_qc must be a mapping.")

    recording_qc_df = preprocessing_qc.get("recording_qc_df")
    if not isinstance(recording_qc_df, pd.DataFrame):
        raise TypeError("preprocessing_qc['recording_qc_df'] must be a pandas DataFrame.")
    if recording_qc_df.empty:
        raise ValueError("preprocessing_qc['recording_qc_df'] is empty.")

    # ------------------------------------------------------------------
    # Optional clean-duration criterion
    # ------------------------------------------------------------------
    clean_duration_criterion_enabled = minimum_clean_minutes is not None
    if clean_duration_criterion_enabled:
        minimum_clean_minutes = float(minimum_clean_minutes)
        if minimum_clean_minutes <= 0:
            raise ValueError("minimum_clean_minutes must be greater than zero or None.")

    data = recording_qc_df.copy()

    # ------------------------------------------------------------------
    # Validate and normalize QC fields
    # ------------------------------------------------------------------
    required_columns = {"recording_id", "subject_id", "processing_status", "usable_clean_minutes"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise KeyError(f"recording_qc_df is missing required columns: {sorted(missing_columns)}")

    numeric_columns = [
        "n_epochs_attempted", "n_epochs_rejected", "n_epochs_retained",
        "epoch_retention_percent", "usable_clean_minutes", "n_bad_channels",
        "n_excluded_ics", "n_eog_candidate_ics",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    def normalize_list(value: Any) -> list[Any]:
        """Convert stored QC values into a safe list representation."""
        if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
            return list(value)
        if value is None:
            return []
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass
        return [value]

    if "bad_channels" in data.columns:
        data["bad_channels"] = data["bad_channels"].apply(lambda value: [str(x) for x in normalize_list(value)])
    if "qc_notes" in data.columns:
        data["qc_notes"] = data["qc_notes"].apply(lambda value: [str(x) for x in normalize_list(value)])

    # =========================================================================
    # 1. RECORDING-LEVEL QC / COMPLETENESS
    # =========================================================================
    data["processing_successful"] = data["processing_status"].astype(str).str.lower().eq("success")

    if {"n_epochs_attempted", "n_epochs_rejected"}.issubset(data.columns):
        data["epoch_rejection_percent"] = np.where(
            data["n_epochs_attempted"] > 0,
            100.0 * data["n_epochs_rejected"] / data["n_epochs_attempted"],
            np.nan,
        )
        data["has_rejected_epochs"] = data["n_epochs_rejected"].fillna(0).gt(0)

    # Create the clean-duration pass/fail flag only when the study defines one.
    if clean_duration_criterion_enabled:
        data["meets_clean_minutes_requirement"] = (
            data["processing_successful"] & data["usable_clean_minutes"].ge(minimum_clean_minutes)
        )

    if "n_bad_channels" in data.columns:
        data["has_bad_or_interpolated_channels"] = data["n_bad_channels"].fillna(0).gt(0)
    if "n_excluded_ics" in data.columns:
        data["has_excluded_ics"] = data["n_excluded_ics"].fillna(0).gt(0)
    if "n_eog_candidate_ics" in data.columns:
        data["has_eog_supported_ics"] = data["n_eog_candidate_ics"].fillna(0).gt(0)
    if "qc_notes" in data.columns:
        data["has_qc_notes"] = data["qc_notes"].apply(len).gt(0)

    preferred_columns = [
        "recording_id", "source_recording_id", "subject_id", "label",
        "cohort", "visit", "timepoint", "dose", "condition", "analysis_condition", "eye_state",
        "processing_status", "processing_successful",
        "n_epochs_attempted", "n_epochs_rejected", "epoch_rejection_percent",
        "n_epochs_retained", "epoch_retention_percent",
        "usable_clean_minutes", "meets_clean_minutes_requirement",
        "bad_channels", "n_bad_channels", "has_bad_or_interpolated_channels",
        "excluded_ics", "n_excluded_ics", "has_excluded_ics",
        "eog_available", "eog_candidate_ics", "n_eog_candidate_ics", "has_eog_supported_ics",
        "qc_flag", "qc_notes", "has_qc_notes", "processing_error",
    ]
    all_recordings_df = data[[column for column in preferred_columns if column in data.columns]].copy()

    # =========================================================================
    # 2. DETERMINE LOGICAL-RECORDING AGGREGATION COLUMNS
    # =========================================================================
    if group_columns is not None:
        group_columns_used = list(_normalize_group_columns(group_columns))
        missing = [column for column in group_columns_used if column not in data.columns]
        if missing:
            raise KeyError(f"Requested group columns are missing: {missing}")
    else:
        # Preserve the established study fields when present. If no study/condition
        # grouping exists, fall back to label and finally to one overall group.
        group_columns_used = [
            column for column in ("cohort", "timepoint", "eye_state")
            if column in data.columns and data[column].notna().any()
        ]
        if not group_columns_used:
            for column in ("analysis_condition", "condition", "label"):
                if column in data.columns and data[column].notna().any():
                    group_columns_used = [column]
                    break

    if not group_columns_used:
        data["_report_group"] = "Overall"
        group_columns_used = ["_report_group"]

    # =========================================================================
    # 3. AGGREGATE QC / COMPLETENESS
    # =========================================================================
    aggregate_rows: list[dict[str, Any]] = []

    for group_key, group_df in data.groupby(group_columns_used, observed=True, dropna=False, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_columns_used, group_key))

        # Processing success.
        n_attempted = int(len(group_df))
        n_successful = int(group_df["processing_successful"].sum())
        n_failed = n_attempted - n_successful
        row.update({
            "n_attempted": n_attempted,
            "n_successful": n_successful,
            "n_failed": n_failed,
            "percent_processing_successful": 100.0 * n_successful / n_attempted if n_attempted else np.nan,
        })

        # Optional clean-duration criterion.
        if clean_duration_criterion_enabled:
            n_meeting = int(group_df["meets_clean_minutes_requirement"].sum())
            row.update({
                "n_meeting_clean_minutes": n_meeting,
                "percent_meeting_clean_minutes": 100.0 * n_meeting / n_attempted if n_attempted else np.nan,
                "minimum_clean_minutes_required": minimum_clean_minutes,
            })

        # Clean-duration descriptive statistics are always retained.
        clean_minutes = pd.to_numeric(group_df["usable_clean_minutes"], errors="coerce").dropna()
        row.update({
            "usable_clean_minutes_mean": float(clean_minutes.mean()) if not clean_minutes.empty else np.nan,
            "usable_clean_minutes_sd": float(clean_minutes.std(ddof=1)) if len(clean_minutes) > 1 else np.nan,
            "usable_clean_minutes_median": float(clean_minutes.median()) if not clean_minutes.empty else np.nan,
            "usable_clean_minutes_minimum": float(clean_minutes.min()) if not clean_minutes.empty else np.nan,
            "usable_clean_minutes_maximum": float(clean_minutes.max()) if not clean_minutes.empty else np.nan,
        })

        # Epoch rejection / retention.
        if {"n_epochs_attempted", "n_epochs_rejected"}.issubset(group_df.columns):
            attempted = pd.to_numeric(group_df["n_epochs_attempted"], errors="coerce")
            rejected = pd.to_numeric(group_df["n_epochs_rejected"], errors="coerce")
            retained = pd.to_numeric(
                group_df.get("n_epochs_retained", pd.Series(np.nan, index=group_df.index)), errors="coerce"
            )
            total_attempted = int(attempted.fillna(0).sum())
            total_rejected = int(rejected.fillna(0).sum())
            total_retained = int(retained.fillna(0).sum())
            rejection_percent = pd.to_numeric(group_df["epoch_rejection_percent"], errors="coerce").dropna()
            n_with_rejected = int(group_df["has_rejected_epochs"].sum())

            row.update({
                "total_epochs_attempted": total_attempted,
                "total_epochs_rejected": total_rejected,
                "total_epochs_retained": total_retained,
                "overall_epoch_rejection_percent": 100.0 * total_rejected / total_attempted if total_attempted else np.nan,
                "epoch_rejection_percent_mean": float(rejection_percent.mean()) if not rejection_percent.empty else np.nan,
                "epoch_rejection_percent_sd": float(rejection_percent.std(ddof=1)) if len(rejection_percent) > 1 else np.nan,
                "epoch_rejection_percent_maximum": float(rejection_percent.max()) if not rejection_percent.empty else np.nan,
                "n_with_rejected_epochs": n_with_rejected,
                "percent_with_rejected_epochs": 100.0 * n_with_rejected / n_attempted if n_attempted else np.nan,
            })

        if "epoch_retention_percent" in group_df.columns:
            retention = pd.to_numeric(group_df["epoch_retention_percent"], errors="coerce").dropna()
            row.update({
                "epoch_retention_percent_mean": float(retention.mean()) if not retention.empty else np.nan,
                "epoch_retention_percent_sd": float(retention.std(ddof=1)) if len(retention) > 1 else np.nan,
            })

        # Bad / interpolated channels.
        if "has_bad_or_interpolated_channels" in group_df.columns:
            n_with_bad = int(group_df["has_bad_or_interpolated_channels"].sum())
            row["n_with_bad_or_interpolated_channels"] = n_with_bad
            row["percent_with_bad_or_interpolated_channels"] = 100.0 * n_with_bad / n_attempted if n_attempted else np.nan

        if "bad_channels" in group_df.columns:
            observed_bad_channels = sorted({channel for channels in group_df["bad_channels"] for channel in channels})
            row["bad_channels_observed"] = observed_bad_channels
            row["n_unique_bad_channels"] = len(observed_bad_channels)

        # ICA / EOG.
        if "has_excluded_ics" in group_df.columns:
            row["n_with_excluded_ics"] = int(group_df["has_excluded_ics"].sum())

        if "eog_available" in group_df.columns:
            eog_available = group_df["eog_available"].fillna(False).astype(bool)
            row["n_with_eog_available"] = int(eog_available.sum())
            row["percent_with_eog_available"] = 100.0 * eog_available.sum() / n_attempted if n_attempted else np.nan

        if "has_eog_supported_ics" in group_df.columns:
            row["n_with_eog_supported_ics"] = int(group_df["has_eog_supported_ics"].sum())

        if "has_qc_notes" in group_df.columns:
            n_with_notes = int(group_df["has_qc_notes"].sum())
            row["n_with_qc_notes"] = n_with_notes
            row["percent_with_qc_notes"] = 100.0 * n_with_notes / n_attempted if n_attempted else np.nan

        aggregate_rows.append(row)

    aggregate_df = pd.DataFrame(aggregate_rows)

    # =========================================================================
    # 4. BAD-CHANNEL RECURRENCE — PHYSICAL-RECORDING LEVEL
    # =========================================================================
    bad_channel_recurrence_rows: list[dict[str, Any]] = []

    if "bad_channels" in data.columns:
        recurrence_unit_column = "source_recording_id" if "source_recording_id" in data.columns else "recording_id"

        # Keep only grouping dimensions that are constant within a physical
        # recording. This automatically removes EO/EC or any future annotation-
        # derived condition that creates multiple logical records.
        recurrence_group_columns = _infer_physical_level_group_columns(
            data, group_columns_used, physical_recording_col=recurrence_unit_column
        )
        recurrence_grouped = (
            data.groupby(recurrence_group_columns, observed=True, dropna=False, sort=False)
            if recurrence_group_columns else [((), data)]
        )

        for group_key, group_df in recurrence_grouped:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_values = dict(zip(recurrence_group_columns, group_key))
            n_units = int(group_df[recurrence_unit_column].dropna().astype(str).nunique())
            channel_units: dict[str, set[str]] = {}

            for _, recording_row in group_df.iterrows():
                unit_id = recording_row.get(recurrence_unit_column)
                if pd.isna(unit_id):
                    continue
                for channel in recording_row.get("bad_channels", []):
                    channel_units.setdefault(str(channel), set()).add(str(unit_id))

            for channel in sorted(channel_units):
                n_affected = len(channel_units[channel])
                bad_channel_recurrence_rows.append({
                    **group_values,
                    "bad_channel": channel,
                    "n_recordings_affected": n_affected,
                    "n_recordings_total": n_units,
                    "percent_recordings_affected": 100.0 * n_affected / n_units if n_units else np.nan,
                })

    return {
        "all_recordings_df": all_recordings_df,
        "aggregate_df": aggregate_df,
        "bad_channel_recurrence_df": pd.DataFrame(bad_channel_recurrence_rows),
    }


# =============================================================================
# BAD-CHANNEL DETECTOR + ARTIFACT PROVENANCE QC
# =============================================================================

def build_bad_channel_method_qc(
    recording_qc_df: pd.DataFrame,
    *,
    physical_recording_col: str = "source_recording_id",
) -> dict[str, pd.DataFrame]:
    """
    Build physical-recording-level bad-channel provenance tables.

    For every interpolated channel, identify:
      - which bad-channel stage detected it: MAD, RANSAC, or both
      - whether the same physical recording showed EOG-supported ocular ICs
      - whether ICLabel excluded eye-blink or muscle-artifact components
      - whether an EOG-supported IC overlapped an actually excluded ICA component

    IMPORTANT
    ---------
    ICA/EOG occurs AFTER bad-channel interpolation in the current pipeline.
    These artifact fields therefore describe ASSOCIATION within the same
    physical recording; they do not prove that ocular/muscle activity caused
    the earlier bad-channel flag.
    """
    if not isinstance(recording_qc_df, pd.DataFrame):
        raise TypeError("recording_qc_df must be a pandas DataFrame.")
    if recording_qc_df.empty:
        raise ValueError("recording_qc_df is empty.")

    if physical_recording_col not in recording_qc_df.columns:
        if "recording_id" not in recording_qc_df.columns:
            raise KeyError(f"Neither '{physical_recording_col}' nor 'recording_id' is available.")
        physical_recording_col = "recording_id"

    data = recording_qc_df.copy()

    # Normalize stored list-like QC fields.
    def _as_list(value):
        if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)): return list(value)
        if value is None: return []
        try:
            if pd.isna(value): return []
        except (TypeError, ValueError):
            pass
        return [value]

    for col in ("mad_bad_channels", "ransac_bad_channels", "bad_channels",
                "excluded_ics", "excluded_ic_labels", "eog_candidate_ics"):
        if col not in data.columns: data[col] = [[] for _ in range(len(data))]
        else: data[col] = data[col].apply(_as_list)

    # Bad-channel detection and ICA/EOG are physical-recording operations.
    # EO/EC rows from the same EEG therefore must not be counted twice.
    physical_df = data.sort_values(physical_recording_col).drop_duplicates(physical_recording_col).copy()

    if "processing_status" in physical_df.columns:
        physical_df = physical_df[
            physical_df["processing_status"].astype(str).str.lower().eq("success")
        ].copy()

    n_physical_recordings = physical_df[physical_recording_col].dropna().astype(str).nunique()
    channel_rows = []

    # -------------------------------------------------------------------------
    # Create one row per physical recording × bad/interpolated channel.
    # -------------------------------------------------------------------------
    for _, row in physical_df.iterrows():
        mad = {str(x) for x in row["mad_bad_channels"]}
        ransac = {str(x) for x in row["ransac_bad_channels"]}
        final = {str(x) for x in row["bad_channels"]} | mad | ransac

        excluded_ics = {int(x) for x in row["excluded_ics"]}
        eog_ics = {int(x) for x in row["eog_candidate_ics"]}
        labels = [str(x).strip().lower() for x in row["excluded_ic_labels"]]

        # Recording-level downstream artifact evidence.
        has_eog_candidate = len(eog_ics) > 0
        has_eog_exclusion_overlap = len(excluded_ics & eog_ics) > 0
        has_eye_blink_ic = "eye blink" in labels
        has_muscle_ic = "muscle artifact" in labels
        has_ocular_or_muscle_evidence = (
            has_eog_candidate or has_eye_blink_ic or has_muscle_ic
        )

        for channel in sorted(final):
            by_mad, by_ransac = channel in mad, channel in ransac

            if by_mad and by_ransac: method = "MAD + RANSAC"
            elif by_mad: method = "MAD only"
            elif by_ransac: method = "RANSAC only"
            else: method = "Unknown / legacy"

            channel_rows.append({
                physical_recording_col: str(row[physical_recording_col]),
                "subject_id": row.get("subject_id"),
                "timepoint": row.get("timepoint"),
                "bad_channel": channel,

                # Bad-channel detector provenance
                "flagged_by_mad": by_mad,
                "flagged_by_ransac": by_ransac,
                "detection_method": method,

                # Downstream ICA/EOG artifact evidence
                "eog_available": bool(row.get("eog_available", False)),
                "n_eog_candidate_ics": len(eog_ics),
                "n_excluded_ics": len(excluded_ics),
                "excluded_ic_labels": list(row["excluded_ic_labels"]),
                "has_eog_candidate": has_eog_candidate,
                "has_eog_exclusion_overlap": has_eog_exclusion_overlap,
                "has_eye_blink_ic": has_eye_blink_ic,
                "has_muscle_ic": has_muscle_ic,
                "has_ocular_or_muscle_evidence": has_ocular_or_muscle_evidence,
            })

    channel_df = pd.DataFrame(channel_rows)

    # -------------------------------------------------------------------------
    # Summarize detector + artifact provenance for EVERY recurring channel.
    # -------------------------------------------------------------------------
    summary_rows = []

    for channel, group in channel_df.groupby("bad_channel", observed=True):
        n_affected = group[physical_recording_col].nunique()

        summary_rows.append({
            "bad_channel": channel,
            "n_recordings_affected": int(n_affected),
            "n_recordings_total": int(n_physical_recordings),
            "percent_recordings_affected": (
                100.0 * n_affected / n_physical_recordings
                if n_physical_recordings else np.nan
            ),

            # Detector stage
            "n_mad_only": int(group["detection_method"].eq("MAD only").sum()),
            "n_ransac_only": int(group["detection_method"].eq("RANSAC only").sum()),
            "n_mad_and_ransac": int(group["detection_method"].eq("MAD + RANSAC").sum()),

            # Downstream ocular / muscle evidence
            "n_with_eog_candidates": int(group["has_eog_candidate"].sum()),
            "n_with_eog_exclusion_overlap": int(group["has_eog_exclusion_overlap"].sum()),
            "n_with_eye_blink_ic": int(group["has_eye_blink_ic"].sum()),
            "n_with_muscle_ic": int(group["has_muscle_ic"].sum()),
            "n_with_ocular_or_muscle_evidence": int(group["has_ocular_or_muscle_evidence"].sum()),
        })

    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        # Add percentages relative to recordings in which that channel was flagged.
        for count_col, percent_col in (
            ("n_with_eog_candidates", "percent_with_eog_candidates"),
            ("n_with_eog_exclusion_overlap", "percent_with_eog_exclusion_overlap"),
            ("n_with_eye_blink_ic", "percent_with_eye_blink_ic"),
            ("n_with_muscle_ic", "percent_with_muscle_ic"),
            ("n_with_ocular_or_muscle_evidence", "percent_with_ocular_or_muscle_evidence"),
        ):
            summary_df[percent_col] = (
                100.0 * summary_df[count_col] / summary_df["n_recordings_affected"]
            )

        summary_df = summary_df.sort_values(
            ["n_recordings_affected", "bad_channel"],
            ascending=[False, True],
        ).reset_index(drop=True)

    return {
        "channel_df": channel_df,
        "summary_df": summary_df,
    }



# =============================================================================
# BAD-CHANNEL QC — REPORT-READY OUTPUT
# =============================================================================
def prepare_bad_channel_qc(bad_channel_method_qc: Mapping[str, Any], *, top_n: int = 10) -> dict[str, pd.DataFrame]:
    """
    Prepare final bad-channel QC tables for reporting.

    All recurrence, detector, and artifact calculations must already exist in
    build_bad_channel_method_qc(). This function only creates the final
    human-readable table so notebooks require no post-processing.
    """
    summary_df = bad_channel_method_qc["summary_df"].copy()
    detail_df = bad_channel_method_qc["channel_df"].copy()
    report_df = summary_df.head(int(top_n)).copy()

    report_df["Recordings affected"] = report_df["n_recordings_affected"].astype(int).astype(str) + "/" + report_df["n_recordings_total"].astype(int).astype(str)
    report_df["Percent affected"] = report_df["percent_recordings_affected"].map(lambda x: f"{x:.1f}%")

    report_df = report_df.rename(columns={
        "bad_channel": "Channel", "n_mad_only": "MAD only", "n_ransac_only": "RANSAC only",
        "n_mad_and_ransac": "MAD + RANSAC", "n_with_eog_candidates": "EOG-supported IC",
        "n_with_eog_exclusion_overlap": "EOG / ICA overlap", "n_with_eye_blink_ic": "Eye-blink IC",
        "n_with_muscle_ic": "Muscle IC", "n_with_ocular_or_muscle_evidence": "Ocular / muscle evidence",
    })

    report_df = report_df[[
        "Channel", "Recordings affected", "Percent affected",
        "MAD only", "RANSAC only", "MAD + RANSAC",
        "EOG-supported IC", "EOG / ICA overlap", "Eye-blink IC",
        "Muscle IC", "Ocular / muscle evidence",
    ]].reset_index(drop=True)

    return {"report_df": report_df, "summary_df": summary_df, "detail_df": detail_df}


# =============================================================================
# PREPROCESSING QC / DATA COMPLETENESS — SLIDE SUMMARY
# =============================================================================

# def build_qc_completeness_slide_summary(
#     qc_completeness_tables: Mapping[str, Any],
#     *,
#     group_filters: Mapping[str, Any] | None = None,
#     clean_minutes_decimals: int = 2,
#     retention_decimals: int = 1,
#     rejection_decimals: int = 1,
#     percent_decimals: int = 0,
# ) -> pd.DataFrame:
#     """
#     Build a compact, presentation-ready preprocessing QC table.

#     Purpose
#     -------
#     Summarize the major recording-quality and completeness findings needed
#     for cohort-level reporting.

#     The slide table includes:
#       - processing success
#       - usable clean EEG duration
#       - rejected epochs
#       - epoch retention
#       - specific bad/interpolated channel names
#       - EOG availability

#     Parameters
#     ----------
#     qc_completeness_tables
#         Output returned by summarize_qc_completeness().

#     group_filters
#         Optional filters used to select one cohort/timepoint/data cut.

#         Example:
#             {
#                 "cohort": "Cohort 1",
#                 "timepoint": "H1",
#             }

#     clean_minutes_decimals
#         Decimal places for usable clean EEG duration.

#     retention_decimals
#         Decimal places for epoch-retention percentage.

#     rejection_decimals
#         Decimal places for rejected-epoch percentage.

#     percent_decimals
#         Decimal places for count percentages.

#     Returns
#     -------
#     pd.DataFrame
#         Presentation-ready QC table with one row per eye state.
#     """
#     if not isinstance(qc_completeness_tables, Mapping):
#         raise TypeError(
#             "qc_completeness_tables must be a mapping."
#         )

#     aggregate_df = qc_completeness_tables.get(
#         "aggregate_df"
#     )

#     if not isinstance(aggregate_df, pd.DataFrame):
#         raise TypeError(
#             "qc_completeness_tables['aggregate_df'] must be "
#             "a pandas DataFrame."
#         )

#     if aggregate_df.empty:
#         raise ValueError(
#             "qc_completeness_tables['aggregate_df'] is empty."
#         )

#     data = aggregate_df.copy()

#     # ------------------------------------------------------------
#     # Apply optional cohort / timepoint / study filters
#     # ------------------------------------------------------------
#     if group_filters is not None:
#         for column, value in group_filters.items():

#             if column not in data.columns:
#                 raise KeyError(
#                     f"Filter column '{column}' is not present "
#                     "in aggregate_df."
#                 )

#             data = data.loc[
#                 data[column] == value
#             ].copy()

#         if data.empty:
#             raise ValueError(
#                 "No QC rows remain after applying "
#                 f"group_filters={dict(group_filters)}."
#             )

#     if "eye_state" not in data.columns:
#         raise KeyError(
#             "aggregate_df must contain 'eye_state'."
#         )

#     # ------------------------------------------------------------
#     # Prevent accidental mixing of multiple study cuts
#     # ------------------------------------------------------------
#     study_group_columns = [
#         column
#         for column in (
#             "cohort",
#             "visit",
#             "timepoint",
#             "dose",
#         )
#         if (
#             column in data.columns
#             and data[column].notna().any()
#         )
#     ]

#     multiple_group_columns = [
#         column
#         for column in study_group_columns
#         if data[column].dropna().nunique() > 1
#     ]

#     if multiple_group_columns:
#         raise ValueError(
#             "The selected QC data contain multiple study groups "
#             f"for {multiple_group_columns}. "
#             "Use group_filters to select the study cut intended "
#             "for this slide."
#         )

#     # ============================================================
#     # Formatting helpers
#     # ============================================================
#     def format_count_percent(
#         count: Any,
#         total: Any,
#         percent: Any,
#     ) -> str:
#         """Format recording count as n/N (percent)."""
#         if pd.isna(count) or pd.isna(total):
#             return "N/A"

#         count = int(count)
#         total = int(total)

#         percent = pd.to_numeric(
#             pd.Series([percent]),
#             errors="coerce",
#         ).iloc[0]

#         if np.isfinite(percent):
#             return (
#                 f"{count}/{total} "
#                 f"({percent:.{percent_decimals}f}%)"
#             )

#         return f"{count}/{total}"

#     def format_mean_sd(
#         mean_value: Any,
#         sd_value: Any,
#         *,
#         decimals: int,
#         suffix: str = "",
#     ) -> str:
#         """
#         Format mean ± SD.

#         When N=1, SD is undefined and only the observed value is shown.
#         """
#         mean_value = pd.to_numeric(
#             pd.Series([mean_value]),
#             errors="coerce",
#         ).iloc[0]

#         sd_value = pd.to_numeric(
#             pd.Series([sd_value]),
#             errors="coerce",
#         ).iloc[0]

#         if not np.isfinite(mean_value):
#             return "N/A"

#         if np.isfinite(sd_value):
#             return (
#                 f"{mean_value:.{decimals}f} ± "
#                 f"{sd_value:.{decimals}f}{suffix}"
#             )

#         return (
#             f"{mean_value:.{decimals}f}{suffix}"
#         )

#     def format_rejected_epochs(
#         rejected: Any,
#         attempted: Any,
#         percent: Any,
#     ) -> str:
#         """Format rejected epochs as rejected/attempted (percent)."""
#         if pd.isna(rejected) or pd.isna(attempted):
#             return "N/A"

#         rejected = int(rejected)
#         attempted = int(attempted)

#         percent = pd.to_numeric(
#             pd.Series([percent]),
#             errors="coerce",
#         ).iloc[0]

#         if np.isfinite(percent):
#             return (
#                 f"{rejected}/{attempted} "
#                 f"({percent:.{rejection_decimals}f}%)"
#             )

#         return f"{rejected}/{attempted}"

#     def format_bad_channels(
#         channels: Any,
#     ) -> str:
#         """Show the specific bad/interpolated electrode names."""
#         if not isinstance(
#             channels,
#             (
#                 list,
#                 tuple,
#                 set,
#                 np.ndarray,
#                 pd.Series,
#             ),
#         ):
#             if channels is None:
#                 return "None"

#             try:
#                 if pd.isna(channels):
#                     return "None"
#             except (TypeError, ValueError):
#                 pass

#             channels = [
#                 channels
#             ]

#         channel_names = [
#             str(channel)
#             for channel in channels
#             if str(channel).strip()
#         ]

#         if not channel_names:
#             return "None"

#         return ", ".join(
#             channel_names
#         )

#     # ============================================================
#     # Build slide-level QC rows
#     # ============================================================
#     slide_rows: list[dict[str, Any]] = []

#     for eye_state in (
#         "EO",
#         "EC",
#     ):

#         eye_rows = data.loc[
#             data["eye_state"] == eye_state
#         ]

#         if eye_rows.empty:
#             continue

#         if len(eye_rows) > 1:
#             raise ValueError(
#                 f"More than one aggregate row remains for "
#                 f"eye_state='{eye_state}'. "
#                 "Use group_filters to select one study group."
#             )

#         row = eye_rows.iloc[0]

#         # --------------------------------------------------------
#         # Processing success
#         # --------------------------------------------------------
#         n_attempted = row.get(
#             "n_attempted"
#         )

#         n_successful = row.get(
#             "n_successful"
#         )

#         percent_successful = row.get(
#             "percent_processing_successful"
#         )

#         # --------------------------------------------------------
#         # EOG availability
#         # --------------------------------------------------------
#         n_with_eog = row.get(
#             "n_with_eog_available"
#         )

#         percent_with_eog = row.get(
#             "percent_with_eog_available"
#         )

#         # --------------------------------------------------------
#         # Presentation-ready row
#         # --------------------------------------------------------
#         slide_rows.append({
#             "Eye State":
#                 eye_state,

#             "Successful":
#                 format_count_percent(
#                     n_successful,
#                     n_attempted,
#                     percent_successful,
#                 ),

#             "Usable Clean EEG":
#                 format_mean_sd(
#                     row.get(
#                         "usable_clean_minutes_mean"
#                     ),
#                     row.get(
#                         "usable_clean_minutes_sd"
#                     ),
#                     decimals=clean_minutes_decimals,
#                     suffix=" min",
#                 ),

#             "Rejected Epochs":
#                 format_rejected_epochs(
#                     row.get(
#                         "total_epochs_rejected"
#                     ),
#                     row.get(
#                         "total_epochs_attempted"
#                     ),
#                     row.get(
#                         "overall_epoch_rejection_percent"
#                     ),
#                 ),

#             "Epoch Retention":
#                 format_mean_sd(
#                     row.get(
#                         "epoch_retention_percent_mean"
#                     ),
#                     row.get(
#                         "epoch_retention_percent_sd"
#                     ),
#                     decimals=retention_decimals,
#                     suffix="%",
#                 ),

#             "Bad / Interpolated Channels":
#                 format_bad_channels(
#                     row.get(
#                         "bad_channels_observed"
#                     )
#                 ),

#             "EOG Available":
#                 format_count_percent(
#                     n_with_eog,
#                     n_attempted,
#                     percent_with_eog,
#                 ),
#         })

#     slide_summary_df = pd.DataFrame(
#         slide_rows
#     )

#     return slide_summary_df


def build_qc_completeness_slide_summary(
    qc_completeness_tables: Mapping[str, Any],
    *,
    group_col: str | None = None,
    group_order: Sequence[Any] | None = None,
    group_label: str | None = None,
    group_filters: Mapping[str, Any] | None = None,
    clean_minutes_decimals: int = 2,
    retention_decimals: int = 1,
    rejection_decimals: int = 1,
    percent_decimals: int = 0,
) -> pd.DataFrame:
    """
    Build a compact preprocessing QC table for any logical-recording grouping.

    Examples:
        NeuShen -> group_col="eye_state", group_order=("EO", "EC"), group_label="Eye State"
        ABC-CT  -> group_col="label", group_order=("ASD",), group_label="Group"
        Overall -> group_col=None
    """
    if not isinstance(qc_completeness_tables, Mapping):
        raise TypeError("qc_completeness_tables must be a mapping.")

    aggregate_df = qc_completeness_tables.get("aggregate_df")
    if not isinstance(aggregate_df, pd.DataFrame):
        raise TypeError("qc_completeness_tables['aggregate_df'] must be a pandas DataFrame.")
    if aggregate_df.empty:
        raise ValueError("qc_completeness_tables['aggregate_df'] is empty.")

    data = aggregate_df.copy()

    # ------------------------------------------------------------------
    # Apply optional study/group filters
    # ------------------------------------------------------------------
    for column, value in dict(group_filters or {}).items():
        if column not in data.columns:
            raise KeyError(f"Filter column '{column}' is not present in aggregate_df.")
        data = data.loc[data[column].astype(str) == str(value)].copy()

    if data.empty:
        raise ValueError("No QC rows remain after applying group_filters.")

    # Auto-select a useful row grouping when none is supplied.
    if group_col is None:
        for candidate in ("eye_state", "analysis_condition", "condition", "label"):
            if candidate in data.columns and data[candidate].notna().any():
                group_col = candidate
                break

    if group_col is None:
        data["_report_group"] = "Overall"
        group_col_used = "_report_group"
        group_order_used = ["Overall"]
        group_label_used = group_label or "Group"
    else:
        if group_col not in data.columns:
            raise KeyError(f"aggregate_df does not contain group_col='{group_col}'.")
        group_col_used = group_col
        observed = data[group_col_used].dropna().drop_duplicates().tolist()
        group_order_used = list(group_order) if group_order is not None else observed
        group_label_used = group_label or group_col.replace("_", " ").title()

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def format_count_percent(count, total, percent):
        if pd.isna(count) or pd.isna(total):
            return "N/A"
        percent = pd.to_numeric(pd.Series([percent]), errors="coerce").iloc[0]
        return (
            f"{int(count)}/{int(total)} ({percent:.{percent_decimals}f}%)"
            if np.isfinite(percent) else f"{int(count)}/{int(total)}"
        )

    def format_mean_sd(mean_value, sd_value, *, decimals, suffix=""):
        mean_value = pd.to_numeric(pd.Series([mean_value]), errors="coerce").iloc[0]
        sd_value = pd.to_numeric(pd.Series([sd_value]), errors="coerce").iloc[0]
        if not np.isfinite(mean_value):
            return "N/A"
        return (
            f"{mean_value:.{decimals}f} ± {sd_value:.{decimals}f}{suffix}"
            if np.isfinite(sd_value) else f"{mean_value:.{decimals}f}{suffix}"
        )

    def format_rejected_epochs(rejected, attempted, percent):
        if pd.isna(rejected) or pd.isna(attempted):
            return "N/A"
        percent = pd.to_numeric(pd.Series([percent]), errors="coerce").iloc[0]
        return (
            f"{int(rejected)}/{int(attempted)} ({percent:.{rejection_decimals}f}%)"
            if np.isfinite(percent) else f"{int(rejected)}/{int(attempted)}"
        )

    def format_bad_channels(channels):
        if isinstance(channels, (list, tuple, set, np.ndarray, pd.Series)):
            names = [str(channel) for channel in channels if str(channel).strip()]
            return ", ".join(names) if names else "None"
        if channels is None:
            return "None"
        try:
            if pd.isna(channels):
                return "None"
        except (TypeError, ValueError):
            pass
        return str(channels)

    # ------------------------------------------------------------------
    # Build one compact row per requested group
    # ------------------------------------------------------------------
    rows = []

    for group in group_order_used:
        current = data.loc[data[group_col_used].astype(str) == str(group)]
        if current.empty:
            continue
        if len(current) > 1:
            raise ValueError(
                f"More than one aggregate QC row remains for {group_col_used}='{group}'. "
                "Use group_filters or choose additional aggregation before building this table."
            )

        row = current.iloc[0]
        n_attempted = row.get("n_attempted")
        report_row = {
            group_label_used: str(group),
            "Successful": format_count_percent(
                row.get("n_successful"), n_attempted, row.get("percent_processing_successful")
            ),
            "Usable Clean EEG": format_mean_sd(
                row.get("usable_clean_minutes_mean"), row.get("usable_clean_minutes_sd"),
                decimals=clean_minutes_decimals, suffix=" min",
            ),
            "Rejected Epochs": format_rejected_epochs(
                row.get("total_epochs_rejected"), row.get("total_epochs_attempted"),
                row.get("overall_epoch_rejection_percent"),
            ),
            "Epoch Retention": format_mean_sd(
                row.get("epoch_retention_percent_mean"), row.get("epoch_retention_percent_sd"),
                decimals=retention_decimals, suffix="%",
            ),
            "Bad / Interpolated Channels": format_bad_channels(row.get("bad_channels_observed")),
            "EOG Available": format_count_percent(
                row.get("n_with_eog_available"), n_attempted, row.get("percent_with_eog_available")
            ),
        }

        if "n_meeting_clean_minutes" in row.index:
            minimum = row.get("minimum_clean_minutes_required")
            label = f">= {minimum:g} min" if pd.notna(minimum) else "Clean-duration criterion"
            report_row[label] = format_count_percent(
                row.get("n_meeting_clean_minutes"), n_attempted, row.get("percent_meeting_clean_minutes")
            )

        rows.append(report_row)

    return pd.DataFrame(rows)



# =============================================================================
# RECORDING PREPARATION AND INSPECTION
# =============================================================================



# Pair cleaned MNE Epochs objects with their recording-level metadata.
def build_recordings_from_epochs(
    label_epoch_arrays: Mapping[str, Sequence[mne.BaseEpochs | Mapping[str, mne.BaseEpochs]]],
    metadata: Sequence[Mapping[str, Any]],
    *,
    label: str | Sequence[str] | None = None,
    condition_to_eye_state: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Pair cleaned Epochs with metadata and create logical analysis records.

    This is the canonical bridge between preprocessing outputs and Part 1 qEEG.
    It supports both ordinary one-recording/one-Epochs inputs and condition-aware
    inputs in which one physical recording yields multiple logical records.

    Parameters
    ----------
    label_epoch_arrays
        Mapping returned by ``build_label_epoch_arrays``. Each label contains
        one entry per physical recording. An entry may be either a single
        MNE Epochs object or a mapping such as ``{"EC": epochs_ec, "EO": epochs_eo}``.

    metadata
        Recording metadata returned by ``build_label_epoch_arrays``.

    label
        Label or labels to include. If None, all labels in ``label_epoch_arrays``
        are processed. This removes the need for notebook-level loops over labels.

    condition_to_eye_state
        Optional mapping from analysis-condition names to eye-state labels.
        Defaults to ``{"EO": "EO", "EC": "EC"}``.

    Returns
    -------
    list[dict[str, Any]]
        Standardized logical recording dictionaries. Each returned recording
        contains a unique ``recording_id``, ``source_recording_id``, condition
        metadata when applicable, ``qc_idx`` when available, and ``epochs_clean``.

    Notes
    -----
    The function now writes both ``analysis_condition`` and the compatibility
    alias ``condition`` directly. Downstream code should not need to patch
    condition metadata manually.
    """
    if not isinstance(label_epoch_arrays, Mapping) or not label_epoch_arrays:
        raise ValueError("label_epoch_arrays must be a non-empty mapping.")
    if not metadata:
        raise ValueError("metadata is empty.")

    if condition_to_eye_state is None:
        condition_to_eye_state = {"EO": "EO", "EC": "EC"}
    if not isinstance(condition_to_eye_state, Mapping):
        raise TypeError("condition_to_eye_state must be a mapping or None.")
    condition_to_eye_state = {
        str(key): str(value)
        for key, value in condition_to_eye_state.items()
    }

    # Resolve which labels to process. None means all available labels.
    if label is None:
        labels_to_process = list(label_epoch_arrays.keys())
    elif isinstance(label, str):
        labels_to_process = [label]
    else:
        labels_to_process = [str(value) for value in label]

    if not labels_to_process:
        raise ValueError("No labels were selected.")
    if len(set(labels_to_process)) != len(labels_to_process):
        raise ValueError("label contains duplicate values.")

    missing_labels = [
        current_label
        for current_label in labels_to_process
        if current_label not in label_epoch_arrays
    ]
    if missing_labels:
        raise KeyError(
            "The following labels are not present in label_epoch_arrays: "
            f"{missing_labels}"
        )

    recordings: list[dict[str, Any]] = []
    used_recording_ids: set[str] = set()

    for current_label in labels_to_process:
        epochs_list = list(label_epoch_arrays[current_label])
        label_metadata = [
            dict(row)
            for row in metadata
            if row.get("label") == current_label
        ]

        # Index metadata explicitly by label_idx so the relationship between
        # metadata and the preprocessing Epochs list is validated once here.
        metadata_by_label_idx: dict[int, dict[str, Any]] = {}
        for row in label_metadata:
            if "label_idx" not in row:
                raise KeyError(
                    f"Metadata for label '{current_label}' is missing 'label_idx'."
                )
            label_idx = int(row["label_idx"])
            if label_idx in metadata_by_label_idx:
                raise ValueError(
                    f"Duplicate label_idx={label_idx} for label '{current_label}'."
                )
            metadata_by_label_idx[label_idx] = row

        expected_indices = set(range(len(epochs_list)))
        if set(metadata_by_label_idx) != expected_indices:
            raise ValueError(
                f"Metadata/Epochs index mismatch for label '{current_label}'. "
                f"Expected label_idx values {sorted(expected_indices)}, "
                f"received {sorted(metadata_by_label_idx)}."
            )

        for label_idx, epochs_entry in enumerate(epochs_list):
            row = dict(metadata_by_label_idx[label_idx])
            source_recording_id = str(
                row.get("source_recording_id")
                or row.get("recording_id")
                or Path(str(row.get("file_path", f"recording_{label_idx}"))).stem
            )

            # Map named analysis conditions to their condition-specific QC rows.
            condition_names = [str(x) for x in row.get("analysis_conditions", [])]
            qc_indices = list(row.get("qc_indices", []))
            condition_qc_map = (
                dict(zip(condition_names, qc_indices))
                if len(condition_names) == len(qc_indices)
                else {}
            )

            # ------------------------------------------------------------
            # Standard recording: one physical recording -> one logical record
            # ------------------------------------------------------------
            if isinstance(epochs_entry, mne.BaseEpochs):
                if len(epochs_entry) == 0:
                    raise ValueError(
                        f"Recording '{source_recording_id}' contains no retained epochs."
                    )

                recording = dict(row)
                recording_id = str(
                    recording.get("recording_id")
                    or source_recording_id
                )
                analysis_condition = recording.get(
                    "analysis_condition",
                    recording.get("condition"),
                )

                recording.update({
                    "recording_id": recording_id,
                    "source_recording_id": source_recording_id,
                    "label": current_label,
                    "label_idx": label_idx,
                    "epochs_clean": epochs_entry,
                })

                if recording.get("global_idx") is not None:
                    recording["global_idx"] = int(recording["global_idx"])

                if analysis_condition is not None:
                    analysis_condition = str(analysis_condition)
                    recording["analysis_condition"] = analysis_condition
                    recording["condition"] = analysis_condition
                    if recording.get("eye_state") is None:
                        mapped_eye_state = condition_to_eye_state.get(analysis_condition)
                        if mapped_eye_state is not None:
                            recording["eye_state"] = mapped_eye_state

                if recording_id in used_recording_ids:
                    raise ValueError(f"Duplicate recording_id: {recording_id}")
                used_recording_ids.add(recording_id)
                recordings.append(recording)
                continue

            # ------------------------------------------------------------
            # Condition-aware recording: one physical file -> logical records
            # ------------------------------------------------------------
            if not isinstance(epochs_entry, Mapping):
                raise TypeError(
                    f"Epoch entry at label_idx={label_idx} must be an MNE Epochs "
                    f"object or condition mapping; got {type(epochs_entry).__name__}."
                )
            if not epochs_entry:
                raise ValueError(
                    f"Condition mapping at label_idx={label_idx} is empty."
                )

            for condition_idx, (condition, epochs) in enumerate(epochs_entry.items()):
                condition = str(condition)
                if not isinstance(epochs, mne.BaseEpochs):
                    raise TypeError(
                        f"Condition '{condition}' at label_idx={label_idx} must contain "
                        f"an MNE Epochs object; got {type(epochs).__name__}."
                    )
                if len(epochs) == 0:
                    raise ValueError(
                        f"Condition '{condition}' at label_idx={label_idx} "
                        "contains no retained epochs."
                    )

                recording_id = f"{source_recording_id}__{condition}"
                if recording_id in used_recording_ids:
                    raise ValueError(
                        f"Duplicate logical recording_id: {recording_id}"
                    )
                used_recording_ids.add(recording_id)

                recording = dict(row)
                recording.update({
                    "recording_id": recording_id,
                    "source_recording_id": source_recording_id,
                    "label": current_label,
                    "analysis_condition": condition,
                    "condition": condition,
                    "condition_idx": condition_idx,
                    "label_idx": label_idx,
                    "epochs_clean": epochs,
                })

                if recording.get("global_idx") is not None:
                    recording["global_idx"] = int(recording["global_idx"])

                # Point each logical record to its own condition-specific QC row.
                if condition in condition_qc_map:
                    recording["qc_idx"] = int(condition_qc_map[condition])
                elif condition_idx < len(qc_indices):
                    recording["qc_idx"] = int(qc_indices[condition_idx])

                # Only mapped conditions populate eye_state automatically.
                mapped_eye_state = condition_to_eye_state.get(condition)
                if mapped_eye_state is not None:
                    recording["eye_state"] = mapped_eye_state

                recordings.append(recording)

    if not recordings:
        raise ValueError("No logical recordings were created.")

    return recordings


# -----------------------------------------------------------------------------
# Print identifying information and dimensions for a sample of recordings.
def inspect_recordings(
    recordings: Sequence[Mapping[str, Any]],
    *,
    n: int = 5,
    picks: str | Sequence[str] = "eeg",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Inspect a sample of logical recordings and return a compact QC table.

    The function remains useful during development, but it is no longer a
    required notebook step. ``run_qeeg_part1_pipeline`` can call it optionally.

    Parameters
    ----------
    recordings
        Recording dictionaries containing ``epochs_clean``.

    n
        Maximum number of recordings to inspect.

    picks
        Channels expected to enter qEEG. The production default is ``"eeg"``.

    verbose
        If True, print a readable summary for each inspected recording.

    Returns
    -------
    pd.DataFrame
        One row per inspected logical recording.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    if not recordings:
        raise ValueError("recordings is empty.")

    rows: list[dict[str, Any]] = []

    for recording in recordings[:n]:
        epochs = recording.get("epochs_clean")
        if not isinstance(epochs, mne.BaseEpochs):
            raise TypeError("epochs_clean must be an MNE Epochs object.")

        n_channels_total = len(epochs.ch_names)

        # Match the production qEEG selection for the common EEG-only case.
        # For custom picks, use an MNE copy so channel-type/name selection follows
        # the same public MNE picking rules used elsewhere in the framework.
        if picks == "eeg":
            qeeg_indices = mne.pick_types(
                epochs.info,
                eeg=True,
                meg=False,
                stim=False,
                misc=False,
                exclude="bads",
            )
            qeeg_ch_names = [epochs.ch_names[index] for index in qeeg_indices]
        else:
            picked_epochs = epochs.copy().pick(picks, exclude="bads")
            qeeg_ch_names = list(picked_epochs.ch_names)

        row = {
            "recording_id": recording.get("recording_id"),
            "source_recording_id": recording.get("source_recording_id"),
            "subject_id": recording.get("subject_id"),
            "label": recording.get("label"),
            "analysis_condition": recording.get("analysis_condition"),
            "eye_state": recording.get("eye_state"),
            "timepoint": recording.get("timepoint"),
            "n_epochs_clean": int(len(epochs)),
            "n_channels_total": int(n_channels_total),
            "n_channels_qeeg": int(len(qeeg_ch_names)),
            "qeeg_ch_names": qeeg_ch_names,
            "sfreq_hz": float(epochs.info["sfreq"]),
            "n_samples_per_epoch": int(len(epochs.times)),
            "stored_data_shape": (
                int(len(epochs)),
                int(n_channels_total),
                int(len(epochs.times)),
            ),
            "qeeg_data_shape": (
                int(len(epochs)),
                int(len(qeeg_ch_names)),
                int(len(epochs.times)),
            ),
        }
        rows.append(row)

        if verbose:
            print("=" * 100)
            print(f"Recording ID:         {row['recording_id'] or 'Unknown'}")
            print(f"Subject ID:           {row['subject_id'] or 'Unknown'}")
            print(f"Source file:          {recording.get('file_path', 'Unknown')}")
            print(f"Analysis condition:   {row['analysis_condition'] or 'N/A'}")
            print(f"Eye state:            {row['eye_state'] or 'N/A'}")
            print(f"Timepoint:            {row['timepoint'] or 'N/A'}")
            print(f"Clean epochs:         {row['n_epochs_clean']}")
            print(f"Total channels:       {row['n_channels_total']}")
            print(f"qEEG channels:        {row['n_channels_qeeg']}")
            print(f"Sampling rate:        {row['sfreq_hz']:.1f} Hz")
            print(f"Samples/epoch:        {row['n_samples_per_epoch']}")
            print(f"Stored data shape:    {row['stored_data_shape']}")
            print(f"qEEG data shape:      {row['qeeg_data_shape']}")

    return pd.DataFrame(rows)

        

# =============================================================================
# BATCH qEEG ANALYSIS
# =============================================================================
# Run the complete qEEG calculation pipeline across cleaned recordings.
def run_qeeg_batch_analysis(
    recordings: Sequence[Mapping[str, Any]],
    *,
    bands: Mapping[str, tuple[float, float]],
    ratio_definitions: Mapping[str, tuple[str, str]],
    psd_range_hz: tuple[float, float] = (0.5, 45.0),
    total_range_hz: tuple[float, float] = (1.0, 45.0),
    relative_power_bands: Sequence[str] | None = None,
    picks: str | Sequence[str] = "eeg",
    ratio_summary_method: Literal["ratio_of_means", "mean_of_ratios"] = "ratio_of_means",
    psd_kwargs: Mapping[str, Any] | None = None,
    log_mode: Literal["summary", "detailed", "silent"] = "summary",
    progress_every: int = 25,
) -> dict[str, dict[str, Any]]:
    """
    Run conventional qEEG calculations for multiple logical recordings.

    Each logical recording is analyzed independently for:
      1. Mean power spectral density (PSD).
      2. Absolute band power.
      3. Relative band power for the configured subset of bands.
      4. Configured spectral-power ratios.

    The absolute-power band definitions and relative-power band definitions
    may now differ. This allows the PSD and absolute-power analysis to include
    bands outside the established relative-power denominator.

    Example
    -------
    NeuShen expanded-Gamma configuration:

        bands = {
            "delta": (1.0, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "low_gamma": (30.0, 45.0),
            "high_gamma": (55.0, 80.0),
        }

        relative_power_bands = (
            "delta", "theta", "alpha", "beta", "low_gamma"
        )

    This preserves the original 1-45 Hz relative-power denominator while
    allowing High Gamma to be retained as an absolute-power endpoint.

    Parameters
    ----------
    recordings
        Standardized logical recording dictionaries containing ``epochs_clean``.

    bands
        Frequency-band definitions used for absolute power.

    ratio_definitions
        Ratio names mapped to numerator and denominator band names.

    psd_range_hz
        Frequency range retained for PSD calculation.

    total_range_hz
        Frequency range used as the relative-power denominator.

    relative_power_bands
        Optional subset of ``bands`` converted to relative-power endpoints.
        If None, all configured bands are used, preserving legacy behavior.

    picks
        Channels included in qEEG spectral analysis. Default ``"eeg"`` keeps
        EOG, trigger, and other auxiliary channels out of qEEG endpoints.

    ratio_summary_method
        Primary spectral-ratio summary method.

    psd_kwargs
        Optional keyword arguments passed to ``calculate_mean_psd_curve``.
        Core pipeline values such as frequency range, picks, and plotting
        controls remain managed by this function.

    log_mode
        ``"summary"``, ``"detailed"``, or ``"silent"``.

    progress_every
        Number of completed recordings between summary progress messages.

    Returns
    -------
    dict[str, dict[str, Any]]
        qEEG results and complete logical-record metadata keyed by recording ID.
    """
    # ------------------------------------------------------------------
    # Validate core inputs
    # ------------------------------------------------------------------
    if not recordings: raise ValueError("recordings is empty.")
    if not bands: raise ValueError("bands is empty.")
    if not ratio_definitions: raise ValueError("ratio_definitions is empty.")
    if log_mode not in {"summary", "detailed", "silent"}:
        raise ValueError("log_mode must be 'summary', 'detailed', or 'silent'.")
    if progress_every < 1: raise ValueError("progress_every must be at least 1.")

    # ------------------------------------------------------------------
    # Resolve the relative-power band subset once for the complete batch
    # ------------------------------------------------------------------
    # None preserves the original behavior: every absolute-power band also
    # becomes a relative-power endpoint.
    absolute_band_names = tuple(str(band) for band in bands)
    relative_power_bands_used = (
        absolute_band_names if relative_power_bands is None
        else tuple(str(band) for band in relative_power_bands)
    )

    if not relative_power_bands_used:
        raise ValueError("relative_power_bands must contain at least one band.")
    if len(set(relative_power_bands_used)) != len(relative_power_bands_used):
        raise ValueError("relative_power_bands contains duplicate band names.")

    missing_relative_bands = [band for band in relative_power_bands_used if band not in absolute_band_names]
    if missing_relative_bands:
        raise ValueError(
            "relative_power_bands contains bands not present in bands: "
            f"{missing_relative_bands}"
        )

    # ------------------------------------------------------------------
    # Validate PSD range and optional Welch overrides
    # ------------------------------------------------------------------
    fmin, fmax = map(float, psd_range_hz)
    if fmin >= fmax: raise ValueError("psd_range_hz must satisfy fmin < fmax.")

    psd_options = dict(psd_kwargs or {})
    protected_psd_keys = {"fmin", "fmax", "picks", "plot", "show_channel_curves"}
    conflicting_psd_keys = protected_psd_keys.intersection(psd_options)
    if conflicting_psd_keys:
        raise ValueError(
            "psd_kwargs cannot override pipeline-controlled arguments: "
            f"{sorted(conflicting_psd_keys)}"
        )

    # ------------------------------------------------------------------
    # Initialize batch processing
    # ------------------------------------------------------------------
    n_recordings = len(recordings)
    results_by_recording: dict[str, dict[str, Any]] = {}
    batch_start = perf_counter()

    if log_mode == "summary":
        print(f"Starting qEEG analysis for {n_recordings} recordings...")

    # ==================================================================
    # Process each logical recording independently
    # ==================================================================
    for index, recording in enumerate(recordings, start=1):
        if "epochs_clean" not in recording:
            raise KeyError(f"Recording {index} is missing 'epochs_clean'.")

        epochs = recording["epochs_clean"]
        if not isinstance(epochs, mne.BaseEpochs):
            raise TypeError(f"Recording {index} epochs_clean must be an MNE Epochs object.")

        recording_id = recording.get("recording_id")
        if not recording_id:
            file_path = recording.get("file_path")
            if not file_path:
                raise KeyError(f"Recording {index} requires 'recording_id' or 'file_path'.")
            recording_id = Path(str(file_path)).stem

        recording_id = str(recording_id)
        if recording_id in results_by_recording:
            raise ValueError(f"Duplicate recording_id: {recording_id}")

        # Preserve all logical-record metadata except the large Epochs object.
        recording_metadata = {key: value for key, value in recording.items() if key != "epochs_clean"}

        # Ensure the compatibility condition field remains available for older
        # downstream code when only analysis_condition was originally supplied.
        if recording_metadata.get("condition") is None and recording_metadata.get("analysis_condition") is not None:
            recording_metadata["condition"] = recording_metadata["analysis_condition"]

        recording_start = perf_counter()

        if log_mode == "detailed":
            print("\n" + "=" * 80)
            print(f"[{index}/{n_recordings}] qEEG analysis: {recording_id}")
            print(
                f"Subject: {recording_metadata.get('subject_id', 'Unknown')} | "
                f"Condition: {recording_metadata.get('analysis_condition', 'N/A')} | "
                f"Timepoint: {recording_metadata.get('timepoint', 'N/A')} | "
                f"Epochs: {len(epochs)} | Channels: {len(epochs.ch_names)} | "
                f"Sampling rate: {epochs.info['sfreq']:.1f} Hz"
            )
            print("=" * 80)

        try:
            # ----------------------------------------------------------
            # 1. Mean power spectral density
            # ----------------------------------------------------------
            if log_mode == "detailed":
                print(f"[1/4] Calculating mean PSD from {fmin:g}-{fmax:g} Hz...")

            mean_psd_result = calculate_mean_psd_curve(
                epochs, fmin=fmin, fmax=fmax, picks=picks, plot=False,
                show_channel_curves=False, **psd_options
            )

            qeeg_ch_names = list(mean_psd_result["ch_names"])
            n_channels_qeeg, n_channels_total = len(qeeg_ch_names), len(epochs.ch_names)

            if log_mode == "detailed":
                print(f"qEEG channels: {n_channels_qeeg} of {n_channels_total} retained channels.")

            # ----------------------------------------------------------
            # 2. Absolute band power
            # ----------------------------------------------------------
            # Every configured band is calculated here. For NeuShen this
            # includes both Low Gamma (30-45 Hz) and High Gamma (55-80 Hz).
            if log_mode == "detailed":
                print(f"[2/4] Calculating absolute power: {', '.join(absolute_band_names)}...")

            absolute_power_result = calculate_absolute_band_power(
                mean_psd_result, bands=dict(bands), plot=False
            )

            # ----------------------------------------------------------
            # 3. Relative band power
            # ----------------------------------------------------------
            # Only relative_power_bands_used enter this calculation.
            # This allows High Gamma to remain an absolute endpoint while
            # preserving NeuShen's established 1-45 Hz relative denominator.
            if log_mode == "detailed":
                print(
                    f"[3/4] Calculating relative power: "
                    f"{', '.join(relative_power_bands_used)} | "
                    f"denominator {total_range_hz[0]:g}-{total_range_hz[1]:g} Hz..."
                )

            relative_power_result = calculate_relative_band_power(
                mean_psd_result, absolute_power_result,
                total_range_hz=total_range_hz,
                relative_bands=relative_power_bands_used,
                plot=False,
            )

            # ----------------------------------------------------------
            # 4. Spectral-power ratios
            # ----------------------------------------------------------
            # Ratios continue to use the absolute-power band results and
            # therefore remain independent of the relative-power subset.
            if log_mode == "detailed":
                print(f"[4/4] Calculating ratios: {', '.join(ratio_definitions)}...")

            spectral_ratio_result = calculate_spectral_power_ratios(
                absolute_power_result, ratio_definitions=dict(ratio_definitions),
                summary_method=ratio_summary_method, plot=False
            )

        except Exception as exc:
            raise RuntimeError(f"qEEG analysis failed for recording '{recording_id}'.") from exc

        # ------------------------------------------------------------------
        # Preserve metadata, calculated outputs, and analysis settings
        # ------------------------------------------------------------------
        recording_result = dict(recording_metadata)
        recording_result.update({
            "recording_id": recording_id,
            "n_epochs_clean": int(len(epochs)),
            "sfreq_hz": float(epochs.info["sfreq"]),
            "n_samples_per_epoch": int(len(epochs.times)),
            "n_channels_total": int(n_channels_total),
            "n_channels_qeeg": int(n_channels_qeeg),
            "qeeg_ch_names": qeeg_ch_names,

            # Backward-compatible legacy field. New reporting should use the
            # explicit total/qEEG channel fields above.
            "n_channels": int(n_channels_total),

            # Retain the cleaned Epochs reference for topomap geometry and
            # later subject-level or longitudinal visualizations.
            "epochs_clean": epochs,

            "mean_psd_result": mean_psd_result,
            "absolute_power_result": absolute_power_result,
            "relative_power_result": relative_power_result,
            "spectral_ratio_result": spectral_ratio_result,

            # Store the exact numerical configuration for traceability.
            "analysis_settings": {
                "psd_range_hz": (fmin, fmax),
                "total_range_hz": tuple(map(float, total_range_hz)),
                "bands": {str(name): tuple(map(float, limits)) for name, limits in bands.items()},
                "relative_power_bands": tuple(relative_power_bands_used),
                "ratio_definitions": {
                    str(name): (str(value[0]), str(value[1]))
                    for name, value in ratio_definitions.items()
                },
                "ratio_summary_method": str(ratio_summary_method),
                "picks": picks,
                "psd_kwargs": dict(psd_options),
            },
        })
        results_by_recording[recording_id] = recording_result

        # ------------------------------------------------------------------
        # Progress reporting
        # ------------------------------------------------------------------
        if log_mode == "detailed":
            print(f"Completed {recording_id} in {perf_counter() - recording_start:.2f} seconds.")

        if log_mode == "summary" and (index % progress_every == 0 or index == n_recordings):
            elapsed = perf_counter() - batch_start
            average_seconds = elapsed / index
            remaining_seconds = average_seconds * (n_recordings - index)
            print(
                f"Processed {index}/{n_recordings} ({100 * index / n_recordings:.1f}%) | "
                f"Elapsed: {elapsed / 60:.1f} min | "
                f"Estimated remaining: {remaining_seconds / 60:.1f} min"
            )

    # ------------------------------------------------------------------
    # Final batch summary
    # ------------------------------------------------------------------
    if log_mode != "silent":
        elapsed = perf_counter() - batch_start
        print(
            f"Completed qEEG analysis for {len(results_by_recording)} recordings "
            f"in {elapsed / 60:.1f} minutes."
        )

    return results_by_recording



# =============================================================================
# PHYSIOLOGICAL QC — POSTERIOR ALPHA
# =============================================================================
def build_posterior_alpha_qc(
    qeeg_results_by_recording: Mapping[str, Mapping[str, Any]],
    *,
    posterior_channels: Sequence[str] = ("Pz", "O1", "Oz", "O2"),
    alpha_band: str = "alpha",
    metadata_fields: Sequence[str] = DEFAULT_QEEG_METADATA_FIELDS,
) -> dict[str, pd.DataFrame]:
    """
    Build a compact posterior-alpha physiological QC summary.

    Purpose
    -------
    This is a physiological sanity check, not a new qEEG endpoint.

    It evaluates whether:
      1. Alpha activity shows posterior/occipital predominance.
      2. Posterior alpha behavior can be compared between EO and EC.

    Existing PSD and relative-power results are reused; nothing is recalculated
    from the raw EEG.

    Important
    ---------
    Posterior PSD is averaged across channels in LINEAR power units first.
    The posterior-average PSD is converted to dB only after averaging.

    Returns
    -------
    dict[str, pd.DataFrame]

        posterior_psd_df
            Posterior-average PSD curve for each logical recording.

        summary_df
            Compact recording-level physiological QC summary containing:
              - posterior relative alpha
              - whole-scalp relative alpha
              - posterior/scalp alpha ratio
    """
    if not qeeg_results_by_recording:
        raise ValueError(
            "qeeg_results_by_recording is empty."
        )

    posterior_channels = tuple(
        str(channel)
        for channel in posterior_channels
    )

    if not posterior_channels:
        raise ValueError(
            "posterior_channels must contain at least one channel."
        )

    psd_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    tiny = np.finfo(float).tiny

    for recording_id, result in qeeg_results_by_recording.items():

        # --------------------------------------------------------
        # Recording metadata
        # --------------------------------------------------------
        metadata_values = _extract_qeeg_metadata(
            str(recording_id),
            result,
            metadata_fields,
        )

        # --------------------------------------------------------
        # Existing qEEG outputs
        # --------------------------------------------------------
        psd_result = result.get(
            "mean_psd_result"
        )

        relative_result = result.get(
            "relative_power_result"
        )

        if not isinstance(psd_result, Mapping):
            raise TypeError(
                f"Recording '{recording_id}' is missing "
                "mean_psd_result."
            )

        if not isinstance(relative_result, Mapping):
            raise TypeError(
                f"Recording '{recording_id}' is missing "
                "relative_power_result."
            )

        ch_names = list(
            psd_result["ch_names"]
        )

        # Use requested posterior channels that exist in this recording.
        available_channels = [
            channel
            for channel in posterior_channels
            if channel in ch_names
        ]

        if not available_channels:
            raise ValueError(
                f"Recording '{recording_id}' contains none of the "
                f"requested posterior channels: "
                f"{list(posterior_channels)}"
            )

        channel_indices = [
            ch_names.index(channel)
            for channel in available_channels
        ]

        # ========================================================
        # 1. Posterior-average PSD
        # ========================================================
        freqs = np.asarray(
            psd_result["freqs_hz"],
            dtype=float,
        )

        channel_psd_uv2 = np.asarray(
            psd_result[
                "mean_psd_by_channel_uv2_per_hz"
            ],
            dtype=float,
        )

        # Average LINEAR PSD across posterior channels.
        posterior_psd_uv2 = np.mean(
            channel_psd_uv2[
                channel_indices,
                :
            ],
            axis=0,
        )

        # Convert to dB only after averaging linear power.
        posterior_psd_db = 10.0 * np.log10(
            np.maximum(
                posterior_psd_uv2,
                tiny,
            )
        )

        for frequency, psd_uv2, psd_db in zip(
            freqs,
            posterior_psd_uv2,
            posterior_psd_db,
        ):
            psd_rows.append({
                **metadata_values,
                "posterior_channels":
                    list(available_channels),
                "frequency_hz":
                    float(frequency),
                "mean_psd_uv2_per_hz":
                    float(psd_uv2),
                "mean_psd_db":
                    float(psd_db),
            })

        # ========================================================
        # 2. Posterior relative-alpha summary
        # ========================================================
        relative_channel_df = (
            relative_result[
                "channel_relative_power_df"
            ]
        )

        if alpha_band not in relative_channel_df.columns:
            raise KeyError(
                f"Band '{alpha_band}' is missing from "
                f"relative-power results for '{recording_id}'."
            )

        # Mean alpha across posterior channels.
        posterior_alpha_percent = float(
            relative_channel_df.loc[
                available_channels,
                alpha_band,
            ].mean()
        )

        # Whole-scalp alpha from existing qEEG result.
        overall_relative_df = (
            relative_result[
                "overall_relative_power_df"
            ]
        )

        alpha_rows = overall_relative_df.loc[
            overall_relative_df["band"] == alpha_band,
            "mean_relative_power_percent",
        ]

        if alpha_rows.empty:
            raise KeyError(
                f"Band '{alpha_band}' is missing from the "
                f"whole-scalp relative-power table for "
                f"'{recording_id}'."
            )

        scalp_alpha_percent = float(
            alpha_rows.iloc[0]
        )

        # Posterior predominance ratio:
        # >1 means posterior alpha exceeds the whole-scalp mean.
        posterior_to_scalp_ratio = (
            posterior_alpha_percent
            / scalp_alpha_percent
            if scalp_alpha_percent > 0
            else np.nan
        )

        summary_rows.append({
            **metadata_values,

            # Channels contributing to the QC summary
            "posterior_channels":
                list(available_channels),
            "n_posterior_channels":
                len(available_channels),

            # Main physiological QC values
            "posterior_alpha_percent":
                posterior_alpha_percent,
            "scalp_alpha_percent":
                scalp_alpha_percent,
            "posterior_to_scalp_alpha_ratio":
                posterior_to_scalp_ratio,

            # Simple within-recording posterior predominance check
            "posterior_alpha_predominant":
                bool(
                    np.isfinite(
                        posterior_to_scalp_ratio
                    )
                    and posterior_to_scalp_ratio > 1.0
                ),
        })

    # ============================================================
    # Final compact outputs
    # ============================================================
    posterior_psd_df = pd.DataFrame(
        psd_rows
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    return {
        "posterior_psd_df": posterior_psd_df,
        "summary_df": summary_df,
    }

# =============================================================================
# PHYSIOLOGICAL QC — FRONTAL / FRONTOTEMPORAL HIGH-FREQUENCY POWER
# =============================================================================

def build_frontal_high_frequency_qc(
    qeeg_results_by_recording: Mapping[str, Mapping[str, Any]],
    *,
    left_channels: Sequence[str] = (
        "Fp1",
        "F7",
        "F3",
        "FC5",
        "T7",
    ),
    right_channels: Sequence[str] = (
        "Fp2",
        "F8",
        "F4",
        "FC6",
        "T8",
    ),
    midline_channels: Sequence[str] = (
        "Fz",
    ),
    bands: Sequence[str] = (
        "beta",
        "gamma",
    ),
    metadata_fields: Sequence[str] = DEFAULT_QEEG_METADATA_FIELDS,
) -> dict[str, pd.DataFrame]:
    """
    Build compact frontal/frontotemporal high-frequency QC summaries.

    Purpose
    -------
    This is an artifact-review sanity check, not a new qEEG endpoint.

    It evaluates whether beta/gamma power shows:
      1. Frontal/frontotemporal concentration relative to the rest of the scalp.
      2. Strong left-versus-right frontal/frontotemporal asymmetry.
      3. A localized channel containing the maximum band power.

    Existing channel-level absolute band-power results are reused.
    No PSD or band-power calculations are repeated from the EEG epochs.

    Notes
    -----
    The function deliberately does NOT assign an automatic artifact/pass/fail
    classification. Frontal high-frequency power can have multiple causes, so
    the numerical summary should be reviewed together with the existing
    beta/gamma topographic maps.

    Left-right asymmetry is calculated as:

        100 * (left - right) / ((left + right) / 2)

    Therefore:
        positive value -> left > right
        negative value -> right > left
        value near zero -> relatively symmetric

    Absolute left-right asymmetry is also retained because signed asymmetry
    can cancel across patients during cohort-level aggregation.

    Returns
    -------
    dict[str, pd.DataFrame]

        summary_df
            One compact QC row per logical recording.
    """
    if not qeeg_results_by_recording:
        raise ValueError(
            "qeeg_results_by_recording is empty."
        )

    left_channels = tuple(
        str(channel)
        for channel in left_channels
    )

    right_channels = tuple(
        str(channel)
        for channel in right_channels
    )

    midline_channels = tuple(
        str(channel)
        for channel in midline_channels
    )

    bands = tuple(
        str(band)
        for band in bands
    )

    if not left_channels:
        raise ValueError(
            "left_channels must contain at least one channel."
        )

    if not right_channels:
        raise ValueError(
            "right_channels must contain at least one channel."
        )

    if not bands:
        raise ValueError(
            "bands must contain at least one frequency band."
        )

    # ------------------------------------------------------------
    # Prevent accidental double-counting across anatomical groups
    # ------------------------------------------------------------
    left_set = set(
        left_channels
    )

    right_set = set(
        right_channels
    )

    midline_set = set(
        midline_channels
    )

    overlapping_channels = (
        (left_set & right_set)
        | (left_set & midline_set)
        | (right_set & midline_set)
    )

    if overlapping_channels:
        raise ValueError(
            "Channel groups overlap: "
            f"{sorted(overlapping_channels)}"
        )

    summary_rows: list[dict[str, Any]] = []

    # ============================================================
    # Process each logical recording independently
    # ============================================================
    for recording_id, result in qeeg_results_by_recording.items():

        # --------------------------------------------------------
        # Recording metadata
        # --------------------------------------------------------
        metadata_values = _extract_qeeg_metadata(
            str(recording_id),
            result,
            metadata_fields,
        )

        # --------------------------------------------------------
        # Existing absolute band-power output
        # --------------------------------------------------------
        absolute_result = result.get(
            "absolute_power_result"
        )

        if not isinstance(absolute_result, Mapping):
            raise TypeError(
                f"Recording '{recording_id}' is missing "
                "absolute_power_result."
            )

        channel_power_df = absolute_result.get(
            "channel_band_power_df"
        )

        if not isinstance(channel_power_df, pd.DataFrame):
            raise TypeError(
                f"Recording '{recording_id}' is missing a valid "
                "channel_band_power_df."
            )

        missing_bands = [
            band
            for band in bands
            if band not in channel_power_df.columns
        ]

        if missing_bands:
            raise KeyError(
                f"Recording '{recording_id}' is missing bands: "
                f"{missing_bands}"
            )

        available_channels = [
            str(channel)
            for channel in channel_power_df.index
        ]

        available_channel_set = set(
            available_channels
        )

        # --------------------------------------------------------
        # Resolve anatomical channel groups
        # --------------------------------------------------------
        available_left = [
            channel
            for channel in left_channels
            if channel in available_channel_set
        ]

        available_right = [
            channel
            for channel in right_channels
            if channel in available_channel_set
        ]

        available_midline = [
            channel
            for channel in midline_channels
            if channel in available_channel_set
        ]

        if not available_left:
            raise ValueError(
                f"Recording '{recording_id}' contains none of the "
                "requested left frontal/frontotemporal channels."
            )

        if not available_right:
            raise ValueError(
                f"Recording '{recording_id}' contains none of the "
                "requested right frontal/frontotemporal channels."
            )

        # Combined frontal/frontotemporal region used for comparison
        # against the remaining scalp channels.
        available_frontotemporal = list(
            dict.fromkeys(
                available_left
                + available_right
                + available_midline
            )
        )

        frontotemporal_set = set(
            available_frontotemporal
        )

        # Reference region = every analyzed EEG channel outside the
        # frontal/frontotemporal set.
        rest_channels = [
            channel
            for channel in available_channels
            if channel not in frontotemporal_set
        ]

        if not rest_channels:
            raise ValueError(
                f"Recording '{recording_id}' has no remaining scalp "
                "channels available for the reference region."
            )

        missing_requested_channels = [
            channel
            for channel in (
                left_channels
                + right_channels
                + midline_channels
            )
            if channel not in available_channel_set
        ]

        # --------------------------------------------------------
        # Base recording-level QC row
        # --------------------------------------------------------
        row: dict[str, Any] = {
            **metadata_values,

            # Channel availability / traceability
            "left_channels":
                list(available_left),

            "right_channels":
                list(available_right),

            "midline_channels":
                list(available_midline),

            "frontotemporal_channels":
                list(available_frontotemporal),

            "rest_channels":
                list(rest_channels),

            "n_left_channels":
                len(available_left),

            "n_right_channels":
                len(available_right),

            "n_frontotemporal_channels":
                len(available_frontotemporal),

            "n_rest_channels":
                len(rest_channels),

            "missing_requested_channels":
                list(missing_requested_channels),
        }

        # ========================================================
        # Calculate beta / gamma QC metrics
        # ========================================================
        for band in bands:

            band_values = pd.to_numeric(
                channel_power_df[band],
                errors="coerce",
            )

            # ----------------------------------------------------
            # Regional channel values
            # ----------------------------------------------------
            left_values = band_values.reindex(
                available_left
            ).dropna()

            right_values = band_values.reindex(
                available_right
            ).dropna()

            frontotemporal_values = band_values.reindex(
                available_frontotemporal
            ).dropna()

            rest_values = band_values.reindex(
                rest_channels
            ).dropna()

            # ----------------------------------------------------
            # Regional means
            # ----------------------------------------------------
            left_mean = (
                float(left_values.mean())
                if not left_values.empty
                else np.nan
            )

            right_mean = (
                float(right_values.mean())
                if not right_values.empty
                else np.nan
            )

            frontotemporal_mean = (
                float(frontotemporal_values.mean())
                if not frontotemporal_values.empty
                else np.nan
            )

            rest_mean = (
                float(rest_values.mean())
                if not rest_values.empty
                else np.nan
            )

            # ----------------------------------------------------
            # Frontal/frontotemporal concentration
            # ----------------------------------------------------
            # >1 means mean frontal/frontotemporal band power exceeds
            # mean band power across the remaining scalp.
            frontotemporal_to_rest_ratio = (
                frontotemporal_mean / rest_mean
                if (
                    np.isfinite(frontotemporal_mean)
                    and np.isfinite(rest_mean)
                    and rest_mean > 0
                )
                else np.nan
            )

            # ----------------------------------------------------
            # Signed left-right asymmetry
            # ----------------------------------------------------
            # Positive = left > right
            # Negative = right > left
            lr_mean = (
                (left_mean + right_mean) / 2.0
                if (
                    np.isfinite(left_mean)
                    and np.isfinite(right_mean)
                )
                else np.nan
            )

            lr_asymmetry_percent = (
                100.0
                * (left_mean - right_mean)
                / lr_mean
                if (
                    np.isfinite(lr_mean)
                    and lr_mean > 0
                )
                else np.nan
            )

            # ----------------------------------------------------
            # Absolute left-right asymmetry
            # ----------------------------------------------------
            # Retains asymmetry magnitude regardless of direction.
            # This is useful for cohort aggregation because strong
            # left and strong right asymmetries should not cancel.
            abs_lr_asymmetry_percent = (
                abs(lr_asymmetry_percent)
                if np.isfinite(lr_asymmetry_percent)
                else np.nan
            )

            # ----------------------------------------------------
            # Maximum-power channel
            # ----------------------------------------------------
            valid_band_values = (
                band_values
                .dropna()
            )

            if valid_band_values.empty:
                peak_channel = None
                peak_power = np.nan
                peak_is_frontotemporal = False

            else:
                peak_channel = str(
                    valid_band_values.idxmax()
                )

                peak_power = float(
                    valid_band_values.loc[
                        peak_channel
                    ]
                )

                peak_is_frontotemporal = (
                    peak_channel
                    in frontotemporal_set
                )

            # ----------------------------------------------------
            # Store band-specific QC metrics
            # ----------------------------------------------------
            row.update({
                f"{band}_frontotemporal_mean_uv2":
                    frontotemporal_mean,

                f"{band}_rest_mean_uv2":
                    rest_mean,

                f"{band}_frontotemporal_to_rest_ratio":
                    frontotemporal_to_rest_ratio,

                f"{band}_left_mean_uv2":
                    left_mean,

                f"{band}_right_mean_uv2":
                    right_mean,

                f"{band}_lr_asymmetry_percent":
                    lr_asymmetry_percent,

                f"{band}_abs_lr_asymmetry_percent":
                    abs_lr_asymmetry_percent,

                f"{band}_peak_channel":
                    peak_channel,

                f"{band}_peak_power_uv2":
                    peak_power,

                f"{band}_peak_is_frontotemporal":
                    bool(peak_is_frontotemporal),
            })

        summary_rows.append(
            row
        )

    # ============================================================
    # Final recording-level QC output
    # ============================================================
    summary_df = pd.DataFrame(
        summary_rows
    )

    return {
        "summary_df": summary_df,
    }



# =============================================================================
# FRONTAL / FRONTOTEMPORAL HIGH-FREQUENCY QC — SUMMARY TABLES
# =============================================================================

# def summarize_frontal_high_frequency_qc(
#     frontal_high_frequency_qc: Mapping[str, Any],
#     *,
#     group_columns: Sequence[str] | None = None,
# ) -> dict[str, pd.DataFrame]:
#     """
#     Create patient-level and aggregate frontal high-frequency QC tables.

#     Parameters
#     ----------
#     frontal_high_frequency_qc
#         Output returned by build_frontal_high_frequency_qc().

#     group_columns
#         Columns used to create the aggregate summary.

#         If None, the function automatically uses available meaningful
#         study fields from:
#             cohort
#             timepoint
#             eye_state

#         Columns that are completely missing are ignored.

#     Returns
#     -------
#     dict[str, pd.DataFrame]

#         all_patients_df
#             One compact QC row per logical recording.

#         aggregate_df
#             Group-level QC summary across recordings/patients.
#     """
#     if not isinstance(frontal_high_frequency_qc, Mapping):
#         raise TypeError(
#             "frontal_high_frequency_qc must be a mapping."
#         )

#     source_df = frontal_high_frequency_qc.get(
#         "summary_df"
#     )

#     if not isinstance(source_df, pd.DataFrame):
#         raise TypeError(
#             "frontal_high_frequency_qc['summary_df'] "
#             "must be a pandas DataFrame."
#         )

#     if source_df.empty:
#         raise ValueError(
#             "frontal_high_frequency_qc['summary_df'] is empty."
#         )

#     # ------------------------------------------------------------
#     # Validate required QC metrics
#     # ------------------------------------------------------------
#     required_columns = {
#         "recording_id",
#         "eye_state",

#         "beta_frontotemporal_to_rest_ratio",
#         "gamma_frontotemporal_to_rest_ratio",

#         "beta_lr_asymmetry_percent",
#         "gamma_lr_asymmetry_percent",

#         "beta_abs_lr_asymmetry_percent",
#         "gamma_abs_lr_asymmetry_percent",

#         "beta_peak_channel",
#         "gamma_peak_channel",
#     }

#     missing_columns = (
#         required_columns
#         - set(source_df.columns)
#     )

#     if missing_columns:
#         raise KeyError(
#             "Frontal high-frequency QC summary is missing "
#             f"required columns: {sorted(missing_columns)}"
#         )

#     # ============================================================
#     # 1. ALL-PATIENT / ALL-RECORDING QC TABLE
#     # ============================================================
#     metadata_columns = [
#         column
#         for column in (
#             "recording_id",
#             "subject_id",
#             "cohort",
#             "visit",
#             "timepoint",
#             "dose",
#             "eye_state",
#         )
#         if column in source_df.columns
#     ]

#     qc_columns = [
#         "beta_frontotemporal_to_rest_ratio",
#         "gamma_frontotemporal_to_rest_ratio",

#         "beta_lr_asymmetry_percent",
#         "gamma_lr_asymmetry_percent",

#         "beta_abs_lr_asymmetry_percent",
#         "gamma_abs_lr_asymmetry_percent",

#         "beta_peak_channel",
#         "gamma_peak_channel",
#     ]

#     all_patients_df = (
#         source_df[
#             metadata_columns
#             + qc_columns
#         ]
#         .copy()
#     )

#     # Spell out "left-right" rather than using the abbreviated "lr".
#     all_patients_df = all_patients_df.rename(
#         columns={
#             "beta_lr_asymmetry_percent":
#                 "beta_left_right_asymmetry_percent",

#             "gamma_lr_asymmetry_percent":
#                 "gamma_left_right_asymmetry_percent",

#             "beta_abs_lr_asymmetry_percent":
#                 "beta_absolute_left_right_asymmetry_percent",

#             "gamma_abs_lr_asymmetry_percent":
#                 "gamma_absolute_left_right_asymmetry_percent",
#         }
#     )

#     # ============================================================
#     # 2. DETERMINE AGGREGATION COLUMNS
#     # ============================================================
#     if group_columns is None:

#         group_columns_used = []

#         for column in (
#             "cohort",
#             "timepoint",
#             "eye_state",
#         ):
#             if column not in all_patients_df.columns:
#                 continue

#             # Eye state is always meaningful for this QC.
#             if column == "eye_state":
#                 group_columns_used.append(
#                     column
#                 )

#             # Ignore metadata columns that are completely empty.
#             elif all_patients_df[column].notna().any():
#                 group_columns_used.append(
#                     column
#                 )

#     else:
#         group_columns_used = [
#             str(column)
#             for column in group_columns
#         ]

#         missing_group_columns = [
#             column
#             for column in group_columns_used
#             if column not in all_patients_df.columns
#         ]

#         if missing_group_columns:
#             raise KeyError(
#                 "Requested group columns are missing: "
#                 f"{missing_group_columns}"
#             )

#     if not group_columns_used:
#         raise ValueError(
#             "No usable aggregation columns were identified."
#         )

#     # ============================================================
#     # 3. AGGREGATE ACROSS PATIENTS / RECORDINGS
#     # ============================================================
#     numeric_metrics = [
#         "beta_frontotemporal_to_rest_ratio",
#         "gamma_frontotemporal_to_rest_ratio",

#         # Signed values preserve direction:
#         # positive = left > right
#         # negative = right > left
#         "beta_left_right_asymmetry_percent",
#         "gamma_left_right_asymmetry_percent",

#         # Absolute values preserve asymmetry magnitude and avoid
#         # left/right cancellation when averaging across patients.
#         "beta_absolute_left_right_asymmetry_percent",
#         "gamma_absolute_left_right_asymmetry_percent",
#     ]

#     aggregate_rows: list[dict[str, Any]] = []

#     grouped = all_patients_df.groupby(
#         group_columns_used,
#         observed=True,
#         dropna=False,
#         sort=False,
#     )

#     for group_key, group_df in grouped:

#         if not isinstance(group_key, tuple):
#             group_key = (
#                 group_key,
#             )

#         aggregate_row = {
#             column: value
#             for column, value in zip(
#                 group_columns_used,
#                 group_key,
#             )
#         }

#         # --------------------------------------------------------
#         # Number of contributing recordings and subjects
#         # --------------------------------------------------------
#         aggregate_row["n_recordings"] = int(
#             len(group_df)
#         )

#         if "subject_id" in group_df.columns:
#             aggregate_row["n_subjects"] = int(
#                 group_df["subject_id"]
#                 .dropna()
#                 .nunique()
#             )
#         else:
#             aggregate_row["n_subjects"] = int(
#                 len(group_df)
#             )

#         # --------------------------------------------------------
#         # Numeric group summaries
#         # --------------------------------------------------------
#         for metric in numeric_metrics:

#             values = pd.to_numeric(
#                 group_df[metric],
#                 errors="coerce",
#             ).dropna()

#             aggregate_row[
#                 f"{metric}_mean"
#             ] = (
#                 float(values.mean())
#                 if not values.empty
#                 else np.nan
#             )

#             # Sample SD is undefined for N=1, so leave as NaN.
#             aggregate_row[
#                 f"{metric}_sd"
#             ] = (
#                 float(values.std(ddof=1))
#                 if len(values) > 1
#                 else np.nan
#             )

#         # --------------------------------------------------------
#         # Most common beta / gamma peak channels
#         # --------------------------------------------------------
#         for band in (
#             "beta",
#             "gamma",
#         ):
#             peak_column = (
#                 f"{band}_peak_channel"
#             )

#             peak_values = (
#                 group_df[peak_column]
#                 .dropna()
#                 .astype(str)
#             )

#             if peak_values.empty:
#                 peak_channel = None
#                 peak_count = 0
#                 peak_percent = np.nan

#             else:
#                 peak_counts = (
#                     peak_values
#                     .value_counts()
#                 )

#                 peak_channel = str(
#                     peak_counts.index[0]
#                 )

#                 peak_count = int(
#                     peak_counts.iloc[0]
#                 )

#                 peak_percent = float(
#                     100.0
#                     * peak_count
#                     / len(peak_values)
#                 )

#             aggregate_row[
#                 f"{band}_most_common_peak_channel"
#             ] = peak_channel

#             aggregate_row[
#                 f"{band}_most_common_peak_channel_n"
#             ] = peak_count

#             aggregate_row[
#                 f"{band}_most_common_peak_channel_percent"
#             ] = peak_percent

#         aggregate_rows.append(
#             aggregate_row
#         )

#     aggregate_df = pd.DataFrame(
#         aggregate_rows
#     )

#     return {
#         "all_patients_df":
#             all_patients_df,

#         "aggregate_df":
#             aggregate_df,
#     }

def summarize_frontal_high_frequency_qc(
    frontal_high_frequency_qc: Mapping[str, Any],
    *,
    group_columns: Sequence[str] | str | None = None,
    bands: Sequence[str] = ("beta", "gamma"),
) -> dict[str, pd.DataFrame]:
    """
    Create recording-level and aggregate frontal high-frequency QC tables.

    No eye-state field is required. Grouping can be eye state, timepoint,
    phenotype label, another annotation-derived condition, or one overall group.
    """
    if not isinstance(frontal_high_frequency_qc, Mapping):
        raise TypeError("frontal_high_frequency_qc must be a mapping.")

    source_df = frontal_high_frequency_qc.get("summary_df")
    if not isinstance(source_df, pd.DataFrame) or source_df.empty:
        raise ValueError("frontal_high_frequency_qc['summary_df'] must be a non-empty DataFrame.")

    bands = tuple(str(band) for band in bands)
    if not bands:
        raise ValueError("bands must contain at least one band.")

    required_columns = {"recording_id"}
    for band in bands:
        required_columns.update({
            f"{band}_frontotemporal_to_rest_ratio",
            f"{band}_lr_asymmetry_percent",
            f"{band}_abs_lr_asymmetry_percent",
            f"{band}_peak_channel",
        })

    missing = required_columns - set(source_df.columns)
    if missing:
        raise KeyError(f"Frontal high-frequency QC summary is missing required columns: {sorted(missing)}")

    metadata_columns = [
        column for column in (
            "recording_id", "source_recording_id", "subject_id", "label",
            "condition", "analysis_condition", "eye_state",
            "cohort", "visit", "timepoint", "dose",
        )
        if column in source_df.columns
    ]

    qc_columns = []
    rename_columns = {}
    for band in bands:
        qc_columns.extend([
            f"{band}_frontotemporal_to_rest_ratio",
            f"{band}_lr_asymmetry_percent",
            f"{band}_abs_lr_asymmetry_percent",
            f"{band}_peak_channel",
        ])
        rename_columns.update({
            f"{band}_lr_asymmetry_percent": f"{band}_left_right_asymmetry_percent",
            f"{band}_abs_lr_asymmetry_percent": f"{band}_absolute_left_right_asymmetry_percent",
        })

    all_patients_df = source_df[metadata_columns + qc_columns].copy().rename(columns=rename_columns)

    # ------------------------------------------------------------------
    # Resolve grouping without requiring eye_state
    # ------------------------------------------------------------------
    if group_columns is not None:
        group_columns_used = list(_normalize_group_columns(group_columns))
    else:
        group_columns_used = [
            column for column in ("cohort", "timepoint", "eye_state")
            if column in all_patients_df.columns and all_patients_df[column].notna().any()
        ]
        if not group_columns_used:
            for column in ("analysis_condition", "condition", "label"):
                if column in all_patients_df.columns and all_patients_df[column].notna().any():
                    group_columns_used = [column]
                    break

    missing_groups = [column for column in group_columns_used if column not in all_patients_df.columns]
    if missing_groups:
        raise KeyError(f"Requested group columns are missing: {missing_groups}")

    if not group_columns_used:
        all_patients_df["_report_group"] = "Overall"
        group_columns_used = ["_report_group"]

    # ------------------------------------------------------------------
    # Aggregate numerical QC and peak-channel recurrence
    # ------------------------------------------------------------------
    aggregate_rows = []

    for group_key, group_df in all_patients_df.groupby(group_columns_used, observed=True, dropna=False, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = dict(zip(group_columns_used, group_key))
        row["n_recordings"] = int(len(group_df))
        row["n_subjects"] = (
            int(group_df["subject_id"].dropna().nunique())
            if "subject_id" in group_df.columns else int(len(group_df))
        )

        for band in bands:
            numeric_metrics = (
                f"{band}_frontotemporal_to_rest_ratio",
                f"{band}_left_right_asymmetry_percent",
                f"{band}_absolute_left_right_asymmetry_percent",
            )
            for metric in numeric_metrics:
                values = pd.to_numeric(group_df[metric], errors="coerce").dropna()
                row[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
                row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan

            peak_values = group_df[f"{band}_peak_channel"].dropna().astype(str)
            peak_counts = peak_values.value_counts()
            row[f"{band}_most_common_peak_channel"] = str(peak_counts.index[0]) if not peak_counts.empty else None
            row[f"{band}_most_common_peak_channel_n"] = int(peak_counts.iloc[0]) if not peak_counts.empty else 0
            row[f"{band}_most_common_peak_channel_percent"] = (
                float(100.0 * peak_counts.iloc[0] / len(peak_values)) if not peak_counts.empty else np.nan
            )

        aggregate_rows.append(row)

    return {"all_patients_df": all_patients_df, "aggregate_df": pd.DataFrame(aggregate_rows)}



# =============================================================================
# FRONTAL / FRONTOTEMPORAL HIGH-FREQUENCY QC — SLIDE SUMMARY
# =============================================================================

def build_frontal_high_frequency_slide_summary(
    frontal_hf_tables: Mapping[str, Any],
    *,
    group_columns: Sequence[str] | str | None = None,
    group_order: Mapping[str, Sequence[Any]] | None = None,
    group_labels: Mapping[str, str] | None = None,
    group_filters: Mapping[str, Any] | None = None,
    bands: Sequence[str] = ("beta", "gamma"),
    ratio_decimals: int = 2,
    asymmetry_decimals: int = 1,
    peak_decimals: int = 0,
) -> pd.DataFrame:
    """
    Build a compact high-frequency QC table for arbitrary grouping dimensions.

    Examples:
        NeuShen -> group_columns=("timepoint", "eye_state")
        ABC-CT  -> group_columns=("label",) if high-frequency QC is enabled
        Overall -> group_columns=None
    """
    if not isinstance(frontal_hf_tables, Mapping):
        raise TypeError("frontal_hf_tables must be a mapping.")

    aggregate_df = frontal_hf_tables.get("aggregate_df")
    if not isinstance(aggregate_df, pd.DataFrame) or aggregate_df.empty:
        raise ValueError("frontal_hf_tables['aggregate_df'] must be a non-empty DataFrame.")

    data = aggregate_df.copy()
    bands = tuple(str(band) for band in bands)

    for column, value in dict(group_filters or {}).items():
        if column not in data.columns:
            raise KeyError(f"Filter column '{column}' is not present in aggregate_df.")
        data = data.loc[data[column].astype(str) == str(value)].copy()

    if data.empty:
        raise ValueError("No aggregate high-frequency QC rows remain after filtering.")

    if group_columns is not None:
        group_columns_used = list(_normalize_group_columns(group_columns))
    else:
        group_columns_used = [
            column for column in ("timepoint", "eye_state")
            if column in data.columns and data[column].notna().any()
        ]
        if not group_columns_used:
            for column in ("analysis_condition", "condition", "label"):
                if column in data.columns and data[column].notna().any():
                    group_columns_used = [column]
                    break

    missing_groups = [column for column in group_columns_used if column not in data.columns]
    if missing_groups:
        raise KeyError(f"High-frequency aggregate table is missing grouping columns: {missing_groups}")

    if not group_columns_used:
        data["_report_group"] = "Overall"
        group_columns_used = ["_report_group"]

    order_map = {str(column): list(values) for column, values in dict(group_order or {}).items()}
    label_map = {str(column): str(label) for column, label in dict(group_labels or {}).items()}

    for column in group_columns_used:
        observed = data[column].dropna().drop_duplicates().tolist()
        preferred = order_map.get(column, [])
        order_used = preferred + [value for value in observed if value not in preferred]
        if order_used:
            data[column] = pd.Categorical(data[column], categories=order_used, ordered=True)

    data = data.sort_values(group_columns_used, kind="stable").reset_index(drop=True)

    def format_mean_sd(mean_value, sd_value, *, decimals, suffix=""):
        mean_value = pd.to_numeric(pd.Series([mean_value]), errors="coerce").iloc[0]
        sd_value = pd.to_numeric(pd.Series([sd_value]), errors="coerce").iloc[0]
        if not np.isfinite(mean_value):
            return "N/A"
        return (
            f"{mean_value:.{decimals}f} ± {sd_value:.{decimals}f}{suffix}"
            if np.isfinite(sd_value) else f"{mean_value:.{decimals}f}{suffix}"
        )

    def format_peak_channel(channel, percent):
        if channel is None or pd.isna(channel):
            return "N/A"
        percent = pd.to_numeric(pd.Series([percent]), errors="coerce").iloc[0]
        return f"{channel} ({percent:.{peak_decimals}f}%)" if np.isfinite(percent) else str(channel)

    rows = []

    for _, row in data.iterrows():
        for band in bands:
            ratio_mean = row.get(f"{band}_frontotemporal_to_rest_ratio_mean")
            ratio_sd = row.get(f"{band}_frontotemporal_to_rest_ratio_sd")
            asymmetry_mean = row.get(f"{band}_absolute_left_right_asymmetry_percent_mean")
            asymmetry_sd = row.get(f"{band}_absolute_left_right_asymmetry_percent_sd")
            peak_channel = row.get(f"{band}_most_common_peak_channel")
            peak_percent = row.get(f"{band}_most_common_peak_channel_percent")
            n_value = row.get("n_subjects", row.get("n_recordings"))

            report_row = {}
            for column in group_columns_used:
                if column == "_report_group":
                    report_row["Group"] = str(row[column])
                else:
                    report_row[label_map.get(column, column.replace("_", " ").title())] = str(row[column])

            report_row.update({
                "Band": band.capitalize(),
                "N": int(n_value) if pd.notna(n_value) else None,
                "Frontal/Rest Power Ratio": format_mean_sd(ratio_mean, ratio_sd, decimals=ratio_decimals, suffix="×"),
                "Absolute Left-Right Asymmetry": format_mean_sd(
                    asymmetry_mean, asymmetry_sd, decimals=asymmetry_decimals, suffix="%"
                ),
                "Most Common Peak Channel": format_peak_channel(peak_channel, peak_percent),
            })
            rows.append(report_row)

    return pd.DataFrame(rows)



# =============================================================================
# CORE SPECTRAL CALCULATIONS
# =============================================================================

# Calculate epoch-, channel-, and scalp-level Welch PSD summaries.
def calculate_mean_psd_curve(
    epochs: mne.BaseEpochs,
    *,
    fmin: float = 0.5,
    fmax: float = 45.0,
    picks: str | Sequence[str] = "eeg",
    n_fft: int | None = None,
    n_per_seg: int | None = None,
    n_overlap: int = 0,
    window: str = "hamming",
    remove_dc: bool = True,
    plot: bool = True,
    show_channel_curves: bool = False,
    title: str = "Mean Power Spectral Density",
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Calculate the mean power spectral density curve from cleaned EEG epochs.

    The function performs the following steps:

      1. Calculates a Welch PSD separately for every retained epoch and channel.
      2. Averages the PSD across epochs while preserving individual channels.
      3. Calculates an overall scalp mean by averaging across EEG channels.
      4. Retains linear PSD values for later band-power calculations.
      5. Converts PSD to decibels for plotting.

    Parameters
    ----------
    epochs
        Clean MNE Epochs object, such as state["epochs_clean"].

    fmin, fmax
        Frequency range to retain, in Hz.

    picks
        EEG channels to include. Default is "eeg".

    n_fft
        FFT length. If None, the number of samples in one epoch is used.

        For a 2-second epoch sampled at 250 Hz, this will normally be
        500 samples and produce 0.5-Hz frequency resolution.

    n_per_seg
        Number of samples in each Welch segment. If None, one complete
        epoch is used as the segment.

    n_overlap
        Number of overlapping samples between Welch segments.
        Default is zero because each clean epoch is already an
        independent 2-second segment.

    window
        Window applied before the Fourier transform.
        Default is "hamming".

    remove_dc
        Whether to remove the mean from each segment before calculating
        the spectrum.

    plot
        If True, display the overall mean PSD curve.

    show_channel_curves
        If True, display the mean curve for every channel behind the
        overall scalp mean.

    title
        Plot title.

    verbose
        Passed to MNE.

    Returns
    -------
    dict
        Dictionary containing:

        - spectrum:
            Original MNE EpochsSpectrum object.

        - freqs_hz:
            Frequency-bin values.

        - ch_names:
            EEG channel names.

        - psd_by_epoch_v2_per_hz:
            PSD for every epoch and channel.
            Shape: (epochs, channels, frequencies).

        - mean_psd_by_channel_v2_per_hz:
            Mean across epochs, preserving channels.
            Shape: (channels, frequencies).

        - mean_psd_by_channel_uv2_per_hz:
            Same channel-level values converted to microvolt squared / Hz.

        - mean_psd_by_channel_db:
            Channel-level PSD in dB relative to 1 microvolt squared / Hz.

        - overall_mean_psd_uv2_per_hz:
            Mean across both epochs and EEG channels.

        - overall_mean_psd_db:
            Overall scalp mean in dB relative to
            1 microvolt squared / Hz.

        - channel_psd_df:
            Frequency-by-channel DataFrame in microvolt squared / Hz.

        - overall_psd_df:
            DataFrame containing the overall mean curve.

        - settings:
            Spectral settings used for this calculation.

        - figure:
            Matplotlib figure, or None when plot=False.
    """
    # ------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------
    if not isinstance(epochs, mne.BaseEpochs):
        raise TypeError(
            "epochs must be an MNE Epochs object. "
            f"Received {type(epochs).__name__}."
        )

    if len(epochs) == 0:
        raise ValueError("epochs contains no retained epochs.")

    if fmin < 0:
        raise ValueError("fmin must be greater than or equal to zero.")

    if fmax <= fmin:
        raise ValueError("fmax must be greater than fmin.")

    sfreq = float(epochs.info["sfreq"])
    nyquist = sfreq / 2.0

    if fmax >= nyquist:
        raise ValueError(
            f"fmax must be below Nyquist ({nyquist:.2f} Hz). "
            f"Received fmax={fmax:.2f} Hz."
        )

    n_times = len(epochs.times)

    if n_fft is None:
        n_fft = n_times
    else:
        n_fft = int(n_fft)

    if n_per_seg is None:
        n_per_seg = min(n_fft, n_times)
    else:
        n_per_seg = int(n_per_seg)

    n_overlap = int(n_overlap)

    if n_fft <= 0:
        raise ValueError("n_fft must be greater than zero.")

    if n_per_seg <= 0:
        raise ValueError("n_per_seg must be greater than zero.")

    if n_per_seg > n_times:
        raise ValueError(
            f"n_per_seg={n_per_seg} exceeds the number of samples "
            f"in one epoch ({n_times})."
        )

    if n_fft < n_per_seg:
        raise ValueError(
            f"n_fft={n_fft} must be greater than or equal to "
            f"n_per_seg={n_per_seg}."
        )

    if n_overlap < 0 or n_overlap >= n_per_seg:
        raise ValueError(
            "n_overlap must be greater than or equal to zero and "
            "smaller than n_per_seg."
        )

    # ------------------------------------------------------------
    # Calculate PSD for every retained epoch and channel
    # ------------------------------------------------------------
    spectrum = epochs.compute_psd(
        method="welch",
        fmin=float(fmin),
        fmax=float(fmax),
        picks=picks,
        exclude="bads",
        remove_dc=bool(remove_dc),
        n_jobs=1,
        verbose=verbose,
        n_fft=n_fft,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        average="mean",
        window=window,
    )

    psd_by_epoch = spectrum.get_data()
    freqs = np.asarray(spectrum.freqs, dtype=float)
    ch_names = list(spectrum.ch_names)

    # Expected shape:
    # (number of epochs, number of channels, number of frequencies)
    if psd_by_epoch.ndim != 3:
        raise RuntimeError(
            "Expected PSD data with shape "
            "(epochs, channels, frequencies), "
            f"but received shape {psd_by_epoch.shape}."
        )

    # ------------------------------------------------------------
    # Average in linear power units
    # ------------------------------------------------------------

    # Mean across retained epochs, while preserving each EEG channel.
    mean_psd_by_channel_v2 = np.mean(
        psd_by_epoch,
        axis=0,
    )

    # Overall scalp mean: mean across EEG channels.
    overall_mean_psd_v2 = np.mean(
        mean_psd_by_channel_v2,
        axis=0,
    )

    # MNE EEG PSD values are represented in V²/Hz.
    # Convert to µV²/Hz for readable EEG-scale values.
    volts_squared_to_microvolts_squared = 1e12

    psd_by_epoch_uv2 = (
        psd_by_epoch * volts_squared_to_microvolts_squared
    )

    mean_psd_by_channel_uv2 = (
        mean_psd_by_channel_v2
        * volts_squared_to_microvolts_squared
    )

    overall_mean_psd_uv2 = (
        overall_mean_psd_v2
        * volts_squared_to_microvolts_squared
    )

    # ------------------------------------------------------------
    # Convert to dB only after averaging linear power
    # ------------------------------------------------------------
    tiny = np.finfo(float).tiny

    mean_psd_by_channel_db = 10.0 * np.log10(
        np.maximum(mean_psd_by_channel_uv2, tiny)
    )

    overall_mean_psd_db = 10.0 * np.log10(
        np.maximum(overall_mean_psd_uv2, tiny)
    )

    # ------------------------------------------------------------
    # Create convenient DataFrames
    # ------------------------------------------------------------
    channel_psd_df = pd.DataFrame(
        mean_psd_by_channel_uv2.T,
        columns=ch_names,
    )

    channel_psd_df.insert(
        0,
        "frequency_hz",
        freqs,
    )

    overall_psd_df = pd.DataFrame({
        "frequency_hz": freqs,
        "mean_psd_uv2_per_hz": overall_mean_psd_uv2,
        "mean_psd_db": overall_mean_psd_db,
    })

    # ------------------------------------------------------------
    # Optional plot
    # ------------------------------------------------------------
    figure = None

    if plot:
        figure, ax = plt.subplots(figsize=(10, 5))

        if show_channel_curves:
            for channel_curve in mean_psd_by_channel_db:
                ax.plot(
                    freqs,
                    channel_curve,
                    linewidth=0.7,
                    alpha=0.20,
                )

        ax.plot(
            freqs,
            overall_mean_psd_db,
            linewidth=2.0,
            label="Overall scalp mean",
        )

        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (dB re 1 µV²/Hz)")
        ax.set_xlim(float(fmin), float(fmax))
        ax.grid(True, alpha=0.25)
        ax.legend()
        figure.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # Return all useful outputs
    # ------------------------------------------------------------
    return {
        "spectrum": spectrum,
        "freqs_hz": freqs,
        "ch_names": ch_names,

        "psd_by_epoch_v2_per_hz": psd_by_epoch,
        "psd_by_epoch_uv2_per_hz": psd_by_epoch_uv2,

        "mean_psd_by_channel_v2_per_hz":
            mean_psd_by_channel_v2,

        "mean_psd_by_channel_uv2_per_hz":
            mean_psd_by_channel_uv2,

        "mean_psd_by_channel_db":
            mean_psd_by_channel_db,

        "overall_mean_psd_v2_per_hz":
            overall_mean_psd_v2,

        "overall_mean_psd_uv2_per_hz":
            overall_mean_psd_uv2,

        "overall_mean_psd_db":
            overall_mean_psd_db,

        "channel_psd_df": channel_psd_df,
        "overall_psd_df": overall_psd_df,

        "settings": {
            "method": "welch",
            "fmin_hz": float(fmin),
            "fmax_hz": float(fmax),
            "sfreq_hz": sfreq,
            "n_epochs": int(len(epochs)),
            "n_channels": int(len(ch_names)),
            "n_times_per_epoch": int(n_times),
            "n_fft": int(n_fft),
            "n_per_seg": int(n_per_seg),
            "n_overlap": int(n_overlap),
            "window": str(window),
            "remove_dc": bool(remove_dc),
            "frequency_resolution_hz": float(sfreq / n_fft),
        },

        "figure": figure,
    }


# -----------------------------------------------------------------------------
# Integrate linear PSD values to obtain absolute power for each band.
def calculate_absolute_band_power(
    psd_result: Mapping[str, Any],
    *,
    bands: Mapping[str, tuple[float, float]] | None = None,
    plot: bool = True,
    title: str = "Absolute Power by Frequency Band",
) -> dict[str, Any]:
    """
    Calculate absolute EEG power within predefined frequency bands.

    This function integrates the linear PSD values over frequency.
    Because the PSD is expressed in µV²/Hz, integrating over Hz
    produces absolute power in µV².

    Parameters
    ----------
    psd_result
        Output returned by calculate_mean_psd_curve().

    bands
        Mapping from band name to (lower_frequency, upper_frequency).

        Default bands:
            delta: 1-4 Hz
            theta: 4-8 Hz
            alpha: 8-13 Hz
            beta: 13-30 Hz
            gamma: 30-45 Hz

    plot
        If True, plot the overall scalp-mean absolute band powers.

    title
        Title used for the optional plot.

    Returns
    -------
    dict
        Contains epoch-level, channel-level, and overall absolute
        band-power results.
    """
    if bands is None:
        bands = {
            "delta": (1.0, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 45.0),
        }

    required_keys = {
        "freqs_hz",
        "ch_names",
        "psd_by_epoch_uv2_per_hz",
    }

    missing_keys = required_keys - set(psd_result.keys())

    if missing_keys:
        raise KeyError(
            "psd_result is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    freqs = np.asarray(
        psd_result["freqs_hz"],
        dtype=float,
    )

    psd_by_epoch = np.asarray(
        psd_result["psd_by_epoch_uv2_per_hz"],
        dtype=float,
    )

    ch_names = list(psd_result["ch_names"])

    if psd_by_epoch.ndim != 3:
        raise ValueError(
            "Expected PSD shape "
            "(epochs, channels, frequencies), "
            f"but received {psd_by_epoch.shape}."
        )

    if psd_by_epoch.shape[-1] != len(freqs):
        raise ValueError(
            "PSD frequency dimension does not match freqs_hz."
        )

    if psd_by_epoch.shape[1] != len(ch_names):
        raise ValueError(
            "PSD channel dimension does not match ch_names."
        )

    band_names = list(bands.keys())

    n_epochs = psd_by_epoch.shape[0]
    n_channels = psd_by_epoch.shape[1]
    n_bands = len(band_names)

    absolute_power_by_epoch = np.empty(
        (n_epochs, n_channels, n_bands),
        dtype=float,
    )

    # Compatibility across NumPy versions.
    integrate = getattr(np, "trapezoid", np.trapz)

    for band_index, band_name in enumerate(band_names):
        low_frequency, high_frequency = bands[band_name]

        low_frequency = float(low_frequency)
        high_frequency = float(high_frequency)

        if high_frequency <= low_frequency:
            raise ValueError(
                f"Invalid limits for {band_name}: "
                f"{low_frequency}-{high_frequency} Hz."
            )

        if low_frequency < freqs.min():
            raise ValueError(
                f"{band_name} begins at {low_frequency} Hz, "
                f"but the PSD begins at {freqs.min()} Hz."
            )

        if high_frequency > freqs.max():
            raise ValueError(
                f"{band_name} ends at {high_frequency} Hz, "
                f"but the PSD ends at {freqs.max()} Hz."
            )

        frequency_mask = (
            (freqs >= low_frequency)
            & (freqs <= high_frequency)
        )

        band_frequencies = freqs[frequency_mask]

        if len(band_frequencies) < 2:
            raise ValueError(
                f"Not enough frequency bins to integrate "
                f"{band_name} ({low_frequency}-{high_frequency} Hz)."
            )

        band_psd = psd_by_epoch[..., frequency_mask]

        # Integrate µV²/Hz across Hz to obtain µV².
        absolute_power_by_epoch[:, :, band_index] = integrate(
            band_psd,
            x=band_frequencies,
            axis=-1,
        )

    # Average across retained epochs while preserving channels.
    mean_absolute_power_by_channel = np.mean(
        absolute_power_by_epoch,
        axis=0,
    )

    # Average the channel-level powers across the scalp.
    overall_mean_absolute_power = np.mean(
        mean_absolute_power_by_channel,
        axis=0,
    )

    channel_band_power_df = pd.DataFrame(
        mean_absolute_power_by_channel,
        index=ch_names,
        columns=band_names,
    )

    channel_band_power_df.index.name = "channel"

    overall_band_power_df = pd.DataFrame({
        "band": band_names,
        "mean_absolute_power_uv2": overall_mean_absolute_power,
        "fmin_hz": [
            float(bands[name][0])
            for name in band_names
        ],
        "fmax_hz": [
            float(bands[name][1])
            for name in band_names
        ],
    })

    figure = None

    if plot:
        figure, ax = plt.subplots(figsize=(9, 5))

        ax.bar(
            band_names,
            overall_mean_absolute_power,
        )

        ax.set_title(title)
        ax.set_xlabel("Frequency band")
        ax.set_ylabel("Absolute power (µV²)")
        ax.grid(axis="y", alpha=0.25)

        figure.tight_layout()
        plt.show()

    return {
        "band_definitions_hz": {
            name: (
                float(limits[0]),
                float(limits[1]),
            )
            for name, limits in bands.items()
        },

        "band_names": band_names,
        "ch_names": ch_names,

        # Shape: epochs × channels × bands
        "absolute_power_by_epoch_uv2":
            absolute_power_by_epoch,

        # Shape: channels × bands
        "mean_absolute_power_by_channel_uv2":
            mean_absolute_power_by_channel,

        # Shape: bands
        "overall_mean_absolute_power_uv2":
            overall_mean_absolute_power,

        "channel_band_power_df":
            channel_band_power_df,

        "overall_band_power_df":
            overall_band_power_df,

        "settings": {
            "n_epochs": int(n_epochs),
            "n_channels": int(n_channels),
            "n_bands": int(n_bands),
            "integration_method": "trapezoidal",
            "input_psd_units": "µV²/Hz",
            "output_power_units": "µV²",
        },

        "figure": figure,
    }



# -----------------------------------------------------------------------------
# Express each band as a percentage of total spectral power.
def calculate_relative_band_power(
    psd_result: Mapping[str, Any],
    absolute_power_result: Mapping[str, Any],
    *,
    total_range_hz: tuple[float, float] = (1.0, 45.0),
    relative_bands: Sequence[str] | str | None = None,
    plot: bool = True,
    title: str = "Relative Power by Frequency Band",
) -> dict[str, Any]:
    """
    Calculate relative EEG band power as a percentage of total power.

    For each epoch and channel:

        relative band power (%) = absolute band power / total power * 100

    relative_bands controls which already-calculated absolute-power bands are
    converted to relative power. None preserves the original behavior and uses
    every absolute-power band.

    This allows absolute-power analysis to extend beyond the relative-power
    denominator without changing the established denominator itself.
    """
    required_psd_keys = {"freqs_hz", "ch_names", "psd_by_epoch_uv2_per_hz"}
    required_absolute_keys = {
        "band_names", "ch_names", "band_definitions_hz",
        "absolute_power_by_epoch_uv2",
    }

    missing_psd = required_psd_keys - set(psd_result.keys())
    missing_absolute = required_absolute_keys - set(absolute_power_result.keys())

    if missing_psd:
        raise KeyError(f"psd_result is missing: {sorted(missing_psd)}")
    if missing_absolute:
        raise KeyError(
            "absolute_power_result is missing: "
            f"{sorted(missing_absolute)}"
        )

    freqs = np.asarray(psd_result["freqs_hz"], dtype=float)
    psd_by_epoch = np.asarray(
        psd_result["psd_by_epoch_uv2_per_hz"], dtype=float
    )
    all_absolute_power = np.asarray(
        absolute_power_result["absolute_power_by_epoch_uv2"], dtype=float
    )

    ch_names = list(psd_result["ch_names"])
    absolute_ch_names = list(absolute_power_result["ch_names"])
    absolute_band_names = list(absolute_power_result["band_names"])

    if ch_names != absolute_ch_names:
        raise ValueError(
            "Channel names differ between psd_result and absolute_power_result."
        )
    if psd_by_epoch.ndim != 3:
        raise ValueError(
            "Expected psd_by_epoch to have shape "
            "(epochs, channels, frequencies)."
        )
    if all_absolute_power.ndim != 3:
        raise ValueError(
            "Expected absolute_power_by_epoch to have shape "
            "(epochs, channels, bands)."
        )
    if psd_by_epoch.shape[:2] != all_absolute_power.shape[:2]:
        raise ValueError(
            "Epoch and channel dimensions do not match between "
            "PSD and absolute-power results."
        )
    if all_absolute_power.shape[2] != len(absolute_band_names):
        raise ValueError(
            "Absolute-power band dimension does not match band_names."
        )

    # ------------------------------------------------------------------
    # Select which absolute-power bands become relative-power endpoints.
    # ------------------------------------------------------------------
    if relative_bands is None:
        band_names = list(absolute_band_names)
    elif isinstance(relative_bands, str):
        band_names = [relative_bands]
    else:
        band_names = [str(band) for band in relative_bands]

    if not band_names:
        raise ValueError("relative_bands must contain at least one band.")

    if len(set(band_names)) != len(band_names):
        raise ValueError("relative_bands contains duplicate band names.")

    missing_bands = [
        band for band in band_names
        if band not in absolute_band_names
    ]
    if missing_bands:
        raise ValueError(
            f"Requested relative-power bands are unavailable: {missing_bands}. "
            f"Available absolute-power bands: {absolute_band_names}"
        )

    band_index = {
        band: index for index, band in enumerate(absolute_band_names)
    }
    selected_indices = [band_index[band] for band in band_names]

    # Keep only the selected relative-power bands.
    absolute_power_by_epoch = all_absolute_power[..., selected_indices]

    band_definitions_hz = {
        band: tuple(
            map(float, absolute_power_result["band_definitions_hz"][band])
        )
        for band in band_names
    }

    # ------------------------------------------------------------------
    # Define and validate the total-power denominator.
    # ------------------------------------------------------------------
    total_low, total_high = map(float, total_range_hz)

    if total_high <= total_low:
        raise ValueError(
            "total_range_hz must have high frequency greater than low frequency."
        )

    if total_low < freqs.min() or total_high > freqs.max():
        raise ValueError(
            f"Requested total range {total_low}-{total_high} Hz is outside "
            f"the PSD range {freqs.min()}-{freqs.max()} Hz."
        )

    # Only RELATIVE-power bands must lie inside the denominator.
    for band_name, limits in band_definitions_hz.items():
        band_low, band_high = limits
        if band_low < total_low or band_high > total_high:
            raise ValueError(
                f"Relative-power band '{band_name}' "
                f"({band_low}-{band_high} Hz) lies outside the total-power "
                f"range ({total_low}-{total_high} Hz)."
            )

    total_mask = (freqs >= total_low) & (freqs <= total_high)
    total_freqs = freqs[total_mask]

    if len(total_freqs) < 2:
        raise ValueError(
            "Not enough frequency bins to calculate total power."
        )

    integrate = getattr(np, "trapezoid", np.trapz)

    # Shape: epochs x channels
    total_power_by_epoch = integrate(
        psd_by_epoch[..., total_mask],
        x=total_freqs,
        axis=-1,
    )

    # ------------------------------------------------------------------
    # Calculate relative power.
    # ------------------------------------------------------------------
    valid_total = total_power_by_epoch > 0
    relative_power_by_epoch = np.full(
        absolute_power_by_epoch.shape, np.nan, dtype=float
    )

    np.divide(
        absolute_power_by_epoch,
        total_power_by_epoch[..., np.newaxis],
        out=relative_power_by_epoch,
        where=valid_total[..., np.newaxis],
    )
    relative_power_by_epoch *= 100.0

    mean_relative_power_by_channel = np.nanmean(
        relative_power_by_epoch, axis=0
    )
    overall_mean_relative_power = np.nanmean(
        mean_relative_power_by_channel, axis=0
    )

    # ------------------------------------------------------------------
    # Create output tables.
    # ------------------------------------------------------------------
    channel_relative_power_df = pd.DataFrame(
        mean_relative_power_by_channel,
        index=ch_names,
        columns=band_names,
    )
    channel_relative_power_df.index.name = "channel"

    overall_relative_power_df = pd.DataFrame({
        "band": band_names,
        "mean_relative_power_percent": overall_mean_relative_power,
        "fmin_hz": [band_definitions_hz[band][0] for band in band_names],
        "fmax_hz": [band_definitions_hz[band][1] for band in band_names],
    })

    figure = None
    if plot:
        figure, ax = plt.subplots(figsize=(9, 5))
        ax.bar(band_names, overall_mean_relative_power)
        ax.set_title(title)
        ax.set_xlabel("Frequency band")
        ax.set_ylabel("Relative power (%)")
        ax.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        plt.show()

    return {
        "band_definitions_hz": band_definitions_hz,
        "total_power_range_hz": (total_low, total_high),
        "band_names": band_names,
        "ch_names": ch_names,

        # Shape: epochs x channels
        "total_power_by_epoch_uv2": total_power_by_epoch,

        # Shape: epochs x channels x selected relative bands
        "relative_power_by_epoch_percent": relative_power_by_epoch,

        # Shape: channels x selected relative bands
        "mean_relative_power_by_channel_percent":
            mean_relative_power_by_channel,

        # Shape: selected relative bands
        "overall_mean_relative_power_percent":
            overall_mean_relative_power,

        "channel_relative_power_df": channel_relative_power_df,
        "overall_relative_power_df": overall_relative_power_df,

        "settings": {
            "n_epochs": int(psd_by_epoch.shape[0]),
            "n_channels": int(psd_by_epoch.shape[1]),
            "n_bands": int(len(band_names)),
            "relative_bands": tuple(band_names),
            "total_power_range_hz": (total_low, total_high),
            "output_units": "percent",
        },

        "figure": figure,
    }


# -----------------------------------------------------------------------------
# Calculate configurable ratios between absolute band-power measures.
def calculate_spectral_power_ratios(
    absolute_power_result: Mapping[str, Any],
    *,
    ratio_definitions: Mapping[str, tuple[str, str]],
    summary_method: str = "ratio_of_means",
    plot: bool = True,
    title: str = "Spectral Power Ratios",
) -> dict[str, Any]:
    """
    Calculate configurable spectral-power ratios.

    Each ratio definition has the form:

        "ratio_name": ("numerator_band", "denominator_band")

    Example
    -------
    {
        "theta_beta": ("theta", "beta"),
        "alpha_theta": ("alpha", "theta"),
    }

    Parameters
    ----------
    absolute_power_result
        Output returned by calculate_absolute_band_power().

    ratio_definitions
        Mapping defining the requested ratios.

        Example:
            {
                "theta_beta": ("theta", "beta"),
                "alpha_theta": ("alpha", "theta"),
            }

    summary_method
        Determines the primary channel-level ratio:

        "ratio_of_means"
            Mean numerator power divided by mean denominator power.
            This is the recommended default because it uses the stabilized
            band-power estimates after averaging across retained epochs.

        "mean_of_ratios"
            Calculates the ratio separately for each epoch and then
            averages those epoch-level ratios.

        Both versions are returned regardless of which method is selected.

    plot
        If True, display the overall scalp-mean ratio values.

    title
        Plot title.

    Returns
    -------
    dict
        Contains epoch-level ratios, channel-level ratios, overall ratios,
        tables, definitions, and calculation settings.
    """
    # ------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------
    if not isinstance(ratio_definitions, Mapping):
        raise TypeError(
            "ratio_definitions must be a mapping such as "
            "{'theta_beta': ('theta', 'beta')}."
        )

    if len(ratio_definitions) == 0:
        raise ValueError(
            "ratio_definitions must contain at least one ratio."
        )

    summary_method = str(summary_method).lower().strip()

    allowed_methods = {
        "ratio_of_means",
        "mean_of_ratios",
    }

    if summary_method not in allowed_methods:
        raise ValueError(
            "summary_method must be either "
            "'ratio_of_means' or 'mean_of_ratios'."
        )

    required_keys = {
        "band_names",
        "ch_names",
        "absolute_power_by_epoch_uv2",
        "mean_absolute_power_by_channel_uv2",
    }

    missing_keys = required_keys - set(
        absolute_power_result.keys()
    )

    if missing_keys:
        raise KeyError(
            "absolute_power_result is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    band_names = list(
        absolute_power_result["band_names"]
    )

    ch_names = list(
        absolute_power_result["ch_names"]
    )

    absolute_power_by_epoch = np.asarray(
        absolute_power_result[
            "absolute_power_by_epoch_uv2"
        ],
        dtype=float,
    )

    mean_absolute_power_by_channel = np.asarray(
        absolute_power_result[
            "mean_absolute_power_by_channel_uv2"
        ],
        dtype=float,
    )

    # ------------------------------------------------------------
    # Validate array dimensions
    # ------------------------------------------------------------
    if absolute_power_by_epoch.ndim != 3:
        raise ValueError(
            "absolute_power_by_epoch_uv2 must have shape "
            "(epochs, channels, bands). "
            f"Received {absolute_power_by_epoch.shape}."
        )

    if mean_absolute_power_by_channel.ndim != 2:
        raise ValueError(
            "mean_absolute_power_by_channel_uv2 must have shape "
            "(channels, bands). "
            f"Received {mean_absolute_power_by_channel.shape}."
        )

    n_epochs, n_channels, n_bands = (
        absolute_power_by_epoch.shape
    )

    expected_mean_shape = (
        n_channels,
        n_bands,
    )

    if (
        mean_absolute_power_by_channel.shape
        != expected_mean_shape
    ):
        raise ValueError(
            "Mean absolute-power dimensions do not match the "
            "epoch-level absolute-power dimensions."
        )

    if n_channels != len(ch_names):
        raise ValueError(
            "Absolute-power channel dimension does not match "
            "ch_names."
        )

    if n_bands != len(band_names):
        raise ValueError(
            "Absolute-power band dimension does not match "
            "band_names."
        )

    band_to_index = {
        band_name: index
        for index, band_name in enumerate(band_names)
    }

    # ------------------------------------------------------------
    # Prepare outputs
    # ------------------------------------------------------------
    ratio_names = list(ratio_definitions.keys())
    n_ratios = len(ratio_names)

    ratio_by_epoch = np.full(
        (
            n_epochs,
            n_channels,
            n_ratios,
        ),
        np.nan,
        dtype=float,
    )

    ratio_of_mean_power_by_channel = np.full(
        (
            n_channels,
            n_ratios,
        ),
        np.nan,
        dtype=float,
    )

    mean_of_epoch_ratios_by_channel = np.full(
        (
            n_channels,
            n_ratios,
        ),
        np.nan,
        dtype=float,
    )

    numerator_bands = []
    denominator_bands = []
    invalid_denominator_counts = {}

    # ------------------------------------------------------------
    # Calculate each requested ratio
    # ------------------------------------------------------------
    for ratio_index, ratio_name in enumerate(ratio_names):
        definition = ratio_definitions[ratio_name]

        if (
            not isinstance(definition, (tuple, list))
            or len(definition) != 2
        ):
            raise ValueError(
                f"Ratio '{ratio_name}' must be defined as "
                "('numerator_band', 'denominator_band')."
            )

        numerator_band = str(definition[0])
        denominator_band = str(definition[1])

        if numerator_band not in band_to_index:
            raise ValueError(
                f"Numerator band '{numerator_band}' for ratio "
                f"'{ratio_name}' is unavailable. "
                f"Available bands: {band_names}"
            )

        if denominator_band not in band_to_index:
            raise ValueError(
                f"Denominator band '{denominator_band}' for ratio "
                f"'{ratio_name}' is unavailable. "
                f"Available bands: {band_names}"
            )

        numerator_index = band_to_index[numerator_band]
        denominator_index = band_to_index[
            denominator_band
        ]

        numerator_bands.append(numerator_band)
        denominator_bands.append(denominator_band)

        # --------------------------------------------------------
        # Epoch-level ratios
        # Shape: epochs × channels
        # --------------------------------------------------------
        epoch_numerator = absolute_power_by_epoch[
            :,
            :,
            numerator_index,
        ]

        epoch_denominator = absolute_power_by_epoch[
            :,
            :,
            denominator_index,
        ]

        valid_epoch_denominator = (
            np.isfinite(epoch_denominator)
            & (epoch_denominator > 0)
            & np.isfinite(epoch_numerator)
        )

        np.divide(
            epoch_numerator,
            epoch_denominator,
            out=ratio_by_epoch[
                :,
                :,
                ratio_index,
            ],
            where=valid_epoch_denominator,
        )

        invalid_denominator_counts[ratio_name] = int(
            valid_epoch_denominator.size
            - np.count_nonzero(valid_epoch_denominator)
        )

        # --------------------------------------------------------
        # Ratio of mean band powers for every channel
        # --------------------------------------------------------
        mean_numerator = mean_absolute_power_by_channel[
            :,
            numerator_index,
        ]

        mean_denominator = mean_absolute_power_by_channel[
            :,
            denominator_index,
        ]

        valid_mean_denominator = (
            np.isfinite(mean_denominator)
            & (mean_denominator > 0)
            & np.isfinite(mean_numerator)
        )

        np.divide(
            mean_numerator,
            mean_denominator,
            out=ratio_of_mean_power_by_channel[
                :,
                ratio_index,
            ],
            where=valid_mean_denominator,
        )

        # --------------------------------------------------------
        # Mean of the epoch-level ratios for every channel
        # --------------------------------------------------------
        mean_of_epoch_ratios_by_channel[
            :,
            ratio_index,
        ] = np.nanmean(
            ratio_by_epoch[
                :,
                :,
                ratio_index,
            ],
            axis=0,
        )

    # ------------------------------------------------------------
    # Choose the primary channel-level summary
    # ------------------------------------------------------------
    if summary_method == "ratio_of_means":
        mean_ratio_by_channel = (
            ratio_of_mean_power_by_channel.copy()
        )
    else:
        mean_ratio_by_channel = (
            mean_of_epoch_ratios_by_channel.copy()
        )

    # Overall scalp mean of the channel-level ratios.
    overall_mean_ratio = np.nanmean(
        mean_ratio_by_channel,
        axis=0,
    )

    # Also calculate each ratio after averaging power across
    # all scalp channels.
    overall_absolute_power = np.nanmean(
        mean_absolute_power_by_channel,
        axis=0,
    )

    global_power_ratio = np.full(
        n_ratios,
        np.nan,
        dtype=float,
    )

    for ratio_index, ratio_name in enumerate(ratio_names):
        numerator_index = band_to_index[
            numerator_bands[ratio_index]
        ]

        denominator_index = band_to_index[
            denominator_bands[ratio_index]
        ]

        denominator_value = overall_absolute_power[
            denominator_index
        ]

        if (
            np.isfinite(denominator_value)
            and denominator_value > 0
        ):
            global_power_ratio[ratio_index] = (
                overall_absolute_power[numerator_index]
                / denominator_value
            )

    # ------------------------------------------------------------
    # Create convenient DataFrames
    # ------------------------------------------------------------
    channel_ratio_df = pd.DataFrame(
        mean_ratio_by_channel,
        index=ch_names,
        columns=ratio_names,
    )

    channel_ratio_df.index.name = "channel"

    ratio_of_means_channel_df = pd.DataFrame(
        ratio_of_mean_power_by_channel,
        index=ch_names,
        columns=ratio_names,
    )

    ratio_of_means_channel_df.index.name = "channel"

    mean_of_ratios_channel_df = pd.DataFrame(
        mean_of_epoch_ratios_by_channel,
        index=ch_names,
        columns=ratio_names,
    )

    mean_of_ratios_channel_df.index.name = "channel"

    overall_ratio_df = pd.DataFrame({
        "ratio": ratio_names,
        "numerator_band": numerator_bands,
        "denominator_band": denominator_bands,
        "mean_channel_ratio": overall_mean_ratio,
        "global_power_ratio": global_power_ratio,
        "summary_method": summary_method,
    })

    # ------------------------------------------------------------
    # Optional plot
    # ------------------------------------------------------------
    figure = None

    if plot:
        figure, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.bar(
            ratio_names,
            overall_mean_ratio,
        )

        ax.set_title(title)
        ax.set_xlabel("Spectral-power ratio")
        ax.set_ylabel("Power ratio")
        ax.grid(
            axis="y",
            alpha=0.25,
        )

        figure.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # Return results
    # ------------------------------------------------------------
    return {
        "ratio_definitions": {
            ratio_name: (
                numerator_bands[index],
                denominator_bands[index],
            )
            for index, ratio_name in enumerate(ratio_names)
        },

        "ratio_names": ratio_names,
        "ch_names": ch_names,

        # Shape: epochs × channels × ratios
        "ratio_by_epoch":
            ratio_by_epoch,

        # Shape: channels × ratios
        "ratio_of_mean_power_by_channel":
            ratio_of_mean_power_by_channel,

        # Shape: channels × ratios
        "mean_of_epoch_ratios_by_channel":
            mean_of_epoch_ratios_by_channel,

        # Primary channel-level result selected by summary_method
        "mean_ratio_by_channel":
            mean_ratio_by_channel,

        # Shape: ratios
        "overall_mean_ratio":
            overall_mean_ratio,

        # Ratio calculated from globally averaged band powers
        "global_power_ratio":
            global_power_ratio,

        "channel_ratio_df":
            channel_ratio_df,

        "ratio_of_means_channel_df":
            ratio_of_means_channel_df,

        "mean_of_ratios_channel_df":
            mean_of_ratios_channel_df,

        "overall_ratio_df":
            overall_ratio_df,

        "invalid_denominator_counts":
            invalid_denominator_counts,

        "settings": {
            "n_epochs": int(n_epochs),
            "n_channels": int(n_channels),
            "n_ratios": int(n_ratios),
            "summary_method": summary_method,
            "ratio_units": "unitless",
        },

        "figure": figure,
    }



# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def save_plot_png(plot_obj, filename, *, output_dir, dpi=300):
    """Save any notebook matplotlib figure as a high-resolution PNG."""
    if isinstance(plot_obj, dict): fig = plot_obj["figure"]
    elif hasattr(plot_obj, "get_figure"): fig = plot_obj.get_figure()
    else: fig = plot_obj

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return output_path

# Create a reusable publication-ready line plot with optional uncertainty.
def plot_professional_line(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,

    # Multiple-line grouping
    hue: str | None = None,
    hue_order: Sequence[Any] | None = None,
    style: str | None = None,
    style_order: Sequence[Any] | None = None,
    units: str | None = None,

    # Display labels
    series_alias: Mapping[Any, str] | None = None,
    series_label: str | None = None,

    # Line appearance
    palette: Mapping[Any, str] | Sequence[str] | str | None = None,
    color: str | None = None,
    linewidth: float = 2.5,
    linestyle: str = "-",
    marker: str | None = None,
    markers: bool | Mapping[Any, str] = False,
    markersize: float = 6.0,
    dashes: bool | Mapping[Any, Any] = True,
    line_alpha: float = 1.0,

    # Aggregation and automatically calculated uncertainty
    estimator: str | Callable[..., float] | None = "mean",
    errorbar: str | tuple[str, float] | None = None,
    error_band_alpha: float = 0.18,

    # Optional precomputed uncertainty columns
    lower: str | None = None,
    upper: str | None = None,
    fill_alpha: float = 0.18,
    fill_edgecolor: str | None = None,

    # Figure and typography
    figsize: tuple[float, float] = (9.0, 5.0),
    font_size: float = 12.0,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,

    # Axis controls
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: Sequence[float] | None = None,
    yticks: Sequence[float] | None = None,
    x_tick_rotation: float = 0.0,

    # Optional reference lines and shaded regions
    horizontal_lines: Sequence[Mapping[str, Any]] | None = None,
    vertical_lines: Sequence[Mapping[str, Any]] | None = None,
    x_spans: Sequence[Mapping[str, Any]] | None = None,

    # Legend
    show_legend: bool = True,
    legend_loc: str = "best",
    legend_title: str | None = None,
    legend_ncol: int = 1,

    # General styling
    sns_style: str = "whitegrid",
    grid: bool = True,
    grid_axis: str = "both",
    remove_top_right_spines: bool = True,
    sort: bool = True,
    show: bool = True,
) -> tuple[Any, Any]:
    """
    Create a publication- and presentation-ready line plot.

    Supports single or multiple lines, aliases, custom palettes,
    automatically estimated uncertainty bands, and precomputed
    lower/upper uncertainty intervals.

    Parameters
    ----------
    data
        DataFrame containing the plotting data.

    x, y
        Columns used for the horizontal and vertical axes.

    hue
        Optional column defining separate colored lines.

    hue_order
        Order of the raw hue values.

    style
        Optional column defining separate markers or line styles.

    units
        Optional observation identifier. When supplied, estimator must
        be None and one line is drawn for each unit.

    series_alias
        Mapping from raw hue/style values to display labels.

        Example:
            {
                "EO": "Eyes open",
                "EC": "Eyes closed",
            }

    series_label
        Optional legend label for a single line when hue and style are None.

    estimator
        Aggregation used by seaborn.lineplot.

        Examples:
            "mean"
            np.median
            None

    errorbar
        Automatically calculated uncertainty around the estimator.

        Examples:
            None
            "sd"
            "se"
            ("ci", 95)

        This is useful when multiple subjects or observations contribute
        values at the same x-coordinate.

    lower, upper
        Optional columns containing precomputed lower and upper uncertainty
        bounds. Supply both or neither.

    horizontal_lines
        Optional reference-line specifications.

        Example:
            [
                {
                    "y": 0,
                    "label": "Baseline",
                    "color": "#6B7280",
                    "linestyle": "--",
                }
            ]

    vertical_lines
        Optional vertical reference-line specifications.

    x_spans
        Optional shaded x-axis regions.

        Example:
            [
                {
                    "xmin": 8,
                    "xmax": 13,
                    "label": "Alpha",
                    "color": "#D4A72C",
                    "alpha": 0.08,
                }
            ]

    Returns
    -------
    figure, axis
        Matplotlib Figure and Axes.
    """
    # ------------------------------------------------------------
    # Validate input DataFrame and columns
    # ------------------------------------------------------------
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame. "
            f"Received {type(data).__name__}."
        )

    required_columns = {x, y}

    for optional_column in (
        hue,
        style,
        units,
        lower,
        upper,
    ):
        if optional_column is not None:
            required_columns.add(optional_column)

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise KeyError(
            "data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("data is empty.")

    if (lower is None) != (upper is None):
        raise ValueError(
            "lower and upper must either both be supplied or both be None."
        )

    if lower is not None and errorbar is not None:
        raise ValueError(
            "Use either precomputed lower/upper bounds or errorbar, "
            "not both."
        )

    if units is not None and estimator is not None:
        raise ValueError(
            "When units is supplied, estimator must be None."
        )

    if units is not None and errorbar is not None:
        raise ValueError(
            "errorbar must be None when units is supplied."
        )

    if grid_axis not in {"x", "y", "both"}:
        raise ValueError(
            "grid_axis must be 'x', 'y', or 'both'."
        )

    if legend_ncol < 1:
        raise ValueError(
            "legend_ncol must be at least 1."
        )

    plot_data = data.copy()

    numeric_columns = [y]

    if lower is not None:
        numeric_columns.extend([lower, upper])

    for column in numeric_columns:
        plot_data[column] = pd.to_numeric(
            plot_data[column],
            errors="coerce",
        )

    drop_columns = list(required_columns)

    plot_data = plot_data.dropna(
        subset=drop_columns,
    )

    if plot_data.empty:
        raise ValueError(
            "No usable rows remain after removing missing or "
            "non-numeric values."
        )

    # ------------------------------------------------------------
    # Resolve line and style ordering
    # ------------------------------------------------------------
    if hue is not None:
        available_hue_values = (
            plot_data[hue]
            .drop_duplicates()
            .tolist()
        )

        if hue_order is None:
            hue_order_used = available_hue_values
        else:
            hue_order_used = list(hue_order)

            missing_hue_values = [
                value
                for value in hue_order_used
                if value not in set(available_hue_values)
            ]

            if missing_hue_values:
                raise ValueError(
                    "The following hue_order values are absent "
                    f"from data['{hue}']: {missing_hue_values}"
                )
    else:
        hue_order_used = None

    if style is not None:
        available_style_values = (
            plot_data[style]
            .drop_duplicates()
            .tolist()
        )

        if style_order is None:
            style_order_used = available_style_values
        else:
            style_order_used = list(style_order)

            missing_style_values = [
                value
                for value in style_order_used
                if value not in set(available_style_values)
            ]

            if missing_style_values:
                raise ValueError(
                    "The following style_order values are absent "
                    f"from data['{style}']: {missing_style_values}"
                )
    else:
        style_order_used = None

    # ------------------------------------------------------------
    # Resolve aliases
    # ------------------------------------------------------------
    if series_alias is None:
        series_alias = {}

    if not isinstance(series_alias, Mapping):
        raise TypeError(
            "series_alias must be a mapping such as "
            "{'EO': 'Eyes open', 'EC': 'Eyes closed'}."
        )

    alias_lookup = {
        str(raw_value): str(display_value)
        for raw_value, display_value in series_alias.items()
    }

    # ------------------------------------------------------------
    # Resolve a stable palette
    # ------------------------------------------------------------
    default_single_color = "#355C8A"
    single_line_color = color or default_single_color

    palette_used: Mapping[Any, Any] | Sequence[Any] | str | None

    if hue is not None:
        if isinstance(palette, Mapping):
            missing_palette_values = [
                value
                for value in hue_order_used
                if value not in palette
            ]

            if missing_palette_values:
                raise ValueError(
                    "palette is missing colors for: "
                    f"{missing_palette_values}"
                )

            palette_used = dict(palette)

        else:
            if isinstance(palette, str) or palette is None:
                resolved_colors = sns.color_palette(
                    palette=palette,
                    n_colors=len(hue_order_used),
                )
            else:
                resolved_colors = list(palette)

                if len(resolved_colors) < len(hue_order_used):
                    raise ValueError(
                        "The palette sequence contains fewer colors "
                        "than the number of hue values."
                    )

            palette_used = {
                hue_value: resolved_colors[index]
                for index, hue_value in enumerate(hue_order_used)
            }

    else:
        palette_used = None

    # ------------------------------------------------------------
    # Figure and Seaborn line plot
    # ------------------------------------------------------------
    sns.set_theme(
        style=sns_style,
        context="notebook",
    )

    figure, axis = plt.subplots(
        figsize=figsize,
    )

    lineplot_kwargs: dict[str, Any] = {
        "data": plot_data,
        "x": x,
        "y": y,
        "estimator": estimator,
        "errorbar": errorbar,
        "sort": sort,
        "linewidth": linewidth,
        "alpha": line_alpha,
        "err_style": "band",
        "err_kws": {
            "alpha": error_band_alpha,
        },
        "ax": axis,
    }

    if hue is not None:
        lineplot_kwargs.update({
            "hue": hue,
            "hue_order": hue_order_used,
            "palette": palette_used,
        })
    else:
        lineplot_kwargs.update({
            "color": single_line_color,
        })

    if style is not None:
        lineplot_kwargs.update({
            "style": style,
            "style_order": style_order_used,
            "markers": markers,
            "dashes": dashes,
        })
    else:
        lineplot_kwargs["linestyle"] = linestyle

        if marker is not None:
            lineplot_kwargs.update({
                "marker": marker,
                "markersize": markersize,
            })

    if units is not None:
        lineplot_kwargs["units"] = units

    if (
        hue is None
        and style is None
        and series_label is not None
    ):
        lineplot_kwargs["label"] = series_label

    sns.lineplot(
        **lineplot_kwargs,
    )

    # ------------------------------------------------------------
    # Draw precomputed uncertainty intervals
    # ------------------------------------------------------------
    if lower is not None and upper is not None:
        grouping_column = (
            hue
            if hue is not None
            else style
        )

        if grouping_column is None:
            grouped_data = [
                (
                    None,
                    plot_data.sort_values(x),
                )
            ]
        else:
            group_order = (
                hue_order_used
                if grouping_column == hue
                else style_order_used
            )

            grouped_data = [
                (
                    group_value,
                    plot_data[
                        plot_data[grouping_column]
                        == group_value
                    ].sort_values(x),
                )
                for group_value in group_order
            ]

        for group_value, group_df in grouped_data:
            if group_df.empty:
                continue

            if hue is not None:
                fill_color = palette_used[group_value]
            else:
                fill_color = single_line_color

            axis.fill_between(
                group_df[x].to_numpy(),
                group_df[lower].to_numpy(dtype=float),
                group_df[upper].to_numpy(dtype=float),
                color=fill_color,
                alpha=fill_alpha,
                edgecolor=fill_edgecolor,
                linewidth=0.0 if fill_edgecolor is None else 0.8,
                zorder=1,
            )

    # ------------------------------------------------------------
    # Optional shaded x-axis regions
    # ------------------------------------------------------------
    if x_spans is not None:
        for span in x_spans:
            if not isinstance(span, Mapping):
                raise TypeError(
                    "Each x_spans entry must be a mapping."
                )

            if "xmin" not in span or "xmax" not in span:
                raise KeyError(
                    "Each x_spans entry requires xmin and xmax."
                )

            axis.axvspan(
                span["xmin"],
                span["xmax"],
                color=span.get("color", "#9CA3AF"),
                alpha=float(span.get("alpha", 0.10)),
                label=span.get("label"),
                zorder=0,
            )

    # ------------------------------------------------------------
    # Optional horizontal reference lines
    # ------------------------------------------------------------
    if horizontal_lines is not None:
        for line in horizontal_lines:
            if not isinstance(line, Mapping):
                raise TypeError(
                    "Each horizontal_lines entry must be a mapping."
                )

            if "y" not in line:
                raise KeyError(
                    "Each horizontal_lines entry requires y."
                )

            axis.axhline(
                y=float(line["y"]),
                color=line.get("color", "#6B7280"),
                linewidth=float(line.get("linewidth", 1.5)),
                linestyle=line.get("linestyle", "--"),
                alpha=float(line.get("alpha", 1.0)),
                label=line.get("label"),
                zorder=float(line.get("zorder", 2)),
            )

    # ------------------------------------------------------------
    # Optional vertical reference lines
    # ------------------------------------------------------------
    if vertical_lines is not None:
        for line in vertical_lines:
            if not isinstance(line, Mapping):
                raise TypeError(
                    "Each vertical_lines entry must be a mapping."
                )

            if "x" not in line:
                raise KeyError(
                    "Each vertical_lines entry requires x."
                )

            axis.axvline(
                x=float(line["x"]),
                color=line.get("color", "#6B7280"),
                linewidth=float(line.get("linewidth", 1.5)),
                linestyle=line.get("linestyle", "--"),
                alpha=float(line.get("alpha", 1.0)),
                label=line.get("label"),
                zorder=float(line.get("zorder", 2)),
            )

    # ------------------------------------------------------------
    # Titles and axis labels
    # ------------------------------------------------------------
    axis.set_title(
        title or "",
        fontsize=font_size + 2,
        fontweight="bold",
        pad=14,
    )

    axis.set_xlabel(
        xlabel if xlabel is not None else x,
        fontsize=font_size,
        fontweight="bold",
        labelpad=8,
    )

    axis.set_ylabel(
        ylabel if ylabel is not None else y,
        fontsize=font_size,
        fontweight="bold",
        labelpad=8,
    )

    axis.tick_params(
        axis="both",
        labelsize=font_size,
    )

    axis.tick_params(
        axis="x",
        rotation=x_tick_rotation,
    )

    for label in axis.get_xticklabels():
        label.set_fontweight("bold")

    for label in axis.get_yticklabels():
        label.set_fontweight("bold")

    # ------------------------------------------------------------
    # Axis limits and ticks
    # ------------------------------------------------------------
    if xlim is not None:
        axis.set_xlim(*xlim)

    if ylim is not None:
        axis.set_ylim(*ylim)

    if xticks is not None:
        axis.set_xticks(
            list(xticks)
        )

    if yticks is not None:
        axis.set_yticks(
            list(yticks)
        )

    # ------------------------------------------------------------
    # Grid and spines
    # ------------------------------------------------------------
    axis.set_axisbelow(True)

    if grid:
        axis.grid(
            axis=grid_axis,
            linestyle="--",
            linewidth=0.8,
            alpha=0.28,
        )
    else:
        axis.grid(False)

    if remove_top_right_spines:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axis.spines["left"].set_linewidth(1.0)
    axis.spines["bottom"].set_linewidth(1.0)

    # ------------------------------------------------------------
    # Legend and aliases
    # ------------------------------------------------------------
    handles, labels = axis.get_legend_handles_labels()

    renamed_labels = [
        alias_lookup.get(label, label)
        for label in labels
    ]

    unique_handles = []
    unique_labels = []
    seen_labels = set()

    for handle, label in zip(
        handles,
        renamed_labels,
    ):
        if not label or label.startswith("_"):
            continue

        if label in seen_labels:
            continue

        seen_labels.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)

    if show_legend and unique_handles:
        legend = axis.legend(
            unique_handles,
            unique_labels,
            title=legend_title,
            loc=legend_loc,
            ncol=legend_ncol,
            frameon=True,
            prop={
                "size": font_size,
                "weight": "bold",
            },
        )

        if legend is not None:
            legend.get_frame().set_alpha(0.95)

            if legend.get_title() is not None:
                legend.get_title().set_fontsize(
                    font_size
                )
                legend.get_title().set_fontweight(
                    "bold"
                )

    else:
        existing_legend = axis.get_legend()

        if existing_legend is not None:
            existing_legend.remove()

    figure.tight_layout()

    if show:
        plt.show()

    return figure, axis

# -----------------------------------------------------------------------------
# Plot absolute or relative channel-level band power as EEG topomaps.
def plot_band_power_topomaps(
    epochs: mne.BaseEpochs,
    band_power_result: Mapping[str, Any],
    *,
    power_type: str = "absolute",

    # Band selection and display
    band_order: Sequence[str] | None = None,
    band_alias: Mapping[str, str] | None = None,

    # Color scaling
    shared_scale: bool = False,
    vlim: tuple[float, float] | None = None,
    zero_based: bool = True,
    vmax_percentile: float | None = None,

    # Topomap appearance
    cmap: str = "viridis",
    contours: int = 6,
    sensors: bool | str = "k.",
    outlines: str = "head",
    extrapolate: str = "auto",
    sphere: str | tuple[float, float, float, float] = "auto",
    image_interp: str = "cubic",

    # Figure layout
    n_cols: int = 3,
    figsize: tuple[float, float] | None = None,
    font_size: float = 12.0,

    # Labels
    title: str | None = None,
    show_scale_note: bool = True,
    colorbar_label: str | None = None,
    colorbar_decimals: int = 1,

    # Output
    show: bool = True,
) -> dict[str, Any]:
    """
    Create publication- and presentation-ready EEG band-power topographic maps.

    Parameters
    ----------
    epochs
        Clean MNE Epochs object used for the qEEG analysis.
        Sensor locations are obtained from epochs.info.

    band_power_result
        For power_type="absolute":
            Output from calculate_absolute_band_power().

        For power_type="relative":
            Output from calculate_relative_band_power().

    power_type
        Either "absolute" or "relative".

    band_order
        Optional order of frequency bands.

        Example:
            ["delta", "theta", "alpha", "beta", "gamma"]

        If None, the order stored in band_power_result["band_names"] is used.

    band_alias
        Optional mapping from raw band names to displayed labels.

        Example:
            {
                "delta": "Delta",
                "theta": "Theta",
                "alpha": "Alpha",
                "beta": "Beta",
                "gamma": "Gamma",
            }

    shared_scale
        If False, each band uses its own color scale.

        If True, all bands use the same color scale.

    vlim
        Optional explicit color limits applied to every map.

        Example:
            (0.0, 30.0)

        When provided, this overrides automatic shared or independent scaling.

    zero_based
        If True, the lower color limit is fixed at zero when vlim is not
        explicitly supplied.

    vmax_percentile
        Optional upper percentile used to determine automatic color limits.

        Example:
            98.0

        This can reduce the effect of an unusually large single-channel value.
        If None, the maximum observed value is used.

    cmap
        Matplotlib colormap name.

    contours
        Number of contour lines.

    sensors
        Whether and how to display electrode locations.

        Examples:
            True
            False
            "k."

    n_cols
        Number of topographic maps per row.

    figsize
        Overall figure size. If None, it is determined from the number of rows
        and columns.

    font_size
        Base font size.

    title
        Overall figure title. If None, an automatic title is used.

    show_scale_note
        Whether to display shared-versus-independent scaling below the title.

    colorbar_label
        Optional custom colorbar label. If None:
            absolute -> "Absolute power (µV²)"
            relative -> "Relative power (%)"

    colorbar_decimals
        Number of decimal places shown on colorbar ticks.

    show
        Whether to display the figure immediately.

    Returns
    -------
    dict
        Contains the figure, axes, colorbars, plotted values, channel names,
        band names, display labels, units, and applied color limits.
    """
    # ------------------------------------------------------------
    # Validate epochs
    # ------------------------------------------------------------
    if not isinstance(epochs, mne.BaseEpochs):
        raise TypeError(
            "epochs must be an MNE Epochs object. "
            f"Received {type(epochs).__name__}."
        )

    power_type = str(power_type).lower().strip()

    # ------------------------------------------------------------
    # Select the appropriate result array
    # ------------------------------------------------------------
    if power_type == "absolute":
        value_key = "mean_absolute_power_by_channel_uv2"
        units = "µV²"
        default_colorbar_label = "Absolute power (µV²)"
        default_title = "Absolute Band-Power Topographic Maps"

    elif power_type == "relative":
        value_key = "mean_relative_power_by_channel_percent"
        units = "%"
        default_colorbar_label = "Relative power (%)"
        default_title = "Relative Band-Power Topographic Maps"

    else:
        raise ValueError(
            "power_type must be either 'absolute' or 'relative'."
        )

    required_keys = {
        value_key,
        "band_names",
        "ch_names",
    }

    missing_keys = required_keys - set(
        band_power_result.keys()
    )

    if missing_keys:
        raise KeyError(
            "band_power_result is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    values_original = np.asarray(
        band_power_result[value_key],
        dtype=float,
    )

    original_band_names = list(
        band_power_result["band_names"]
    )

    ch_names = list(
        band_power_result["ch_names"]
    )

    # Expected shape: channels × bands
    expected_shape = (
        len(ch_names),
        len(original_band_names),
    )

    if values_original.ndim != 2:
        raise ValueError(
            f"{value_key} must have shape (channels, bands). "
            f"Received {values_original.shape}."
        )

    if values_original.shape != expected_shape:
        raise ValueError(
            "Band-power array dimensions do not match the stored "
            "channel and band names. "
            f"Expected {expected_shape}, "
            f"received {values_original.shape}."
        )

    if not np.isfinite(values_original).all():
        raise ValueError(
            "Topographic-map values contain NaN or infinite values."
        )

    # ------------------------------------------------------------
    # Resolve band order
    # ------------------------------------------------------------
    if band_order is None:
        band_names = original_band_names
    else:
        band_names = list(band_order)

        missing_bands = [
            band
            for band in band_names
            if band not in original_band_names
        ]

        if missing_bands:
            raise ValueError(
                "The following requested bands are unavailable: "
                f"{missing_bands}. "
                f"Available bands: {original_band_names}"
            )

        if len(set(band_names)) != len(band_names):
            raise ValueError(
                "band_order contains duplicate band names."
            )

    band_index_lookup = {
        band: index
        for index, band in enumerate(original_band_names)
    }

    selected_band_indices = [
        band_index_lookup[band]
        for band in band_names
    ]

    values_by_channel = values_original[
        :,
        selected_band_indices,
    ]

    # ------------------------------------------------------------
    # Resolve displayed band labels
    # ------------------------------------------------------------
    if band_alias is None:
        band_alias = {}

    if not isinstance(band_alias, Mapping):
        raise TypeError(
            "band_alias must be a mapping such as "
            "{'delta': 'Delta', 'theta': 'Theta'}."
        )

    band_display_labels = [
        str(band_alias.get(band, band.capitalize()))
        for band in band_names
    ]

    if len(set(band_display_labels)) != len(
        band_display_labels
    ):
        raise ValueError(
            "band_alias creates duplicate displayed labels."
        )

    # ------------------------------------------------------------
    # Align epochs.info with result channel order
    # ------------------------------------------------------------
    missing_channels = [
        channel
        for channel in ch_names
        if channel not in epochs.ch_names
    ]

    if missing_channels:
        raise ValueError(
            "The following result channels are not present in epochs: "
            f"{missing_channels}"
        )

    channel_indices = [
        epochs.ch_names.index(channel)
        for channel in ch_names
    ]

    topo_info = mne.pick_info(
        epochs.info,
        sel=channel_indices,
        copy=True,
    )

    if list(topo_info["ch_names"]) != ch_names:
        raise RuntimeError(
            "Channel order in topo_info does not match the "
            "band-power result."
        )

    # ------------------------------------------------------------
    # Validate electrode coordinates
    # ------------------------------------------------------------
    invalid_locations = []

    for channel_info in topo_info["chs"]:
        xyz = np.asarray(
            channel_info["loc"][:3],
            dtype=float,
        )

        if (
            not np.isfinite(xyz).all()
            or np.allclose(xyz, 0.0)
        ):
            invalid_locations.append(
                channel_info["ch_name"]
            )

    if invalid_locations:
        raise ValueError(
            "The following channels do not have valid sensor "
            f"locations: {invalid_locations}"
        )

    # ------------------------------------------------------------
    # Validate plotting controls
    # ------------------------------------------------------------
    n_bands = len(band_names)

    if n_bands == 0:
        raise ValueError(
            "No frequency bands were selected."
        )

    if n_cols < 1:
        raise ValueError(
            "n_cols must be at least 1."
        )

    n_cols_used = min(
        int(n_cols),
        n_bands,
    )

    n_rows = int(
        np.ceil(n_bands / n_cols_used)
    )

    if vmax_percentile is not None:
        vmax_percentile = float(vmax_percentile)

        if not 0 < vmax_percentile <= 100:
            raise ValueError(
                "vmax_percentile must be greater than zero "
                "and less than or equal to 100."
            )

    if colorbar_decimals < 0:
        raise ValueError(
            "colorbar_decimals must be zero or greater."
        )

    if vlim is not None:
        if len(vlim) != 2:
            raise ValueError(
                "vlim must contain exactly two values."
            )

        explicit_vmin = float(vlim[0])
        explicit_vmax = float(vlim[1])

        if explicit_vmax <= explicit_vmin:
            raise ValueError(
                "The upper vlim value must be greater "
                "than the lower value."
            )

        explicit_vlim = (
            explicit_vmin,
            explicit_vmax,
        )

    else:
        explicit_vlim = None

    # ------------------------------------------------------------
    # Resolve figure size
    # ------------------------------------------------------------
    if figsize is None:
        figure_width = 4.6 * n_cols_used
        figure_height = 4.9 * n_rows + 1.0

        figsize_used = (
            figure_width,
            figure_height,
        )
    else:
        figsize_used = figsize

    figure, axes = plt.subplots(
        n_rows,
        n_cols_used,
        figsize=figsize_used,
        squeeze=False,
    )

    figure.patch.set_facecolor("white")

    axes_flat = axes.ravel()

    # ------------------------------------------------------------
    # Determine automatic color limits
    # ------------------------------------------------------------
    # Resolve a positive maximum for automatic topomap color scaling.
    def _get_maximum(
        values: np.ndarray,
    ) -> float:
        if vmax_percentile is None:
            maximum = float(
                np.nanmax(values)
            )
        else:
            maximum = float(
                np.nanpercentile(
                    values,
                    vmax_percentile,
                )
            )

        if maximum <= 0:
            raise ValueError(
                "Topographic-map values must contain "
                "positive power values."
            )

        return maximum

    if explicit_vlim is not None:
        shared_vlim = explicit_vlim

    elif shared_scale:
        shared_maximum = _get_maximum(
            values_by_channel
        )

        shared_minimum = (
            0.0
            if zero_based
            else float(
                np.nanmin(values_by_channel)
            )
        )

        shared_vlim = (
            shared_minimum,
            shared_maximum,
        )

    else:
        shared_vlim = None

    applied_vlims: dict[str, tuple[float, float]] = {}
    colorbars = []

    # ------------------------------------------------------------
    # Plot each band
    # ------------------------------------------------------------
    for band_index, (
        band_name,
        display_label,
    ) in enumerate(
        zip(
            band_names,
            band_display_labels,
        )
    ):
        axis = axes_flat[band_index]
        axis.set_facecolor("white")

        band_values = values_by_channel[
            :,
            band_index,
        ]

        if shared_vlim is not None:
            current_vlim = shared_vlim

        else:
            band_maximum = _get_maximum(
                band_values
            )

            band_minimum = (
                0.0
                if zero_based
                else float(
                    np.nanmin(band_values)
                )
            )

            current_vlim = (
                band_minimum,
                band_maximum,
            )

        applied_vlims[band_name] = (
            float(current_vlim[0]),
            float(current_vlim[1]),
        )

        image, _ = mne.viz.plot_topomap(
            band_values,
            topo_info,
            ch_type="eeg",
            axes=axis,
            show=False,
            sensors=sensors,
            contours=contours,
            cmap=cmap,
            vlim=current_vlim,
            outlines=outlines,
            extrapolate=extrapolate,
            sphere=sphere,
            image_interp=image_interp,
        )

        axis.set_title(
            display_label,
            fontsize=font_size + 1,
            fontweight="bold",
            pad=14,
        )

        colorbar = figure.colorbar(
            image,
            ax=axis,
            shrink=0.74,
            pad=0.045,
            aspect=24,
        )

        colorbar.set_label(
            colorbar_label or default_colorbar_label,
            fontsize=font_size - 1,
            fontweight="bold",
            labelpad=8,
        )

        colorbar.ax.tick_params(
            labelsize=font_size - 2,
            width=1.0,
        )

        for tick_label in colorbar.ax.get_yticklabels():
            tick_label.set_fontweight("bold")

        colorbar.formatter.set_powerlimits(
            (-3, 4)
        )

        colorbar.ax.yaxis.set_major_formatter(
            plt.FormatStrFormatter(
                f"%.{colorbar_decimals}f"
            )
        )

        colorbar.update_ticks()

        colorbar.outline.set_linewidth(0.8)

        colorbars.append(colorbar)

    # ------------------------------------------------------------
    # Hide unused subplot positions
    # ------------------------------------------------------------
    for unused_index in range(
        n_bands,
        len(axes_flat),
    ):
        axes_flat[unused_index].axis("off")

    # ------------------------------------------------------------
    # Overall title
    # ------------------------------------------------------------
    if explicit_vlim is not None:
        scale_note = "Fixed color scale"
    elif shared_scale:
        scale_note = "Shared color scale across bands"
    else:
        scale_note = "Independent color scale for each band"

    if show_scale_note:
        complete_title = (
            f"{title or default_title}\n"
            f"{scale_note}"
        )
    else:
        complete_title = (
            title or default_title
        )

    figure.suptitle(
        complete_title,
        fontsize=font_size + 4,
        fontweight="bold",
        y=0.975,
    )

    # Explicit spacing keeps lower-row labels visible.
    figure.subplots_adjust(
        top=0.84 if n_rows > 1 else 0.79,
        bottom=0.06,
        left=0.04,
        right=0.97,
        hspace=0.48,
        wspace=0.30,
    )

    if show:
        plt.show()

    return {
        "figure": figure,
        "axes": axes,
        "colorbars": colorbars,

        "power_type": power_type,
        "units": units,

        "band_names": band_names,
        "band_display_labels": band_display_labels,
        "ch_names": ch_names,

        "values_by_channel": values_by_channel,
        "applied_vlims": applied_vlims,

        "shared_scale": bool(shared_scale),
        "explicit_vlim": explicit_vlim,
        "zero_based": bool(zero_based),
        "vmax_percentile": vmax_percentile,

        "settings": {
            "n_bands": int(n_bands),
            "n_channels": int(len(ch_names)),
            "n_rows": int(n_rows),
            "n_cols": int(n_cols_used),
            "cmap": cmap,
            "contours": contours,
            "figsize": figsize_used,
        },
    }


# -----------------------------------------------------------------------------
# Plot the recording-level mean PSD with a shaded plus-or-minus one SD band.
def plot_mean_psd_with_std(
    data: pd.DataFrame,
    *,
    frequency_col: str = "frequency_hz",
    psd_col: str = "mean_psd_db",
    recording_col: str = "recording_id",

    # Optional grouping, such as EO/EC, cohort, dose, or timepoint
    group_col: str | None = None,
    group_order: Sequence[Any] | None = None,
    group_alias: Mapping[Any, str] | None = None,

    # Appearance
    palette: Mapping[Any, str] | Sequence[str] | str | None = None,
    color: str = "#355C8A",
    linewidth: float = 2.8,
    fill_alpha: float = 0.20,

    # Figure labels
    title: str = "Mean Power Spectral Density",
    xlabel: str = "Frequency (Hz)",
    ylabel: str = "PSD (dB re 1 µV²/Hz)",

    # Axis controls
    xlim: tuple[float, float] | None = (0.5, 45.0),
    ylim: tuple[float, float] | None = None,
    xticks: Sequence[float] | None = None,
    yticks: Sequence[float] | None = None,

    # Layout
    figsize: tuple[float, float] = (9.0, 5.0),
    font_size: float = 12.0,

    # Legend
    show_legend: bool | None = None,
    legend_title: str | None = None,
    legend_loc: str = "best",
    legend_ncol: int = 1,

    show: bool = True,
) -> tuple[Any, Any, pd.DataFrame]:
    """
    Plot group-level mean PSD curves with shaded ±1 SD regions.

    Each recording first contributes one PSD value at each frequency.
    The function then calculates, at every frequency:

        mean PSD across recordings
        standard deviation across recordings
        lower bound = mean - SD
        upper bound = mean + SD

    When `group_col` is supplied, one mean curve and SD ribbon are drawn
    for each group.

    Examples of useful grouping columns:
        - condition: EO versus EC
        - cohort
        - treatment
        - dose
        - timepoint
        - visit

    Parameters
    ----------
    data
        Tidy DataFrame containing one PSD curve per recording.

    frequency_col
        Frequency column.

    psd_col
        PSD measurement column.

    recording_col
        Unique recording identifier.

    group_col
        Optional column defining separate mean curves.

    group_order
        Optional order of the raw group values.

    group_alias
        Optional mapping from raw group values to displayed labels.

    Returns
    -------
    figure
        Matplotlib Figure.

    axis
        Matplotlib Axes.

    summary_df
        Frequency-level summary containing mean, SD, bounds, and the
        number of recordings contributing at each frequency.
    """
    # ------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame. "
            f"Received {type(data).__name__}."
        )

    required_columns = {
        frequency_col,
        psd_col,
        recording_col,
    }

    if group_col is not None:
        required_columns.add(group_col)

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise KeyError(
            "data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("data is empty.")

    plot_data = data.copy()

    plot_data[frequency_col] = pd.to_numeric(
        plot_data[frequency_col],
        errors="coerce",
    )

    plot_data[psd_col] = pd.to_numeric(
        plot_data[psd_col],
        errors="coerce",
    )

    drop_columns = [
        frequency_col,
        psd_col,
        recording_col,
    ]

    if group_col is not None:
        drop_columns.append(group_col)

    plot_data = plot_data.dropna(
        subset=drop_columns,
    )

    if plot_data.empty:
        raise ValueError(
            "No usable rows remain after removing missing or "
            "non-numeric values."
        )

    # ------------------------------------------------------------
    # Validate group ordering
    # ------------------------------------------------------------
    if group_col is not None:
        available_groups = (
            plot_data[group_col]
            .drop_duplicates()
            .tolist()
        )

        if group_order is None:
            group_order_used = available_groups
        else:
            group_order_used = list(group_order)

            missing_groups = [
                group
                for group in group_order_used
                if group not in set(available_groups)
            ]

            if missing_groups:
                raise ValueError(
                    "The following group_order values are absent "
                    f"from data['{group_col}']: {missing_groups}"
                )
    else:
        group_order_used = None

    if group_alias is None:
        group_alias = {}

    # ------------------------------------------------------------
    # First ensure one value per recording and frequency
    # ------------------------------------------------------------
    recording_group_columns = [
        recording_col,
        frequency_col,
    ]

    if group_col is not None:
        recording_group_columns.insert(
            0,
            group_col,
        )

    recording_level_df = (
        plot_data
        .groupby(
            recording_group_columns,
            observed=True,
            sort=False,
        )[psd_col]
        .mean()
        .reset_index()
    )

    # ------------------------------------------------------------
    # Calculate mean ± SD across recordings
    # ------------------------------------------------------------
    summary_group_columns = [
        frequency_col,
    ]

    if group_col is not None:
        summary_group_columns.insert(
            0,
            group_col,
        )

    summary_df = (
        recording_level_df
        .groupby(
            summary_group_columns,
            observed=True,
            sort=False,
        )[psd_col]
        .agg(
            mean_psd="mean",
            std_psd=lambda values: values.std(ddof=1),
            n_recordings="count",
        )
        .reset_index()
    )

    # A single recording has an undefined sample SD.
    # Use a zero-width ribbon in that case.
    summary_df["std_psd"] = summary_df["std_psd"].fillna(0.0) 

    summary_df["lower_psd"] = (summary_df["mean_psd"]- summary_df["std_psd"])

    summary_df["upper_psd"] = (summary_df["mean_psd"]+ summary_df["std_psd"])









    sort_columns = ( [group_col, frequency_col]
        if group_col is not None
        else [frequency_col]
    )

    summary_df = ( summary_df.sort_values(sort_columns).reset_index(drop=True))

    # ------------------------------------------------------------
    # Legend defaults
    # ------------------------------------------------------------
    if show_legend is None:
        show_legend_used = (
            group_col is not None
            and summary_df[group_col].nunique() > 1
        )
    else:
        show_legend_used = bool(show_legend)

    # ------------------------------------------------------------
    # Plot using the shared professional line function
    # ------------------------------------------------------------
    if group_col is None:
        figure, axis = plot_professional_line(
            summary_df,
            x=frequency_col,
            y="mean_psd",
            lower="lower_psd",
            upper="upper_psd",

            color=color,
            linewidth=linewidth,
            fill_alpha=fill_alpha,

            title=title,
            xlabel=xlabel,
            ylabel=ylabel,

            xlim=xlim,
            ylim=ylim,
            xticks=xticks,
            yticks=yticks,

            figsize=figsize,
            font_size=font_size,

            show_legend=False,
            grid_axis="both",
            show=show,
        )

    else:
        figure, axis = plot_professional_line(
            summary_df,
            x=frequency_col,
            y="mean_psd",

            hue=group_col,
            hue_order=group_order_used,
            series_alias=group_alias,

            lower="lower_psd",
            upper="upper_psd",

            palette=palette,
            linewidth=linewidth,
            fill_alpha=fill_alpha,

            title=title,
            xlabel=xlabel,
            ylabel=ylabel,

            xlim=xlim,
            ylim=ylim,
            xticks=xticks,
            yticks=yticks,

            figsize=figsize,
            font_size=font_size,

            show_legend=show_legend_used,
            legend_title=legend_title,
            legend_loc=legend_loc,
            legend_ncol=legend_ncol,

            grid_axis="both",
            show=show,
        )

    return figure, axis, summary_df


# -----------------------------------------------------------------------------
# Create a reusable publication-ready bar plot with optional annotations.
def plot_professional_bar(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str | None = None,
    order: Sequence[Any] | None = None,
    hue_order: Sequence[Any] | None = None,

    # Display aliases
    category_alias: Mapping[Any, str] | None = None,
    hue_alias: Mapping[Any, str] | None = None,

    # Bar appearance
    palette: Mapping[Any, str] | Sequence[str] | str | None = None,
    color: str | None = None,
    estimator: str | Callable[..., float] = "mean",
    errorbar: str | tuple[str, float] | None = None,
    errorbar_capsize: float = 0.12,

    # Figure and typography
    figsize: tuple[float, float] = (9.0, 5.0),
    font_size: float = 12.0,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,

    # Axes
    x_tick_rotation: float = 0.0,
    ylim: tuple[float, float] | None = None,

    # Legend
    show_legend: bool = True,
    legend_loc: str = "best",
    legend_title: str | None = None,
    legend_ncol: int = 1,

    # Annotations
    annotate: bool = True,
    annotation_mode: Literal["value", "mean_sd"] = "value",
    annotate_decimals: int = 2,
    annotate_suffix: str = "",
    annotate_font_size: float | None = None,
    annotate_offset_fraction: float = 0.02,
    annotation_ddof: int = 1,

    # Optional reference line
    baseline: float | None = None,
    baseline_label: str | None = None,
    baseline_color: str = "#6B7280",
    baseline_linewidth: float = 1.5,
    baseline_linestyle: str = "--",

    # General styling
    bar_edgecolor: str = "#2F2F2F",
    bar_linewidth: float = 0.8,
    saturation: float = 0.95,
    grid: bool = True,
    show: bool = True,
) -> tuple[Any, Any]:
    """
    Create a publication- and presentation-ready bar plot.

    Annotation modes
    ----------------
    value
        Displays the plotted bar height.

    mean_sd
        Displays mean ± SD for each category or category/hue combination.
        This is intended for cohort-level summaries where each row represents
        one recording or subject.

    Notes
    -----
    - category_alias changes only visible x-axis labels.
    - hue_alias changes only visible legend labels.
    - Raw category names should still be used in order, hue_order, and palette.
    """
    # ============================================================
    # Validate input
    # ============================================================
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame. "
            f"Received {type(data).__name__}."
        )

    required_columns = {x, y}

    if hue is not None:
        required_columns.add(hue)

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise KeyError(
            "data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("data is empty.")

    if annotation_mode not in {"value", "mean_sd"}:
        raise ValueError(
            "annotation_mode must be either 'value' or 'mean_sd'."
        )

    if annotate_decimals < 0:
        raise ValueError(
            "annotate_decimals must be zero or greater."
        )

    if annotate_offset_fraction < 0:
        raise ValueError(
            "annotate_offset_fraction must be zero or greater."
        )

    if annotation_ddof < 0:
        raise ValueError(
            "annotation_ddof must be zero or greater."
        )

    if legend_ncol < 1:
        raise ValueError(
            "legend_ncol must be at least 1."
        )

    # Mean ± SD annotations should correspond to mean bars.
    if annotation_mode == "mean_sd":
        estimator_name = (
            estimator
            if isinstance(estimator, str)
            else getattr(estimator, "__name__", "")
        )

        if str(estimator_name).lower() not in {
            "mean",
            "nanmean",
        }:
            raise ValueError(
                "annotation_mode='mean_sd' requires a mean estimator."
            )

    # ============================================================
    # Clean plotting data
    # ============================================================
    plot_data = data.copy()

    plot_data[y] = pd.to_numeric(
        plot_data[y],
        errors="coerce",
    )

    drop_columns = [x, y]

    if hue is not None:
        drop_columns.append(hue)

    plot_data = plot_data.dropna(
        subset=drop_columns,
    )

    if plot_data.empty:
        raise ValueError(
            "No usable rows remain after removing missing or "
            "non-numeric plotting values."
        )

    # ============================================================
    # Resolve x-axis category order
    # ============================================================
    available_categories = (
        plot_data[x]
        .drop_duplicates()
        .tolist()
    )

    if order is None:
        order_used = available_categories
    else:
        order_used = list(order)

        missing_categories = [
            category
            for category in order_used
            if category not in set(available_categories)
        ]

        if missing_categories:
            raise ValueError(
                "The following values in order are not present "
                f"in data['{x}']: {missing_categories}"
            )

    # ============================================================
    # Resolve x-axis aliases
    # ============================================================
    if category_alias is None:
        category_alias = {}

    if not isinstance(category_alias, Mapping):
        raise TypeError(
            "category_alias must be a mapping."
        )

    display_labels = [
        str(category_alias.get(category, category))
        for category in order_used
    ]

    if len(set(display_labels)) != len(display_labels):
        raise ValueError(
            "category_alias creates duplicate displayed labels."
        )

    # ============================================================
    # Resolve hue ordering
    # ============================================================
    if hue is not None:
        available_hue_categories = (
            plot_data[hue]
            .drop_duplicates()
            .tolist()
        )

        if hue_order is None:
            hue_order_used = available_hue_categories
        else:
            hue_order_used = list(hue_order)

            missing_hue_categories = [
                category
                for category in hue_order_used
                if category not in set(available_hue_categories)
            ]

            if missing_hue_categories:
                raise ValueError(
                    "The following values in hue_order are not "
                    f"present in data['{hue}']: "
                    f"{missing_hue_categories}"
                )
    else:
        hue_order_used = None

    if hue_alias is None:
        hue_alias = {}

    if not isinstance(hue_alias, Mapping):
        raise TypeError(
            "hue_alias must be a mapping."
        )

    hue_alias_lookup = {
        str(raw_value): str(display_value)
        for raw_value, display_value in hue_alias.items()
    }

    # ============================================================
    # Validate dictionary palette
    # ============================================================
    if isinstance(palette, Mapping):
        expected_palette_keys = (
            hue_order_used
            if hue is not None
            else order_used
        )

        missing_palette_keys = [
            category
            for category in expected_palette_keys
            if category not in palette
        ]

        if missing_palette_keys:
            raise ValueError(
                "palette is missing colors for the following "
                f"raw category values: {missing_palette_keys}"
            )

    # ============================================================
    # Calculate statistics used for annotations
    # ============================================================
    annotation_group_columns = [x]

    if hue is not None:
        annotation_group_columns.append(hue)

    annotation_summary = (
        plot_data
        .groupby(
            annotation_group_columns,
            observed=True,
            sort=False,
        )[y]
        .agg(
            mean="mean",
            sd=lambda values: np.std(
                values,
                ddof=annotation_ddof,
            ),
            n="count",
        )
        .reset_index()
    )

    annotation_summary["sd"] = (
        annotation_summary["sd"]
        .fillna(0.0)
    )

    if hue is None:
        annotation_stats = {
            row[x]: {
                "mean": float(row["mean"]),
                "sd": float(row["sd"]),
                "n": int(row["n"]),
            }
            for _, row in annotation_summary.iterrows()
        }

    else:
        annotation_stats = {
            (row[x], row[hue]): {
                "mean": float(row["mean"]),
                "sd": float(row["sd"]),
                "n": int(row["n"]),
            }
            for _, row in annotation_summary.iterrows()
        }

    # ============================================================
    # Create figure
    # ============================================================
    sns.set_theme(
        style="whitegrid",
        context="notebook",
    )

    figure, axis = plt.subplots(
        figsize=figsize,
    )

    barplot_kwargs: dict[str, Any] = {
        "data": plot_data,
        "x": x,
        "y": y,
        "order": order_used,
        "estimator": estimator,
        "errorbar": errorbar,
        "saturation": saturation,
        "ax": axis,
    }

    if errorbar is not None:
        barplot_kwargs["capsize"] = errorbar_capsize

    if hue is not None:
        barplot_kwargs.update({
            "hue": hue,
            "hue_order": hue_order_used,
            "palette": palette,
            "dodge": True,
        })

    elif palette is not None:
        # Assign one configured color to each x category.
        barplot_kwargs.update({
            "hue": x,
            "hue_order": order_used,
            "palette": palette,
            "dodge": False,
            "legend": False,
        })

    else:
        barplot_kwargs["color"] = (
            color or "#2F6B9A"
        )

    sns.barplot(
        **barplot_kwargs,
    )

    # ============================================================
    # Apply x-axis display aliases
    # ============================================================
    axis.set_xticks(
        range(len(order_used))
    )

    axis.set_xticklabels(
        display_labels,
        rotation=x_tick_rotation,
        ha="right" if x_tick_rotation else "center",
    )

    # ============================================================
    # Style bars
    # ============================================================
    for bar in axis.patches:
        bar.set_edgecolor(bar_edgecolor)
        bar.set_linewidth(bar_linewidth)

    # ============================================================
    # Optional baseline
    # ============================================================
    if baseline is not None:
        axis.axhline(
            float(baseline),
            color=baseline_color,
            linewidth=baseline_linewidth,
            linestyle=baseline_linestyle,
            label=baseline_label,
            zorder=1,
        )

    # ============================================================
    # Titles and labels
    # ============================================================
    axis.set_title(
        title or "",
        fontsize=font_size + 2,
        fontweight="bold",
        pad=14,
    )

    axis.set_xlabel(
        xlabel if xlabel is not None else x,
        fontsize=font_size,
        fontweight="bold",
        labelpad=8,
    )

    axis.set_ylabel(
        ylabel if ylabel is not None else y,
        fontsize=font_size,
        fontweight="bold",
        labelpad=8,
    )

    axis.tick_params(
        axis="both",
        labelsize=font_size,
    )

    for label in axis.get_xticklabels():
        label.set_fontweight("bold")

    for label in axis.get_yticklabels():
        label.set_fontweight("bold")

    # ============================================================
    # Grid and spines
    # ============================================================
    axis.set_axisbelow(True)

    if grid:
        axis.grid(
            axis="y",
            linestyle="--",
            linewidth=0.8,
            alpha=0.28,
        )
    else:
        axis.grid(False)

    axis.grid(
        axis="x",
        visible=False,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.0)
    axis.spines["bottom"].set_linewidth(1.0)

    # ============================================================
    # Legend with aliases
    # ============================================================
    handles, labels = axis.get_legend_handles_labels()

    renamed_labels = [
        hue_alias_lookup.get(label, label)
        for label in labels
    ]

    unique_handles = []
    unique_labels = []
    seen_labels = set()

    for handle, label in zip(
        handles,
        renamed_labels,
    ):
        if not label or label.startswith("_"):
            continue

        if label in seen_labels:
            continue

        seen_labels.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)

    should_show_legend = (
        show_legend
        and len(unique_handles) > 0
        and (
            hue is not None
            or baseline_label is not None
        )
    )

    if should_show_legend:
        legend = axis.legend(
            unique_handles,
            unique_labels,
            title=legend_title,
            loc=legend_loc,
            ncol=legend_ncol,
            frameon=True,
            prop={
                "size": font_size,
                "weight": "bold",
            },
        )

        if legend is not None:
            legend.get_frame().set_alpha(0.95)

            if legend.get_title() is not None:
                legend.get_title().set_fontsize(
                    font_size
                )
                legend.get_title().set_fontweight(
                    "bold"
                )

    else:
        existing_legend = axis.get_legend()

        if existing_legend is not None:
            existing_legend.remove()

    # ============================================================
    # Apply explicit y-axis limits
    # ============================================================
    if ylim is not None:
        axis.set_ylim(*ylim)

    # ============================================================
    # Add annotations
    # ============================================================
    if annotate:
        annotation_font_size_used = (
            annotate_font_size
            if annotate_font_size is not None
            else max(8.0, font_size - 2.0)
        )

        current_ymin, current_ymax = axis.get_ylim()
        current_span = current_ymax - current_ymin

        if current_span <= 0:
            current_span = 1.0

        annotation_offset = (
            current_span
            * annotate_offset_fraction
        )

        annotation_positions = []
        plotted_values = []

        # --------------------------------------------------------
        # No explicit hue: one visible bar per x category
        # --------------------------------------------------------
        if hue is None:
            visible_bars = [
                bar
                for bar in axis.patches
                if (
                    np.isfinite(bar.get_height())
                    and bar.get_width() > 0
                )
            ]

            visible_bars = sorted(
                visible_bars,
                key=lambda bar: (
                    bar.get_x()
                    + bar.get_width() / 2.0
                ),
            )

            if len(visible_bars) < len(order_used):
                raise RuntimeError(
                    "Could not align plotted bars with x-axis categories."
                )

            for category, bar in zip(
                order_used,
                visible_bars[:len(order_used)],
            ):
                stats = annotation_stats[category]

                mean_value = stats["mean"]
                sd_value = stats["sd"]

                if annotation_mode == "mean_sd":
                    text = (
                        f"{mean_value:.{annotate_decimals}f} ± "
                        f"{sd_value:.{annotate_decimals}f}"
                        f"{annotate_suffix}"
                    )

                    if mean_value >= 0:
                        anchor_value = mean_value + sd_value
                        y_position = (
                            anchor_value
                            + annotation_offset
                        )
                        vertical_alignment = "bottom"
                    else:
                        anchor_value = mean_value - sd_value
                        y_position = (
                            anchor_value
                            - annotation_offset
                        )
                        vertical_alignment = "top"

                else:
                    bar_value = float(
                        bar.get_height()
                    )

                    text = (
                        f"{bar_value:.{annotate_decimals}f}"
                        f"{annotate_suffix}"
                    )

                    if bar_value >= 0:
                        y_position = (
                            bar_value
                            + annotation_offset
                        )
                        vertical_alignment = "bottom"
                    else:
                        y_position = (
                            bar_value
                            - annotation_offset
                        )
                        vertical_alignment = "top"

                x_position = (
                    bar.get_x()
                    + bar.get_width() / 2.0
                )

                axis.text(
                    x_position,
                    y_position,
                    text,
                    ha="center",
                    va=vertical_alignment,
                    fontsize=annotation_font_size_used,
                    fontweight="bold",
                )

                annotation_positions.append(
                    y_position
                )

                plotted_values.append(
                    mean_value
                )

        # --------------------------------------------------------
        # Explicit hue: one BarContainer per hue group
        # --------------------------------------------------------
        else:
            bar_containers = [
                container
                for container in axis.containers
                if isinstance(container, BarContainer)
            ]

            if len(bar_containers) < len(hue_order_used):
                raise RuntimeError(
                    "Could not align plotted bars with hue categories."
                )

            for hue_value, container in zip(
                hue_order_used,
                bar_containers[:len(hue_order_used)],
            ):
                for category, bar in zip(
                    order_used,
                    container,
                ):
                    stats_key = (
                        category,
                        hue_value,
                    )

                    if stats_key not in annotation_stats:
                        continue

                    bar_height = float(
                        bar.get_height()
                    )

                    if not np.isfinite(bar_height):
                        continue

                    stats = annotation_stats[
                        stats_key
                    ]

                    mean_value = stats["mean"]
                    sd_value = stats["sd"]

                    if annotation_mode == "mean_sd":
                        text = (
                            f"{mean_value:.{annotate_decimals}f} ± "
                            f"{sd_value:.{annotate_decimals}f}"
                            f"{annotate_suffix}"
                        )

                        if mean_value >= 0:
                            anchor_value = (
                                mean_value
                                + sd_value
                            )

                            y_position = (
                                anchor_value
                                + annotation_offset
                            )

                            vertical_alignment = "bottom"

                        else:
                            anchor_value = (
                                mean_value
                                - sd_value
                            )

                            y_position = (
                                anchor_value
                                - annotation_offset
                            )

                            vertical_alignment = "top"

                    else:
                        text = (
                            f"{bar_height:.{annotate_decimals}f}"
                            f"{annotate_suffix}"
                        )

                        if bar_height >= 0:
                            y_position = (
                                bar_height
                                + annotation_offset
                            )

                            vertical_alignment = "bottom"

                        else:
                            y_position = (
                                bar_height
                                - annotation_offset
                            )

                            vertical_alignment = "top"

                    x_position = (
                        bar.get_x()
                        + bar.get_width() / 2.0
                    )

                    axis.text(
                        x_position,
                        y_position,
                        text,
                        ha="center",
                        va=vertical_alignment,
                        fontsize=annotation_font_size_used,
                        fontweight="bold",
                    )

                    annotation_positions.append(
                        y_position
                    )

                    plotted_values.append(
                        mean_value
                    )

        # --------------------------------------------------------
        # Expand automatically selected limits
        # --------------------------------------------------------
        if (
            ylim is None
            and annotation_positions
        ):
            plotted_ymin, plotted_ymax = (
                axis.get_ylim()
            )

            span = (
                plotted_ymax
                - plotted_ymin
            )

            if span <= 0:
                span = 1.0

            new_ymax = max(
                plotted_ymax,
                max(annotation_positions)
                + 0.08 * span,
            )

            new_ymin = min(
                plotted_ymin,
                min(annotation_positions)
                - 0.04 * span,
            )

            if (
                plotted_values
                and min(plotted_values) >= 0
            ):
                new_ymin = 0.0

            axis.set_ylim(
                new_ymin,
                new_ymax,
            )

    axis.margins(
        x=0.04,
    )

    figure.tight_layout()

    if show:
        plt.show()

    return figure, axis
    

# =============================================================================
# BATCH AGGREGATION AND SUMMARY HELPERS
# =============================================================================

# Combine recording-level qEEG metrics into tidy batch-level tables.
def build_qeeg_batch_metric_tables(
    qeeg_results_by_recording: Mapping[str, Mapping[str, Any]],
    *,
    metadata_fields: Sequence[str] = DEFAULT_QEEG_METADATA_FIELDS,
) -> dict[str, pd.DataFrame]:
    """
    Combine recording-level qEEG metrics into tidy batch-level tables.

    Metadata propagation is handled internally so notebook code does not need
    to map ``source_recording_id``, condition, eye state, timepoint, or other
    study fields back into the tables after they are built.

    Returns
    -------
    dict
        ``absolute_power_df``, ``relative_power_df``, and
        ``spectral_ratio_df``.
    """
    if not isinstance(qeeg_results_by_recording, Mapping):
        raise TypeError("qeeg_results_by_recording must be a mapping.")
    if not qeeg_results_by_recording:
        raise ValueError("qeeg_results_by_recording is empty.")

    metadata_fields_used = tuple(dict.fromkeys(str(x) for x in metadata_fields))
    absolute_tables: list[pd.DataFrame] = []
    relative_tables: list[pd.DataFrame] = []
    ratio_tables: list[pd.DataFrame] = []

    for recording_id, result in qeeg_results_by_recording.items():
        required_result_keys = {
            "absolute_power_result",
            "relative_power_result",
            "spectral_ratio_result",
        }
        missing_result_keys = required_result_keys - set(result.keys())
        if missing_result_keys:
            raise KeyError(
                f"Recording '{recording_id}' is missing: "
                f"{sorted(missing_result_keys)}"
            )

        metadata_values = _extract_qeeg_metadata(
            recording_id,
            result,
            metadata_fields_used,
        )

        absolute_tables.append(
            result["absolute_power_result"]["overall_band_power_df"]
            .copy()
            .assign(**metadata_values)
        )
        relative_tables.append(
            result["relative_power_result"]["overall_relative_power_df"]
            .copy()
            .assign(**metadata_values)
        )
        ratio_tables.append(
            result["spectral_ratio_result"]["overall_ratio_df"]
            .copy()
            .assign(**metadata_values)
        )

    return {
        "absolute_power_df": pd.concat(absolute_tables, ignore_index=True),
        "relative_power_df": pd.concat(relative_tables, ignore_index=True),
        "spectral_ratio_df": pd.concat(ratio_tables, ignore_index=True),
    }


# -----------------------------------------------------------------------------
# Average channel-by-band power values across selected recordings.
def aggregate_band_power_for_group_topomap(
    qeeg_results_by_recording: Mapping[str, Mapping[str, Any]],
    *,
    power_type: Literal["absolute", "relative"],
    recording_ids: Sequence[str] | None = None,
    ddof: int = 1,
) -> dict[str, Any]:
    """
    Aggregate channel-level band power across selected recordings.

    The function averages numerical channel-by-band values; rendered topographic
    images are never averaged. When the selected group contains too few
    recordings for the requested sample SD, the returned SD array is NaN rather
    than zero because the sample SD is undefined.
    """
    if not isinstance(qeeg_results_by_recording, Mapping):
        raise TypeError("qeeg_results_by_recording must be a mapping.")
    if not qeeg_results_by_recording:
        raise ValueError("qeeg_results_by_recording is empty.")
    if ddof < 0:
        raise ValueError("ddof must be zero or greater.")

    if power_type == "absolute":
        result_key = "absolute_power_result"
        value_key = "mean_absolute_power_by_channel_uv2"
        sd_key = "std_absolute_power_by_channel_uv2"
    elif power_type == "relative":
        result_key = "relative_power_result"
        value_key = "mean_relative_power_by_channel_percent"
        sd_key = "std_relative_power_by_channel_percent"
    else:
        raise ValueError("power_type must be 'absolute' or 'relative'.")

    if recording_ids is None:
        selected_recording_ids = list(qeeg_results_by_recording.keys())
    else:
        selected_recording_ids = [str(value) for value in recording_ids]
        missing_recording_ids = [
            recording_id
            for recording_id in selected_recording_ids
            if recording_id not in qeeg_results_by_recording
        ]
        if missing_recording_ids:
            raise KeyError(
                "The following recording IDs are unavailable: "
                f"{missing_recording_ids}"
            )

    if not selected_recording_ids:
        raise ValueError("No recordings were selected.")

    stacked_values: list[np.ndarray] = []
    reference_band_names: list[str] | None = None
    reference_ch_names: list[str] | None = None

    for recording_id in selected_recording_ids:
        recording_result = qeeg_results_by_recording[recording_id]
        if result_key not in recording_result:
            raise KeyError(
                f"Recording '{recording_id}' is missing '{result_key}'."
            )

        band_result = recording_result[result_key]
        required_keys = {value_key, "band_names", "ch_names"}
        missing_keys = required_keys - set(band_result.keys())
        if missing_keys:
            raise KeyError(
                f"Recording '{recording_id}' is missing band-power keys: "
                f"{sorted(missing_keys)}"
            )

        current_values = np.asarray(band_result[value_key], dtype=float)
        current_band_names = list(band_result["band_names"])
        current_ch_names = list(band_result["ch_names"])

        if reference_band_names is None:
            reference_band_names = current_band_names
            reference_ch_names = current_ch_names
        else:
            if current_band_names != reference_band_names:
                raise ValueError(
                    "Band names or band ordering differ for recording "
                    f"'{recording_id}'."
                )
            if current_ch_names != reference_ch_names:
                raise ValueError(
                    "Channel names or channel ordering differ for recording "
                    f"'{recording_id}'."
                )

        expected_shape = (len(current_ch_names), len(current_band_names))
        if current_values.shape != expected_shape:
            raise ValueError(
                f"Unexpected channel-by-band shape for recording '{recording_id}'. "
                f"Expected {expected_shape}, received {current_values.shape}."
            )
        if not np.isfinite(current_values).all():
            raise ValueError(
                f"Recording '{recording_id}' contains NaN or infinite "
                "topographic values."
            )

        stacked_values.append(current_values)

    stacked_values_array = np.stack(stacked_values, axis=0)
    mean_values = np.mean(stacked_values_array, axis=0)

    if len(selected_recording_ids) > ddof:
        std_values = np.std(stacked_values_array, axis=0, ddof=ddof)
    else:
        std_values = np.full_like(mean_values, np.nan, dtype=float)

    return {
        value_key: mean_values,
        sd_key: std_values,
        "band_names": reference_band_names,
        "ch_names": reference_ch_names,
        "n_recordings": int(len(selected_recording_ids)),
        "recording_ids": selected_recording_ids,
        "stacked_values_by_recording": stacked_values_array,
    }


# -----------------------------------------------------------------------------
# Combine recording-level PSD curves and metadata into one tidy table.
def build_combined_psd_df(
    qeeg_results_by_recording: Mapping[str, Mapping[str, Any]],
    *,
    metadata_fields: Sequence[str] = DEFAULT_QEEG_METADATA_FIELDS,
) -> pd.DataFrame:
    """
    Combine recording-level mean PSD curves into one tidy DataFrame.

    All requested metadata are appended inside this helper, eliminating the
    notebook-level metadata patching previously required for EO/EC, timepoint,
    source-recording ID, and other study fields.
    """
    if not qeeg_results_by_recording:
        raise ValueError("qeeg_results_by_recording is empty.")

    metadata_fields_used = tuple(dict.fromkeys(str(x) for x in metadata_fields))
    psd_tables: list[pd.DataFrame] = []

    for recording_id, result in qeeg_results_by_recording.items():
        if "mean_psd_result" not in result:
            raise KeyError(
                f"Recording '{recording_id}' is missing 'mean_psd_result'."
            )
        psd_df = result["mean_psd_result"]["overall_psd_df"].copy()
        metadata_values = _extract_qeeg_metadata(
            recording_id,
            result,
            metadata_fields_used,
        )
        for field, value in metadata_values.items():
            psd_df[field] = value
        psd_tables.append(psd_df)

    return pd.concat(psd_tables, ignore_index=True)


# -----------------------------------------------------------------------------
# Select an Epochs object that supplies channel locations for group topomaps.
def get_topomap_template_epochs(
    qeeg_results_by_recording: Mapping[str, Mapping[str, Any]],
    recording_id: str | None = None,
    *,
    validate_channel_compatibility: bool = True,
) -> mne.BaseEpochs:
    """
    Return an Epochs object that supplies sensor locations for group topomaps.

    When ``validate_channel_compatibility`` is True, all analyzed recordings are
    checked against the selected template's qEEG channel names and ordering.
    """
    if not qeeg_results_by_recording:
        raise ValueError("qeeg_results_by_recording is empty.")

    selected_id = recording_id or next(iter(qeeg_results_by_recording))
    if selected_id not in qeeg_results_by_recording:
        raise KeyError(f"Unknown recording_id: {selected_id}")

    selected_result = qeeg_results_by_recording[selected_id]
    epochs = selected_result.get("epochs_clean")
    if not isinstance(epochs, mne.BaseEpochs):
        raise TypeError(
            f"epochs_clean for '{selected_id}' must be an MNE Epochs object."
        )

    if validate_channel_compatibility:
        reference_ch_names = list(
            selected_result.get(
                "qeeg_ch_names",
                selected_result["mean_psd_result"]["ch_names"],
            )
        )
        for current_id, result in qeeg_results_by_recording.items():
            current_ch_names = list(
                result.get(
                    "qeeg_ch_names",
                    result["mean_psd_result"]["ch_names"],
                )
            )
            if current_ch_names != reference_ch_names:
                raise ValueError(
                    "qEEG channel names/order differ between topomap template "
                    f"'{selected_id}' and recording '{current_id}'."
                )

    return epochs




# -----------------------------------------------------------------------------
# Create one core analysis-summary row per completed logical recording.
def build_qeeg_recording_summary_df(
    qeeg_results_by_recording: Mapping[str, Mapping[str, Any]],
    *,
    metadata_fields: Sequence[str] = DEFAULT_QEEG_METADATA_FIELDS,
) -> pd.DataFrame:
    """
    Create one analysis-summary row per completed logical recording.

    The summary is now the authoritative recording-level table for Part 1 qEEG.
    It preserves condition-aware study metadata and explicitly distinguishes all
    retained channels from the channels that actually entered qEEG.
    """
    if not qeeg_results_by_recording:
        raise ValueError("qeeg_results_by_recording is empty.")

    metadata_fields_used = tuple(dict.fromkeys(str(x) for x in metadata_fields))
    summary_rows: list[dict[str, Any]] = []

    for recording_id, result in qeeg_results_by_recording.items():
        n_channels_total = int(
            result.get("n_channels_total", result.get("n_channels", 0))
        )
        n_channels_qeeg = int(
            result.get(
                "n_channels_qeeg",
                len(result["mean_psd_result"]["ch_names"]),
            )
        )

        row = _extract_qeeg_metadata(
            recording_id,
            result,
            metadata_fields_used,
        )
        row.update({
            "n_epochs_clean": int(result["n_epochs_clean"]),
            "n_channels_total": n_channels_total,
            "n_channels_qeeg": n_channels_qeeg,
            "qeeg_ch_names": list(
                result.get(
                    "qeeg_ch_names",
                    result["mean_psd_result"]["ch_names"],
                )
            ),
            "sfreq_hz": float(result["sfreq_hz"]),
            "n_samples_per_epoch": int(result["n_samples_per_epoch"]),
        })
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)

# =============================================================================
# PART 1 qEEG PIPELINE ORCHESTRATION
# =============================================================================
# These functions remove notebook-level glue code while preserving reusable
# low-level calculation and plotting helpers above.


def build_qeeg_analysis_groups(
    qeeg_results_by_recording: Mapping[str, Mapping[str, Any]],
    *,
    group_columns: Sequence[str] | str = DEFAULT_QEEG_GROUP_COLUMNS,
    drop_missing_groups: bool = False,
) -> dict[tuple[Any, ...], list[str]]:
    """
    Group logical recording IDs by condition/timepoint or other metadata.

    Group keys are always tuples, even when only one grouping column is used.
    """
    if not qeeg_results_by_recording:
        raise ValueError("qeeg_results_by_recording is empty.")

    group_columns_used = _normalize_group_columns(group_columns)
    groups: dict[tuple[Any, ...], list[str]] = {}

    for recording_id, result in qeeg_results_by_recording.items():
        group_key = tuple(result.get(column) for column in group_columns_used)
        if drop_missing_groups and any(value is None for value in group_key):
            continue
        groups.setdefault(group_key, []).append(str(recording_id))

    if not groups:
        raise ValueError("No qEEG analysis groups were created.")

    return groups


# -----------------------------------------------------------------------------
# Prepare absolute and relative channel-level inputs for every analysis group.
def build_grouped_topomap_inputs(
    qeeg_results_by_recording: Mapping[str, Mapping[str, Any]],
    recording_ids_by_group: Mapping[tuple[Any, ...], Sequence[str]],
    *,
    ddof: int = 1,
) -> dict[str, dict[tuple[Any, ...], dict[str, Any]]]:
    """Build grouped numerical topomap inputs for absolute and relative power."""
    if not recording_ids_by_group:
        raise ValueError("recording_ids_by_group is empty.")

    absolute_by_group: dict[tuple[Any, ...], dict[str, Any]] = {}
    relative_by_group: dict[tuple[Any, ...], dict[str, Any]] = {}

    for group_key, recording_ids in recording_ids_by_group.items():
        absolute_by_group[group_key] = aggregate_band_power_for_group_topomap(
            qeeg_results_by_recording,
            power_type="absolute",
            recording_ids=recording_ids,
            ddof=ddof,
        )
        relative_by_group[group_key] = aggregate_band_power_for_group_topomap(
            qeeg_results_by_recording,
            power_type="relative",
            recording_ids=recording_ids,
            ddof=ddof,
        )

    return {
        "absolute": absolute_by_group,
        "relative": relative_by_group,
    }


# -----------------------------------------------------------------------------
# Convert qEEG analysis outputs into one standardized, plot-ready result object.
def prepare_qeeg_part1_results(
    qeeg_results_by_recording: Mapping[str, Mapping[str, Any]],
    *,
    qc_records: Sequence[Mapping[str, Any]] | None = None,
    group_columns: Sequence[str] | str = DEFAULT_QEEG_GROUP_COLUMNS,
    metadata_fields: Sequence[str] = DEFAULT_QEEG_METADATA_FIELDS,
    topomap_ddof: int = 1,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Prepare all Part 1 qEEG tables, groups, QC, and topomap inputs.

    This is the preparation layer that replaces notebook-level DataFrame
    construction, metadata reinjection, manual grouping, and topomap
    comprehensions.
    """
    if not qeeg_results_by_recording:
        raise ValueError("qeeg_results_by_recording is empty.")

    group_columns_used = _normalize_group_columns(group_columns)
    metadata_fields_used = tuple(dict.fromkeys(str(x) for x in metadata_fields))

    preprocessing_qc = None
    if qc_records is not None:
        preprocessing_qc = build_preprocessing_qc_summary(
            qc_records,
            group_columns=group_columns_used,
            verbose=verbose,
        )

    recording_summary_df = build_qeeg_recording_summary_df(
        qeeg_results_by_recording,
        metadata_fields=metadata_fields_used,
    )

    channel_qc_columns = [
        column
        for column in (
            "recording_id",
            "source_recording_id",
            "subject_id",
            "analysis_condition",
            "eye_state",
            "timepoint",
            "n_epochs_clean",
            "n_channels_total",
            "n_channels_qeeg",
            "qeeg_ch_names",
        )
        if column in recording_summary_df.columns
    ]
    channel_qc_df = recording_summary_df[channel_qc_columns].copy()

    combined_psd_df = build_combined_psd_df(
        qeeg_results_by_recording,
        metadata_fields=metadata_fields_used,
    )
    metric_tables = build_qeeg_batch_metric_tables(
        qeeg_results_by_recording,
        metadata_fields=metadata_fields_used,
    )

    recording_ids_by_group = build_qeeg_analysis_groups(
        qeeg_results_by_recording,
        group_columns=group_columns_used,
    )
    grouped_topomaps = build_grouped_topomap_inputs(
        qeeg_results_by_recording,
        recording_ids_by_group,
        ddof=topomap_ddof,
    )
    topomap_template_epochs = get_topomap_template_epochs(
        qeeg_results_by_recording,
        validate_channel_compatibility=True,
    )

    prepared = {
        "results_by_recording": qeeg_results_by_recording,
        "qc": preprocessing_qc,
        "recording_summary_df": recording_summary_df,
        "channel_qc_df": channel_qc_df,
        "tables": {
            "combined_psd_df": combined_psd_df,
            **metric_tables,
        },
        "groups": {
            "columns": group_columns_used,
            "recording_ids_by_group": recording_ids_by_group,
        },
        "topomaps": {
            "template_epochs": topomap_template_epochs,
            "absolute_by_group": grouped_topomaps["absolute"],
            "relative_by_group": grouped_topomaps["relative"],
        },
    }

    if verbose:
        print("\nPrepared Part 1 qEEG results")
        print("-" * 52)
        print(f"Logical recordings: {len(qeeg_results_by_recording)}")
        print(f"Analysis groups:    {len(recording_ids_by_group)}")
        print(f"Group columns:      {', '.join(group_columns_used)}")
        print("-" * 52)

    return prepared


# -----------------------------------------------------------------------------
# User-facing Part 1 qEEG pipeline: build logical records, analyze, and prepare.
def run_qeeg_part1_pipeline(
    label_epoch_arrays: Mapping[str, Sequence[mne.BaseEpochs | Mapping[str, mne.BaseEpochs]]],
    metadata: Sequence[Mapping[str, Any]],
    qc_records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any] | None = None,
    inspect_n: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the complete Part 1 qEEG workflow and return one prepared result object.

    The notebook-level workflow becomes:

        qeeg_results = run_qeeg_part1_pipeline(...)

    followed by:

        qeeg_plot_data = prepare_qeeg_plot_data(
            qeeg_results,
            timepoint="H1",
        )

    Plotting functions are then called explicitly so figure colors, fonts,
    titles, sizes, scales, and other aesthetics remain easy to edit manually.

    Parameters
    ----------
    label_epoch_arrays, metadata, qc_records
        Outputs from the EEG preprocessing/batch-preparation workflow.

    config
        Optional overrides applied to ``get_default_qeeg_part1_config()``.

    inspect_n
        Optional number of logical recordings to print for development QC.
        Zero disables inspection.

    verbose
        Controls pipeline-level reporting. Scientific calculations are unchanged.
    """
    cfg = _merge_nested_config(get_default_qeeg_part1_config(), config)

    recordings = build_recordings_from_epochs(
        label_epoch_arrays,
        metadata,
        label=cfg["labels"],
        condition_to_eye_state=cfg["condition_to_eye_state"],
    )

    if inspect_n < 0:
        raise ValueError("inspect_n must be zero or greater.")
    if inspect_n > 0:
        inspect_recordings(
            recordings,
            n=min(inspect_n, len(recordings)),
            picks=cfg["picks"],
            verbose=verbose,
        )

    analysis_log_mode = cfg["log_mode"] if verbose else "silent"

    qeeg_results_by_recording = run_qeeg_batch_analysis(
        recordings,
        bands=cfg["bands"],
        ratio_definitions=cfg["ratio_definitions"],
        psd_range_hz=tuple(cfg["psd_range_hz"]),
        total_range_hz=tuple(cfg["total_range_hz"]),
        relative_power_bands=cfg.get("relative_power_bands"),
        picks=cfg["picks"],
        ratio_summary_method=cfg["ratio_summary_method"],
        psd_kwargs=cfg.get("psd_kwargs"),
        log_mode=analysis_log_mode,
        progress_every=int(cfg["progress_every"]),
    )

    prepared = prepare_qeeg_part1_results(
        qeeg_results_by_recording,
        qc_records=qc_records,
        group_columns=cfg["group_columns"],
        metadata_fields=cfg["metadata_fields"],
        topomap_ddof=int(cfg["topomap_ddof"]),
        verbose=verbose,
    )

    prepared["recordings"] = recordings
    prepared["config"] = cfg
    # prepared["module_version"] = __version__
    # prepared["pipeline_milestone"] = PIPELINE_MILESTONE

    if verbose:
        eye_states = [
            value
            for value in recording_summary_values(prepared, "eye_state")
            if value is not None
        ]
        timepoints = [
            value
            for value in recording_summary_values(prepared, "timepoint")
            if value is not None
        ]
        print("\nPart 1 qEEG pipeline complete")
        print("=" * 52)
        print(f"Logical recordings: {len(recordings)}")
        print(f"Subjects:           {prepared['recording_summary_df']['subject_id'].nunique(dropna=True) if 'subject_id' in prepared['recording_summary_df'] else 'N/A'}")
        print(f"Eye states:         {', '.join(map(str, eye_states)) if eye_states else 'N/A'}")
        print(f"Timepoints:         {', '.join(map(str, timepoints)) if timepoints else 'N/A'}")
        print("=" * 52)

    return prepared


# -----------------------------------------------------------------------------
# Small helper used only for concise pipeline reporting.
def recording_summary_values(
    qeeg_results: Mapping[str, Any],
    column: str,
) -> list[Any]:
    """Return unique non-duplicated values from the recording summary table."""
    summary_df = qeeg_results.get("recording_summary_df")
    if not isinstance(summary_df, pd.DataFrame) or column not in summary_df.columns:
        return []
    return summary_df[column].drop_duplicates().tolist()



# =============================================================================
# PART 1 qEEG QC ORCHESTRATION
# =============================================================================
def run_qeeg_part1_qc(
    qeeg_results: Mapping[str, Any],
    *,
    group_columns: Sequence[str] | str | None = None,
    minimum_clean_minutes: float | None = None,
    physical_group_col: str | None = None,
    posterior_alpha: Mapping[str, Any] | None = None,
    high_frequency: Mapping[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the complete Part 1 QC workflow from an existing qEEG result object.

    Whole-recording and condition-aware datasets use the same workflow.
    Grouping comes from group_columns; physical-recording QC automatically
    excludes grouping dimensions that vary within one physical recording.

    QC includes:
      1. Preprocessing / recording-quality QC
      2. Bad-channel detector + downstream artifact provenance
      3. Physical-recording ICA / EOG QC
      4. Optional posterior-alpha physiological QC
      5. Optional frontal/frontotemporal high-frequency QC

    No preprocessing or Part 1 qEEG spectral calculations are repeated.
    """
    if not isinstance(qeeg_results, Mapping):
        raise TypeError("qeeg_results must be returned by run_qeeg_part1_pipeline().")

    missing_sections = {"results_by_recording", "qc"} - set(qeeg_results)
    if missing_sections:
        raise KeyError(f"qeeg_results is missing required sections: {sorted(missing_sections)}")

    results_by_recording, qc_source = qeeg_results["results_by_recording"], qeeg_results["qc"]

    if not isinstance(results_by_recording, Mapping) or not results_by_recording:
        raise ValueError("qeeg_results['results_by_recording'] is empty or invalid.")
    if not isinstance(qc_source, Mapping) or "recording_qc_df" not in qc_source:
        raise KeyError("qeeg_results['qc'] must contain 'recording_qc_df'.")

    recording_qc_source_df = qc_source["recording_qc_df"]
    if not isinstance(recording_qc_source_df, pd.DataFrame) or recording_qc_source_df.empty:
        raise ValueError("qeeg_results['qc']['recording_qc_df'] is empty or invalid.")

    # -------------------------------------------------------------------------
    # Resolve logical-recording grouping and propagated metadata.
    # -------------------------------------------------------------------------
    if group_columns is None:
        group_columns = qeeg_results.get("groups", {}).get("columns")
    if group_columns is None:
        group_columns = qeeg_results.get("config", {}).get("group_columns", DEFAULT_QEEG_GROUP_COLUMNS)

    group_columns_used = _normalize_group_columns(group_columns)
    metadata_fields = tuple(
        qeeg_results.get("config", {}).get("metadata_fields", DEFAULT_QEEG_METADATA_FIELDS)
    )

    # Retained for shared reporting helpers that may need a fallback overall
    # grouping when requested grouping dimensions are unavailable.
    def _resolve_grouping(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        data = data.copy()
        columns = [
            column for column in group_columns_used
            if column in data.columns and data[column].notna().any()
        ]
        if not columns:
            data["_report_group"] = "Overall"
            columns = ["_report_group"]
        return data, columns

    # =========================================================================
    # 1. PREPROCESSING / RECORDING-QUALITY QC
    # =========================================================================
    preprocessing_qc = build_preprocessing_qc_summary(
        recording_qc_source_df.to_dict("records"),
        group_columns=group_columns_used,
        verbose=False,
    )

    if minimum_clean_minutes is not None:
        minimum_clean_minutes = float(minimum_clean_minutes)
        if minimum_clean_minutes <= 0:
            raise ValueError("minimum_clean_minutes must be greater than zero or None.")

    completeness_qc = summarize_qc_completeness(
        preprocessing_qc,
        minimum_clean_minutes=minimum_clean_minutes,
        group_columns=group_columns_used,
    )

    preprocessing_results = {
        "recording_df": preprocessing_qc["recording_qc_df"],
        "metric_summary_df": preprocessing_qc["qc_metric_summary_df"],
        "completeness": completeness_qc,
    }

    recording_qc_df = preprocessing_results["recording_df"]
    physical_recording_col = (
        "source_recording_id"
        if "source_recording_id" in recording_qc_df.columns
        else "recording_id"
    )

    # -------------------------------------------------------------------------
    # Bad-channel detector + artifact provenance QC
    # -------------------------------------------------------------------------
    # For EVERY bad/interpolated channel, preserve:
    #   - whether it was flagged by MAD, RANSAC, or both
    #   - downstream EOG-supported ocular-component evidence
    #   - downstream ICLabel eye-blink / muscle-artifact evidence
    #   - overlap between EOG-supported ICs and actually excluded ICA components
    #
    # IMPORTANT:
    # ICA/EOG occurs AFTER bad-channel interpolation. Artifact evidence therefore
    # describes association within the same physical recording and does not prove
    # that ocular/muscle activity caused the earlier bad-channel rejection.
    #
    # The calculation is performed once per physical EEG recording so EO/EC
    # logical records originating from the same file are not counted twice.
    bad_channel_method_qc = build_bad_channel_method_qc(recording_qc_df,physical_recording_col=physical_recording_col )

    # Store the general detector/artifact provenance tables with preprocessing
    # QC for recurrence summaries, targeted investigation, reporting, and slides.
    preprocessing_results["bad_channel_methods"] = bad_channel_method_qc

    preprocessing_results["bad_channels"] = prepare_bad_channel_qc(bad_channel_method_qc)

    # =========================================================================
    # 2. PHYSICAL-RECORDING ICA / EOG QC
    # =========================================================================
    # Infer physical-level dimensions instead of hard-coding eye_state.
    # Any grouping dimension that varies within the same physical recording
    # is automatically excluded from physical-recording ICA/EOG summaries.
    if physical_group_col is None:
        physical_candidates = _infer_physical_level_group_columns(
            recording_qc_df,
            group_columns_used,
            physical_recording_col=physical_recording_col,
        )

        if "timepoint" in physical_candidates:
            physical_group_col = "timepoint"
        elif len(physical_candidates) == 1:
            physical_group_col = physical_candidates[0]

    ocular_ica_results = prepare_ocular_ica_qc(
        recording_qc_df,
        group_col=physical_group_col,
        physical_recording_col=physical_recording_col,
    )

    # =========================================================================
    # 3. POSTERIOR-ALPHA PHYSIOLOGICAL QC — OPTIONAL
    # =========================================================================
    posterior_alpha_results = None

    if posterior_alpha is not None:
        posterior_cfg = dict(posterior_alpha)
        posterior_cfg.setdefault("metadata_fields", metadata_fields)

        posterior_raw = build_posterior_alpha_qc(
            results_by_recording,
            **posterior_cfg,
        )

        # Reuse the shared preparation helper so standalone and orchestrated
        # posterior-alpha QC use identical numeric aggregation.
        posterior_prepared = prepare_posterior_alpha_qc(
            posterior_raw,
            group_columns=group_columns_used,
        )

        posterior_alpha_results = {
            "posterior_psd_df": posterior_raw["posterior_psd_df"],
            "recording_df": posterior_raw["summary_df"].copy(),
            "group_df": posterior_prepared["numeric_df"],
            "report_df": posterior_prepared["report_df"],
        }

    # =========================================================================
    # 4. FRONTAL / FRONTOTEMPORAL HIGH-FREQUENCY QC — OPTIONAL
    # =========================================================================
    high_frequency_results = None

    if high_frequency is not None:
        hf_cfg = dict(high_frequency)
        hf_cfg.setdefault("metadata_fields", metadata_fields)

        # A study-specific configuration can supply Beta, Low Gamma, and
        # High Gamma. The legacy default remains for backward compatibility.
        hf_bands = tuple(str(band) for band in hf_cfg.get("bands", ("beta", "gamma")))

        hf_raw = build_frontal_high_frequency_qc(
            results_by_recording,
            **hf_cfg,
        )

        # Reuse the shared summarizer so standalone and orchestrated high-
        # frequency QC have identical recording-level and aggregate structure.
        hf_tables = summarize_frontal_high_frequency_qc(
            hf_raw,
            group_columns=group_columns_used,
            bands=hf_bands,
        )

        high_frequency_results = {
            "recording_df": hf_raw["summary_df"].copy(),
            "group_df": hf_tables["aggregate_df"],
            "summary_tables": hf_tables,
            "bands": hf_bands,
        }

    # =========================================================================
    # 5. QC SETTINGS / TRACEABILITY
    # =========================================================================
    settings = {
        "group_columns": group_columns_used,
        "minimum_clean_minutes": minimum_clean_minutes,
        "physical_recording_col": physical_recording_col,
        "physical_group_col": physical_group_col,
        "bad_channel_method_qc_enabled": True,
        "posterior_alpha_enabled": posterior_alpha is not None,
        "high_frequency_enabled": high_frequency is not None,
    }

    # =========================================================================
    # CONSOLE SUMMARY
    # =========================================================================
    if verbose:
        clean_duration_text = (
            f">= {minimum_clean_minutes:g} min"
            if minimum_clean_minutes is not None
            else "None"
        )

        n_bad_channels_profiled = len(
            bad_channel_method_qc.get("summary_df", pd.DataFrame())
        )

        print("\nPart 1 qEEG QC complete")
        print("=" * 60)
        print(f"Logical recordings:       {len(recording_qc_df)}")
        print(f"QC grouping:              {', '.join(group_columns_used)}")
        print(f"Clean-duration criterion: {clean_duration_text}")
        print(f"Bad-channel provenance:   enabled ({n_bad_channels_profiled} channels profiled)")
        print(f"Posterior-alpha QC:       {'enabled' if posterior_alpha_results is not None else 'disabled'}")
        print(f"High-frequency QC:        {'enabled' if high_frequency_results is not None else 'disabled'}")
        print("=" * 60)

    return {
        "preprocessing": preprocessing_results,
        "ocular_ica": ocular_ica_results,
        "posterior_alpha": posterior_alpha_results,
        "high_frequency": high_frequency_results,
        "settings": settings,
    }

# =============================================================================
# PART 1 qEEG PLOT-DATA PREPARATION
# =============================================================================
# The analysis pipeline prepares and validates all numerical results. This
# function performs the final lightweight selection needed for plotting while
# deliberately leaving the actual plotting calls explicit in the notebook.
#
# Intended workflow:
#
#   qeeg_results = run_qeeg_part1_pipeline(...)
#   qeeg_plot_data = prepare_qeeg_plot_data(qeeg_results, timepoint="H1")
#
#   plot_mean_psd_with_std(qeeg_plot_data["psd_df"], ...)
#   plot_professional_bar(qeeg_plot_data["absolute_power_df"], ...)
#   plot_band_power_topomaps(
#       qeeg_plot_data["topomap_template_epochs"],
#       qeeg_plot_data["topomaps"]["EO"]["absolute"],
#       ...
#   )
#
# This keeps data preparation automated while preserving full manual control
# over colors, fonts, titles, figure size, color maps, scaling, and annotations.


def prepare_qeeg_plot_data(
    qeeg_results: Mapping[str, Any],
    *,
    timepoint: Any | None = None,
    condition_column: str = "eye_state",
    timepoint_column: str = "timepoint",
    preferred_condition_order: Sequence[Any] = ("EO", "EC"),
) -> dict[str, Any]:
    """
    Prepare one timepoint of Part 1 qEEG results for explicit plotting calls.

    The function performs only plot-data selection and organization. It does
    not create figures and does not modify the underlying qEEG calculations.

    Parameters
    ----------
    qeeg_results
        Prepared result object returned by ``run_qeeg_part1_pipeline``.

    timepoint
        Study timepoint to prepare, such as ``"H1"``. If None, the function
        automatically selects the timepoint only when exactly one non-missing
        timepoint is available. If multiple timepoints are present, an explicit
        value is required to prevent accidental longitudinal pooling.

    condition_column
        Column identifying the plotting condition. Default is ``"eye_state"``
        so EO and EC remain separate.

    timepoint_column
        Column identifying study timepoint. Default is ``"timepoint"``.

    preferred_condition_order
        Preferred display/access order for conditions. Available conditions not
        listed here are appended rather than dropped.

    Returns
    -------
    dict[str, Any]
        Plot-ready data containing:

        - ``timepoint``
        - ``condition_column``
        - ``conditions``
        - ``psd_df``
        - ``absolute_power_df``
        - ``relative_power_df``
        - ``spectral_ratio_df``
        - ``topomap_template_epochs``
        - ``topomaps`` keyed directly by condition
        - ``recording_ids_by_condition``

    Notes
    -----
    No EO/EC or timepoint averaging is performed here. Group aggregation for
    topomaps has already been completed numerically by the Part 1 preparation
    pipeline using the configured grouping columns.
    """
    if not isinstance(qeeg_results, Mapping):
        raise TypeError(
            "qeeg_results must be a mapping returned by "
            "run_qeeg_part1_pipeline()."
        )

    required_sections = {"tables", "groups", "topomaps"}
    missing_sections = required_sections - set(qeeg_results.keys())
    if missing_sections:
        raise KeyError(
            "qeeg_results is missing required prepared sections: "
            f"{sorted(missing_sections)}"
        )

    tables = qeeg_results["tables"]
    required_tables = {
        "combined_psd_df",
        "absolute_power_df",
        "relative_power_df",
        "spectral_ratio_df",
    }
    missing_tables = required_tables - set(tables.keys())
    if missing_tables:
        raise KeyError(
            "qeeg_results['tables'] is missing required tables: "
            f"{sorted(missing_tables)}"
        )

    table_lookup = {
        "psd_df": tables["combined_psd_df"],
        "absolute_power_df": tables["absolute_power_df"],
        "relative_power_df": tables["relative_power_df"],
        "spectral_ratio_df": tables["spectral_ratio_df"],
    }

    # Confirm every plotting table carries condition and timepoint metadata.
    for table_name, table in table_lookup.items():
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"{table_name} must be a pandas DataFrame.")
        if condition_column not in table.columns:
            raise KeyError(
                f"{table_name} is missing condition column "
                f"'{condition_column}'."
            )
        if timepoint_column not in table.columns:
            raise KeyError(
                f"{table_name} is missing timepoint column "
                f"'{timepoint_column}'."
            )

    # Use the absolute-power table as the authoritative list of available
    # timepoints because it has one compact set of recording-level metrics.
    available_timepoints = (
        table_lookup["absolute_power_df"][timepoint_column]
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    if not available_timepoints:
        raise ValueError("No non-missing qEEG timepoints are available.")

    if timepoint is None:
        if len(available_timepoints) != 1:
            raise ValueError(
                "Multiple qEEG timepoints are available. Supply timepoint=... "
                "explicitly so longitudinal timepoints are not mixed. "
                f"Available timepoints: {available_timepoints}"
            )
        timepoint_used = available_timepoints[0]
    else:
        timepoint_used = timepoint
        if timepoint_used not in available_timepoints:
            raise ValueError(
                f"Requested timepoint '{timepoint_used}' is unavailable. "
                f"Available timepoints: {available_timepoints}"
            )

    # Filter every table to the requested timepoint. The returned DataFrames
    # are independent copies, so notebook plotting edits cannot alter the
    # pipeline's stored result tables.
    filtered_tables: dict[str, pd.DataFrame] = {}
    for table_name, table in table_lookup.items():
        filtered = table.loc[
            table[timepoint_column] == timepoint_used
        ].copy()
        if filtered.empty:
            raise ValueError(
                f"{table_name} contains no rows for "
                f"timepoint '{timepoint_used}'."
            )
        filtered_tables[table_name] = filtered

    available_conditions = (
        filtered_tables["psd_df"][condition_column]
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    if not available_conditions:
        raise ValueError(
            f"No non-missing '{condition_column}' values are available for "
            f"timepoint '{timepoint_used}'."
        )

    preferred_order = list(preferred_condition_order)
    conditions = [
        value for value in preferred_order
        if value in available_conditions
    ]
    conditions.extend([
        value for value in available_conditions
        if value not in conditions
    ])

    group_columns = tuple(qeeg_results["groups"].get("columns", ()))
    if not group_columns:
        raise ValueError("qeeg_results does not contain analysis group columns.")
    if condition_column not in group_columns:
        raise KeyError(
            f"condition_column '{condition_column}' is not one of the prepared "
            f"analysis group columns: {group_columns}"
        )
    if timepoint_column not in group_columns:
        raise KeyError(
            f"timepoint_column '{timepoint_column}' is not one of the prepared "
            f"analysis group columns: {group_columns}"
        )

    recording_ids_by_group = qeeg_results["groups"].get(
        "recording_ids_by_group",
        {},
    )
    absolute_by_group = qeeg_results["topomaps"].get(
        "absolute_by_group",
        {},
    )
    relative_by_group = qeeg_results["topomaps"].get(
        "relative_by_group",
        {},
    )

    # Re-key grouped topomap inputs directly by condition so notebook calls are
    # simple: qeeg_plot_data["topomaps"]["EO"]["absolute"].
    topomaps_by_condition: dict[Any, dict[str, Any]] = {}
    recording_ids_by_condition: dict[Any, list[str]] = {}

    for group_key, recording_ids in recording_ids_by_group.items():
        group_key_tuple = (
            group_key if isinstance(group_key, tuple) else (group_key,)
        )
        if len(group_key_tuple) != len(group_columns):
            raise ValueError(
                "Prepared analysis group key length does not match the stored "
                "group columns."
            )

        group_values = dict(zip(group_columns, group_key_tuple))
        if group_values.get(timepoint_column) != timepoint_used:
            continue

        condition = group_values.get(condition_column)
        if condition is None:
            continue
        if condition in topomaps_by_condition:
            raise ValueError(
                f"More than one prepared analysis group maps to condition "
                f"'{condition}' at timepoint '{timepoint_used}'. Add additional "
                "group dimensions explicitly before plotting."
            )
        if group_key not in absolute_by_group or group_key not in relative_by_group:
            raise KeyError(
                f"Missing grouped topomap input for analysis group {group_key}."
            )

        topomaps_by_condition[condition] = {
            "absolute": absolute_by_group[group_key],
            "relative": relative_by_group[group_key],
            "group_key": group_key,
        }
        recording_ids_by_condition[condition] = [
            str(recording_id) for recording_id in recording_ids
        ]

    missing_topomap_conditions = [
        condition for condition in conditions
        if condition not in topomaps_by_condition
    ]
    if missing_topomap_conditions:
        raise KeyError(
            "Grouped topomap inputs are unavailable for conditions: "
            f"{missing_topomap_conditions}"
        )

    return {
        "timepoint": timepoint_used,
        "available_timepoints": available_timepoints,
        "condition_column": condition_column,
        "timepoint_column": timepoint_column,
        "conditions": conditions,
        "psd_df": filtered_tables["psd_df"],
        "absolute_power_df": filtered_tables["absolute_power_df"],
        "relative_power_df": filtered_tables["relative_power_df"],
        "spectral_ratio_df": filtered_tables["spectral_ratio_df"],
        "topomap_template_epochs": qeeg_results["topomaps"]["template_epochs"],
        "topomaps": topomaps_by_condition,
        "recording_ids_by_condition": recording_ids_by_condition,
    }


# =============================================================================
# OPTIONAL AUTOMATIC PART 1 qEEG PLOTTING PIPELINE
# =============================================================================
# ``plot_qeeg_part1_results`` is retained as an optional convenience wrapper.
# The preferred notebook workflow is ``prepare_qeeg_plot_data`` followed by
# explicit calls to the reusable plotting functions so aesthetics remain easy
# to edit manually.


def plot_qeeg_part1_results(
    qeeg_results: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """
    Plot a prepared Part 1 qEEG result object with one user-facing call.

    Timepoints remain separate. Conditions such as EO and EC are compared within
    each timepoint for PSD/bar plots and remain separate for topographic maps.
    """
    if not isinstance(qeeg_results, Mapping):
        raise TypeError("qeeg_results must be a mapping returned by the pipeline.")
    required_top_level = {"tables", "groups", "topomaps"}
    missing = required_top_level - set(qeeg_results.keys())
    if missing:
        raise KeyError(
            "qeeg_results is missing required prepared sections: "
            f"{sorted(missing)}"
        )

    cfg = _merge_nested_config(get_default_qeeg_plot_config(), config)
    tables = qeeg_results["tables"]
    combined_psd_df = tables["combined_psd_df"]
    absolute_power_df = tables["absolute_power_df"]
    relative_power_df = tables["relative_power_df"]
    spectral_ratio_df = tables["spectral_ratio_df"]

    condition_col = str(cfg["condition_column"])
    timepoint_col = str(cfg["timepoint_column"])

    for name, table in {
        "combined_psd_df": combined_psd_df,
        "absolute_power_df": absolute_power_df,
        "relative_power_df": relative_power_df,
        "spectral_ratio_df": spectral_ratio_df,
    }.items():
        if condition_col not in table.columns:
            raise KeyError(f"{name} is missing condition column '{condition_col}'.")
        if timepoint_col not in table.columns:
            raise KeyError(f"{name} is missing timepoint column '{timepoint_col}'.")

    available_timepoints = (
        absolute_power_df[timepoint_col]
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    if not available_timepoints:
        raise ValueError("No non-missing timepoints are available for qEEG plotting.")

    preferred_order = list(cfg["preferred_condition_order"])
    condition_alias = dict(cfg["condition_alias"])
    condition_palette = dict(cfg["condition_palette"])
    band_order = list(cfg["band_order"])
    band_alias = dict(cfg["band_alias"])
    ratio_order = list(cfg["ratio_order"])
    ratio_alias = dict(cfg["ratio_alias"])
    psd_cfg = dict(cfg["psd"])
    bar_cfg = dict(cfg["bar"])
    topomap_cfg = dict(cfg["topomap"])

    figures_by_timepoint: dict[Any, dict[str, Any]] = {}

    for timepoint in available_timepoints:
        psd_tp = combined_psd_df.loc[
            combined_psd_df[timepoint_col] == timepoint
        ].copy()
        abs_tp = absolute_power_df.loc[
            absolute_power_df[timepoint_col] == timepoint
        ].copy()
        rel_tp = relative_power_df.loc[
            relative_power_df[timepoint_col] == timepoint
        ].copy()
        ratio_tp = spectral_ratio_df.loc[
            spectral_ratio_df[timepoint_col] == timepoint
        ].copy()

        available_conditions = (
            psd_tp[condition_col].dropna().drop_duplicates().tolist()
        )
        condition_order = [
            value for value in preferred_order if value in available_conditions
        ]
        condition_order.extend([
            value for value in available_conditions if value not in condition_order
        ])
        if not condition_order:
            raise ValueError(
                f"No condition data are available for timepoint '{timepoint}'."
            )

        # Ensure any newly encountered condition receives a deterministic color.
        missing_palette = [
            value for value in condition_order if value not in condition_palette
        ]
        if missing_palette:
            fallback_colors = sns.color_palette(n_colors=len(missing_palette))
            for value, color in zip(missing_palette, fallback_colors):
                condition_palette[value] = color

        psd_figure, psd_axis, psd_summary_df = plot_mean_psd_with_std(
            psd_tp,
            frequency_col="frequency_hz",
            psd_col="mean_psd_db",
            recording_col="recording_id",
            group_col=condition_col,
            group_order=condition_order,
            group_alias=condition_alias,
            palette=condition_palette,
            linewidth=float(psd_cfg["linewidth"]),
            fill_alpha=float(psd_cfg["fill_alpha"]),
            title=f"Mean Power Spectral Density — {timepoint}",
            xlabel="Frequency (Hz)",
            ylabel="PSD (dB re 1 µV²/Hz)",
            xlim=tuple(psd_cfg["xlim"]),
            xticks=psd_cfg["xticks"],
            figsize=tuple(psd_cfg["figsize"]),
            font_size=float(psd_cfg["font_size"]),
            show_legend=True,
            legend_title="Eye state" if condition_col == "eye_state" else condition_col,
            legend_loc="best",
            show=show,
        )

        absolute_figure, absolute_axis = plot_professional_bar(
            abs_tp,
            x="band",
            y="mean_absolute_power_uv2",
            hue=condition_col,
            order=band_order,
            hue_order=condition_order,
            category_alias=band_alias,
            hue_alias=condition_alias,
            palette=condition_palette,
            estimator="mean",
            errorbar=bar_cfg["errorbar"],
            title=f"Mean Absolute Band Power — {timepoint}",
            xlabel="Frequency band",
            ylabel="Absolute power (µV²)",
            show_legend=True,
            legend_title="Eye state" if condition_col == "eye_state" else condition_col,
            annotate=bool(bar_cfg["annotate"]),
            annotation_mode=bar_cfg["annotation_mode"],
            annotate_decimals=2,
            figsize=tuple(bar_cfg["figsize"]),
            font_size=float(bar_cfg["font_size"]),
            show=show,
        )

        relative_figure, relative_axis = plot_professional_bar(
            rel_tp,
            x="band",
            y="mean_relative_power_percent",
            hue=condition_col,
            order=band_order,
            hue_order=condition_order,
            category_alias=band_alias,
            hue_alias=condition_alias,
            palette=condition_palette,
            estimator="mean",
            errorbar=bar_cfg["errorbar"],
            title=f"Mean Relative Band Power — {timepoint}",
            xlabel="Frequency band",
            ylabel="Relative power (%)",
            show_legend=True,
            legend_title="Eye state" if condition_col == "eye_state" else condition_col,
            annotate=bool(bar_cfg["annotate"]),
            annotation_mode=bar_cfg["annotation_mode"],
            annotate_decimals=1,
            annotate_suffix="%",
            figsize=tuple(bar_cfg["figsize"]),
            font_size=float(bar_cfg["font_size"]),
            show=show,
        )

        ratio_figure, ratio_axis = plot_professional_bar(
            ratio_tp,
            x="ratio",
            y="mean_channel_ratio",
            hue=condition_col,
            order=ratio_order,
            hue_order=condition_order,
            category_alias=ratio_alias,
            hue_alias=condition_alias,
            palette=condition_palette,
            estimator="mean",
            errorbar=bar_cfg["errorbar"],
            title=f"Mean Spectral Power Ratios — {timepoint}",
            xlabel="Spectral-power ratio",
            ylabel="Power ratio",
            show_legend=True,
            legend_title="Eye state" if condition_col == "eye_state" else condition_col,
            annotate=bool(bar_cfg["annotate"]),
            annotation_mode=bar_cfg["annotation_mode"],
            annotate_decimals=2,
            figsize=(8, tuple(bar_cfg["figsize"])[1]),
            font_size=float(bar_cfg["font_size"]),
            show=show,
        )

        figures_by_timepoint[timepoint] = {
            "psd": {
                "figure": psd_figure,
                "axis": psd_axis,
                "summary_df": psd_summary_df,
            },
            "absolute_power": {
                "figure": absolute_figure,
                "axis": absolute_axis,
            },
            "relative_power": {
                "figure": relative_figure,
                "axis": relative_axis,
            },
            "spectral_ratio": {
                "figure": ratio_figure,
                "axis": ratio_axis,
            },
        }

    group_columns = tuple(qeeg_results["groups"]["columns"])
    template_epochs = qeeg_results["topomaps"]["template_epochs"]
    absolute_topomap_results: dict[tuple[Any, ...], dict[str, Any]] = {}
    relative_topomap_results: dict[tuple[Any, ...], dict[str, Any]] = {}

    for power_type, source_by_group, output_by_group in (
        (
            "absolute",
            qeeg_results["topomaps"]["absolute_by_group"],
            absolute_topomap_results,
        ),
        (
            "relative",
            qeeg_results["topomaps"]["relative_by_group"],
            relative_topomap_results,
        ),
    ):
        for group_key, topomap_input in source_by_group.items():
            group_values = dict(zip(group_columns, group_key))
            condition = group_values.get(condition_col)
            timepoint = group_values.get(timepoint_col)
            condition_display = condition_alias.get(condition, str(condition))
            title_prefix = "Absolute Band Power" if power_type == "absolute" else "Relative Band Power"
            title = f"{title_prefix} — {condition_display}, {timepoint}"

            output_by_group[group_key] = plot_band_power_topomaps(
                template_epochs,
                topomap_input,
                power_type=power_type,
                band_order=band_order,
                band_alias=band_alias,
                title=title,
                shared_scale=bool(topomap_cfg["shared_scale"]),
                sphere=topomap_cfg["sphere"],
                cmap=topomap_cfg["cmap"],
                contours=int(topomap_cfg["contours"]),
                sensors=topomap_cfg["sensors"],
                n_cols=int(topomap_cfg["n_cols"]),
                font_size=float(topomap_cfg["font_size"]),
                colorbar_decimals=int(topomap_cfg["colorbar_decimals"]),
                show=show,
            )

    return {
        "by_timepoint": figures_by_timepoint,
        "topomaps": {
            "absolute_by_group": absolute_topomap_results,
            "relative_by_group": relative_topomap_results,
        },
        "config": cfg,
    }



# =============================================================================
# PART 2 — LONGITUDINAL ENDPOINT ENGINE 
# =============================================================================



# IMPORTANT: plot_qeeg_time_course() calls the existing plot_professional_line()
# already present in qeeg_analysis_pipeline.py. Do not duplicate or replace that
# Part 1 plotting helper when applying this patch.

def build_subject_level_qeeg_endpoint_df(
    qeeg_results_by_recording: Mapping[str, Mapping[str, Any]],
    *,
    metadata_overrides: Mapping[str, Any] | None = None,
    recording_id_suffix: str | None = None,
    ratio_value_column: Literal[
        "mean_channel_ratio",
        "global_power_ratio",
    ] = "mean_channel_ratio",
) -> pd.DataFrame:
    """
    Build a long-format subject-level qEEG endpoint dataset.

    The function extracts recording-level scalar qEEG endpoints from the
    outputs of ``run_qeeg_batch_analysis``. Absolute band power, relative
    band power, and spectral ratios are converted into one standardized
    long-format table.

    Parameters
    ----------
    qeeg_results_by_recording
        Mapping returned by ``run_qeeg_batch_analysis``. Each recording
        should contain absolute-power, relative-power, and spectral-ratio
        result dictionaries.

    metadata_overrides
        Optional study metadata to add or overwrite for every recording.

        This is useful while testing longitudinal workflows before the real
        study metadata are available.

        Example::

            {
                "eye_state": "EO",
                "timepoint": "Baseline",
            }

    recording_id_suffix
        Optional suffix appended to each recording ID. This is useful when
        the same qEEG results are reused to simulate multiple timepoints.

        Example:
            ``"Baseline"`` produces
            ``"NDARAH371ZT7_eeg_Baseline"``.

    ratio_value_column
        Column from ``overall_ratio_df`` used as the scalar spectral-ratio
        endpoint. Supported options are ``"mean_channel_ratio"`` and
        ``"global_power_ratio"``.

    Returns
    -------
    pd.DataFrame
        Long-format subject-level endpoint dataset containing one row per
        recording and qEEG endpoint.

    Notes
    -----
    PSD curves and channel-level topographic values are intentionally not
    included here. They remain in their existing Part 1 result tables.

    The function does not calculate change from baseline. That is Part 2B.
    """
    if not qeeg_results_by_recording:
        raise ValueError("qeeg_results_by_recording is empty.")

    if metadata_overrides is not None and not isinstance(
        metadata_overrides,
        Mapping,
    ):
        raise TypeError("metadata_overrides must be a mapping.")

    endpoint_rows: list[dict[str, Any]] = []

    # ============================================================
    # Process each analyzed recording
    # ============================================================
    for result_key, result in qeeg_results_by_recording.items():

        base_recording_id = str(
            result.get("recording_id", result_key)
        )

        # Preserve the physical source recording separately from the logical
        # EO/EC recording ID used by the qEEG analysis.
        source_recording_id = str(
            result.get("source_recording_id", base_recording_id)
        )

        if recording_id_suffix is None:
            recording_id = base_recording_id
        else:
            recording_id = (
                f"{base_recording_id}_{recording_id_suffix}"
            )

        # --------------------------------------------------------
        # Store recording-level identifiers and traceability fields
        # --------------------------------------------------------
        recording_metadata: dict[str, Any] = {
            "recording_id": recording_id,
            "source_recording_id": source_recording_id,
            "subject_id": result.get("subject_id"),
            "label": result.get("label"),
            "file_path": result.get("file_path"),
            "n_epochs_clean": result.get("n_epochs_clean"),
            "n_channels": result.get("n_channels"),
            "n_channels_total": result.get(
                "n_channels_total",
                result.get("n_channels"),
            ),
            "n_channels_qeeg": result.get("n_channels_qeeg"),
            "sfreq_hz": result.get("sfreq_hz"),
        }

        # Preserve study metadata if it already exists.
        for field in (
            "cohort",
            "condition",
            "eye_state",
            "timepoint",
            "visit",
            "dose",
        ):
            if field in result:
                recording_metadata[field] = result[field]

        # Explicit test/study values override existing values.
        if metadata_overrides is not None:
            recording_metadata.update(metadata_overrides)

        # ========================================================
        # Absolute band-power endpoints
        # ========================================================
        absolute_result = result.get("absolute_power_result")

        if not isinstance(absolute_result, Mapping):
            raise KeyError(
                f"{source_recording_id}: "
                "'absolute_power_result' is missing."
            )

        absolute_df = absolute_result.get(
            "overall_band_power_df"
        )

        if not isinstance(absolute_df, pd.DataFrame):
            raise TypeError(
                f"{source_recording_id}: "
                "'overall_band_power_df' must be a DataFrame."
            )

        required_absolute_columns = {
            "band",
            "mean_absolute_power_uv2",
            "fmin_hz",
            "fmax_hz",
        }

        missing = (
            required_absolute_columns
            - set(absolute_df.columns)
        )

        if missing:
            raise KeyError(
                f"{source_recording_id}: absolute-power table "
                f"is missing columns {sorted(missing)}."
            )

        for _, row in absolute_df.iterrows():
            endpoint_rows.append({
                **recording_metadata,
                "endpoint_type": "absolute_power",
                "endpoint": str(row["band"]),
                "value": float(
                    row["mean_absolute_power_uv2"]
                ),
                "unit": "uV^2",
                "fmin_hz": float(row["fmin_hz"]),
                "fmax_hz": float(row["fmax_hz"]),
                "numerator_band": np.nan,
                "denominator_band": np.nan,
                "ratio_summary_method": np.nan,
            })

        # ========================================================
        # Relative band-power endpoints
        # ========================================================
        relative_result = result.get("relative_power_result")

        if not isinstance(relative_result, Mapping):
            raise KeyError(
                f"{source_recording_id}: "
                "'relative_power_result' is missing."
            )

        relative_df = relative_result.get(
            "overall_relative_power_df"
        )

        if not isinstance(relative_df, pd.DataFrame):
            raise TypeError(
                f"{source_recording_id}: "
                "'overall_relative_power_df' must be a DataFrame."
            )

        required_relative_columns = {
            "band",
            "mean_relative_power_percent",
            "fmin_hz",
            "fmax_hz",
        }

        missing = (
            required_relative_columns
            - set(relative_df.columns)
        )

        if missing:
            raise KeyError(
                f"{source_recording_id}: relative-power table "
                f"is missing columns {sorted(missing)}."
            )

        for _, row in relative_df.iterrows():
            endpoint_rows.append({
                **recording_metadata,
                "endpoint_type": "relative_power",
                "endpoint": str(row["band"]),
                "value": float(
                    row["mean_relative_power_percent"]
                ),
                "unit": "%",
                "fmin_hz": float(row["fmin_hz"]),
                "fmax_hz": float(row["fmax_hz"]),
                "numerator_band": np.nan,
                "denominator_band": np.nan,
                "ratio_summary_method": np.nan,
            })

        # ========================================================
        # Spectral-ratio endpoints
        # ========================================================
        ratio_result = result.get("spectral_ratio_result")

        if not isinstance(ratio_result, Mapping):
            raise KeyError(
                f"{source_recording_id}: "
                "'spectral_ratio_result' is missing."
            )

        ratio_df = ratio_result.get("overall_ratio_df")

        if not isinstance(ratio_df, pd.DataFrame):
            raise TypeError(
                f"{source_recording_id}: "
                "'overall_ratio_df' must be a DataFrame."
            )

        required_ratio_columns = {
            "ratio",
            "numerator_band",
            "denominator_band",
            ratio_value_column,
        }

        missing = required_ratio_columns - set(
            ratio_df.columns
        )

        if missing:
            raise KeyError(
                f"{source_recording_id}: spectral-ratio table "
                f"is missing columns {sorted(missing)}."
            )

        for _, row in ratio_df.iterrows():
            endpoint_rows.append({
                **recording_metadata,
                "endpoint_type": "spectral_ratio",
                "endpoint": str(row["ratio"]),
                "value": float(row[ratio_value_column]),
                "unit": "ratio",
                "fmin_hz": np.nan,
                "fmax_hz": np.nan,
                "numerator_band": row["numerator_band"],
                "denominator_band": row[
                    "denominator_band"
                ],
                "ratio_summary_method": row.get(
                    "summary_method",
                    ratio_value_column,
                ),
            })

    # ============================================================
    # Build and organize the final endpoint table
    # ============================================================
    endpoint_df = pd.DataFrame(endpoint_rows)

    preferred_columns = [
        "recording_id",
        "source_recording_id",
        "subject_id",
        "label",
        "cohort",
        "condition",
        "eye_state",
        "timepoint",
        "visit",
        "dose",
        "endpoint_type",
        "endpoint",
        "value",
        "unit",
        "fmin_hz",
        "fmax_hz",
        "numerator_band",
        "denominator_band",
        "ratio_summary_method",
        "n_epochs_clean",
        "n_channels",
        "n_channels_total",
        "n_channels_qeeg",
        "sfreq_hz",
        "file_path",
    ]

    ordered_columns = [
        column
        for column in preferred_columns
        if column in endpoint_df.columns
    ]

    remaining_columns = [
        column
        for column in endpoint_df.columns
        if column not in ordered_columns
    ]

    endpoint_df = endpoint_df[
        ordered_columns + remaining_columns
    ].copy()

    return endpoint_df


def calculate_change_from_baseline(
    endpoint_df: pd.DataFrame,
    *,
    baseline_timepoint: str = "Baseline",
    timepoint_col: str = "timepoint",
    value_col: str = "value",
    match_columns: Sequence[str] = (
        "subject_id",
        "eye_state",
        "endpoint_type",
        "endpoint",
    ),
    require_baseline: bool = True,
) -> pd.DataFrame:
    """
    Add subject-level baseline values and change-from-baseline values.

    Each endpoint is matched to the baseline value from the same subject,
    eye state, endpoint type, and endpoint. Additional or alternative
    matching columns can be supplied through ``match_columns``.

    Parameters
    ----------
    endpoint_df
        Long-format subject-level qEEG endpoint dataset produced by
        ``build_subject_level_qeeg_endpoint_df``.

    baseline_timepoint
        Value in ``timepoint_col`` identifying baseline recordings.

    timepoint_col
        Column containing the longitudinal timepoint label.

    value_col
        Column containing the endpoint value.

    match_columns
        Columns defining which baseline observation belongs to each
        longitudinal observation. Defaults to subject, eye state,
        endpoint type, and endpoint.

    require_baseline
        If True, raise an error when any row cannot be matched to a
        baseline value. If False, unmatched rows receive NaN baseline
        and change-from-baseline values.

    Returns
    -------
    pd.DataFrame
        Copy of the input endpoint dataset with additional columns:

        - ``baseline_value``
        - ``change_from_baseline``
        - ``baseline_recording_id``
        - ``is_baseline``

    Notes
    -----
    Change from baseline is calculated as:

        current endpoint value - subject-specific baseline endpoint value

    Baseline rows therefore have a change-from-baseline value of zero.
    """
    if endpoint_df.empty:
        raise ValueError("endpoint_df is empty.")

    match_columns = tuple(match_columns)

    required_columns = {
        *match_columns,
        timepoint_col,
        value_col,
    }

    missing_columns = required_columns - set(endpoint_df.columns)

    if missing_columns:
        raise KeyError(
            "endpoint_df is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df = endpoint_df.copy()

    # Ensure endpoint values are numeric before subtraction.
    df[value_col] = pd.to_numeric(
        df[value_col],
        errors="coerce",
    )

    # Longitudinal identity fields must be complete. Failing here keeps
    # notebook-level code free of ad hoc missing-value checks.
    identity_columns = list(
        dict.fromkeys([
            *match_columns,
            timepoint_col,
        ])
    )

    missing_identity = (
        df[identity_columns]
        .isna()
        .sum()
    )

    if (missing_identity > 0).any():
        missing_identity = missing_identity.loc[
            missing_identity > 0
        ]

        raise ValueError(
            "Longitudinal endpoint rows contain missing identity values:\n"
            + missing_identity.to_string()
        )

    if require_baseline and df[value_col].isna().any():
        raise ValueError(
            f"Longitudinal endpoint column '{value_col}' contains "
            f"{int(df[value_col].isna().sum())} missing value(s)."
        )

    # ============================================================
    # Identify the subject-specific baseline endpoint values
    # ============================================================
    baseline_df = df.loc[
        df[timepoint_col] == baseline_timepoint
    ].copy()

    if baseline_df.empty:
        raise ValueError(
            f"No baseline rows were found where "
            f"{timepoint_col} == {baseline_timepoint!r}."
        )

    # Each subject/eye-state/endpoint combination should have one
    # and only one baseline observation.
    duplicate_baselines = baseline_df.duplicated(
        subset=list(match_columns),
        keep=False,
    )

    if duplicate_baselines.any():
        duplicate_keys = (
            baseline_df.loc[
                duplicate_baselines,
                list(match_columns),
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        raise ValueError(
            "Multiple baseline observations were found for one or more "
            "matching keys. Each endpoint must have a unique baseline.\n\n"
            f"{duplicate_keys.to_string(index=False)}"
        )

    # Keep the baseline recording ID for traceability when available.
    baseline_columns = [
        *match_columns,
        value_col,
    ]

    if "recording_id" in baseline_df.columns:
        baseline_columns.append("recording_id")

    baseline_lookup = baseline_df[
        baseline_columns
    ].copy()

    baseline_lookup = baseline_lookup.rename(
        columns={
            value_col: "baseline_value",
            "recording_id": "baseline_recording_id",
        }
    )

    # ============================================================
    # Match every longitudinal endpoint to its own baseline
    # ============================================================
    df = df.merge(
        baseline_lookup,
        on=list(match_columns),
        how="left",
        validate="many_to_one",
    )

    missing_baseline = df["baseline_value"].isna()

    if require_baseline and missing_baseline.any():
        missing_keys = (
            df.loc[
                missing_baseline,
                list(match_columns),
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        raise ValueError(
            "One or more endpoint rows could not be matched to a "
            "baseline observation.\n\n"
            f"{missing_keys.to_string(index=False)}"
        )

    # ============================================================
    # Calculate subject-level change from baseline
    # ============================================================
    df["change_from_baseline"] = (
        df[value_col] - df["baseline_value"]
    )

    df["is_baseline"] = (
        df[timepoint_col] == baseline_timepoint
    )

    return df


def build_qeeg_metric_summary_df(
    longitudinal_summary_df: pd.DataFrame,
    *,
    decimals: int = 2,
    timepoint_alias: Mapping[str, str] | None = None,
    timepoint_order: Sequence[str] | None = None,
    eye_state_order: Sequence[str] | None = None,
    endpoint_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Build a compact presentation-ready qEEG longitudinal summary table.

    The function reformats ``qeeg_longitudinal_summary_df`` into a simpler
    table containing eye state, timepoint, a human-readable qEEG metric,
    sample size, mean ± SD, and change from baseline ± SD.

    Optional display orders are applied before the compact columns are created,
    so notebook code does not need to manipulate hidden ``endpoint_type`` or
    ``endpoint`` columns after formatting.
    """
    if longitudinal_summary_df.empty:
        raise ValueError("longitudinal_summary_df is empty.")

    required_columns = {
        "eye_state",
        "timepoint",
        "endpoint_type",
        "endpoint",
        "unit",
        "n_subjects",
        "mean_value",
        "sd_value",
        "mean_change_from_baseline",
        "sd_change_from_baseline",
    }

    missing_columns = (
        required_columns
        - set(longitudinal_summary_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "longitudinal_summary_df is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df = longitudinal_summary_df.copy()

    # ============================================================
    # Apply stable study/report ordering while the raw endpoint
    # identity columns are still available.
    # ============================================================
    if timepoint_order is not None:
        timepoint_order = list(timepoint_order)
        observed = set(df["timepoint"].dropna())
        missing = [
            value
            for value in timepoint_order
            if value not in observed
        ]
        if missing:
            raise ValueError(
                f"Requested timepoints are absent from the longitudinal "
                f"summary: {missing}"
            )
        df["timepoint"] = pd.Categorical(
            df["timepoint"],
            categories=timepoint_order,
            ordered=True,
        )

    if eye_state_order is not None:
        eye_state_order = list(eye_state_order)
        observed = set(df["eye_state"].dropna())
        missing = [
            value
            for value in eye_state_order
            if value not in observed
        ]
        if missing:
            raise ValueError(
                f"Requested eye states are absent from the longitudinal "
                f"summary: {missing}"
            )
        df["eye_state"] = pd.Categorical(
            df["eye_state"],
            categories=eye_state_order,
            ordered=True,
        )

    endpoint_type_order = [
        "absolute_power",
        "relative_power",
        "spectral_ratio",
    ]
    df["_endpoint_type_order"] = pd.Categorical(
        df["endpoint_type"],
        categories=endpoint_type_order,
        ordered=True,
    )

    if endpoint_order is not None:
        endpoint_order = list(endpoint_order)
        observed = set(df["endpoint"].dropna())
        missing = [
            value
            for value in endpoint_order
            if value not in observed
        ]
        if missing:
            raise ValueError(
                f"Requested endpoints are absent from the longitudinal "
                f"summary: {missing}"
            )
        df["_endpoint_order"] = pd.Categorical(
            df["endpoint"],
            categories=endpoint_order,
            ordered=True,
        )
    else:
        df["_endpoint_order"] = df["endpoint"]

    sort_columns = [
        "timepoint",
        "eye_state",
        "_endpoint_type_order",
        "_endpoint_order",
    ]

    df = (
        df.sort_values(
            sort_columns,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # ============================================================
    # Create human-readable qEEG metric names
    # ============================================================
    band_alias = {
        "delta": "Delta",
        "theta": "Theta",
        "alpha": "Alpha",
        "beta": "Beta",
        "gamma": "Gamma",
    }

    ratio_alias = {
        "theta_beta": "Theta/Beta Ratio",
        "alpha_theta": "Alpha/Theta Ratio",
        "alpha_beta": "Alpha/Beta Ratio",
        "delta_alpha": "Delta/Alpha Ratio",
    }

    def _format_metric(row: pd.Series) -> str:
        endpoint_type = row["endpoint_type"]
        endpoint = row["endpoint"]

        if endpoint_type == "absolute_power":
            band = band_alias.get(endpoint, str(endpoint).title())
            return f"Absolute {band} Power"

        if endpoint_type == "relative_power":
            band = band_alias.get(endpoint, str(endpoint).title())
            return f"Relative {band} Power"

        if endpoint_type == "spectral_ratio":
            return ratio_alias.get(
                endpoint,
                str(endpoint).replace("_", "/").title(),
            )

        return str(endpoint).replace("_", " ").title()

    df["metric"] = df.apply(
        _format_metric,
        axis=1,
    )

    # ============================================================
    # Apply optional display aliases
    # ============================================================
    if timepoint_alias is not None:
        df["timepoint"] = df["timepoint"].replace(
            dict(timepoint_alias)
        )

    df["unit"] = df["unit"].replace({
        "uV^2": "µV²",
        "ratio": "ratio",
    })

    # ============================================================
    # Format mean ± SD and change-from-baseline ± SD
    # ============================================================
    def _format_mean_sd(
        mean_value: Any,
        sd_value: Any,
    ) -> str:
        if pd.isna(mean_value):
            return ""

        if pd.isna(sd_value):
            return f"{mean_value:.{decimals}f}"

        return (
            f"{mean_value:.{decimals}f} "
            f"± {sd_value:.{decimals}f}"
        )

    df["mean_sd"] = [
        _format_mean_sd(mean, sd)
        for mean, sd in zip(
            df["mean_value"],
            df["sd_value"],
        )
    ]

    df["change_from_baseline_sd"] = [
        _format_mean_sd(mean, sd)
        for mean, sd in zip(
            df["mean_change_from_baseline"],
            df["sd_change_from_baseline"],
        )
    ]

    # ============================================================
    # Keep only presentation-ready columns
    # ============================================================
    output_columns = []

    if "label" in df.columns:
        output_columns.append("label")

    output_columns.extend([
        "eye_state",
        "timepoint",
        "metric",
        "unit",
        "n_subjects",
        "mean_sd",
        "change_from_baseline_sd",
    ])

    qeeg_metric_summary_df = df[
        output_columns
    ].copy()

    qeeg_metric_summary_df = qeeg_metric_summary_df.rename(
        columns={
            "label": "Label",
            "eye_state": "Eye state",
            "timepoint": "Timepoint",
            "metric": "Metric",
            "unit": "Unit",
            "n_subjects": "N",
            "mean_sd": "Mean ± SD",
            "change_from_baseline_sd": (
                "Change from baseline ± SD"
            ),
        }
    )

    return qeeg_metric_summary_df


def plot_qeeg_time_course(
    longitudinal_summary_df: pd.DataFrame,
    *,
    endpoint_type: str,
    timepoint_order: Sequence[str],
    endpoint_order: Sequence[str] | None = None,
    label: str | None = None,
    eye_state: str | None = None,
    series_alias: Mapping[str, str] | None = None,
    palette: Mapping[str, str] | Sequence[str] | str | None = None,
    title: str | None = None,
    ylabel: str = "Change from baseline",
    legend_title: str | None = None,
    figsize: tuple[float, float] = (9.0, 5.0),
    font_size: float = 12.0,
    show: bool = True,
) -> tuple[Any, Any]:
    """
    Plot longitudinal mean qEEG change from baseline with ±1 SD bands.

    The function uses the group-level output from
    ``build_qeeg_longitudinal_summary`` and delegates the actual plotting
    to ``plot_professional_line``.

    Parameters
    ----------
    longitudinal_summary_df
        Group-level longitudinal qEEG summary containing mean and SD
        change-from-baseline values.

    endpoint_type
        qEEG endpoint family to plot.

        Examples:
            ``"absolute_power"``
            ``"relative_power"``
            ``"spectral_ratio"``

    timepoint_order
        Ordered longitudinal timepoints.

        Example:
            ``["Baseline", "H1", "H2", "H4", "H8", "H24"]``

    endpoint_order
        Optional order of frequency bands or spectral ratios.

    label
        Optional label/cohort filter.

    eye_state
        Optional eye-state filter, such as ``"EO"`` or ``"EC"``.

    series_alias
        Optional mapping from endpoint names to display labels.

    palette
        Optional colors passed to ``plot_professional_line``.

    title
        Figure title.

    ylabel
        Y-axis label.

    legend_title
        Optional legend title. When None, frequency-band plots use
        ``"Frequency band"`` and spectral-ratio plots use
        ``"Spectral ratio"``.

    figsize
        Figure size.

    font_size
        Base plot font size.

    show
        Whether to display the figure immediately.

    Returns
    -------
    figure, axis
        Matplotlib Figure and Axes objects.

    Notes
    -----
    The plotted line represents the group mean change from baseline.
    The shaded region represents ±1 standard deviation across subjects.
    """
    if longitudinal_summary_df.empty:
        raise ValueError("longitudinal_summary_df is empty.")

    required_columns = {
        "endpoint_type",
        "endpoint",
        "timepoint",
        "mean_change_from_baseline",
        "sd_change_from_baseline",
    }

    missing_columns = (
        required_columns
        - set(longitudinal_summary_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "longitudinal_summary_df is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ------------------------------------------------------------
    # Select the requested qEEG endpoint family
    # ------------------------------------------------------------
    plot_df = longitudinal_summary_df.loc[
        longitudinal_summary_df["endpoint_type"] == endpoint_type
    ].copy()

    if label is not None:
        if "label" not in plot_df.columns:
            raise KeyError("The summary does not contain a 'label' column.")

        plot_df = plot_df.loc[
            plot_df["label"] == label
        ].copy()

    if eye_state is not None:
        if "eye_state" not in plot_df.columns:
            raise KeyError(
                "The summary does not contain an 'eye_state' column."
            )

        plot_df = plot_df.loc[
            plot_df["eye_state"] == eye_state
        ].copy()

    if plot_df.empty:
        raise ValueError(
            "No rows remain after applying the requested filters."
        )

    # ------------------------------------------------------------
    # Resolve endpoint and timepoint ordering
    # ------------------------------------------------------------
    if endpoint_order is None:
        endpoint_order_used = (
            plot_df["endpoint"]
            .drop_duplicates()
            .tolist()
        )
    else:
        endpoint_order_used = list(endpoint_order)

    timepoint_order_used = list(timepoint_order)

    missing_timepoints = [
        timepoint
        for timepoint in timepoint_order_used
        if timepoint not in set(plot_df["timepoint"])
    ]

    if missing_timepoints:
        raise ValueError(
            f"Missing requested timepoints: {missing_timepoints}"
        )

    # Use numeric positions internally so the uncertainty ribbon can
    # be drawn reliably while preserving readable timepoint labels.
    timepoint_positions = {
        timepoint: index
        for index, timepoint in enumerate(timepoint_order_used)
    }

    plot_df = plot_df.loc[
        plot_df["timepoint"].isin(timepoint_order_used)
        & plot_df["endpoint"].isin(endpoint_order_used)
    ].copy()

    plot_df["timepoint_position"] = (
        plot_df["timepoint"]
        .map(timepoint_positions)
        .astype(float)
    )

    # ------------------------------------------------------------
    # Calculate precomputed mean ± SD uncertainty bounds
    # ------------------------------------------------------------
    plot_df["sd_change_from_baseline"] = (
        pd.to_numeric(
            plot_df["sd_change_from_baseline"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    plot_df["lower_change"] = (
        plot_df["mean_change_from_baseline"]
        - plot_df["sd_change_from_baseline"]
    )

    plot_df["upper_change"] = (
        plot_df["mean_change_from_baseline"]
        + plot_df["sd_change_from_baseline"]
    )

    # ------------------------------------------------------------
    # Draw the professional longitudinal line plot
    # ------------------------------------------------------------
    legend_title_used = (
        legend_title
        if legend_title is not None
        else (
            "Spectral ratio"
            if endpoint_type == "spectral_ratio"
            else "Frequency band"
        )
    )

    figure, axis = plot_professional_line(
        plot_df,
        x="timepoint_position",
        y="mean_change_from_baseline",

        hue="endpoint",
        hue_order=endpoint_order_used,
        series_alias=series_alias,

        lower="lower_change",
        upper="upper_change",

        palette=palette,
        linewidth=2.6,
        marker="o",
        markersize=7.0,
        fill_alpha=0.15,

        horizontal_lines=[
            {
                "y": 0.0,
                "color": "#6B7280",
                "linestyle": "--",
                "linewidth": 1.5,
            }
        ],

        title=title,
        xlabel="Timepoint",
        ylabel=ylabel,

        figsize=figsize,
        font_size=font_size,

        show_legend=True,
        legend_title=legend_title_used,
        legend_loc="best",

        grid_axis="both",
        show=False,
    )

    # Replace numeric x positions with the actual study timepoint labels.
    axis.set_xticks(
        list(timepoint_positions.values())
    )

    axis.set_xticklabels(
        timepoint_order_used,
        fontweight="bold",
    )

    figure.tight_layout()

    if show:
        plt.show()

    return figure, axis


def build_qeeg_longitudinal_summary(
    subject_level_change_df: pd.DataFrame,
    *,
    group_columns: Sequence[str] = (
        "label",
        "eye_state",
        "timepoint",
        "endpoint_type",
        "endpoint",
        "unit",
    ),
    subject_col: str = "subject_id",
    value_col: str = "value",
    change_col: str = "change_from_baseline",
) -> pd.DataFrame:
    """
    Build a longitudinal group-level summary of subject-level qEEG endpoints.

    The function summarizes both the observed endpoint values and their
    subject-specific changes from baseline within each requested group.

    Parameters
    ----------
    subject_level_change_df
        Long-format subject-level qEEG dataset produced by
        ``calculate_change_from_baseline``.

    group_columns
        Columns defining each longitudinal summary group. By default,
        results are summarized by label, eye state, timepoint, endpoint
        type, endpoint, and unit.

    subject_col
        Column containing the subject identifier.

    value_col
        Column containing the observed qEEG endpoint value.

    change_col
        Column containing the subject-level change from baseline.

    Returns
    -------
    pd.DataFrame
        One row per qEEG endpoint and longitudinal group containing
        descriptive statistics for both the raw endpoint value and
        change from baseline.

    Notes
    -----
    This function performs descriptive summarization only. It does not
    perform hypothesis testing or statistical modeling.
    """
    if subject_level_change_df.empty:
        raise ValueError("subject_level_change_df is empty.")

    group_columns = tuple(group_columns)

    required_columns = {
        *group_columns,
        subject_col,
        value_col,
        change_col,
    }

    missing_columns = (
        required_columns
        - set(subject_level_change_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "subject_level_change_df is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df = subject_level_change_df.copy()

    # Ensure endpoint and change values are numeric.
    df[value_col] = pd.to_numeric(
        df[value_col],
        errors="coerce",
    )

    df[change_col] = pd.to_numeric(
        df[change_col],
        errors="coerce",
    )

    # ============================================================
    # Build one descriptive row per longitudinal endpoint group
    # ============================================================
    summary_rows: list[dict[str, Any]] = []

    grouped_df = df.groupby(
        list(group_columns),
        observed=True,
        dropna=False,
        sort=False,
    )

    for group_key, group_df in grouped_df:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_values = dict(
            zip(group_columns, group_key)
        )

        endpoint_values = group_df[value_col].dropna()
        change_values = group_df[change_col].dropna()

        n_subjects = int(
            group_df.loc[
                group_df[value_col].notna(),
                subject_col,
            ].nunique()
        )

        summary_rows.append({
            **group_values,

            # Number of contributing subjects.
            "n_subjects": n_subjects,

            # --------------------------------------------
            # Observed endpoint values
            # --------------------------------------------
            "mean_value": (
                float(endpoint_values.mean())
                if not endpoint_values.empty
                else np.nan
            ),
            "sd_value": (
                float(endpoint_values.std(ddof=1))
                if len(endpoint_values) > 1
                else np.nan
            ),
            "sem_value": (
                float(
                    endpoint_values.std(ddof=1)
                    / np.sqrt(len(endpoint_values))
                )
                if len(endpoint_values) > 1
                else np.nan
            ),
            "median_value": (
                float(endpoint_values.median())
                if not endpoint_values.empty
                else np.nan
            ),
            "minimum_value": (
                float(endpoint_values.min())
                if not endpoint_values.empty
                else np.nan
            ),
            "maximum_value": (
                float(endpoint_values.max())
                if not endpoint_values.empty
                else np.nan
            ),

            # --------------------------------------------
            # Subject-level change from baseline
            # --------------------------------------------
            "mean_change_from_baseline": (
                float(change_values.mean())
                if not change_values.empty
                else np.nan
            ),
            "sd_change_from_baseline": (
                float(change_values.std(ddof=1))
                if len(change_values) > 1
                else np.nan
            ),
            "sem_change_from_baseline": (
                float(
                    change_values.std(ddof=1)
                    / np.sqrt(len(change_values))
                )
                if len(change_values) > 1
                else np.nan
            ),
            "median_change_from_baseline": (
                float(change_values.median())
                if not change_values.empty
                else np.nan
            ),
            "minimum_change_from_baseline": (
                float(change_values.min())
                if not change_values.empty
                else np.nan
            ),
            "maximum_change_from_baseline": (
                float(change_values.max())
                if not change_values.empty
                else np.nan
            ),
        })

    longitudinal_summary_df = pd.DataFrame(
        summary_rows
    )

    return longitudinal_summary_df

# =============================================================================
# LONGITUDINAL CHANNEL-LEVEL CHANGE FROM PREDOSE
# =============================================================================
def build_longitudinal_topomap_change(qeeg_results_by_recording, *, power_type, baseline_timepoint="PREDOSE", ddof=1):
    """Build cohort mean ± SD channel-level band-power change from PREDOSE."""

    if power_type == "absolute":
        result_key, value_key, unit = "absolute_power_result", "mean_absolute_power_by_channel_uv2", "µV²"
    elif power_type == "relative":
        result_key, value_key, unit = "relative_power_result", "mean_relative_power_by_channel_percent", "percentage points"
    else:
        raise ValueError("power_type must be 'absolute' or 'relative'.")

    # Index every logical recording by subject, eye state, and timepoint.
    recordings = {}
    for recording_id, result in qeeg_results_by_recording.items():
        subject, eye_state, timepoint = result.get("subject_id"), result.get("eye_state"), result.get("timepoint")
        if any(value is None for value in (subject, eye_state, timepoint)):
            raise ValueError(f"Recording '{recording_id}' is missing subject_id, eye_state, or timepoint.")
        key = (str(subject), str(eye_state), str(timepoint))
        if key in recordings: raise ValueError(f"Duplicate longitudinal recording key: {key}")
        recordings[key] = (str(recording_id), result)

    # Calculate each subject's channel-level change from the matching PREDOSE map.
    groups = {}
    for (subject, eye_state, timepoint), (recording_id, result) in recordings.items():
        baseline_key = (subject, eye_state, str(baseline_timepoint))
        if baseline_key not in recordings: raise ValueError(f"Missing {baseline_timepoint} recording for {(subject, eye_state)}.")

        baseline_id, baseline_result = recordings[baseline_key]
        current, baseline = result[result_key], baseline_result[result_key]
        current_values, baseline_values = np.asarray(current[value_key], float), np.asarray(baseline[value_key], float)
        band_names, ch_names = list(current["band_names"]), list(current["ch_names"])

        if band_names != list(baseline["band_names"]) or ch_names != list(baseline["ch_names"]):
            raise ValueError(f"Channel/band mismatch between '{recording_id}' and baseline '{baseline_id}'.")

        group = groups.setdefault((eye_state, timepoint), {
            "changes": [], "subject_ids": [], "recording_ids": [], "baseline_recording_ids": [],
            "band_names": band_names, "ch_names": ch_names, "template_epochs": result["epochs_clean"],
        })
        if group["band_names"] != band_names or group["ch_names"] != ch_names:
            raise ValueError(f"Inconsistent channel/band order in group {(eye_state, timepoint)}.")

        group["changes"].append(current_values - baseline_values)
        group["subject_ids"].append(subject)
        group["recording_ids"].append(recording_id)
        group["baseline_recording_ids"].append(baseline_id)

    # Cohort summary of the paired subject-level spatial changes.
    by_group = {}
    for key, group in groups.items():
        values = np.stack(group.pop("changes"), axis=0)
        group["n_subjects"] = values.shape[0]
        group["mean_change_by_channel"] = values.mean(axis=0)
        group["sd_change_by_channel"] = values.std(axis=0, ddof=ddof) if values.shape[0] > ddof else np.full(values.shape[1:], np.nan)
        by_group[key] = group

    return {"by_group": by_group, "power_type": power_type, "baseline_timepoint": baseline_timepoint, "unit": unit}



# =============================================================================
# PLOT LONGITUDINAL CHANGE-FROM-PREDOSE TOPOMAPS
# =============================================================================
def plot_longitudinal_topomap_change(change_results, *, eye_state, band, timepoint_order, band_alias=None,
                                     cmap="RdBu_r", vlim=None, vmax_percentile=None, outlines="head",
                                     extrapolate="head", sphere=(0.0, 0.0, 0.0, 0.110), sensors="k.",
                                     contours=6, n_cols=4, figsize=(12, 6.5), font_size=11, show=True):
    """Plot cohort mean channel-level change from PREDOSE for one band across timepoints."""

    by_group = change_results["by_group"]
    baseline = str(change_results["baseline_timepoint"])
    eye_state = str(eye_state)
    timepoints = [str(tp) for tp in timepoint_order if str(tp) != baseline and (eye_state, str(tp)) in by_group]

    if not timepoints:
        raise ValueError(f"No post-{baseline} groups found for eye_state='{eye_state}'.")

    # -------------------------------------------------------------------------
    # Retrieve the requested band and establish channel/sensor geometry.
    # -------------------------------------------------------------------------
    first = by_group[(eye_state, timepoints[0])]
    if band not in first["band_names"]:
        raise ValueError(f"Band '{band}' is unavailable. Available bands: {first['band_names']}")

    band_idx = first["band_names"].index(band)
    ch_names = list(first["ch_names"])
    epochs = first["template_epochs"]

    missing_channels = [ch for ch in ch_names if ch not in epochs.ch_names]
    if missing_channels:
        raise ValueError(f"Channels missing from template epochs: {missing_channels}")

    topo_info = mne.pick_info(epochs.info, [epochs.ch_names.index(ch) for ch in ch_names], copy=True)

    # -------------------------------------------------------------------------
    # Collect cohort mean change maps across the requested timepoints.
    # -------------------------------------------------------------------------
    values, n_subjects = [], []

    for tp in timepoints:
        group = by_group[(eye_state, tp)]

        if group["ch_names"] != ch_names or group["band_names"] != first["band_names"]:
            raise ValueError(f"Inconsistent channel/band order at {eye_state} {tp}.")

        values.append(np.asarray(group["mean_change_by_channel"], dtype=float)[:, band_idx])
        n_subjects.append(int(group["n_subjects"]))

    values = np.stack(values)

    if not np.isfinite(values).all():
        raise ValueError("Topomap change values contain NaN or infinite values.")

    # -------------------------------------------------------------------------
    # Use one symmetric zero-centered color scale across all timepoints.
    # -------------------------------------------------------------------------
    if vlim is None:
        vmax = np.percentile(np.abs(values), vmax_percentile) if vmax_percentile is not None else np.max(np.abs(values))
        vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
    elif np.isscalar(vlim):
        vmax = abs(float(vlim))
    else:
        if len(vlim) != 2:
            raise ValueError("vlim must contain two values.")
        vmax = max(abs(float(vlim[0])), abs(float(vlim[1])))

    vlim = (-vmax, vmax)

    # -------------------------------------------------------------------------
    # Plot all post-PREDOSE timepoints using the NeuShen Part 1 head geometry.
    # -------------------------------------------------------------------------
    n_rows = int(np.ceil(len(timepoints) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    image = None

    for i, tp in enumerate(timepoints):
        image, _ = mne.viz.plot_topomap(
            values[i], topo_info, axes=axes_flat[i], show=False, cmap=cmap, vlim=vlim,
            sensors=sensors, contours=contours, outlines=outlines,
            extrapolate=extrapolate, sphere=sphere,
        )
        axes_flat[i].set_title(f"{tp}\nN={n_subjects[i]}", fontsize=font_size, fontweight="bold", pad=8)

    for ax in axes_flat[len(timepoints):]:
        ax.axis("off")

    # -------------------------------------------------------------------------
    # Figure title and external shared colorbar.
    # -------------------------------------------------------------------------
    band_label = (band_alias or {}).get(band, band.capitalize())
    eye_label = {"EO": "Eyes Open", "EC": "Eyes Closed"}.get(eye_state, eye_state)
    power_label = "Absolute Power" if change_results["power_type"] == "absolute" else "Relative Band Power"

    fig.suptitle(
        f"{band_label} {power_label} Change from PREDOSE — {eye_label}",
        fontsize=font_size + 3, fontweight="bold", y=0.98,
    )

    fig.subplots_adjust(top=0.86, bottom=0.06, left=0.04, right=0.88, hspace=0.35, wspace=0.22)

    cbar_ax = fig.add_axes([0.91, 0.22, 0.018, 0.55])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label(
        f"Change from PREDOSE ({change_results['unit']})",
        fontsize=font_size, fontweight="bold",
    )
    cbar.ax.tick_params(labelsize=font_size - 1)

    if show:
        plt.show()

    return {
        "figure": fig,
        "axes": axes,
        "timepoints": timepoints,
        "n_subjects": n_subjects,
        "values": values,
        "vlim": vlim,
        "eye_state": eye_state,
        "band": band,
        "power_type": change_results["power_type"],
    }

# =============================================================================
# ID-UNLINKED LONGITUDINAL CHANGE DISTRIBUTIONS
# =============================================================================

def plot_qeeg_change_distribution(subject_level_change_df, *, eye_state, endpoint_type, endpoint, timepoint_order,
                                  endpoint_alias=None, figsize=(10, 5.5), font_size=11, show=True):
    """Plot ID-unlinked subject-level change-from-PREDOSE distributions."""

    required = {"subject_id", "eye_state", "timepoint", "endpoint_type", "endpoint", "change_from_baseline"}
    missing = required - set(subject_level_change_df.columns)
    if missing: raise KeyError(f"subject_level_change_df is missing required columns: {sorted(missing)}")

    # -------------------------------------------------------------------------
    # Select one qEEG endpoint and remove PREDOSE because its change is zero.
    # -------------------------------------------------------------------------
    timepoints = [str(tp) for tp in timepoint_order if str(tp) != "PREDOSE"]

    plot_df = subject_level_change_df.loc[
        subject_level_change_df["eye_state"].astype(str).eq(str(eye_state))
        & subject_level_change_df["endpoint_type"].astype(str).eq(str(endpoint_type))
        & subject_level_change_df["endpoint"].astype(str).eq(str(endpoint))
        & subject_level_change_df["timepoint"].astype(str).isin(timepoints)
    ].copy()

    if plot_df.empty: raise ValueError(f"No rows found for {eye_state} / {endpoint_type} / {endpoint}.")

    plot_df["change_from_baseline"] = pd.to_numeric(plot_df["change_from_baseline"], errors="coerce")
    plot_df = plot_df.dropna(subset=["change_from_baseline"])
    plot_df["timepoint"] = pd.Categorical(plot_df["timepoint"], categories=timepoints, ordered=True)

    # Each subject should contribute no more than one value per timepoint.
    if plot_df.duplicated(["subject_id", "timepoint"]).any():
        raise ValueError("Multiple values were found for the same subject and timepoint.")

    # -------------------------------------------------------------------------
    # Build an internal descriptive distribution summary.
    # -------------------------------------------------------------------------
    summary_df = (
        plot_df.groupby("timepoint", observed=True)["change_from_baseline"]
        .agg(N="count", Median="median", Minimum="min", Maximum="max",
             Q1=lambda x: x.quantile(0.25), Q3=lambda x: x.quantile(0.75))
        .reset_index()
    )

    # -------------------------------------------------------------------------
    # Plot unlinked individual observations over the distribution boxplot.
    # Subject IDs are never displayed and observations are not connected.
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=plot_df, x="timepoint", y="change_from_baseline", order=timepoints,
        width=0.45, showfliers=False, color="white", ax=ax,
    )

    sns.stripplot(
        data=plot_df, x="timepoint", y="change_from_baseline", order=timepoints,
        jitter=0.16, size=6, alpha=0.80, ax=ax,
    )

    # Zero represents no change from the subject-specific PREDOSE value.
    ax.axhline(0, linestyle="--", linewidth=1.4, alpha=0.7)

    # -------------------------------------------------------------------------
    # Labels and title.
    # -------------------------------------------------------------------------
    ylabel = {
        "absolute_power": "Change from PREDOSE (µV²)",
        "relative_power": "Change from PREDOSE (percentage points)",
        "spectral_ratio": "Change from PREDOSE (ratio units)",
    }.get(endpoint_type, "Change from PREDOSE")

    eye_label = {"EO": "Eyes Open", "EC": "Eyes Closed"}.get(str(eye_state), str(eye_state))
    endpoint_label = (endpoint_alias or {}).get(endpoint, str(endpoint).replace("_", " ").title())

    if endpoint_type == "absolute_power": metric_label = f"Absolute {endpoint_label} Power"
    elif endpoint_type == "relative_power": metric_label = f"Relative {endpoint_label} Power"
    else: metric_label = endpoint_label

    ax.set_xlabel("Timepoint", fontsize=font_size, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=font_size, fontweight="bold")
    ax.set_title(f"{metric_label} Change from PREDOSE — {eye_label}",
                 fontsize=font_size + 2, fontweight="bold", pad=12)

    # -------------------------------------------------------------------------
    # Add N below each timepoint.
    #
    # Explicitly setting the tick positions before the labels prevents the
    # Matplotlib FixedLocator/set_ticklabels warning.
    # -------------------------------------------------------------------------
    n_lookup = {str(row["timepoint"]): int(row["N"]) for _, row in summary_df.iterrows()}
    tick_positions = np.arange(len(timepoints))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{tp}\nN={n_lookup.get(tp, 0)}" for tp in timepoints], fontweight="bold")

    ax.tick_params(axis="y", labelsize=font_size)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if show: plt.show()

    return {"figure": fig, "axis": ax, "plot_df": plot_df, "summary_df": summary_df}

def plot_qeeg_subject_trajectories(
    subject_level_change_df,
    *,
    eye_state,
    band,
    power_type,
    timepoint_order,
    subject_id_col="subject_id",
    eye_state_col="eye_state",
    band_col="endpoint",
    power_type_col="endpoint_type",
    timepoint_col="timepoint",
    value_col="change_from_baseline",
    band_alias=None,
    eye_state_alias=None,
    subject_mask_map=None,
    subject_color_map=None,
    include_predose=True,
    figsize=(12.5, 6),
    linewidth=1.8,
    marker_size=5,
    alpha=0.95,
    legend_ncol=2,
    title=None,
    y_label=None,
    show=True,
):
    """
    Plot masked subject-level longitudinal qEEG trajectories.

    One connected line is shown per masked subject so within-subject temporal
    patterns can be followed without displaying the actual subject IDs.

    The input dataframe should contain subject-specific change-from-PREDOSE
    values produced by the longitudinal qEEG workflow.

    Notes
    -----
    - PREDOSE is retained by default so every trajectory starts at zero.
    - The same masked subject receives the same label and color across plots.
    - Missing timepoints are shown as gaps rather than connecting across
      unavailable observations.
    - This function changes only visualization; no qEEG values are recalculated.
    """

    # -------------------------------------------------------------------------
    # Validate required longitudinal columns
    # -------------------------------------------------------------------------
    required_cols = {
        subject_id_col, eye_state_col, band_col, power_type_col,
        timepoint_col, value_col,
    }
    missing_cols = sorted(required_cols - set(subject_level_change_df.columns))
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    # -------------------------------------------------------------------------
    # Create a stable masked-ID map using ALL subjects in the dataset.
    # This keeps Masked 01, Masked 02, etc. consistent across every figure.
    # -------------------------------------------------------------------------
    all_subject_ids = sorted(subject_level_change_df[subject_id_col].dropna().astype(str).unique())

    if subject_mask_map is None:
        subject_mask_map = {
            subject_id: f"Masked {i:02d}"
            for i, subject_id in enumerate(all_subject_ids, start=1)
        }
    else:
        subject_mask_map = dict(subject_mask_map)

        # Add any subjects not already represented in a supplied mapping.
        next_index = len(subject_mask_map) + 1
        for subject_id in all_subject_ids:
            if subject_id not in subject_mask_map:
                subject_mask_map[subject_id] = f"Masked {next_index:02d}"
                next_index += 1

    # -------------------------------------------------------------------------
    # Create a stable subject-color map across ALL figures.
    # Colors are assigned from the complete masked-subject list rather than
    # independently within each plot, preventing colors from changing when
    # individual subjects are missing from a particular feature/timepoint.
    # -------------------------------------------------------------------------
    all_masked_ids = [subject_mask_map[subject_id] for subject_id in all_subject_ids]

    if subject_color_map is None:
        cmap = plt.get_cmap("tab20")
        subject_color_map = {
            masked_id: cmap(i % cmap.N)
            for i, masked_id in enumerate(all_masked_ids)
        }
    else:
        subject_color_map = dict(subject_color_map)

    # -------------------------------------------------------------------------
    # Select the requested eye state, endpoint, and power type.
    # -------------------------------------------------------------------------
    plot_df = subject_level_change_df[
        (subject_level_change_df[eye_state_col] == eye_state)
        & (subject_level_change_df[band_col] == band)
        & (subject_level_change_df[power_type_col] == power_type)
    ].copy()

    if plot_df.empty:
        raise ValueError(
            f"No longitudinal data found for "
            f"eye_state={eye_state}, band={band}, power_type={power_type}."
        )

    # -------------------------------------------------------------------------
    # Restrict data to the requested timepoint order.
    # PREDOSE remains visible by default so every trajectory begins at zero.
    # -------------------------------------------------------------------------
    ordered_timepoints = list(timepoint_order)

    if not include_predose:
        ordered_timepoints = [tp for tp in ordered_timepoints if tp != "PREDOSE"]

    plot_df = plot_df[plot_df[timepoint_col].isin(ordered_timepoints)].copy()
    plot_df["_subject_id"] = plot_df[subject_id_col].astype(str)
    plot_df["masked_subject"] = plot_df["_subject_id"].map(subject_mask_map)

    # -------------------------------------------------------------------------
    # Count subjects contributing usable values at each timepoint.
    # These counts are displayed directly beneath the x-axis labels.
    # -------------------------------------------------------------------------
    timepoint_n = (
        plot_df.dropna(subset=[value_col])
        .groupby(timepoint_col)[subject_id_col]
        .nunique()
        .reindex(ordered_timepoints, fill_value=0)
    )

    # -------------------------------------------------------------------------
    # Build presentation labels.
    # -------------------------------------------------------------------------
    band_label = (
        band_alias.get(band, band.replace("_", " ").title())
        if band_alias else band.replace("_", " ").title()
    )

    eye_label = (
        eye_state_alias.get(eye_state, eye_state)
        if eye_state_alias
        else {"EO": "Eyes Open", "EC": "Eyes Closed"}.get(eye_state, eye_state)
    )

    power_label = {
        "absolute_power": "Absolute",
        "relative_power": "Relative",
    }.get(power_type, power_type.replace("_", " ").title())

    if title is None:
        title = f"{power_label} {band_label} Power Change from PREDOSE — {eye_label}"

    if y_label is None:
        y_label = (
            "Change from PREDOSE (µV²)"
            if power_type == "absolute_power"
            else "Change from PREDOSE (percentage points)"
        )

    # -------------------------------------------------------------------------
    # Create the subject-level trajectory figure.
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    x_positions = np.arange(len(ordered_timepoints))

    masked_id_order = [
        masked_id for masked_id in all_masked_ids
        if masked_id in set(plot_df["masked_subject"].dropna())
    ]

    # Plot one line per masked subject.
    # Reindexing each subject to the complete timepoint order inserts NaN where
    # observations are missing, so Matplotlib leaves a visible gap rather than
    # incorrectly connecting across an unavailable visit.
    for masked_id in masked_id_order:
        subject_df = plot_df[plot_df["masked_subject"] == masked_id]

        # Plot only the timepoints actually available for this subject.
        # Missing visits have no marker, but the available longitudinal observations
        # remain connected so the subject's overall temporal pattern can be followed.
        subject_df = subject_df[subject_df[value_col].notna()].copy()
        subject_df["_time_index"] = subject_df[timepoint_col].map(
            {timepoint: i for i, timepoint in enumerate(ordered_timepoints)}
        )
        subject_df = subject_df.sort_values("_time_index")

        ax.plot(
            subject_df["_time_index"],
            subject_df[value_col],
            marker="o", markersize=marker_size,
            linewidth=linewidth, alpha=alpha,
            color=subject_color_map[masked_id],
            label=masked_id,
        )


    # -------------------------------------------------------------------------
    # Reference line and presentation formatting.
    # -------------------------------------------------------------------------
    ax.axhline(0, linestyle="--", linewidth=1.5)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([
        f"{timepoint}\nN={int(timepoint_n.get(timepoint, 0))}"
        for timepoint in ordered_timepoints
    ])

    ax.set_xlabel("Timepoint")
    ax.set_ylabel(y_label)
    ax.set_title(title, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Keep the subject legend outside the plotting area so the trajectories
    # remain readable while still identifying each masked subject/color.
    ax.legend(
        title="Masked subject",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=legend_ncol,
        frameon=True,
    )

    fig.tight_layout()

    if show:
        plt.show()

    # Return only the masked plotting dataframe for downstream review.
    # The real-to-masked mapping is retained separately for internal use and
    # should NOT be included in sponsor-facing exports.
    masked_data = plot_df.drop(columns=["_subject_id"]).copy()

    return {
        "figure": fig,
        "axis": ax,
        "data": masked_data,
        "subject_mask_map": subject_mask_map,
        "subject_color_map": subject_color_map,
        "masked_id_order": masked_id_order,
        "timepoint_n": timepoint_n,
    }