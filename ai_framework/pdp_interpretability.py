"""
PDP-based interpretability utilities for external validation outputs.

This module contains the full PDP-based external interpretability workflow:

1. Build patient-level PDP signal/contribution tables from `all_results`.
2. Prepare shared plotting data for selected patients, models, and calibrations.
3. Plot patient-level PDP waterfall summaries.
4. Plot patient-level stacked PDP contribution summaries.
5. Plot cohort-level PDP feature contribution summaries.

Main expected inputs
--------------------
The construction function expects `all_results` to contain fold/trial-level
model records with stored external predictions, model-development PDP grids,
PDP average probability curves, and external validation feature data.

The plotting functions expect a long-format `pdp_signal_summary` DataFrame,
usually created by `build_external_pdp_signal_allocation_from_results(...)`,
with columns such as:

- patient_idx
- model
- calibration
- feature
- patient_value_mean
- allocated_prediction_signal
- pdp_signal_share
- patient_predicted_probability_mean

Terminology
-----------
The internal column name may remain `allocated_prediction_signal`, but the
user-facing interpretation is PDP-based probability contribution. These plots
are PDP-based response-curve summaries, not SHAP-style additive attributions.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Mapping, Sequence

import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt







# ---------------------------------------------------------------------
# PDP signal/allocation construction
# ---------------------------------------------------------------------

def _as_1d_float_array(x):
    """Convert input to a 1D float array."""
    return np.asarray(x, dtype=float).ravel()


def _interp_pdp_value(
    *,
    grid_values,
    pdp_values,
    patient_value,
    interpolation="linear",
    clip_to_grid=True,
):
    """
    Interpolate one PDP curve at one patient feature value.
    """

    if interpolation not in {"linear", "nearest"}:
        raise ValueError("interpolation must be either 'linear' or 'nearest'.")

    x = _as_1d_float_array(grid_values)
    y = _as_1d_float_array(pdp_values)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return np.nan, np.nan, "empty_pdp_curve"

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Drop duplicate grid values, keeping the first sorted occurrence.
    unique_x, unique_idx = np.unique(x, return_index=True)
    x = unique_x
    y = y[unique_idx]

    patient_value = float(patient_value)

    if patient_value < x.min() or patient_value > x.max():
        if clip_to_grid:
            lookup_value = float(np.clip(patient_value, x.min(), x.max()))
            status = "clipped_to_pdp_range"
        else:
            return np.nan, patient_value, "outside_pdp_range"
    else:
        lookup_value = patient_value
        status = "ok"

    if interpolation == "linear":
        pdp_probability = float(np.interp(lookup_value, x, y))
    else:
        nearest_i = int(np.argmin(np.abs(x - lookup_value)))
        pdp_probability = float(y[nearest_i])

    return pdp_probability, lookup_value, status


def _detect_real_model_names_from_results(all_results):
    """
    Detect real model keys from all_results.

    Skips common summary/meta keys and any entries that are not non-empty lists.
    """

    skip_keys = {
        "summary",
        "external_shap_summary",
        "progress_log",
        "metadata",
        "meta",
    }

    model_names = []

    for key, value in all_results.items():
        if key in skip_keys:
            continue
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            model_names.append(key)

    return model_names


def _prediction_key_from_calibration(calibration):
    """
    Map project-style calibration names to stored external prediction keys.
    """

    mapping = {
        "uncalib": "y_external_scores",
        "raw": "y_external_scores",
        "platt": "calib_external_predictions_platt",
        "beta": "calib_external_predictions_beta",
    }

    if calibration not in mapping:
        raise ValueError(
            f"Unknown calibration={calibration!r}. "
            f"Expected one of {list(mapping.keys())}."
        )

    return mapping[calibration]


def _build_external_pdp_signal_allocation_single(
    *,
    all_results,
    model_name,
    calibration,
    interpolation="linear",
    clip_to_grid=True,
    rescale_to_prediction=True,
    pdp_feature_names_key="pdp_feature_names",
    pdp_grid_key="pdp_grid_values_test",
    pdp_average_key="pdp_average_values_test",
    external_feature_names_key="external_shap_feature_names",
    external_data_key="external_shap_data",
    external_idx_key="external_shap_idx",
):
    """
    Build PDP signal allocation for one model and one calibration.
    """

    prediction_key = _prediction_key_from_calibration(calibration)

    records = all_results[model_name]
    detail_rows = []

    for record_i, record in enumerate(records):
        required_keys = [
            pdp_feature_names_key,
            pdp_grid_key,
            pdp_average_key,
            external_feature_names_key,
            external_data_key,
            external_idx_key,
            prediction_key,
        ]

        missing = [k for k in required_keys if k not in record]
        if missing:
            continue

        pdp_feature_names = list(record[pdp_feature_names_key])
        external_feature_names = list(record[external_feature_names_key])

        pdp_grid_values = np.asarray(record[pdp_grid_key], dtype=float)
        pdp_average_values = np.asarray(record[pdp_average_key], dtype=float)
        external_data = np.asarray(record[external_data_key], dtype=float)
        external_idx = np.asarray(record[external_idx_key])
        predictions = np.asarray(record[prediction_key], dtype=float)

        n_external = min(len(external_idx), external_data.shape[0], len(predictions))

        # Use self-contained feature intersection from this model/fold record.
        record_features = [
            f for f in pdp_feature_names
            if f in external_feature_names
        ]

        for external_pos in range(n_external):
            patient_idx = external_idx[external_pos]
            patient_prediction = float(predictions[external_pos])

            for feature in record_features:
                pdp_feature_i = pdp_feature_names.index(feature)
                external_feature_i = external_feature_names.index(feature)

                patient_value = float(external_data[external_pos, external_feature_i])

                pdp_probability, lookup_value, status = _interp_pdp_value(
                    grid_values=pdp_grid_values[pdp_feature_i],
                    pdp_values=pdp_average_values[pdp_feature_i],
                    patient_value=patient_value,
                    interpolation=interpolation,
                    clip_to_grid=clip_to_grid,
                )

                detail_rows.append(
                    {
                        "model": model_name,
                        "calibration": calibration,
                        "patient_idx": patient_idx,
                        "record_i": record_i,
                        "trial": record.get("trial", np.nan),
                        "outer_fold": record.get("outer_fold", np.nan),
                        "feature": feature,
                        "patient_value": patient_value,
                        "lookup_value": lookup_value,
                        "pdp_probability": pdp_probability,
                        "patient_predicted_probability": patient_prediction,
                        "status": status,
                    }
                )

    detail_df = pd.DataFrame(detail_rows)

    if detail_df.empty:
        summary_df = pd.DataFrame()
        return summary_df, detail_df

    valid_detail = detail_df.loc[
        detail_df["pdp_probability"].notna()
        & detail_df["patient_predicted_probability"].notna()
    ].copy()

    if valid_detail.empty:
        summary_df = pd.DataFrame()
        return summary_df, detail_df

    summary_df = (
        valid_detail
        .groupby(["model", "calibration", "patient_idx", "feature"], as_index=False)
        .agg(
            patient_value_mean=("patient_value", "mean"),
            patient_value_std=("patient_value", "std"),
            pdp_probability_mean=("pdp_probability", "mean"),
            pdp_probability_std=("pdp_probability", "std"),
            pdp_probability_min=("pdp_probability", "min"),
            pdp_probability_max=("pdp_probability", "max"),
            patient_predicted_probability_mean=("patient_predicted_probability", "mean"),
            patient_predicted_probability_std=("patient_predicted_probability", "std"),
            n_records=("pdp_probability", "size"),
            n_clipped=("status", lambda s: int((s == "clipped_to_pdp_range").sum())),
        )
    )

    # Total PDP signal per model/calibration/patient.
    summary_df["total_pdp_signal"] = (
        summary_df
        .groupby(["model", "calibration", "patient_idx"])["pdp_probability_mean"]
        .transform("sum")
    )

    summary_df["pdp_signal_share"] = np.where(
        summary_df["total_pdp_signal"] > 0,
        summary_df["pdp_probability_mean"] / summary_df["total_pdp_signal"],
        np.nan,
    )

    # One final prediction per model/calibration/patient.
    summary_df["allocated_prediction_signal_total"] = (
        summary_df
        .groupby(["model", "calibration", "patient_idx"])["patient_predicted_probability_mean"]
        .transform("mean")
    )

    if rescale_to_prediction:
        summary_df["allocated_prediction_signal"] = (
            summary_df["pdp_signal_share"]
            * summary_df["allocated_prediction_signal_total"]
        )
    else:
        summary_df["allocated_prediction_signal"] = np.nan

    summary_df = summary_df.sort_values(
        ["model", "calibration", "patient_idx", "pdp_signal_share", "feature"],
        ascending=[True, True, True, False, True],
        na_position="last",
    ).reset_index(drop=True)

    return summary_df, detail_df


def build_external_pdp_signal_allocation_from_results(
    *,
    all_results,
    model_name=None,
    calibrations=("uncalib", "beta"),
    interpolation="linear",
    clip_to_grid=True,
    rescale_to_prediction=True,
    include_detail=True,
    return_long=True,
    warn_on_skip=False,
):
    """
    Build raw PDP-derived signal allocation tables for the external validation dataset.

    This function uses model-development PDP curves stored in all_results and maps
    every external validation patient's observed model-ready feature values onto
    those PDP curves. For each patient, feature-level PDP probabilities are treated
    as raw average-response signal weights. These weights are normalized across
    features and optionally rescaled to the patient's final external predicted
    probability.

    The output is not a SHAP-style additive attribution. It is a PDP-derived
    allocation of each patient's external predicted probability using development
    PDP curves as feature-level signal weights.

    Parameters
    ----------
    all_results : dict
        Dictionary keyed by model name. Each model maps to fold/trial result records.

    model_name : str, list[str], or None
        Model(s) to include. If None, all real model keys are detected.

    calibrations : list[str] or tuple[str]
        Calibration outputs to include. Common options are:
            "uncalib"
            "platt"
            "beta"

    interpolation : {"linear", "nearest"}
        How to read PDP curves at external patient feature values.

    clip_to_grid : bool
        If True, values outside a PDP grid are clipped to the closest grid endpoint.

    rescale_to_prediction : bool
        If True, PDP signal shares are multiplied by the patient's external
        predicted probability, so allocated_prediction_signal sums to the
        patient prediction.

    include_detail : bool
        If True, include fold/trial-level detail outputs.

    return_long : bool
        If True, include concatenated summary_long and detail_long tables.

    warn_on_skip : bool
        If True, print messages when model/calibration combinations are skipped.

    Returns
    -------
    dict
        Dictionary with:
            by_model[model][calibration]["summary"]
            by_model[model][calibration]["detail"]
            summary_long
            detail_long
    """

    if model_name is None:
        model_names = _detect_real_model_names_from_results(all_results)
    elif isinstance(model_name, str):
        model_names = [model_name]
    else:
        model_names = list(model_name)

    if isinstance(calibrations, str):
        calibrations = [calibrations]
    else:
        calibrations = list(calibrations)

    outputs = {
        "by_model": {},
    }

    summary_tables = []
    detail_tables = []

    for current_model in model_names:
        if current_model not in all_results:
            if warn_on_skip:
                print(f"[skip] model {current_model!r} not found in all_results.")
            continue

        outputs["by_model"][current_model] = {}

        for calibration in calibrations:
            try:
                summary_df, detail_df = _build_external_pdp_signal_allocation_single(
                    all_results=all_results,
                    model_name=current_model,
                    calibration=calibration,
                    interpolation=interpolation,
                    clip_to_grid=clip_to_grid,
                    rescale_to_prediction=rescale_to_prediction,
                )
            except Exception as exc:
                if warn_on_skip:
                    print(
                        f"[skip] model={current_model!r}, "
                        f"calibration={calibration!r}: {exc}"
                    )
                continue

            outputs["by_model"][current_model][calibration] = {
                "summary": summary_df,
            }

            if include_detail:
                outputs["by_model"][current_model][calibration]["detail"] = detail_df

            if not summary_df.empty:
                summary_tables.append(summary_df)

            if include_detail and not detail_df.empty:
                detail_tables.append(detail_df)

    if return_long:
        outputs["summary_long"] = (
            pd.concat(summary_tables, ignore_index=True)
            if summary_tables
            else pd.DataFrame()
        )

        outputs["detail_long"] = (
            pd.concat(detail_tables, ignore_index=True)
            if detail_tables
            else pd.DataFrame()
        )

    return outputs

# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _as_list_or_all(
    value: Any,
    all_values: list[Any],
) -> list[Any]:
    """
    Convert a scalar/list/None input into a list.

    If value is None, return all_values.
    """
    if value is None:
        return list(all_values)

    if isinstance(value, (list, tuple, set, pd.Index, np.ndarray)):
        return list(value)

    return [value]


def _kwargs_for_function(
    func: Callable[..., Any],
    values: dict[str, Any],
) -> dict[str, Any]:
    """
    Return only the key-value pairs from values that are accepted by func.
    """
    valid_keys = set(inspect.signature(func).parameters)

    return {
        key: value
        for key, value in values.items()
        if key in valid_keys
    }


def _resolve_feature_colors(
    features: list[str],
    feature_colors: Mapping[str, str] | None = None,
    *,
    cmap_name: str = "Set2",
) -> dict[str, Any]:
    """
    Resolve a feature-to-color mapping.

    User-provided feature colors are used first. Missing features are assigned
    colors from the selected matplotlib colormap.
    """
    if feature_colors is None:
        cmap = plt.get_cmap(cmap_name)
        return {
            feature: cmap(i % cmap.N)
            for i, feature in enumerate(features)
        }

    resolved = dict(feature_colors)
    missing = [feature for feature in features if feature not in resolved]

    if missing:
        cmap = plt.get_cmap(cmap_name)
        start_i = len(resolved)

        for offset, feature in enumerate(missing):
            resolved[feature] = cmap((start_i + offset) % cmap.N)

    return resolved


def _resolve_feature_order_from_long(
    plot_df: pd.DataFrame,
    *,
    feature_col: str,
    value_col: str,
    feature_order: Literal[
        "mean_allocation",
        "total_allocation",
        "mean_contribution",
        "total_contribution",
        "original",
    ] | Sequence[str],
) -> list[str]:
    """
    Resolve a stable feature order from a long-format PDP allocation table.

    This is shared by patient-level stacked plots and cohort-level summaries.
    """
    if isinstance(feature_order, str):
        if feature_order in {"mean_allocation", "mean_contribution"}:
            return (
                plot_df.groupby(feature_col, as_index=True)[value_col]
                .mean()
                .sort_values(ascending=False)
                .index
                .tolist()
            )

        if feature_order in {"total_allocation", "total_contribution"}:
            return (
                plot_df.groupby(feature_col, as_index=True)[value_col]
                .sum()
                .sort_values(ascending=False)
                .index
                .tolist()
            )

        if feature_order == "original":
            return plot_df[feature_col].drop_duplicates().tolist()

        raise ValueError(
            "feature_order must be one of 'mean_allocation', "
            "'total_allocation', 'mean_contribution', 'total_contribution', "
            "'original', or a sequence of feature names."
        )

    provided = list(feature_order)
    observed = plot_df[feature_col].drop_duplicates().tolist()

    return provided + [feature for feature in observed if feature not in provided]


def _prepare_pdp_allocation_plot_data_single_model(
    pdp_signal_summary: pd.DataFrame,
    *,
    model_name: str,
    patient_idx: int | str | list[int | str] | None = None,
    calibration: str = "beta",
    feature_col: str = "feature",
    value_col: str = "allocated_prediction_signal",
    prediction_col: str = "patient_predicted_probability_mean",
    patient_value_col: str = "patient_value_mean",

    # Patient selection
    min_prediction: float | None = None,
    max_prediction: float | None = None,
    top_n: int | None = None,
    patient_sort_ascending: bool = False,

    # Feature ordering and colors
    feature_order: Literal[
        "mean_allocation",
        "total_allocation",
        "mean_contribution",
        "total_contribution",
        "original",
    ] | Sequence[str] = "mean_allocation",
    feature_colors: Mapping[str, str] | None = None,
    cmap_name: str = "Set2",
) -> dict[str, Any]:
    """
    Prepare shared PDP allocation plotting data for one model.

    This function centralizes the logic used by the patient-level stack plot and
    the cohort-level contribution plot:

    - model filtering
    - calibration filtering
    - patient_idx handling
    - prediction-range filtering
    - top_n selection
    - patient prediction table
    - feature ordering
    - feature colors
    - patient x feature wide table
    - patient x feature-value wide table

    Each patient’s observed feature values are mapped onto the model-development
    PDP curves. The PDP probability at each observed value is used as a
    feature-level response signal. These signals are normalized and rescaled so
    their cumulative allocation sums to the patient’s predicted probability.
    """

    d = pdp_signal_summary.copy()

    required_cols = {
        "patient_idx",
        "model",
        "calibration",
        feature_col,
        value_col,
        prediction_col,
        patient_value_col,
    }
    missing_cols = required_cols - set(d.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    d = d[
        (d["model"] == model_name)
        & (d["calibration"] == calibration)
    ].copy()

    if d.empty:
        raise ValueError(
            f"No rows found for model_name={model_name}, calibration={calibration}."
        )

    all_patients = d["patient_idx"].drop_duplicates().tolist()
    patient_list = _as_list_or_all(patient_idx, all_patients)

    d = d[d["patient_idx"].isin(patient_list)].copy()

    if d.empty:
        raise ValueError(
            f"No rows found after patient filtering for model_name={model_name}."
        )

    patient_predictions = (
        d.groupby("patient_idx", as_index=False)[prediction_col]
        .first()
        .rename(columns={prediction_col: "_prediction"})
    )

    if min_prediction is not None:
        patient_predictions = patient_predictions[
            patient_predictions["_prediction"] >= min_prediction
        ].copy()

    if max_prediction is not None:
        patient_predictions = patient_predictions[
            patient_predictions["_prediction"] <= max_prediction
        ].copy()

    patient_predictions = patient_predictions.sort_values(
        "_prediction",
        ascending=patient_sort_ascending,
    ).reset_index(drop=True)

    if top_n is not None:
        patient_predictions = patient_predictions.head(top_n).copy()

    selected_patients = patient_predictions["patient_idx"].tolist()

    if not selected_patients:
        raise ValueError(
            "No patients remain after prediction-range and top_n filtering."
        )

    d = d[d["patient_idx"].isin(selected_patients)].copy()

    plot_df = d.merge(
        patient_predictions,
        on="patient_idx",
        how="left",
    )

    features = _resolve_feature_order_from_long(
        plot_df,
        feature_col=feature_col,
        value_col=value_col,
        feature_order=feature_order,
    )

    colors = _resolve_feature_colors(
        features,
        feature_colors=feature_colors,
        cmap_name=cmap_name,
    )

    wide_df = (
        plot_df.pivot_table(
            index="patient_idx",
            columns=feature_col,
            values=value_col,
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=selected_patients)
        .reindex(columns=features, fill_value=0.0)
    )

    feature_values_wide = (
        plot_df.pivot_table(
            index="patient_idx",
            columns=feature_col,
            values=patient_value_col,
            aggfunc="first",
        )
        .reindex(index=selected_patients)
        .reindex(columns=features)
    )

    pred_map = patient_predictions.set_index("patient_idx")["_prediction"].to_dict()

    return {
        "plot_df": plot_df,
        "wide_df": wide_df,
        "feature_values_wide": feature_values_wide,
        "patient_predictions": patient_predictions,
        "selected_patients": selected_patients,
        "feature_order": features,
        "feature_colors": colors,
        "pred_map": pred_map,
        "model": model_name,
        "calibration": calibration,
        "min_prediction": min_prediction,
        "max_prediction": max_prediction,
        "top_n": top_n,
    }


def _format_probability_value(
    value: float,
    *,
    signed: bool = False,
    fixed_decimals: int = 2,
    small_decimals: int = 4,
    mid_small_decimals: int = 3,
    scientific_decimals: int = 1,
) -> str:
    """
    Format probability-like values for plot labels.

    This avoids misleading labels near 0 or 1, where fixed 2-decimal
    formatting turns values like 0.0042 into 0.00 and 0.9999 into 1.00.
    """

    if value is None or pd.isna(value):
        return "NA"

    value = float(value)
    abs_value = abs(value)

    if abs_value == 0:
        fmt = f"{{value:{'+' if signed else ''}.{fixed_decimals}f}}"
    elif abs_value < 1e-3:
        fmt = f"{{value:{'+' if signed else ''}.{scientific_decimals}e}}"
    elif abs_value < 1e-2:
        fmt = f"{{value:{'+' if signed else ''}.{small_decimals}f}}"
    elif abs_value < 1e-1:
        fmt = f"{{value:{'+' if signed else ''}.{mid_small_decimals}f}}"
    elif 0.99 < abs_value < 1.0:
        fmt = f"{{value:{'+' if signed else ''}.{small_decimals}f}}"
    else:
        fmt = f"{{value:{'+' if signed else ''}.{fixed_decimals}f}}"

    return fmt.format(value=value)


# ---------------------------------------------------------------------
# Cohort-level PDP summary helper
# ---------------------------------------------------------------------

def _prepare_cohort_pdp_feature_contribution_summary(
    pdp_signal_summary: pd.DataFrame,
    *,
    model_name: str,
    calibration: str = "beta",
    feature_col: str = "feature",
    value_col: str = "allocated_prediction_signal",
    prediction_col: str = "patient_predicted_probability_mean",
    patient_value_col: str = "patient_value_mean",

    # Patient selection
    patient_idx: int | str | list[int | str] | None = None,
    min_prediction: float | None = None,
    max_prediction: float | None = None,
    top_n: int | None = None,
    patient_sort_ascending: bool = False,

    # Feature ordering
    feature_order: Literal[
        "mean_contribution",
        "total_contribution",
        "mean_allocation",
        "total_allocation",
        "original",
    ] | Sequence[str] = "mean_contribution",
) -> dict[str, Any]:
    """
    Prepare a cohort-level PDP feature contribution summary for one model.

    This helper centralizes the cohort summary logic so the single-cohort plot
    and grouped cohort-comparison plot use the same patient filtering,
    feature ordering, and aggregation.
    """

    prepared = _prepare_pdp_allocation_plot_data_single_model(
        pdp_signal_summary,
        model_name=model_name,
        patient_idx=patient_idx,
        calibration=calibration,
        feature_col=feature_col,
        value_col=value_col,
        prediction_col=prediction_col,
        patient_value_col=patient_value_col,
        min_prediction=min_prediction,
        max_prediction=max_prediction,
        top_n=top_n,
        patient_sort_ascending=patient_sort_ascending,
        feature_order=feature_order,
        feature_colors=None,
        cmap_name="Set2",
    )

    plot_df_long = prepared["plot_df"]

    summary_df = (
        plot_df_long.groupby(feature_col, as_index=False)
        .agg(
            n_patients=("patient_idx", "nunique"),
            mean_contribution=(value_col, "mean"),
            sd_contribution=(value_col, "std"),
            median_contribution=(value_col, "median"),
            total_contribution=(value_col, "sum"),
            mean_feature_value=(patient_value_col, "mean"),
            sd_feature_value=(patient_value_col, "std"),
            median_feature_value=(patient_value_col, "median"),
        )
    )

    summary_df["sd_contribution"] = summary_df["sd_contribution"].fillna(0.0)
    summary_df["sd_feature_value"] = summary_df["sd_feature_value"].fillna(0.0)

    features = prepared["feature_order"]

    summary_df[feature_col] = pd.Categorical(
        summary_df[feature_col],
        categories=features,
        ordered=True,
    )
    summary_df = summary_df.sort_values(feature_col)
    summary_df[feature_col] = summary_df[feature_col].astype(str)
    summary_df = summary_df.reset_index(drop=True)

    return {
        "summary_df": summary_df,
        "prepared": prepared,
        "feature_order": features,
        "model": model_name,
        "calibration": calibration,
        "min_prediction": min_prediction,
        "max_prediction": max_prediction,
        "top_n": top_n,
    }


# ---------------------------------------------------------------------
# Patient-level waterfall plot
# ---------------------------------------------------------------------

def _plot_patient_pdp_allocation_waterfall_single(
    pdp_signal_summary: pd.DataFrame,
    *,
    patient_idx: int | str,
    model_name: str,
    calibration: str = "beta",
    model_alias: Mapping[str, str] | None = None,
    feature_col: str = "feature",
    patient_value_col: str = "patient_value_mean",
    value_col: str = "allocated_prediction_signal",
    share_col: str = "pdp_signal_share",
    prediction_col: str = "patient_predicted_probability_mean",
    sort_by: str = "allocated_prediction_signal",

    # Ordering
    display_ascending: bool = False,
    cumulative_ascending: bool = True,

    # Arrow bars
    bar_color: str = "#536D88",
    alpha: float = 1.0,
    bar_height: float = 0.5,
    y_gap: float = 0.0,
    head_length: float = 0.08,

    # Figure
    figsize: tuple[float, float] = (10, 4.5),
    font_size: int = 12,
    title_suffix: str | None = "PDP-based feature allocation of predicted probability",
    xlabel: str = "Predicted probability",
    title_pad: float = 30,

    # Final prediction reference
    show_prediction_vline: bool = True,
    show_prediction_label: bool = True,
    prediction_label: str = "Predicted probability",
    prediction_line_color: str = "#000000",
    prediction_linewidth: float = 1.5,
    prediction_linestyle: str = "--",
    prediction_label_y: float = 1.03,
    prediction_label_fontsize: int | None = None,

    # Bar value labels
    show_value_labels: bool = True,
    label_mode: Literal["value", "share", "none"] = "value",
    label_position: Literal["inside", "outside", "auto"] = "auto",
    value_decimals: int = 2,
    share_decimals: int = 1,
    show_value_sign: bool = True,
    value_label_fontsize: int | None = None,
    label_color_inside: str = "white",
    label_color_outside: str = "black",
    label_pad: float = 0.01,
    adaptive_probability_labels: bool = True,

    # Axis
    xlim: tuple[float, float] | None = None,
    x_padding: float = 0.10,
    top_ylim_pad: float = 0.35,
    bottom_ylim_pad: float = 0.15,

    # Grid
    show_grid: bool = True,
    grid_color: str = "#cccccc",
    grid_linewidth: float = 1.0,
    grid_alpha: float = 1.0,
    grid_linestyle: str = "-",

    show: bool = True,
) -> dict[str, Any]:
    """
    Plot one patient-level PDP allocation waterfall for one model.

    Each patient’s observed feature values are mapped onto the model-development
    PDP curves. The PDP probability at each observed value is used as a
    feature-level response signal. These signals are normalized and rescaled so
    their cumulative allocation sums to the patient’s predicted probability.

    Each feature is drawn as a right-facing arrow segment. The segments accumulate
    to the patient's predicted probability.

    Display order and cumulative order are separated:
    - display_ascending=False shows strongest features at the top.
    - cumulative_ascending=True builds from smallest to largest, so the largest
      feature tends to land closest to the final prediction line.

    This is a PDP-based response-curve summary, not a SHAP-style additive
    attribution.
    """

    if model_alias is None:
        model_alias = {}

    if label_mode not in {"value", "share", "none"}:
        raise ValueError("label_mode must be one of: 'value', 'share', or 'none'.")

    if label_position not in {"inside", "outside", "auto"}:
        raise ValueError("label_position must be one of: 'inside', 'outside', or 'auto'.")

    # Use the shared PDP allocation preparation backbone.
    prepared = _prepare_pdp_allocation_plot_data_single_model(
        pdp_signal_summary,
        model_name=model_name,
        patient_idx=patient_idx,
        calibration=calibration,
        feature_col=feature_col,
        value_col=value_col,
        prediction_col=prediction_col,
        patient_value_col=patient_value_col,
        feature_order=sort_by if sort_by in {
            "mean_allocation",
            "total_allocation",
            "mean_contribution",
            "total_contribution",
            "original",
        } else "original",
    )

    d = prepared["plot_df"].copy()

    d = d[
        (d["patient_idx"] == patient_idx)
        & (d["model"] == model_name)
        & (d["calibration"] == calibration)
    ].copy()

    if d.empty:
        raise ValueError(
            f"No rows found for patient_idx={patient_idx}, "
            f"model_name={model_name}, calibration={calibration}."
        )

    required_cols = {
        feature_col,
        patient_value_col,
        value_col,
        share_col,
        prediction_col,
        sort_by,
    }
    missing_cols = required_cols - set(d.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    pred = float(d[prediction_col].iloc[0])

    # ------------------------------------------------------------------
    # 1. Compute cumulative positions using cumulative order.
    # ------------------------------------------------------------------
    accum_df = d.sort_values(
        sort_by,
        ascending=cumulative_ascending,
    ).reset_index(drop=True)

    accum_df["segment_start"] = (
        accum_df[value_col].cumsum().shift(fill_value=0.0)
    )
    accum_df["segment_end"] = (
        accum_df["segment_start"] + accum_df[value_col]
    )

    merge_cols = ["model", "calibration", "patient_idx", feature_col]

    d = d.merge(
        accum_df[merge_cols + ["segment_start", "segment_end"]],
        on=merge_cols,
        how="left",
    )

    # ------------------------------------------------------------------
    # 2. Display order is separate from cumulative order.
    # ------------------------------------------------------------------
    d = d.sort_values(
        sort_by,
        ascending=display_ascending,
    ).reset_index(drop=True)

    def _fmt_feature_value(v: Any) -> str:
        if pd.isna(v):
            return "NA"
        return f"{v:.2f}"

    d["feature_label"] = [
        f"{_fmt_feature_value(v)} = {f}"
        for v, f in zip(d[patient_value_col], d[feature_col])
    ]

    # Reverse for barh coordinates so first row appears at top.
    plot_df = d.iloc[::-1].reset_index(drop=True)

    if value_label_fontsize is None:
        value_label_fontsize = max(font_size - 2, 8)

    if prediction_label_fontsize is None:
        prediction_label_fontsize = font_size

    fig, ax = plt.subplots(figsize=figsize)

    y_step = bar_height + y_gap
    y = np.arange(len(plot_df)) * y_step

    xmax_needed = max(float(plot_df["segment_end"].max()), pred)

    if xmax_needed <= 0:
        xmax_needed = 1e-8

    if xlim is None:
        ax.set_xlim(0, xmax_needed * (1 + x_padding))
    else:
        ax.set_xlim(*xlim)

    ax.set_ylim(
        y.min() - bar_height / 2 - bottom_ylim_pad,
        y.max() + bar_height / 2 + top_ylim_pad,
    )

    if show_grid:
        ax.grid(
            True,
            axis="both",
            color=grid_color,
            linewidth=grid_linewidth,
            alpha=grid_alpha,
            linestyle=grid_linestyle,
            zorder=0,
        )
        ax.set_axisbelow(True)

    # ------------------------------------------------------------------
    # Arrow geometry, matching the SHAP-style waterfall scaling.
    # ------------------------------------------------------------------
    fig.canvas.draw()

    xlen = ax.get_xlim()[1] - ax.get_xlim()[0]
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    bbox_to_xscale = xlen / max(bbox.width, 1e-6)
    hl_scaled = bbox_to_xscale * head_length

    for yi, (_, row) in zip(y, plot_df.iterrows()):
        x0 = float(row["segment_start"])
        x1 = float(row["segment_end"])
        width = x1 - x0

        if width <= 0 or not np.isfinite(width):
            continue

        local_head = min(width * 0.80, hl_scaled)

        ax.arrow(
            x0,
            yi,
            width,
            0,
            head_length=local_head,
            head_width=bar_height,
            width=bar_height,
            length_includes_head=True,
            color=bar_color,
            alpha=alpha,
            linewidth=0,
            zorder=3,
        )

    # ------------------------------------------------------------------
    # Value labels.
    # ------------------------------------------------------------------
    if show_value_labels and label_mode != "none":
        x0_lim, x1_lim = ax.get_xlim()
        x_range = x1_lim - x0_lim
        label_pad_data = label_pad * x_range

        for yi, (_, row) in zip(y, plot_df.iterrows()):
            x0 = float(row["segment_start"])
            x1 = float(row["segment_end"])
            width = x1 - x0
            xc = x0 + width / 2.0

            if width <= 0 or not np.isfinite(width):
                continue


            if label_mode == "value":
                if adaptive_probability_labels:
                    label_text = _format_probability_value(
                        width,
                        signed=show_value_sign,
                        fixed_decimals=value_decimals,
                    )
                else:
                    if show_value_sign:
                        label_text = f"{width:+.{value_decimals}f}"
                    else:
                        label_text = f"{width:.{value_decimals}f}"
            elif label_mode == "share":
                label_text = f"{100.0 * float(row[share_col]):.{share_decimals}f}%"
            else:
                label_text = ""


            if not label_text:
                continue

            if label_position == "inside":
                use_inside = True
            elif label_position == "outside":
                use_inside = False
            else:
                use_inside = width >= 0.12 * x_range

            if use_inside:
                ax.text(
                    xc,
                    yi,
                    label_text,
                    ha="center",
                    va="center",
                    fontsize=value_label_fontsize,
                    fontweight="bold",
                    color=label_color_inside,
                    zorder=5,
                )
            else:
                ax.text(
                    x1 + label_pad_data,
                    yi,
                    label_text,
                    ha="left",
                    va="center",
                    fontsize=value_label_fontsize,
                    fontweight="bold",
                    color=label_color_outside,
                    zorder=5,
                    clip_on=False,
                )

    # ------------------------------------------------------------------
    # Prediction reference line and label.
    # ------------------------------------------------------------------
    if show_prediction_vline:
        ax.axvline(
            pred,
            color=prediction_line_color,
            linewidth=prediction_linewidth,
            linestyle=prediction_linestyle,
            zorder=2,
        )

    if show_prediction_label:

        if adaptive_probability_labels:
            pred_text = _format_probability_value(
                pred,
                signed=False,
                fixed_decimals=value_decimals,
            )
        else:
            pred_text = f"{pred:.{value_decimals}f}"

        ax.text(
            pred,
            prediction_label_y,
            f"{prediction_label} = {pred_text}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=prediction_label_fontsize,
            fontweight="bold",
            color=prediction_line_color,
            clip_on=False,
        )



    # ------------------------------------------------------------------
    # Titles and labels.
    # ------------------------------------------------------------------
    model_label = model_alias.get(model_name, model_name)
    final_title = f"{model_label}: Patient {patient_idx}"

    if title_suffix is not None:
        final_title = f"{final_title}\n{title_suffix}"

    ax.set_title(
        final_title,
        fontsize=font_size + 2,
        fontweight="bold",
        pad=title_pad,
    )

    ax.set_xlabel(
        xlabel,
        fontsize=font_size,
        fontweight="bold",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        plot_df["feature_label"].tolist(),
        fontsize=font_size,
        fontweight="bold",
    )

    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()

    if show:
        plt.show()

    return {
        "fig": fig,
        "ax": ax,
        "plot_df": d,
        "prepared": prepared,
        "predicted_probability": pred,
        "patient_idx": patient_idx,
        "model": model_name,
        "calibration": calibration,
        "display_order": "ascending" if display_ascending else "descending",
        "cumulative_order": "ascending" if cumulative_ascending else "descending",
    }


def plot_patient_pdp_allocation_waterfall(
    pdp_signal_summary: pd.DataFrame,
    *,
    patient_idx: int | str | list[int | str] | None = None,
    model_name: str | list[str] | None = None,
    calibration: str = "beta",
    model_alias: Mapping[str, str] | None = None,
    feature_col: str = "feature",
    patient_value_col: str = "patient_value_mean",
    value_col: str = "allocated_prediction_signal",
    share_col: str = "pdp_signal_share",
    prediction_col: str = "patient_predicted_probability_mean",
    sort_by: str = "allocated_prediction_signal",

    # Ordering
    display_ascending: bool = False,
    cumulative_ascending: bool = True,

    # Arrow bars
    bar_color: str = "#536D88",
    alpha: float = 1.0,
    bar_height: float = 0.5,
    y_gap: float = 0.0,
    head_length: float = 0.08,

    # Figure
    figsize: tuple[float, float] = (10, 4.5),
    font_size: int = 12,
    title_suffix: str | None = "PDP-based feature allocation of predicted probability",
    xlabel: str = "Predicted probability",
    title_pad: float = 30,

    # Final prediction reference
    show_prediction_vline: bool = True,
    show_prediction_label: bool = True,
    prediction_label: str = "Predicted probability",
    prediction_line_color: str = "#000000",
    prediction_linewidth: float = 1.5,
    prediction_linestyle: str = "--",
    prediction_label_y: float = 1.03,
    prediction_label_fontsize: int | None = None,

    # Bar value labels
    show_value_labels: bool = True,
    label_mode: Literal["value", "share", "none"] = "value",
    label_position: Literal["inside", "outside", "auto"] = "auto",
    value_decimals: int = 2,
    share_decimals: int = 1,
    show_value_sign: bool = True,
    value_label_fontsize: int | None = None,
    label_color_inside: str = "white",
    label_color_outside: str = "black",
    label_pad: float = 0.01,
    adaptive_probability_labels: bool = True,

    # Axis
    xlim: tuple[float, float] | None = None,
    x_padding: float = 0.10,
    top_ylim_pad: float = 0.35,
    bottom_ylim_pad: float = 0.15,

    # Grid
    show_grid: bool = True,
    grid_color: str = "#cccccc",
    grid_linewidth: float = 1.0,
    grid_alpha: float = 1.0,
    grid_linestyle: str = "-",

    show: bool = True,
    warn_on_skip: bool = True,
) -> dict[str, Any]:
    """
    Plot patient-level PDP allocation waterfall plots.

    Supports single or multiple patients and models:
    - patient_idx=None plots all patients.
    - patient_idx=66 plots one patient.
    - patient_idx=[66, 0] plots selected patients.
    - model_name=None plots all models.
    - model_name="logistic_regression" plots one model.
    - model_name=["logistic_regression", "xgboost"] plots selected models.

    Each patient’s observed feature values are mapped onto the model-development
    PDP curves. The PDP probability at each observed value is used as a
    feature-level response signal. These signals are normalized and rescaled so
    their cumulative allocation sums to the patient’s predicted probability.

    This is a PDP-based response-curve summary, not a SHAP-style additive
    attribution.
    """

    d = pdp_signal_summary.copy()

    required_base_cols = {"patient_idx", "model", "calibration"}
    missing_base_cols = required_base_cols - set(d.columns)

    if missing_base_cols:
        raise ValueError(
            f"pdp_signal_summary is missing required columns: "
            f"{sorted(missing_base_cols)}"
        )

    d = d[d["calibration"] == calibration].copy()

    if d.empty:
        raise ValueError(f"No rows found for calibration={calibration}.")

    all_patients = d["patient_idx"].drop_duplicates().tolist()
    all_models = d["model"].drop_duplicates().tolist()

    patient_list = _as_list_or_all(patient_idx, all_patients)
    model_list = _as_list_or_all(model_name, all_models)

    base_plot_kwargs = _kwargs_for_function(
        _plot_patient_pdp_allocation_waterfall_single,
        locals(),
    )

    base_plot_kwargs.pop("patient_idx", None)
    base_plot_kwargs.pop("model_name", None)
    base_plot_kwargs["pdp_signal_summary"] = d

    outputs: list[dict[str, Any]] = []
    by_model: dict[Any, dict[Any, dict[str, Any]]] = {}
    by_patient: dict[Any, dict[Any, dict[str, Any]]] = {}

    for current_model in model_list:
        by_model.setdefault(current_model, {})

        for current_patient in patient_list:
            try:
                out = _plot_patient_pdp_allocation_waterfall_single(
                    **base_plot_kwargs,
                    patient_idx=current_patient,
                    model_name=current_model,
                )

                outputs.append(out)
                by_model[current_model][current_patient] = out

                by_patient.setdefault(current_patient, {})
                by_patient[current_patient][current_model] = out

            except ValueError as e:
                if warn_on_skip:
                    print(
                        f"[SKIP] model={current_model}, "
                        f"patient_idx={current_patient}: {e}"
                    )

    return {
        "outputs": outputs,
        "by_model": by_model,
        "by_patient": by_patient,
        "patient_idx": patient_list,
        "model_name": model_list,
        "calibration": calibration,
        "n_plots": len(outputs),
    }


# ---------------------------------------------------------------------
# Patient-level stacked PDP plot
# ---------------------------------------------------------------------

def _plot_patient_pdp_allocation_stack_single(
    pdp_signal_summary: pd.DataFrame,
    *,
    model_name: str,
    patient_idx: int | str | list[int | str] | None = None,
    calibration: str = "beta",
    model_alias: Mapping[str, str] | None = None,
    feature_col: str = "feature",
    value_col: str = "allocated_prediction_signal",
    prediction_col: str = "patient_predicted_probability_mean",
    patient_value_col: str = "patient_value_mean",

    # Patient selection
    min_prediction: float | None = None,
    max_prediction: float | None = None,
    top_n: int | None = None,
    patient_sort_ascending: bool = False,

    # Feature ordering and colors
    feature_order: Literal[
        "mean_allocation",
        "total_allocation",
        "mean_contribution",
        "total_contribution",
        "original",
    ] | Sequence[str] = "mean_allocation",
    feature_colors: Mapping[str, str] | None = None,
    cmap_name: str = "Set2",

    # Figure
    figsize: tuple[float, float] | None = None,
    font_size: int = 12,
    title_suffix: str | None = "PDP-based feature allocation of predicted probability",
    xlabel: str = "Predicted probability",
    title_pad: float = 24,

    # Bars
    bar_height: float = 0.75,
    edge_color: str = "white",
    edge_linewidth: float = 0.8,
    alpha: float = 1.0,

    # Prediction labels
    show_prediction_labels: bool = True,
    prediction_label_decimals: int = 2,
    prediction_label_fontsize: int | None = None,
    prediction_label_color: str = "black",
    prediction_label_pad: float = 0.01,
    prediction_label_x: float | None = None,

    show_prediction_label_header: bool = True,
    prediction_label_header: str = "Predicted probability",
    prediction_label_header_y: float = 1.01,
    prediction_label_header_fontsize: int | None = None,

    # Segment labels
    show_segment_labels: bool = False,
    segment_label_mode: Literal[
        "none",
        "value",
        "feature_value",
        "value_and_feature_value",
    ] = "none",
    segment_label_position: Literal["inside", "outside", "auto"] = "auto",
    segment_label_fontsize: int | None = None,
    segment_value_decimals: int = 2,
    feature_value_decimals: int = 2,
    show_segment_value_sign: bool = True,
    segment_label_min_width: float | None = None,
    segment_label_min_width_frac: float = 0.08,
    segment_label_pad: float = 0.006,
    segment_label_color_inside: str = "white",
    segment_label_color_outside: str = "black",

    # Axis
    xlim: tuple[float, float] | None = (0, 1),
    x_padding: float = 0.08,

    # Legend
    show_legend: bool = True,
    legend_title: str = "Feature",
    legend_loc: str = "center left",
    legend_bbox_to_anchor: tuple[float, float] = (1.02, 0.5),
    legend_fontsize: int | None = None,
    legend_title_fontsize: int | None = None,

    # Grid
    show_grid: bool = True,
    grid_color: str = "#cccccc",
    grid_linewidth: float = 1.0,
    grid_alpha: float = 1.0,
    grid_linestyle: str = "-",

    show: bool = True,
) -> dict[str, Any]:
    """
    Plot one model-level stacked PDP allocation chart across patients.

    Each row is one patient. Each stacked segment is a feature-level PDP-based
    probability contribution, and the total bar length equals the patient's
    predicted probability.
    """

    if model_alias is None:
        model_alias = {}

    if segment_label_mode not in {
        "none",
        "value",
        "feature_value",
        "value_and_feature_value",
    }:
        raise ValueError(
            "segment_label_mode must be one of: 'none', 'value', "
            "'feature_value', or 'value_and_feature_value'."
        )

    if segment_label_position not in {"inside", "outside", "auto"}:
        raise ValueError(
            "segment_label_position must be one of: 'inside', 'outside', or 'auto'."
        )

    prepared = _prepare_pdp_allocation_plot_data_single_model(
        pdp_signal_summary,
        model_name=model_name,
        patient_idx=patient_idx,
        calibration=calibration,
        feature_col=feature_col,
        value_col=value_col,
        prediction_col=prediction_col,
        patient_value_col=patient_value_col,
        min_prediction=min_prediction,
        max_prediction=max_prediction,
        top_n=top_n,
        patient_sort_ascending=patient_sort_ascending,
        feature_order=feature_order,
        feature_colors=feature_colors,
        cmap_name=cmap_name,
    )

    selected_patients = prepared["selected_patients"]
    display_patients = selected_patients[::-1]

    features = prepared["feature_order"]
    colors = prepared["feature_colors"]
    wide_plot = prepared["wide_df"].loc[display_patients]
    feature_values_plot = prepared["feature_values_wide"].loc[display_patients]
    pred_map = prepared["pred_map"]

    n_patients = len(display_patients)

    if figsize is None:
        figsize = (10, max(3.0, 0.45 * n_patients + 1.8))

    if prediction_label_fontsize is None:
        prediction_label_fontsize = max(font_size - 2, 8)

    if prediction_label_header_fontsize is None:
        prediction_label_header_fontsize = prediction_label_fontsize

    if segment_label_fontsize is None:
        segment_label_fontsize = max(font_size - 4, 7)

    if legend_fontsize is None:
        legend_fontsize = max(font_size - 2, 8)

    if legend_title_fontsize is None:
        legend_title_fontsize = max(font_size - 1, 9)

    fig, ax = plt.subplots(figsize=figsize)

    y = np.arange(n_patients)
    left = np.zeros(n_patients, dtype=float)
    bar_containers = {}

    for feature in features:
        values = wide_plot[feature].to_numpy(dtype=float)

        bars = ax.barh(
            y,
            values,
            left=left,
            height=bar_height,
            color=colors[feature],
            edgecolor=edge_color,
            linewidth=edge_linewidth,
            alpha=alpha,
            label=feature,
            zorder=3,
        )

        bar_containers[feature] = bars
        left = left + values

    if xlim is None:
        xmax_needed = max(float(left.max()), 1e-8)
        ax.set_xlim(0, xmax_needed * (1 + x_padding))
    else:
        ax.set_xlim(*xlim)

    x0_lim, x1_lim = ax.get_xlim()
    x_range = x1_lim - x0_lim

    if segment_label_min_width is None:
        effective_segment_label_min_width = segment_label_min_width_frac * x_range
    else:
        effective_segment_label_min_width = segment_label_min_width

    # Segment labels.
    if show_segment_labels and segment_label_mode != "none":
        segment_pad_data = segment_label_pad * x_range
        cumulative_left = np.zeros(n_patients, dtype=float)

        for feature in features:
            values = wide_plot[feature].to_numpy(dtype=float)

            for i, value in enumerate(values):
                if value <= 0 or not np.isfinite(value):
                    continue

                x_start = cumulative_left[i]
                x_end = x_start + value
                x_center = x_start + value / 2.0
                patient = display_patients[i]

                if show_segment_value_sign:
                    value_text = f"{value:+.{segment_value_decimals}f}"
                else:
                    value_text = f"{value:.{segment_value_decimals}f}"

                raw_feature_value = feature_values_plot.loc[patient, feature]

                if pd.isna(raw_feature_value):
                    feature_value_text = "NA"
                else:
                    feature_value_text = f"{raw_feature_value:.{feature_value_decimals}f}"

                if segment_label_mode == "value":
                    label_text = value_text
                elif segment_label_mode == "feature_value":
                    label_text = feature_value_text
                elif segment_label_mode == "value_and_feature_value":
                    label_text = f"{feature_value_text} | {value_text}"
                else:
                    label_text = ""

                if not label_text:
                    continue

                if segment_label_position == "inside":
                    use_inside = True
                elif segment_label_position == "outside":
                    use_inside = False
                else:
                    use_inside = value >= effective_segment_label_min_width

                if use_inside:
                    ax.text(
                        x_center,
                        y[i],
                        label_text,
                        ha="center",
                        va="center",
                        fontsize=segment_label_fontsize,
                        fontweight="bold",
                        color=segment_label_color_inside,
                        zorder=5,
                    )
                else:
                    ax.text(
                        x_end + segment_pad_data,
                        y[i],
                        label_text,
                        ha="left",
                        va="center",
                        fontsize=segment_label_fontsize,
                        fontweight="bold",
                        color=segment_label_color_outside,
                        zorder=5,
                        clip_on=False,
                    )

            cumulative_left = cumulative_left + values

    # Prediction labels.
    if show_prediction_labels:
        pad = prediction_label_pad * x_range

        max_display_prediction = max(
            float(pred_map[patient])
            for patient in display_patients
        )

        if prediction_label_x is None:
            prediction_label_x_current = max_display_prediction + pad
        else:
            prediction_label_x_current = prediction_label_x

        if show_prediction_label_header:
            ax.text(
                prediction_label_x_current,
                prediction_label_header_y,
                prediction_label_header,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=prediction_label_header_fontsize,
                fontweight="bold",
                color=prediction_label_color,
                clip_on=False,
                zorder=6,
            )

        for i, patient in enumerate(display_patients):
            pred = float(pred_map[patient])

            ax.text(
                prediction_label_x_current,
                y[i],
                f"{pred:.{prediction_label_decimals}f}",
                ha="center",
                va="center",
                fontsize=prediction_label_fontsize,
                fontweight="bold",
                color=prediction_label_color,
                zorder=6,
                clip_on=False,
            )
    else:
        prediction_label_x_current = None

    if show_grid:
        ax.grid(
            True,
            axis="x",
            color=grid_color,
            linewidth=grid_linewidth,
            alpha=grid_alpha,
            linestyle=grid_linestyle,
            zorder=0,
        )
        ax.set_axisbelow(True)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"Patient {patient}" for patient in display_patients],
        fontsize=font_size,
        fontweight="bold",
    )

    ax.set_xlabel(
        xlabel,
        fontsize=font_size,
        fontweight="bold",
    )

    model_label = model_alias.get(model_name, model_name)
    final_title = model_label

    if title_suffix is not None:
        final_title = f"{final_title}\n{title_suffix}"

    ax.set_title(
        final_title,
        fontsize=font_size + 2,
        fontweight="bold",
        pad=title_pad,
    )

    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)

    if show_legend:
        ax.legend(
            title=legend_title,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            fontsize=legend_fontsize,
            title_fontsize=legend_title_fontsize,
            frameon=False,
        )

    plt.tight_layout()

    if show:
        plt.show()

    return {
        "fig": fig,
        "ax": ax,
        "plot_df": prepared["plot_df"],
        "wide_df": prepared["wide_df"],
        "selected_patients": selected_patients,
        "display_patients": display_patients,
        "feature_order": features,
        "feature_colors": colors,
        "model": model_name,
        "calibration": calibration,
        "min_prediction": min_prediction,
        "max_prediction": max_prediction,
        "top_n": top_n,
        "bar_containers": bar_containers,
        "prediction_label_x": prediction_label_x_current,
        "effective_segment_label_min_width": effective_segment_label_min_width,
        "prepared": prepared,
    }


def plot_patient_pdp_allocation_stack(
    pdp_signal_summary: pd.DataFrame,
    *,
    patient_idx: int | str | list[int | str] | None = None,
    model_name: str | list[str] | None = None,
    calibration: str = "beta",
    model_alias: Mapping[str, str] | None = None,
    feature_col: str = "feature",
    value_col: str = "allocated_prediction_signal",
    prediction_col: str = "patient_predicted_probability_mean",
    patient_value_col: str = "patient_value_mean",

    # Patient selection
    min_prediction: float | None = None,
    max_prediction: float | None = None,
    top_n: int | None = None,
    patient_sort_ascending: bool = False,

    # Feature ordering and colors
    feature_order: Literal[
        "mean_allocation",
        "total_allocation",
        "mean_contribution",
        "total_contribution",
        "original",
    ] | Sequence[str] = "mean_allocation",
    feature_colors: Mapping[str, str] | None = None,
    cmap_name: str = "Set2",

    # Figure
    figsize: tuple[float, float] | None = None,
    font_size: int = 12,
    title_suffix: str | None = "PDP-based feature allocation of predicted probability",
    xlabel: str = "Predicted probability",
    title_pad: float = 24,

    # Bars
    bar_height: float = 0.75,
    edge_color: str = "white",
    edge_linewidth: float = 0.8,
    alpha: float = 1.0,

    # Prediction labels
    show_prediction_labels: bool = True,
    prediction_label_decimals: int = 2,
    prediction_label_fontsize: int | None = None,
    prediction_label_color: str = "black",
    prediction_label_pad: float = 0.01,
    prediction_label_x: float | None = None,
    show_prediction_label_header: bool = True,
    prediction_label_header: str = "Predicted probability",
    prediction_label_header_y: float = 1.01,
    prediction_label_header_fontsize: int | None = None,

    # Segment labels
    show_segment_labels: bool = False,
    segment_label_mode: Literal[
        "none",
        "value",
        "feature_value",
        "value_and_feature_value",
    ] = "none",
    segment_label_position: Literal["inside", "outside", "auto"] = "auto",
    segment_label_fontsize: int | None = None,
    segment_value_decimals: int = 2,
    feature_value_decimals: int = 2,
    show_segment_value_sign: bool = True,
    segment_label_min_width: float | None = None,
    segment_label_min_width_frac: float = 0.08,
    segment_label_pad: float = 0.006,
    segment_label_color_inside: str = "white",
    segment_label_color_outside: str = "black",

    # Axis
    xlim: tuple[float, float] | None = (0, 1),
    x_padding: float = 0.08,

    # Legend
    show_legend: bool = True,
    legend_title: str = "Feature",
    legend_loc: str = "center left",
    legend_bbox_to_anchor: tuple[float, float] = (1.02, 0.5),
    legend_fontsize: int | None = None,
    legend_title_fontsize: int | None = None,

    # Grid
    show_grid: bool = True,
    grid_color: str = "#cccccc",
    grid_linewidth: float = 1.0,
    grid_alpha: float = 1.0,
    grid_linestyle: str = "-",

    show: bool = True,
    warn_on_skip: bool = True,
) -> dict[str, Any]:
    """
    Plot stacked patient-level PDP allocation charts.

    This wrapper supports one or multiple models. Each model gets one figure.
    """

    d = pdp_signal_summary.copy()

    required_base_cols = {"patient_idx", "model", "calibration"}
    missing_base_cols = required_base_cols - set(d.columns)

    if missing_base_cols:
        raise ValueError(
            f"pdp_signal_summary is missing required columns: "
            f"{sorted(missing_base_cols)}"
        )

    d = d[d["calibration"] == calibration].copy()

    if d.empty:
        raise ValueError(f"No rows found for calibration={calibration}.")

    all_models = d["model"].drop_duplicates().tolist()
    model_list = _as_list_or_all(model_name, all_models)

    base_plot_kwargs = _kwargs_for_function(
        _plot_patient_pdp_allocation_stack_single,
        locals(),
    )

    base_plot_kwargs.pop("model_name", None)
    base_plot_kwargs["pdp_signal_summary"] = d

    outputs: list[dict[str, Any]] = []
    by_model: dict[Any, dict[str, Any]] = {}

    for current_model in model_list:
        try:
            out = _plot_patient_pdp_allocation_stack_single(
                **base_plot_kwargs,
                model_name=current_model,
            )

            outputs.append(out)
            by_model[current_model] = out

        except ValueError as e:
            if warn_on_skip:
                print(f"[SKIP] model={current_model}: {e}")

    return {
        "outputs": outputs,
        "by_model": by_model,
        "patient_idx": patient_idx,
        "model_name": model_list,
        "calibration": calibration,
        "min_prediction": min_prediction,
        "max_prediction": max_prediction,
        "top_n": top_n,
        "n_plots": len(outputs),
    }


# ---------------------------------------------------------------------
# Cohort-level PDP contribution plot
# ---------------------------------------------------------------------

def _plot_cohort_pdp_feature_contribution_single(
    pdp_signal_summary: pd.DataFrame,
    *,
    model_name: str,
    patient_idx: int | str | list[int | str] | None = None,
    calibration: str = "beta",
    model_alias: Mapping[str, str] | None = None,
    feature_col: str = "feature",
    value_col: str = "allocated_prediction_signal",
    prediction_col: str = "patient_predicted_probability_mean",
    patient_value_col: str = "patient_value_mean",

    # Patient selection
    min_prediction: float | None = None,
    max_prediction: float | None = None,
    top_n: int | None = None,
    patient_sort_ascending: bool = False,

    # Feature ordering and colors
    feature_order: Literal[
        "mean_contribution",
        "total_contribution",
        "mean_allocation",
        "total_allocation",
        "original",
    ] | Sequence[str] = "mean_contribution",
    feature_colors: Mapping[str, str] | None = None,
    cmap_name: str = "Set2",

    # Figure
    orientation: Literal["vertical", "horizontal"] = "vertical",
    figsize: tuple[float, float] | None = None,
    font_size: int = 12,
    title_suffix: str | None = "Cohort-level PDP-based probability contribution",
    xlabel: str = "Feature",
    ylabel: str = "Average probability contribution",
    title_pad: float = 24,

    # Bars
    bar_width: float = 0.75,
    bar_height: float = 0.75,
    alpha: float = 1.0,
    error_color: str = "black",
    error_linewidth: float = 1.5,
    error_capsize: float = 4,

    # Labels
    show_value_labels: bool = True,
    feature_value_decimals: int = 2,
    contribution_decimals: int = 2,
    label_fontsize: int | None = None,
    label_color: str = "black",
    label_pad: float = 0.02,

    # Axis
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    axis_padding: float = 0.18,

    # Grid
    show_grid: bool = True,
    grid_color: str = "#cccccc",
    grid_linewidth: float = 1.0,
    grid_alpha: float = 1.0,
    grid_linestyle: str = "-",

    show: bool = True,
) -> dict[str, Any]:
    """
    Plot one model-level cohort PDP feature contribution summary.

    Bar height/length shows the average PDP-based probability contribution.
    Error bars always show SD across selected patients.
    """

    if model_alias is None:
        model_alias = {}

    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be either 'vertical' or 'horizontal'.")

    prepared = _prepare_pdp_allocation_plot_data_single_model(
        pdp_signal_summary,
        model_name=model_name,
        patient_idx=patient_idx,
        calibration=calibration,
        feature_col=feature_col,
        value_col=value_col,
        prediction_col=prediction_col,
        patient_value_col=patient_value_col,
        min_prediction=min_prediction,
        max_prediction=max_prediction,
        top_n=top_n,
        patient_sort_ascending=patient_sort_ascending,
        feature_order=feature_order,
        feature_colors=feature_colors,
        cmap_name=cmap_name,
    )

    plot_df_long = prepared["plot_df"]

    summary_df = (
        plot_df_long.groupby(feature_col, as_index=False)
        .agg(
            n_patients=("patient_idx", "nunique"),
            mean_contribution=(value_col, "mean"),
            sd_contribution=(value_col, "std"),
            median_contribution=(value_col, "median"),
            total_contribution=(value_col, "sum"),
            mean_feature_value=(patient_value_col, "mean"),
            sd_feature_value=(patient_value_col, "std"),
            median_feature_value=(patient_value_col, "median"),
        )
    )

    summary_df["sd_contribution"] = summary_df["sd_contribution"].fillna(0.0)
    summary_df["sd_feature_value"] = summary_df["sd_feature_value"].fillna(0.0)

    features = prepared["feature_order"]

    summary_df[feature_col] = pd.Categorical(
        summary_df[feature_col],
        categories=features,
        ordered=True,
    )
    summary_df = summary_df.sort_values(feature_col)
    summary_df[feature_col] = summary_df[feature_col].astype(str)
    summary_df = summary_df.reset_index(drop=True)

    if orientation == "horizontal":
        plot_df = summary_df.iloc[::-1].reset_index(drop=True)
    else:
        plot_df = summary_df.reset_index(drop=True)

    plot_features = plot_df[feature_col].tolist()
    colors = prepared["feature_colors"]

    n_features = len(plot_df)

    if figsize is None:
        if orientation == "horizontal":
            figsize = (9.5, max(3.0, 0.55 * n_features + 1.8))
        else:
            figsize = (8.5, 5.0)

    if label_fontsize is None:
        label_fontsize = max(font_size - 2, 8)

    fig, ax = plt.subplots(figsize=figsize)

    means = plot_df["mean_contribution"].to_numpy(dtype=float)
    sds = plot_df["sd_contribution"].to_numpy(dtype=float)
    bar_colors = [colors[feature] for feature in plot_features]

    if orientation == "horizontal":
        y = np.arange(n_features)

        ax.barh(
            y,
            means,
            xerr=sds,
            height=bar_height,
            color=bar_colors,
            alpha=alpha,
            ecolor=error_color,
            capsize=error_capsize,
            error_kw={
                "elinewidth": error_linewidth,
                "capthick": error_linewidth,
            },
            zorder=3,
        )

        xmax_needed = max(float(np.nanmax(means + sds)), 1e-8)

        if xlim is None:
            ax.set_xlim(0, xmax_needed * (1 + axis_padding))
        else:
            ax.set_xlim(*xlim)

        x0_lim, x1_lim = ax.get_xlim()
        x_range = x1_lim - x0_lim
        label_pad_data = label_pad * x_range

        if show_value_labels:
            for i, row in plot_df.iterrows():
                mean_contribution = float(row["mean_contribution"])
                sd_contribution = float(row["sd_contribution"])
                mean_value = float(row["mean_feature_value"])
                sd_value = float(row["sd_feature_value"])

                label_text = (
                    f"{row[feature_col]}: {mean_value:.{feature_value_decimals}f} ± "
                    f"{sd_value:.{feature_value_decimals}f}\n"
                    f"{mean_contribution:.{contribution_decimals}f} ± "
                    f"{sd_contribution:.{contribution_decimals}f}"
                )

                ax.text(
                    mean_contribution + sd_contribution + label_pad_data,
                    y[i],
                    label_text,
                    ha="left",
                    va="center",
                    fontsize=label_fontsize,
                    fontweight="bold",
                    color=label_color,
                    clip_on=False,
                    zorder=5,
                )

        ax.set_yticks(y)
        ax.set_yticklabels(
            plot_df[feature_col].tolist(),
            fontsize=font_size,
            fontweight="bold",
        )

        ax.set_xlabel(
            ylabel,
            fontsize=font_size,
            fontweight="bold",
        )

        if show_grid:
            ax.grid(
                True,
                axis="x",
                color=grid_color,
                linewidth=grid_linewidth,
                alpha=grid_alpha,
                linestyle=grid_linestyle,
                zorder=0,
            )
            ax.set_axisbelow(True)

    else:
        x = np.arange(n_features)

        ax.bar(
            x,
            means,
            yerr=sds,
            width=bar_width,
            color=bar_colors,
            alpha=alpha,
            ecolor=error_color,
            capsize=error_capsize,
            error_kw={
                "elinewidth": error_linewidth,
                "capthick": error_linewidth,
            },
            zorder=3,
        )

        ymax_needed = max(float(np.nanmax(means + sds)), 1e-8)

        if ylim is None:
            ax.set_ylim(0, ymax_needed * (1 + axis_padding))
        else:
            ax.set_ylim(*ylim)

        y0_lim, y1_lim = ax.get_ylim()
        y_range = y1_lim - y0_lim
        label_pad_data = label_pad * y_range

        if show_value_labels:
            for i, row in plot_df.iterrows():
                mean_contribution = float(row["mean_contribution"])
                sd_contribution = float(row["sd_contribution"])
                mean_value = float(row["mean_feature_value"])
                sd_value = float(row["sd_feature_value"])

                label_text = (
                    f"{row[feature_col]}: {mean_value:.{feature_value_decimals}f} ± "
                    f"{sd_value:.{feature_value_decimals}f}\n"
                    f"{mean_contribution:.{contribution_decimals}f} ± "
                    f"{sd_contribution:.{contribution_decimals}f}"
                )

                ax.text(
                    x[i],
                    mean_contribution + sd_contribution + label_pad_data,
                    label_text,
                    ha="center",
                    va="bottom",
                    fontsize=label_fontsize,
                    fontweight="bold",
                    color=label_color,
                    clip_on=False,
                    zorder=5,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            plot_df[feature_col].tolist(),
            fontsize=font_size,
            fontweight="bold",
        )

        ax.set_xlabel(
            xlabel,
            fontsize=font_size,
            fontweight="bold",
        )

        ax.set_ylabel(
            ylabel,
            fontsize=font_size,
            fontweight="bold",
        )

        if show_grid:
            ax.grid(
                True,
                axis="y",
                color=grid_color,
                linewidth=grid_linewidth,
                alpha=grid_alpha,
                linestyle=grid_linestyle,
                zorder=0,
            )
            ax.set_axisbelow(True)

    model_label = model_alias.get(model_name, model_name)
    final_title = model_label

    if title_suffix is not None:
        final_title = f"{final_title}\n{title_suffix}"

    ax.set_title(
        final_title,
        fontsize=font_size + 2,
        fontweight="bold",
        pad=title_pad,
    )

    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none" if orientation == "horizontal" else "left")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if orientation == "horizontal":
        ax.spines["left"].set_visible(False)

    plt.tight_layout()

    if show:
        plt.show()

    return {
        "fig": fig,
        "ax": ax,
        "summary_df": summary_df,
        "plot_df": plot_df,
        "selected_patients": prepared["selected_patients"],
        "patient_predictions": prepared["patient_predictions"],
        "feature_order": features,
        "feature_colors": colors,
        "model": model_name,
        "calibration": calibration,
        "min_prediction": min_prediction,
        "max_prediction": max_prediction,
        "top_n": top_n,
        "orientation": orientation,
        "prepared": prepared,
    }


def plot_cohort_pdp_feature_contribution(
    pdp_signal_summary: pd.DataFrame,
    *,
    patient_idx: int | str | list[int | str] | None = None,
    model_name: str | list[str] | None = None,
    calibration: str = "beta",
    model_alias: Mapping[str, str] | None = None,
    feature_col: str = "feature",
    value_col: str = "allocated_prediction_signal",
    prediction_col: str = "patient_predicted_probability_mean",
    patient_value_col: str = "patient_value_mean",

    # Patient selection
    min_prediction: float | None = None,
    max_prediction: float | None = None,
    top_n: int | None = None,
    patient_sort_ascending: bool = False,

    # Feature ordering and colors
    feature_order: Literal[
        "mean_contribution",
        "total_contribution",
        "mean_allocation",
        "total_allocation",
        "original",
    ] | Sequence[str] = "mean_contribution",
    feature_colors: Mapping[str, str] | None = None,
    cmap_name: str = "Set2",

    # Figure
    orientation: Literal["vertical", "horizontal"] = "vertical",
    figsize: tuple[float, float] | None = None,
    font_size: int = 12,
    title_suffix: str | None = "Cohort-level PDP-based probability contribution",
    xlabel: str = "Feature",
    ylabel: str = "Average probability contribution",
    title_pad: float = 24,

    # Bars
    bar_width: float = 0.75,
    bar_height: float = 0.75,
    alpha: float = 1.0,
    error_color: str = "black",
    error_linewidth: float = 1.5,
    error_capsize: float = 4,

    # Labels
    show_value_labels: bool = True,
    feature_value_decimals: int = 2,
    contribution_decimals: int = 2,
    label_fontsize: int | None = None,
    label_color: str = "black",
    label_pad: float = 0.02,

    # Axis
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    axis_padding: float = 0.18,

    # Grid
    show_grid: bool = True,
    grid_color: str = "#cccccc",
    grid_linewidth: float = 1.0,
    grid_alpha: float = 1.0,
    grid_linestyle: str = "-",

    show: bool = True,
    warn_on_skip: bool = True,
) -> dict[str, Any]:
    """
    Plot cohort-level PDP-based feature probability contribution summaries.

    This wrapper supports one or multiple models. Each model gets one figure.
    """

    d = pdp_signal_summary.copy()

    required_base_cols = {"patient_idx", "model", "calibration"}
    missing_base_cols = required_base_cols - set(d.columns)

    if missing_base_cols:
        raise ValueError(
            f"pdp_signal_summary is missing required columns: "
            f"{sorted(missing_base_cols)}"
        )

    d = d[d["calibration"] == calibration].copy()

    if d.empty:
        raise ValueError(f"No rows found for calibration={calibration}.")

    all_models = d["model"].drop_duplicates().tolist()
    model_list = _as_list_or_all(model_name, all_models)

    base_plot_kwargs = _kwargs_for_function(
        _plot_cohort_pdp_feature_contribution_single,
        locals(),
    )

    base_plot_kwargs.pop("model_name", None)
    base_plot_kwargs["pdp_signal_summary"] = d

    outputs: list[dict[str, Any]] = []
    by_model: dict[Any, dict[str, Any]] = {}

    for current_model in model_list:
        try:
            out = _plot_cohort_pdp_feature_contribution_single(
                **base_plot_kwargs,
                model_name=current_model,
            )

            outputs.append(out)
            by_model[current_model] = out

        except ValueError as e:
            if warn_on_skip:
                print(f"[SKIP] model={current_model}: {e}")

    return {
        "outputs": outputs,
        "by_model": by_model,
        "patient_idx": patient_idx,
        "model_name": model_list,
        "calibration": calibration,
        "min_prediction": min_prediction,
        "max_prediction": max_prediction,
        "top_n": top_n,
        "n_plots": len(outputs),
    }



# ---------------------------------------------------------------------
# Cohort-level PDP contribution comparison plot
# ---------------------------------------------------------------------

def _format_pdp_contribution_label(
    mean_contribution: float,
    sd_contribution: float,
    *,
    contribution_decimals: int = 2,
    adaptive_probability_labels: bool = True,
) -> str:
    """
    Format mean ± SD contribution label for PDP contribution bars.
    """

    if adaptive_probability_labels:
        mean_text = _format_probability_value(
            mean_contribution,
            fixed_decimals=contribution_decimals,
        )
        sd_text = _format_probability_value(
            sd_contribution,
            fixed_decimals=contribution_decimals,
        )
    else:
        mean_text = f"{mean_contribution:.{contribution_decimals}f}"
        sd_text = f"{sd_contribution:.{contribution_decimals}f}"

    return f"{mean_text} ± {sd_text}"


def _plot_cohort_pdp_feature_contribution_comparison_single(
    pdp_signal_summary: pd.DataFrame,
    *,
    model_name: str,
    cohorts: Mapping[str, Mapping[str, Any]],
    calibration: str = "beta",
    model_alias: Mapping[str, str] | None = None,
    feature_col: str = "feature",
    value_col: str = "allocated_prediction_signal",
    prediction_col: str = "patient_predicted_probability_mean",
    patient_value_col: str = "patient_value_mean",


    # Default patient selection values.
    # Each cohort can override these inside the cohorts dictionary.
    patient_idx: int | str | list[int | str] | None = None,
    top_n: int | None = None,
    patient_sort_ascending: bool = False,

    # Feature ordering
    feature_order: Literal[
        "mean_contribution",
        "total_contribution",
        "mean_allocation",
        "total_allocation",
        "original",
    ] | Sequence[str] = "mean_contribution",

    # Figure
    figsize: tuple[float, float] | None = None,
    font_size: int = 12,
    title_suffix: str | None = "Cohort-level PDP-based probability contribution comparison",
    xlabel: str = "Feature",
    ylabel: str = "Average probability contribution",
    title_pad: float = 24,

    # Bars
    group_width: float = 0.75,
    alpha: float = 1.0,
    error_color: str = "black",
    error_linewidth: float = 1.3,
    error_capsize: float = 3,

    # Cohort colors
    cohort_colors: Mapping[str, str] | None = None,
    cohort_cmap_name: str = "Set2",

    # Labels
    show_bar_labels: bool = True,
    contribution_decimals: int = 2,
    adaptive_probability_labels: bool = True,
    label_fontsize: int | None = None,
    label_color: str = "black",
    label_pad: float = 0.015,
    label_group_x_offset: float = 0.0,
    bar_label_mode: Literal[
        "contribution",
        "feature_value",
        "feature_value_and_contribution",
    ] = "contribution",
    feature_value_decimals: int = 2,

    # Axis
    ylim: tuple[float, float] | None = None,
    axis_padding: float = 0.20,

    # Legend
    show_legend: bool = True,
    legend_title: str = "Cohort",
    legend_loc: str = "best",
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    legend_fontsize: int | None = None,
    legend_title_fontsize: int | None = None,
    legend_frameon: bool = True,
    legend_edgecolor: str = "black",
    legend_facecolor: str = "white",
    legend_framealpha: float = 1.0,
    legend_linewidth: float = 1.0,

    # Grid
    show_grid: bool = True,
    grid_color: str = "#cccccc",
    grid_linewidth: float = 1.0,
    grid_alpha: float = 1.0,
    grid_linestyle: str = "-",

    show: bool = True,
    warn_on_skip: bool = True,
) -> dict[str, Any]:
    """
    Compare cohort-level PDP-based probability contributions in one grouped plot.

    Each cohort is defined by patient-selection filters, usually min_prediction
    and max_prediction. Bars show the average PDP-based probability contribution;
    error bars show the SD across patients.
    """

    if model_alias is None:
        model_alias = {}

    if not cohorts:
        raise ValueError("cohorts must be a non-empty mapping.")

    cohort_order = list(cohorts.keys())
    cohort_summaries: list[pd.DataFrame] = []
    prepared_by_cohort: dict[str, dict[str, Any]] = {}

    for cohort_label, cohort_kwargs in cohorts.items():
        try:
            cohort_summary = _prepare_cohort_pdp_feature_contribution_summary(
                pdp_signal_summary,
                model_name=model_name,
                calibration=calibration,
                feature_col=feature_col,
                value_col=value_col,
                prediction_col=prediction_col,
                patient_value_col=patient_value_col,
                patient_idx=cohort_kwargs.get("patient_idx", patient_idx),
                min_prediction=cohort_kwargs.get("min_prediction", None),
                max_prediction=cohort_kwargs.get("max_prediction", None),
                top_n=cohort_kwargs.get("top_n", top_n),
                patient_sort_ascending=cohort_kwargs.get(
                    "patient_sort_ascending",
                    patient_sort_ascending,
                ),
                feature_order=feature_order,
            )

            temp = cohort_summary["summary_df"].copy()
            temp["cohort"] = cohort_label

            cohort_summaries.append(temp)
            prepared_by_cohort[cohort_label] = cohort_summary["prepared"]

        except ValueError as e:
            if warn_on_skip:
                print(f"[SKIP] cohort={cohort_label!r}: {e}")

    if not cohort_summaries:
        raise ValueError("No cohort summaries could be built.")

    summary_df = pd.concat(cohort_summaries, ignore_index=True)

    # Use feature order from the combined summaries.
    features = (
        summary_df.groupby(feature_col, as_index=True)["mean_contribution"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    if not isinstance(feature_order, str):
        provided = list(feature_order)
        features = provided + [feature for feature in features if feature not in provided]
    elif feature_order == "original":
        features = summary_df[feature_col].drop_duplicates().tolist()
    elif feature_order in {"total_contribution", "total_allocation"}:
        features = (
            summary_df.groupby(feature_col, as_index=True)["total_contribution"]
            .sum()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
    elif feature_order not in {"mean_contribution", "mean_allocation"}:
        raise ValueError(
            "feature_order must be one of 'mean_contribution', "
            "'total_contribution', 'mean_allocation', 'total_allocation', "
            "'original', or a sequence of feature names."
        )

    summary_df[feature_col] = pd.Categorical(
        summary_df[feature_col],
        categories=features,
        ordered=True,
    )
    summary_df["cohort"] = pd.Categorical(
        summary_df["cohort"],
        categories=cohort_order,
        ordered=True,
    )
    summary_df = summary_df.sort_values([feature_col, "cohort"]).reset_index(drop=True)

    if cohort_colors is None:
        cmap = plt.get_cmap(cohort_cmap_name)
        colors = {
            cohort: cmap(i % cmap.N)
            for i, cohort in enumerate(cohort_order)
        }
    else:
        colors = dict(cohort_colors)
        missing = [cohort for cohort in cohort_order if cohort not in colors]
        if missing:
            cmap = plt.get_cmap(cohort_cmap_name)
            start_i = len(colors)
            for offset, cohort in enumerate(missing):
                colors[cohort] = cmap((start_i + offset) % cmap.N)

    n_features = len(features)
    n_cohorts = len(cohort_order)

    if figsize is None:
        figsize = (10, 5.2)

    if label_fontsize is None:
        label_fontsize = max(font_size - 3, 7)

    if legend_fontsize is None:
        legend_fontsize = max(font_size - 2, 8)

    if legend_title_fontsize is None:
        legend_title_fontsize = max(font_size - 1, 9)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_features)
    bar_width = group_width / max(n_cohorts, 1)
    offsets = (
        np.arange(n_cohorts) - (n_cohorts - 1) / 2.0
    ) * bar_width

    ymax_needed = 0.0
    bar_containers: dict[str, Any] = {}

    for cohort_i, cohort_label in enumerate(cohort_order):
        cohort_df = (
            summary_df[summary_df["cohort"] == cohort_label]
            .set_index(feature_col)
            .reindex(features)
            .reset_index()
        )

        means = cohort_df["mean_contribution"].fillna(0.0).to_numpy(dtype=float)
        sds = cohort_df["sd_contribution"].fillna(0.0).to_numpy(dtype=float)
        xpos = x + offsets[cohort_i]

        # Probability contributions are bounded below by 0.
        # Use asymmetric error bars so mean - SD does not visually extend below 0.
        lower_errors = np.minimum(sds, means)
        upper_errors = sds
        yerr = np.vstack([lower_errors, upper_errors])

        bars = ax.bar(
            xpos,
            means,
            yerr=yerr,
            width=bar_width,
            color=colors[cohort_label],
            alpha=alpha,
            ecolor=error_color,
            capsize=error_capsize,
            error_kw={
                "elinewidth": error_linewidth,
                "capthick": error_linewidth,
            },
            label=cohort_label,
            zorder=3,
        )




        bar_containers[cohort_label] = bars
        ymax_needed = max(ymax_needed, float(np.nanmax(means + sds)))

        if show_bar_labels:
            for i, row in cohort_df.iterrows():
                if pd.isna(row["mean_contribution"]):
                    continue

                # label_text = _format_pdp_contribution_label(
                #     float(row["mean_contribution"]),
                #     float(row["sd_contribution"]),
                #     contribution_decimals=contribution_decimals,
                #     adaptive_probability_labels=adaptive_probability_labels,
                # )

                mean_contribution = float(row["mean_contribution"])
                sd_contribution = float(row["sd_contribution"])
                mean_value = float(row["mean_feature_value"])
                sd_value = float(row["sd_feature_value"])

                contribution_text = _format_pdp_contribution_label(
                    mean_contribution,
                    sd_contribution,
                    contribution_decimals=contribution_decimals,
                    adaptive_probability_labels=adaptive_probability_labels,
                )

                feature_value_text = (
                    f"{mean_value:.{feature_value_decimals}f} ± "
                    f"{sd_value:.{feature_value_decimals}f}"
                )

                if bar_label_mode == "contribution":
                    label_text = contribution_text
                elif bar_label_mode == "feature_value":
                    label_text = f"x: {feature_value_text}"
                elif bar_label_mode == "feature_value_and_contribution":
                    label_text = f"x: {feature_value_text}\nPDP: {contribution_text}"
                else:
                    raise ValueError(
                        "bar_label_mode must be 'contribution', 'feature_value', "
                        "or 'feature_value_and_contribution'."
                    )
                

                label_x = xpos[i] + (
                    cohort_i - (n_cohorts - 1) / 2.0
                ) * label_group_x_offset

                ax.text(
                    label_x,
                    float(row["mean_contribution"]) + float(row["sd_contribution"]),
                    label_text,
                    ha="center",
                    va="bottom",
                    fontsize=label_fontsize,
                    fontweight="bold",
                    color=label_color,
                    clip_on=False,
                    zorder=5,
                )



    ymax_needed = max(ymax_needed, 1e-8)

    if ylim is None:
        ax.set_ylim(0, ymax_needed * (1 + axis_padding))
    else:
        ax.set_ylim(*ylim)

    # After ylim is set, add label padding in axis-data units.
    if show_bar_labels:
        y0_lim, y1_lim = ax.get_ylim()
        y_range = y1_lim - y0_lim
        label_pad_data = label_pad * y_range

        # Move existing text labels up by the requested padding.
        for text in ax.texts:
            x_text, y_text = text.get_position()
            text.set_position((x_text, y_text + label_pad_data))

    ax.set_xticks(x)
    ax.set_xticklabels(
        features,
        fontsize=font_size,
        fontweight="bold",
    )

    ax.set_xlabel(
        xlabel,
        fontsize=font_size,
        fontweight="bold",
    )

    ax.set_ylabel(
        ylabel,
        fontsize=font_size,
        fontweight="bold",
    )

    if show_grid:
        ax.grid(
            True,
            axis="y",
            color=grid_color,
            linewidth=grid_linewidth,
            alpha=grid_alpha,
            linestyle=grid_linestyle,
            zorder=0,
        )
        ax.set_axisbelow(True)

    model_label = model_alias.get(model_name, model_name)
    final_title = model_label

    if title_suffix is not None:
        final_title = f"{final_title}\n{title_suffix}"

    ax.set_title(
        final_title,
        fontsize=font_size + 2,
        fontweight="bold",
        pad=title_pad,
    )

    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if show_legend:
        legend_kwargs = {
            "title": legend_title,
            "loc": legend_loc,
            "fontsize": legend_fontsize,
            "title_fontsize": legend_title_fontsize,
            "frameon": legend_frameon,
            "edgecolor": legend_edgecolor,
            "facecolor": legend_facecolor,
            "framealpha": legend_framealpha,
            "fancybox": False,
        }

        if legend_bbox_to_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor

        legend = ax.legend(**legend_kwargs)

        if legend_frameon:
            legend.get_frame().set_linewidth(legend_linewidth)

    plt.tight_layout()

    if show:
        plt.show()

    return {
        "fig": fig,
        "ax": ax,
        "summary_df": summary_df,
        "prepared_by_cohort": prepared_by_cohort,
        "selected_patients_by_cohort": {
            cohort_label: prepared["selected_patients"]
            for cohort_label, prepared in prepared_by_cohort.items()
        },
        "patient_predictions_by_cohort": {
            cohort_label: prepared["patient_predictions"]
            for cohort_label, prepared in prepared_by_cohort.items()
        },
        "feature_order": features,
        "cohort_order": cohort_order,
        "cohort_colors": colors,
        "model": model_name,
        "calibration": calibration,
        "bar_containers": bar_containers,
    }


def plot_cohort_pdp_feature_contribution_comparison(
    pdp_signal_summary: pd.DataFrame,
    *,
    model_name: str | list[str] | None = None,
    cohorts: Mapping[str, Mapping[str, Any]],
    calibration: str = "beta",
    model_alias: Mapping[str, str] | None = None,
    feature_col: str = "feature",
    value_col: str = "allocated_prediction_signal",
    prediction_col: str = "patient_predicted_probability_mean",
    patient_value_col: str = "patient_value_mean",

    # Default patient selection values.
    # Each cohort can override these inside the cohorts dictionary.
    patient_idx: int | str | list[int | str] | None = None,
    top_n: int | None = None,
    patient_sort_ascending: bool = False,

    # Feature ordering
    feature_order: Literal[
        "mean_contribution",
        "total_contribution",
        "mean_allocation",
        "total_allocation",
        "original",
    ] | Sequence[str] = "mean_contribution",

    # Figure
    figsize: tuple[float, float] | None = None,
    font_size: int = 12,
    title_suffix: str | None = "Cohort-level PDP-based probability contribution comparison",
    xlabel: str = "Feature",
    ylabel: str = "Average probability contribution",
    title_pad: float = 24,

    # Bars
    group_width: float = 0.75,
    alpha: float = 1.0,
    error_color: str = "black",
    error_linewidth: float = 1.3,
    error_capsize: float = 3,

    # Cohort colors
    cohort_colors: Mapping[str, str] | None = None,
    cohort_cmap_name: str = "Set2",

    # Labels
    show_bar_labels: bool = True,
    contribution_decimals: int = 2,
    adaptive_probability_labels: bool = True,
    label_fontsize: int | None = None,
    label_color: str = "black",
    label_pad: float = 0.015,
    label_group_x_offset: float = 0.0,
    bar_label_mode: Literal[
        "contribution",
        "feature_value",
        "feature_value_and_contribution",
    ] = "contribution",
    feature_value_decimals: int = 2,

    # Axis
    ylim: tuple[float, float] | None = None,
    axis_padding: float = 0.20,

    # Legend
    show_legend: bool = True,
    legend_title: str = "Cohort",
    legend_loc: str = "best",
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    legend_fontsize: int | None = None,
    legend_title_fontsize: int | None = None,
    legend_frameon: bool = True,
    legend_edgecolor: str = "black",
    legend_facecolor: str = "white",
    legend_framealpha: float = 1.0,
    legend_linewidth: float = 1.0,

    # Grid
    show_grid: bool = True,
    grid_color: str = "#cccccc",
    grid_linewidth: float = 1.0,
    grid_alpha: float = 1.0,
    grid_linestyle: str = "-",

    show: bool = True,
    warn_on_skip: bool = True,
) -> dict[str, Any]:
    """
    Compare cohort-level PDP-based probability contributions for one or more models.

    Each cohort is defined by a dictionary of patient-selection filters, such as:

        cohorts={
            "Selected for enrichment": {"min_prediction": 0.70, "max_prediction": 1.0},
            "Below threshold": {"min_prediction": 0.0, "max_prediction": 0.70},
        }

    Each model gets one grouped contribution figure.
    """

    d = pdp_signal_summary.copy()

    required_base_cols = {"patient_idx", "model", "calibration"}
    missing_base_cols = required_base_cols - set(d.columns)

    if missing_base_cols:
        raise ValueError(
            f"pdp_signal_summary is missing required columns: "
            f"{sorted(missing_base_cols)}"
        )

    d = d[d["calibration"] == calibration].copy()

    if d.empty:
        raise ValueError(f"No rows found for calibration={calibration}.")

    all_models = d["model"].drop_duplicates().tolist()
    model_list = _as_list_or_all(model_name, all_models)

    base_plot_kwargs = _kwargs_for_function(
        _plot_cohort_pdp_feature_contribution_comparison_single,
        locals(),
    )

    base_plot_kwargs.pop("model_name", None)
    base_plot_kwargs["pdp_signal_summary"] = d

    outputs: list[dict[str, Any]] = []
    by_model: dict[Any, dict[str, Any]] = {}

    for current_model in model_list:
        try:
            out = _plot_cohort_pdp_feature_contribution_comparison_single(
                **base_plot_kwargs,
                model_name=current_model,
            )

            outputs.append(out)
            by_model[current_model] = out

        except ValueError as e:
            if warn_on_skip:
                print(f"[SKIP] model={current_model}: {e}")

    return {
        "outputs": outputs,
        "by_model": by_model,
        "model_name": model_list,
        "calibration": calibration,
        "cohorts": cohorts,
        "n_plots": len(outputs),
    }



__all__ = [
    "build_external_pdp_signal_allocation_from_results",
    "plot_patient_pdp_allocation_waterfall",
    "plot_patient_pdp_allocation_stack",
    "plot_cohort_pdp_feature_contribution",
    "plot_cohort_pdp_feature_contribution_comparison",
]