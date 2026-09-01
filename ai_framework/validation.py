# validation .py
# ML external validation datase

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Type, Mapping, Literal, Callable

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_validate
from sklearn.model_selection._split import BaseCrossValidator  # for typing
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


from tqdm.auto import trange
from tqdm.auto import tqdm


import shap
import warnings


import copy
import time

import ai_framework.post_analysis as post


from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve, log_loss, brier_score_loss
)

from .pdp_interpretability import build_external_pdp_signal_allocation_from_results



def add_external_predictions_to_results(
    all_results: Dict[str, List[Dict[str, Any]]],
    model_data_dict: Dict[str, pd.DataFrame],
    *,
    y_col: Optional[str] = None,
    external_tag: str = "external",
    feature_names_key: str = "feature_names_used",
    strict_features: bool = True,
    inplace: bool = True,
    warn_on_skip: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Score model-specific external dataframes with every fold model in all_results
    and store predictions back into each fold record.

    Processes only overlapping model names between all_results and model_data_dict.

    Parameters
    ----------
    all_results:
        Dict mapping model_name -> list of fold-record dicts.

    model_data_dict:
        Dict mapping model_name -> external dataframe for that model.

    y_col:
        Optional label column name present in each model dataframe.
        If provided, external metrics are computed.

    external_tag:
        Prefix used for written keys, e.g. "external" -> y_external_scores.

    feature_names_key:
        Fold-record key containing the selected feature names.
        Default: "feature_names_used"

    strict_features:
        If True, error if a fold record is missing feature_names_key or if
        required columns are missing from the model dataframe.
        If False, and feature_names_key is missing, use all dataframe columns
        except y_col.

    inplace:
        If True, modify all_results in place. If False, shallow-copy records first.

    warn_on_skip:
        If True, warn when models are skipped due to missing overlap.

    Returns
    -------
    Updated all_results dict.
    """
    if not isinstance(model_data_dict, dict):
        raise TypeError("model_data_dict must be a dict of {model_name: dataframe}")

    out = all_results if inplace else {
        model_name: [dict(rec) for rec in recs]
        for model_name, recs in all_results.items()
    }

    all_result_models = set(out.keys())
    data_models = set(model_data_dict.keys())

    overlap_models = sorted(all_result_models & data_models)
    missing_in_data = sorted(all_result_models - data_models)
    extra_in_data = sorted(data_models - all_result_models)

    if not overlap_models:
        raise KeyError(
            "No overlapping model names between all_results and model_data_dict. "
            f"all_results models={sorted(all_result_models)}, "
            f"model_data_dict models={sorted(data_models)}"
        )

    if warn_on_skip and missing_in_data:
        warnings.warn(
            "Skipping models in all_results with no matching dataframe in model_data_dict: "
            f"{missing_in_data}"
        )

    if warn_on_skip and extra_in_data:
        warnings.warn(
            "model_data_dict contains models not present in all_results; they will be ignored: "
            f"{extra_in_data}"
        )

    for model_name in overlap_models:
        fold_records = out[model_name]
        external_df = model_data_dict[model_name]

        if not isinstance(external_df, pd.DataFrame):
            raise TypeError(f"model_data_dict[{model_name!r}] must be a pandas DataFrame")

        if y_col is not None and y_col not in external_df.columns:
            raise KeyError(
                f"model_data_dict[{model_name!r}] is missing y_col={y_col!r}"
            )

        y_ext = None if y_col is None else np.asarray(external_df[y_col])
        has_labels = y_ext is not None
        idx_ext = external_df.index.to_numpy()

        for rec in fold_records:
            if "final_model" not in rec:
                raise KeyError(f"{model_name} record missing 'final_model'")

            selected_feature_names = rec.get(feature_names_key, None)

            if selected_feature_names is None:
                if strict_features:
                    raise KeyError(
                        f"{model_name} record missing {feature_names_key!r}"
                    )
                selected_feature_names = [
                    c for c in external_df.columns
                    if c != y_col
                ]

            selected_feature_names = list(selected_feature_names)

            missing = [c for c in selected_feature_names if c not in external_df.columns]
            if missing:
                raise KeyError(
                    f"{model_name} external dataframe missing required features: {missing}"
                )

            X_ext = external_df.loc[:, selected_feature_names].to_numpy()

            final_model = rec["final_model"]
            p_ext = final_model.predict_proba(X_ext)[:, 1]

            rec[f"{external_tag}_feature_names"] = selected_feature_names
            rec[f"n_{external_tag}"] = int(len(external_df))
            rec[f"{external_tag}_idx"] = idx_ext
            rec[f"y_{external_tag}_scores"] = p_ext

            if rec.get("calibrator_platt", None) is not None:
                rec[f"calib_{external_tag}_predictions_platt"] = (
                    rec["calibrator_platt"].predict_proba(p_ext.reshape(-1, 1))[:, 1]
                )

            if rec.get("calibrator_beta", None) is not None:
                rec[f"calib_{external_tag}_predictions_beta"] = (
                    rec["calibrator_beta"].predict(p_ext)
                )

            if has_labels:
                rec[f"y_{external_tag}"] = y_ext
                rec[f"{external_tag}_metrics"] = {
                    "average_precision": float(average_precision_score(y_ext, p_ext)),
                    "roc_auc": float(roc_auc_score(y_ext, p_ext)),
                }

                pp = rec.get(f"calib_{external_tag}_predictions_platt", None)
                if pp is not None:
                    rec[f"{external_tag}_metrics_platt"] = {
                        "average_precision": float(average_precision_score(y_ext, pp)),
                        "roc_auc": float(roc_auc_score(y_ext, pp)),
                    }

                pb = rec.get(f"calib_{external_tag}_predictions_beta", None)
                if pb is not None:
                    rec[f"{external_tag}_metrics_beta"] = {
                        "average_precision": float(average_precision_score(y_ext, pb)),
                        "roc_auc": float(roc_auc_score(y_ext, pb)),
                    }

    return out



def build_long_predictions_df(
    all_results: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    model_name: str | Sequence[str] | None = None,
    methods: Optional[Sequence[str]] = None,
    include_uncalibrated: bool = True,
    external_idx_key: str = "external_idx",
    external_y_key: str = "y_external",
    external_prob_key_uncalib: str = "y_external_scores",
    external_prob_key_prefix_calib: str = "calib_external_predictions_",
) -> pd.DataFrame:
    """
    Build a long-form predictions dataframe for EXTERNAL predictions only.

    This function converts the external prediction outputs stored inside
    `all_results` into a single tidy dataframe with one row per predicted
    external sample.

    The intended use is after calling your external-scoring function that
    attaches keys such as:
      - y_external_scores
      - calib_external_predictions_beta
      - calib_external_predictions_platt
      - y_external (optional)
      - external_idx (optional)

    Each output row corresponds to one prediction for one external sample
    for a given:
      - model
      - calibration setting
      - trial
      - outer fold

    Parameters
    ----------
    all_results:
        Mapping of:
            model_name -> sequence of fold result dictionaries

        Each fold dictionary is expected to contain the external prediction
        arrays created earlier in your pipeline.

    model_name:
        Which model(s) to include:
          - None: include all models in all_results
          - str: include only that model
          - Sequence[str]: include only the listed models

    methods:
        Calibration methods to include, for example:
            ["beta"]
            ["beta", "platt"]

        If None, only uncalibrated predictions are included when
        include_uncalibrated=True.

    include_uncalibrated:
        If True, include the raw uncalibrated external probabilities using
        `external_prob_key_uncalib`.

    external_idx_key:
        Key in each fold dictionary containing the external sample indices.
        If missing, the function falls back to np.arange(n_external).

    external_y_key:
        Key in each fold dictionary containing the external labels.
        If missing, labels are treated as unavailable and output y is set
        to np.nan.

    external_prob_key_uncalib:
        Key containing uncalibrated external probabilities.

    external_prob_key_prefix_calib:
        Prefix used for calibrated external probabilities.
        For example, if method="beta", the function looks for:
            "calib_external_predictions_beta"

    Returns
    -------
    pd.DataFrame
        Long-form dataframe with columns:
          - model
          - calibration
          - split
          - trial
          - outer_fold
          - idx
          - y
          - p

        Notes:
          - split is always "external"
          - y is float so missing labels can be represented with np.nan
          - p is the predicted probability

    Raises
    ------
    KeyError
        If requested model_name values are not found in all_results.

    ValueError
        If no prediction variants were requested, or if idx / y / p lengths
        do not match for any fold.
    """

    # -------------------------
    # Resolve which model names to include
    # -------------------------
    # If model_name is None, we include every model present in all_results.
    # Otherwise we normalize the input into a list of model names.
    if model_name is None:
        model_names = list(all_results.keys())
    elif isinstance(model_name, str):
        model_names = [model_name]
    else:
        model_names = list(model_name)

    # Validate that every requested model is actually present.
    missing_models = [m for m in model_names if m not in all_results]
    if missing_models:
        raise KeyError(
            f"Model(s) not found in all_results: {missing_models}. "
            f"Available: {list(all_results.keys())}"
        )

    # -------------------------
    # Resolve which calibration settings to include
    # -------------------------
    # We use "calibration" instead of "variant" because it is more explicit:
    #   - "uncalib" means raw model probabilities
    #   - "beta" / "platt" etc. mean calibrated probabilities
    methods_list = [] if methods is None else list(methods)

    calibrations: List[str] = []
    if include_uncalibrated:
        calibrations.append("uncalib")
    calibrations.extend(methods_list)

    # If neither uncalibrated nor calibrated methods were requested,
    # there is nothing to build.
    if not calibrations:
        raise ValueError(
            "No predictions requested. "
            "Set include_uncalibrated=True and/or provide methods."
        )

    # We will accumulate one dict per output row here, then convert to DataFrame.
    rows: List[Dict[str, Any]] = []

    # -------------------------
    # Loop over selected models
    # -------------------------
    for mname in model_names:
        # Each model has a sequence of fold-level result dictionaries.
        folds = all_results[mname]

        # -------------------------
        # Loop over fold records
        # -------------------------
        for r in folds:
            # Trial / outer fold are stored for traceability in the output.
            trial = r.get("trial", None)
            outer_fold = r.get("outer_fold", None)

            # -------------------------
            # Resolve external indices
            # -------------------------
            # Preferred behavior:
            #   use explicitly stored external indices if available.
            # Fallback behavior:
            #   use 0..n_external-1 so the function still works even if
            #   explicit indices were not stored.
            if external_idx_key in r:
                idx_ex = np.asarray(r[external_idx_key], dtype=int)
            else:
                n_ex = int(r.get("n_external", len(r.get(external_prob_key_uncalib, []))))
                idx_ex = np.arange(n_ex, dtype=int)

            # -------------------------
            # Resolve external labels (optional)
            # -------------------------
            # If labels exist, we use them.
            # If not, we keep y as missing (np.nan) in the output.
            y_ex = np.asarray(r[external_y_key], dtype=float) if external_y_key in r else None

            # -------------------------
            # Loop over requested calibration settings
            # -------------------------
            for cal in calibrations:
                # Determine which probability key to read from the fold record.
                if cal == "uncalib":
                    key = external_prob_key_uncalib
                else:
                    key = f"{external_prob_key_prefix_calib}{cal}"

                # If this fold does not contain that calibration output,
                # we silently skip it.
                #
                # Example:
                #   methods=["beta", "platt"]
                # but this record only has beta predictions.
                if key not in r:
                    continue

                # Convert predicted probabilities to a numeric numpy array.
                p_ex = np.asarray(r[key], dtype=float)

                # -------------------------
                # Validate array lengths
                # -------------------------
                # idx and p must always align one-to-one.
                # If labels exist, y must also align with them.
                if y_ex is None:
                    if len(idx_ex) != len(p_ex):
                        raise ValueError(
                            f"Length mismatch for model={mname}, trial={trial}, "
                            f"outer_fold={outer_fold}, calibration={cal}: "
                            f"len(idx)={len(idx_ex)}, len(p)={len(p_ex)}"
                        )
                else:
                    if len(idx_ex) != len(y_ex) or len(idx_ex) != len(p_ex):
                        raise ValueError(
                            f"Length mismatch for model={mname}, trial={trial}, "
                            f"outer_fold={outer_fold}, calibration={cal}: "
                            f"len(idx)={len(idx_ex)}, len(y)={len(y_ex)}, len(p)={len(p_ex)}"
                        )

                # -------------------------
                # Append one output row per external sample
                # -------------------------
                if y_ex is None:
                    # Labels unavailable: y is stored as NaN.
                    for i, pp in zip(idx_ex, p_ex):
                        rows.append(
                            {
                                "model": mname,
                                "calibration": cal,
                                "split": "external",
                                "trial": trial,
                                "outer_fold": outer_fold,
                                "idx": int(i),
                                "y": np.nan,
                                "p": float(pp),
                            }
                        )
                else:
                    # Labels available: store the paired y and probability p.
                    for i, yy, pp in zip(idx_ex, y_ex, p_ex):
                        rows.append(
                            {
                                "model": mname,
                                "calibration": cal,
                                "split": "external",
                                "trial": trial,
                                "outer_fold": outer_fold,
                                "idx": int(i),
                                "y": float(yy),
                                "p": float(pp),
                            }
                        )

    # -------------------------
    # Build final DataFrame
    # -------------------------
    # If no rows were collected, return an empty DataFrame with the expected schema.
    if not rows:
        return pd.DataFrame(
            columns=["model", "calibration", "split", "trial", "outer_fold", "idx", "y", "p"]
        )

    df_long = pd.DataFrame(rows)

    # -------------------------
    # Enforce clean column dtypes
    # -------------------------
    # Keep text columns as strings and numeric columns as numeric types.
    df_long["model"] = df_long["model"].astype(str)
    df_long["calibration"] = df_long["calibration"].astype(str)
    df_long["split"] = "external"
    df_long["idx"] = df_long["idx"].astype(int)
    df_long["y"] = pd.to_numeric(df_long["y"], errors="coerce").astype(float)
    df_long["p"] = pd.to_numeric(df_long["p"], errors="coerce").astype(float)

    # -------------------------
    # Stable sorting for reproducibility
    # -------------------------
    # This makes output order deterministic and easier to debug / compare.
    df_long = df_long.sort_values(
        ["model", "calibration", "split", "trial", "outer_fold", "idx"],
        kind="mergesort",
    ).reset_index(drop=True)

    return df_long


def aggregate_predictions_by_idx(
    df_long: pd.DataFrame,
    *,
    model_name: str | Sequence[str] | None = None,
    calibrations: Optional[Sequence[str]] = None,
    agg_stats: Sequence[str] = ("mean", "median", "std", "min", "max"),
    add_y_label: bool = True,
    prevalence: Union[bool, float] = True,
    add_ensemble: bool = True,
    ensemble_name: str = "Ensemble model",
    ensemble_models: Sequence[str] | None = None,
    truncate_decimals: Optional[int] = None,
) -> pd.DataFrame:
    """
    Aggregate repeated EXTERNAL predictions per idx into a single row per
    (model, calibration, idx), and optionally add an ensemble model by pooling
    predictions across models.

    This function is designed to consume the output of the simplified
    `build_long_predictions_df(...)`, where df_long contains EXTERNAL predictions
    only and columns like:
        ["model", "calibration", "split", "trial", "outer_fold", "idx", "y", "p"]

    Because nested CV produces repeated predictions for the same external sample
    across trials / outer folds, this function collapses those repeated predictions
    into summary statistics per idx.

    Parameters
    ----------
    df_long:
        Long-form dataframe containing at least:
            ["model", "calibration", "idx", "y", "p"]

        Optional columns such as "split", "trial", and "outer_fold" may also be
        present, but are not required for the aggregation itself.

    model_name:
        Which models to include:
          - None: include all models in df_long
          - str: include only that model
          - Sequence[str]: include only those models

    calibrations:
        Which calibration settings to include, e.g.:
            ["uncalib", "beta"]
        If None, use all calibration values present in df_long.

    agg_stats:
        Which summary statistics to compute over repeated probabilities p.
        Supported values:
            "mean", "median", "std", "min", "max"

    add_y_label:
        If True and labels exist, add y_label using:
            0 -> "0 (neg)"
            1 -> "1 (pos)"

    prevalence:
        Controls whether to add prevalence_used:
          - True: compute prevalence per model from unique labeled idx
          - False: do not add prevalence_used
          - float: use the provided prevalence value for all rows

    add_ensemble:
        If True, append an ensemble "model" by pooling predictions across
        multiple models for each (calibration, idx).

    ensemble_name:
        Name to assign to the pooled ensemble rows in df_agg["model"].

    ensemble_models:
        Which models to pool for the ensemble.
        If None, pool all selected models after model_name filtering.

    truncate_decimals:
        If not None, truncate probability-style output columns to this many decimal
        places after all calculations are complete. This is truncation, not rounding.

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe with one row per:
            (model, calibration, idx)

        Includes:
          - y
          - n_preds
          - p_mean / p_median / p_std / p_min / p_max (depending on agg_stats)
          - optional y_label
          - optional prevalence_used
          - split="external" for consistency
    """
    def _truncate_decimals(x: float, decimals: int):
        if pd.isna(x):
            return x
        factor = 10 ** decimals
        return np.trunc(float(x) * factor) / factor


    # ---------------------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------------------
    # d = df_long.copy()

    # # Older/newer prediction tables may use either:
    # #   variant      = uncalib / beta / isotonic / ...
    # #   calibration  = uncalib / beta / isotonic / ...
    # #
    # # Internally, this aggregation function uses "calibration".
    # if "calibration" not in d.columns:
    #     if "variant" in d.columns:
    #         d["calibration"] = d["variant"]
    #     else:
    #         raise KeyError(
    #             "df_long is missing required calibration column. "
    #             "Expected either 'calibration' or 'variant'."
    #         )

    # required = {"model", "calibration", "idx", "y", "p"}
    # missing = required - set(d.columns)

    # if missing:
    #     raise KeyError(
    #         f"df_long is missing required columns: {sorted(missing)}"
    #     )
    
    required = {"model", "calibration", "idx", "y", "p"}
    missing = required - set(df_long.columns)
    if missing:
        raise KeyError(
            f"df_long is missing required columns: {sorted(missing)}"
        )

    # Work on a copy so we do not modify the caller's dataframe.
    d = df_long.copy()

    # ---------------------------------------------------------------------
    # Filter models if requested
    # ---------------------------------------------------------------------
    if model_name is None:
        selected_models = sorted(d["model"].astype(str).unique().tolist())
    elif isinstance(model_name, str):
        selected_models = [model_name]
    else:
        selected_models = list(model_name)

    d["model"] = d["model"].astype(str)
    d = d[d["model"].isin(selected_models)].copy()

    if d.empty:
        raise ValueError(f"No rows found after filtering model_name={model_name}.")

    # ---------------------------------------------------------------------
    # Filter calibrations if requested
    # ---------------------------------------------------------------------
    d["calibration"] = d["calibration"].astype(str)

    if calibrations is None:
        selected_calibrations = sorted(d["calibration"].unique().tolist())
    else:
        selected_calibrations = list(calibrations)

    d = d[d["calibration"].isin(selected_calibrations)].copy()

    if d.empty:
        raise ValueError(
            f"No rows found after filtering calibrations={selected_calibrations}."
        )

    # ---------------------------------------------------------------------
    # Normalize dtypes
    # ---------------------------------------------------------------------
    d["idx"] = pd.to_numeric(d["idx"], errors="coerce").astype(int)
    d["p"] = pd.to_numeric(d["p"], errors="coerce").astype(float)
    d["y"] = pd.to_numeric(d["y"], errors="coerce").astype(float)

    # ---------------------------------------------------------------------
    # Helper to carry forward the first non-missing y value within a group
    # ---------------------------------------------------------------------
    def _first_non_nan(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce")
        x = x[~x.isna()]
        return float(x.iloc[0]) if len(x) else np.nan

    # ---------------------------------------------------------------------
    # Define the aggregation operations
    # ---------------------------------------------------------------------
    agg_dict = {
        "y": ("y", _first_non_nan),
        "n_preds": ("p", "size"),
    }

    if "mean" in agg_stats:
        agg_dict["p_mean"] = ("p", "mean")
    if "median" in agg_stats:
        agg_dict["p_median"] = ("p", "median")
    if "std" in agg_stats:
        agg_dict["p_std"] = ("p", "std")
    if "min" in agg_stats:
        agg_dict["p_min"] = ("p", "min")
    if "max" in agg_stats:
        agg_dict["p_max"] = ("p", "max")

    # ---------------------------------------------------------------------
    # Aggregate repeated predictions per (model, calibration, idx)
    # ---------------------------------------------------------------------
    grp = d.groupby(
        ["model", "calibration", "idx"],
        as_index=False,
        observed=False,
    )

    df_agg = grp.agg(**agg_dict)

    # Keep split for consistency with the rest of the pipeline.
    df_agg["split"] = "external"

    # ---------------------------------------------------------------------
    # Optionally create an ensemble by pooling predictions across models
    # ---------------------------------------------------------------------
    if add_ensemble:
        if ensemble_models is None:
            pool_models = sorted(d["model"].unique().tolist())
        else:
            available_models = set(d["model"].unique())
            pool_models = [m for m in ensemble_models if m in available_models]

        if len(pool_models) == 0:
            raise ValueError(
                "add_ensemble=True but no models available to pool. "
                "Check ensemble_models / model_name filters."
            )

        # Restrict to only the models selected for ensemble pooling.
        d_pool = d[d["model"].isin(pool_models)].copy()

        # Pool across models by grouping only on calibration and idx.
        grp_e = d_pool.groupby(
            ["calibration", "idx"],
            as_index=False,
            observed=False,
        )

        df_e = grp_e.agg(**agg_dict)
        df_e.insert(0, "model", ensemble_name)
        df_e["split"] = "external"

        # Append ensemble rows to the main aggregated dataframe.
        df_agg = pd.concat([df_agg, df_e], ignore_index=True)

    # ---------------------------------------------------------------------
    # Ensure prediction summary columns are numeric floats
    # ---------------------------------------------------------------------
    for c in df_agg.columns:
        if c.startswith("p_"):
            df_agg[c] = pd.to_numeric(df_agg[c], errors="coerce").astype(float)

    # ---------------------------------------------------------------------
    # Add y_label only when labels actually exist
    # ---------------------------------------------------------------------
    labels_exist = df_agg["y"].notna().any()

    if add_y_label:
        if labels_exist:
            y_map = {
                0.0: "0 (neg)",
                1.0: "1 (pos)",
            }
            df_agg["y_label"] = df_agg["y"].map(y_map)
            df_agg["y_label"] = pd.Categorical(
                df_agg["y_label"],
                categories=["0 (neg)", "1 (pos)"],
                ordered=True,
            )
        else:
            df_agg["y_label"] = np.nan

    # ---------------------------------------------------------------------
    # Add prevalence_used if requested
    # ---------------------------------------------------------------------
    if prevalence is not False:
        if isinstance(prevalence, bool):
            if prevalence is True and labels_exist:
                # Compute prevalence per model using unique labeled idx.
                base = (
                    df_agg[df_agg["y"].notna()]
                    .drop_duplicates(["model", "idx"])[["model", "y"]]
                )

                prev_map = base.groupby("model")["y"].mean().to_dict()

                df_agg["prevalence_used"] = [
                    float(prev_map.get(m, np.nan))
                    for m in df_agg["model"]
                ]
            else:
                df_agg["prevalence_used"] = np.nan
        else:
            prev_val = float(prevalence)
            if not (0.0 <= prev_val <= 1.0):
                raise ValueError(f"prevalence must be in [0,1]; got {prev_val}")
            df_agg["prevalence_used"] = prev_val

    # ---------------------------------------------------------------------
    # Apply truncation to probability-style columns at the very end
    # ---------------------------------------------------------------------
    if truncate_decimals is not None:
        if truncate_decimals < 0:
            raise ValueError("truncate_decimals must be >= 0 or None.")
        prob_cols = [c for c in df_agg.columns if c.startswith("p_")]
        df_agg[prob_cols] = df_agg[prob_cols].apply(
            lambda s: s.map(lambda x: _truncate_decimals(x, truncate_decimals))
        )

    # ---------------------------------------------------------------------
    # Stable sort for reproducibility
    # ---------------------------------------------------------------------
    df_agg = df_agg.sort_values(
        ["model", "calibration", "idx"],
        kind="mergesort",
    ).reset_index(drop=True)

    return df_agg



def compute_logloss_brier_from_df_agg(
    df_agg: pd.DataFrame,
    *,
    split: str | Sequence[str] = "test",
    pred_col: str = "p_mean",
    calibration: Optional[Sequence[str]] = None,
    model_names: str | Sequence[str] | None = None,
    method_alias: Mapping[str, str] | None = None,
    prevalence_col: str | None = "prevalence_used",
    eps: float = 1e-15,
) -> pd.DataFrame:
    """
    Compute Log Loss and Brier score from an aggregated per-idx predictions table (df_agg),
    and also compute prevalence-only baselines for each metric.

    Expected df_agg columns (minimum):
      - model, calibration, split, idx, y, <pred_col>
    Optional:
      - prevalence_used (or a user-specified prevalence_col)

    Label handling
    --------------
    Metrics require labels. Rows with y=NaN are ignored. If nothing labeled remains after
    filtering, raises ValueError.

    Baselines
    ---------
    For each (calibration, split) subset we compute a baseline prevalence π from:
      1) prevalence_col (if provided and present and non-null), else
      2) π = mean(y) on unique idx in that subset.

    Baseline metrics:
      - baseline_log_loss = -[π log(π) + (1-π) log(1-π)]
      - baseline_brier    = π(1-π)

    Returns
    -------
    pd.DataFrame with columns:
      ["model","model_label","calibration","split","n_labeled","prevalence_used",
       "log_loss","brier","baseline_log_loss","baseline_brier"]
    """
    if method_alias is None:
        method_alias = {}

    required = {"model", "calibration", "split", "idx", "y", pred_col}
    missing = required - set(df_agg.columns)
    if missing:
        raise KeyError(f"df_agg is missing required columns: {sorted(missing)}")

    d = df_agg.copy()

    # ---- split filter ----
    splits = [split] if isinstance(split, str) else list(split)
    d = d[d["split"].isin(splits)].copy()
    if d.empty:
        raise ValueError(f"No rows found for split(s)={splits}.")

    # ---- model filter ----
    if model_names is not None:
        mlist = [model_names] if isinstance(model_names, str) else list(model_names)
        d = d[d["model"].isin(mlist)].copy()
        if d.empty:
            raise ValueError(f"No rows found after filtering model_names={mlist} for split(s)={splits}.")

    # ---- calibration filter ----
    if calibration is None:
        calibration = sorted(d["calibration"].astype(str).unique().tolist())
    else:
        calibration = list(calibration)
    d = d[d["calibration"].isin(calibration)].copy()
    if d.empty:
        raise ValueError(f"No rows found after filtering calibration={calibration}.")

    # types
    d["idx"] = pd.to_numeric(d["idx"], errors="coerce").astype(int)
    d["y"] = pd.to_numeric(d["y"], errors="coerce").astype(float)
    d[pred_col] = pd.to_numeric(d[pred_col], errors="coerce").astype(float)

    # display labels
    d["model_label"] = d["model"].map(lambda m: method_alias.get(str(m), str(m))).astype(str)

    # ---- compute prevalence baseline per (calibration, split) ----
    prev_map: dict[tuple[str, str], float] = {}

    for (v, s), sub_vs in d.groupby(["calibration", "split"], observed=False):
        sub_l = sub_vs[sub_vs["y"].notna()].drop_duplicates(["idx"]).copy()
        if sub_l.empty:
            continue

        prev_val: Optional[float] = None
        if prevalence_col is not None and prevalence_col in sub_vs.columns:
            # if prevalence_col exists, use a robust representative value if present
            cand = pd.to_numeric(sub_vs[prevalence_col], errors="coerce").dropna()
            if len(cand) > 0:
                prev_val = float(cand.iloc[0])

        if prev_val is None:
            prev_val = float(sub_l["y"].mean())

        prev_val = float(np.clip(prev_val, eps, 1.0 - eps))
        prev_map[(str(v), str(s))] = prev_val

    # ---- compute per-model metrics ----
    out_rows: list[dict[str, Any]] = []

    for (m, mlabel, v, s), sub in d.groupby(["model", "model_label", "calibration", "split"], observed=False):
        sub = sub.drop_duplicates(["idx"])  # safety
        sub_l = sub[sub["y"].notna()].copy()
        if sub_l.empty:
            continue

        y = sub_l["y"].astype(int).to_numpy()
        p = np.clip(sub_l[pred_col].to_numpy(dtype=float), eps, 1.0 - eps)

        ll = float(log_loss(y, p, labels=[0, 1]))
        br = float(brier_score_loss(y, p))

        pi = prev_map.get((str(v), str(s)), float(np.clip(float(sub_l["y"].mean()), eps, 1.0 - eps)))
        baseline_ll = float(-(pi * np.log(pi) + (1.0 - pi) * np.log(1.0 - pi)))
        baseline_br = float(pi * (1.0 - pi))

        out_rows.append(
            dict(
                model=str(m),
                model_label=str(mlabel),
                calibration=str(v),
                split=str(s),
                n_labeled=int(len(y)),
                prevalence_used=float(pi),
                log_loss=ll,
                brier=br,
                baseline_log_loss=baseline_ll,
                baseline_brier=baseline_br,
            )
        )

    df_metrics = pd.DataFrame(out_rows)
    if df_metrics.empty:
        raise ValueError(
            "No labeled rows available to compute log loss / brier after filtering. "
            "If this is external-unlabeled, that's expected."
        )

    df_metrics = df_metrics.sort_values(
        ["split", "calibration", "model_label"],
        kind="mergesort",
    ).reset_index(drop=True)

    return df_metrics


def plot_logloss_brier_from_df_agg(
    df_agg: pd.DataFrame,
    *,
    split: str | Sequence[str] = "test",
    pred_col: str = "p_mean",
    calibration: Optional[Sequence[str]] = None,
    model_names: str | Sequence[str] | None = None,
    method_alias: Mapping[str, str] | None = None,
    model_palette: Mapping[str, str] | None = None,  # keys should be *model_label*
    prevalence_col: str | None = "prevalence_used",
    figsize: tuple[float, float] = (7, 5),
    font_size: float = 12.0,
    x_tick_rotation: int = 0,
    baseline_color: str = "#D5F713",
    baseline_lw: float = 1.5,
    baseline_ls: str = "--",
    show_calibration_legend: bool | None = None,  # auto: show if len(calibration)>1
    legend_loc: str = "best",
    # y-lims
    logloss_ylim: tuple[float, float] | None = None,
    brier_ylim: tuple[float, float] | None = None,
    annotate: bool = True,
    annotate_decimals: int = 3,
    annotate_font_size: float | None = None,
    annotate_offset: float = 0.01,
) -> pd.DataFrame:
    """
    Barplot Log Loss and Brier score from df_agg (two separate figures), including
    prevalence-only baseline for each metric.

    - Colors are applied per model_label using model_palette.
    - By default, no legend is shown for models (x-axis already labels them).
      If multiple calibration are provided, a calibration legend is shown unless disabled.

    Returns the metrics dataframe (same as compute_logloss_brier_from_df_agg).
    """

    sns.set(style="whitegrid")

    if method_alias is None:
        method_alias = {}

    df_metrics = compute_logloss_brier_from_df_agg(
        df_agg,
        split=split,
        pred_col=pred_col,
        calibration=calibration,
        model_names=model_names,
        method_alias=method_alias,
        prevalence_col=prevalence_col,
    )

    # Decide whether to show calibration legend
    uniq_calibration = sorted(df_metrics["calibration"].unique().tolist())
    if show_calibration_legend is None:
        show_calibration_legend = len(uniq_calibration) > 1

    # Prepare palette (by model label)
    model_labels = df_metrics["model_label"].tolist()
    uniq_models = list(dict.fromkeys(model_labels))  # stable order as seen
    if model_palette is None:
        # fallback colors (matplotlib cycle) — user typically supplies this
        model_palette = {m: None for m in uniq_models}

    # stable ordering for x
    model_order = uniq_models

    def _plot(metric_col: str, baseline_col: str, title: str, ylim: tuple[float, float] | None):
        # aggregate over calibration? no: keep calibration-separated bars if multiple calibration
        # but most of your use is calibration=["beta"], so it becomes single bar per model.
        plot_df = df_metrics.copy()
        plot_df["model_label"] = pd.Categorical(plot_df["model_label"], categories=model_order, ordered=True)

        # If multiple calibration, we plot grouped bars (calibration within model).
        # If single calibration, no grouping needed.
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(model_order), dtype=float)

        if len(uniq_calibration) == 1:
            v = uniq_calibration[0]
            sub = plot_df[plot_df["calibration"] == v].sort_values("model_label")

            heights = sub[metric_col].to_numpy(dtype=float)

            colors = [model_palette.get(m, None) for m in sub["model_label"].astype(str).tolist()]
            bars = ax.bar(x, heights, color=colors)

            # Baseline: same for all models within (calibration, split) (by construction)
            base_val = float(sub[baseline_col].iloc[0])
            ax.axhline(base_val, color=baseline_color, lw=baseline_lw, ls=baseline_ls, label=f"Baseline = {base_val:.3f}")

            ax.set_xticks(x)
            ax.set_xticklabels(sub["model_label"].astype(str).tolist(), rotation=x_tick_rotation, fontsize=font_size, fontweight="bold")

            if annotate:
                ann_fs = annotate_font_size if annotate_font_size is not None else max(8.0, float(font_size) - 3.0)
                for bar, val in zip(bars, heights):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        float(val) + float(annotate_offset),
                        f"{val:.{annotate_decimals}f}",
                        ha="center",
                        va="bottom",
                        fontsize=ann_fs,
                        fontweight="bold",
                    )

            # no model legend (x-axis already labels models)
            # baseline legend only
            ax.legend(loc=legend_loc, prop={"size": font_size, "weight": "bold"}, title="")

        else:
            # grouped bars by calibration (legend for calibration is useful)
            width = 0.8 / max(1, len(uniq_calibration))
            for j, v in enumerate(uniq_calibration):
                sub = plot_df[plot_df["calibration"] == v].sort_values("model_label")
                heights = sub[metric_col].to_numpy(dtype=float)
                xj = x - 0.4 + width / 2.0 + j * width

                # color by model, but calibration differ by bar position, not color.
                colors = [model_palette.get(m, None) for m in sub["model_label"].astype(str).tolist()]
                bars = ax.bar(xj, heights, width=width, color=colors, label=v)

                if annotate:
                    ann_fs = annotate_font_size if annotate_font_size is not None else max(8.0, float(font_size) - 3.0)
                    for bar, val in zip(bars, heights):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            float(val) + float(annotate_offset),
                            f"{val:.{annotate_decimals}f}",
                            ha="center",
                            va="bottom",
                            fontsize=ann_fs,
                            fontweight="bold",
                        )

                # baseline line per calibration (usually same across calibration if same labels;
                # but we keep it correct in case you pass subsets later)
                base_val = float(sub[baseline_col].iloc[0])
                ax.axhline(base_val, color=baseline_color, lw=baseline_lw, ls=baseline_ls)

            ax.set_xticks(x)
            ax.set_xticklabels(model_order, rotation=x_tick_rotation, fontsize=font_size, fontweight="bold")

            if show_calibration_legend:
                ax.legend(loc=legend_loc, prop={"size": font_size, "weight": "bold"}, title="")

        ax.set_title(title, fontsize=font_size + 2, fontweight="bold")
        ax.set_xlabel("Model", fontsize=font_size, fontweight="bold")
        ax.set_ylabel(metric_col.replace("_", " ").title(), fontsize=font_size, fontweight="bold")

        ax.tick_params(axis="y", labelsize=font_size)
        for lab in ax.get_yticklabels():
            lab.set_fontweight("bold")

        if ylim is not None:
            ax.set_ylim(*ylim)

        plt.tight_layout()
        plt.show()

    split_title = split if isinstance(split, str) else ",".join(map(str, split))
    _plot("log_loss", "baseline_log_loss", f"Log loss across models", logloss_ylim)
    _plot("brier", "baseline_brier", f"Brier score across models", brier_ylim)

    return df_metrics



def plot_auroc_auprc_from_df_agg(
    df_agg: pd.DataFrame,
    *,
    split: str = "external",
    pred_col: str = "p_mean",
    prevalence_col: str = "prevalence_used",
    calibration: Optional[Sequence[str]] = None,

    # --- labeling / styling ---
    method_alias: Optional[Mapping[str, str]] = None,      # model_key -> display label
    model_palette: Optional[Mapping[str, str]] = None,     # display label -> color
    figsize: tuple[float, float] = (7, 5),
    font_size: float = 12.0,
    legend_loc: str = "best",
    x_tick_rotation: int = 0,

    # --- baselines ---
    show_prevalence_baseline: bool = True,
    baseline_color: str = "#D5F713",
    baseline_lw: float = 1.5,
    baseline_ls: str = "--",

    # --- y-lims ---
    auprc_ylim: Optional[tuple[float, float]] = None,
    auroc_ylim: Optional[tuple[float, float]] = None,

    # --- annotation ---
    annotate: bool = True,
    annotate_decimals: int = 3,
    annotate_font_size: Optional[float] = None,
    annotate_offset: float = 0.015,
) -> pd.DataFrame:
    """
    Compute and plot AUROC and AUPRC across models from an *already aggregated* prediction table (df_agg).

    This is designed to consume the output of your `aggregate_predictions_by_idx(...)` (or equivalent),
    where each row corresponds to one unit (idx) for a given (model, calibration, split) and contains:
      - predicted probability summary (e.g., p_mean) in `pred_col`
      - optional labels in column `y` (may be NaN for unlabeled external)
      - optional prevalence baseline value in `prevalence_col` (often repeated across rows)

    Behavior
    --------
    - If labels are present (at least one non-NaN y), computes AUROC/AUPRC for each (model, calibration)
      within the requested split using (y, pred_col).
    - If labels are missing (all y NaN), returns a metrics table with NaN metrics and does not
      error (plots will be skipped because metrics can’t be computed).
    - Plots two bar charts:
        1) AUPRC across models (baseline = prevalence if available)
        2) AUROC across models (baseline = 0.50 chance)

    Notes on plotting
    -----------------
    - X-axis shows model display labels. There is NO model legend (since x labels already identify models).
    - Bar colors come from `model_palette` keyed by display label. If not provided, matplotlib defaults.

    Parameters
    ----------
    df_agg:
        Aggregated table with columns: ["model","calibration","split","idx","y", pred_col, prevalence_col(optional)].

    split:
        Which split to evaluate "external"

    pred_col:
        Which probability column to evaluate (e.g., "p_mean").

    prevalence_col:
        Column containing prevalence baseline value (used only for AUPRC baseline). If missing or NaN,
        AUPRC baseline is skipped.

    calibration:
        Which calibration to include. If None, uses all calibration in df_agg for that split.

    method_alias:
        Optional mapping model_key -> display label (used on x-axis and for model_palette lookup).

    model_palette:
        Optional mapping display label -> color.

    Returns
    -------
    pd.DataFrame
        Metrics table with one row per (model, calibration, split), columns:
          ["model","model_display","calibration","split","n","prevalence","auprc","auroc"]
    """

    sns.set(style="whitegrid")
    
    required = {"model", "calibration", "split", "idx", "y", pred_col}
    missing = required - set(df_agg.columns)
    if missing:
        raise KeyError(f"df_agg missing required columns: {sorted(missing)}")

    if method_alias is None:
        method_alias = {}

    d = df_agg.copy()
    d = d[d["split"] == split].copy()
    if d.empty:
        raise ValueError(f"No rows found in df_agg for split='{split}'.")

    # calibration filter
    if calibration is None:
        calibration = sorted(d["calibration"].astype(str).unique().tolist())
    else:
        calibration = list(calibration)
    d = d[d["calibration"].isin(calibration)].copy()
    if d.empty:
        raise ValueError(f"No rows found after filtering calibration={calibration} for split='{split}'.")

    # types
    d["model"] = d["model"].astype(str)
    d["calibration"] = d["calibration"].astype(str)
    d["idx"] = pd.to_numeric(d["idx"], errors="coerce").astype("Int64")
    d[pred_col] = pd.to_numeric(d[pred_col], errors="coerce").astype(float)
    d["y"] = pd.to_numeric(d["y"], errors="coerce").astype(float)

    labels_exist = d["y"].notna().any()

    # model display labels
    d["model_display"] = d["model"].map(lambda m: method_alias.get(m, m)).astype(str)

    # Compute prevalence for baseline (for this split) if available
    prev_val: float | None = None
    if show_prevalence_baseline:
        if prevalence_col in d.columns:
            # take first non-nan (they are typically repeated)
            pv = pd.to_numeric(d[prevalence_col], errors="coerce")
            pv = pv[pv.notna()]
            if len(pv):
                prev_val = float(pv.iloc[0])

    # -------------------------
    # Compute metrics per (model, calibration)
    # -------------------------
    rows = []
    for (m, v), sub in d.groupby(["model", "calibration"], observed=False):
        sub_labeled = sub[sub["y"].notna()].copy()

        n = int(sub_labeled["idx"].nunique()) if labels_exist else int(sub["idx"].nunique())

        if not labels_exist or sub_labeled.empty:
            rows.append(
                {
                    "model": m,
                    "model_display": method_alias.get(m, m),
                    "calibration": v,
                    "split": split,
                    "n": n,
                    "prevalence": np.nan,
                    "auprc": np.nan,
                    "auroc": np.nan,
                }
            )
            continue

        # one row per idx already, but be safe:
        sub_u = sub_labeled.drop_duplicates("idx")[["y", pred_col]]

        y_true = sub_u["y"].astype(int).to_numpy()
        y_score = sub_u[pred_col].astype(float).to_numpy()

        # prevalence from labeled unique idx
        prevalence = float(np.mean(y_true)) if len(y_true) else np.nan

        # metrics
        auprc = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan
        auroc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan

        rows.append(
            {
                "model": m,
                "model_display": method_alias.get(m, m),
                "calibration": v,
                "split": split,
                "n": int(sub_u.shape[0]),
                "prevalence": prevalence,
                "auprc": auprc,
                "auroc": auroc,
            }
        )

    df_metrics = pd.DataFrame(rows)

    # If no labels, just return metrics table (no plot)
    if not labels_exist:
        return df_metrics

    # -------------------------
    # Plot helpers
    # -------------------------
    def _barplot_single_variant(
        metric_col: Literal["auprc", "auroc"],
        title: str,
        ylim: Optional[tuple[float, float]],
    ) -> None:
        # Expecting one calibration or multiple; plot each calibration separately (simple + explicit)
        # Here: we’ll plot a grouped-by-calibrationt bar chart if more than 1 calibration.
        plot_df = df_metrics.copy()

        # Order by display label (stable)
        model_order = plot_df["model_display"].unique().tolist()

        calibration_order = calibration if calibration is not None else sorted(plot_df["calibration"].unique().tolist())

        # build bar positions
        x = np.arange(len(model_order), dtype=float)
        n_var = len(calibration_order)
        width = 0.8 / max(1, n_var)

        fig, ax = plt.subplots(figsize=figsize)

        for j, v in enumerate(calibration_order):
            sub = plot_df[plot_df["calibration"] == v].copy()
            sub = sub.set_index("model_display").reindex(model_order).reset_index()

            vals = sub[metric_col].to_numpy(dtype=float)

            # colors by model label (NOT by calibration)
            if model_palette is not None:
                colors = [model_palette.get(lbl, None) for lbl in sub["model_display"].tolist()]
            else:
                colors = None

            xpos = x + (j - (n_var - 1) / 2.0) * width

            bars = ax.bar(
                xpos,
                vals,
                width=width,
                label=v if n_var > 1 else None,  # only show calibration legend if multiple calibration
                color=colors,
            )

            if annotate:
                ann_fs = annotate_font_size if annotate_font_size is not None else max(8.0, float(font_size) - 3.0)
                for b, val in zip(bars, vals):
                    if np.isnan(val):
                        continue
                    ax.text(
                        b.get_x() + b.get_width() / 2.0,
                        float(val) + float(annotate_offset),
                        f"{val:.{annotate_decimals}f}",
                        ha="center",
                        va="bottom",
                        fontsize=ann_fs,
                        fontweight="bold",
                    )

        # Baselines:
        baseline_handle = None
        baseline_label = None

        if show_prevalence_baseline:
            if metric_col == "auprc":
                if prev_val is not None:
                    baseline_label = f"Baseline = {prev_val:.2f}"
                    baseline_handle = ax.axhline(
                        prev_val, color=baseline_color, lw=baseline_lw, ls=baseline_ls, label=baseline_label
                    )
            elif metric_col == "auroc":
                chance = 0.5
                baseline_label = f"Baseline = {chance:.2f}"
                baseline_handle = ax.axhline(
                    chance, color=baseline_color, lw=baseline_lw, ls=baseline_ls, label=baseline_label
                )

        ax.set_title(title, fontsize=font_size + 1, fontweight="bold")
        ax.set_xlabel("Model", fontsize=font_size, fontweight="bold")
        ax.set_ylabel(metric_col.upper(), fontsize=font_size, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=x_tick_rotation, fontsize=font_size, fontweight="bold")
        ax.tick_params(axis="y", labelsize=font_size)
        for lab in ax.get_yticklabels():
            lab.set_fontweight("bold")

        # y-lims
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            # light auto padding
            top = np.nanmax(plot_df[metric_col].to_numpy(dtype=float))
            if metric_col == "auroc":
                top = np.nanmax([top, 0.5])
            if metric_col == "auprc" and prev_val is not None:
                top = np.nanmax([top, prev_val])
            ax.set_ylim(0.0, min(1.10, float(top) + 0.08))

        # Legend:
        # - NO model legend (models are x-axis labels)
        # - Only show legend if multiple calibration OR baseline exists
        handles, labels = ax.get_legend_handles_labels()
        keep_H, keep_L = [], []

        # keep calibration legend only if we have >1 calibration
        if n_var > 1:
            # keep unique calibration labels
            seen = set()
            for h, l in zip(handles, labels):
                if l in calibration_order and l not in seen:
                    seen.add(l)
                    keep_H.append(h)
                    keep_L.append(l)

        # always include baseline if present
        if baseline_handle is not None and baseline_label is not None:
            keep_H.append(baseline_handle)
            keep_L.append(baseline_label)

        if len(keep_H) > 0:
            ax.legend(keep_H, keep_L, loc=legend_loc, prop={"size": font_size, "weight": "bold"}, title="")

        fig.tight_layout()
        plt.show()

    # -------------------------
    # Plot AUPRC + AUROC
    # -------------------------
    _barplot_single_variant(
        metric_col="auprc",
        title=f"AUPRC across models",
        ylim=auprc_ylim,
    )
    _barplot_single_variant(
        metric_col="auroc",
        title=f"AUROC across models",
        ylim=auroc_ylim,
    )

    return df_metrics



def barplot_balanced_accuracy_from_agg(
    df_agg: pd.DataFrame,
    *,
    model_names: str | Sequence[str] | None = None,
    calibration: str | None = None,
    prob_col: str = "p_mean",
    evaluation_split: str = "external",

    # --- threshold handling ---
    threshold_value: float | Mapping[str, float] | None = None,
    fallback_threshold: float = 0.50,

    # --- labels / aliasing ---
    method_alias: Mapping[str, str] | None = None,

    # --- styling ---
    figsize: tuple[float, float] = (9.0, 5.0),
    font_size: float = 12.0,
    x_tick_rotation: int = 0,
    bar_color: str = "#2E9B4E",
    bar_width: float = 0.55,

    # --- baseline ---
    show_baseline: bool = True,
    baseline_value: float = 0.50,
    baseline_color: str = "#D5F713",
    baseline_lw: float = 1.5,
    baseline_ls: str = "--",

    # --- annotation ---
    annotate: bool = True,
    annotate_decimals: int = 3,
    annotate_font_size: float | None = None,
    annotate_offset: float = 0.015,

    # --- y limits ---
    ylim: tuple[float, float] | None = None,

    # --- console threshold summary ---
    print_threshold_summary: bool = True,
) -> pd.DataFrame:
    """
    Plot balanced accuracy from an aggregated predictions table (`df_agg`) and
    return a per-model summary table.

    This function evaluates balanced accuracy on ONE split (typically "external")
    using an explicitly supplied classification threshold.

    Expected input
    --------------
    `df_agg` should contain one row per aggregated unit (for example one patient)
    for a given:
      - model
      - calibration
      - split
      - idx

    and should include:
      - a binary label column `y`
      - a probability column such as `p_mean`

    Threshold behavior
    ------------------
    Thresholds are handled in this order:

    1) If `threshold_value` is a float:
         use that same threshold for all models.

    2) If `threshold_value` is a mapping:
         use the threshold for each model by model name.
         Example:
             {
                 "logistic_regression": 0.41,
                 "xgboost": 0.53,
                 "Ensemble model": 0.47,
             }

    3) If `threshold_value` is None:
         use `fallback_threshold` for all models.

    Parameters
    ----------
    df_agg:
        Aggregated predictions table with columns including:
            ["model", "calibration", "split", "idx", "y", prob_col]

    model_names:
        Which model(s) to include:
          - None: include all models in df_agg
          - str: include only that model
          - Sequence[str]: include only those models

    calibration:
        Which calibration setting to use (for example "beta").
        If None, the function expects only one calibration to remain after filtering.

    prob_col:
        Probability column used to create hard predictions via thresholding.

    evaluation_split:
        Which split to evaluate balanced accuracy on.
        In your current workflow this is usually "external".

    threshold_value:
        Threshold specification.
          - float: same threshold for every model
          - mapping: per-model thresholds
          - None: use `fallback_threshold`

    fallback_threshold:
        Threshold used when `threshold_value` is None, or when `threshold_value`
        is a mapping and a model is missing from that mapping.

    method_alias:
        Optional mapping from internal model names to display labels.

    figsize, font_size, x_tick_rotation:
        Standard plotting controls.

    bar_color:
        Bar color used for all models.

    bar_width:
        Width of the bars.

    show_baseline:
        Whether to draw a horizontal baseline reference line.

    baseline_value:
        Y-value of the baseline line. For balanced accuracy this is usually 0.50.

    baseline_color, baseline_lw, baseline_ls:
        Styling for the baseline line.

    annotate:
        If True, annotate each bar with its balanced accuracy value.

    annotate_decimals:
        Number of decimals in bar annotations.

    annotate_font_size:
        Font size for annotations. If None, derived from `font_size`.

    annotate_offset:
        Vertical offset above each bar annotation.

    ylim:
        Optional y-axis limits.

    print_threshold_summary:
        If True, print per-model threshold summaries to the console.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per model, including:
          - model
          - model_label
          - calibration
          - evaluation_split
          - balanced_accuracy
          - threshold
          - n
    """
    sns.set(style="whitegrid")

    # ------------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------------
    required = {"model", "calibration", "split", "idx", "y", prob_col}
    missing = required - set(df_agg.columns)
    if missing:
        raise KeyError(f"df_agg is missing required columns: {sorted(missing)}")

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    if method_alias is None:
        method_alias = {}

    # ------------------------------------------------------------------
    # Copy and normalize dtypes
    # ------------------------------------------------------------------
    d = df_agg.copy()
    d["model"] = d["model"].astype(str)
    d["calibration"] = d["calibration"].astype(str)
    d["split"] = d["split"].astype(str)
    d["idx"] = pd.to_numeric(d["idx"], errors="coerce").astype(int)
    d["y"] = pd.to_numeric(d["y"], errors="coerce").astype(float)
    d[prob_col] = pd.to_numeric(d[prob_col], errors="coerce").astype(float)

    # ------------------------------------------------------------------
    # Filter evaluation split first
    # ------------------------------------------------------------------
    d = d[d["split"] == evaluation_split].copy()
    if d.empty:
        raise ValueError(f"No rows found in df_agg for evaluation_split={evaluation_split!r}.")

    # ------------------------------------------------------------------
    # Filter model(s)
    # ------------------------------------------------------------------
    available_models = sorted(d["model"].unique().tolist())

    if model_names is None:
        selected_models = available_models
    elif isinstance(model_names, str):
        selected_models = [model_names]
    else:
        selected_models = list(model_names)

    missing_models = [m for m in selected_models if m not in set(available_models)]
    if missing_models:
        raise KeyError(
            f"Model(s) not found in df_agg for split={evaluation_split!r}: {missing_models}. "
            f"Available: {available_models}"
        )

    d = d[d["model"].isin(selected_models)].copy()
    if d.empty:
        raise ValueError("No rows remain after model filtering.")

    # ------------------------------------------------------------------
    # Filter calibration
    # ------------------------------------------------------------------
    if calibration is not None:
        d = d[d["calibration"] == calibration].copy()
        if d.empty:
            raise ValueError(f"No rows found for calibration={calibration!r}.")
        calibration_value = calibration
    else:
        calibrations_present = sorted(d["calibration"].unique().tolist())
        if len(calibrations_present) != 1:
            raise ValueError(
                "Multiple calibration values remain after filtering. "
                f"Please specify `calibration`. Available: {calibrations_present}"
            )
        calibration_value = calibrations_present[0]

    # ------------------------------------------------------------------
    # Resolve display labels and ensure they are unique
    # ------------------------------------------------------------------
    model_labels = [method_alias.get(m, m) for m in selected_models]

    dupes = sorted({x for x in model_labels if model_labels.count(x) > 1})
    if dupes:
        raise ValueError(f"method_alias causes duplicate display labels: {dupes}")

    # ------------------------------------------------------------------
    # Helper to resolve threshold for one model
    # ------------------------------------------------------------------
    def _resolve_threshold(model: str) -> float:
        # Same threshold for every model
        if isinstance(threshold_value, (int, float, np.floating)):
            t = float(threshold_value)
        # Per-model threshold mapping
        elif isinstance(threshold_value, Mapping):
            t = float(threshold_value.get(model, fallback_threshold))
        # Nothing provided -> fallback
        elif threshold_value is None:
            t = float(fallback_threshold)
        else:
            raise TypeError(
                "threshold_value must be None, a float, or a mapping of {model_name: threshold}."
            )

        if not (0.0 <= t <= 1.0):
            raise ValueError(f"Threshold for model={model!r} must be in [0, 1], got {t}")
        return t

    # ------------------------------------------------------------------
    # Compute balanced accuracy per model
    # ------------------------------------------------------------------
    ba_vals: list[float] = []
    thresholds: list[float] = []
    n_vals: list[int] = []

    for model in selected_models:
        # Keep only labeled rows for the requested model and evaluation split.
        sub = d[(d["model"] == model) & d["y"].notna()].copy()

        if sub.empty:
            raise ValueError(
                f"No labeled rows for model={model!r}, split={evaluation_split!r}."
            )

        # Defensive de-duplication: df_agg should already be one row per idx,
        # but we keep the first just in case duplicates exist.
        sub = sub.drop_duplicates("idx", keep="first")

        y_true = sub["y"].to_numpy(dtype=float)
        y_score = sub[prob_col].to_numpy(dtype=float)

        uniq = set(np.unique(y_true[~np.isnan(y_true)]).tolist())
        if not uniq.issubset({0.0, 1.0}):
            raise ValueError(
                f"Non-binary labels found for model={model!r}, split={evaluation_split!r}: {sorted(uniq)}"
            )

        y_true = y_true.astype(int)
        t_star = _resolve_threshold(model)
        y_pred = (y_score >= t_star).astype(int)

        ba = float(balanced_accuracy_score(y_true, y_pred))

        ba_vals.append(ba)
        thresholds.append(t_star)
        n_vals.append(int(len(sub)))

    ba_means = np.array(ba_vals, dtype=float)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(model_labels), dtype=float)

    bars = ax.bar(
        x,
        ba_means,
        width=float(bar_width),
        color=bar_color,
    )

    if show_baseline:
        ax.axhline(
            float(baseline_value),
            linestyle=baseline_ls,
            linewidth=baseline_lw,
            color=baseline_color,
            label=f"Baseline = {baseline_value:.2f}",
        )

    ax.set_title(
        f"Balanced accuracy on {evaluation_split}",
        fontsize=font_size + 1,
        fontweight="bold",
    )
    ax.set_xlabel("Model", fontsize=font_size, fontweight="bold")
    ax.set_ylabel("Balanced accuracy", fontsize=font_size, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(
        model_labels,
        fontsize=font_size,
        fontweight="bold",
        rotation=x_tick_rotation,
    )
    ax.tick_params(axis="y", labelsize=font_size)
    for lab in ax.get_yticklabels():
        lab.set_fontweight("bold")

    if annotate:
        ann_fs = annotate_font_size if annotate_font_size is not None else max(8.0, float(font_size) - 3.0)
        for bar, val in zip(bars, ba_means):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                float(val) + float(annotate_offset),
                f"{val:.{annotate_decimals}f}",
                ha="center",
                va="bottom",
                fontsize=ann_fs,
                fontweight="bold",
            )

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        top = max(float(np.max(ba_means)), float(baseline_value) if show_baseline else 0.0)
        ax.set_ylim(0.0, min(1.10, top + 0.08))

    if show_baseline:
        ax.legend(
            loc="lower right",
            frameon=True,
            prop={"size": font_size, "weight": "bold"},
            title="",
        )

    fig.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Summary output
    # ------------------------------------------------------------------
    summary = pd.DataFrame(
        {
            "model": selected_models,
            "model_label": model_labels,
            "calibration": calibration_value,
            "evaluation_split": evaluation_split,
            "balanced_accuracy": ba_means,
            "threshold": thresholds,
            "n": n_vals,
        }
    )

    if print_threshold_summary:
        print("Per-model threshold summary:")
        for label, t in zip(model_labels, thresholds):
            print(f"  {label}: {t:.3f}")

    return summary



def plot_screening_predictions(
    df_pred: pd.DataFrame,
    *,
    models: Sequence[str],
    calibration: str = "beta",
    center_col: str = "p_mean",
    std_col: str = "p_std",

    # ------------------------------------------------------------------
    # Enrichment / stratification controls
    # ------------------------------------------------------------------
    cutoff: float | None = 0.70,
    strata: Sequence[tuple[str, float, float]] | None = None,

    reference_order_model: str | None = None,
    method_alias: Mapping[str, str] | None = None,
    model_colors: Sequence[str] = ("#4C97E8", "#EC6868", "#55A868", "#8172B2"),

    # Enrichment colors, used when exactly one model is plotted and cutoff is provided
    selected_color: str = "#4C97E8",
    below_threshold_color: str = "#EC6868",

    # Stratification colors, used when exactly one model is plotted and strata are provided
    strata_colors: Mapping[str, str] | None = None,
    default_strata_colors: Sequence[str] = (
        "#EC6868",  # low / first stratum
        "#F2C94C",  # medium / second stratum
        "#4C97E8",  # high / third stratum
        "#55A868",
        "#8172B2",
        "#CC79A7",
    ),

    ribbon_color_single_model: str = "#ADAAAA",
    ylim: tuple[float, float] = (0.0, 1.0),
    shade_alpha: float = 0.16,
    linewidth: float = 1.8,
    marker: str = "o",
    markersize: float = 3.0,
    markevery: int = 1,
    figsize: tuple[float, float] = (12, 6),
    font_size: int = 12,

    # Enrichment cutoff-line style
    cutoff_color: str = "#222222",
    cutoff_ls: str = "--",
    cutoff_lw: float = 1.5,

    # Stratification boundary-line style
    show_strata_boundaries: bool = True,
    strata_boundary_color: str = "#222222",
    strata_boundary_ls: str = "--",
    strata_boundary_lw: float = 1.2,

    positive_rule: str = "gt",
    title_prefix: str = "Ranked screening risk",
    line_zorder: int = 3,
    ribbon_zorder: int = 1,
    cutoff_zorder: int = 4,
    return_ranked: bool = True,
) -> dict[str, pd.DataFrame] | None:
    """
    Plot ranked screening probabilities using a shared patient order.

    This function supports three plotting modes:

    1. Plain ranked probability plot
       - cutoff=None
       - strata=None

    2. Enrichment mode
       - cutoff is provided
       - strata=None
       - patients are colored as selected vs below threshold in single-model mode

    3. Stratification mode
       - strata is provided
       - cutoff must be None
       - patients are assigned to named probability strata such as Low / Medium / High

    Behavior
    --------
    - Patients are ordered once using `reference_order_model`.
    - That same patient order is reused for all plotted models.
    - If multiple models are plotted, color encodes model identity.
    - If exactly one model is plotted:
        * enrichment mode colors points by selected vs below threshold
        * stratification mode colors points by stratum
    - The uncertainty ribbon uses a neutral color in single-model mode.

    Parameters
    ----------
    df_pred:
        Patient-level prediction summary dataframe. Must contain:
        ["model", "calibration", "idx", center_col, std_col].

    models:
        Model names to plot.

    calibration:
        Calibration variant to filter on.

    center_col:
        Prediction summary column used for sorting patients and plotting the
        ranked screening curve.

    std_col:
        Standard deviation column used for the shaded uncertainty ribbon.

    cutoff:
        Enrichment threshold.

        If provided and `strata=None`, the plot is in enrichment mode.

        In enrichment mode:
            - "gt": selected if center_col > cutoff
            - "ge": selected if center_col >= cutoff

    strata:
        Optional sequence of named probability intervals.

        Each stratum should be:
            (stratum_name, lower_bound, upper_bound)

        The interval is interpreted as:
            lower_bound <= center_col < upper_bound

        Example:
            [
                ("Low", 0.00, 0.30),
                ("Medium", 0.30, 0.70),
                ("High", 0.70, 1.00),
            ]

        If provided, the plot is in stratification mode and `cutoff` must be None.

    reference_order_model:
        Model used to define the shared patient order. Patients are sorted by
        descending `center_col` from this model, and that same order is reused
        for all plotted models. If None, the first model in `models` is used.

    method_alias:
        Optional display-name mapping for plot labels.

    model_colors:
        Colors used for model lines when plotting multiple models.

    selected_color:
        Line color for points above threshold when plotting a single model in
        enrichment mode.

    below_threshold_color:
        Line color for points below threshold when plotting a single model in
        enrichment mode.

    strata_colors:
        Optional mapping from stratum name to color in single-model
        stratification mode.

    default_strata_colors:
        Fallback colors used when strata_colors is not provided.

    ribbon_color_single_model:
        Neutral ribbon color used in single-model mode so the uncertainty band
        does not visually imply membership in a selected group or stratum.

    ylim:
        Y-axis range.

    shade_alpha:
        Transparency of the shaded ±1 SD ribbon.

    positive_rule:
        Rule used to define selected patients in enrichment mode:
          - "gt": selected if center_col > cutoff
          - "ge": selected if center_col >= cutoff

    title_prefix:
        Plot title.

    return_ranked:
        If True, return ranked/ordered dataframes per model.

    Returns
    -------
    dict[str, pd.DataFrame] | None
        Ordered dataframe per model if `return_ranked=True`, else None.
    """
    required_cols = {"model", "calibration", "idx", center_col, std_col}
    missing = required_cols - set(df_pred.columns)
    if missing:
        raise KeyError(f"df_pred is missing required columns: {sorted(missing)}")

    if not models:
        raise ValueError("You must provide at least one model name in `models`.")

    if positive_rule not in {"gt", "ge"}:
        raise ValueError("positive_rule must be either 'gt' or 'ge'.")

    if method_alias is None:
        method_alias = {}

    if reference_order_model is None:
        reference_order_model = str(models[0])

    # ------------------------------------------------------------------
    # Determine plotting mode
    # ------------------------------------------------------------------
    if strata is not None and cutoff is not None:
        raise ValueError(
            "Provide either cutoff or strata, not both. "
            "Use cutoff for enrichment mode and strata for stratification mode."
        )

    if strata is not None:
        plot_mode = "stratification"
    elif cutoff is not None:
        plot_mode = "enrichment"
    else:
        plot_mode = "plain"

    # ------------------------------------------------------------------
    # Validate and normalize strata, if provided
    # ------------------------------------------------------------------
    strata_list: list[tuple[str, float, float]] = []

    if strata is not None:
        if len(list(strata)) == 0:
            raise ValueError("strata must contain at least one stratum.")

        for s in list(strata):
            if len(s) != 3:
                raise ValueError(
                    "Each stratum must be a tuple: "
                    "(stratum_name, lower_bound, upper_bound)."
                )

            name, low, high = s
            low = float(low)
            high = float(high)

            if low >= high:
                raise ValueError(
                    f"Invalid stratum {name!r}: lower bound ({low}) "
                    f"must be < upper bound ({high})."
                )

            strata_list.append((str(name), low, high))

        # Prevent accidental overlap. Gaps are allowed.
        sorted_strata = sorted(strata_list, key=lambda x: (x[1], x[2]))
        for i in range(1, len(sorted_strata)):
            prev_name, prev_low, prev_high = sorted_strata[i - 1]
            curr_name, curr_low, curr_high = sorted_strata[i]

            if curr_low < prev_high:
                raise ValueError(
                    "Overlapping strata detected: "
                    f"{prev_name!r} ({prev_low}, {prev_high}) overlaps with "
                    f"{curr_name!r} ({curr_low}, {curr_high}). "
                    "Use non-overlapping intervals."
                )

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _pretty_prediction_name(col: str) -> str:
        """Convert internal column names into reader-friendly phrases."""
        mapping = {
            "p_mean": "mean predicted probability",
            "p_median": "median predicted probability",
            "p_max": "maximum predicted probability",
            "p_min": "minimum predicted probability",
            "p_std": "prediction standard deviation",
        }
        return mapping.get(col, col.replace("_", " "))

    def _assign_stratum(value: float) -> tuple[str | None, float | None, float | None]:
        """
        Assign a probability value to a named stratum.

        Strata use:
            low <= value < high

        The last stratum is treated as closed on the upper bound if high == 1.0
        so that a value of exactly 1.0 is included.
        """
        if pd.isna(value):
            return None, None, None

        for i, (name, low, high) in enumerate(strata_list):
            is_last = i == len(strata_list) - 1

            if low <= value < high:
                return name, low, high

            if is_last and value == high:
                return name, low, high

        return None, None, None

    def _get_strata_color_map() -> dict[str, str]:
        if strata_colors is not None:
            return {str(k): v for k, v in strata_colors.items()}

        return {
            name: default_strata_colors[i % len(default_strata_colors)]
            for i, (name, _, _) in enumerate(strata_list)
        }

    # ------------------------------------------------------------------
    # Build the reference ordering once
    # ------------------------------------------------------------------
    ref = df_pred.copy()
    ref = ref[
        (ref["model"].astype(str) == str(reference_order_model))
        & (ref["calibration"].astype(str) == str(calibration))
    ].copy()

    if ref.empty:
        raise ValueError(
            f"No rows found for reference_order_model='{reference_order_model}' "
            f"and calibration='{calibration}'."
        )

    ref[center_col] = pd.to_numeric(ref[center_col], errors="coerce")
    ref = ref.dropna(subset=[center_col]).copy()

    if ref.empty:
        raise ValueError(
            f"Reference model '{reference_order_model}' has no valid "
            f"'{center_col}' values."
        )

    # Shared patient order for all models.
    ref = ref.sort_values(center_col, ascending=False).reset_index(drop=True)
    reference_patient_order: list = ref["idx"].tolist()
    reference_rank_map: dict = {
        patient_idx: rank
        for rank, patient_idx in enumerate(reference_patient_order, start=1)
    }

    ranked_results: dict[str, pd.DataFrame] = {}

    fig, ax = plt.subplots(figsize=figsize)

    # Collect valid model data first so we know whether we are in single-model
    # or multi-model mode.
    plot_data: list[tuple[str, str, pd.DataFrame]] = []

    for model_name in models:
        d = df_pred.copy()
        d = d[
            (d["model"].astype(str) == str(model_name))
            & (d["calibration"].astype(str) == str(calibration))
        ].copy()

        # Skip requested models that do not exist in the data.
        if d.empty:
            continue

        d[center_col] = pd.to_numeric(d[center_col], errors="coerce")
        d[std_col] = pd.to_numeric(d[std_col], errors="coerce")
        d = d.dropna(subset=[center_col]).copy()

        if d.empty:
            continue

        # Keep only patients present in the reference ordering so all plotted
        # models share the same patient x-axis meaning.
        d = d[d["idx"].isin(reference_rank_map)].copy()
        if d.empty:
            continue

        # Apply shared patient order from the reference model.
        d["patient_rank"] = d["idx"].map(reference_rank_map)
        d = d.sort_values("patient_rank", ascending=True).reset_index(drop=True)

        if plot_mode == "enrichment":
            assert cutoff is not None
            if positive_rule == "gt":
                selected_mask = d[center_col] > cutoff
            else:
                selected_mask = d[center_col] >= cutoff

            d["selected_for_enrichment"] = selected_mask

        elif plot_mode == "stratification":
            assignments = d[center_col].apply(_assign_stratum)
            d["stratum"] = assignments.apply(lambda x: x[0])
            d["stratum_low"] = assignments.apply(lambda x: x[1])
            d["stratum_high"] = assignments.apply(lambda x: x[2])

            d["in_defined_stratum"] = d["stratum"].notna()

        ranked_results[str(model_name)] = d.copy()

        display_name = method_alias.get(str(model_name), str(model_name))
        plot_data.append((str(model_name), display_name, d))

    if not plot_data:
        raise ValueError(
            "No valid rows were found for the requested model(s) and calibration."
        )

    single_model_mode = len(plot_data) == 1
    strata_color_map = _get_strata_color_map() if plot_mode == "stratification" else {}

    # ------------------------------------------------------------------
    # Plot curves
    # ------------------------------------------------------------------
    for i, (model_name, display_name, d) in enumerate(plot_data):
        x = d["patient_rank"].to_numpy(dtype=int)
        y = d[center_col].to_numpy(dtype=float)
        s = d[std_col].fillna(0.0).to_numpy(dtype=float)

        lo = np.clip(y - s, ylim[0], ylim[1])
        hi = np.clip(y + s, ylim[0], ylim[1])

        # Use a neutral ribbon for the single-model case so the uncertainty band
        # is visually separate from threshold or stratum colors.
        ribbon_color = (
            ribbon_color_single_model
            if single_model_mode
            else model_colors[i % len(model_colors)]
        )

        ax.fill_between(
            x,
            lo,
            hi,
            color=ribbon_color,
            alpha=shade_alpha,
            zorder=ribbon_zorder,
            label="±1 SD" if (single_model_mode and i == 0) else None,
        )

        n_total = int(len(d))

        # --------------------------------------------------------------
        # Single-model enrichment mode:
        # color by selected vs below threshold
        # --------------------------------------------------------------
        if single_model_mode and plot_mode == "enrichment":
            selected_mask = d["selected_for_enrichment"].to_numpy(dtype=bool)

            x_sel = x[selected_mask]
            y_sel = y[selected_mask]

            x_not = x[~selected_mask]
            y_not = y[~selected_mask]

            if len(x_sel) > 0:
                ax.plot(
                    x_sel,
                    y_sel,
                    color=selected_color,
                    linewidth=linewidth,
                    marker=marker,
                    markersize=markersize,
                    markevery=markevery,
                    label=f"Selected for enrichment (n={len(x_sel)})",
                    zorder=line_zorder,
                )

            if len(x_not) > 0:
                ax.plot(
                    x_not,
                    y_not,
                    color=below_threshold_color,
                    linewidth=linewidth,
                    marker=marker,
                    markersize=markersize,
                    markevery=markevery,
                    label=f"Below threshold (n={len(x_not)})",
                    zorder=line_zorder,
                )

        # --------------------------------------------------------------
        # Single-model stratification mode:
        # color by stratum
        # --------------------------------------------------------------
        elif single_model_mode and plot_mode == "stratification":
            for stratum_name, low, high in strata_list:
                mask = d["stratum"].astype(str) == str(stratum_name)
                x_s = x[mask.to_numpy()]
                y_s = y[mask.to_numpy()]

                if len(x_s) == 0:
                    continue

                color = strata_color_map.get(
                    str(stratum_name),
                    default_strata_colors[0],
                )

                ax.plot(
                    x_s,
                    y_s,
                    color=color,
                    linewidth=linewidth,
                    marker=marker,
                    markersize=markersize,
                    markevery=markevery,
                    label=f"{stratum_name} ({low:.2f}–{high:.2f}, n={len(x_s)})",
                    zorder=line_zorder,
                )

            # If any patients do not fall into user-defined strata, show them
            # in gray rather than silently hiding them.
            if "in_defined_stratum" in d.columns:
                mask_unassigned = ~d["in_defined_stratum"].to_numpy(dtype=bool)
                if mask_unassigned.any():
                    ax.plot(
                        x[mask_unassigned],
                        y[mask_unassigned],
                        color="#808080",
                        linewidth=linewidth,
                        marker=marker,
                        markersize=markersize,
                        markevery=markevery,
                        label=f"Unassigned (n={int(mask_unassigned.sum())})",
                        zorder=line_zorder,
                    )

        # --------------------------------------------------------------
        # Multi-model mode:
        # color by model identity
        # --------------------------------------------------------------
        else:
            line_color = model_colors[i % len(model_colors)]

            if plot_mode == "enrichment":
                n_selected = int(d["selected_for_enrichment"].sum())
                label = f"{display_name} (selected {n_selected}/{n_total})"
            elif plot_mode == "stratification":
                label = f"{display_name}"
            else:
                label = f"{display_name}"

            ax.plot(
                x,
                y,
                color=line_color,
                linewidth=linewidth,
                marker=marker,
                markersize=markersize,
                markevery=markevery,
                label=label,
                zorder=line_zorder,
            )

    # ------------------------------------------------------------------
    # Add enrichment cutoff line or stratification boundary lines
    # ------------------------------------------------------------------
    if plot_mode == "enrichment":
        assert cutoff is not None
        ax.axhline(
            y=cutoff,
            color=cutoff_color,
            linestyle=cutoff_ls,
            linewidth=cutoff_lw,
            label=f"Cutoff = {cutoff:.2f}",
            zorder=cutoff_zorder,
        )

    elif plot_mode == "stratification" and show_strata_boundaries:
        # Draw unique internal boundaries only.
        # Example strata:
        #   Low    0.00–0.30
        #   Medium 0.30–0.70
        #   High   0.70–1.00
        # Internal boundaries are 0.30 and 0.70.
        boundaries = sorted(
            {
                high
                for _, _, high in strata_list[:-1]
            }
        )

        for b in boundaries:
            ax.axhline(
                y=float(b),
                color=strata_boundary_color,
                linestyle=strata_boundary_ls,
                linewidth=strata_boundary_lw,
                label=f"Boundary = {b:.2f}",
                zorder=cutoff_zorder,
            )

    # ------------------------------------------------------------------
    # Axis labels and title
    # ------------------------------------------------------------------
    ref_display_name = method_alias.get(
        str(reference_order_model),
        str(reference_order_model),
    )
    center_col_label = _pretty_prediction_name(center_col)

    ax.set_ylim(*ylim)

    ax.set_xlabel(
        f"Patients (ordered by descending {center_col_label} from {ref_display_name})",
        fontsize=font_size,
        fontweight="bold",
    )

    ax.set_ylabel(
        "Predicted probability",
        fontsize=font_size,
        fontweight="bold",
    )

    ax.set_title(
        title_prefix,
        fontsize=font_size + 2,
        fontweight="bold",
    )

    # Patient rank is discrete, so keep x-axis ticks as integers.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Make axis values bold as well.
    ax.tick_params(axis="both", labelsize=font_size)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    ax.legend(prop={"size": font_size, "weight": "bold"})
    fig.tight_layout()
    plt.show()

    if return_ranked:
        return ranked_results

    return None


def plot_selected_patient_screening_comparison(
    comparison_df: pd.DataFrame,
    *,
    reference_model: str = "Ensemble model",
    models: Optional[Sequence[str]] = None,

    # Columns
    patient_idx_col: str = "patient_idx",
    reference_order_col: str = "reference_order",
    reference_reason_col: str = "reference_selection_reason",
    model_col: str = "compared_model",
    model_label_col: str = "compared_model_label",
    score_col: str = "p_mean",
    uncertainty_col: Optional[str] = "p_std",
    selected_col: str = "selected_for_enrichment",

    # Cutoff / axis
    cutoff: Optional[float] = 0.70,
    ylim: tuple[float, float] = (0.0, 1.05),

    # Display labels
    method_alias: Optional[Mapping[str, str]] = None,
    title: str = "Selected patient model comparison",
    xlabel: Optional[str] = None,
    ylabel: str = "Predicted probability",

    # Plot style
    figsize: tuple[float, float] = (10, 5),
    font_size: int = 12,
    linewidth: float = 1.8,
    marker: str = "o",
    markersize: float = 4.0,
    model_colors: Sequence[str] = ("#4C97E8", "#EC6868", "#55A868", "#8172B2"),

    # Uncertainty ribbon
    show_uncertainty: bool = True,
    shade_alpha: float = 0.14,

    # Cutoff line
    cutoff_color: str = "#222222",
    cutoff_ls: str = "--",
    cutoff_lw: float = 1.5,

    # X tick labeling
    x_tick_label_mode: str = "patient_idx",
    max_xticks: int = 30,
    x_tick_rotation: int = 0,

    # Optional annotations
    annotate_points: bool = False,
    annotate_decimals: int = 2,
    annotate_font_size: Optional[int] = None,
    annotate_offset: float = 0.015,

    # Grid
    show_grid: bool = True,
    grid_color: str = "#cccccc",
    grid_linewidth: float = 1.0,
    grid_alpha: float = 1.0,
    grid_linestyle: str = "-",

    # Return
    return_plot_data: bool = True,
):
    """
    Plot model probabilities for patients selected from a reference model.

    This function consumes the long-format output from:

        build_selected_patient_model_comparison_table(..., return_format="long")

    It reuses the visual grammar of `plot_screening_predictions`:
        - patients on the x-axis
        - predicted probability on the y-axis
        - one line per model
        - optional uncertainty ribbon
        - enrichment cutoff line

    Parameters
    ----------
    comparison_df : pandas.DataFrame
        Long-format selected-patient comparison table.

        Expected structure:
            one row per reference patient x compared model

    reference_model : str, default "Ensemble model"
        Model whose selected patients define the comparison cohort.

    models : sequence of str or None, default None
        Compared models to plot.

        If None, all compared models present in `comparison_df` are plotted.

    x_tick_label_mode : {"patient_idx", "patient_idx_and_reason", "rank", "none"}, default "patient_idx"
        Controls x-axis tick labels.

        "patient_idx":
            Show actual patient identifiers.

        "patient_idx_and_reason":
            Show patient identifiers plus reference selection reason.

        "rank":
            Show rank/order labels 1..N.

        "none":
            Hide x tick labels.

        Important:
            The x-position is always ordered by `reference_order_col`, but the
            default labels are patient IDs, not artificial ranks.

    max_xticks : int, default 30
        Maximum number of x ticks to show.

        If there are more patients than max_xticks, the function samples evenly
        spaced ticks. The labels are still based on `x_tick_label_mode`.

    Returns
    -------
    outputs : dict or None
        If return_plot_data=True:
            {
                "fig": fig,
                "ax": ax,
                "plot_data": plot_data,
                "patient_order_df": patient_order_df,
            }
    """

    if not isinstance(comparison_df, pd.DataFrame):
        raise TypeError("comparison_df must be a pandas DataFrame.")

    if x_tick_label_mode not in {
        "patient_idx",
        "patient_idx_and_reason",
        "rank",
        "none",
    }:
        raise ValueError(
            "x_tick_label_mode must be one of "
            "{'patient_idx', 'patient_idx_and_reason', 'rank', 'none'}."
        )

    if method_alias is None:
        method_alias = {}

    required_cols = {
        reference_order_col,
        reference_reason_col,
        patient_idx_col,
        model_col,
        score_col,
    }

    missing = required_cols - set(comparison_df.columns)
    if missing:
        raise KeyError(
            f"comparison_df is missing required columns: {sorted(missing)}"
        )

    if uncertainty_col is not None and uncertainty_col not in comparison_df.columns:
        if show_uncertainty:
            raise KeyError(
                f"comparison_df is missing uncertainty_col={uncertainty_col!r}."
            )

    d = comparison_df.copy()

    # Filter to requested reference model if available.
    if "reference_model" in d.columns:
        d = d[d["reference_model"].astype(str) == str(reference_model)].copy()

        if d.empty:
            raise ValueError(
                f"No rows found for reference_model={reference_model!r}."
            )

    d[patient_idx_col] = pd.to_numeric(d[patient_idx_col], errors="coerce").astype(int)
    d[reference_order_col] = pd.to_numeric(
        d[reference_order_col],
        errors="coerce",
    ).astype(int)
    d[model_col] = d[model_col].astype(str)
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce").astype(float)

    if uncertainty_col is not None and uncertainty_col in d.columns:
        d[uncertainty_col] = pd.to_numeric(
            d[uncertainty_col],
            errors="coerce",
        ).astype(float)

    if models is None:
        model_list = list(dict.fromkeys(d[model_col].tolist()))
    else:
        model_list = list(models)

    if len(model_list) == 0:
        raise ValueError("models must contain at least one model.")

    available_models = sorted(d[model_col].unique().tolist())
    missing_models = [m for m in model_list if m not in available_models]

    if missing_models:
        raise KeyError(
            f"Requested model(s) not found in comparison_df: {missing_models}. "
            f"Available models: {available_models}"
        )

    d = d[d[model_col].isin(model_list)].copy()

    # Build patient order table.
    patient_order_df = (
        d[[reference_order_col, patient_idx_col, reference_reason_col]]
        .drop_duplicates(patient_idx_col, keep="first")
        .sort_values(reference_order_col, kind="mergesort")
        .reset_index(drop=True)
    )

    patient_order_df["x"] = np.arange(1, len(patient_order_df) + 1, dtype=int)

    d = d.merge(
        patient_order_df[[patient_idx_col, "x"]],
        on=patient_idx_col,
        how="left",
        validate="many_to_one",
    )

    d = d.sort_values(
        ["x", model_col],
        kind="mergesort",
    ).reset_index(drop=True)

    n_patients = int(patient_order_df[patient_idx_col].nunique())

    fig, ax = plt.subplots(figsize=figsize)

    # ------------------------------------------------------------------
    # Plot one line per compared model
    # ------------------------------------------------------------------
    for i, model_name in enumerate(model_list):
        sub = d[d[model_col] == str(model_name)].copy()
        sub = sub.sort_values("x", kind="mergesort")

        if sub.empty:
            continue

        x = sub["x"].to_numpy(dtype=float)
        y = sub[score_col].to_numpy(dtype=float)

        display_name = method_alias.get(str(model_name), None)

        if display_name is None:
            if model_label_col in sub.columns:
                display_name = str(sub[model_label_col].iloc[0])
            else:
                display_name = str(model_name)

        label = display_name
        if selected_col in sub.columns:
            selected_count = int(sub[selected_col].astype(bool).sum())
            label = f"{display_name} (selected {selected_count}/{n_patients})"

        color = model_colors[i % len(model_colors)]

        if (
            show_uncertainty
            and uncertainty_col is not None
            and uncertainty_col in sub.columns
        ):
            s = sub[uncertainty_col].fillna(0.0).to_numpy(dtype=float)
            lo = np.clip(y - s, ylim[0], ylim[1])
            hi = np.clip(y + s, ylim[0], ylim[1])

            ax.fill_between(
                x,
                lo,
                hi,
                color=color,
                alpha=shade_alpha,
                zorder=1,
            )

        ax.plot(
            x,
            y,
            color=color,
            linewidth=linewidth,
            marker=marker,
            markersize=markersize,
            label=label,
            zorder=3,
        )

        if annotate_points:
            ann_fs = annotate_font_size if annotate_font_size is not None else max(8, font_size - 3)

            for xx, yy in zip(x, y):
                ax.text(
                    xx,
                    yy + annotate_offset,
                    f"{yy:.{annotate_decimals}f}",
                    ha="center",
                    va="bottom",
                    fontsize=ann_fs,
                    fontweight="bold",
                    color=color,
                )

    # ------------------------------------------------------------------
    # Cutoff line
    # ------------------------------------------------------------------
    if cutoff is not None:
        ax.axhline(
            y=float(cutoff),
            color=cutoff_color,
            linestyle=cutoff_ls,
            linewidth=cutoff_lw,
            label=f"Cutoff = {float(cutoff):.2f}",
            zorder=4,
        )

    # ------------------------------------------------------------------
    # Axis labels
    # ------------------------------------------------------------------
    if xlabel is None:
        ref_label = method_alias.get(str(reference_model), str(reference_model))
        xlabel = f"Selected patients ordered by {ref_label}"

    ax.set_title(
        title,
        fontsize=font_size + 2,
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

    ax.set_ylim(*ylim)

    # ------------------------------------------------------------------
    # X ticks
    # ------------------------------------------------------------------
    if x_tick_label_mode == "none":
        ax.set_xticks([])

    else:
        # Select tick positions.
        if n_patients <= max_xticks:
            tick_df = patient_order_df.copy()
        else:
            tick_positions = np.linspace(
                0,
                n_patients - 1,
                num=max_xticks,
                dtype=int,
            )
            tick_positions = sorted(set(tick_positions.tolist()))
            tick_df = patient_order_df.iloc[tick_positions].copy()

        tick_positions = tick_df["x"].to_numpy(dtype=int)

        if x_tick_label_mode == "patient_idx":
            tick_labels = [
                str(pid)
                for pid in tick_df[patient_idx_col]
            ]

        elif x_tick_label_mode == "patient_idx_and_reason":
            tick_labels = [
                f"{pid}\n{str(reason).replace('_', ' ')}"
                for pid, reason in zip(
                    tick_df[patient_idx_col],
                    tick_df[reference_reason_col],
                )
            ]

        elif x_tick_label_mode == "rank":
            tick_labels = [
                str(x)
                for x in tick_df["x"]
            ]

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
        tick_labels,
        fontsize=font_size,
        fontweight="bold",
        rotation=x_tick_rotation,
        )

    ax.tick_params(axis="y", labelsize=font_size)

    for tick_label in ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    # ------------------------------------------------------------------
    # Grid and legend
    # ------------------------------------------------------------------
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
    else:
        ax.grid(False)

    ax.set_axisbelow(True)

    ax.legend(
        loc="best",
        prop={"size": font_size, 
              "weight": "bold"
              },
        title="",
    )

    fig.tight_layout()
    plt.show()

    if return_plot_data:
        return {
            "fig": fig,
            "ax": ax,
            "plot_data": d,
            "patient_order_df": patient_order_df,
        }

    return None



# ------------------------------------------------------------------
# SHAP values
# ------------------------------------------------------------------

def add_external_shap_to_results(
    all_results: Dict[str, List[Dict[str, Any]]],
    model_data_dict: Dict[str, pd.DataFrame],
    train_data: pd.DataFrame,
    *,
    y_col: Optional[str] = None,
    external_tag: str = "external",
    strict_features: bool = True,
    max_background: Optional[int] = None,
    random_state: int = 42,
    check_additivity: bool = True,
    additivity_tolerance: float = 1e-4,
    warn_on_skip: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute raw-probability SHAP values for model-specific external dataframes
    using every fold model in all_results, and store SHAP outputs back into
    each fold record.

    This function is designed for the existing nested-CV results pipeline.

    Required fold-record keys
    -------------------------
    Each fold record must contain:

        rec["final_model"]
        rec["feature_names_used"]
        rec["outer_train_idx"]

    SHAP target explained
    ---------------------
    This explains the raw model probability:

        final_model.predict_proba(X_ext)[:, 1]

    i.e., the same prediction stored by add_external_predictions_to_results as:

        rec[f"y_{external_tag}_scores"]

    It does NOT explain Platt-calibrated or beta-calibrated probabilities.

    Background/reference data
    -------------------------
    For each fold record, the SHAP background is selected from train_data using:

        rec["outer_train_idx"]

    Therefore each fold model is explained relative to the training rows used
    for that fold model.

    Parameters
    ----------
    all_results:
        Dict mapping model_name -> list of fold-record dicts.

    model_data_dict:
        Dict mapping model_name -> external dataframe for that model.
        These are the rows/patients being explained.

    train_data:
        Full training dataframe containing all possible feature columns.
        This is used as the SHAP background/reference source.

    y_col:
        Optional label column name.
        This is ignored for SHAP except when strict_features=False and fallback
        feature selection is needed.

    external_tag:
        Prefix used for written keys, e.g. "external" ->
        external_shap_values_raw_probability.

    strict_features:
        If True, error if required keys or columns are missing.
        If False, and feature_names_used is missing, use all external dataframe
        columns except y_col.

    max_background:
        Optional cap on the number of SHAP background rows.

        If None, use all available fold-specific training rows as the SHAP
        background.

        If an integer, randomly sample up to that many rows from the fold-specific
        training background.

        The selected background rows are passed to SHAP with an explicit masker so
        SHAP does not internally downsample them again.

    random_state:
        Random state used only when max_background is not None.

    check_additivity:
        If True, store a check comparing:

            base_value + sum(SHAP values)

        against:

            final_model.predict_proba(X_ext)[:, 1]

    additivity_tolerance:
        Tolerance used for the additivity check.

    warn_on_skip:
        If True, warn when models are skipped due to missing overlap.

    Returns
    -------
    Updated all_results dict.
    """
    if not isinstance(model_data_dict, dict):
        raise TypeError("model_data_dict must be a dict of {model_name: dataframe}")

    if not isinstance(train_data, pd.DataFrame):
        raise TypeError("train_data must be a pandas DataFrame")

    if max_background is not None:
        if not isinstance(max_background, int):
            raise TypeError("max_background must be an int or None.")
        if max_background <= 0:
            raise ValueError("max_background must be positive when provided.")
        
    all_result_models = set(all_results.keys())
    data_models = set(model_data_dict.keys())

    overlap_models = sorted(all_result_models & data_models)
    missing_in_data = sorted(all_result_models - data_models)
    extra_in_data = sorted(data_models - all_result_models)

    if not overlap_models:
        raise KeyError(
            "No overlapping model names between all_results and model_data_dict. "
            f"all_results models={sorted(all_result_models)}, "
            f"model_data_dict models={sorted(data_models)}"
        )

    if warn_on_skip and missing_in_data:
        warnings.warn(
            "Skipping models in all_results with no matching dataframe in model_data_dict: "
            f"{missing_in_data}"
        )

    if warn_on_skip and extra_in_data:
        warnings.warn(
            "model_data_dict contains models not present in all_results; they will be ignored: "
            f"{extra_in_data}"
        )

    rng = np.random.default_rng(random_state)

    for model_name in overlap_models:
        fold_records = all_results[model_name]
        external_df = model_data_dict[model_name]

        if not isinstance(external_df, pd.DataFrame):
            raise TypeError(f"model_data_dict[{model_name!r}] must be a pandas DataFrame")

        if y_col is not None and y_col not in external_df.columns:
            raise KeyError(
                f"model_data_dict[{model_name!r}] is missing y_col={y_col!r}"
            )

        idx_ext = external_df.index.to_numpy()

        for rec in fold_records:
            if "final_model" not in rec:
                raise KeyError(f"{model_name} record missing 'final_model'")

            selected_feature_names = rec.get("feature_names_used", None)

            if selected_feature_names is None:
                if strict_features:
                    raise KeyError(
                        f"{model_name} record missing 'feature_names_used'"
                    )
                selected_feature_names = [
                    c for c in external_df.columns
                    if c != y_col
                ]

            if "outer_train_idx" not in rec:
                if strict_features:
                    raise KeyError(
                        f"{model_name} record missing 'outer_train_idx'"
                    )
                train_idx = np.arange(len(train_data))
            else:
                train_idx = np.asarray(rec["outer_train_idx"])

            selected_feature_names = list(selected_feature_names)

            missing_external = [
                c for c in selected_feature_names
                if c not in external_df.columns
            ]
            if missing_external:
                raise KeyError(
                    f"{model_name} external dataframe missing required features: "
                    f"{missing_external}"
                )

            missing_train = [
                c for c in selected_feature_names
                if c not in train_data.columns
            ]
            if missing_train:
                raise KeyError(
                    f"train_data missing required features for {model_name}: "
                    f"{missing_train}"
                )

            X_ext_df = external_df.loc[:, selected_feature_names]

            # CV split indices are positional indices, so iloc is the safest default.
            X_bg_df = train_data.iloc[train_idx].loc[:, selected_feature_names]
            background_idx_used = train_idx.copy()

            if max_background is not None and len(X_bg_df) > max_background:
                sampled_pos = rng.choice(
                    len(X_bg_df),
                    size=max_background,
                    replace=False,
                )
                X_bg_df = X_bg_df.iloc[sampled_pos]
                background_idx_used = train_idx[sampled_pos]

            final_model = rec["final_model"]

            def predict_raw_probability(X):
                X_arr = np.asarray(X)
                return final_model.predict_proba(X_arr)[:, 1]

            # explainer = shap.Explainer(
            #     predict_raw_probability,
            #     X_bg_df,
            #     feature_names=selected_feature_names,
            # )

            # shap_exp = explainer(X_ext_df)

            masker = shap.maskers.Independent(
                X_bg_df,
                max_samples=len(X_bg_df),
            )

            explainer = shap.Explainer(
                predict_raw_probability,
                masker,
                feature_names=selected_feature_names,
            )

            shap_exp = explainer(X_ext_df)

            shap_values = np.asarray(shap_exp.values)
            base_values = np.asarray(shap_exp.base_values)

            if shap_values.ndim != 2:
                raise ValueError(
                    f"Expected SHAP values with shape (n_samples, n_features), "
                    f"got shape={shap_values.shape} for model={model_name}"
                )

            p_ext = predict_raw_probability(X_ext_df)

            rec[f"{external_tag}_shap_feature_names"] = selected_feature_names
            rec[f"{external_tag}_shap_values_raw_probability"] = shap_values
            rec[f"{external_tag}_shap_base_values_raw_probability"] = base_values
            rec[f"{external_tag}_shap_data"] = X_ext_df.to_numpy()
            rec[f"{external_tag}_shap_idx"] = idx_ext
            rec[f"{external_tag}_shap_background_idx"] = background_idx_used
            rec[f"{external_tag}_shap_background_n"] = int(len(X_bg_df))
            rec[f"{external_tag}_shap_explains"] = f"y_{external_tag}_scores"
            rec[f"{external_tag}_shap_output"] = "raw_probability"

            if check_additivity:
                shap_reconstructed = base_values + shap_values.sum(axis=1)
                additivity_error = p_ext - shap_reconstructed

                rec[f"{external_tag}_shap_reconstructed_raw_probability"] = (
                    shap_reconstructed
                )
                rec[f"{external_tag}_shap_additivity_error"] = additivity_error
                rec[f"{external_tag}_shap_additivity_max_abs_error"] = float(
                    np.max(np.abs(additivity_error))
                )
                rec[f"{external_tag}_shap_additivity_passed"] = bool(
                    np.max(np.abs(additivity_error)) <= additivity_tolerance
                )

    return all_results


def add_external_shap_summary_to_results(
    all_results,
    *,
    external_tag="external",
    summary_key="external_shap_summary",
):
    """
    Aggregate fold-level external SHAP values by model type using the mean.

    This should be run after:
        1. add_external_predictions_to_results(...)
        2. add_external_shap_to_results(...)

    It does NOT combine different model types. For example:
        - logistic_regression fold models are aggregated together
        - xgboost fold models are aggregated together

    The summary is stored back into all_results under:

        all_results[summary_key][model_name]

    Example:
        all_results["external_shap_summary"]["logistic_regression"]

    Aggregation
    -----------
    For each model type, this computes:

        mean prediction across fold models
        mean SHAP base value across fold models
        mean SHAP value for each patient-feature pair across fold models

    Because mean aggregation is linear:

        mean(prediction) ~= mean(base value) + sum(mean(SHAP values))

    Required fold-record keys
    -------------------------
    Each fold record must contain:

        rec[f"y_{external_tag}_scores"]
        rec[f"{external_tag}_shap_values_raw_probability"]
        rec[f"{external_tag}_shap_base_values_raw_probability"]
        rec[f"{external_tag}_shap_feature_names"]
        rec[f"{external_tag}_shap_data"]
        rec[f"{external_tag}_shap_idx"]
    """

    shap_values_key = f"{external_tag}_shap_values_raw_probability"
    base_values_key = f"{external_tag}_shap_base_values_raw_probability"
    feature_names_key = f"{external_tag}_shap_feature_names"
    shap_data_key = f"{external_tag}_shap_data"
    shap_idx_key = f"{external_tag}_shap_idx"
    pred_key = f"y_{external_tag}_scores"

    summary = {}

    # Do not treat summary/meta keys as model names.
    model_names = [
        model_name
        for model_name, records in all_results.items()
        if not str(model_name).startswith("_")
    ]

    for model_name in model_names:
        fold_records = all_results[model_name]

        if not isinstance(fold_records, list):
            continue

        if len(fold_records) == 0:
            continue

        required_keys = [
            pred_key,
            shap_values_key,
            base_values_key,
            feature_names_key,
            shap_data_key,
            shap_idx_key,
        ]

        for fold_idx, rec in enumerate(fold_records):
            missing_keys = [k for k in required_keys if k not in rec]
            if missing_keys:
                raise KeyError(
                    f"{model_name} fold index {fold_idx} is missing required keys: "
                    f"{missing_keys}. Run add_external_predictions_to_results and "
                    f"add_external_shap_to_results first."
                )

        reference_feature_names = list(fold_records[0][feature_names_key])
        reference_data = np.asarray(fold_records[0][shap_data_key])
        reference_idx = np.asarray(fold_records[0][shap_idx_key])

        shap_values_list = []
        base_values_list = []
        predictions_list = []

        for fold_idx, rec in enumerate(fold_records):
            feature_names = list(rec[feature_names_key])
            shap_data = np.asarray(rec[shap_data_key])
            shap_idx = np.asarray(rec[shap_idx_key])

            if feature_names != reference_feature_names:
                raise ValueError(
                    f"{model_name} fold index {fold_idx} has different feature names/order. "
                    "This simple mean-aggregation function assumes all folds within a "
                    "model type use the same feature set in the same order."
                )

            if not np.array_equal(shap_idx, reference_idx):
                raise ValueError(
                    f"{model_name} fold index {fold_idx} has different external row indices. "
                    "All folds must explain the same external patients in the same order."
                )

            if not np.allclose(shap_data, reference_data):
                raise ValueError(
                    f"{model_name} fold index {fold_idx} has different external SHAP data. "
                    "All folds must explain the same external feature values."
                )

            shap_values_list.append(np.asarray(rec[shap_values_key]))
            base_values_list.append(np.asarray(rec[base_values_key]))
            predictions_list.append(np.asarray(rec[pred_key]))

        shap_values_stack = np.stack(shap_values_list, axis=0)
        base_values_stack = np.stack(base_values_list, axis=0)
        predictions_stack = np.stack(predictions_list, axis=0)

        shap_values_mean = shap_values_stack.mean(axis=0)
        base_values_mean = base_values_stack.mean(axis=0)
        predictions_mean = predictions_stack.mean(axis=0)

        reconstructed_mean = base_values_mean + shap_values_mean.sum(axis=1)
        additivity_error_mean = predictions_mean - reconstructed_mean

        summary[model_name] = {
            "model_name": model_name,
            "aggregation": "mean",
            "n_models": int(len(fold_records)),
            "external_tag": external_tag,
            "feature_names": reference_feature_names,
            "data": reference_data,
            "idx": reference_idx,
            "predictions_mean": predictions_mean,
            "base_values_mean": base_values_mean,
            "shap_values_mean": shap_values_mean,
            "reconstructed_mean": reconstructed_mean,
            "additivity_error_mean": additivity_error_mean,
            "additivity_max_abs_error_mean": float(
                np.max(np.abs(additivity_error_mean))
            ),
            "explains": f"mean_{pred_key}",
            "output": "raw_probability",
        }

    all_results[summary_key] = summary

    return all_results


def plot_patient_waterfall(
    all_results,
    model_name,
    patient_idx,
    *,
    level="mean",
    fold_idx=None,
    external_tag="external",
    summary_key="external_shap_summary",
    max_display=10,
    show_feature_table=True,
    verbose=False,
):
    """
    Plot a SHAP waterfall plot for one external patient.

    Parameters
    ----------
    all_results:
        Nested results dictionary.

    model_name:
        Model type, e.g. "logistic_regression" or "xgboost".

    patient_idx:
        Row position of the external patient to plot.

    level:
        Which explanation level to plot.

        "mean":
            Plot the mean-aggregated SHAP explanation across fold models for
            this model type. This uses:

                all_results[summary_key][model_name]

        "fold":
            Plot one specific fold model explanation. This uses:

                all_results[model_name][fold_idx]

    fold_idx:
        Required when level="fold". Ignored when level="mean".

    external_tag:
        Prefix used in stored external SHAP/prediction keys.

    summary_key:
        Top-level key containing mean SHAP summaries.

    max_display:
        Passed to shap.plots.waterfall.

    show_feature_table:
        If True, display a dataframe with feature values and SHAP values.

    verbose:
        If False, print only a short one-line summary.
        If True, print detailed reconstruction/additivity information.

    Returns
    -------
    shap.Explanation for the selected patient.
    """

    if level not in {"mean", "fold"}:
        raise ValueError("level must be either 'mean' or 'fold'")

    if level == "mean":
        if summary_key not in all_results:
            raise KeyError(
                f"Missing {summary_key!r}. Run add_external_shap_summary_to_results first."
            )

        if model_name not in all_results[summary_key]:
            raise KeyError(
                f"Missing model_name={model_name!r} in all_results[{summary_key!r}]."
            )

        summary = all_results[summary_key][model_name]

        shap_values_i = summary["shap_values_mean"][patient_idx]
        base_value_i = summary["base_values_mean"][patient_idx]
        data_i = summary["data"][patient_idx]
        feature_names = summary["feature_names"]

        prediction_i = summary["predictions_mean"][patient_idx]
        reconstructed_i = base_value_i + np.sum(shap_values_i)
        reconstruction_error_i = prediction_i - reconstructed_i

        value_label = "mean_shap_value"
        abs_value_label = "abs_mean_shap_value"

        if verbose:
            print("Model:", model_name)
            print("Level: mean aggregation across fold models")
            print("Patient index:", patient_idx)
            print("Aggregation:", summary["aggregation"])
            print("Number of fold models:", summary["n_models"])
            print("Mean predicted raw probability:", prediction_i)
            print("Mean base value:", base_value_i)
            print("Mean base + mean SHAP sum:", reconstructed_i)
            print("Reconstruction error:", reconstruction_error_i)
            print(
                "Model-level max abs additivity error:",
                summary["additivity_max_abs_error_mean"],
            )
        else:
            print(
                f"Model: {model_name} | "
                f"Number of fold models: {summary["n_models"]} | "
                f"Patient: {patient_idx} | "
                f"Mean raw probability: {prediction_i:.3f}"
            )


    else:
        if fold_idx is None:
            raise ValueError("fold_idx is required when level='fold'")

        rec = all_results[model_name][fold_idx]

        shap_values_key = f"{external_tag}_shap_values_raw_probability"
        base_values_key = f"{external_tag}_shap_base_values_raw_probability"
        shap_data_key = f"{external_tag}_shap_data"
        feature_names_key = f"{external_tag}_shap_feature_names"
        pred_key = f"y_{external_tag}_scores"

        required_keys = [
            shap_values_key,
            base_values_key,
            shap_data_key,
            feature_names_key,
            pred_key,
        ]

        missing_keys = [k for k in required_keys if k not in rec]
        if missing_keys:
            raise KeyError(
                "This fold record is missing required SHAP/prediction keys: "
                f"{missing_keys}. Run add_external_predictions_to_results and "
                "add_external_shap_to_results first."
            )

        shap_values_i = rec[shap_values_key][patient_idx]
        base_value_i = rec[base_values_key][patient_idx]
        data_i = rec[shap_data_key][patient_idx]
        feature_names = rec[feature_names_key]

        prediction_i = rec[pred_key][patient_idx]
        reconstructed_i = base_value_i + np.sum(shap_values_i)
        reconstruction_error_i = prediction_i - reconstructed_i

        value_label = "shap_value"
        abs_value_label = "abs_shap_value"

        if verbose:
            print("Model:", model_name)
            print("Level: single fold model")
            print("Fold index:", fold_idx)
            print("Patient index:", patient_idx)
            print("Predicted raw probability:", prediction_i)
            print("Base value:", base_value_i)
            print("Base + SHAP sum:", reconstructed_i)
            print("Reconstruction error:", reconstruction_error_i)

            if f"{external_tag}_shap_background_n" in rec:
                print("SHAP background n:", rec[f"{external_tag}_shap_background_n"])

            if f"{external_tag}_shap_additivity_passed" in rec:
                print(
                    "Fold-level additivity passed:",
                    rec[f"{external_tag}_shap_additivity_passed"],
                )
        else:
            print(
                f"Model: {model_name} | "
                f"Fold: {fold_idx} | "
                f"Patient: {patient_idx} | "
                f"Raw probability: {prediction_i:.3f}"
            )

    patient_exp = shap.Explanation(
        values=shap_values_i,
        base_values=base_value_i,
        data=data_i,
        feature_names=feature_names,
    )

    if show_feature_table:
        patient_shap_df = (
            pd.DataFrame({
                "feature": feature_names,
                "value": data_i,
                value_label: shap_values_i,
                abs_value_label: np.abs(shap_values_i),
            })
            .sort_values(abs_value_label, ascending=False)
            .reset_index(drop=True)
        )

        display(patient_shap_df)

    shap.plots.waterfall(patient_exp, max_display=max_display)

    return patient_exp



def _format_waterfall_value(value, fmt="%0.03f"):
    """
    Lightweight formatter used by plot_shap_style_waterfall.
    """
    if isinstance(value, str):
        return value
    return fmt % value



def plot_shap_style_waterfall(
    all_results,
    patient_idx,
    *,
    model_name=None,
    model_alias=None,
    level="mean",
    fold_idx=None,
    external_tag="external",
    summary_key="external_shap_summary",
    patient_idx_is_id=True,
    max_display=10,
    show=True,

    # colors
    positive_color="#ff0051",
    negative_color="#008bfb",
    vlines_color="#999999",
    tick_labels_color="#999999",
    text_color="white",

    # SHAP vertical connector/reference lines
    vlines_linewidth=0.5,

    # grid: applies to BOTH x and y gridlines
    show_grid=True,
    grid_color="#cccccc",
    grid_linewidth=0.5,
    grid_alpha=1.0,
    grid_linestyle="--",

    # sizing
    plot_width=8,
    row_height=0.5,
    extra_height=1.5,
    bar_width=0.8,
    head_length=0.08,

    # fonts
    tick_fontsize=13,
    bar_text_fontsize=12,
    top_label_fontsize=12,
    xlabel_fontsize=12,
    title_fontsize=14,

    # labels
    show_xlabel=True,
    xlabel="Model output",
    show_title=True,
    title=None,
    title_suffix: Optional[str] = "SHAP contributions to predicted probability",

    # formatting
    feature_value_format="%0.02f",
    shap_value_format="%+0.02f",
    probability_format="%0.02f",
):
    """
    SHAP-style waterfall plot for one or more external patients.

    By default, this plots all model types present in all_results.

    Parameters
    ----------
    all_results:
        Nested results dictionary.

    patient_idx : int or sequence of int
        Patient identifier(s) to plot.

        By default, `patient_idx_is_id=True`, meaning these values are treated
        as external patient IDs and are matched against:
            - all_results[summary_key][model_name]["idx"] when level="mean"
            - all_results[model_name][fold_idx][f"{external_tag}_shap_idx"]
              when level="fold"

        Example:
            patient_idx=137

            patient_idx=[66, 0, 39, 137]

            patient_idx=selected_patients["Ensemble model"]["patient_idx"].tolist()

    patient_idx_is_id : bool, default True
        If True, `patient_idx` is treated as an external patient ID and mapped
        to the correct SHAP row position using the stored external index array.

        If False, `patient_idx` is treated as the row position in the SHAP arrays.
        This matches the older behavior.

    model_name:
        If None, plot all model types in all_results.
        If a string, plot only that model type.
        If a list/tuple, plot those model types.

    model_alias:
        Optional dictionary mapping model keys to display names.

        Example:
            {
                "logistic_regression": "Logistic regression",
                "xgboost": "XGBoost",
            }

    level:
        "mean":
            Plot mean-aggregated SHAP explanation across fold models.
            Uses all_results[summary_key][model_name].

        "fold":
            Plot one specific fold model explanation.
            Uses all_results[model_name][fold_idx].

    fold_idx:
        Required when level="fold".

    external_tag:
        Prefix used for fold-level external SHAP/prediction keys.

    summary_key:
        Top-level key containing mean SHAP summaries.

    max_display:
        Maximum number of features to show.

    show:
        Whether to call plt.show() after each plot.

    Returns
    -------
    plot_outputs:
        If a single patient is requested:
            Dictionary keyed by model_name.

            {
                model_name: {
                    "fig": fig,
                    "ax": ax,
                    "plot_df": plot_df,
                    "patient_idx": patient_id_or_position,
                    "patient_row_position": row_position,
                }
            }

        If multiple patients are requested:
            Nested dictionary keyed by patient_idx, then model_name.

            {
                patient_idx: {
                    model_name: {
                        "fig": fig,
                        "ax": ax,
                        "plot_df": plot_df,
                        "patient_idx": patient_id_or_position,
                        "patient_row_position": row_position,
                    }
                }
            }
    """
    if level not in {"mean", "fold"}:
        raise ValueError("level must be either 'mean' or 'fold'")

    if level == "fold" and fold_idx is None:
        raise ValueError("fold_idx is required when level='fold'")

    if model_alias is None:
        model_alias = {}

    # ------------------------------------------------------------
    # Normalize patient_idx to a list
    # ------------------------------------------------------------
    if isinstance(patient_idx, (list, tuple, set, np.ndarray, pd.Series)):
        patient_indices = list(patient_idx)
        single_patient_input = False
    else:
        patient_indices = [patient_idx]
        single_patient_input = True

    if len(patient_indices) == 0:
        raise ValueError("patient_idx must contain at least one patient.")

    # ------------------------------------------------------------
    # Decide which model(s) to plot
    # ------------------------------------------------------------
    if model_name is None:
        model_names = [
            k for k, v in all_results.items()
            if isinstance(v, list)
        ]
    elif isinstance(model_name, str):
        model_names = [model_name]
    else:
        model_names = list(model_name)

    if len(model_names) == 0:
        raise ValueError("No model names found to plot.")

    # ------------------------------------------------------------
    # Helper: map external patient ID to SHAP row position
    # ------------------------------------------------------------
    def _resolve_patient_row_position(index_array, requested_patient_idx):
        if not patient_idx_is_id:
            row_position = int(requested_patient_idx)

            if row_position < 0 or row_position >= len(index_array):
                raise IndexError(
                    f"patient_idx={requested_patient_idx} was interpreted as a row "
                    f"position, but valid row positions are 0 to {len(index_array) - 1}."
                )

            return row_position

        index_array = np.asarray(index_array)

        matches = np.where(index_array == requested_patient_idx)[0]

        if len(matches) == 0:
            raise KeyError(
                f"patient_idx={requested_patient_idx!r} was not found in the "
                "stored external SHAP index array. If you intended patient_idx "
                "to be a row position, call with patient_idx_is_id=False."
            )

        if len(matches) > 1:
            raise ValueError(
                f"patient_idx={requested_patient_idx!r} appears multiple times "
                "in the stored external SHAP index array. Patient IDs must be unique."
            )

        return int(matches[0])

    # ------------------------------------------------------------
    # Helper for one model and one patient
    # ------------------------------------------------------------
    def _plot_one_model(current_model_name, current_patient_idx):
        display_model_name = model_alias.get(current_model_name, current_model_name)

        # --------------------------------------------------------
        # Extract values from all_results
        # --------------------------------------------------------
        if level == "mean":
            if summary_key not in all_results:
                raise KeyError(
                    f"Missing {summary_key!r}. "
                    "Run add_external_shap_summary_to_results first."
                )

            if current_model_name not in all_results[summary_key]:
                raise KeyError(
                    f"Missing model_name={current_model_name!r} "
                    f"in all_results[{summary_key!r}]."
                )

            summary = all_results[summary_key][current_model_name]

            if "idx" not in summary:
                raise KeyError(
                    f"all_results[{summary_key!r}][{current_model_name!r}] is "
                    "missing key 'idx'. Cannot map patient_idx to SHAP row position."
                )

            row_position = _resolve_patient_row_position(
                summary["idx"],
                current_patient_idx,
            )

            values = np.asarray(summary["shap_values_mean"][row_position], dtype=float)
            base_values = float(summary["base_values_mean"][row_position])
            features = np.asarray(summary["data"][row_position])
            feature_names = list(summary["feature_names"])

            default_title = f"{display_model_name}: Patient {current_patient_idx}"

            if title_suffix is not None:
                default_title = f"{default_title}\n{title_suffix}"

        else:
            if current_model_name not in all_results:
                raise KeyError(
                    f"Missing model_name={current_model_name!r} in all_results."
                )

            rec = all_results[current_model_name][fold_idx]

            shap_values_key = f"{external_tag}_shap_values_raw_probability"
            base_values_key = f"{external_tag}_shap_base_values_raw_probability"
            shap_data_key = f"{external_tag}_shap_data"
            feature_names_key = f"{external_tag}_shap_feature_names"
            shap_idx_key = f"{external_tag}_shap_idx"
            pred_key = f"y_{external_tag}_scores"

            required_keys = [
                shap_values_key,
                base_values_key,
                shap_data_key,
                feature_names_key,
                pred_key,
            ]

            if patient_idx_is_id:
                required_keys.append(shap_idx_key)

            missing_keys = [k for k in required_keys if k not in rec]
            if missing_keys:
                raise KeyError(
                    "This fold record is missing required SHAP/prediction keys: "
                    f"{missing_keys}. Run add_external_predictions_to_results and "
                    "add_external_shap_to_results first."
                )

            if patient_idx_is_id:
                row_position = _resolve_patient_row_position(
                    rec[shap_idx_key],
                    current_patient_idx,
                )
            else:
                row_position = _resolve_patient_row_position(
                    np.arange(len(rec[shap_values_key])),
                    current_patient_idx,
                )

            values = np.asarray(rec[shap_values_key][row_position], dtype=float)
            base_values = float(rec[base_values_key][row_position])
            features = np.asarray(rec[shap_data_key][row_position])
            feature_names = list(rec[feature_names_key])

            default_title = (
                f"{display_model_name}: Fold {fold_idx}, "
                f"Patient {current_patient_idx}"
            )

            if title_suffix is not None:
                default_title = f"{default_title}\n{title_suffix}"


        if values.ndim != 1:
            raise ValueError("This function expects one patient's 1D SHAP values.")

        if len(values) != len(feature_names):
            raise ValueError("Length mismatch between SHAP values and feature names.")

        if features is not None and len(features) != len(feature_names):
            raise ValueError("Length mismatch between feature values and feature names.")

        # --------------------------------------------------------
        # Setup
        # --------------------------------------------------------
        num_features = min(max_display, len(values))
        rng = range(num_features - 1, -1, -1)
        order = np.argsort(-np.abs(values))

        pos_lefts = []
        pos_inds = []
        pos_widths = []

        neg_lefts = []
        neg_inds = []
        neg_widths = []

        loc = base_values + values.sum()
        fx = loc

        yticklabels = ["" for _ in range(num_features + 1)]

        fig = plt.figure(figsize=(plot_width, num_features * row_height + extra_height))

        if num_features == len(values):
            num_individual = num_features
        else:
            num_individual = num_features - 1

        plot_rows = []

        # --------------------------------------------------------
        # Compute arrow locations from f(x) back toward E[f(X)]
        # --------------------------------------------------------
        for i in range(num_individual):
            feature_idx = order[i]
            sval = values[feature_idx]
            loc -= sval

            y_position = rng[i]

            if sval >= 0:
                pos_inds.append(y_position)
                pos_widths.append(sval)
                pos_lefts.append(loc)
            else:
                neg_inds.append(y_position)
                neg_widths.append(sval)
                neg_lefts.append(loc)

            # SHAP-style vertical connector line
            if num_individual != num_features or i + 4 < num_individual:
                plt.plot(
                    [loc, loc],
                    [y_position - 1 - 0.4, y_position + 0.4],
                    color=vlines_color,
                    linestyle="--",
                    linewidth=vlines_linewidth,
                    zorder=-1,
                )

            if features is None:
                label = str(feature_names[feature_idx])
                feature_value = np.nan
            else:
                feature_value = features[feature_idx]

                if np.issubdtype(type(feature_value), np.number):
                    label = (
                        _format_waterfall_value(float(feature_value), feature_value_format)
                        + " = "
                        + str(feature_names[feature_idx])
                    )
                else:
                    label = str(feature_value) + " = " + str(feature_names[feature_idx])

            yticklabels[y_position] = label

            plot_rows.append({
                "model_name": current_model_name,
                "patient_idx": current_patient_idx,
                "patient_row_position": row_position,
                "feature": feature_names[feature_idx],
                "value": feature_value,
                "shap_value": sval,
                "abs_shap_value": abs(sval),
                "y_position": y_position,
                "left": loc,
                "right": loc + sval,
            })

        # --------------------------------------------------------
        # Group remaining features if needed
        # --------------------------------------------------------
        if num_features < len(values):
            yticklabels[0] = f"{len(values) - num_features + 1} other features"

            remaining_impact = base_values - loc

            if remaining_impact < 0:
                pos_inds.append(0)
                pos_widths.append(-remaining_impact)
                pos_lefts.append(loc + remaining_impact)
                grouped_left = loc + remaining_impact
                grouped_right = loc
                grouped_value = -remaining_impact
            else:
                neg_inds.append(0)
                neg_widths.append(-remaining_impact)
                neg_lefts.append(loc + remaining_impact)
                grouped_left = loc + remaining_impact
                grouped_right = loc
                grouped_value = -remaining_impact

            plot_rows.append({
                "model_name": current_model_name,
                "patient_idx": current_patient_idx,
                "patient_row_position": row_position,
                "feature": f"{len(values) - num_features + 1} other features",
                "value": np.nan,
                "shap_value": grouped_value,
                "abs_shap_value": abs(grouped_value),
                "y_position": 0,
                "left": grouped_left,
                "right": grouped_right,
            })

        # --------------------------------------------------------
        # Invisible bars to size axes like SHAP
        # --------------------------------------------------------
        points = (
            pos_lefts
            + list(np.array(pos_lefts) + np.array(pos_widths))
            + neg_lefts
            + list(np.array(neg_lefts) + np.array(neg_widths))
        )

        if len(points) == 0:
            points = [base_values, fx]

        dataw = np.max(points) - np.min(points)
        if dataw == 0:
            dataw = 1e-6

        label_padding = np.array([0.1 * dataw if w < 1 else 0 for w in pos_widths])
        plt.barh(
            pos_inds,
            np.array(pos_widths) + label_padding + 0.02 * dataw,
            left=np.array(pos_lefts) - 0.01 * dataw,
            color=positive_color,
            alpha=0,
        )

        label_padding = np.array([-0.1 * dataw if -w < 1 else 0 for w in neg_widths])
        plt.barh(
            neg_inds,
            np.array(neg_widths) + label_padding - 0.02 * dataw,
            left=np.array(neg_lefts) + 0.01 * dataw,
            color=negative_color,
            alpha=0,
        )

        # --------------------------------------------------------
        # Arrow geometry
        # --------------------------------------------------------
        xlen = plt.xlim()[1] - plt.xlim()[0]
        ax = plt.gca()

        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width = bbox.width
        bbox_to_xscale = xlen / width
        hl_scaled = bbox_to_xscale * head_length

        renderer = fig.canvas.get_renderer()

        # --------------------------------------------------------
        # Draw positive arrows
        # --------------------------------------------------------
        for i in range(len(pos_inds)):
            dist = pos_widths[i]

            arrow_obj = plt.arrow(
                pos_lefts[i],
                pos_inds[i],
                dist - hl_scaled,
                0,
                head_length=min(dist, hl_scaled),
                color=positive_color,
                width=bar_width,
                head_width=bar_width,
            )

            txt_obj = plt.text(
                pos_lefts[i] + 0.5 * dist,
                pos_inds[i],
                _format_waterfall_value(pos_widths[i], shap_value_format),
                horizontalalignment="center",
                verticalalignment="center",
                color=text_color,
                fontsize=bar_text_fontsize,
                fontweight="bold",
            )

            text_bbox = txt_obj.get_window_extent(renderer=renderer)
            arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

            if text_bbox.width > arrow_bbox.width:
                txt_obj.remove()
                plt.text(
                    pos_lefts[i] + (5 / 72) * bbox_to_xscale + dist,
                    pos_inds[i],
                    _format_waterfall_value(pos_widths[i], shap_value_format),
                    horizontalalignment="left",
                    verticalalignment="center",
                    color=positive_color,
                    fontsize=bar_text_fontsize,
                    fontweight="bold",
                )

        # --------------------------------------------------------
        # Draw negative arrows
        # --------------------------------------------------------
        for i in range(len(neg_inds)):
            dist = neg_widths[i]

            arrow_obj = plt.arrow(
                neg_lefts[i],
                neg_inds[i],
                -(-dist - hl_scaled),
                0,
                head_length=min(-dist, hl_scaled),
                color=negative_color,
                width=bar_width,
                head_width=bar_width,
            )

            txt_obj = plt.text(
                neg_lefts[i] + 0.5 * dist,
                neg_inds[i],
                _format_waterfall_value(neg_widths[i], shap_value_format),
                horizontalalignment="center",
                verticalalignment="center",
                color=text_color,
                fontsize=bar_text_fontsize,
                fontweight="bold",
            )

            text_bbox = txt_obj.get_window_extent(renderer=renderer)
            arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

            if text_bbox.width > arrow_bbox.width:
                txt_obj.remove()
                plt.text(
                    neg_lefts[i] - (5 / 72) * bbox_to_xscale + dist,
                    neg_inds[i],
                    _format_waterfall_value(neg_widths[i], shap_value_format),
                    horizontalalignment="right",
                    verticalalignment="center",
                    color=negative_color,
                    fontsize=bar_text_fontsize,
                    fontweight="bold",
                )

        # --------------------------------------------------------
        # Y ticks twice: gray full label, black feature name
        # --------------------------------------------------------
        ytick_pos = list(range(num_features)) + list(np.arange(num_features) + 1e-8)

        black_labels = [
            label.split("=")[-1] if "=" in label else label
            for label in yticklabels[:-1]
        ]

        plt.yticks(
            ytick_pos,
            yticklabels[:-1] + black_labels,
            fontsize=tick_fontsize,
        )

        for tick_label in plt.gca().get_yticklabels():
            tick_label.set_fontweight("bold")

        # --------------------------------------------------------
        # Grid lines: same controls for BOTH x and y
        # --------------------------------------------------------
        if show_grid:
            ax.grid(
                True,
                axis="both",
                color=grid_color,
                linewidth=grid_linewidth,
                alpha=grid_alpha,
                linestyle=grid_linestyle,
                zorder=-2,
            )
        else:
            ax.grid(False)

        ax.set_axisbelow(True)

        # --------------------------------------------------------
        # SHAP-style prior expected value and prediction vertical lines
        # --------------------------------------------------------
        plt.axvline(
            base_values,
            0,
            1 / num_features,
            color=vlines_color,
            linestyle="--",
            linewidth=vlines_linewidth,
            zorder=-1,
        )

        plt.axvline(
            fx,
            0,
            1,
            color=vlines_color,
            linestyle="--",
            linewidth=vlines_linewidth,
            zorder=-1,
        )

        # --------------------------------------------------------
        # Clean main axis
        # --------------------------------------------------------
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("none")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(labelsize=tick_fontsize)

        for tick_label in ax.get_xticklabels():
            tick_label.set_fontweight("bold")

        if show_xlabel:
            ax.set_xlabel(
                xlabel,
                fontsize=xlabel_fontsize,
                fontweight="bold",
            )

        if show_title:
            final_title = title if title is not None else default_title
            ax.set_title(
                final_title,
                fontsize=title_fontsize,
                fontweight="bold",
                pad=12,
            )

        # --------------------------------------------------------
        # Top axis: E[f(X)]
        # --------------------------------------------------------
        xmin, xmax = ax.get_xlim()

        ax2 = ax.twiny()
        ax2.set_xlim(xmin, xmax)

        tiny_offset = min(1e-8, xmax * 1e-10)

        ax2.set_xticks([base_values, base_values + tiny_offset])
        ax2.set_xticklabels(
            [
                "\n$E[f(X)]$",
                "\n$ = " + _format_waterfall_value(base_values, probability_format) + "$",
            ],
            fontsize=top_label_fontsize,
            ha="left",
        )

        for tick_label in ax2.xaxis.get_majorticklabels():
            tick_label.set_fontweight("bold")

        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)

        # --------------------------------------------------------
        # Top axis: f(x)
        # --------------------------------------------------------
        ax3 = ax2.twiny()
        ax3.set_xlim(xmin, xmax)
        ax3.set_xticks([fx, fx + tiny_offset])
        ax3.set_xticklabels(
            [
                "$f(x)$",
                "$ = " + _format_waterfall_value(fx, probability_format) + "$",
            ],
            fontsize=top_label_fontsize,
            ha="left",
        )

        for tick_label in ax3.xaxis.get_majorticklabels():
            tick_label.set_fontweight("bold")

        tick_labels = ax3.xaxis.get_majorticklabels()
        tick_labels[0].set_transform(
            tick_labels[0].get_transform()
            + matplotlib.transforms.ScaledTranslation(
                -10 / 72.0,
                0,
                fig.dpi_scale_trans,
            )
        )
        tick_labels[1].set_transform(
            tick_labels[1].get_transform()
            + matplotlib.transforms.ScaledTranslation(
                12 / 72.0,
                0,
                fig.dpi_scale_trans,
            )
        )
        tick_labels[1].set_color(tick_labels_color)

        ax3.spines["right"].set_visible(False)
        ax3.spines["top"].set_visible(False)
        ax3.spines["left"].set_visible(False)

        # Adjust E[f(X)] labels
        tick_labels = ax2.xaxis.get_majorticklabels()
        tick_labels[0].set_transform(
            tick_labels[0].get_transform()
            + matplotlib.transforms.ScaledTranslation(
                -20 / 72.0,
                0,
                fig.dpi_scale_trans,
            )
        )
        tick_labels[1].set_transform(
            tick_labels[1].get_transform()
            + matplotlib.transforms.ScaledTranslation(
                22 / 72.0,
                -1 / 72.0,
                fig.dpi_scale_trans,
            )
        )
        tick_labels[1].set_color(tick_labels_color)

        # Gray feature-value labels, but keep bold
        tick_labels = ax.yaxis.get_majorticklabels()
        for i in range(num_features):
            tick_labels[i].set_color(tick_labels_color)
            tick_labels[i].set_fontweight("bold")

        plot_df = (
            pd.DataFrame(plot_rows)
            .sort_values("abs_shap_value", ascending=False)
            .reset_index(drop=True)
        )

        if show:
            plt.show()

        return {
            "fig": fig,
            "ax": ax,
            "plot_df": plot_df,
            "patient_idx": current_patient_idx,
            "patient_row_position": row_position,
            "model_name": current_model_name,
        }

    # ------------------------------------------------------------
    # Plot requested patient(s) and model(s)
    # ------------------------------------------------------------
    if single_patient_input:
        plot_outputs = {}

        for current_model_name in model_names:
            plot_outputs[current_model_name] = _plot_one_model(
                current_model_name,
                patient_indices[0],
            )

        return plot_outputs

    plot_outputs = {}

    for current_patient_idx in patient_indices:
        plot_outputs[current_patient_idx] = {}

        for current_model_name in model_names:
            plot_outputs[current_patient_idx][current_model_name] = _plot_one_model(
                current_model_name,
                current_patient_idx,
            )

    return plot_outputs


def plot_cohort_shap_contributions(
    cohort_shap_summary: pd.DataFrame,
    *,
    models: Union[str, Sequence[str]],
    feature_col: str = "feature",
    model_col: str = "model",

    # X-axis mode
    x_axis_mode: str = "contribution",  # "contribution" or "probability"

    # Mean SHAP columns
    selected_mean_col: str = "mean_shap_selected",
    not_selected_mean_col: str = "mean_shap_not_selected",

    # Error bar columns for SHAP contribution mode
    selected_error_col: Optional[str] = "sem_shap_selected",
    not_selected_error_col: Optional[str] = "sem_shap_not_selected",

    # Endpoint columns used when x_axis_mode="probability"
    selected_endpoint_col: str = "mean_endpoint_selected",
    not_selected_endpoint_col: str = "mean_endpoint_not_selected",
    selected_endpoint_error_col: Optional[str] = "sem_endpoint_selected",
    not_selected_endpoint_error_col: Optional[str] = "sem_endpoint_not_selected",

    # Baseline / prediction columns used when x_axis_mode="probability"
    selected_base_col: str = "mean_base_value_selected",
    not_selected_base_col: str = "mean_base_value_not_selected",
    selected_prediction_col: str = "mean_prediction_selected",
    not_selected_prediction_col: str = "mean_prediction_not_selected",

    # Sorting / feature selection
    sort_metric_col: str = "balanced_delta_mean_shap",
    top_n: Optional[int] = 10,
    sort_by: str = "abs",

    # Labels
    model_alias: Optional[Mapping[str, str]] = None,
    title_prefix: str = "Cohort SHAP contributions to predicted probability",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Feature",
    selected_label: str = "Selected",
    not_selected_label: str = "Not selected",

    # Probability-mode reference lines
    show_baseline_line: bool = True,
    show_baseline_value_label: bool = True,
    baseline_label: str = r"Baseline probability ($E[f(X)]$)",
    baseline_value_format: str = "{:.2f}",
    baseline_value_label_y: float = -0.01,

    show_prediction_lines: bool = False,
    selected_prediction_label: Optional[str] = None,
    not_selected_prediction_label: Optional[str] = None,

    # Figure styling
    figsize: tuple[float, float] = (9, 4.8),
    selected_color: str = "#ff0051",
    not_selected_color: str = "#008bfb",
    error_color: str = "#222222",
    baseline_color: str = "#777777",
    zero_line_color: str = "#222222",
    selected_prediction_color: Optional[str] = None,
    not_selected_prediction_color: Optional[str] = None,
    bar_height: float = 0.35,
    error_linewidth: float = 1.2,
    capsize: float = 4,
    reference_linewidth: float = 1.2,
    reference_linestyle: str = "--",

    # Grid
    show_grid: bool = True,
    grid_color: str = "#cccccc",
    grid_linewidth: float = 1.0,
    grid_alpha: float = 1.0,
    grid_linestyle: str = "-",

    # Fonts
    title_fontsize: int = 14,
    label_fontsize: int = 13,
    tick_fontsize: int = 12,
    legend_fontsize: int = 12,
    value_fontsize: int = 11,
    reference_label_fontsize: int = 11,
    fontweight: str = "bold",
    legend_fontweight: str = "bold",

    # Value labels
    show_group_values: bool = True,
    show_shap_in_probability_label: bool = False,
    value_format: str = "{:.2f}",
    contribution_format: str = "{:+.2f}",
    error_format: str = "{:.2f}",
    value_offset_fraction: float = 0.025,

    # Axis limits
    xlim: Optional[tuple[float, float]] = None,

    # Return
    return_plot_data: bool = True,
):

    """
    Plot cohort-level SHAP contributions for one or more models.

    This function compares average feature contributions across two cohort
    groups, usually selected vs below-threshold patients. It is designed to
    visualize the output of `build_cohort_shap_summary_table(...)`.

    The same function supports two complementary views:

    1. Contribution mode
       Shows mean SHAP contributions centered around 0.

       Interpretation:
           positive values push predicted probability upward;
           negative values push predicted probability downward.

       Label format:
           SHAP: +0.31 ± 0.02

       Error bars use:
           selected_error_col / not_selected_error_col

    2. Probability mode
       Shows each feature contribution on the predicted-probability axis.

       For each feature:
           bar start = baseline probability, E[f(X)]
           bar end   = feature-specific endpoint probability

       Default label format:
           P: 0.66 ± 0.02

       If show_shap_in_probability_label=True:
           P: 0.66 ± 0.02
           SHAP: +0.31 ± 0.02

       Error bars use:
           selected_endpoint_error_col / not_selected_endpoint_error_col

    Parameters
    ----------
    cohort_shap_summary : pandas.DataFrame
        Output from `build_cohort_shap_summary_table(...)`.

    models : str or sequence of str
        Model or models to plot. One figure is generated per model.

    x_axis_mode : {"contribution", "probability"}, default "contribution"
        Controls whether bars are displayed as SHAP contributions around 0
        or as endpoint probabilities anchored to the model baseline.

    selected_mean_col, not_selected_mean_col : str
        Columns containing mean SHAP contributions for the two cohort groups.

    selected_error_col, not_selected_error_col : str or None
        Error columns for contribution mode. These usually contain SEM values
        for mean SHAP contributions.

    selected_endpoint_col, not_selected_endpoint_col : str
        Endpoint probability columns for probability mode.

    selected_endpoint_error_col, not_selected_endpoint_error_col : str or None
        Error columns for probability mode. These usually contain SEM values
        for endpoint probabilities.

    selected_base_col, not_selected_base_col : str
        Baseline probability columns used in probability mode.

    sort_metric_col : str, default "balanced_delta_mean_shap"
        Column used to rank/select the displayed features.

    top_n : int or None, default 10
        Number of features to show per model. If None, all features are shown.

    selected_label, not_selected_label : str
        Legend labels for the two cohort groups.

    show_shap_in_probability_label : bool, default False
        If True, probability-mode labels include a second line showing the
        mean SHAP contribution and its uncertainty.

    show_baseline_line : bool, default True
        Whether to show the baseline probability reference line in probability
        mode.

    show_prediction_lines : bool, default False
        Whether to show cohort mean prediction reference lines in probability
        mode. Usually leave False because this plot is feature-specific, not a
        full sequential waterfall.

    model_alias : mapping or None
        Optional mapping from internal model names to display names.

    Returns
    -------
    outputs : dict or None
        If `return_plot_data=True`, returns a dictionary keyed by model name:

            {
                model_name: {
                    "fig": fig,
                    "ax": ax,
                    "plot_data": plot_data,
                    "x_axis_mode": x_axis_mode,
                }
            }

        Otherwise returns None.

    Notes
    -----
    In probability mode, the visible error bars correspond to uncertainty in
    the plotted endpoint probability.

    A label such as:

        P: 0.66 ± 0.02
        SHAP: +0.31 ± 0.02

    means that the feature moves the cohort from the model baseline probability
    to an endpoint probability of 0.66, and the mean SHAP contribution itself
    is +0.31.
    """

    if not isinstance(cohort_shap_summary, pd.DataFrame):
        raise TypeError("cohort_shap_summary must be a pandas DataFrame.")

    if x_axis_mode not in {"contribution", "probability"}:
        raise ValueError("x_axis_mode must be either 'contribution' or 'probability'.")

    if sort_by not in {"abs", "value"}:
        raise ValueError("sort_by must be either 'abs' or 'value'.")

    if isinstance(models, str):
        model_list = [models]
    else:
        model_list = list(models)

    if len(model_list) == 0:
        raise ValueError("models must contain at least one model.")

    if model_alias is None:
        model_alias = {}

    if selected_prediction_color is None:
        selected_prediction_color = selected_color

    if not_selected_prediction_color is None:
        not_selected_prediction_color = not_selected_color

    required_cols = {
        model_col,
        feature_col,
        selected_mean_col,
        not_selected_mean_col,
        sort_metric_col,
    }

    if x_axis_mode == "probability":
        required_cols.update(
            {
                selected_base_col,
                not_selected_base_col,
                selected_endpoint_col,
                not_selected_endpoint_col,
            }
        )

        if show_prediction_lines:
            required_cols.update(
                {
                    selected_prediction_col,
                    not_selected_prediction_col,
                }
            )

    missing_cols = required_cols - set(cohort_shap_summary.columns)
    if missing_cols:
        raise KeyError(
            f"cohort_shap_summary is missing required columns: {sorted(missing_cols)}"
        )

    optional_error_cols = []

    if x_axis_mode == "contribution":
        optional_error_cols.extend([selected_error_col, not_selected_error_col])
    else:
        optional_error_cols.extend(
            [
                selected_endpoint_error_col,
                not_selected_endpoint_error_col,
            ]
        )

        if show_shap_in_probability_label:
            optional_error_cols.extend([selected_error_col, not_selected_error_col])

    missing_error_cols = [
        col for col in optional_error_cols
        if col is not None and col not in cohort_shap_summary.columns
    ]

    if missing_error_cols:
        raise KeyError(
            f"cohort_shap_summary is missing error bar columns: {missing_error_cols}"
        )

    available_models = sorted(
        cohort_shap_summary[model_col].astype(str).unique()
    )

    missing_models = [
        m for m in model_list
        if str(m) not in available_models
    ]

    if missing_models:
        raise ValueError(
            f"Requested model(s) not found: {missing_models}. "
            f"Available models: {available_models}"
        )

    outputs = {}

    def _format_with_error(value, err, fmt, err_fmt):
        if err is None or pd.isna(err):
            return f"{fmt.format(value)} ± NA"
        return f"{fmt.format(value)} ± {err_fmt.format(abs(err))}"

    def _make_value_label(
        contribution: float,
        endpoint: float,
        *,
        endpoint_err=None,
        shap_err=None,
    ) -> str:
        if x_axis_mode == "probability":
            p_text = _format_with_error(
                endpoint,
                endpoint_err,
                value_format,
                error_format,
            )

            label = f"P: {p_text}"

            if show_shap_in_probability_label:
                shap_text = _format_with_error(
                    contribution,
                    shap_err,
                    contribution_format,
                    error_format,
                )
                label = f"{label}\nSHAP: {shap_text}"

            return label

        shap_text = _format_with_error(
            contribution,
            shap_err,
            contribution_format,
            error_format,
        )

        return f"SHAP: {shap_text}"

    def _plot_one_model(current_model_name: str):
        d = cohort_shap_summary.copy()
        d[model_col] = d[model_col].astype(str)
        d = d[d[model_col] == str(current_model_name)].copy()

        numeric_cols = [
            selected_mean_col,
            not_selected_mean_col,
            sort_metric_col,
        ]

        # SHAP contribution errors.
        if selected_error_col is not None:
            numeric_cols.append(selected_error_col)
        if not_selected_error_col is not None:
            numeric_cols.append(not_selected_error_col)

        # Probability-mode endpoint columns.
        if x_axis_mode == "probability":
            numeric_cols.extend(
                [
                    selected_base_col,
                    not_selected_base_col,
                    selected_endpoint_col,
                    not_selected_endpoint_col,
                ]
            )

            if selected_endpoint_error_col is not None:
                numeric_cols.append(selected_endpoint_error_col)
            if not_selected_endpoint_error_col is not None:
                numeric_cols.append(not_selected_endpoint_error_col)

            if show_prediction_lines:
                numeric_cols.extend(
                    [selected_prediction_col, not_selected_prediction_col]
                )

        for col in numeric_cols:
            d[col] = pd.to_numeric(d[col], errors="coerce")

        if d.empty:
            raise ValueError(f"No rows found for model={current_model_name!r}.")

        if d[sort_metric_col].isna().all():
            raise ValueError(
                f"All values in sort_metric_col={sort_metric_col!r} are NaN "
                f"for model={current_model_name!r}."
            )

        # Sort/select features.
        if sort_by == "abs":
            d["_sort_value"] = d[sort_metric_col].abs()
        else:
            d["_sort_value"] = d[sort_metric_col]

        d = d.sort_values("_sort_value", ascending=False, kind="mergesort")

        if top_n is not None:
            d = d.head(int(top_n)).copy()

        # Reverse for horizontal plotting.
        d = d.iloc[::-1].reset_index(drop=True)

        y = np.arange(len(d))

        selected_contrib = d[selected_mean_col].to_numpy(dtype=float)
        not_selected_contrib = d[not_selected_mean_col].to_numpy(dtype=float)

        selected_y = y + bar_height / 2
        not_selected_y = y - bar_height / 2

        # SHAP error arrays for contribution mode or labels.
        selected_shap_err = None
        not_selected_shap_err = None

        if selected_error_col is not None and selected_error_col in d.columns:
            selected_shap_err = d[selected_error_col].to_numpy(dtype=float)

        if not_selected_error_col is not None and not_selected_error_col in d.columns:
            not_selected_shap_err = d[not_selected_error_col].to_numpy(dtype=float)

        # Build bar geometry.
        if x_axis_mode == "contribution":
            selected_base = np.zeros_like(selected_contrib)
            not_selected_base = np.zeros_like(not_selected_contrib)

            selected_end = selected_contrib
            not_selected_end = not_selected_contrib

            selected_left = np.minimum(selected_base, selected_end)
            selected_width = np.abs(selected_contrib)

            not_selected_left = np.minimum(not_selected_base, not_selected_end)
            not_selected_width = np.abs(not_selected_contrib)

            selected_xerr = selected_shap_err
            not_selected_xerr = not_selected_shap_err

            reference_x = 0.0

            final_xlabel = (
                "Mean SHAP contribution to predicted probability"
                if xlabel is None else xlabel
            )

        else:
            selected_base = d[selected_base_col].to_numpy(dtype=float)
            not_selected_base = d[not_selected_base_col].to_numpy(dtype=float)

            selected_end = d[selected_endpoint_col].to_numpy(dtype=float)
            not_selected_end = d[not_selected_endpoint_col].to_numpy(dtype=float)

            selected_left = np.minimum(selected_base, selected_end)
            selected_width = np.abs(selected_end - selected_base)

            not_selected_left = np.minimum(not_selected_base, not_selected_end)
            not_selected_width = np.abs(not_selected_end - not_selected_base)

            selected_xerr = None
            not_selected_xerr = None

            if selected_endpoint_error_col is not None:
                selected_xerr = d[selected_endpoint_error_col].to_numpy(dtype=float)

            if not_selected_endpoint_error_col is not None:
                not_selected_xerr = d[not_selected_endpoint_error_col].to_numpy(dtype=float)

            reference_x = float(
                np.nanmean(np.concatenate([selected_base, not_selected_base]))
            )

            final_xlabel = (
                "Predicted probability"
                if xlabel is None else xlabel
            )

        fig, ax = plt.subplots(figsize=figsize)

        use_manual_endpoint_errors = x_axis_mode == "probability"

        # Bars.
        if use_manual_endpoint_errors:
            ax.barh(
                selected_y,
                selected_width,
                left=selected_left,
                height=bar_height,
                color=selected_color,
                label=selected_label,
                edgecolor="none",
                zorder=3,
            )

            ax.barh(
                not_selected_y,
                not_selected_width,
                left=not_selected_left,
                height=bar_height,
                color=not_selected_color,
                label=not_selected_label,
                edgecolor="none",
                zorder=3,
            )
        else:
            ax.barh(
                selected_y,
                selected_width,
                left=selected_left,
                xerr=selected_xerr,
                height=bar_height,
                color=selected_color,
                label=selected_label,
                edgecolor="none",
                ecolor=error_color,
                capsize=capsize,
                error_kw={
                    "linewidth": error_linewidth,
                    "capthick": error_linewidth,
                },
                zorder=3,
            )

            ax.barh(
                not_selected_y,
                not_selected_width,
                left=not_selected_left,
                xerr=not_selected_xerr,
                height=bar_height,
                color=not_selected_color,
                label=not_selected_label,
                edgecolor="none",
                ecolor=error_color,
                capsize=capsize,
                error_kw={
                    "linewidth": error_linewidth,
                    "capthick": error_linewidth,
                },
                zorder=3,
            )

        # Manual endpoint error bars for probability mode.
        if use_manual_endpoint_errors:
            if selected_xerr is not None:
                ax.errorbar(
                    selected_end,
                    selected_y,
                    xerr=selected_xerr,
                    fmt="none",
                    ecolor=error_color,
                    elinewidth=error_linewidth,
                    capsize=capsize,
                    capthick=error_linewidth,
                    zorder=5,
                )

            if not_selected_xerr is not None:
                ax.errorbar(
                    not_selected_end,
                    not_selected_y,
                    xerr=not_selected_xerr,
                    fmt="none",
                    ecolor=error_color,
                    elinewidth=error_linewidth,
                    capsize=capsize,
                    capthick=error_linewidth,
                    zorder=5,
                )

        # Reference line.
        if x_axis_mode == "contribution":
            ax.axvline(
                reference_x,
                color=zero_line_color,
                linewidth=1.2,
                linestyle="-",
                zorder=4,
            )
        else:
            if show_baseline_line:
                ax.axvline(
                    reference_x,
                    color=baseline_color,
                    linewidth=reference_linewidth,
                    linestyle=reference_linestyle,
                    zorder=4,
                    label=baseline_label,
                )

                if show_baseline_value_label:
                    ax.text(
                        reference_x,
                        baseline_value_label_y,
                        baseline_value_format.format(reference_x),
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=reference_label_fontsize,
                        fontweight=fontweight,
                        color=baseline_color,
                    )

        # Optional mean prediction lines.
        if x_axis_mode == "probability" and show_prediction_lines:
            selected_pred = float(d[selected_prediction_col].iloc[0])
            not_selected_pred = float(d[not_selected_prediction_col].iloc[0])

            selected_pred_label = (
                selected_prediction_label
                if selected_prediction_label is not None
                else f"{selected_label} mean prediction"
            )
            not_selected_pred_label = (
                not_selected_prediction_label
                if not_selected_prediction_label is not None
                else f"{not_selected_label} mean prediction"
            )

            ax.axvline(
                selected_pred,
                color=selected_prediction_color,
                linewidth=reference_linewidth,
                linestyle=reference_linestyle,
                zorder=4,
                label=selected_pred_label,
            )

            ax.axvline(
                not_selected_pred,
                color=not_selected_prediction_color,
                linewidth=reference_linewidth,
                linestyle=reference_linestyle,
                zorder=4,
                label=not_selected_pred_label,
            )

        # Y axis.
        ax.set_yticks(y)
        ax.set_yticklabels(
            d[feature_col].astype(str).tolist(),
            fontsize=tick_fontsize,
            fontweight=fontweight,
        )

        ax.tick_params(axis="x", labelsize=tick_fontsize)
        for tick_label in ax.get_xticklabels():
            tick_label.set_fontweight(fontweight)

        # X limits.
        if xlim is not None:
            ax.set_xlim(*xlim)
        else:
            all_x = []

            all_x += list(selected_left)
            all_x += list(selected_end)
            all_x += list(not_selected_left)
            all_x += list(not_selected_end)

            if selected_xerr is not None:
                all_x += list(selected_end - selected_xerr)
                all_x += list(selected_end + selected_xerr)

            if not_selected_xerr is not None:
                all_x += list(not_selected_end - not_selected_xerr)
                all_x += list(not_selected_end + not_selected_xerr)

            all_x.append(reference_x)

            if x_axis_mode == "probability" and show_prediction_lines:
                all_x.append(float(d[selected_prediction_col].iloc[0]))
                all_x.append(float(d[not_selected_prediction_col].iloc[0]))

            xmin = np.nanmin(all_x)
            xmax = np.nanmax(all_x)

            if xmin == xmax:
                xmin -= 0.1
                xmax += 0.1

            pad = 0.20 * (xmax - xmin)
            ax.set_xlim(xmin - pad, xmax + pad)

        xmin, xmax = ax.get_xlim()
        x_range = xmax - xmin
        value_offset = value_offset_fraction * x_range

        # Value labels.
        if show_group_values:
            for yy, contrib, endpoint, endpoint_err, shap_err in zip(
                selected_y,
                selected_contrib,
                selected_end,
                selected_xerr if selected_xerr is not None else np.full_like(selected_contrib, np.nan),
                selected_shap_err if selected_shap_err is not None else np.full_like(selected_contrib, np.nan),
            ):
                if endpoint >= reference_x:
                    xpos = endpoint + abs(endpoint_err if not pd.isna(endpoint_err) else 0) + value_offset
                    ha = "left"
                else:
                    xpos = endpoint - abs(endpoint_err if not pd.isna(endpoint_err) else 0) - value_offset
                    ha = "right"

                ax.text(
                    xpos,
                    yy,
                    _make_value_label(
                        contrib,
                        endpoint,
                        endpoint_err=endpoint_err,
                        shap_err=shap_err,
                    ),
                    va="center",
                    ha=ha,
                    fontsize=value_fontsize,
                    fontweight=fontweight,
                    color=selected_color,
                )

            for yy, contrib, endpoint, endpoint_err, shap_err in zip(
                not_selected_y,
                not_selected_contrib,
                not_selected_end,
                not_selected_xerr if not_selected_xerr is not None else np.full_like(not_selected_contrib, np.nan),
                not_selected_shap_err if not_selected_shap_err is not None else np.full_like(not_selected_contrib, np.nan),
            ):
                if endpoint >= reference_x:
                    xpos = endpoint + abs(endpoint_err if not pd.isna(endpoint_err) else 0) + value_offset
                    ha = "left"
                else:
                    xpos = endpoint - abs(endpoint_err if not pd.isna(endpoint_err) else 0) - value_offset
                    ha = "right"

                ax.text(
                    xpos,
                    yy,
                    _make_value_label(
                        contrib,
                        endpoint,
                        endpoint_err=endpoint_err,
                        shap_err=shap_err,
                    ),
                    va="center",
                    ha=ha,
                    fontsize=value_fontsize,
                    fontweight=fontweight,
                    color=not_selected_color,
                )

        display_model_name = model_alias.get(
            str(current_model_name),
            str(current_model_name),
        )

        final_title = (
            title if title is not None
            else f"{title_prefix}: {display_model_name}"
        )

        ax.set_title(
            final_title,
            fontsize=title_fontsize,
            fontweight=fontweight,
            pad=12,
        )

        ax.set_xlabel(
            final_xlabel,
            fontsize=label_fontsize,
            fontweight=fontweight,
        )

        ax.set_ylabel(
            ylabel,
            fontsize=label_fontsize,
            fontweight=fontweight,
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
        else:
            ax.grid(False)

        ax.set_axisbelow(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax.legend(
            loc="best",
            prop={"size": legend_fontsize, "weight": legend_fontweight},
            title="",
        )

        fig.tight_layout()
        plt.show()

        d = d.drop(columns=["_sort_value"], errors="ignore")

        d["_selected_bar_start"] = selected_left
        d["_selected_bar_end"] = selected_end
        d["_not_selected_bar_start"] = not_selected_left
        d["_not_selected_bar_end"] = not_selected_end

        if x_axis_mode == "probability":
            d["_baseline_reference"] = reference_x
        else:
            d["_zero_reference"] = reference_x

        return {
            "fig": fig,
            "ax": ax,
            "plot_data": d,
            "x_axis_mode": x_axis_mode,
        }

    for current_model_name in model_list:
        outputs[str(current_model_name)] = _plot_one_model(str(current_model_name))

    if return_plot_data:
        return outputs

    return None



# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Patient risk distributions
# ------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_patient_risk_distributions_by_outcome(
    df_pat: pd.DataFrame,
    *,
    model_name: str,
    split: str = "test",
    variants: Optional[Sequence[str]] = None,
    value_col: str = "p_mean",
    unit_col: Optional[str] = None,  # None -> auto: ["group","idx","subject_id"]

    # ---- colors ----
    outcome_palette: Optional[Mapping[str, str]] = None,
    jitter_palette: Optional[Mapping[str, str]] = None,

    # ---- violin ----
    inner: str = "box",
    cut: float = 0,
    linewidth: float = 1.2,
    saturation: float = 1.0,
    density_norm: str = "width",
    bw_adjust: float = 1.0,

    # ---- jitter ----
    show_jitter: bool = True,
    jitter: float = 0.12,
    point_size: float = 0.9,
    point_alpha: float = 0.25,

    # ---- figure/text ----
    figsize: tuple[float, float] = (10, 5),
    ylim: tuple[float, float] = (0.0, 1.0),
    title: Optional[str] = None,
    font_size: float = 11.0,
    legend_loc: str = "best",
    xlabel: str = "Prediction type",
    ylabel: str = "Predicted P(y=1)",

    # ---- annotations ----
    outcome_legend_alias: Optional[Mapping[str, str]] = None,
    show_counts_in_legend: bool = True,

    # ---- prevalence baseline ----
    prevalence: Union[bool, float] = True,  # True=auto, False=off, float=use value
    prevalence_color: str = "#D5F713",
    baseline_lw: float = 1.5,
    baseline_ls: str = "--",
) -> pd.DataFrame:
    """
    Plot distributions of PATIENT-LEVEL predicted probabilities by outcome, across calibration variants.

    This function consumes a patient-level table (`df_pat`) such as the output of pooled_patient_risk_summary.
    It plots grouped violins where:
      - x = variant (e.g., "uncalib", "beta")
      - hue = true outcome (negative vs positive)
      - y = value_col (e.g., "p_mean", "p_max", "p_q75", "p_softmax")

    Unlike df_long plots, this does NOT re-aggregate window-level rows; it assumes df_pat already contains
    the aggregation you want. If df_pat contains one row per patient×run (e.g., grouping="per_trial_fold"),
    the distribution reflects variability across CV runs. If df_pat contains one row per patient
    (grouping="all_trials"), the distribution reflects across-patient variability only.

    Returns the filtered plotting DataFrame.
    """
    # Defaults
    if outcome_palette is None:
        outcome_palette = {"0 (neg)": "#1587F8", "1 (pos)": "#F14949"}
    if jitter_palette is None:
        jitter_palette = {"0 (neg)": "black", "1 (pos)": "black"}

    required = {"model", "variant", "split", "y", value_col}
    missing = required - set(df_pat.columns)
    if missing:
        raise KeyError(f"df_pat missing required columns: {sorted(missing)}")

    # Resolve unit_col for prevalence/counts
    if unit_col is None:
        for cand in ("group", "idx", "subject_id"):
            if cand in df_pat.columns:
                unit_col = cand
                break
        if unit_col is None:
            # We can still plot, but can't compute unique counts/prevalence reliably
            unit_col = "__row__"
            df_pat = df_pat.copy()
            df_pat[unit_col] = np.arange(len(df_pat), dtype=int)

    # ---- filter to model + split ----
    d = df_pat[(df_pat["model"] == model_name) & (df_pat["split"] == split)].copy()
    if d.empty:
        raise ValueError(f"No rows found for model='{model_name}' and split='{split}'.")

    # types
    d["y"] = pd.to_numeric(d["y"], errors="coerce").astype(int)
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    # select variants
    if variants is None:
        variants = sorted(d["variant"].dropna().astype(str).unique().tolist())
    else:
        variants = [str(v) for v in variants]

    d = d[d["variant"].astype(str).isin(variants)].copy()
    if d.empty:
        raise ValueError(f"No rows found after filtering variants={variants}.")

    # ---- labels ----
    y_map = {0: "0 (neg)", 1: "1 (pos)"}
    d["y_label"] = d["y"].map(y_map)
    d["y_label"] = pd.Categorical(d["y_label"], categories=["0 (neg)", "1 (pos)"], ordered=True)
    d["variant"] = pd.Categorical(d["variant"].astype(str), categories=list(variants), ordered=True)

    # ---- counts/prevalence from unique patients ----
    base_unique = d.drop_duplicates([unit_col])[ [unit_col, "y"] ]
    n_neg = int((base_unique["y"] == 0).sum())
    n_pos = int((base_unique["y"] == 1).sum())

    prev_val: Optional[float] = None
    if isinstance(prevalence, bool):
        if prevalence:
            prev_val = float(base_unique["y"].mean())
        else:
            prev_val = None
    else:
        prev_val = float(prevalence)
        if not (0.0 <= prev_val <= 1.0):
            raise ValueError(f"prevalence must be in [0,1]; got {prev_val}")

    # ---- plotting ----
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    sns.violinplot(
        data=d,
        x="variant",
        y=value_col,
        hue="y_label",
        palette=outcome_palette,
        inner=inner,
        cut=cut,
        linewidth=linewidth,
        saturation=saturation,
        density_norm=density_norm,  # scale=scale,
        bw_adjust=bw_adjust,
        dodge=True,
        ax=ax,
    )


    if show_jitter:
        # draw jitter points but avoid adding extra legend entries
        for ylab in ["0 (neg)", "1 (pos)"]:
            sub = d[d["y_label"] == ylab]
            sns.stripplot(
                data=sub,
                x="variant",
                y=value_col,
                dodge=True,
                jitter=jitter,
                size=point_size,
                alpha=point_alpha,
                color=jitter_palette.get(ylab, "black"),
                linewidth=0,
                ax=ax,
            )

    # prevalence baseline
    baseline_handle = None
    baseline_label = None
    if prev_val is not None:
        baseline_label = f"Prevalence = {prev_val:.2f}"
        baseline_handle = ax.axhline(
            prev_val,
            color=prevalence_color,
            lw=baseline_lw,
            ls=baseline_ls,
            label=baseline_label,
            zorder=0,
        )

    ax.set_ylim(*ylim)

    def _pretty_value(vc: str) -> str:
        return {
            "p_mean": "mean",
            "p_median": "median",
            "p_max": "max",
            "p_softmax": "softmax",
        }.get(vc, vc)

    if title is None:
        title = f"{model_name} — {split.title()} patient-risk distributions ({_pretty_value(value_col)})"
    fig.suptitle(title, fontsize=font_size + 2, fontweight="bold", y=0.92)

    # bold axes
    ax.set_xlabel(xlabel, fontsize=font_size, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=font_size, fontweight="bold")
    ax.tick_params(axis="both", labelsize=font_size)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")

    # Legend labels
    if outcome_legend_alias is None:
        outcome_legend_alias = {"0 (neg)": "Neg", "1 (pos)": "Pos"}

    neg_name = outcome_legend_alias.get("0 (neg)", "Neg")
    pos_name = outcome_legend_alias.get("1 (pos)", "Pos")

    if show_counts_in_legend:
        label_map = {
            "0 (neg)": f"{neg_name} (n={n_neg:,})",
            "1 (pos)": f"{pos_name} (n={n_pos:,})",
        }
    else:
        label_map = {"0 (neg)": neg_name, "1 (pos)": pos_name}

    # Clean legend (keep only two outcomes + baseline)
    handles, labels = ax.get_legend_handles_labels()
    keep_core = ["0 (neg)", "1 (pos)"]

    uniq = {}
    H, L = [], []
    for h, l in zip(handles, labels):
        if l in keep_core and l not in uniq:
            uniq[l] = True
            H.append(h)
            L.append(label_map[l])

    if baseline_handle is not None and baseline_label is not None:
        H.append(baseline_handle)
        L.append(baseline_label)

    leg = ax.legend(H, L, title="True label", loc=legend_loc, prop={"size": font_size, "weight": "bold"})
    leg.get_title().set_fontweight("bold")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

    return d


def plot_risk_distributions_by_outcome(
    df_long: pd.DataFrame,
    *,
    model_name: str,
    split: str = "test",
    variants: Optional[Sequence[str]] = None,
    value_col: str = "p_mean",

    # ---- colors ----
    outcome_palette: Optional[Mapping[str, str]] = None,  # violin fill by y_label
    jitter_palette: Optional[Mapping[str, str]] = None,   # jitter colors by y_label

    # ---- aggregation ----
    agg_stats: Sequence[str] = ("mean", "median", "std", "min", "max"),

    # ---- violin ----
    inner: str = "box",
    cut: float = 0,
    linewidth: float = 1.2,
    saturation: float = 1.0,
    density_norm: str = "width", # scale
    bw_adjust: float = 1.0,

    # ---- jitter ----
    show_jitter: bool = True,
    jitter: float = 0.12,
    point_size: float = 0.9,
    point_alpha: float = 0.25,

    # ---- figure/text ----
    figsize: tuple[float, float] = (10, 4),
    ylim: tuple[float, float] = (0.0, 1.0),
    title: Optional[str] = None,
    font_size: float = 11.0,
    legend_loc: str = "best",
    xlabel: str = "Prediction type",
    ylabel: str = "Predicted P(y=1)",

    # ---- annotations ----
    outcome_legend_alias: Optional[Mapping[str, str]] = None,
    show_counts_in_legend: bool = True,

    # ---- prevalence baseline (SINGLE PARAM) ----
    prevalence: Union[bool, float] = True,   # True=auto, False=off, float=use value
    prevalence_color: str = "#D5F713",
    baseline_lw: float = 1.5,
    baseline_ls: str = "--",
) -> pd.DataFrame:
    """
    Plot grouped violin distributions of predicted positive-class probabilities by outcome, with optional jitter.

    This function consumes a long-format prediction table (`df_long`) containing repeated out-of-sample
    probabilities per patient/row across CV runs/folds. It first aggregates per (model, variant, idx)
    within the requested `split` (e.g., test), then plots grouped violins where:
    - x = calibration/model variant (e.g., "uncalib", "beta")
    - hue = true outcome (negative vs positive)
    Optionally overlays jittered points, adds outcome counts to the legend, and draws a prevalence
    baseline line (auto-computed from df_long when prevalence=True).

    Parameters
    ----------
    df_long:
        Long table with at least columns: ["model","variant","split","idx","y","p"].
    model_name:
        Model key to plot (must match df_long["model"]).
    split:
        Which split to plot (e.g., "test" or "train_oof"), matched against df_long["split"].
    variants:
        Variants to include (subset of df_long["variant"]); if None, uses all available for the model+split.
    value_col:
        Aggregated probability column to plot (e.g., "p_mean" or "p_median").

    outcome_palette / jitter_palette:
        Dict mapping outcome labels {"0 (neg)","1 (pos)"} to colors for violins / jitter points.

    agg_stats:
        Which summary stats to compute from repeated probabilities per idx (controls which p_* columns exist).

    inner, cut, linewidth, saturation, density_norm, bw_adjust:
        Violinplot styling controls.

    show_jitter, jitter, point_size, point_alpha:
        Jitter overlay controls.

    figsize, ylim, title, font_size, legend_loc, xlabel, ylabel:
        Figure and text styling controls. If title is None, a concise title is auto-generated.

    outcome_legend_alias:
        Optional mapping to rename outcomes in legend (e.g., {"0 (neg)":"TD","1 (pos)":"ASD"}).
    show_counts_in_legend:
        If True, appends outcome counts (unique idx) to legend labels.

    prevalence:
        True -> auto-compute prevalence from unique idx in df_long (within model+split),
        False -> disable baseline,
        float -> use the provided prevalence value.
    prevalence_color, baseline_lw, baseline_ls:
        Styling for the prevalence baseline.

    Returns
    -------
    pd.DataFrame
        The aggregated table used for plotting (one row per (model, variant, idx)).
    """
    # Defaults
    if outcome_palette is None:
        outcome_palette = {"0 (neg)": "#1587F8", "1 (pos)": "#F14949"}
    if jitter_palette is None:
        jitter_palette = {"0 (neg)": "black", "1 (pos)": "black"}

    # ---- filter to model + split ----
    d = df_long[(df_long["model"] == model_name) & (df_long["split"] == split)].copy()
    if d.empty:
        raise ValueError(f"No rows found for model='{model_name}' and split='{split}'.")

    # types
    d["y"] = d["y"].astype(int)
    d["p"] = d["p"].astype(float)

    # select variants
    if variants is None:
        variants = sorted(d["variant"].astype(str).unique().tolist())
    else:
        variants = list(variants)

    d = d[d["variant"].isin(variants)].copy()
    if d.empty:
        raise ValueError(f"No rows found after filtering variants={variants}.")

    # ---- labels (avoid numeric-looking categories) ----
    y_map = {0: "0 (neg)", 1: "1 (pos)"}
    d["y_label"] = d["y"].map(y_map)
    d["y_label"] = pd.Categorical(d["y_label"], categories=["0 (neg)", "1 (pos)"], ordered=True)
    d["variant"] = pd.Categorical(d["variant"], categories=list(variants), ordered=True)

    # ---- auto detect number of runs ----
    n_runs = None
    if "trial" in d.columns:
        try:
            n_runs = int(pd.Series(d["trial"]).nunique())
        except Exception:
            n_runs = None

    # ---- counts from unique idx (correct even if df_long repeats) ----
    base_unique = d.drop_duplicates("idx")[["idx", "y"]]
    n_neg = int((base_unique["y"] == 0).sum())
    n_pos = int((base_unique["y"] == 1).sum())

    # ---- prevalence baseline value (single param logic) ----
    prev_val: Optional[float] = None
    if isinstance(prevalence, bool):
        if prevalence:  # auto
            prev_val = float(base_unique["y"].mean())
        else:
            prev_val = None
    else:
        prev_val = float(prevalence)
        if not (0.0 <= prev_val <= 1.0):
            raise ValueError(f"prevalence must be in [0,1]; got {prev_val}")

    # ---- aggregate per (model, variant, idx) ----
    #grp = d.groupby(["model", "variant", "idx"], as_index=False)
    grp = d.groupby(["model", "variant", "idx"], as_index=False, observed=False)


    agg_dict = {"y": ("y", "first"), "n_preds": ("p", "size")}
    if "mean" in agg_stats:   agg_dict["p_mean"]   = ("p", "mean")
    if "median" in agg_stats: agg_dict["p_median"] = ("p", "median")
    if "std" in agg_stats:    agg_dict["p_std"]    = ("p", "std")
    if "min" in agg_stats:    agg_dict["p_min"]    = ("p", "min")
    if "max" in agg_stats:    agg_dict["p_max"]    = ("p", "max")

    df_agg = grp.agg(**agg_dict)

    if value_col not in df_agg.columns:
        raise KeyError(f"value_col='{value_col}' not in aggregated columns: {list(df_agg.columns)}")

    df_agg["y_label"] = df_agg["y"].map(y_map)
    df_agg["y_label"] = pd.Categorical(df_agg["y_label"], categories=["0 (neg)", "1 (pos)"], ordered=True)
    df_agg["variant"] = pd.Categorical(df_agg["variant"], categories=list(variants), ordered=True)
    df_agg[value_col] = df_agg[value_col].astype(float)

    if n_runs is None:
        n_runs = int(df_agg["n_preds"].median())

    # ---- plotting ----
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    sns.violinplot(
        data=df_agg,
        x="variant",
        y=value_col,
        hue="y_label",
        palette=outcome_palette,
        inner=inner,
        cut=cut,
        linewidth=linewidth,
        saturation=saturation,
        density_norm=density_norm,  #scale=scale,
        bw_adjust=bw_adjust,
        dodge=True,
        ax=ax,
    )

    if show_jitter:
        for ylab in ["0 (neg)", "1 (pos)"]:
            sub = df_agg[df_agg["y_label"] == ylab]
            sns.stripplot(
                data=sub,
                x="variant",
                y=value_col,
                hue="y_label",
                dodge=True,
                jitter=jitter,
                size=point_size,
                alpha=point_alpha,
                palette=jitter_palette, #color=jitter_palette.get(ylab, "black"),
                linewidth=0,
                ax=ax,
            )

    # prevalence baseline
    baseline_handle = None
    baseline_label = None
    if prev_val is not None:
        baseline_label = f"Prevalence = {prev_val:.2f}"
        baseline_handle = ax.axhline(
            prev_val,
            color=prevalence_color,
            lw=baseline_lw,
            ls=baseline_ls,
            label=baseline_label,
            zorder=0,
        )

    ax.set_ylim(*ylim)

    # ---- single-line title (no overlap) ----
    def _pretty_split(s: str) -> str:
        s = str(s).strip().lower()
        if s in {"test", "outer_test"}:
            return "Test"
        if s in {"train", "train_oof", "oof", "outer_train"}:
            return "Train"
        # fallback: title-case the raw string
        return s.replace("_", " ").title()

    def _pretty_value(vc: str) -> str:
        vc = str(vc)
        return {"p_mean": "mean", "p_median": "median"}.get(vc, vc)

    if title is None:
        title = f"{model_name} — {_pretty_split(split)} risk distributions ({_pretty_value(value_col)} over {n_runs} trials)"

    fig.suptitle(title, fontsize=font_size + 2, fontweight="bold", y=0.85)


    # ---- bold axes ----
    ax.set_xlabel(xlabel, fontsize=font_size, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=font_size, fontweight="bold")

    ax.tick_params(axis="both", labelsize=font_size)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")

    # ---- Legend cleanup (hue drawn multiple times) ----
    handles, labels = ax.get_legend_handles_labels()
    keep_core = ["0 (neg)", "1 (pos)"]

    # Default display names if user doesn't pass aliases
    if outcome_legend_alias is None:
        outcome_legend_alias = {"0 (neg)": "Neg", "1 (pos)": "Pos"}

    neg_name = outcome_legend_alias.get("0 (neg)", "Neg")
    pos_name = outcome_legend_alias.get("1 (pos)", "Pos")

    if show_counts_in_legend:
        label_map = {
            "0 (neg)": f"{neg_name} (n={n_neg:,})",
            "1 (pos)": f"{pos_name} (n={n_pos:,})",
        }
    else:
        label_map = {
            "0 (neg)": neg_name,
            "1 (pos)": pos_name,
        }


    uniq = {}
    H, L = [], []
    for h, l in zip(handles, labels):
        if l in keep_core and l not in uniq:
            uniq[l] = True
            H.append(h)
            L.append(label_map[l])

    if baseline_handle is not None and baseline_label is not None:
        H.append(baseline_handle)
        L.append(baseline_label)

    leg = ax.legend(H, L, title="True label", loc=legend_loc, prop={"size": font_size, "weight": "bold"})
    leg.get_title().set_fontweight("bold")

    # reserve space for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

    return df_agg


def build_model_prediction_rows(
    all_results: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    model_name: str | Sequence[str] | None = None,   
    groups_all: Optional[np.ndarray] = None,
    group_id_to_key: Optional[Mapping[int, Tuple[str, str]]] = None,  # group -> (label_str, subject_id)
    methods: Optional[Sequence[str]] = None,
    include_uncalibrated: bool = True,
    include_test: bool = True,
    include_train_oof: bool = False,
    unit_col: str = "idx",
) -> pd.DataFrame:
    """
    Build long-format model prediction rows from nested-CV/internal model outputs.

    This function converts fold-level prediction outputs from `all_results` into a
    tidy prediction table with one row per predicted example.

    The output is intended for downstream patient-level risk pooling, risk ranking,
    threshold analyses, diagnostic enrichment, prognostic analyses, and other
    post-modeling workflows.

    This is separate from `build_external_prediction_rows(...)`, which consumes
    externally scored validation predictions and uses a `calibration` column with
    split="external".

    Row granularity
    ---------------
    Each output row corresponds to a single prediction for a single index `idx` within a given:
      - model
      - variant (uncalibrated or a calibration method)
      - split ("test" and/or "train_oof")
      - trial and outer_fold

    Group / patient metadata (optional)
    -----------------------------------
    If both `groups_all` and `group_id_to_key` are provided, the function adds group-level
    identifiers for each row by mapping `idx -> group` and then `group -> (group_label, subject_id)`:
      - group: integer patient/group id
      - group_label: e.g., "ASD" / "TD"
      - subject_id: e.g., "NDAR..."

    If grouping info is not provided, group-level columns are omitted and `unit_col` may be
    included as an identifier for downstream aggregation.

    Parameters
    ----------
    all_results:
        Mapping from model name -> sequence of fold dictionaries (trial/outer_fold) containing
        indices, labels, and prediction arrays (including optional calibrated predictions).

    model_name:
        Controls which models to include:
          - None (default): include ALL models in `all_results`
          - str: include a single model
          - Sequence[str]: include only the specified models
        Model names must match keys in `all_results`.

    methods:
        Calibration methods to include (e.g., ["beta"]).
        If None, methods are discovered per model by scanning keys that start with
        "calib_test_predictions_" in that model's fold dictionaries.

    include_uncalibrated:
        If True, include the uncalibrated variant ("uncalib") using:
          - test:  "y_test_scores"
          - train: "cv_uncalib_train_predictions" (when include_train_oof=True)

    include_test:
        If True, include outer test predictions (split="test") using "outer_test_idx"/"y_test".

    include_train_oof:
        If True, include outer-train out-of-fold predictions (split="train_oof") using
        "outer_train_idx"/"y_train" and CV prediction keys.

    unit_col:
        In the non-grouped setting (no groups_all / group_id_to_key), an additional identifier
        column name to include per row (default "idx"). This can be useful if downstream code
        expects a "unit id" column even when patient groups are unavailable.

    Returns
    -------
    pd.DataFrame
        Long-form predictions table.

        Always included columns:
          ["model", "variant", "split", "trial", "outer_fold", "idx", "y", "p"]

        Included only when grouping info is provided:
          + ["group", "group_label", "subject_id"]

        Included only in the non-grouped case:
          + [unit_col] (if unit_col is not "idx", it will still be present explicitly)

    Raises
    ------
    KeyError:
        If requested model(s) are not present in `all_results`, or required prediction keys are missing.
    ValueError:
        If idx/y/p lengths mismatch for any fold/variant/split.
    IndexError:
        If idx values are out of bounds for `groups_all` when grouping info is provided.
    """

    # -------------------------
    # Resolve model list
    # -------------------------
    if model_name is None:
        model_names: List[str] = list(all_results.keys())
    elif isinstance(model_name, str):
        model_names = [model_name]
    else:
        model_names = list(model_name)

    missing = [m for m in model_names if m not in all_results]
    if missing:
        raise KeyError(
            f"Model(s) not found in all_results: {missing}. "
            f"Available: {list(all_results.keys())}"
        )

    have_groups = (groups_all is not None) and (group_id_to_key is not None)
    if have_groups:
        groups_all = np.asarray(groups_all)

    all_dfs: List[pd.DataFrame] = []

    # -------------------------
    # Loop models, reuse your existing logic
    # -------------------------
    for mname in model_names:
        folds = all_results[mname]

        # Discover calibration methods if not provided (PER MODEL)
        if methods is None:
            discovered = set()
            for r in folds:
                for k in r.keys():
                    if k.startswith("calib_test_predictions_"):
                        discovered.add(k.replace("calib_test_predictions_", "", 1))
            methods_list = sorted(discovered)
        else:
            methods_list = list(methods)

        variants: List[str] = []
        if include_uncalibrated:
            variants.append("uncalib")
        variants.extend(methods_list)

        rows: List[Dict[str, Any]] = []

        def _append_rows(
            *,
            idx_arr: np.ndarray,
            y_arr: np.ndarray,
            p_arr: np.ndarray,
            split_name: str,
            trial: Any,
            outer_fold: Any,
            variant: str,
        ) -> None:
            idx_arr = np.asarray(idx_arr, dtype=int)
            y_arr = np.asarray(y_arr, dtype=int)
            p_arr = np.asarray(p_arr, dtype=float)

            if len(idx_arr) != len(y_arr) or len(idx_arr) != len(p_arr):
                raise ValueError(
                    f"Length mismatch: model={mname}, trial={trial}, outer_fold={outer_fold}, "
                    f"variant={variant}, split={split_name} "
                    f"len(idx)={len(idx_arr)}, len(y)={len(y_arr)}, len(p)={len(p_arr)}"
                )

            if have_groups:
                assert groups_all is not None and group_id_to_key is not None

                if idx_arr.max(initial=-1) >= len(groups_all) or idx_arr.min(initial=0) < 0:
                    raise IndexError(
                        f"Some idx values are out of bounds for groups_all (len={len(groups_all)}). "
                        f"idx range: [{idx_arr.min()}, {idx_arr.max()}]"
                    )

                g_arr = groups_all[idx_arr]

                # lookup label_str and subject_id per group
                label_strs: List[Optional[str]] = []
                subject_ids: List[Optional[str]] = []
                for g in g_arr:
                    lab, sid = group_id_to_key.get(int(g), (None, None))
                    label_strs.append(lab)
                    subject_ids.append(sid)

                for i, g, lab, sid, yy, pp in zip(idx_arr, g_arr, label_strs, subject_ids, y_arr, p_arr):
                    rows.append({
                        "model": mname,
                        "variant": variant,
                        "split": split_name,
                        "trial": trial,
                        "outer_fold": outer_fold,
                        "idx": int(i),
                        "group": int(g),
                        "group_label": lab,
                        "subject_id": sid,
                        "y": int(yy),
                        "p": float(pp),
                    })
            else:
                for i, yy, pp in zip(idx_arr, y_arr, p_arr):
                    rows.append({
                        "model": mname,
                        "variant": variant,
                        "split": split_name,
                        "trial": trial,
                        "outer_fold": outer_fold,
                        "idx": int(i),
                        unit_col: int(i) if unit_col != "idx" else int(i),
                        "y": int(yy),
                        "p": float(pp),
                    })

        for r in folds:
            trial = r.get("trial", None)
            outer_fold = r.get("outer_fold", None)

            # ---------- outer test ----------
            if include_test:
                idx = np.asarray(r["outer_test_idx"], dtype=int)
                y = np.asarray(r["y_test"], dtype=int)

                for v in variants:
                    key = "y_test_scores" if v == "uncalib" else f"calib_test_predictions_{v}"
                    if key not in r:
                        continue
                    p = np.asarray(r[key], dtype=float)

                    _append_rows(
                        idx_arr=idx,
                        y_arr=y,
                        p_arr=p,
                        split_name="test",
                        trial=trial,
                        outer_fold=outer_fold,
                        variant=v,
                    )

            # ---------- train OOF (optional) ----------
            if include_train_oof:
                idx_tr = np.asarray(r["outer_train_idx"], dtype=int)
                y_tr = np.asarray(r["y_train"], dtype=int)

                for v in variants:
                    key = "cv_uncalib_train_predictions" if v == "uncalib" else f"cv_calib_train_predictions_{v}"
                    if key not in r:
                        continue
                    p_tr = np.asarray(r[key], dtype=float)

                    _append_rows(
                        idx_arr=idx_tr,
                        y_arr=y_tr,
                        p_arr=p_tr,
                        split_name="train_oof",
                        trial=trial,
                        outer_fold=outer_fold,
                        variant=v,
                    )

        df_m = pd.DataFrame(rows)

        # dtypes
        if not df_m.empty:
            df_m["model"] = df_m["model"].astype(str)
            df_m["variant"] = df_m["variant"].astype(str)
            df_m["split"] = df_m["split"].astype(str)
            df_m["idx"] = df_m["idx"].astype(int)
            df_m["y"] = df_m["y"].astype(int)
            df_m["p"] = df_m["p"].astype(float)
            if have_groups:
                df_m["group"] = df_m["group"].astype(int)

        all_dfs.append(df_m)

    # Combine across models
    if len(all_dfs) == 0:
        return pd.DataFrame()

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Optional: stable ordering (nice for debugging)
    sort_cols = ["model", "variant", "split", "trial", "outer_fold", "idx"]
    sort_cols = [c for c in sort_cols if c in df_all.columns]
    if sort_cols:
        df_all = df_all.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    return df_all


def pooled_patient_risk_summary(
    df_long: pd.DataFrame,
    *,
    agg: Literal["mean", "median", "max", "quantile", "softmax"] = "mean",
    quantile: float = 0.75,
    beta: float = 5.0,
    eps: float = 1e-6,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
    ddof: int = 0,
    grouping: Literal["all_trials", "per_trial_fold"] | None = "all_trials",
    unit_col: Optional[str] = "group",
    splits: Optional[Sequence[str]] = None,
    include_test: bool = True,
    include_train_oof: bool = False,
    truncate_decimals: Optional[int] = None,
) -> pd.DataFrame:
    """
    Aggregate row-/window-level predicted probabilities into unit-level (e.g., patient-level) summaries.

    This function takes a "long" prediction table where each unit (patient/subject) may appear multiple
    times (e.g., many EEG windows, and/or repeated cross-validation runs), and returns one row per unit
    (within each model/variant/split), summarizing the distribution of predicted probabilities.

    Expected input (df_long)
    ------------------------
    Must include:
    - model:   model name/identifier
    - variant: calibration/variant label (e.g., "beta", "uncalib")
    - split:   split label (e.g., "test", "train_oof")
    - p:       predicted probability in [0, 1]
    - y:       true label (0/1)
    - unit_col: column identifying the unit you want to aggregate to (default "group"; can be "idx")

    May include (optional):
    - subject_id: string subject identifier
    - group_label: human-readable label (if missing, the output uses y as group_label)

    If grouping="per_trial_fold", df_long must also include:
    - trial, outer_fold

    Key idea
    --------
    Within each grouping bucket, probabilities are optionally winsorized (capped at within-bucket
    quantiles), then aggregated to a single "center" probability (mean/median/max/quantile/softmax),
    and a within-bucket spread (std) is computed.

    Parameters
    ----------
    agg:
        How to summarize probabilities within each unit bucket:
        - "mean": mean(p_used)
        - "median": median(p_used)
        - "max": max(p_used)
        - "quantile": quantile(p_used, quantile)
        - "softmax": softmax-pooled weighted mean emphasizing higher-evidence windows
            weights = softmax(beta * logit(p_used)), p_softmax = sum_i w_i * p_used_i
    quantile:
        Quantile used when agg="quantile" (e.g., 0.75 for 75th percentile).
    beta, eps:
        Softmax sharpness and numerical stability when agg="softmax".
    lower_q, upper_q:
        Winsorization cutoffs applied *within each group bucket*.
        - lower_q == 0.0 disables lower capping
        - upper_q == 1.0 disables upper capping
        - lower_q=0.0 and upper_q=1.0 disables winsorization entirely (p_used == p)
    ddof:
        Degrees of freedom for std computation (np.std).
    grouping:
        Defines the groupby key (the aggregation unit):
        - "all_trials":      ["model","variant","split", unit_col]
            Pools all rows for a unit across all trials/folds (if present).
        - "per_trial_fold":  ["model","variant","split","trial","outer_fold", unit_col]
            Produces one unit summary per CV run (trial × outer_fold).
        - None: alias for "all_trials".
    unit_col:
        Column name identifying the unit being summarized (e.g., "group" or "idx").
    splits / include_test / include_train_oof:
        Controls which split rows are kept before aggregation.
        If `splits` is provided it is used directly; otherwise it is built from the include_* flags.
    truncate_decimals:
        If not None, truncate probability-style output columns to this many decimal places
        after all calculations are complete. This is truncation, not rounding.

    Returns
    -------
    pd.DataFrame
        One row per grouping key with:
        - grouping key columns (depends on grouping)
        - subject_id (if present, else NaN)
        - group_label (if present, else y)
        - y
        - n_windows: number of non-NaN probabilities used
        - p_mean / p_median / p_max / p_qXX / p_softmax (depending on agg)
        - p_total_std: std of p_used within the bucket
        - winsorization metadata: lower_q, upper_q, p_cap_low, p_cap_high
        - softmax metadata: beta, eps (if agg="softmax")
        - quantile metadata: quantile (if agg="quantile")
    """

    def _truncate_decimals(x: float, decimals: int):
        if pd.isna(x):
            return x
        factor = 10 ** decimals
        return np.trunc(float(x) * factor) / factor

    # -------------------------
    # Split filtering
    # -------------------------
    if "split" not in df_long.columns:
        raise KeyError("df_long must contain a 'split' column for split filtering.")

    if splits is not None:
        splits_list = list(splits)
        if len(splits_list) == 0:
            raise ValueError("If provided, splits must be a non-empty list/sequence of split names.")
    else:
        splits_list = []
        if include_test:
            splits_list.append("test")
        if include_train_oof:
            splits_list.append("train_oof")
        if len(splits_list) == 0:
            raise ValueError(
                "No splits selected. Set include_test/include_train_oof to True, "
                "or pass splits=['test', 'train_oof', ...]."
            )

    d = df_long[df_long["split"].isin(splits_list)].copy()

    if d.empty:
        present = sorted(df_long["split"].dropna().unique().tolist())
        raise ValueError(
            f"After filtering, no rows remain for splits={splits_list}. "
            f"Splits present in df_long: {present}"
        )

    # -------------------------
    # Infer unit_col if needed
    # -------------------------
    if unit_col is None:
        for cand in ("group", "subject_id", "idx"):
            if cand in d.columns:
                unit_col = cand
                break
        if unit_col is None:
            raise KeyError("Could not infer unit_col. Please pass unit_col='group' or 'idx' (or another id column).")

    if unit_col not in d.columns:
        raise KeyError(f"unit_col='{unit_col}' not found in df_long columns.")

    # -------------------------
    # grouping=None -> "all_trials"
    # -------------------------
    if grouping is None:
        grouping = "all_trials"

    # -------------------------
    # Grouping schemes
    # -------------------------
    GROUPING_SCHEMES = {
        "all_trials": ["model", "variant", "split", unit_col],
        "per_trial_fold": ["model", "variant", "split", "trial", "outer_fold", unit_col],
    }
    if grouping not in GROUPING_SCHEMES:
        raise ValueError(f"Unknown grouping='{grouping}'. Choose one of {list(GROUPING_SCHEMES)}")

    group_cols = GROUPING_SCHEMES[grouping]

    # -------------------------
    # Validation
    # -------------------------
    required_base = {"model", "variant", "split", "y", "p", unit_col}
    missing = required_base - set(d.columns)
    if missing:
        raise KeyError(f"df_long missing required columns: {sorted(missing)}")

    need_cols = set(group_cols) - set(d.columns)
    if need_cols:
        raise KeyError(f"grouping='{grouping}' requires missing columns: {sorted(need_cols)}")

    d["p"] = pd.to_numeric(d["p"], errors="coerce")

    # -------------------------
    # Output column naming
    # -------------------------
    if agg == "quantile":
        q_tag = int(round(quantile * 100))
        center_col = f"p_q{q_tag}"
    elif agg == "softmax":
        center_col = "p_softmax"
    else:
        center_col = {"mean": "p_mean", "median": "p_median", "max": "p_max"}[agg]

    out_rows: list[dict] = []

    apply_low = (lower_q > 0.0)
    apply_high = (upper_q < 1.0)

    for keys, gdf in d.groupby(group_cols, sort=False):
        p = gdf["p"].to_numpy(dtype=float)
        p = p[~np.isnan(p)]
        if p.size == 0:
            continue

        # Winsorize (optional, per side)
        if not apply_low and not apply_high:
            lo = np.nan
            hi = np.nan
            p_used = p
        else:
            lo = float(np.quantile(p, lower_q)) if apply_low else np.nan
            hi = float(np.quantile(p, upper_q)) if apply_high else np.nan
            lo_clip = lo if apply_low else -np.inf
            hi_clip = hi if apply_high else np.inf
            p_used = np.clip(p, lo_clip, hi_clip)

        # Aggregate
        if agg == "mean":
            p_center = float(np.mean(p_used))
        elif agg == "median":
            p_center = float(np.median(p_used))
        elif agg == "max":
            p_center = float(np.max(p_used))
        elif agg == "quantile":
            p_center = float(np.quantile(p_used, quantile))
        else:  # "softmax"
            p_clip = np.clip(p_used, eps, 1.0 - eps)
            s = np.log(p_clip) - np.log1p(-p_clip)  # logit(p)
            t = beta * s
            t = t - np.max(t)
            w = np.exp(t)
            w_sum = np.sum(w)
            if not np.isfinite(w_sum) or w_sum == 0.0:
                p_center = float(np.mean(p_used))
            else:
                w = w / w_sum
                p_center = float(np.sum(w * p_used))

        # Spread
        p_std = float(np.std(p_used, ddof=ddof))

        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            {
                "grouping": grouping,
                "unit_col": unit_col,
                "subject_id": gdf["subject_id"].iloc[0] if "subject_id" in gdf.columns else np.nan,
                "group_label": gdf["group_label"].iloc[0] if "group_label" in gdf.columns else int(gdf["y"].iloc[0]),
                "y": int(gdf["y"].iloc[0]),
                "n_windows": int(p.size),
                center_col: p_center,
                "p_total_std": p_std,
                "lower_q": float(lower_q),
                "upper_q": float(upper_q),
                "p_cap_low": lo,
                "p_cap_high": hi,
            }
        )

        if agg == "quantile":
            row["quantile"] = float(quantile)
        if agg == "softmax":
            row["beta"] = float(beta)
            row["eps"] = float(eps)

        out_rows.append(row)

    out = pd.DataFrame(out_rows)

    if truncate_decimals is not None:
        if truncate_decimals < 0:
            raise ValueError("truncate_decimals must be >= 0 or None.")
        prob_cols = [c for c in out.columns if c.startswith("p_")]
        out[prob_cols] = out[prob_cols].apply(
            lambda s: s.map(lambda x: _truncate_decimals(x, truncate_decimals))
        )

    return out



def plot_ranked_patients_patient_level(
    df_pat: pd.DataFrame,
    *,
    model: Optional[str] = None,
    variants: Optional[Sequence[str]] = None,                  # REQUIRED
    colors: Sequence[str] = ("#5BA8F5", "#EC6868"),            # label colors in order of labels_to_plot
    split: str = "test",
    group_label: Optional[Union[str, Sequence[str]]] = None,   # None => auto-detect all labels
    center_col: Optional[str] = None,                          # auto-detect if None (supports mean/median/max/softmax/quantiles)
    std_col: str = "p_total_std",
    prob_label: str = "ASD",
    # plot toggles
    make_overlay: bool = True,
    make_separate: bool = True,
    # prevalence baseline
    show_prevalence_baseline: bool = True,
    prevalence_color: str = "#D5F713",
    prevalence_lw: float = 1.5,
    prevalence_ls: str = "--",
    # style
    clip: tuple[float, float] = (0.0, 1.0),
    shade_alpha: float = 0.22,
    linewidth: float = 1.6,
    marker: str = "o",
    markersize: float = 2.5,
    markevery: int = 1,
    figsize_overlay: tuple[float, float] = (12, 4),
    figsize_single: tuple[float, float] = (8, 4),
    font_size: int = 12,
    # cutoff line(s)
    show_cutoff_lines: bool = True,
    cutoffs: float | Sequence[float] | None = None,   # e.g. 0.7 or [0.3, 0.6, 0.8]
    cutoff_color: str = "#222222",
    cutoff_lw: float = 1.5,
    cutoff_ls: str = ":",
    cutoff_labels: bool = True,                       # whether to add legend labels
    cutoff_label_fmt: str = "Cutoff = {c:.3f}",
    x_mode: str = "index",  # "index" or "percentile"

) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Plot ranked (sorted) patient-level predicted probabilities per calibration variant.

    Parameters
    ----------
    df_pat:
        Patient-level summary DataFrame (one row per patient per variant/split; typically output of
        pooled_patient_risk_summary). Must include: ["variant","split","group_label","y", std_col, center_col].
    model:
        Optional model name to filter on (requires df_pat["model"]).
    variants:
        Sequence of variant names to plot (e.g., ["uncalib","beta"]). Required.
    colors:
        Line/shade colors for each group_label in plotting order.
    split:
        Which split to plot (default "test").
    group_label:
        Which labels to plot. None auto-detects all labels present after filtering.
    center_col:
        Patient-level probability column to sort/plot (e.g., "p_mean", "p_median", "p_max", "p_softmax", "p_q75").
        If None, attempts to auto-detect a suitable column.
    std_col:
        Column used for shading ±1 std around the center curve (default "p_total_std").
    prob_label:
        Label name used in y-axis text, i.e., "Predicted P(prob_label)".
    make_overlay, make_separate:
        Toggle producing an overlay plot (all labels on one axis) and/or separate plots (one per label).
    show_prevalence_baseline:
        If True, draw a horizontal line at prevalence computed from the filtered data (mean(y)).
    prevalence_color, prevalence_lw, prevalence_ls:
        Styling for the prevalence baseline line.
    clip:
        y-axis limits (default (0, 1)).
    shade_alpha:
        Alpha for the ±std shaded band.
    linewidth, marker, markersize, markevery:
        Line/marker style controls.
    figsize_overlay, figsize_single:
        Figure sizes for overlay and single-label plots.
    font_size:
        Base font size for labels/titles/legend.
    show_cutoff_lines:
        If True, draw one or more horizontal cutoff line(s) at probability threshold(s) provided by `cutoffs`.
        Useful for visualizing subgroup definitions such as “high-risk = p >= cutoff”.

    cutoffs:
        Probability cutoff(s) to plot as horizontal line(s). May be a single float (e.g., 0.75) or a sequence
        of floats (e.g., [0.30, 0.60, 0.80]). If None, no cutoff lines are drawn (even if show_cutoff_lines=True).

    cutoff_color, cutoff_lw, cutoff_ls:
        Styling for cutoff line(s): color, line width, and line style.

    cutoff_labels:
        If True, add cutoff line label(s) to the legend (e.g., "Cutoff = 0.750"). If False, lines are drawn
        without legend entries.

    cutoff_label_fmt:
        Format string for cutoff legend labels. Must contain "{c}" which will be replaced by the cutoff value.
        Example: "Threshold = {c:.2f}".

    x_mode:
        X-axis scaling for ranked curves.
        - "index": use within-label rank index (0..n-1).
        - "percentile": use within-label percentile rank (0..100), which makes overlays comparable when
        label sizes differ. For very small label sizes, percentile spacing can appear stretched; in that
        case the function may fall back to "index" for clearer visualization (see min_n_for_percentile).

    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        results[variant][label] -> ranked DataFrame used for plotting that (variant,label).
    """
    # -------------------------
    # Validate columns
    # -------------------------
    required_cols = {"variant", "split", "group_label", std_col, "y"}
    missing = required_cols - set(df_pat.columns)
    if missing:
        raise KeyError(f"df_pat missing required columns: {sorted(missing)}")

    if model is not None and "model" not in df_pat.columns:
        raise KeyError("You passed model=... but df_pat has no 'model' column.")

    # -------------------------
    # Auto-detect center_col (mean/median/max/softmax/quantiles)
    # -------------------------
    def _detect_center_col(cols: Sequence[str]) -> str:
        preferred = ["p_mean", "p_median", "p_max", "p_softmax"]
        for c in preferred:
            if c in cols:
                return c

        q_cols = []
        for c in cols:
            m = re.fullmatch(r"p_q(\d{1,3})", str(c))
            if m:
                q_cols.append((int(m.group(1)), c))
        if q_cols:
            q_cols.sort(key=lambda t: t[0])  # smallest by default
            return q_cols[0][1]

        raise KeyError(
            "Could not auto-detect a center column. Expected one of "
            "{'p_mean','p_median','p_max','p_softmax','p_qXX'} or pass center_col explicitly."
        )

    if center_col is None:
        center_col = _detect_center_col(df_pat.columns)

    if center_col not in df_pat.columns:
        raise KeyError(f"center_col='{center_col}' not found in df_pat columns.")

    # variants required
    if variants is None or len(list(variants)) == 0:
        raise ValueError("You must provide variants, e.g. variants=['uncalib','beta'].")

    # -------------------------
    # Filter to split/model
    # -------------------------
    d = df_pat.copy()
    d = d[d["split"] == split]
    if model is not None:
        d = d[d["model"] == model]

    if d.empty:
        avail_splits = sorted(df_pat["split"].dropna().astype(str).unique().tolist())
        msg = f"No rows after filtering split='{split}'"
        if model is not None:
            msg += f" and model='{model}'"
        msg += f". Available splits: {avail_splits}"
        raise ValueError(msg)

    available_variants = sorted(d["variant"].dropna().astype(str).unique().tolist())
    selected_variants = [str(v) for v in list(variants)]
    missing_variants = [v for v in selected_variants if v not in set(available_variants)]
    if missing_variants:
        raise KeyError(f"Requested variants not found: {missing_variants}. Available: {available_variants}")

    # labels to plot
    available_labels = sorted(d["group_label"].dropna().astype(str).unique().tolist())
    if group_label is None:
        labels_to_plot = available_labels
    elif isinstance(group_label, str):
        labels_to_plot = [group_label]
    else:
        labels_to_plot = [str(x) for x in group_label]

    if not labels_to_plot:
        raise ValueError("No group_label values to plot after filtering.")

    missing_labels = [lab for lab in labels_to_plot if str(lab) not in set(available_labels)]
    if missing_labels:
        raise KeyError(f"Requested group_label not found: {missing_labels}. Available: {available_labels}")

    d = d[d["variant"].astype(str).isin(selected_variants)].copy()

    # -------------------------
    # Pretty names
    # -------------------------
    def _center_word(cc: str) -> str:
        if cc == "p_mean":
            return "mean"
        if cc == "p_median":
            return "median"
        if cc == "p_max":
            return "max"
        if cc == "p_softmax":
            return "softmax"
        m = re.fullmatch(r"p_q(\d{1,3})", str(cc))
        if m:
            return f"q{m.group(1)}"
        return str(cc)

    center_word = _center_word(center_col)

    def _resolve_model_name(d_sub: pd.DataFrame) -> str:
        if model is not None:
            return str(model)
        if "model" not in d_sub.columns:
            return "model_unknown"
        uniq = sorted(d_sub["model"].dropna().astype(str).unique().tolist())
        if len(uniq) == 1:
            return uniq[0]
        if len(uniq) == 0:
            return "model_unknown"
        return "multiple_models"

    # -------------------------
    # Styling helpers
    # -------------------------
    def _style_axes(ax, *, xlabel: str, ylabel: str, title: str):
        ax.set_xlabel(xlabel, fontsize=font_size, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=font_size, fontweight="bold")
        ax.set_title(title, fontsize=font_size + 2, fontweight="bold")
        ax.tick_params(axis="both", labelsize=font_size)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontweight("bold")

    def _bold_legend(ax):
        leg = ax.legend(prop={"size": font_size, "weight": "bold"})
        if leg is not None and leg.get_title() is not None:
            leg.get_title().set_fontweight("bold")

    def _plot_curve(ax, x, yvals, svals, color, label):
        lo = np.clip(yvals - svals, clip[0], clip[1])
        hi = np.clip(yvals + svals, clip[0], clip[1])
        ax.plot(
            x,
            yvals,
            linewidth=linewidth,
            label=label,
            color=color,
            marker=marker,
            markersize=markersize,
            markevery=markevery,
        )
        ax.fill_between(x, lo, hi, alpha=shade_alpha, color=color)

    def _add_prevalence_line(ax, d_for_prev: pd.DataFrame):
        # prevalence computed from current filtered rows
        y = pd.to_numeric(d_for_prev["y"], errors="coerce").dropna()
        if y.empty:
            return
        prev = float(y.mean())
        ax.axhline(
            prev,
            color=prevalence_color,
            linewidth=prevalence_lw,
            linestyle=prevalence_ls,
            label=f"Prevalence (mean y) = {prev:.3f}",
        )

    def _add_cutoff_lines(ax):
        if not show_cutoff_lines or cutoffs is None:
            return

        # normalize to list[float]
        if isinstance(cutoffs, (int, float, np.floating)):
            cs = [float(cutoffs)]
        else:
            cs = [float(c) for c in cutoffs]

        # draw (optionally clipped to y-range)
        for c in cs:
            ax.axhline(
                y=c,
                color=cutoff_color,
                linewidth=cutoff_lw,
                linestyle=cutoff_ls,
                label=(cutoff_label_fmt.format(c=c) if cutoff_labels else None),
            )
    def _make_x(n: int) -> np.ndarray:
        if x_mode == "index":
            return np.arange(n, dtype=float)
        if x_mode == "percentile":
            if n <= 1:
                return np.array([0.0], dtype=float)
            return np.linspace(0.0, 100.0, n, dtype=float)
        raise ValueError(f"x_mode must be 'index' or 'percentile', got: {x_mode}")


    # -------------------------
    # Plot style
    # -------------------------
    sns.set(style="whitegrid")

    # -------------------------
    # Build plots per variant
    # -------------------------
    results: Dict[str, Dict[str, pd.DataFrame]] = {}

    for v in selected_variants:
        dv = d[d["variant"].astype(str) == v].copy()
        if dv.empty:
            continue

        model_name = _resolve_model_name(dv)

        # Prevalence is computed per-variant (after filtering), so it's consistent with what's shown.
        # (If you prefer global prevalence across selected_variants, compute once from d instead.)
        # We add it to each plot for this variant.
        # -------------------------
        # Build ranked tables per label
        # -------------------------
        per_label_tables: Dict[str, pd.DataFrame] = {}
        for lab in labels_to_plot:
            dl = dv[dv["group_label"].astype(str) == str(lab)].copy()
            if dl.empty:
                continue

            dl[center_col] = pd.to_numeric(dl[center_col], errors="coerce")
            dl[std_col] = pd.to_numeric(dl[std_col], errors="coerce")

            dl = dl.sort_values(center_col, ascending=False).reset_index(drop=True)
            per_label_tables[str(lab)] = dl

        if not per_label_tables:
            continue

        # Overlay (one chart: multiple labels)
        if make_overlay:
            fig, ax = plt.subplots(figsize=figsize_overlay)

            if show_prevalence_baseline:
                _add_prevalence_line(ax, dv)

            _add_cutoff_lines(ax)

            for j, lab in enumerate(labels_to_plot):
                key = str(lab)
                if key not in per_label_tables:
                    continue
                dl = per_label_tables[key]

                #x = np.arange(len(dl), dtype=int)
                x = _make_x(len(dl))

                yvals = dl[center_col].astype(float).to_numpy()
                svals = dl[std_col].astype(float).to_numpy()

                c = colors[j] if j < len(colors) else colors[j % len(colors)]
                _plot_curve(ax, x, yvals, svals, c, f"{lab} (n={len(dl)})")

            ax.set_ylim(*clip)

            title = f"Sorted patient-level predicted risk — {model_name} ({split} set) "
            if x_mode == "percentile":
                xlabel = f"Patients (sorted within each label by {center_word} pooled risk), percentile rank"
            else:
                xlabel = f"Patients (sorted within each label by {center_word} pooled risk)"

            #xlabel = f"Patients (sorted within each label by {center_word} pooled risk)"
            _style_axes(ax, xlabel=xlabel, ylabel=f"Predicted P({prob_label})", title=title)
            _bold_legend(ax)

            fig.tight_layout()
            plt.show()

        # Separate (one chart per label)
        if make_separate:
            for j, lab in enumerate(labels_to_plot):
                key = str(lab)
                if key not in per_label_tables:
                    continue
                dl = per_label_tables[key]

                fig, ax = plt.subplots(figsize=figsize_single)

                if show_prevalence_baseline:
                    _add_prevalence_line(ax, dv)

                _add_cutoff_lines(ax)
                
                #x = np.arange(len(dl), dtype=int)
                x = _make_x(len(dl))

                yvals = dl[center_col].astype(float).to_numpy()
                svals = dl[std_col].astype(float).to_numpy()

                c = colors[j] if j < len(colors) else colors[j % len(colors)]
                _plot_curve(ax, x, yvals, svals, c, f"{lab} (n={len(dl)})")

                ax.set_ylim(*clip)

                title = f"Sorted patient predicted risk — {model_name} ({split} set) | {lab}"
                xlabel = f"Patients ({lab}; sorted by {center_word} pooled risk)"
                _style_axes(ax, xlabel=xlabel, ylabel=f"Predicted P({prob_label})", title=title)
                _bold_legend(ax)

                fig.tight_layout()
                plt.show()

        results[str(v)] = per_label_tables

    if not results:
        raise ValueError("No plots produced. Check that your requested variants/labels exist after filtering.")

    return results



# ---------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------

def diagnostic_enrichment_threshold_sweep_by_model(
    df,
    thresholds,
    *,
    model=None,
    score_col="p_mean",
    variants="beta",
    split="test",
    meta_cols=None,
    drop_subject_ids=None,
    subject_col="idx",
    y_col="y",
    label_col="group_label",
    confidence=0.95,
    precision=0.05,
    compute_power=True,
    power_alpha=0.05,
    power_alternative="larger",
    power_endpoint="binary",
    power_method="binomial",
    skip_empty=True,
    verbose=True,
):
    """
    Run diagnostic enrichment across multiple thresholds.

    If a threshold selects zero patients for a model, skip it by default.
    """

    rows = []
    skipped = []

    for threshold in thresholds:
        try:
            out = post.diagnostic_enrichment_pipeline_by_model(
                df=df,
                threshold=float(threshold),
                model=model,
                score_col=score_col,
                variants=variants,
                split=split,
                meta_cols=meta_cols,
                drop_subject_ids=drop_subject_ids,
                subject_col=subject_col,
                y_col=y_col,
                label_col=label_col,
                confidence=confidence,
                precision=precision,
                compute_power=compute_power,
                power_alpha=power_alpha,
                power_alternative=power_alternative,
                power_endpoint=power_endpoint,
                power_method=power_method,
            )

            for model_name, block in out.items():
                row = block["planning_summary"].copy()
                row["model"] = model_name
                rows.append(row)

        except ValueError as e:
            msg = str(e)

            if skip_empty and "df_hi is empty" in msg:
                skipped.append(
                    {
                        "threshold": float(threshold),
                        "reason": msg,
                    }
                )
                continue

            raise

    if len(rows) == 0:
        raise ValueError(
            "No threshold-sweep rows were produced. "
            "All thresholds may have selected zero patients."
        )

    sweep_table = pd.concat(rows, axis=0, ignore_index=True)

    skipped_table = pd.DataFrame(skipped)

    if verbose and not skipped_table.empty:
        print("Skipped thresholds with no selected patients:")
        display(skipped_table)

    return sweep_table, skipped_table

def make_threshold_decision_table(
    sweep_table,
    *,
    threshold_col="thr_low",
    round_digits=3,
):
    """
    Reduce the full threshold sweep table to the clinically useful
    threshold-decision columns.
    """

    decision_cols = [
        # Identity
        "model",
        threshold_col,

        # Feasibility
        "n_selected",
        "pct_selected",
        "nns",

        # Enrichment quality
        "ppv",
        "fdr",
        "enrichment_factor",

        # Operating behavior
        "sensitivity",
        "specificity",
        "npv",

        # Prospective planning
        "required_selected_n",
        "implied_screened_n",
        "power",
    ]

    cols = [c for c in decision_cols if c in sweep_table.columns]

    out = sweep_table.loc[:, cols].copy()

    if threshold_col in out.columns:
        out = out.rename(columns={threshold_col: "threshold"})

    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].round(round_digits)

    out = out.sort_values(["model", "threshold"]).reset_index(drop=True)

    return out

def plot_threshold_decision_bars(
    decision_table,
    *,
    metric_groups=None,
    model_col="model",
    threshold_col="threshold",
    model_alias=None,
    model_order=None,

    # Colors
    model_palette=None,  # dict keyed by displayed model label
    model_colors=("#4C72B0", "#DD8452", "#55A868", "#8172B2"),
    bar_alpha=1.0,
    bar_edgecolor=None,
    bar_linewidth=0.0,

    # Layout
    ncols=3,
    figsize_per_panel=(4.2, 3.4),
    font_size=11,
    x_tick_rotation=0,
    bar_width=0.35,

    # Y-axis limits
    metric_ylim=None,  # optional dict: {"ppv": (0, 1), "nns": (0, 50)}

    # Value annotations
    show_values=True,
    value_decimals=2,
    value_font_size=None,
    value_offset=0.01,
    value_color="black",
    value_fontweight="bold",
    rotate_value_labels=False,
    padding=0.18,

    # Grid
    show_grid=True,
    grid_axis="y",
    grid_color="#cccccc",
    grid_linewidth=1.0,
    grid_alpha=0.35,
    grid_linestyle="-",

    # Legend
    legend_loc="upper center",
    legend_bbox_to_anchor=(0.5, 1.02),
    legend_ncol=None,
    legend_frameon=False,
):
    """
    Plot threshold-decision metrics as grouped barplots.

    One figure per metric group.
    One subplot per metric.
    x-axis = threshold.
    grouped bars = model.

    Notes
    -----
    - Group names are used only to organize figures and are not shown.
    - The model legend is shown once only, outside the first figure.
    - metric_groups supports either:

        ("ppv", "Positive predictive value")

      or:

        ("ppv", "Positive predictive value", "Proportion")

      where the second item is the subplot title and the third item is
      the y-axis label.

    Color control
    -------------
    model_palette:
        Explicit dictionary keyed by displayed model labels, e.g.
        {"Logistic regression": "#4C97E8", "XGBoost": "#EC6868"}

    model_colors:
        Sequence of fallback colors assigned to models in order.
    """

    if metric_groups is None:
        metric_groups = {
            "Enrichment quality": [
                ("ppv", "Positive predictive value", "Proportion"),
                ("fdr", "FDR / contamination among selected patients", "Proportion"),
                ("enrichment_factor", "Enrichment factor", "Fold enrichment"),
            ],
            "Operating behavior": [
                ("sensitivity", "Sensitivity", "Proportion"),
                ("specificity", "Specificity", "Proportion"),
                ("npv", "Negative predictive value", "Proportion"),
            ],
            "Recruitment feasibility and study planning": [
                ("pct_selected", "Proportion selected for enriched cohort", "Proportion"),
                ("n_selected", "Enriched cohort size", "Patients selected"),
                (
                    "nns",
                    "Patients screened to identify one enrichment-eligible patient",
                    "Patients screened",
                ),
                (
                    "required_selected_n",
                    "Estimated selected patients required for study",
                    "Selected patients",
                ),
                (
                    "implied_screened_n",
                    "Estimated total screening population required",
                    "Patients screened",
                ),
            ],
        }

    if model_alias is None:
        model_alias = {}

    if metric_ylim is None:
        metric_ylim = {}

    d = decision_table.copy()

    if model_col not in d.columns:
        raise KeyError(f"decision_table missing model_col='{model_col}'.")

    if threshold_col not in d.columns:
        raise KeyError(f"decision_table missing threshold_col='{threshold_col}'.")

    d[model_col] = d[model_col].astype(str)
    d["model_display"] = d[model_col].map(lambda x: model_alias.get(x, x))

    d[threshold_col] = pd.to_numeric(d[threshold_col], errors="coerce")
    d = d.dropna(subset=[threshold_col]).copy()

    thresholds = sorted(d[threshold_col].unique())

    if len(thresholds) == 0:
        raise ValueError("No valid threshold values found.")

    models_found = list(d["model_display"].dropna().unique())

    if len(models_found) == 0:
        raise ValueError("No valid model values found.")

    if model_order is None:
        models = models_found
    else:
        mapped_order = [model_alias.get(m, m) for m in model_order]
        models = [m for m in mapped_order if m in models_found]
        extra_models = [m for m in models_found if m not in models]
        models.extend(extra_models)

    # ------------------------------------------------------------------
    # Build model palette
    # ------------------------------------------------------------------
    if model_palette is None:
        if model_colors is None:
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            model_colors = color_cycle

        model_palette = {
            model_name: model_colors[i % len(model_colors)]
            for i, model_name in enumerate(models)
        }
    else:
        # Keep user-supplied palette, but fill missing models from model_colors.
        model_palette = dict(model_palette)

        if model_colors is None:
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            model_colors = color_cycle

        missing_models = [m for m in models if m not in model_palette]

        for i, model_name in enumerate(missing_models):
            model_palette[model_name] = model_colors[i % len(model_colors)]

    x = np.arange(len(thresholds))
    plot_outputs = {}
    legend_shown = False

    def _parse_metric_spec(metric_spec):
        if len(metric_spec) == 2:
            metric, title = metric_spec
            ylabel = title
        elif len(metric_spec) == 3:
            metric, title, ylabel = metric_spec
        else:
            raise ValueError(
                "Each metric specification must have length 2 or 3: "
                "(column, title) or (column, title, ylabel)."
            )

        return metric, title, ylabel

    def _format_bar_label(value):
        if not np.isfinite(value):
            return ""

        abs_value = abs(value)

        if abs_value >= 1000:
            return f"{value:,.0f}"
        if abs_value >= 100:
            return f"{value:.0f}"
        if abs_value >= 10:
            return f"{value:.1f}"

        return f"{value:.{value_decimals}f}"

    for group_name, metric_specs in metric_groups.items():
        parsed_specs = [_parse_metric_spec(s) for s in metric_specs]

        metrics_present = [
            (metric, title, ylabel)
            for metric, title, ylabel in parsed_specs
            if metric in d.columns
        ]

        if len(metrics_present) == 0:
            continue

        n_metrics = len(metrics_present)
        nrows = int(np.ceil(n_metrics / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
            squeeze=False,
        )

        axes = axes.ravel()

        for ax_i, (metric, metric_title, metric_ylabel) in enumerate(metrics_present):
            ax = axes[ax_i]
            panel_values = []

            for j, model_name in enumerate(models):
                sub = d[d["model_display"] == model_name].copy()

                values = []
                for threshold in thresholds:
                    val = sub.loc[sub[threshold_col] == threshold, metric]

                    if len(val) == 0:
                        values.append(np.nan)
                    else:
                        values.append(float(val.iloc[0]))

                finite_values = [v for v in values if np.isfinite(v)]
                panel_values.extend(finite_values)

                offset = (j - (len(models) - 1) / 2) * bar_width

                bars = ax.bar(
                    x + offset,
                    values,
                    width=bar_width,
                    color=model_palette.get(model_name, None),
                    alpha=bar_alpha,
                    edgecolor=bar_edgecolor,
                    linewidth=bar_linewidth,
                    label=model_name,
                )

                if show_values:
                    ann_fs = (
                        value_font_size
                        if value_font_size is not None
                        else max(8, font_size - 3)
                    )

                    for bar in bars:
                        height = bar.get_height()

                        if not np.isfinite(height):
                            continue

                        label = _format_bar_label(float(height))

                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            height + value_offset,
                            label,
                            ha="center",
                            va="bottom",
                            fontsize=ann_fs,
                            fontweight=value_fontweight,
                            color=value_color,
                            rotation=90 if rotate_value_labels else 0,
                        )

            if metric in metric_ylim and metric_ylim[metric] is not None:
                ax.set_ylim(*metric_ylim[metric])
            else:
                panel_max = max(panel_values) if len(panel_values) > 0 else 1.0

                if panel_max <= 0 or not np.isfinite(panel_max):
                    panel_max = 1.0

                y_top = panel_max * (1.0 + padding)
                ax.set_ylim(0, y_top)

            ax.set_title(
                metric_title,
                fontsize=font_size + 1,
                fontweight="bold",
                pad=font_size,
            )

            ax.set_xlabel("Threshold", fontsize=font_size, fontweight="bold")
            ax.set_ylabel(metric_ylabel, fontsize=font_size, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{t:.2f}" for t in thresholds],
                rotation=x_tick_rotation,
                fontsize=font_size,
                fontweight="bold",
            )

            ax.tick_params(axis="y", labelsize=font_size)

            for tick in ax.get_yticklabels():
                tick.set_fontweight("bold")

            if show_grid:
                ax.grid(
                    True,
                    axis=grid_axis,
                    color=grid_color,
                    linewidth=grid_linewidth,
                    alpha=grid_alpha,
                    linestyle=grid_linestyle,
                    zorder=0,
                )
            else:
                ax.grid(False)

            ax.set_axisbelow(True)

        for k in range(len(metrics_present), len(axes)):
            axes[k].axis("off")

        handles, labels = axes[0].get_legend_handles_labels()

        if not legend_shown:
            if legend_ncol is None:
                legend_ncol_use = len(labels)
            else:
                legend_ncol_use = legend_ncol

            fig.legend(
                handles,
                labels,
                loc=legend_loc,
                bbox_to_anchor=legend_bbox_to_anchor,
                ncol=legend_ncol_use,
                frameon=legend_frameon,
                prop={"size": font_size, "weight": "bold"},
            )

            fig.tight_layout(rect=[0, 0, 1, 0.92])
            legend_shown = True
        else:
            fig.tight_layout()

        plt.show()

        plot_outputs[group_name] = {
            "fig": fig,
            "axes": axes,
            "metrics": [m for m, _, _ in metrics_present],
            "model_palette": model_palette,
        }

    if len(plot_outputs) == 0:
        raise ValueError(
            "No plots were produced. None of the requested metrics were found "
            "in decision_table."
        )

    return plot_outputs



# ---------------------------------------------------------------------
# External validation pipeline helpers
# ---------------------------------------------------------------------

def _detect_models_from_table(
    table: pd.DataFrame,
    *,
    model_col: str = "model",
) -> list[str]:
    """
    Detect available model names from a table.
    """
    if model_col not in table.columns:
        raise KeyError(f"table is missing model_col={model_col!r}.")

    models = (
        table[model_col]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    if len(models) == 0:
        raise ValueError("No model names were detected.")

    return models


def _resolve_requested_models(
    *,
    requested_model=None,
    available_models: list[str],
) -> list[str]:
    """
    Resolve requested models.

    requested_model=None means use all available models.
    """
    available_models = [str(m) for m in available_models]

    if requested_model is None:
        return available_models

    if isinstance(requested_model, str):
        requested = [requested_model]
    else:
        requested = list(requested_model)

    requested = [str(m) for m in requested]

    missing = [m for m in requested if m not in available_models]

    if missing:
        raise ValueError(
            f"Requested model(s) not found: {missing}. "
            f"Available models: {available_models}"
        )

    return requested


def _concat_model_output_tables(
    model_outputs: dict,
    *,
    output_key: str,
) -> pd.DataFrame:
    """
    Concatenate one model-output table across all models.
    """
    rows = []

    for model_name, block in model_outputs.items():
        if output_key not in block:
            continue

        table = block[output_key].copy()

        if "model" not in table.columns:
            table["model"] = model_name

        rows.append(table)

    if len(rows) == 0:
        raise ValueError(f"No tables found for output_key={output_key!r}.")

    return pd.concat(rows, axis=0, ignore_index=True)


def _extract_model_output_tables(
    model_outputs: dict,
    *,
    output_key: str,
) -> dict[str, pd.DataFrame]:
    """
    Extract one table per model from model_outputs.
    """
    out = {}

    for model_name, block in model_outputs.items():
        if output_key in block:
            out[model_name] = block[output_key].copy()

    return out


def _enabled_kwargs(kwargs: dict | None) -> tuple[bool, dict]:
    """
    Split an optional kwargs dictionary into enabled flag + call kwargs.
    """
    if kwargs is None:
        return False, {}

    kwargs = dict(kwargs)
    enabled = bool(kwargs.pop("enabled", False))

    return enabled, kwargs


def _merge_kwargs(defaults: dict, user_kwargs: Optional[dict]) -> dict:
    """
    Merge user kwargs into defaults without modifying either input.
    """
    out = dict(defaults)
    if user_kwargs is not None:
        out.update(user_kwargs)
    return out


def _format_pipeline_message(
    status: str,
    step: str,
    detail: Optional[str] = None,
) -> str:
    """
    Format one persistent progress-tracker line.
    """
    icons = {
        "running": ">>",
        "done": "[OK]",
        "skipped": "[--]",
        "failed": "[FAIL]",
    }

    icon = icons.get(status, status)
    msg = f"{icon} {step}"

    if detail:
        msg += f" -> {detail}"

    return msg


def _summarize_pipeline_output(obj: Any) -> str:
    """
    Create a compact output summary for the progress tracker.
    """
    if isinstance(obj, pd.DataFrame):
        return f"DataFrame shape={obj.shape}"

    if isinstance(obj, dict):
        return f"dict keys={list(obj.keys())}"

    if isinstance(obj, list):
        return f"list len={len(obj)}"

    if hasattr(obj, "shape"):
        return f"{type(obj).__name__} shape={obj.shape}"

    return type(obj).__name__


def _run_pipeline_step(
    progress_rows: list[dict[str, Any]],
    *,
    step: str,
    func,
    progress_enabled: bool = True,
    show_output_shapes: bool = True,
):
    """
    Run one pipeline step, print persistent progress, and record a progress row.
    """
    if progress_enabled:
        print(_format_pipeline_message("running", step))

    start = time.perf_counter()

    try:
        result = func()
        elapsed = time.perf_counter() - start

        detail = (
            _summarize_pipeline_output(result)
            if show_output_shapes
            else None
        )

        if progress_enabled:
            print(_format_pipeline_message("done", step, detail))

        progress_rows.append(
            {
                "step": step,
                "status": "done",
                "detail": detail,
                "elapsed_seconds": elapsed,
                "error": None,
            }
        )

        return result

    except Exception as exc:
        elapsed = time.perf_counter() - start

        if progress_enabled:
            print(_format_pipeline_message("failed", step, str(exc)))

        progress_rows.append(
            {
                "step": step,
                "status": "failed",
                "detail": None,
                "elapsed_seconds": elapsed,
                "error": str(exc),
            }
        )

        raise


def make_bundle_df(
    bundle: Mapping[str, Any],
    *,
    X_key: str = "X_scaled",
    y_key: str = "y",
    feature_names_key: str = "feature_names",
    y_col: str = "target",
) -> pd.DataFrame:
    """
    Convert a train/validation bundle into a single DataFrame containing X and y.

    Parameters
    ----------
    bundle : mapping
        Bundle dictionary containing feature matrix, target vector, and feature names.

    X_key : str, default "X_scaled"
        Key in `bundle` containing the feature matrix.

    y_key : str, default "y"
        Key in `bundle` containing the target vector.

    feature_names_key : str, default "feature_names"
        Key in `bundle` containing feature names aligned with columns of
        `bundle[X_key]`.

    y_col : str, default "target"
        Name to assign to the target column in the returned DataFrame.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing feature columns followed by the target column.
    """
    required = [X_key, y_key, feature_names_key]
    missing = [k for k in required if k not in bundle]

    if missing:
        raise KeyError(f"Bundle is missing required keys: {missing}")

    X = np.asarray(bundle[X_key])
    y = np.asarray(bundle[y_key])
    feature_names = list(bundle[feature_names_key])

    if X.ndim != 2:
        raise ValueError(f"bundle[{X_key!r}] must be 2D. Got shape={X.shape}.")

    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature-name mismatch: bundle[{X_key!r}].shape[1]={X.shape[1]} "
            f"but len(bundle[{feature_names_key!r}])={len(feature_names)}."
        )

    if len(y) != X.shape[0]:
        raise ValueError(
            f"Target-length mismatch: len(bundle[{y_key!r}])={len(y)} "
            f"but bundle[{X_key!r}].shape[0]={X.shape[0]}."
        )

    X_df = pd.DataFrame(X, columns=feature_names)
    y_df = pd.DataFrame(y, columns=[y_col])

    return pd.concat([X_df, y_df], axis=1)


def make_model_data_dict_from_results(
    all_results: Mapping[str, Sequence[Mapping[str, Any]]],
    external_df: pd.DataFrame,
    *,
    y_col: Optional[str] = "target",
    results_feature_names_key: str = "feature_names_used",
    feature_strategy: str = "union",
    strict_features: bool = True,
    include_private_keys: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Build model-specific external DataFrames from `all_results`.

    For each model in `all_results`, this function creates one validation/external
    DataFrame containing the feature columns required by that model plus the
    optional target column.

    Parameters
    ----------
    all_results : mapping
        Nested results dictionary:
            all_results[model_name] = list of fold/trial result dictionaries

    external_df : pandas.DataFrame
        External/validation DataFrame containing candidate feature columns and
        optionally the target column.

    y_col : str or None, default "target"
        Target column to include in each model-specific DataFrame.
        Use None for unlabeled external data.

    results_feature_names_key : str, default "feature_names_used"
        Key inside each fold/trial record containing selected feature names.

    feature_strategy : {"union", "first"}, default "union"
        Strategy for deciding which features to include for each model.

        "union":
            Include the union of all features used across all fold/trial records
            for that model. This is safest when selected features may vary across
            folds/trials.

        "first":
            Include only the features from the first fold/trial record for that
            model. This assumes feature sets are identical across records.

    strict_features : bool, default True
        If True, raise an error if a fold/trial record is missing
        `results_feature_names_key`, or if required feature columns are missing
        from `external_df`.

        If False, records missing `results_feature_names_key` are skipped.

    include_private_keys : bool, default False
        If False, ignore top-level keys beginning with "_", such as summary keys.

    Returns
    -------
    model_data_dict : dict
        Dictionary mapping:
            model_name -> model-specific external DataFrame
    """
    if not isinstance(external_df, pd.DataFrame):
        raise TypeError("external_df must be a pandas DataFrame.")

    if feature_strategy not in {"union", "first"}:
        raise ValueError("feature_strategy must be either 'union' or 'first'.")

    if y_col is not None and y_col not in external_df.columns:
        raise KeyError(f"external_df is missing y_col={y_col!r}.")

    model_data_dict: Dict[str, pd.DataFrame] = {}

    for model_name, records in all_results.items():
        if not include_private_keys and str(model_name).startswith("_"):
            continue

        if not isinstance(records, Sequence) or isinstance(records, (str, bytes, dict)):
            continue

        if len(records) == 0:
            continue

        if feature_strategy == "first":
            first_record = records[0]

            if results_feature_names_key not in first_record:
                if strict_features:
                    raise KeyError(
                        f"{model_name} first record is missing "
                        f"{results_feature_names_key!r}."
                    )
                else:
                    continue

            feature_names = list(first_record[results_feature_names_key])

        else:
            feature_names_seen = []

            for record_idx, record in enumerate(records):
                if results_feature_names_key not in record:
                    if strict_features:
                        raise KeyError(
                            f"{model_name} record index {record_idx} is missing "
                            f"{results_feature_names_key!r}."
                        )
                    else:
                        continue

                for feature_name in list(record[results_feature_names_key]):
                    if feature_name not in feature_names_seen:
                        feature_names_seen.append(feature_name)

            feature_names = feature_names_seen

        missing_features = [
            feature for feature in feature_names
            if feature not in external_df.columns
        ]

        if missing_features:
            raise KeyError(
                f"external_df is missing required features for {model_name}: "
                f"{missing_features}"
            )

        cols = list(feature_names)

        if y_col is not None:
            cols = cols + [y_col]

        model_data_dict[str(model_name)] = external_df.loc[:, cols].copy()

    if not model_data_dict:
        raise ValueError(
            "No model-specific external DataFrames were created. "
            "Check all_results structure and feature-selection settings."
        )

    return model_data_dict


# ---------------------------------------------------------------------
# Enrichment and explanation pipeline helpers
# ---------------------------------------------------------------------

def build_patient_enrichment_table(
    df_pred: pd.DataFrame,
    *,
    models: Union[str, Sequence[str]],
    calibration: str = "beta",
    score_col: str = "p_mean",
    uncertainty_col: Optional[str] = "p_std",
    cutoff: float = 0.70,
    positive_rule: str = "gt",
    borderline_margin: float = 0.05,
    model_alias: Optional[Mapping[str, str]] = None,
    sort_descending: bool = True,
    return_dict: bool = True,
    patient_idx_col: str = "patient_idx",
) -> Union[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Build model-specific patient enrichment tables from an aggregated prediction table.

    This function separates patient-selection logic from plotting. For each
    requested model, it ranks patients by a chosen score column, applies a fixed
    enrichment cutoff, and adds useful columns for interpretation and downstream
    SHAP example selection.

    Notes
    -----
    The input prediction table is expected to use `idx` as the patient identifier,
    as in `df_agg`.

    The output enrichment table renames `idx` to `patient_idx` by default because
    that is more readable for patient-level enrichment and SHAP workflows.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        Aggregated patient-level prediction table, usually `df_agg`.

        Required columns:
            - "model"
            - "calibration"
            - "idx"
            - score_col

    models : str or sequence of str
        Model or models to build enrichment tables for.

        Examples:
            models="logistic_regression"

            models=["logistic_regression", "xgboost"]

            models=["logistic_regression", "xgboost", "Ensemble model"]

    calibration : str, default "beta"
        Calibration variant to use.

    score_col : str, default "p_mean"
        Prediction score used for ranking and thresholding.

    uncertainty_col : str or None, default "p_std"
        Optional uncertainty column to carry into the enrichment table.

    cutoff : float, default 0.70
        Enrichment threshold.

    positive_rule : {"gt", "ge"}, default "gt"
        Rule used to define selected patients.

        "gt":
            selected_for_enrichment = score_col > cutoff

        "ge":
            selected_for_enrichment = score_col >= cutoff

    borderline_margin : float, default 0.05
        Margin around the cutoff used to label borderline patients.

        This does not affect selection. It only creates `selection_group`.

    model_alias : mapping or None, default None
        Optional mapping from model key to display label.

    sort_descending : bool, default True
        If True, rank patients from highest score to lowest score.

    return_dict : bool, default True
        If True, return a dictionary keyed by model name.

    patient_idx_col : str, default "patient_idx"
        Output column name for the patient identifier. The source column in
        `df_pred` is still expected to be `"idx"`.

    Returns
    -------
    enrichment_tables : dict[str, pandas.DataFrame] or pandas.DataFrame
        If return_dict=True:
            Dictionary where each key is a model name and each value is that
            model's enrichment table.

        If return_dict=False:
            Combined enrichment table for all requested models.
    """

    if not isinstance(df_pred, pd.DataFrame):
        raise TypeError("df_pred must be a pandas DataFrame.")

    if isinstance(models, str):
        model_list = [models]
    else:
        model_list = list(models)

    if len(model_list) == 0:
        raise ValueError("models must contain at least one model name.")

    if positive_rule not in {"gt", "ge"}:
        raise ValueError("positive_rule must be either 'gt' or 'ge'.")

    cutoff = float(cutoff)
    borderline_margin = float(borderline_margin)

    if not (0.0 <= cutoff <= 1.0):
        raise ValueError(f"cutoff must be in [0, 1]. Got {cutoff}.")

    if borderline_margin < 0:
        raise ValueError(
            f"borderline_margin must be >= 0. Got {borderline_margin}."
        )

    if model_alias is None:
        model_alias = {}

    required_cols = {"model", "calibration", "idx", score_col}
    missing = required_cols - set(df_pred.columns)

    if missing:
        raise KeyError(
            f"df_pred is missing required columns: {sorted(missing)}"
        )

    if uncertainty_col is not None and uncertainty_col not in df_pred.columns:
        raise KeyError(
            f"df_pred is missing uncertainty_col={uncertainty_col!r}."
        )

    available_models = sorted(df_pred["model"].astype(str).unique().tolist())
    missing_models = [m for m in model_list if m not in available_models]

    if missing_models:
        raise KeyError(
            f"Requested model(s) not found in df_pred: {missing_models}. "
            f"Available models: {available_models}"
        )

    def _build_one_model_table(model_name: str) -> pd.DataFrame:
        d = df_pred.copy()

        d["model"] = d["model"].astype(str)
        d["calibration"] = d["calibration"].astype(str)

        d = d[
            (d["model"] == str(model_name))
            & (d["calibration"] == str(calibration))
        ].copy()

        if d.empty:
            raise ValueError(
                f"No rows found for model={model_name!r} "
                f"and calibration={calibration!r}."
            )

        d["idx"] = pd.to_numeric(d["idx"], errors="coerce").astype(int)
        d[score_col] = pd.to_numeric(d[score_col], errors="coerce").astype(float)

        if d[score_col].isna().all():
            raise ValueError(
                f"All values in score_col={score_col!r} are NaN for "
                f"model={model_name!r}, calibration={calibration!r}."
            )

        if uncertainty_col is not None:
            d[uncertainty_col] = pd.to_numeric(
                d[uncertainty_col],
                errors="coerce",
            ).astype(float)

        # df_agg should already be one row per model/calibration/idx,
        # but keep this defensive de-duplication.
        d = d.drop_duplicates(["model", "calibration", "idx"], keep="first")

        # Rename patient identifier for readability in enrichment workflow.
        d = d.rename(columns={"idx": patient_idx_col})

        # Sort and rank within this model.
        d = d.sort_values(
            score_col,
            ascending=not sort_descending,
            kind="mergesort",
        ).reset_index(drop=True)

        d["patient_rank"] = np.arange(1, len(d) + 1, dtype=int)

        # Selection rule.
        if positive_rule == "gt":
            selected = d[score_col] > cutoff
        else:
            selected = d[score_col] >= cutoff

        d["cutoff"] = cutoff
        d["distance_to_cutoff"] = d[score_col] - cutoff
        d["abs_distance_to_cutoff"] = d["distance_to_cutoff"].abs()
        d["selected_for_enrichment"] = selected.astype(bool)

        # Borderline grouping.
        upper = cutoff + borderline_margin
        lower = cutoff - borderline_margin

        if positive_rule == "gt":
            selected_borderline = (d[score_col] > cutoff) & (d[score_col] <= upper)
            not_selected_borderline = (d[score_col] <= cutoff) & (d[score_col] >= lower)
        else:
            selected_borderline = (d[score_col] >= cutoff) & (d[score_col] <= upper)
            not_selected_borderline = (d[score_col] < cutoff) & (d[score_col] >= lower)

        selection_group = np.where(
            d["selected_for_enrichment"],
            "selected_high_confidence",
            "not_selected_low_risk",
        )

        selection_group = np.where(
            selected_borderline,
            "selected_borderline",
            selection_group,
        )

        selection_group = np.where(
            not_selected_borderline,
            "not_selected_borderline",
            selection_group,
        )

        d["selection_group"] = pd.Categorical(
            selection_group,
            categories=[
                "selected_high_confidence",
                "selected_borderline",
                "not_selected_borderline",
                "not_selected_low_risk",
            ],
            ordered=True,
        )

        # Add model display label.
        d["model_label"] = d["model"].map(
            lambda m: model_alias.get(str(m), str(m))
        )

        # Add model-level selection summary columns.
        n_total = int(len(d))
        n_selected = int(d["selected_for_enrichment"].sum())
        n_below = int(n_total - n_selected)
        selection_rate = float(n_selected / n_total) if n_total > 0 else np.nan

        d["n_selected_for_enrichment"] = n_selected
        d["n_below_threshold"] = n_below
        d["selection_rate"] = selection_rate

        # Put most useful columns first, but preserve any extra columns after.
        preferred_cols = [
            "model",
            "model_label",
            "calibration",
            patient_idx_col,
            "patient_rank",
            "y",
            "y_label",
            score_col,
        ]

        if uncertainty_col is not None:
            preferred_cols.append(uncertainty_col)

        preferred_cols += [
            "p_median",
            "p_min",
            "p_max",
            "cutoff",
            "distance_to_cutoff",
            "abs_distance_to_cutoff",
            "selected_for_enrichment",
            "selection_group",
            "n_selected_for_enrichment",
            "n_below_threshold",
            "selection_rate",
            "split",
            "prevalence_used",
            "n_preds",
        ]

        preferred_cols = [c for c in preferred_cols if c in d.columns]
        remaining_cols = [c for c in d.columns if c not in preferred_cols]

        d = d[preferred_cols + remaining_cols].reset_index(drop=True)

        return d

    enrichment_tables = {
        str(model_name): _build_one_model_table(str(model_name))
        for model_name in model_list
    }

    if return_dict:
        return enrichment_tables

    return pd.concat(
        enrichment_tables.values(),
        axis=0,
        ignore_index=True,
    )

def select_enrichment_patients_for_explanation(
    enrichment_tables: Mapping[str, pd.DataFrame],
    *,
    manual_patient_ids: Optional[
        Union[Sequence[int], Mapping[str, Sequence[int]]]
    ] = None,
    representative_types: Optional[Sequence[str]] = None,
    n_per_type: Union[int, Mapping[str, int]] = 1,
    patient_idx_col: str = "patient_idx",
    score_col: str = "p_mean",
    selected_col: str = "selected_for_enrichment",
    selection_group_col: str = "selection_group",
    distance_col: str = "abs_distance_to_cutoff",
    allow_missing: bool = True,
    return_log: bool = True,
) -> Union[
    dict[str, pd.DataFrame],
    tuple[dict[str, pd.DataFrame], pd.DataFrame],
]:
    """
    Select patients from enrichment tables for downstream SHAP explanation.

    This function supports two workflows:

    1. Manual selection
       If `manual_patient_ids` is provided, the function selects those patient
       identifiers from each model-specific enrichment table.

    2. Automatic representative selection
       If `manual_patient_ids=None`, the function selects representative patient
       examples from each model-specific enrichment table.

       Default representative types:
           - "top_selected"
           - "borderline_selected"
           - "borderline_not_selected"
           - "lowest_not_selected"

    Parameters
    ----------
    enrichment_tables : mapping
        Dictionary of enrichment tables, usually from:

            build_patient_enrichment_table(...)

        Expected structure:
            enrichment_tables[model_name] = model-specific enrichment DataFrame

    manual_patient_ids : sequence[int] or mapping[str, sequence[int]] or None, default None
        Manual patient selection.

        If None:
            Use automatic representative selection.

        If a sequence:
            Use the same patient IDs for every model.

            Example:
                manual_patient_ids=[63, 137]

        If a mapping:
            Use model-specific patient IDs.

            Example:
                manual_patient_ids={
                    "logistic_regression": [63, 137],
                    "xgboost": [12, 88],
                    "Ensemble model": [63, 137],
                }

    representative_types : sequence[str] or None, default None
        Representative examples to select in automatic mode.

        If None, defaults to:
            [
                "top_selected",
                "borderline_selected",
                "borderline_not_selected",
                "lowest_not_selected",
            ]

        Supported values:
            "top_selected"
                Highest-scoring selected patient(s).

            "borderline_selected"
                Selected patient(s) closest to the cutoff among patients labeled
                "selected_borderline".

            "borderline_not_selected"
                Not-selected patient(s) closest to the cutoff among patients labeled
                "not_selected_borderline".

            "lowest_not_selected"
                Lowest-scoring not-selected patient(s).

    n_per_type : int or mapping[str, int], default 1
        Number of patients to select per representative type.

        If int:
            Use the same number for every representative type.

            Example:
                n_per_type=5

        If mapping:
            Use type-specific counts.

            Example:
                n_per_type={
                    "top_selected": 5,
                    "borderline_selected": 5,
                    "borderline_not_selected": 5,
                    "lowest_not_selected": 5,
                }

        If fewer patients are available than requested, the function returns the
        available patients and records the shortfall in the selection log.

    patient_idx_col : str, default "patient_idx"
        Column containing the patient identifier.

    score_col : str, default "p_mean"
        Score column used for ranking patients.

    selected_col : str, default "selected_for_enrichment"
        Boolean column indicating whether a patient was selected.

    selection_group_col : str, default "selection_group"
        Column containing enrichment group labels.

    distance_col : str, default "abs_distance_to_cutoff"
        Column containing absolute distance from the enrichment cutoff.

    allow_missing : bool, default True
        If True, skip missing patient IDs or unavailable representative types
        and record them in the log.

        If False, raise an error when a requested patient or representative type
        cannot be found.

    return_log : bool, default True
        If True, return:
            selected_patients, selection_log

        If False, return only:
            selected_patients

    Returns
    -------
    selected_patients : dict[str, pandas.DataFrame]
        Dictionary keyed by model name.

        Each value is a DataFrame containing selected patients for explanation.
        The DataFrame includes a `selection_reason` column.

    selection_log : pandas.DataFrame, optional
        Returned only when `return_log=True`.

        One row per attempted selection with:
            - model
            - selection_reason
            - patient_idx
            - requested_n
            - selected_n
            - status
            - detail
    """

    if not isinstance(enrichment_tables, Mapping):
        raise TypeError("enrichment_tables must be a mapping of model -> DataFrame.")

    if representative_types is None:
        representative_types = [
            "top_selected",
            "borderline_selected",
            "borderline_not_selected",
            "lowest_not_selected",
        ]

    representative_types = list(representative_types)

    valid_representative_types = {
        "top_selected",
        "borderline_selected",
        "borderline_not_selected",
        "lowest_not_selected",
    }

    invalid_types = [
        x for x in representative_types
        if x not in valid_representative_types
    ]

    if invalid_types:
        raise ValueError(
            f"Unsupported representative_types: {invalid_types}. "
            f"Supported: {sorted(valid_representative_types)}"
        )

    def _get_n_for_type(selection_reason: str) -> int:
        if isinstance(n_per_type, Mapping):
            n = int(n_per_type.get(selection_reason, 1))
        else:
            n = int(n_per_type)

        if n < 0:
            raise ValueError("n_per_type values must be >= 0.")

        return n

    selected_patients: dict[str, pd.DataFrame] = {}
    log_rows: list[dict[str, Any]] = []

    def _log(
        *,
        model: str,
        selection_reason: str,
        patient_idx: Any = np.nan,
        requested_n: Any = np.nan,
        selected_n: Any = np.nan,
        status: str,
        detail: str,
    ) -> None:
        log_rows.append(
            {
                "model": model,
                "selection_reason": selection_reason,
                "patient_idx": patient_idx,
                "requested_n": requested_n,
                "selected_n": selected_n,
                "status": status,
                "detail": detail,
            }
        )

    def _handle_missing(
        *,
        model: str,
        selection_reason: str,
        patient_idx: Any = np.nan,
        requested_n: Any = np.nan,
        selected_n: Any = 0,
        detail: str,
    ) -> None:
        _log(
            model=model,
            selection_reason=selection_reason,
            patient_idx=patient_idx,
            requested_n=requested_n,
            selected_n=selected_n,
            status="missing",
            detail=detail,
        )

        if not allow_missing:
            raise ValueError(
                f"Missing selection for model={model!r}, "
                f"selection_reason={selection_reason!r}, "
                f"patient_idx={patient_idx!r}: {detail}"
            )

    def _get_manual_ids_for_model(model_name: str) -> Optional[list[int]]:
        if manual_patient_ids is None:
            return None

        if isinstance(manual_patient_ids, Mapping):
            if model_name not in manual_patient_ids:
                return []
            return list(manual_patient_ids[model_name])

        return list(manual_patient_ids)

    def _select_top_selected(d: pd.DataFrame, n: int) -> pd.DataFrame:
        sub = d[d[selected_col].astype(bool)].copy()
        if sub.empty or n == 0:
            return sub.iloc[0:0]

        return (
            sub.sort_values(score_col, ascending=False, kind="mergesort")
            .head(n)
        )

    def _select_borderline_selected(d: pd.DataFrame, n: int) -> pd.DataFrame:
        sub = d[d[selection_group_col].astype(str).eq("selected_borderline")].copy()
        if sub.empty or n == 0:
            return sub.iloc[0:0]

        return (
            sub.sort_values(distance_col, ascending=True, kind="mergesort")
            .head(n)
        )

    def _select_borderline_not_selected(d: pd.DataFrame, n: int) -> pd.DataFrame:
        sub = d[d[selection_group_col].astype(str).eq("not_selected_borderline")].copy()
        if sub.empty or n == 0:
            return sub.iloc[0:0]

        return (
            sub.sort_values(distance_col, ascending=True, kind="mergesort")
            .head(n)
        )

    def _select_lowest_not_selected(d: pd.DataFrame, n: int) -> pd.DataFrame:
        sub = d[~d[selected_col].astype(bool)].copy()
        if sub.empty or n == 0:
            return sub.iloc[0:0]

        return (
            sub.sort_values(score_col, ascending=True, kind="mergesort")
            .head(n)
        )

    selector_map = {
        "top_selected": _select_top_selected,
        "borderline_selected": _select_borderline_selected,
        "borderline_not_selected": _select_borderline_not_selected,
        "lowest_not_selected": _select_lowest_not_selected,
    }

    # ------------------------------------------------------------------
    # Main loop over models
    # ------------------------------------------------------------------
    for model_name, table in enrichment_tables.items():
        model_name = str(model_name)

        if not isinstance(table, pd.DataFrame):
            raise TypeError(
                f"enrichment_tables[{model_name!r}] must be a pandas DataFrame."
            )

        required_cols = {
            patient_idx_col,
            score_col,
            selected_col,
            selection_group_col,
            distance_col,
        }

        missing_cols = required_cols - set(table.columns)

        if missing_cols:
            raise KeyError(
                f"enrichment table for model={model_name!r} is missing "
                f"required columns: {sorted(missing_cols)}"
            )

        d = table.copy()
        d[patient_idx_col] = pd.to_numeric(
            d[patient_idx_col],
            errors="coerce",
        ).astype(int)

        d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
        d[distance_col] = pd.to_numeric(d[distance_col], errors="coerce")

        rows_to_keep: list[pd.DataFrame] = []

        # --------------------------------------------------------------
        # Mode 1: manual selection
        # --------------------------------------------------------------
        manual_ids_for_model = _get_manual_ids_for_model(model_name)

        if manual_ids_for_model is not None:
            if len(manual_ids_for_model) == 0:
                _handle_missing(
                    model=model_name,
                    selection_reason="manual",
                    requested_n=0,
                    selected_n=0,
                    detail="No manual patient IDs provided for this model.",
                )
                selected_patients[model_name] = pd.DataFrame(columns=list(d.columns) + ["selection_reason"])
                continue

            selected_manual_rows = []

            for patient_idx in manual_ids_for_model:
                patient_idx = int(patient_idx)

                sub = d[d[patient_idx_col] == patient_idx].copy()

                if sub.empty:
                    _handle_missing(
                        model=model_name,
                        selection_reason="manual",
                        patient_idx=patient_idx,
                        requested_n=1,
                        selected_n=0,
                        detail="Patient ID not found in enrichment table.",
                    )
                    continue

                row_df = sub.head(1).copy()
                row_df["selection_reason"] = "manual"
                selected_manual_rows.append(row_df)

                _log(
                    model=model_name,
                    selection_reason="manual",
                    patient_idx=patient_idx,
                    requested_n=1,
                    selected_n=1,
                    status="selected",
                    detail="Manual patient selected.",
                )

            if selected_manual_rows:
                rows_to_keep.append(pd.concat(selected_manual_rows, ignore_index=True))

        # --------------------------------------------------------------
        # Mode 2: automatic representative selection
        # --------------------------------------------------------------
        else:
            for selection_reason in representative_types:
                n_requested = _get_n_for_type(selection_reason)
                selector = selector_map[selection_reason]

                selected_df = selector(d, n_requested).copy()
                n_selected = int(len(selected_df))

                if n_selected == 0:
                    _handle_missing(
                        model=model_name,
                        selection_reason=selection_reason,
                        requested_n=n_requested,
                        selected_n=0,
                        detail="No patient available for this representative type.",
                    )
                    continue

                selected_df["selection_reason"] = selection_reason
                rows_to_keep.append(selected_df)

                status = "selected" if n_selected == n_requested else "partial"

                detail = (
                    "Representative patients selected."
                    if status == "selected"
                    else f"Requested {n_requested}, but only {n_selected} available."
                )

                _log(
                    model=model_name,
                    selection_reason=selection_reason,
                    patient_idx=selected_df[patient_idx_col].tolist(),
                    requested_n=n_requested,
                    selected_n=n_selected,
                    status=status,
                    detail=detail,
                )

        # --------------------------------------------------------------
        # Build selected DataFrame for this model
        # --------------------------------------------------------------
        if rows_to_keep:
            selected_df = pd.concat(rows_to_keep, ignore_index=True)

            # If the same patient is selected for multiple reasons,
            # keep one row and combine the reasons.
            reason_summary = (
                selected_df
                .groupby(patient_idx_col)["selection_reason"]
                .apply(lambda x: ", ".join(dict.fromkeys(map(str, x))))
                .reset_index()
            )

            selected_df = (
                selected_df
                .drop_duplicates(patient_idx_col, keep="first")
                .drop(columns=["selection_reason"])
                .merge(reason_summary, on=patient_idx_col, how="left")
            )

            # Move useful columns to the front.
            front_cols = [
                "model",
                "model_label",
                "selection_reason",
                patient_idx_col,
                "patient_rank",
                "y",
                "y_label",
                score_col,
                "p_std",
                "cutoff",
                "distance_to_cutoff",
                selected_col,
                selection_group_col,
            ]

            front_cols = [c for c in front_cols if c in selected_df.columns]
            rest_cols = [c for c in selected_df.columns if c not in front_cols]

            selected_df = selected_df[front_cols + rest_cols].reset_index(drop=True)

        else:
            selected_df = pd.DataFrame(columns=list(d.columns) + ["selection_reason"])

        selected_patients[model_name] = selected_df

    selection_log = pd.DataFrame(log_rows)

    if return_log:
        return selected_patients, selection_log

    return selected_patients

def build_selected_patient_model_comparison_table(
    enrichment_tables: Mapping[str, pd.DataFrame],
    selected_patients: Mapping[str, pd.DataFrame],
    *,
    reference_model: str = "Ensemble model",
    models: Optional[Sequence[str]] = None,
    patient_idx_col: str = "patient_idx",
    score_col: str = "p_mean",
    uncertainty_col: Optional[str] = "p_std",
    return_format: str = "long",
) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Compare model predictions for patients selected from a reference model.

    This function is useful when patient selection is based on one model,
    such as an ensemble, but you want to inspect what each component model
    predicted for those same patients.

    Parameters
    ----------
    enrichment_tables : mapping
        Dictionary of model-specific enrichment tables from:

            build_patient_enrichment_table(...)

        Expected structure:
            enrichment_tables[model_name] = enrichment_table

    selected_patients : mapping
        Dictionary of selected patient tables from:

            select_enrichment_patients_for_explanation(...)

        Expected structure:
            selected_patients[model_name] = selected_patient_table

        The `reference_model` entry is used to define which patients are compared.

    reference_model : str, default "Ensemble model"
        Model whose selected patients define the comparison cohort.

        Example:
            If reference_model="Ensemble model", then the function takes
            selected_patients["Ensemble model"]["patient_idx"] and pulls those
            same patients from each model in `models`.

    models : sequence of str or None, default None
        Models to compare.

        If None, compare all models present in `enrichment_tables`.

        Example:
            models=["logistic_regression", "xgboost", "Ensemble model"]

    patient_idx_col : str, default "patient_idx"
        Patient identifier column.

    score_col : str, default "p_mean"
        Prediction score column to compare.

    uncertainty_col : str or None, default "p_std"
        Optional uncertainty column to include.

    return_format : {"long", "wide", "both"}, default "long"
        Format of the returned comparison table.

        "long":
            One row per reference patient x compared model.

        "wide":
            One row per reference patient, with model-specific columns.

        "both":
            Return {"long": long_df, "wide": wide_df}.

    Returns
    -------
    comparison : pandas.DataFrame or dict[str, pandas.DataFrame]
        Long, wide, or both comparison formats.

    Long-format columns include:
        - reference_model
        - reference_selection_reason
        - patient_idx
        - compared_model
        - compared_model_label
        - patient_rank
        - y
        - y_label
        - p_mean
        - p_std
        - selected_for_enrichment
        - selection_group
        - distance_to_cutoff

    Notes
    -----
    This function does not choose patients by itself. It compares model outputs
    for patients already chosen by `select_enrichment_patients_for_explanation`.
    """

    if return_format not in {"long", "wide", "both"}:
        raise ValueError("return_format must be one of {'long', 'wide', 'both'}.")

    if reference_model not in selected_patients:
        raise KeyError(
            f"reference_model={reference_model!r} not found in selected_patients. "
            f"Available: {list(selected_patients.keys())}"
        )

    if reference_model not in enrichment_tables:
        raise KeyError(
            f"reference_model={reference_model!r} not found in enrichment_tables. "
            f"Available: {list(enrichment_tables.keys())}"
        )

    ref_selected = selected_patients[reference_model].copy()

    if not isinstance(ref_selected, pd.DataFrame):
        raise TypeError(
            f"selected_patients[{reference_model!r}] must be a pandas DataFrame."
        )

    if ref_selected.empty:
        raise ValueError(
            f"selected_patients[{reference_model!r}] is empty. "
            "No patients are available for comparison."
        )

    required_ref_cols = {patient_idx_col, "selection_reason"}

    missing_ref_cols = required_ref_cols - set(ref_selected.columns)
    if missing_ref_cols:
        raise KeyError(
            f"selected_patients[{reference_model!r}] is missing required columns: "
            f"{sorted(missing_ref_cols)}"
        )

    if models is None:
        model_list = list(enrichment_tables.keys())
    else:
        model_list = list(models)

    if len(model_list) == 0:
        raise ValueError("models must contain at least one model name.")

    missing_models = [m for m in model_list if m not in enrichment_tables]
    if missing_models:
        raise KeyError(
            f"Requested model(s) not found in enrichment_tables: {missing_models}. "
            f"Available: {list(enrichment_tables.keys())}"
        )

    # Keep selected reference patients in the order provided by selected_patients.
    ref_selected[patient_idx_col] = pd.to_numeric(
        ref_selected[patient_idx_col],
        errors="coerce",
    ).astype(int)

    reference_patients = (
        ref_selected[[patient_idx_col, "selection_reason"]]
        .drop_duplicates(patient_idx_col, keep="first")
        .copy()
    )

    reference_patients["reference_order"] = np.arange(
        1,
        len(reference_patients) + 1,
        dtype=int,
    )

    reference_patients = reference_patients.rename(
        columns={"selection_reason": "reference_selection_reason"}
    )

    long_rows = []

    for compared_model in model_list:
        table = enrichment_tables[compared_model].copy()

        if not isinstance(table, pd.DataFrame):
            raise TypeError(
                f"enrichment_tables[{compared_model!r}] must be a pandas DataFrame."
            )

        required_cols = {
            patient_idx_col,
            score_col,
            "selected_for_enrichment",
            "selection_group",
        }

        missing_cols = required_cols - set(table.columns)
        if missing_cols:
            raise KeyError(
                f"enrichment table for model={compared_model!r} is missing "
                f"required columns: {sorted(missing_cols)}"
            )

        table[patient_idx_col] = pd.to_numeric(
            table[patient_idx_col],
            errors="coerce",
        ).astype(int)

        table = table.drop_duplicates(patient_idx_col, keep="first")

        merged = reference_patients.merge(
            table,
            on=patient_idx_col,
            how="left",
            validate="one_to_one",
        )

        missing_patient_mask = merged[score_col].isna()
        if missing_patient_mask.any():
            missing_ids = merged.loc[
                missing_patient_mask,
                patient_idx_col,
            ].tolist()

            raise KeyError(
                f"Compared model={compared_model!r} is missing selected "
                f"reference patient(s): {missing_ids}"
            )

        merged["reference_model"] = reference_model
        merged["compared_model"] = str(compared_model)

        if "model_label" in merged.columns:
            merged["compared_model_label"] = merged["model_label"].astype(str)
        else:
            merged["compared_model_label"] = str(compared_model)

        keep_cols = [
            "reference_model",
            "reference_selection_reason",
            "reference_order",
            patient_idx_col,
            "compared_model",
            "compared_model_label",
            "patient_rank",
            "y",
            "y_label",
            score_col,
        ]

        if uncertainty_col is not None and uncertainty_col in merged.columns:
            keep_cols.append(uncertainty_col)

        keep_cols += [
            "p_median",
            "p_min",
            "p_max",
            "cutoff",
            "distance_to_cutoff",
            "abs_distance_to_cutoff",
            "selected_for_enrichment",
            "selection_group",
            "n_selected_for_enrichment",
            "n_below_threshold",
            "selection_rate",
            "split",
            "prevalence_used",
            "n_preds",
        ]

        keep_cols = [c for c in keep_cols if c in merged.columns]

        long_rows.append(merged[keep_cols].copy())

    long_df = pd.concat(long_rows, axis=0, ignore_index=True)

    long_df = long_df.sort_values(
        ["reference_order", "compared_model"],
        kind="mergesort",
    ).reset_index(drop=True)

    if return_format == "long":
        return long_df

    # ------------------------------------------------------------------
    # Build wide format
    # ------------------------------------------------------------------
    id_cols = [
        "reference_model",
        "reference_selection_reason",
        "reference_order",
        patient_idx_col,
    ]

    patient_context_cols = [
        c for c in ["y", "y_label", "split"]
        if c in long_df.columns
    ]

    context_df = (
        long_df[id_cols + patient_context_cols]
        .drop_duplicates(patient_idx_col, keep="first")
        .sort_values("reference_order", kind="mergesort")
        .reset_index(drop=True)
    )

    value_cols = [
        score_col,
        "selected_for_enrichment",
        "selection_group",
        "patient_rank",
        "distance_to_cutoff",
    ]

    if uncertainty_col is not None:
        value_cols.append(uncertainty_col)

    value_cols = [c for c in value_cols if c in long_df.columns]

    wide_parts = [context_df]

    for value_col in value_cols:
        pivot = (
            long_df
            .pivot_table(
                index=id_cols,
                columns="compared_model",
                values=value_col,
                aggfunc="first",
                observed=False,
            )
        )

        pivot.columns = [
            f"{model}__{value_col}"
            for model in pivot.columns
        ]

        pivot = pivot.reset_index()

        wide_parts.append(pivot)

    wide_df = wide_parts[0]

    for part in wide_parts[1:]:
        wide_df = wide_df.merge(
            part,
            on=id_cols,
            how="left",
            validate="one_to_one",
        )

    wide_df = wide_df.sort_values(
        "reference_order",
        kind="mergesort",
    ).reset_index(drop=True)

    if return_format == "wide":
        return wide_df

    return {
        "long": long_df,
        "wide": wide_df,
    }

def build_cohort_shap_summary_table(
    all_results,
    enrichment_tables: Mapping[str, pd.DataFrame],
    *,
    reference_selection_model: str = "Ensemble model",
    shap_models: Optional[Union[str, Sequence[str]]] = None,
    summary_key: str = "external_shap_summary",
    patient_idx_col: str = "patient_idx",
    selected_col: str = "selected_for_enrichment",
    selection_group_col: str = "selection_group",
    score_col: str = "p_mean",
    balance_method: Optional[str] = "downsample_larger_group",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
    min_group_size: int = 2,
) -> pd.DataFrame:
    """
    Build a cohort-level SHAP summary comparing selected vs not-selected patients.

    This function summarizes SHAP values across cohorts defined by an enrichment
    table. For example, if `reference_selection_model="Ensemble model"`, then
    the selected/not-selected groups come from the Ensemble model's enrichment
    decisions, while SHAP values can be summarized for component models such as
    logistic regression and XGBoost.

    This is designed to complement local SHAP waterfall plots.

    Local waterfall question:
        Why was this patient selected?

    Cohort SHAP summary question:
        Across the selected enrichment cohort, which features consistently
        pushed predictions upward, and how does that compare with not-selected
        patients?

    Parameters
    ----------
    all_results : dict
        Nested results dictionary containing external SHAP summaries.

        Expected:
            all_results[summary_key][model_name]

        Each model summary should contain:
            - "idx"
            - "feature_names"
            - "shap_values_mean"
            - "base_values_mean"
            - "data"

        Optional:
            - "predictions_mean"

    enrichment_tables : mapping[str, pandas.DataFrame]
        Dictionary of enrichment tables from build_patient_enrichment_table(...).

        The table for `reference_selection_model` defines selected vs not-selected
        patients.

    reference_selection_model : str, default "Ensemble model"
        Model whose enrichment table defines the selected/not-selected groups.

    shap_models : str, sequence of str, or None, default None
        Model(s) whose SHAP summaries should be analyzed.

        If None, all models present under `all_results[summary_key]` are used.

    summary_key : str, default "external_shap_summary"
        Top-level key in `all_results` containing mean SHAP summaries.

    patient_idx_col : str, default "patient_idx"
        Patient identifier column in the enrichment table.

    selected_col : str, default "selected_for_enrichment"
        Boolean column defining selected vs not-selected patients.

    selection_group_col : str, default "selection_group"
        Optional enrichment group column. Used only for cohort counts if present.

    score_col : str, default "p_mean"
        Optional prediction score column from the enrichment table.

    balance_method : {None, "downsample_not_selected", "downsample_larger_group"}, default "downsample_larger_group"
        Optional bootstrap strategy to address selected/not-selected group-size
        imbalance.

    n_bootstrap : int, default 1000
        Number of bootstrap/downsampling iterations.

    ci : float, default 0.95
        Confidence interval width for bootstrap summaries.

    random_state : int, default 42
        Random seed for bootstrap/downsampling.

    min_group_size : int, default 2
        Minimum selected and not-selected group size required to compute
        bootstrap intervals.

    Returns
    -------
    summary_df : pandas.DataFrame
        One row per SHAP model and feature.

        Important columns include:
            - model
            - feature
            - n_selected
            - n_not_selected

            SHAP contribution columns:
            - mean_shap_selected
            - mean_shap_not_selected
            - sem_shap_selected
            - sem_shap_not_selected
            - delta_mean_shap

            Probability endpoint columns:
            - mean_endpoint_selected
            - mean_endpoint_not_selected
            - sem_endpoint_selected
            - sem_endpoint_not_selected

            Baseline / prediction columns:
            - mean_base_value_selected
            - mean_base_value_not_selected
            - mean_prediction_selected
            - mean_prediction_not_selected

            Balanced comparison columns:
            - balanced_delta_mean_shap
            - balanced_delta_ci_low
            - balanced_delta_ci_high

    Notes
    -----
    This is an explanation audit, not a formal hypothesis test.

    The endpoint columns are feature-specific:

        endpoint = base probability + feature SHAP contribution

    These are useful when plotting cohort SHAP bars on the predicted-probability
    scale, because the error bars should represent uncertainty around the
    plotted probability endpoint.
    """

    # ------------------------------------------------------------------
    # Validate reference selection table
    # ------------------------------------------------------------------
    if summary_key not in all_results:
        raise KeyError(
            f"Missing all_results[{summary_key!r}]. "
            "Run add_external_shap_summary_to_results first."
        )

    if reference_selection_model not in enrichment_tables:
        raise KeyError(
            f"reference_selection_model={reference_selection_model!r} not found "
            f"in enrichment_tables. Available: {list(enrichment_tables.keys())}"
        )

    ref_table = enrichment_tables[reference_selection_model].copy()

    required_ref_cols = {patient_idx_col, selected_col}
    missing_ref_cols = required_ref_cols - set(ref_table.columns)

    if missing_ref_cols:
        raise KeyError(
            f"Reference enrichment table is missing required columns: "
            f"{sorted(missing_ref_cols)}"
        )

    ref_table[patient_idx_col] = pd.to_numeric(
        ref_table[patient_idx_col],
        errors="coerce",
    ).astype(int)

    ref_table[selected_col] = ref_table[selected_col].astype(bool)
    ref_table = ref_table.drop_duplicates(patient_idx_col, keep="first").copy()

    selected_ids = ref_table.loc[
        ref_table[selected_col],
        patient_idx_col,
    ].to_numpy()

    not_selected_ids = ref_table.loc[
        ~ref_table[selected_col],
        patient_idx_col,
    ].to_numpy()

    if len(selected_ids) == 0:
        raise ValueError(
            f"No selected patients found in enrichment table for "
            f"{reference_selection_model!r}."
        )

    if len(not_selected_ids) == 0:
        raise ValueError(
            f"No not-selected patients found in enrichment table for "
            f"{reference_selection_model!r}."
        )

    # ------------------------------------------------------------------
    # Resolve SHAP models
    # ------------------------------------------------------------------
    available_shap_models = list(all_results[summary_key].keys())

    if shap_models is None:
        shap_model_list = available_shap_models
    elif isinstance(shap_models, str):
        shap_model_list = [shap_models]
    else:
        shap_model_list = list(shap_models)

    missing_models = [
        m for m in shap_model_list
        if m not in all_results[summary_key]
    ]

    if missing_models:
        raise KeyError(
            f"Requested shap_models not found under all_results[{summary_key!r}]: "
            f"{missing_models}. Available: {available_shap_models}"
        )

    if balance_method not in {None, "downsample_not_selected", "downsample_larger_group"}:
        raise ValueError(
            "balance_method must be one of "
            "{None, 'downsample_not_selected', 'downsample_larger_group'}."
        )

    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be >= 0.")

    if not (0 < ci < 1):
        raise ValueError("ci must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    alpha = 1.0 - ci
    q_low = 100 * alpha / 2
    q_high = 100 * (1 - alpha / 2)

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _safe_mean(x):
        x = np.asarray(x, dtype=float)
        return float(np.nanmean(x)) if len(x) else np.nan

    def _safe_median(x):
        x = np.asarray(x, dtype=float)
        return float(np.nanmedian(x)) if len(x) else np.nan

    def _safe_std(x):
        x = np.asarray(x, dtype=float)
        return float(np.nanstd(x, ddof=1)) if len(x) > 1 else np.nan

    def _safe_sem(x):
        x = np.asarray(x, dtype=float)
        if len(x) <= 1:
            return np.nan
        return float(np.nanstd(x, ddof=1) / np.sqrt(len(x)))

    def _pct_positive(x):
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            return np.nan
        return float(np.mean(x > 0))

    def _pooled_sd(x1, x0):
        x1 = np.asarray(x1, dtype=float)
        x0 = np.asarray(x0, dtype=float)

        n1 = len(x1)
        n0 = len(x0)

        if n1 <= 1 or n0 <= 1:
            return np.nan

        s1 = np.nanvar(x1, ddof=1)
        s0 = np.nanvar(x0, ddof=1)

        denom = n1 + n0 - 2
        if denom <= 0:
            return np.nan

        pooled = np.sqrt(((n1 - 1) * s1 + (n0 - 1) * s0) / denom)

        if pooled == 0:
            return np.nan

        return float(pooled)

    def _balanced_bootstrap_delta(selected_values, not_selected_values):
        selected_values = np.asarray(selected_values, dtype=float)
        not_selected_values = np.asarray(not_selected_values, dtype=float)

        n_sel = len(selected_values)
        n_not = len(not_selected_values)

        if (
            balance_method is None
            or n_bootstrap == 0
            or n_sel < min_group_size
            or n_not < min_group_size
        ):
            return {
                "balanced_delta_mean_shap": np.nan,
                "balanced_delta_ci_low": np.nan,
                "balanced_delta_ci_high": np.nan,
                "balanced_delta_std": np.nan,
                "balanced_n_per_group": np.nan,
            }

        deltas = []

        if balance_method == "downsample_not_selected":
            n = n_sel

            if n_not < n:
                n = min(n_sel, n_not)

            for _ in range(n_bootstrap):
                sel_sample = rng.choice(selected_values, size=n, replace=True)
                not_sample = rng.choice(not_selected_values, size=n, replace=False)
                deltas.append(np.nanmean(sel_sample) - np.nanmean(not_sample))

        elif balance_method == "downsample_larger_group":
            n = min(n_sel, n_not)

            for _ in range(n_bootstrap):
                sel_replace = n_sel < n
                not_replace = n_not < n

                sel_sample = rng.choice(selected_values, size=n, replace=sel_replace)
                not_sample = rng.choice(not_selected_values, size=n, replace=not_replace)
                deltas.append(np.nanmean(sel_sample) - np.nanmean(not_sample))

        deltas = np.asarray(deltas, dtype=float)

        return {
            "balanced_delta_mean_shap": float(np.nanmean(deltas)),
            "balanced_delta_ci_low": float(np.nanpercentile(deltas, q_low)),
            "balanced_delta_ci_high": float(np.nanpercentile(deltas, q_high)),
            "balanced_delta_std": float(np.nanstd(deltas, ddof=1)),
            "balanced_n_per_group": int(min(n_sel, n_not)),
        }

    # ------------------------------------------------------------------
    # Build summary rows
    # ------------------------------------------------------------------
    rows = []

    for shap_model in shap_model_list:
        summary = all_results[summary_key][shap_model]

        required_summary_keys = {
            "idx",
            "feature_names",
            "shap_values_mean",
            "base_values_mean",
            "data",
        }

        missing_summary_keys = required_summary_keys - set(summary.keys())
        if missing_summary_keys:
            raise KeyError(
                f"SHAP summary for model={shap_model!r} is missing keys: "
                f"{sorted(missing_summary_keys)}"
            )

        shap_idx = np.asarray(summary["idx"])
        feature_names = list(summary["feature_names"])
        shap_values = np.asarray(summary["shap_values_mean"], dtype=float)
        base_values = np.asarray(summary["base_values_mean"], dtype=float)
        feature_data = np.asarray(summary["data"])

        if shap_values.ndim != 2:
            raise ValueError(
                f"summary['shap_values_mean'] for model={shap_model!r} must be 2D."
            )

        if shap_values.shape[1] != len(feature_names):
            raise ValueError(
                f"Feature-name mismatch for model={shap_model!r}: "
                f"shap_values_mean.shape[1]={shap_values.shape[1]} but "
                f"len(feature_names)={len(feature_names)}."
            )

        shap_id_to_pos = {
            int(pid): pos
            for pos, pid in enumerate(shap_idx)
        }

        selected_positions = [
            shap_id_to_pos[int(pid)]
            for pid in selected_ids
            if int(pid) in shap_id_to_pos
        ]

        not_selected_positions = [
            shap_id_to_pos[int(pid)]
            for pid in not_selected_ids
            if int(pid) in shap_id_to_pos
        ]

        missing_selected = [
            int(pid)
            for pid in selected_ids
            if int(pid) not in shap_id_to_pos
        ]

        missing_not_selected = [
            int(pid)
            for pid in not_selected_ids
            if int(pid) not in shap_id_to_pos
        ]

        if len(selected_positions) == 0:
            raise ValueError(
                f"No selected reference patients were found in SHAP summary idx "
                f"for model={shap_model!r}."
            )

        if len(not_selected_positions) == 0:
            raise ValueError(
                f"No not-selected reference patients were found in SHAP summary idx "
                f"for model={shap_model!r}."
            )

        selected_shap = shap_values[selected_positions, :]
        not_selected_shap = shap_values[not_selected_positions, :]

        selected_data = feature_data[selected_positions, :]
        not_selected_data = feature_data[not_selected_positions, :]

        selected_base = base_values[selected_positions]
        not_selected_base = base_values[not_selected_positions]

        n_selected = int(selected_shap.shape[0])
        n_not_selected = int(not_selected_shap.shape[0])

        mean_base_value_selected = _safe_mean(selected_base)
        mean_base_value_not_selected = _safe_mean(not_selected_base)

        if "predictions_mean" in summary:
            predictions = np.asarray(summary["predictions_mean"], dtype=float)

            selected_predictions = predictions[selected_positions]
            not_selected_predictions = predictions[not_selected_positions]

            mean_prediction_selected = _safe_mean(selected_predictions)
            mean_prediction_not_selected = _safe_mean(not_selected_predictions)
        else:
            mean_prediction_selected = (
                mean_base_value_selected
                + float(np.nanmean(selected_shap, axis=0).sum())
            )

            mean_prediction_not_selected = (
                mean_base_value_not_selected
                + float(np.nanmean(not_selected_shap, axis=0).sum())
            )

        for j, feature in enumerate(feature_names):
            x_sel = selected_shap[:, j]
            x_not = not_selected_shap[:, j]

            endpoint_sel = selected_base + x_sel
            endpoint_not = not_selected_base + x_not

            mean_sel = _safe_mean(x_sel)
            mean_not = _safe_mean(x_not)
            delta_mean = mean_sel - mean_not

            median_sel = _safe_median(x_sel)
            median_not = _safe_median(x_not)
            delta_median = median_sel - median_not

            abs_sel = np.abs(x_sel)
            abs_not = np.abs(x_not)

            mean_abs_sel = _safe_mean(abs_sel)
            mean_abs_not = _safe_mean(abs_not)
            delta_mean_abs = mean_abs_sel - mean_abs_not

            pct_pos_sel = _pct_positive(x_sel)
            pct_pos_not = _pct_positive(x_not)
            delta_pct_pos = pct_pos_sel - pct_pos_not

            pooled = _pooled_sd(x_sel, x_not)
            standardized_delta = (
                delta_mean / pooled
                if pooled is not None and not np.isnan(pooled) and pooled != 0
                else np.nan
            )

            bootstrap_stats = _balanced_bootstrap_delta(x_sel, x_not)

            row = {
                "reference_selection_model": reference_selection_model,
                "model": shap_model,
                "feature": feature,

                "n_selected": n_selected,
                "n_not_selected": n_not_selected,
                "n_total_used": n_selected + n_not_selected,
                "n_missing_selected_from_shap": len(missing_selected),
                "n_missing_not_selected_from_shap": len(missing_not_selected),

                "mean_base_value_selected": mean_base_value_selected,
                "mean_base_value_not_selected": mean_base_value_not_selected,
                "mean_prediction_selected": mean_prediction_selected,
                "mean_prediction_not_selected": mean_prediction_not_selected,

                "mean_endpoint_selected": _safe_mean(endpoint_sel),
                "mean_endpoint_not_selected": _safe_mean(endpoint_not),
                "median_endpoint_selected": _safe_median(endpoint_sel),
                "median_endpoint_not_selected": _safe_median(endpoint_not),
                "std_endpoint_selected": _safe_std(endpoint_sel),
                "std_endpoint_not_selected": _safe_std(endpoint_not),
                "sem_endpoint_selected": _safe_sem(endpoint_sel),
                "sem_endpoint_not_selected": _safe_sem(endpoint_not),
                "delta_mean_endpoint": (
                    _safe_mean(endpoint_sel) - _safe_mean(endpoint_not)
                ),

                "mean_shap_selected": mean_sel,
                "mean_shap_not_selected": mean_not,
                "delta_mean_shap": delta_mean,

                "median_shap_selected": median_sel,
                "median_shap_not_selected": median_not,
                "delta_median_shap": delta_median,

                "std_shap_selected": _safe_std(x_sel),
                "std_shap_not_selected": _safe_std(x_not),
                "sem_shap_selected": _safe_sem(x_sel),
                "sem_shap_not_selected": _safe_sem(x_not),

                "mean_abs_shap_selected": mean_abs_sel,
                "mean_abs_shap_not_selected": mean_abs_not,
                "delta_mean_abs_shap": delta_mean_abs,

                "pct_positive_shap_selected": pct_pos_sel,
                "pct_positive_shap_not_selected": pct_pos_not,
                "delta_pct_positive_shap": delta_pct_pos,

                "standardized_delta_mean_shap": standardized_delta,

                "mean_feature_value_selected": _safe_mean(selected_data[:, j]),
                "mean_feature_value_not_selected": _safe_mean(not_selected_data[:, j]),
                "delta_mean_feature_value": (
                    _safe_mean(selected_data[:, j])
                    - _safe_mean(not_selected_data[:, j])
                ),

                "balance_method": balance_method,
                "n_bootstrap": n_bootstrap,
                **bootstrap_stats,
            }

            rows.append(row)

    summary_df = pd.DataFrame(rows)

    summary_df["rank_abs_delta_mean_shap_within_model"] = (
        summary_df
        .groupby("model")["delta_mean_shap"]
        .transform(lambda s: s.abs().rank(method="dense", ascending=False))
        .astype(int)
    )

    summary_df["rank_mean_abs_shap_selected_within_model"] = (
        summary_df
        .groupby("model")["mean_abs_shap_selected"]
        .transform(lambda s: s.rank(method="dense", ascending=False))
        .astype(int)
    )

    summary_df = summary_df.sort_values(
        ["model", "rank_abs_delta_mean_shap_within_model", "feature"],
        kind="mergesort",
    ).reset_index(drop=True)

    return summary_df



# ---------------------------------------------------------------------
# Main external validation pipeline
# ---------------------------------------------------------------------

def run_diagnostic_enrichment_workflow(
    all_results,
    *,
    threshold: float | None = None,
    threshold_sweep_values=None,
    model=None,

    # Pass-through kwargs
    long_prediction_kwargs: dict | None = None,
    patient_pooling_kwargs: dict | None = None,
    threshold_sweep_kwargs: dict | None = None,
    decision_table_kwargs: dict | None = None,
    enrichment_kwargs: dict | None = None,

    # Optional plot kwargs
    threshold_plot_kwargs: dict | None = None,
    final_patient_plot_kwargs: dict | None = None,

    # Progress
    progress_kwargs: dict | None = None,
) -> dict:
    """
    Run the diagnostic enrichment workflow from nested-CV prediction results.

    This function is intentionally a thin workflow coordinator. It builds
    patient-level risk summaries, runs a threshold sweep for decision support,
    runs the final fixed-threshold diagnostic enrichment analysis, optionally
    creates plots, and returns all intermediate and final outputs in one
    structured dictionary.

    The workflow does not duplicate the full parameter defaults of the
    downstream functions. Instead, it accepts pass-through kwargs dictionaries.
    To customize a step, pass the relevant keyword arguments in that step's
    kwargs block.

    Workflow
    --------
    1. Build long-format prediction rows from `all_results`.
    2. Pool repeated predictions into a patient-level risk table.
    3. Detect available model names and resolve requested models.
    4. Run a threshold sweep across candidate thresholds.
    5. Build a compact threshold decision table.
    6. Optionally plot threshold-sweep decision summaries.
    7. Run the final fixed-threshold diagnostic enrichment analysis.
    8. Build pocket, operating, and planning summary tables.
    9. Extract selected and eligible patient tables by model.
    10. Optionally plot final patient-level figures.
    11. Return one workflow output dictionary.

    Parameters
    ----------
    all_results : mapping
        Nested model-results dictionary from nested cross-validation.

        Expected structure:
            all_results[model_name] = list of fold/trial result dictionaries

        This is passed to `build_model_prediction_rows(...)`.

    threshold : float or None, default None
        Final diagnostic enrichment cutoff used for the fixed-threshold
        analysis.

        If None, the workflow uses 0.70.

        Patients with predicted probability >= threshold are treated as
        selected into the diagnostic-enriched subgroup.

    threshold_sweep_values : sequence of float or None, default None
        Candidate thresholds used for the threshold-sweep decision-support
        step.

        If None, the workflow uses:
            np.round(np.arange(0.65, 0.95, 0.05), 2)

        The threshold sweep does not automatically choose the final threshold.
        It is used to help inspect tradeoffs across possible cutoffs.

    model : str, sequence of str, or None, default None
        Model or models to include.

        If None, models are detected automatically from the patient-level risk
        table after prediction pooling. This avoids hardcoding model names and
        supports workflows with any number of models.

    long_prediction_kwargs : dict or None, default None
        Pass-through keyword arguments for:

            build_model_prediction_rows(all_results, **long_prediction_kwargs)

        Common examples include:
            - model_name
            - methods
            - groups_all
            - group_id_to_key
            - include_uncalibrated
            - include_test
            - include_train_oof
            - unit_col

        The workflow sets only minimal defaults needed for this workflow:
            {"model_name": None, "methods": ["beta"], "unit_col": "idx"}

        All other defaults are controlled by `build_model_prediction_rows`.

    patient_pooling_kwargs : dict or None, default None
        Pass-through keyword arguments for:

            pooled_patient_risk_summary(
                prediction_rows,
                **patient_pooling_kwargs
            )

        Common examples include:
            - grouping
            - unit_col
            - agg
            - lower_q
            - upper_q
            - include_test
            - include_train_oof
            - truncate_decimals

        The workflow sets only minimal defaults needed for this workflow:
            {"grouping": "all_trials", "unit_col": "idx"}

        All other defaults are controlled by `pooled_patient_risk_summary`.

    threshold_sweep_kwargs : dict or None, default None
        Pass-through keyword arguments for:

            diagnostic_enrichment_threshold_sweep_by_model(
                df=patient_risk_table,
                thresholds=threshold_sweep_values,
                **threshold_sweep_kwargs
            )

        Common examples include:
            - score_col
            - variants
            - split
            - subject_col
            - y_col
            - label_col
            - confidence
            - precision
            - compute_power

        If `model` is not supplied inside this dictionary, the workflow inserts
        the resolved model list automatically.

    decision_table_kwargs : dict or None, default None
        Pass-through keyword arguments for:

            make_threshold_decision_table(
                threshold_sweep_table,
                **decision_table_kwargs
            )

        Common examples include:
            - threshold_col
            - round_digits

    enrichment_kwargs : dict or None, default None
        Pass-through keyword arguments for:

            post.diagnostic_enrichment_pipeline_by_model(
                df=patient_risk_table,
                threshold=threshold,
                **enrichment_kwargs
            )

        Common examples include:
            - score_col
            - variants
            - split
            - subject_col
            - y_col
            - label_col
            - confidence
            - precision
            - compute_power
            - power_alpha
            - power_alternative
            - power_endpoint
            - power_method

        If `model` is not supplied inside this dictionary, the workflow inserts
        the resolved model list automatically.

    threshold_plot_kwargs : dict or None, default None
        Optional kwargs controlling threshold-sweep decision plots.

        If None or if `enabled=False`, threshold-sweep plots are skipped.

        If enabled, the remaining kwargs are passed to:

            plot_threshold_decision_bars(
                decision_table,
                **threshold_plot_kwargs_without_enabled
            )

        Example:
            threshold_plot_kwargs={
                "enabled": True,
                "model_alias": {...},
                "model_palette": {...},
                "figsize_per_panel": (4.3, 3.4),
            }

    final_patient_plot_kwargs : dict or None, default None
        Optional kwargs controlling final patient-level plots at the selected
        threshold.

        If None or if `enabled=False`, final patient-level plots are skipped.

        Expected optional sub-dictionaries are:

            full_ranked_plot_kwargs
                Passed to:
                    plot_ranked_patients_patient_level(
                        patient_risk_table,
                        model=model_name,
                        **full_ranked_plot_kwargs
                    )

            distribution_plot_kwargs
                Passed to:
                    plot_patient_risk_distributions_by_outcome(
                        patient_risk_table,
                        model_name=model_name,
                        **distribution_plot_kwargs
                    )

            selected_ranked_plot_kwargs
                Passed to:
                    plot_ranked_patients_patient_level(
                        selected_patients_by_model[model_name],
                        **selected_ranked_plot_kwargs
                    )

        Any of these sub-dictionaries may be omitted. A plot type only runs
        when its corresponding sub-dictionary is provided.

        Example:
            final_patient_plot_kwargs={
                "enabled": True,
                "full_ranked_plot_kwargs": {...},
                "distribution_plot_kwargs": {...},
                "selected_ranked_plot_kwargs": {...},
            }

    progress_kwargs : dict or None, default None
        Parameters controlling progress logging.

        Supported keys:
            enabled : bool, default True
                Print progress messages.

            show_output_shapes : bool, default True
                Include compact output summaries in progress messages.

            return_progress_log : bool, default True
                Include a progress_log DataFrame in the returned output.

    Returns
    -------
    outputs : dict
        Structured workflow output dictionary.

        Top-level keys:
            config
                Resolved workflow configuration and kwargs.

            prediction_rows
                Long-format prediction table from
                `build_model_prediction_rows`.

            patient_risk_table
                Patient-level pooled risk table from
                `pooled_patient_risk_summary`.

            models
                Resolved model names included in the workflow.

            threshold_sweep
                Dictionary containing:
                    - thresholds
                    - sweep_table
                    - skipped_thresholds
                    - decision_table
                    - plots

            final_threshold
                Dictionary containing:
                    - threshold
                    - model_outputs
                    - pocket_table
                    - operating_table
                    - planning_table
                    - selected_patients_by_model
                    - eligible_patients_by_model
                    - plots

            progress_log
                Progress log DataFrame, if requested.

    Notes
    -----
    This workflow does not automatically select the best threshold. The
    threshold sweep is intended for decision support. The final enrichment
    analysis uses the user-provided threshold, or 0.70 if threshold is None.

    The workflow also does not hardcode model names. If `model=None`, models
    are detected from the patient-level risk table after prediction pooling.
    """

    # ------------------------------------------------------------------
    # Minimal workflow defaults only
    # ------------------------------------------------------------------
    if threshold is None:
        threshold = 0.70

    if threshold_sweep_values is None:
        threshold_sweep_values = np.round(np.arange(0.65, 0.95, 0.05), 2)

    long_prediction_kwargs = _merge_kwargs(
        {
            "model_name": None,
            "methods": ["beta"],
            "unit_col": "idx",
        },
        long_prediction_kwargs,
    )

    patient_pooling_kwargs = _merge_kwargs(
        {
            "grouping": "all_trials",
            "unit_col": "idx",
        },
        patient_pooling_kwargs,
    )

    threshold_sweep_kwargs = dict(threshold_sweep_kwargs or {})
    decision_table_kwargs = dict(decision_table_kwargs or {})
    enrichment_kwargs = dict(enrichment_kwargs or {})

    threshold_plot_enabled, threshold_plot_call_kwargs = _enabled_kwargs(
        threshold_plot_kwargs
    )

    final_patient_plot_enabled = bool(
        (final_patient_plot_kwargs or {}).get("enabled", False)
    )
    final_patient_plot_kwargs = dict(final_patient_plot_kwargs or {})

    progress_kwargs = _merge_kwargs(
        {
            "enabled": True,
            "show_output_shapes": True,
            "return_progress_log": True,
        },
        progress_kwargs,
    )

    progress_enabled = bool(progress_kwargs["enabled"])
    show_output_shapes = bool(progress_kwargs["show_output_shapes"])
    progress_rows = []

    if progress_enabled:
        print("Diagnostic enrichment workflow")
        print("-" * 38)

    outputs = {
        "config": {
            "threshold": threshold,
            "threshold_sweep_values": list(threshold_sweep_values),
            "model": model,
            "long_prediction_kwargs": copy.deepcopy(long_prediction_kwargs),
            "patient_pooling_kwargs": copy.deepcopy(patient_pooling_kwargs),
            "threshold_sweep_kwargs": copy.deepcopy(threshold_sweep_kwargs),
            "decision_table_kwargs": copy.deepcopy(decision_table_kwargs),
            "enrichment_kwargs": copy.deepcopy(enrichment_kwargs),
            "threshold_plot_kwargs": copy.deepcopy(threshold_plot_kwargs),
            "final_patient_plot_kwargs": copy.deepcopy(final_patient_plot_kwargs),
            "progress_kwargs": copy.deepcopy(progress_kwargs),
        }
    }

    # ------------------------------------------------------------------
    # 1. Build prediction rows
    # ------------------------------------------------------------------
    outputs["prediction_rows"] = _run_pipeline_step(
        progress_rows,
        step="Build prediction rows",
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
        func=lambda: build_model_prediction_rows(
            all_results,
            **long_prediction_kwargs,
        ),
    )

    # ------------------------------------------------------------------
    # 2. Build patient risk table
    # ------------------------------------------------------------------
    outputs["patient_risk_table"] = _run_pipeline_step(
        progress_rows,
        step="Build patient risk table",
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
        func=lambda: pooled_patient_risk_summary(
            outputs["prediction_rows"],
            **patient_pooling_kwargs,
        ),
    )

    # ------------------------------------------------------------------
    # 3. Detect / resolve models
    # ------------------------------------------------------------------
    available_models = _run_pipeline_step(
        progress_rows,
        step="Detect available models",
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
        func=lambda: _detect_models_from_table(
            outputs["patient_risk_table"],
            model_col="model",
        ),
    )

    requested_models = _run_pipeline_step(
        progress_rows,
        step="Resolve requested models",
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
        func=lambda: _resolve_requested_models(
            requested_model=model,
            available_models=available_models,
        ),
    )

    outputs["models"] = requested_models

    # If user did not specify model inside these kwargs, use resolved models.
    # This avoids hardcoding while preventing accidental inclusion of non-model rows.
    threshold_sweep_call_kwargs = dict(threshold_sweep_kwargs)
    threshold_sweep_call_kwargs.setdefault("model", requested_models)

    enrichment_call_kwargs = dict(enrichment_kwargs)
    enrichment_call_kwargs.setdefault("model", requested_models)

    # ------------------------------------------------------------------
    # 4. Run threshold sweep
    # ------------------------------------------------------------------
    sweep_result = _run_pipeline_step(
        progress_rows,
        step="Run threshold sweep",
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
        func=lambda: diagnostic_enrichment_threshold_sweep_by_model(
            df=outputs["patient_risk_table"],
            thresholds=threshold_sweep_values,
            **threshold_sweep_call_kwargs,
        ),
    )

    threshold_sweep_table, skipped_thresholds = sweep_result

    outputs["threshold_sweep"] = {
        "thresholds": list(threshold_sweep_values),
        "sweep_table": threshold_sweep_table,
        "skipped_thresholds": skipped_thresholds,
    }

    # ------------------------------------------------------------------
    # 5. Build decision table
    # ------------------------------------------------------------------
    outputs["threshold_sweep"]["decision_table"] = _run_pipeline_step(
        progress_rows,
        step="Build threshold decision table",
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
        func=lambda: make_threshold_decision_table(
            threshold_sweep_table,
            **decision_table_kwargs,
        ),
    )

    # ------------------------------------------------------------------
    # 6. Optional threshold-sweep plots
    # ------------------------------------------------------------------
    if threshold_plot_enabled:
        outputs["threshold_sweep"]["plots"] = _run_pipeline_step(
            progress_rows,
            step="Plot threshold decision panels",
            progress_enabled=progress_enabled,
            show_output_shapes=show_output_shapes,
            func=lambda: plot_threshold_decision_bars(
                outputs["threshold_sweep"]["decision_table"],
                **threshold_plot_call_kwargs,
            ),
        )
    else:
        outputs["threshold_sweep"]["plots"] = None

        progress_rows.append(
            {
                "step": "Plot threshold decision panels",
                "status": "skipped",
                "detail": "threshold_plot_kwargs is None or enabled=False",
                "elapsed_seconds": 0.0,
                "error": None,
            }
        )

        if progress_enabled:
            print(
                _format_pipeline_message(
                    "skipped",
                    "Plot threshold decision panels",
                    "threshold_plot_kwargs is None or enabled=False",
                )
            )

    # ------------------------------------------------------------------
    # 7. Final fixed-threshold enrichment analysis
    # ------------------------------------------------------------------
    enrichment_outputs = _run_pipeline_step(
        progress_rows,
        step=f"Run final enrichment analysis at threshold={threshold:.2f}",
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
        func=lambda: post.diagnostic_enrichment_pipeline_by_model(
            df=outputs["patient_risk_table"],
            threshold=threshold,
            **enrichment_call_kwargs,
        ),
    )

    outputs["final_threshold"] = {
        "threshold": threshold,
        "model_outputs": enrichment_outputs,
    }

    # ------------------------------------------------------------------
    # 8. Build summary tables
    # ------------------------------------------------------------------
    outputs["final_threshold"]["pocket_table"] = _run_pipeline_step(
        progress_rows,
        step="Build pocket summary table",
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
        func=lambda: _concat_model_output_tables(
            enrichment_outputs,
            output_key="pocket_summary",
        ),
    )

    outputs["final_threshold"]["operating_table"] = _run_pipeline_step(
        progress_rows,
        step="Build operating summary table",
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
        func=lambda: _concat_model_output_tables(
            enrichment_outputs,
            output_key="operating_summary",
        ),
    )

    outputs["final_threshold"]["planning_table"] = _run_pipeline_step(
        progress_rows,
        step="Build planning summary table",
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
        func=lambda: _concat_model_output_tables(
            enrichment_outputs,
            output_key="planning_summary",
        ),
    )

    # ------------------------------------------------------------------
    # 9. Extract selected / eligible patient tables
    # ------------------------------------------------------------------
    outputs["final_threshold"]["selected_patients_by_model"] = _run_pipeline_step(
        progress_rows,
        step="Extract selected patients by model",
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
        func=lambda: _extract_model_output_tables(
            enrichment_outputs,
            output_key="df_hi",
        ),
    )

    outputs["final_threshold"]["eligible_patients_by_model"] = _run_pipeline_step(
        progress_rows,
        step="Extract eligible patients by model",
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
        func=lambda: _extract_model_output_tables(
            enrichment_outputs,
            output_key="df_all",
        ),
    )

    # ------------------------------------------------------------------
    # 10. Optional final patient-level plots
    # ------------------------------------------------------------------
    outputs["final_threshold"]["plots"] = {
        "full_patient_rankings": {},
        "full_patient_distributions": {},
        "selected_patient_rankings": {},
    }

    if final_patient_plot_enabled:
        full_ranked_plot_kwargs = dict(
            final_patient_plot_kwargs.get("full_ranked_plot_kwargs", {})
        )
        distribution_plot_kwargs = dict(
            final_patient_plot_kwargs.get("distribution_plot_kwargs", {})
        )
        selected_ranked_plot_kwargs = dict(
            final_patient_plot_kwargs.get("selected_ranked_plot_kwargs", {})
        )

        run_full_ranked = len(full_ranked_plot_kwargs) > 0
        run_distribution = len(distribution_plot_kwargs) > 0
        run_selected_ranked = len(selected_ranked_plot_kwargs) > 0

        for model_name in requested_models:
            if run_full_ranked:
                outputs["final_threshold"]["plots"]["full_patient_rankings"][model_name] = _run_pipeline_step(
                    progress_rows,
                    step=f"Plot full patient ranking: {model_name}",
                    progress_enabled=progress_enabled,
                    show_output_shapes=show_output_shapes,
                    func=lambda model_name=model_name: plot_ranked_patients_patient_level(
                        outputs["patient_risk_table"],
                        model=model_name,
                        **full_ranked_plot_kwargs,
                    ),
                )

            if run_distribution:
                outputs["final_threshold"]["plots"]["full_patient_distributions"][model_name] = _run_pipeline_step(
                    progress_rows,
                    step=f"Plot patient risk distribution: {model_name}",
                    progress_enabled=progress_enabled,
                    show_output_shapes=show_output_shapes,
                    func=lambda model_name=model_name: plot_patient_risk_distributions_by_outcome(
                        outputs["patient_risk_table"],
                        model_name=model_name,
                        **distribution_plot_kwargs,
                    ),
                )

            if run_selected_ranked:
                selected_df = outputs["final_threshold"]["selected_patients_by_model"].get(
                    model_name
                )

                if selected_df is not None:
                    outputs["final_threshold"]["plots"]["selected_patient_rankings"][model_name] = _run_pipeline_step(
                        progress_rows,
                        step=f"Plot selected patient ranking: {model_name}",
                        progress_enabled=progress_enabled,
                        show_output_shapes=show_output_shapes,
                        func=lambda selected_df=selected_df: plot_ranked_patients_patient_level(
                            selected_df,
                            **selected_ranked_plot_kwargs,
                        ),
                    )
    else:
        progress_rows.append(
            {
                "step": "Plot final patient-level figures",
                "status": "skipped",
                "detail": "final_patient_plot_kwargs is None or enabled=False",
                "elapsed_seconds": 0.0,
                "error": None,
            }
        )

        if progress_enabled:
            print(
                _format_pipeline_message(
                    "skipped",
                    "Plot final patient-level figures",
                    "final_patient_plot_kwargs is None or enabled=False",
                )
            )

    # ------------------------------------------------------------------
    # 11. Progress log and return
    # ------------------------------------------------------------------
    progress_log = pd.DataFrame(progress_rows)

    if progress_kwargs.get("return_progress_log", True):
        outputs["progress_log"] = progress_log

    if progress_enabled:
        print("-" * 38)
        print("[OK] Diagnostic enrichment workflow complete")

    return outputs

    
def run_external_validation_pipeline(
    all_results: Dict[str, list[dict[str, Any]]],
    train_bundle: Mapping[str, Any],
    validation_bundle: Mapping[str, Any],
    *,
    bundle_df_kwargs: Optional[dict[str, Any]] = None,
    model_data_kwargs: Optional[dict[str, Any]] = None,
    external_prediction_kwargs: Optional[dict[str, Any]] = None,
    long_prediction_kwargs: Optional[dict[str, Any]] = None,
    aggregation_kwargs: Optional[dict[str, Any]] = None,
    shap_kwargs: Optional[dict[str, Any]] = None,
    pdp_kwargs: Optional[dict[str, Any]] = None,
    progress_kwargs: Optional[dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the non-plotting external validation pipeline.

    This function converts train/validation bundles into DataFrames, builds
    model-specific validation datasets, scores every fold-level model on the
    validation set, builds long-form and aggregated prediction tables, and
    optionally computes external SHAP explanations and PDP-based probability contribution summaries.

    Plotting functions are intentionally not called here. Use the returned
    `all_results`, `df_long`, and `df_agg` with standalone plotting functions.

    Parameters
    ----------
    all_results : dict
        Nested model-results dictionary from nested cross-validation.

        Expected structure:
            all_results[model_name] = list of fold/trial result dictionaries

        Non-model summary keys are allowed. For example:
            all_results["external_shap_summary"]

        These summary/meta keys are ignored when building prediction tables
        because valid model keys are detected as top-level values that are
        non-empty lists of fold/trial dictionaries.

    train_bundle : mapping
        Training bundle produced by the data-preparation/preprocessing workflow.

        Common expected keys are:
            - "X_scaled"
            - "y"
            - "feature_names"

        The exact keys are controlled by `bundle_df_kwargs`.

    validation_bundle : mapping
        Final validation/external bundle produced by the data-preparation/
        preprocessing workflow.

        Common expected keys are:
            - "X_scaled"
            - "y"
            - "feature_names"

        The exact keys are controlled by `bundle_df_kwargs`.

    bundle_df_kwargs : dict, optional
        Parameters passed to `make_bundle_df(...)`.

        Available keys:
            X_key : str, default "X_scaled"
                Key in each bundle containing the feature matrix.

            y_key : str, default "y"
                Key in each bundle containing the target vector.

            feature_names_key : str, default "feature_names"
                Key in each bundle containing feature names aligned with
                columns of `bundle[X_key]`.

            y_col : str, default "target"
                Name assigned to the target column in the returned DataFrames.

    model_data_kwargs : dict, optional
        Parameters passed to `make_model_data_dict_from_results(...)`.

        Available keys:
            results_feature_names_key : str, default "feature_names_used"
                Key inside each fold/trial record containing selected features.

            feature_strategy : {"union", "first"}, default "union"
                "union" includes all selected features used across all records
                for each model. This is safest if selected features differ
                across folds/trials.

                "first" uses only the first record's feature set for each model.

            strict_features : bool, default True
                If True, raise an error when required features are missing.

            include_private_keys : bool, default False
                Kept for compatibility with `make_model_data_dict_from_results`.
                However, this pipeline also protects downstream steps by only
                treating top-level list values as model records.

    external_prediction_kwargs : dict, optional
        Parameters passed to `add_external_predictions_to_results(...)`.

        Available keys:
            external_tag : str, default "external"
                Prefix/tag used when writing prediction keys back into
                `all_results`.

            feature_names_key : str, optional
                Key inside each fold/trial record containing selected features.
                If omitted, this defaults to
                `model_data_kwargs["results_feature_names_key"]`.

            strict_features : bool, optional
                Whether to error if selected model features are missing.
                If omitted, this defaults to
                `model_data_kwargs["strict_features"]`.

            inplace : bool, default True
                If True, update `all_results` in place.

            warn_on_skip : bool, default True
                If True, warn when model names do not overlap between
                `all_results` and `model_data_dict`.

    long_prediction_kwargs : dict, optional
        Parameters passed to `build_long_predictions_df(...)`.

        Available keys:
            model_name : str, sequence of str, or None, default None
                Which model(s) to include. If None, this pipeline automatically
                restricts to real model keys and excludes summary/meta keys such
                as "external_shap_summary".

            methods : sequence of str or None, default ["beta"]
                Calibrated prediction methods to include.

            include_uncalibrated : bool
                Optional. If not provided, the lower-level function default is
                used, which is True.

            external_idx_key : str, default "external_idx"
            external_y_key : str, default "y_external"
            external_prob_key_uncalib : str, default "y_external_scores"
            external_prob_key_prefix_calib : str, default "calib_external_predictions_"

    aggregation_kwargs : dict, optional
        Parameters passed to `aggregate_predictions_by_idx(...)`.

        Available keys:
            model_name : str, sequence of str, or None, default None
                Which model(s) to include. If None, this pipeline automatically
                restricts to real model keys and excludes summary/meta keys.

            calibrations : sequence of str or None, default ["uncalib", "beta"]
            agg_stats : sequence of str, default ("mean", "median", "std", "min", "max")
            add_y_label : bool, default True
            prevalence : bool or float, default True
            add_ensemble : bool, default True
            ensemble_name : str, default "Ensemble model"
            ensemble_models : sequence of str or None, default None
            truncate_decimals : int or None, default None

    shap_kwargs : dict, optional
        Parameters controlling SHAP computation.

        Available keys:
            enabled : bool, default False
                If True, compute external SHAP values.

            external_tag : str, optional
                Prefix/tag used when writing SHAP keys into `all_results`.
                If omitted, this defaults to
                `external_prediction_kwargs["external_tag"]`.

            strict_features : bool, optional
                If omitted, this defaults to
                `model_data_kwargs["strict_features"]`.

            max_background : int or None, default None
            random_state : int, default 42
            check_additivity : bool, default True
            additivity_tolerance : float, default 1e-4
            warn_on_skip : bool, default True

            add_summary : bool, default True
                If True, compute model-level mean SHAP summaries.

            summary_key : str, default "external_shap_summary"
                Top-level key where SHAP summaries are stored inside
                `all_results`.

                If this key already exists, it is overwritten by
                `add_external_shap_summary_to_results(...)`.

    pdp_kwargs : dict, optional
        Parameters controlling PDP-based probability contribution construction.

        Available keys:
            enabled : bool, default False
                If True, build patient-level PDP signal/contribution tables.

            model_name : str, sequence of str, or None, default None
                Which model(s) to include. If None, all real model keys are
                detected from `all_results`.

            calibrations : sequence of str, default ("uncalib", "beta")
                Prediction outputs to use when rescaling PDP signal shares.

            interpolation : {"linear", "nearest"}, default "linear"
                How to read PDP curves at external patient feature values.

            clip_to_grid : bool, default True
                If True, external feature values outside a PDP grid are clipped
                to the nearest PDP grid endpoint.

            rescale_to_prediction : bool, default True
                If True, PDP signal shares are multiplied by the patient's
                external predicted probability, so feature contributions sum to
                the patient prediction.

            include_detail : bool, default True
                If True, return fold/trial-level PDP lookup details.

            return_long : bool, default True
                If True, return concatenated summary_long and detail_long tables.

            warn_on_skip : bool, default False
                If True, print warnings when a model/calibration combination is
                skipped.

    progress_kwargs : dict, optional
        Parameters controlling printed pipeline progress.

        Available keys:
            enabled : bool, default True
            show_output_shapes : bool, default True
            return_progress_log : bool, default True

    Returns
    -------
    outputs : dict
        Dictionary containing:
            - config
            - all_results
            - train_data
            - validation_df
            - model_data_dict
            - df_long
            - df_agg
            - progress_log, if requested
    """

    # ------------------------------------------------------------------
    # Resolve kwargs
    # ------------------------------------------------------------------
    bundle_df_kwargs = _merge_kwargs(
        {
            "X_key": "X_scaled",
            "y_key": "y",
            "feature_names_key": "feature_names",
            "y_col": "target",
        },
        bundle_df_kwargs,
    )

    model_data_kwargs = _merge_kwargs(
        {
            "results_feature_names_key": "feature_names_used",
            "feature_strategy": "union",
            "strict_features": True,
            "include_private_keys": False,
        },
        model_data_kwargs,
    )

    external_prediction_kwargs = _merge_kwargs(
        {
            "external_tag": "external",
            "inplace": True,
            "warn_on_skip": True,
        },
        external_prediction_kwargs,
    )

    long_prediction_kwargs = _merge_kwargs(
        {
            "model_name": None,
            "methods": ["beta"],
        },
        long_prediction_kwargs,
    )

    aggregation_kwargs = _merge_kwargs(
        {
            "model_name": None,
            "calibrations": ["uncalib", "beta"],
            "agg_stats": ("mean", "median", "std", "min", "max"),
            "add_y_label": True,
            "prevalence": True,
            "add_ensemble": True,
            "ensemble_name": "Ensemble model",
            "ensemble_models": None,
            "truncate_decimals": None,
        },
        aggregation_kwargs,
    )

    shap_kwargs = _merge_kwargs(
        {
            "enabled": False,
            "external_tag": external_prediction_kwargs["external_tag"],
            "strict_features": model_data_kwargs["strict_features"],
            "max_background": None,
            "random_state": 42,
            "check_additivity": True,
            "additivity_tolerance": 1e-4,
            "warn_on_skip": True,
            "add_summary": True,
            "summary_key": "external_shap_summary",
        },
        shap_kwargs,
    )

    pdp_kwargs = _merge_kwargs(
        {
            "enabled": False,
            "model_name": None,
            "calibrations": ("uncalib", "beta"),
            "interpolation": "linear",
            "clip_to_grid": True,
            "rescale_to_prediction": True,
            "include_detail": True,
            "return_long": True,
            "warn_on_skip": False,
        },
        pdp_kwargs,
    )

    progress_kwargs = _merge_kwargs(
        {
            "enabled": True,
            "show_output_shapes": True,
            "return_progress_log": True,
        },
        progress_kwargs,
    )

    # If user did not explicitly pass these, inherit from upstream kwargs.
    external_prediction_kwargs.setdefault(
        "feature_names_key",
        model_data_kwargs["results_feature_names_key"],
    )
    external_prediction_kwargs.setdefault(
        "strict_features",
        model_data_kwargs["strict_features"],
    )

    shap_kwargs.setdefault(
        "external_tag",
        external_prediction_kwargs["external_tag"],
    )
    shap_kwargs.setdefault(
        "strict_features",
        model_data_kwargs["strict_features"],
    )

    y_col = bundle_df_kwargs["y_col"]

    progress_enabled = bool(progress_kwargs["enabled"])
    show_output_shapes = bool(progress_kwargs["show_output_shapes"])
    progress_rows: list[dict[str, Any]] = []

    if progress_enabled:
        print("External validation pipeline")
        print("-" * 36)

    outputs: Dict[str, Any] = {
        "config": {
            "bundle_df_kwargs": copy.deepcopy(bundle_df_kwargs),
            "model_data_kwargs": copy.deepcopy(model_data_kwargs),
            "external_prediction_kwargs": copy.deepcopy(external_prediction_kwargs),
            "long_prediction_kwargs": copy.deepcopy(long_prediction_kwargs),
            "aggregation_kwargs": copy.deepcopy(aggregation_kwargs),
            "shap_kwargs": copy.deepcopy(shap_kwargs),
            "pdp_kwargs": copy.deepcopy(pdp_kwargs),
            "progress_kwargs": copy.deepcopy(progress_kwargs),
        }
    }

    # ------------------------------------------------------------------
    # Step 1: train bundle -> DataFrame
    # ------------------------------------------------------------------
    outputs["train_data"] = _run_pipeline_step(
        progress_rows,
        step="Build train dataframe",
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
        func=lambda: make_bundle_df(
            train_bundle,
            **bundle_df_kwargs,
        ),
    )

    # ------------------------------------------------------------------
    # Step 2: validation bundle -> DataFrame
    # ------------------------------------------------------------------
    outputs["validation_df"] = _run_pipeline_step(
        progress_rows,
        step="Build validation dataframe",
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
        func=lambda: make_bundle_df(
            validation_bundle,
            **bundle_df_kwargs,
        ),
    )

    # ------------------------------------------------------------------
    # Step 3: build model-specific external data
    # ------------------------------------------------------------------
    outputs["model_data_dict"] = _run_pipeline_step(
        progress_rows,
        step="Build model-specific external data",
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
        func=lambda: make_model_data_dict_from_results(
            all_results=all_results,
            external_df=outputs["validation_df"],
            y_col=y_col,
            **model_data_kwargs,
        ),
    )

    # ------------------------------------------------------------------
    # Step 4: add external predictions to all_results
    # ------------------------------------------------------------------
    outputs["all_results"] = _run_pipeline_step(
        progress_rows,
        step="Add external predictions",
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
        func=lambda: add_external_predictions_to_results(
            all_results,
            model_data_dict=outputs["model_data_dict"],
            y_col=y_col,
            **external_prediction_kwargs,
        ),
    )

    # ------------------------------------------------------------------
    # Identify real model keys after prediction update
    # ------------------------------------------------------------------
    valid_model_names = [
        model_name
        for model_name, records in outputs["all_results"].items()
        if isinstance(records, list)
        and len(records) > 0
        and isinstance(records[0], dict)
    ]

    if len(valid_model_names) == 0:
        raise ValueError(
            "No valid model keys found in all_results. Expected top-level model "
            "keys to map to non-empty lists of fold/trial result dictionaries."
        )

    # ------------------------------------------------------------------
    # Step 5: build long-form prediction table
    # ------------------------------------------------------------------
    long_prediction_call_kwargs = dict(long_prediction_kwargs)

    if long_prediction_call_kwargs.get("model_name", None) is None:
        long_prediction_call_kwargs["model_name"] = valid_model_names

    outputs["df_long"] = _run_pipeline_step(
        progress_rows,
        step="Build long prediction dataframe",
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
        func=lambda: build_long_predictions_df(
            outputs["all_results"],
            **long_prediction_call_kwargs,
        ),
    )

    # ------------------------------------------------------------------
    # Step 6: aggregate predictions by external index
    # ------------------------------------------------------------------
    aggregation_call_kwargs = dict(aggregation_kwargs)

    if aggregation_call_kwargs.get("model_name", None) is None:
        aggregation_call_kwargs["model_name"] = valid_model_names

    outputs["df_agg"] = _run_pipeline_step(
        progress_rows,
        step="Aggregate predictions by idx",
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
        func=lambda: aggregate_predictions_by_idx(
            outputs["df_long"],
            **aggregation_call_kwargs,
        ),
    )

    # ------------------------------------------------------------------
    # Step 7: optional SHAP values
    # ------------------------------------------------------------------
    if shap_kwargs.get("enabled", False):
        shap_call_kwargs = {
            "y_col": y_col,
            "external_tag": shap_kwargs["external_tag"],
            "strict_features": shap_kwargs["strict_features"],
            "max_background": shap_kwargs["max_background"],
            "random_state": shap_kwargs["random_state"],
            "check_additivity": shap_kwargs["check_additivity"],
            "additivity_tolerance": shap_kwargs["additivity_tolerance"],
            "warn_on_skip": shap_kwargs["warn_on_skip"],
        }

        outputs["all_results"] = _run_pipeline_step(
            progress_rows,
            step="Add external SHAP values",
            progress_enabled=progress_enabled,
            show_output_shapes=show_output_shapes,
            func=lambda: add_external_shap_to_results(
                outputs["all_results"],
                model_data_dict=outputs["model_data_dict"],
                train_data=outputs["train_data"],
                **shap_call_kwargs,
            ),
        )

        if shap_kwargs.get("add_summary", True):
            outputs["all_results"] = _run_pipeline_step(
                progress_rows,
                step="Add external SHAP summary",
                progress_enabled=progress_enabled,
                show_output_shapes=show_output_shapes,
                func=lambda: add_external_shap_summary_to_results(
                    outputs["all_results"],
                    external_tag=shap_kwargs["external_tag"],
                    summary_key=shap_kwargs["summary_key"],
                ),
            )
    else:
        detail = "shap_kwargs['enabled'] is False"

        if progress_enabled:
            print(_format_pipeline_message("skipped", "Add external SHAP values", detail))
            print(_format_pipeline_message("skipped", "Add external SHAP summary", detail))

        progress_rows.append(
            {
                "step": "Add external SHAP values",
                "status": "skipped",
                "detail": detail,
                "elapsed_seconds": 0.0,
                "error": None,
            }
        )
        progress_rows.append(
            {
                "step": "Add external SHAP summary",
                "status": "skipped",
                "detail": detail,
                "elapsed_seconds": 0.0,
                "error": None,
            }
        )

    # ------------------------------------------------------------------
    # Step 8: optional PDP signal allocation
    # ------------------------------------------------------------------
    if pdp_kwargs.get("enabled", False):
        pdp_call_kwargs = {
            "model_name": pdp_kwargs["model_name"],
            "calibrations": pdp_kwargs["calibrations"],
            "interpolation": pdp_kwargs["interpolation"],
            "clip_to_grid": pdp_kwargs["clip_to_grid"],
            "rescale_to_prediction": pdp_kwargs["rescale_to_prediction"],
            "include_detail": pdp_kwargs["include_detail"],
            "return_long": pdp_kwargs["return_long"],
            "warn_on_skip": pdp_kwargs["warn_on_skip"],
        }

        outputs["external_pdp_signal_outputs"] = _run_pipeline_step(
            progress_rows,
            step="Build external PDP signal allocation",
            progress_enabled=progress_enabled,
            show_output_shapes=show_output_shapes,
            func=lambda: build_external_pdp_signal_allocation_from_results(
                all_results=outputs["all_results"],
                **pdp_call_kwargs,
            ),
        )

        if pdp_kwargs.get("return_long", True):
            outputs["external_pdp_signal_summary"] = outputs[
                "external_pdp_signal_outputs"
            ].get("summary_long", pd.DataFrame())

            outputs["external_pdp_signal_detail"] = outputs[
                "external_pdp_signal_outputs"
            ].get("detail_long", pd.DataFrame())

    else:
        detail = "pdp_kwargs['enabled'] is False"

        if progress_enabled:
            print(
                _format_pipeline_message(
                    "skipped",
                    "Build external PDP signal allocation",
                    detail,
                )
            )

        progress_rows.append(
            {
                "step": "Build external PDP signal allocation",
                "status": "skipped",
                "detail": detail,
                "elapsed_seconds": 0.0,
                "error": None,
            }
        )
        
    # ------------------------------------------------------------------
    # Progress log
    # ------------------------------------------------------------------
    progress_log = pd.DataFrame(progress_rows)

    if progress_kwargs.get("return_progress_log", True):
        outputs["progress_log"] = progress_log

    if progress_enabled:
        print("-" * 36)
        print("[OK] Pipeline complete")

    return outputs






def run_enrichment_explanation_pipeline(
    *,
    all_results: dict,
    df_agg: pd.DataFrame,
    enrichment_kwargs: Optional[Mapping[str, Any]] = None,
    explanation_patient_kwargs: Optional[Mapping[str, Any]] = None,
    patient_comparison_kwargs: Optional[Mapping[str, Any]] = None,
    cohort_shap_kwargs: Optional[Mapping[str, Any]] = None,
    progress_kwargs: Optional[Mapping[str, Any]] = None,
):
    """
    Run the enrichment/explanation processing workflow after external validation.

    This pipeline starts from the main outputs of
    `run_external_validation_pipeline(...)`, especially:

        all_results
        df_agg

    It does not create plots. Instead, it prepares the tables/dictionaries used
    by downstream visualization functions, including patient screening plots,
    selected-patient model comparisons, patient-level SHAP waterfalls, and
    cohort-level SHAP contribution plots.

    Processing steps
    ----------------
    1. Build patient enrichment tables
       Uses `build_patient_enrichment_table(...)`.

    2. Select patients for explanation
       Uses `select_enrichment_patients_for_explanation(...)`.

    3. Build selected-patient model comparison table
       Uses `build_selected_patient_model_comparison_table(...)`.

    4. Build cohort-level SHAP summary table
       Uses `build_cohort_shap_summary_table(...)`.

    Parameters
    ----------
    all_results : dict
        Nested model-results dictionary after external predictions and SHAP
        summaries have been added.

        Expected when `cohort_shap_kwargs["enabled"]` is True:
            all_results[summary_key][model_name]

        where `summary_key` is usually "external_shap_summary".

    df_agg : pandas.DataFrame
        Aggregated patient-level prediction table, usually from
        `aggregate_predictions_by_idx(...)`.

        Expected columns for enrichment:
            - "model"
            - "calibration"
            - "idx"
            - score column such as "p_mean"

    enrichment_kwargs : dict, optional
        Parameters passed to `build_patient_enrichment_table(...)`.

        This step is required because all downstream steps depend on the
        generated `enrichment_tables`.

        Available keys:
            models : str or sequence of str
                Model or models to build enrichment tables for.

            calibration : str, default "beta"
                Calibration variant to use.

            score_col : str, default "p_mean"
                Prediction score used for ranking and thresholding.

            uncertainty_col : str or None, default "p_std"
                Optional uncertainty column to carry into the enrichment tables.

            cutoff : float, default 0.70
                Enrichment threshold.

            positive_rule : {"gt", "ge"}, default "gt"
                Rule used to define selected patients.

                "gt":
                    selected_for_enrichment = score_col > cutoff

                "ge":
                    selected_for_enrichment = score_col >= cutoff

            borderline_margin : float, default 0.05
                Margin around the cutoff used to label borderline patients.
                This does not affect selection.

            model_alias : mapping or None, default None
                Optional mapping from model key to display label.

            sort_descending : bool, default True
                If True, rank patients from highest score to lowest score.

            return_dict : bool, default True
                If True, return a dictionary keyed by model name.

            patient_idx_col : str, default "patient_idx"
                Output column name for the patient identifier. The source
                column in `df_agg` is still expected to be "idx".

    explanation_patient_kwargs : dict, optional
        Parameters passed to `select_enrichment_patients_for_explanation(...)`,
        plus one pipeline-only key.

        Add `"enabled": False` to skip this step.

        Available keys:
            enabled : bool, default True
                Pipeline-only key. If False, skip patient selection.

            waterfall_reference_model : str, default "Ensemble model"
                Pipeline-only key. Controls which selected-patient table is
                converted into `outputs["waterfall_patient_ids"]`.

                This key is not passed to
                `select_enrichment_patients_for_explanation(...)`.

            manual_patient_ids : sequence[int], mapping[str, sequence[int]], or None, default None
                Manual patient selection.

                If None, representative patients are selected automatically.

                If a sequence, the same patient IDs are used for every model.

                If a mapping, model-specific patient IDs are used.

            representative_types : sequence[str] or None, default None
                Representative example types to select when
                `manual_patient_ids=None`.

                If None, defaults to:
                    [
                        "top_selected",
                        "borderline_selected",
                        "borderline_not_selected",
                        "lowest_not_selected",
                    ]

                Supported values:
                    "top_selected"
                        Highest-scoring selected patient(s).

                    "borderline_selected"
                        Selected patient(s) closest to the cutoff among
                        patients labeled "selected_borderline".

                    "borderline_not_selected"
                        Not-selected patient(s) closest to the cutoff among
                        patients labeled "not_selected_borderline".

                    "lowest_not_selected"
                        Lowest-scoring not-selected patient(s).

            n_per_type : int or mapping[str, int], default 1
                Number of patients to select per representative type.

                If an int, the same number is used for every representative
                type. If a mapping, type-specific counts are used.

            patient_idx_col : str, default "patient_idx"
                Column containing the patient identifier.

            score_col : str, default "p_mean"
                Score column used for ranking patients.

            selected_col : str, default "selected_for_enrichment"
                Boolean column indicating whether a patient was selected.

            selection_group_col : str, default "selection_group"
                Column containing enrichment group labels.

            distance_col : str, default "abs_distance_to_cutoff"
                Column containing absolute distance from the enrichment cutoff.

            allow_missing : bool, default True
                If True, skip missing patient IDs or unavailable representative
                types and record them in the log. If False, raise an error.

            return_log : bool, default True
                If True, return both selected patients and a selection log.

    patient_comparison_kwargs : dict, optional
        Parameters passed to `build_selected_patient_model_comparison_table(...)`.

        Add `"enabled": False` to skip this step.

        Available keys:
            enabled : bool, default True
                Pipeline-only key. If False, skip model comparison.

            reference_model : str, default "Ensemble model"
                Model whose selected patients define the comparison cohort.

            models : sequence of str or None, default None
                Models to compare. If None, compare all models present in
                `enrichment_tables`.

            patient_idx_col : str, default "patient_idx"
                Patient identifier column.

            score_col : str, default "p_mean"
                Prediction score column to compare.

            uncertainty_col : str or None, default "p_std"
                Optional uncertainty column to include.

            return_format : {"long", "wide", "both"}, default "long"
                Format of the returned comparison table.

                "long":
                    One row per reference patient x compared model.

                "wide":
                    One row per reference patient, with model-specific columns.

                "both":
                    Return {"long": long_df, "wide": wide_df}.

    cohort_shap_kwargs : dict, optional
        Parameters passed to `build_cohort_shap_summary_table(...)`.

        Add `"enabled": False` to skip this step.

        Available keys:
            enabled : bool, default True
                Pipeline-only key. If False, skip cohort SHAP summary.

            reference_selection_model : str, default "Ensemble model"
                Model whose enrichment table defines the selected/not-selected
                groups.

            shap_models : str, sequence of str, or None, default None
                Model(s) whose SHAP summaries should be analyzed. If None, all
                models present under `all_results[summary_key]` are used.

            summary_key : str, default "external_shap_summary"
                Top-level key in `all_results` containing mean SHAP summaries.

            patient_idx_col : str, default "patient_idx"
                Patient identifier column in the enrichment table.

            selected_col : str, default "selected_for_enrichment"
                Boolean column defining selected vs not-selected patients.

            selection_group_col : str, default "selection_group"
                Optional enrichment group column. Used only for cohort counts
                if present.

            score_col : str, default "p_mean"
                Optional prediction score column from the enrichment table.

            balance_method : {None, "downsample_not_selected", "downsample_larger_group"}, default "downsample_larger_group"
                Optional bootstrap strategy to address selected/not-selected
                group-size imbalance.

            n_bootstrap : int, default 1000
                Number of bootstrap/downsampling iterations.

            ci : float, default 0.95
                Confidence interval width for bootstrap summaries.

            random_state : int, default 42
                Random seed for bootstrap/downsampling.

            min_group_size : int, default 2
                Minimum selected and not-selected group size required to compute
                bootstrap intervals.

    progress_kwargs : dict, optional
        Controls progress logging.

        Available keys:
            enabled : bool, default True
                Print pipeline progress messages.

            show_output_shapes : bool, default True
                Include compact output summaries such as DataFrame shapes and
                dictionary keys in the progress log.

            return_progress_log : bool, default True
                Store the progress log in `outputs["progress_log"]`.

    Returns
    -------
    outputs : dict
        Dictionary containing pipeline outputs.

        Main outputs:
            all_results : dict
                Same object passed into the pipeline.

            df_agg : pandas.DataFrame
                Same aggregated prediction table passed into the pipeline.

            enrichment_tables : dict[str, pandas.DataFrame] or pandas.DataFrame
                Output from `build_patient_enrichment_table(...)`.

            selected_patients : dict[str, pandas.DataFrame] or None
                Output from `select_enrichment_patients_for_explanation(...)`.

            waterfall_patient_ids : list or None
                Ready-to-use patient IDs for `plot_shap_style_waterfall(...)`.
                This is built from `selected_patients[waterfall_reference_model]`.

            waterfall_reference_model : str or None
                Model key used to create `waterfall_patient_ids`.

            selection_log : pandas.DataFrame or None
                Selection log returned by
                `select_enrichment_patients_for_explanation(...)`, if enabled.

            patient_comparison : pandas.DataFrame or dict[str, pandas.DataFrame] or None
                Output from `build_selected_patient_model_comparison_table(...)`.

            comparison_long : pandas.DataFrame or None
                Convenience output when `patient_comparison` is a dictionary
                containing a "long" table.

            comparison_wide : pandas.DataFrame or None
                Convenience output when `patient_comparison` is a dictionary
                containing a "wide" table.

            cohort_shap_summary : pandas.DataFrame or None
                Output from `build_cohort_shap_summary_table(...)`.

            progress_log : list[dict], optional
                Returned when `progress_kwargs["return_progress_log"]` is True.

    Notes
    -----
    This pipeline intentionally does not call plotting functions. Use its
    outputs with standalone plotting functions such as:

        plot_screening_predictions(...)
        plot_selected_patient_screening_comparison(...)
        plot_shap_style_waterfall(...)
        plot_cohort_shap_contributions(...)

    A typical workflow is:

        validation_outputs = run_external_validation_pipeline(...)
        explanation_outputs = run_enrichment_explanation_pipeline(...)

    Then use:

        explanation_outputs["waterfall_patient_ids"]

    directly as `patient_idx` in `plot_shap_style_waterfall(...)`.
    """

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    if enrichment_kwargs is None:
        enrichment_kwargs = {}

    if explanation_patient_kwargs is None:
        explanation_patient_kwargs = {}

    if patient_comparison_kwargs is None:
        patient_comparison_kwargs = {}

    if cohort_shap_kwargs is None:
        cohort_shap_kwargs = {}

    if progress_kwargs is None:
        progress_kwargs = {}

    progress_enabled = progress_kwargs.get("enabled", True)
    show_output_shapes = progress_kwargs.get("show_output_shapes", True)
    return_progress_log = progress_kwargs.get("return_progress_log", True)

    progress_log = []

    outputs = {
        "all_results": all_results,
        "df_agg": df_agg,

        "enrichment_tables": None,

        "selected_patients": None,
        "waterfall_patient_ids": None,
        "waterfall_reference_model": None,
        "selection_log": None,

        "patient_comparison": None,
        "comparison_long": None,
        "comparison_wide": None,

        "cohort_shap_summary": None,
    }

    # ------------------------------------------------------------------
    # Small progress helpers
    # ------------------------------------------------------------------
    def _describe_object(obj):
        if not show_output_shapes:
            return ""

        if obj is None:
            return "None"

        if isinstance(obj, pd.DataFrame):
            return f"DataFrame shape={obj.shape}"

        if isinstance(obj, dict):
            return f"dict keys={list(obj.keys())}"

        if isinstance(obj, (list, tuple)):
            return f"{type(obj).__name__} len={len(obj)}"

        return type(obj).__name__

    def _start_step(name):
        if progress_enabled:
            print(f">> {name}")

    def _ok_step(name, obj=None):
        desc = _describe_object(obj)

        message = f"[OK] {name}"
        if desc:
            message += f" -> {desc}"

        progress_log.append(
            {
                "step": name,
                "status": "ok",
                "detail": desc,
            }
        )

        if progress_enabled:
            print(message)

    def _skip_step(name, reason):
        progress_log.append(
            {
                "step": name,
                "status": "skipped",
                "detail": reason,
            }
        )

        if progress_enabled:
            print(f"[SKIP] {name} -> {reason}")

    def _fail_step(name, err):
        progress_log.append(
            {
                "step": name,
                "status": "fail",
                "detail": str(err),
            }
        )

        if progress_enabled:
            print(f"[FAIL] {name} -> {err}")

    if progress_enabled:
        print("Enrichment explanation pipeline")
        print("------------------------------------")

    # ------------------------------------------------------------------
    # Step 1: Build enrichment tables
    # ------------------------------------------------------------------
    step_name = "Build patient enrichment tables"

    try:
        _start_step(step_name)

        enrichment_kwargs_clean = dict(enrichment_kwargs)
        enrichment_kwargs_clean.pop("enabled", None)

        enrichment_tables = build_patient_enrichment_table(
            df_pred=df_agg,
            **enrichment_kwargs_clean,
        )

        outputs["enrichment_tables"] = enrichment_tables

        _ok_step(step_name, enrichment_tables)

    except Exception as err:
        _fail_step(step_name, err)
        raise

    # ------------------------------------------------------------------
    # Step 2: Select patients for explanation
    # ------------------------------------------------------------------
    step_name = "Select patients for explanation"

    explanation_enabled = explanation_patient_kwargs.get("enabled", True)

    if not explanation_enabled:
        _skip_step(step_name, "explanation_patient_kwargs['enabled'] is False")

    else:
        try:
            _start_step(step_name)

            explanation_patient_kwargs_clean = dict(explanation_patient_kwargs)
            explanation_patient_kwargs_clean.pop("enabled", None)

            # This key is for the pipeline only.
            # It decides which selected-patient table will be converted into
            # outputs["waterfall_patient_ids"].
            waterfall_reference_model = explanation_patient_kwargs_clean.pop(
                "waterfall_reference_model",
                "Ensemble model",
            )

            selection_output = select_enrichment_patients_for_explanation(
                enrichment_tables,
                **explanation_patient_kwargs_clean,
            )

            if (
                isinstance(selection_output, tuple)
                and len(selection_output) == 2
            ):
                selected_patients, selection_log = selection_output
            else:
                selected_patients = selection_output
                selection_log = None

            outputs["selected_patients"] = selected_patients
            outputs["selection_log"] = selection_log
            outputs["waterfall_reference_model"] = waterfall_reference_model

            # ----------------------------------------------------------
            # Ready-to-use patient IDs for plot_shap_style_waterfall(...)
            # ----------------------------------------------------------
            if isinstance(selected_patients, dict):
                if waterfall_reference_model not in selected_patients:
                    raise KeyError(
                        f"waterfall_reference_model={waterfall_reference_model!r} "
                        f"not found in selected_patients. Available keys: "
                        f"{list(selected_patients.keys())}"
                    )

                waterfall_table = selected_patients[waterfall_reference_model]

                if not isinstance(waterfall_table, pd.DataFrame):
                    raise TypeError(
                        "selected_patients[waterfall_reference_model] must be "
                        "a pandas DataFrame."
                    )

                if "patient_idx" not in waterfall_table.columns:
                    raise KeyError(
                        "The selected patient table used for waterfall IDs "
                        "must contain a 'patient_idx' column."
                    )

                outputs["waterfall_patient_ids"] = (
                    waterfall_table["patient_idx"]
                    .tolist()
                )

            elif isinstance(selected_patients, pd.DataFrame):
                if "patient_idx" not in selected_patients.columns:
                    raise KeyError(
                        "selected_patients DataFrame must contain a "
                        "'patient_idx' column."
                    )

                outputs["waterfall_patient_ids"] = (
                    selected_patients["patient_idx"]
                    .tolist()
                )

            else:
                outputs["waterfall_patient_ids"] = None

            _ok_step(step_name, selected_patients)
            _ok_step(
                "Prepare waterfall patient IDs",
                outputs["waterfall_patient_ids"],
            )

        except Exception as err:
            _fail_step(step_name, err)
            raise

    # ------------------------------------------------------------------
    # Step 3: Build selected-patient model comparison table
    # ------------------------------------------------------------------
    step_name = "Build selected-patient model comparison"

    comparison_enabled = patient_comparison_kwargs.get("enabled", True)

    if not comparison_enabled:
        _skip_step(step_name, "patient_comparison_kwargs['enabled'] is False")

    elif outputs["selected_patients"] is None:
        _skip_step(step_name, "selected_patients is None")

    else:
        try:
            _start_step(step_name)

            patient_comparison_kwargs_clean = dict(patient_comparison_kwargs)
            patient_comparison_kwargs_clean.pop("enabled", None)

            patient_comparison = build_selected_patient_model_comparison_table(
                enrichment_tables=enrichment_tables,
                selected_patients=outputs["selected_patients"],
                **patient_comparison_kwargs_clean,
            )

            outputs["patient_comparison"] = patient_comparison

            if isinstance(patient_comparison, dict):
                outputs["comparison_long"] = patient_comparison.get("long")
                outputs["comparison_wide"] = patient_comparison.get("wide")

            _ok_step(step_name, patient_comparison)

        except Exception as err:
            _fail_step(step_name, err)
            raise

    # ------------------------------------------------------------------
    # Step 4: Build cohort SHAP summary table
    # ------------------------------------------------------------------
    step_name = "Build cohort SHAP summary"

    cohort_shap_enabled = cohort_shap_kwargs.get("enabled", True)

    if not cohort_shap_enabled:
        _skip_step(step_name, "cohort_shap_kwargs['enabled'] is False")

    else:
        try:
            _start_step(step_name)

            cohort_shap_kwargs_clean = dict(cohort_shap_kwargs)
            cohort_shap_kwargs_clean.pop("enabled", None)

            cohort_shap_summary = build_cohort_shap_summary_table(
                all_results=all_results,
                enrichment_tables=enrichment_tables,
                **cohort_shap_kwargs_clean,
            )

            outputs["cohort_shap_summary"] = cohort_shap_summary

            _ok_step(step_name, cohort_shap_summary)

        except Exception as err:
            _fail_step(step_name, err)
            raise

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    if return_progress_log:
        outputs["progress_log"] = progress_log

    if progress_enabled:
        print("------------------------------------")
        print("[OK] Pipeline complete")

    return outputs




















# ---------------------------------------------------------------------
# Old Code without the pipeline of run_external_validation_pipeline()
# ---------------------------------------------------------------------
# # Load train data
# X_train= pd.DataFrame(train_bundle['X_scaled'], columns=train_bundle['feature_names'])
# y_train = pd.DataFrame(train_bundle['y'], columns=['target'])
# train_data  = pd.concat([X_train, y_train], axis=1)



# # Load validation data
# X_validation = pd.DataFrame(validation_bundle['X_scaled'], columns=validation_bundle['feature_names'])
# y_validation = pd.DataFrame(validation_bundle['y'], columns=['target'])
# validation_df = pd.concat([X_validation, y_validation], axis=1)


# # Features used for model development: Select top k features
# top_k = 5
# lr_top = all_results['logistic_regression'][0]['feature_names_used']
# xgb_top  = all_results['xgboost'][0]['feature_names_used']


# test_data = {}
# test_data['logistic_regression'] = validation_df[lr_top+['target']]
# test_data['xgboost']             = validation_df[xgb_top+['target']]




# # Generate external-set predictions for every fold model in `all_results` and store them back into each fold record using `*_external_*` keys (metrics only if labels exist).
# all_results = vld.add_external_predictions_to_results(
#     all_results,
#     model_data_dict=test_data,
#     y_col='target',   # or None if unlabeled, or just add the label column if present
#     external_tag="external",
#     feature_names_key="feature_names_used",
#     strict_features=True,
# )


# # Build a single long-form  predictions table with one row per predicted example that is pooled across models. 
# df_long = vld.build_long_predictions_df(
#     all_results,
#     model_name=None,
#     methods=["beta"],
#     include_uncalibrated=False,
# )

# # Aggregate repeated predictions per idx into a single row per (model, variant, idx), and optionally add an "ensemble" model by pooling predictions across multiple models.
# df_agg = vld.aggregate_predictions_by_idx(
#     df_long,
#     calibrations=["uncalib", "beta"],
#     add_ensemble=True,
#     ensemble_name="Ensemble model",
#     ensemble_models=None, #["logistic_regression", "xgboost"],  # optional; otherwise pools all selected
#     truncate_decimals=4,
# )

# all_results = vld.add_external_shap_to_results(
#     all_results,
#     model_data_dict=test_data,
#     train_data=train_data,
#     y_col="target",
#     external_tag="external",
#     strict_features=True,
#     max_background=None,
#     random_state=42,
# )

# all_results = vld.add_external_shap_summary_to_results(
#     all_results,
#     external_tag="external",
# )



# enrichment_tables = build_patient_enrichment_table(
#     df_pred=df_agg,
#     models=["logistic_regression", "xgboost", "Ensemble model"],
#     calibration="beta",
#     score_col="p_mean",
#     uncertainty_col="p_std",
#     cutoff=0.70,
#     positive_rule="gt",
#     borderline_margin=0.05,
#     model_alias={
#         "logistic_regression": "Logistic regression",
#         "xgboost": "XGBoost",
#         "Ensemble model": "Ensemble model",
#     },
#     return_dict=True,
# )



# selected_patients, selection_log = select_enrichment_patients_for_explanation(
#     enrichment_tables,
#     manual_patient_ids=None,
#     representative_types=[
#         "top_selected",
#         "borderline_selected",
#         "borderline_not_selected",
#         "lowest_not_selected",
#     ],
#     n_per_type=5,
#     patient_idx_col="patient_idx",
#     score_col="p_mean",
#     allow_missing=True,
#     return_log=True,
# )


# ensemble_patient_comparison = build_selected_patient_model_comparison_table(
#     enrichment_tables=enrichment_tables,
#     selected_patients=selected_patients,
#     reference_model="Ensemble model",
#     models=["logistic_regression", "xgboost", "Ensemble model"],
#     patient_idx_col="patient_idx",
#     score_col="p_mean",
#     uncertainty_col="p_std",
#     return_format="both",
# )

# comparison_long = ensemble_patient_comparison["long"]
# comparison_wide = ensemble_patient_comparison["wide"]


# cohort_shap_summary = build_cohort_shap_summary_table(
#     all_results=all_results,
#     enrichment_tables=enrichment_tables,
#     reference_selection_model="Ensemble model",
#     shap_models=["logistic_regression", "xgboost"],
#     summary_key="external_shap_summary",
#     patient_idx_col="patient_idx",
#     selected_col="selected_for_enrichment",
#     balance_method="downsample_larger_group",
#     n_bootstrap=1000,
#     ci=0.95,
#     random_state=42,
# )


