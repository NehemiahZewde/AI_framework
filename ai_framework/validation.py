# validation .py
# ML external validation datase

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Type, Mapping, Literal, Callable

import numpy as np
import pandas as pd



from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_validate
from sklearn.model_selection._split import BaseCrossValidator  # for typing
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt


from tqdm.auto import trange
from tqdm.auto import tqdm




from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve, log_loss, brier_score_loss
)


def check_bundle_alignment_for_preprocessing(
    bundle_orig: Dict[str, Any],
    bundle: Dict[str, Any],
    preproc_key: str = "preproc",
    require_scaler: bool = True,
) -> Dict[str, Any]:
    """
    Check whether a new bundle can safely reuse fitted preprocessing artifacts
    from an original/reference bundle.

    This function is intended to validate alignment before applying preprocessing
    learned on `bundle_orig` to `bundle`. In particular, it checks that the new
    bundle has the expected raw feature matrix, feature names, and feature order,
    and that the original bundle contains the fitted preprocessing objects stored
    under `bundle_orig[preproc_key]`.

    The function does NOT transform any data. It only performs validation and
    returns a structured PASS/FAIL result.

    Parameters
    ----------
    bundle_orig : dict
        Reference bundle that already contains fitted preprocessing artifacts.
        Expected to contain:
          - "X_raw": np.ndarray of shape (n_samples, n_features)
          - "feature_names": list[str]
          - preproc_key (default "preproc"), containing fitted preprocessing info

    bundle : dict
        New bundle to validate against `bundle_orig` before applying the fitted
        preprocessing artifacts. Expected to contain:
          - "X_raw": np.ndarray of shape (n_samples, n_features)
          - "feature_names": list[str]

    preproc_key : str, default "preproc"
        Key in `bundle_orig` where fitted preprocessing artifacts are stored.

    require_scaler : bool, default True
        If True, require that a fitted scaler exists in
        `bundle_orig[preproc_key]["scaler"]`. Set to False if you want to validate
        compatibility for a preprocessing flow that does not require scaling.

    Returns
    -------
    result : dict
        Dictionary with the following keys:
          - "status": str
              "PASS" if all required checks succeed, otherwise "FAIL".
          - "errors": list[str]
              Validation failures that should block preprocessing transfer.
          - "warnings": list[str]
              Non-fatal issues or soft consistency concerns.

    Checks performed
    ----------------
    The function may check for:
      - required top-level keys in both bundles
      - presence of fitted preprocessing artifacts in `bundle_orig[preproc_key]`
      - consistency between `X_raw.shape[1]` and `len(feature_names)`
      - exact feature-name alignment between the fitted preprocessing space and
        the new bundle
      - exact feature ordering
      - basic consistency of stored objects such as `caps_df`, `imputer`,
        and optionally `scaler`

    Notes
    -----
    - Exact feature order matters, not just feature-name membership. Even if two
      bundles contain the same feature names, preprocessing transfer should fail
      if the column order differs.
    - This function is designed as Part 1 of a safer preprocessing-transfer
      workflow:
          1) validate alignment
          2) apply fitted preprocessing artifacts
          3) generate the transformed output (e.g. `bundle["X_scaled"]`)
    """
    errors = []
    warnings = []

    # --- Required top-level keys ---
    for name, obj in [("bundle_orig", bundle_orig), ("bundle", bundle)]:
        if "X_raw" not in obj:
            errors.append(f"{name} is missing required key: 'X_raw'")
        if "feature_names" not in obj:
            errors.append(f"{name} is missing required key: 'feature_names'")

    if preproc_key not in bundle_orig:
        errors.append(f"bundle_orig is missing required key: '{preproc_key}'")

    if errors:
        return {"status": "FAIL", "errors": errors, "warnings": warnings}

    X_orig = bundle_orig["X_raw"]
    X_new = bundle["X_raw"]
    feature_names_orig = list(bundle_orig["feature_names"])
    feature_names_new = list(bundle["feature_names"])
    preproc = bundle_orig[preproc_key]

    # --- Shape checks ---
    if X_orig.shape[1] != len(feature_names_orig):
        errors.append(
            "bundle_orig mismatch: X_raw.shape[1] != len(feature_names)"
        )

    if X_new.shape[1] != len(feature_names_new):
        errors.append(
            "bundle mismatch: X_raw.shape[1] != len(feature_names)"
        )

    # --- Fitted preproc checks ---
    if "feature_names" not in preproc:
        errors.append(f"bundle_orig['{preproc_key}'] is missing 'feature_names'")
    if "caps_df" not in preproc:
        errors.append(f"bundle_orig['{preproc_key}'] is missing 'caps_df'")
    if "imputer" not in preproc:
        errors.append(f"bundle_orig['{preproc_key}'] is missing 'imputer'")

    if require_scaler and "scaler" not in preproc:
        errors.append(f"bundle_orig['{preproc_key}'] is missing 'scaler'")

    if errors:
        return {"status": "FAIL", "errors": errors, "warnings": warnings}

    fitted_feature_names = list(preproc["feature_names"])

    # --- Feature count checks ---
    if len(feature_names_new) != len(fitted_feature_names):
        errors.append(
            f"Feature count mismatch: new bundle has {len(feature_names_new)} "
            f"features but fitted preproc expects {len(fitted_feature_names)}"
        )

    # --- Exact feature order check ---
    if feature_names_new != fitted_feature_names:
        if set(feature_names_new) == set(fitted_feature_names):
            errors.append(
                "Feature names match as a set, but column order differs. "
                "Bundles are not aligned."
            )
        else:
            missing_in_new = [f for f in fitted_feature_names if f not in feature_names_new]
            extra_in_new = [f for f in feature_names_new if f not in fitted_feature_names]

            if missing_in_new:
                errors.append(
                    f"New bundle is missing fitted features: {missing_in_new[:10]}"
                    + (" ..." if len(missing_in_new) > 10 else "")
                )
            if extra_in_new:
                errors.append(
                    f"New bundle has extra features not seen in fitted bundle: {extra_in_new[:10]}"
                    + (" ..." if len(extra_in_new) > 10 else "")
                )

    # --- caps_df alignment check ---
    caps_df = preproc["caps_df"]
    if list(caps_df.index) != fitted_feature_names:
        errors.append(
            f"bundle_orig['{preproc_key}']['caps_df'] index does not match fitted feature_names"
        )

    # --- scaler sanity check ---
    scaler = preproc.get("scaler", None)
    if require_scaler:
        if scaler is None:
            errors.append("Scaler is None, but require_scaler=True")
        else:
            if hasattr(scaler, "n_features_in_"):
                if scaler.n_features_in_ != len(fitted_feature_names):
                    errors.append(
                        f"Scaler expects {scaler.n_features_in_} features, "
                        f"but fitted feature_names has {len(fitted_feature_names)}"
                    )

    # --- imputer sanity check ---
    imputer = preproc.get("imputer", None)
    if imputer is None:
        errors.append("Imputer is None")
    else:
        if hasattr(imputer, "n_features_in_"):
            skipped = preproc.get("skipped_feature_names", [])
            expected_imputer_features = len(fitted_feature_names) - len(skipped)
            if imputer.n_features_in_ != expected_imputer_features:
                warnings.append(
                    f"Imputer expects {imputer.n_features_in_} features; "
                    f"expected approximately {expected_imputer_features} based on skipped features"
                )

    status = "PASS" if not errors else "FAIL"
    return {"status": status, "errors": errors, "warnings": warnings}


def apply_preprocessing_from_bundle(
    bundle_orig: Dict[str, Any],
    bundle: Dict[str, Any],
    preproc_key: str = "preproc",
) -> Dict[str, Any]:
    """
    Apply fitted preprocessing artifacts from a reference bundle to a new bundle.

    This function reuses preprocessing objects already fit on `bundle_orig`
    (for example capping thresholds, imputers, and scaler) and applies them
    to `bundle` without refitting. It assumes the two bundles have already
    passed an alignment check and share the same feature space and column order.

    Parameters
    ----------
    bundle_orig : dict
        Reference/original bundle containing fitted preprocessing artifacts
        under `bundle_orig[preproc_key]`.

    bundle : dict
        New bundle whose raw features (`bundle["X_raw"]`) will be transformed
        using the fitted preprocessing from `bundle_orig`.

    preproc_key : str, default "preproc"
        Key in `bundle_orig` where the fitted preprocessing artifacts are stored.

    Returns
    -------
    bundle : dict
        The input `bundle`, updated in-place with:
          - `bundle["X_scaled"]`: transformed feature matrix
          - `bundle["feature_name_to_idx"]`: feature-to-column mapping
    """
    if "X_raw" not in bundle:
        raise KeyError("bundle must contain key 'X_raw'")
    if "feature_names" not in bundle:
        raise KeyError("bundle must contain key 'feature_names'")
    if preproc_key not in bundle_orig:
        raise KeyError(f"bundle_orig must contain key '{preproc_key}'")

    preproc = bundle_orig[preproc_key]
    X_raw = np.asarray(bundle["X_raw"], dtype=np.float32)
    feature_names = list(bundle["feature_names"])
    fitted_feature_names = list(preproc["feature_names"])

    if feature_names != fitted_feature_names:
        raise ValueError(
            "Feature names/order mismatch between bundle and fitted preprocessing."
        )

    bundle["feature_name_to_idx"] = {name: i for i, name in enumerate(feature_names)}

    caps_df = preproc["caps_df"]
    imputer = preproc["imputer"]
    scaler = preproc["scaler"]
    cat_ord_imputer = preproc.get("cat_ord_imputer", None)
    skipped_feature_names = preproc.get("skipped_feature_names", [])

    skipped_set = set(skipped_feature_names)
    skip_idx = [i for i, name in enumerate(feature_names) if name in skipped_set]
    cont_idx = [i for i, name in enumerate(feature_names) if name not in skipped_set]

    # Case 1: all columns were treated as continuous during fitting
    if len(skip_idx) == 0:
        lower = caps_df.loc[feature_names, "lower"].to_numpy(dtype=np.float32)
        upper = caps_df.loc[feature_names, "upper"].to_numpy(dtype=np.float32)

        X_capped = np.clip(X_raw, lower, upper).astype(np.float32, copy=False)
        X_imputed = imputer.transform(X_capped).astype(np.float32, copy=False)
        X_scaled = scaler.transform(X_imputed).astype(np.float32, copy=False)

        bundle["X_scaled"] = X_scaled
        return bundle

    # Case 2: continuous + categorical/ordinal split
    X_out = X_raw.copy()

    # Continuous subset
    feature_names_cont = [feature_names[i] for i in cont_idx]
    X_cont = X_raw[:, cont_idx]

    lower = caps_df.loc[feature_names_cont, "lower"].to_numpy(dtype=np.float32)
    upper = caps_df.loc[feature_names_cont, "upper"].to_numpy(dtype=np.float32)

    X_cont_capped = np.clip(X_cont, lower, upper).astype(np.float32, copy=False)
    X_cont_imputed = imputer.transform(X_cont_capped).astype(np.float32, copy=False)
    X_cont_scaled = scaler.transform(X_cont_imputed).astype(np.float32, copy=False)

    X_out[:, cont_idx] = X_cont_scaled

    # Categorical / ordinal subset
    if len(skip_idx) > 0 and cat_ord_imputer is not None:
        X_cat = X_out[:, skip_idx]
        X_cat_imputed = cat_ord_imputer.transform(X_cat).astype(np.float32, copy=False)
        X_out[:, skip_idx] = X_cat_imputed

    bundle["X_scaled"] = X_out.astype(np.float32, copy=False)
    return bundle


def preprocessing_transfer_pipeline(
    bundle_orig: Dict[str, Any],
    bundle: Dict[str, Any],
    preproc_key: str = "preproc",
    require_scaler: bool = True,
) -> Dict[str, Any]:
    """
    Validate alignment between a reference bundle and a new bundle, then apply
    the fitted preprocessing artifacts from the reference bundle to the new bundle.

    This function is a wrapper around:
      1) `check_bundle_alignment_for_preprocessing(...)`
      2) `apply_preprocessing_from_bundle(...)`

    It first checks whether `bundle` is compatible with the preprocessing
    artifacts already fit on `bundle_orig`. If validation passes, it applies
    those fitted preprocessing objects to `bundle` without refitting.

    Parameters
    ----------
    bundle_orig : dict
        Reference/original bundle containing fitted preprocessing artifacts
        under `bundle_orig[preproc_key]`.

    bundle : dict
        New bundle whose raw features (`bundle["X_raw"]`) will be transformed
        using the fitted preprocessing from `bundle_orig`.

    preproc_key : str, default "preproc"
        Key in `bundle_orig` where fitted preprocessing artifacts are stored.

    require_scaler : bool, default True
        If True, require that a fitted scaler exists during validation.
        Set to False if using a preprocessing flow that does not require scaling.

    Returns
    -------
    bundle : dict
        The input `bundle`, updated in-place with transformed features
        (typically `bundle["X_scaled"]`) if validation succeeds.

    Raises
    ------
    ValueError
        If bundle alignment validation fails.
    """
    check = check_bundle_alignment_for_preprocessing(
        bundle_orig=bundle_orig,
        bundle=bundle,
        preproc_key=preproc_key,
        require_scaler=require_scaler,
    )

    if check["status"] != "PASS":
        error_msg = "Bundle alignment check failed:\n- " + "\n- ".join(check["errors"])
        if check["warnings"]:
            error_msg += "\nWarnings:\n- " + "\n- ".join(check["warnings"])
        raise ValueError(error_msg)

    bundle = apply_preprocessing_from_bundle(
        bundle_orig=bundle_orig,
        bundle=bundle,
        preproc_key=preproc_key,
    )

    return bundle


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
    # ---------------------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------------------
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
    # This is useful because repeated predictions for the same idx should all
    # correspond to the same ground-truth label when labels are available.
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



