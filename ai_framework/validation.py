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
    external_bundle: Dict[str, Any],
    *,
    x_key: str = "combined_X_raw",
    y_key: str = "combined_y",
    external_tag: str = "external",
    strict_features: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate external-set predictions for every fold model in `all_results` and store them
    back into each fold record using `*_external_*` keys (metrics only if labels exist).

    Parameters
    ----------
    all_results:
        Dict mapping model_name -> list of fold-record dicts. Each record must contain
        `final_model` and should contain `calib_feature_names` (used to select columns).
        If present, `calibrator_platt` and/or `calibrator_beta` are used to produce
        calibrated external predictions.
    external_bundle:
        Dataset bundle for the external set. Must contain `feature_names` and `x_key`
        (e.g., 'combined_X_raw'). If labels are available, include `y_key`
        (e.g., 'combined_y') to compute external metrics.
    x_key:
        Key in `external_bundle` pointing to the feature matrix to score (default:
        'combined_X_raw', i.e., aggregated-by-group features).
    y_key:
        Optional key in `external_bundle` pointing to labels. If missing or None, metrics
        are not computed (prediction-only mode).
    external_tag:
        Tag used in the keys written into each record (default 'external'), e.g.
        `y_external_scores`, `calib_external_predictions_platt`, `external_metrics`.
    strict_features:
        If True, require that each fold record has a feature list (`calib_feature_names`
        or `feature_names`). If False, fall back to using all external features.

    Returns
    -------
    all_results:
        The same object, modified in-place (returned for convenience).
    """
    if "feature_names" not in external_bundle:
        raise KeyError("external_bundle must contain 'feature_names'.")

    if x_key not in external_bundle:
        raise KeyError(
            f"external_bundle missing x_key='{x_key}'. Available keys: {list(external_bundle.keys())}"
        )

    X_full = np.asarray(external_bundle[x_key])
    if X_full.ndim != 2:
        raise ValueError(f"external_bundle[{x_key}] must be 2D, got shape {X_full.shape}")

    feat_names_full = list(external_bundle["feature_names"])
    if len(feat_names_full) != X_full.shape[1]:
        raise ValueError(
            f"Mismatch: external X has {X_full.shape[1]} cols but feature_names has {len(feat_names_full)}"
        )

    # Labels are optional
    y_ext = external_bundle.get(y_key, None)
    has_labels = y_ext is not None
    if has_labels:
        y_ext = np.asarray(y_ext)
        if y_ext.ndim != 1 or len(y_ext) != X_full.shape[0]:
            raise ValueError(
                f"external y must be 1D and aligned with X rows. Got y shape {y_ext.shape}, X rows {X_full.shape[0]}"
            )

    # Map feature name -> column index for selection
    col_index = {name: i for i, name in enumerate(feat_names_full)}

    for model_name, fold_records in all_results.items():
        for rec in fold_records:
            if "final_model" not in rec:
                raise KeyError(f"{model_name} record missing 'final_model'.")

            # ---- Determine which features to use ----
            selected_feature_names = rec.get("calib_feature_names", None)
            if selected_feature_names is None:
                selected_feature_names = rec.get("feature_names", None)

            if selected_feature_names is None:
                if strict_features:
                    raise KeyError(
                        f"{model_name} fold record is missing 'calib_feature_names' (and 'feature_names'). "
                        "Store it during training or set strict_features=False to use all features."
                    )
                selected_feature_names = feat_names_full

            selected_feature_names = list(selected_feature_names)



            missing = [f for f in selected_feature_names if f not in col_index]
            if missing:
                raise KeyError(f"External bundle missing required features for {model_name}: {missing}")

            cols = [col_index[f] for f in selected_feature_names]
            X_ext = X_full[:, cols]

            # ---- SIMPLE SAFETY CHECK ----
            if X_ext.shape[1] != len(selected_feature_names):
                raise ValueError(
                    f"[{model_name}] Feature mismatch: X_ext has {X_ext.shape[1]} cols "
                    f"but calib_feature_names has {len(selected_feature_names)}. "
                    f"Check feature selection/mapping (e.g., wrong feature list)."
                )

            # ---- Predict (uncalibrated) ----
            final_model = rec["final_model"]
            p_ext = final_model.predict_proba(X_ext)[:, 1]

            # ---- Attach uncalibrated external outputs ----
            rec[f"{external_tag}_x_key"] = x_key
            rec[f"{external_tag}_feature_names_key"] = "feature_names"
            rec[f"{external_tag}_feature_names"] = selected_feature_names
            rec[f"n_{external_tag}"] = int(X_ext.shape[0])
            rec[f"y_{external_tag}_scores"] = p_ext  # mirrors y_test_scores

            # ---- Calibrated external predictions ----
            if rec.get("calibrator_platt", None) is not None:
                rec[f"calib_{external_tag}_predictions_platt"] = rec["calibrator_platt"].predict_proba(
                    p_ext.reshape(-1, 1)
                )[:, 1]

            if rec.get("calibrator_beta", None) is not None:
                rec[f"calib_{external_tag}_predictions_beta"] = rec["calibrator_beta"].predict(p_ext)

            # ---- Metrics only if labels exist ----
            if has_labels:
                rec[f"y_{external_tag}"] = y_ext
                rec[f"{external_tag}_metrics"] = {
                    "average_precision": float(average_precision_score(y_ext, p_ext)),
                    "roc_auc": float(roc_auc_score(y_ext, p_ext)),
                }

                if rec.get(f"calib_{external_tag}_predictions_platt", None) is not None:
                    pp = rec[f"calib_{external_tag}_predictions_platt"]
                    rec[f"{external_tag}_metrics_platt"] = {
                        "average_precision": float(average_precision_score(y_ext, pp)),
                        "roc_auc": float(roc_auc_score(y_ext, pp)),
                    }

                if rec.get(f"calib_{external_tag}_predictions_beta", None) is not None:
                    pb = rec[f"calib_{external_tag}_predictions_beta"]
                    rec[f"{external_tag}_metrics_beta"] = {
                        "average_precision": float(average_precision_score(y_ext, pb)),
                        "roc_auc": float(roc_auc_score(y_ext, pb)),
                    }

    return all_results


def build_long_predictions_df(
    all_results: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    model_name: str | Sequence[str] | None = None,
    groups_all: Optional[np.ndarray] = None,
    group_id_to_key: Optional[Mapping[int, Tuple[str, str]]] = None,  # group -> (label_str, subject_id)
    methods: Optional[Sequence[str]] = None,
    include_uncalibrated: bool = True,
    include_test: bool = True,
    include_train: bool = False,
    include_external: bool = True,
    unit_col: str = "idx",
    # --- external config ---
    external_idx_key: str | None = None,   # if you later store explicit external indices
    external_y_key: str = "y_external",
    external_prob_key_uncalib: str = "y_external_scores",
    external_prob_key_prefix_calib: str = "calib_external_predictions_",
) -> pd.DataFrame:
    """
    Build a single long-form ("tidy") predictions table with one row per predicted example,
    pooled across models, folds, splits, and probability variants (uncalibrated + calibrated).

    This function converts your nested-CV `all_results` structure:

        all_results[model_name] = [fold_dict_1, fold_dict_2, ...]

    into a single pandas DataFrame suitable for downstream aggregation (e.g., pooling by
    subject/patient), plotting, and metric computation.

    Output row granularity
    ----------------------
    Each output row corresponds to a single prediction for a single index `idx` within a given:
      - model (e.g., "logistic_regression")
      - variant: "uncalib" or a calibration method name (e.g., "beta")
      - split: "test", "train" (optional), and "external" (optional)
      - trial and outer_fold (as stored in each fold dict)

    Labels are optional for external
    --------------------------------
    External predictions may or may not have labels. If external labels are absent, the output
    still includes external rows, but sets:
      - y = np.nan
      - and (per your rule) group_label = np.nan as well (when grouping is enabled)

    Group/patient metadata (optional)
    ---------------------------------
    If BOTH `groups_all` and `group_id_to_key` are provided, the function will add patient/group
    columns by mapping:
      idx -> group_id via groups_all[idx]
      group_id -> (group_label, subject_id) via group_id_to_key[group_id]

    In this grouped mode, the output includes additional columns:
      - group (int)
      - group_label (str or np.nan)
      - subject_id (str or None)

    If grouping metadata is NOT provided, the function omits those columns and optionally
    includes a "unit id" column named by `unit_col` (default "idx") for downstream code that
    expects a unit identifier even without groups.

    What variants are included
    --------------------------
    - If include_uncalibrated=True:
        variant="uncalib" is included using:
          * test:      r["y_test_scores"]
          * train r["cv_uncalib_train_predictions"] (if include_train=True)
          * external:  r[external_prob_key_uncalib]      (if include_external=True)
    - Calibrated variants:
        If `methods` is provided, those method names are used (e.g., ["beta"]).
        If `methods` is None, methods are discovered per-model by scanning keys that start with:
          "calib_test_predictions_"
        Calibrated keys are expected to follow:
          * test:      r[f"calib_test_predictions_{method}"]
          * train: r[f"cv_calib_train_predictions_{method}"]
          * external:  r[f"{external_prob_key_prefix_calib}{method}"]

    External indexing
    -----------------
    External rows need an `idx` vector. This is determined as:
      1) If external_idx_key is not None and external_idx_key exists in the fold dict:
           idx_ex = r[external_idx_key]
      2) Else:
           idx_ex = np.arange(n_external)
           where n_external is taken from r["n_external"] if present; otherwise inferred from
           the length of r[external_prob_key_uncalib].

    Parameters
    ----------
    all_results:
        Mapping model_name -> sequence of fold dictionaries.

    model_name:
        Which models to include:
          - None (default): include all models in all_results
          - str: include one model
          - sequence[str]: include listed models

    groups_all:
        Optional array mapping dataset index -> integer group id.

    group_id_to_key:
        Optional mapping group id -> (group_label_str, subject_id_str).

    methods:
        Optional list of calibration methods (e.g., ["beta"]).
        If None, discovered per model by scanning "calib_test_predictions_*" keys.

    include_uncalibrated:
        Include the "uncalib" variant.

    include_test:
        Include outer test predictions (split="test").

    include_train:
        Include train fold predictions (split="train").

    include_external:
        Include external predictions (split="external"), with optional labels.

    unit_col:
        Column name to use as a "unit id" when grouping metadata is not provided. The column
        value will mirror idx.

    external_idx_key:
        Optional key in each fold dict containing explicit external indices.

    external_y_key:
        Key for external labels if they exist (default "y_external"). If missing, y=np.nan.

    external_prob_key_uncalib:
        Key for uncalibrated external probabilities (default "y_external_scores").

    external_prob_key_prefix_calib:
        Prefix for calibrated external probability keys (default "calib_external_predictions_").

    Returns
    -------
    pd.DataFrame
        Long-form predictions table.

        Always included columns:
          ["model", "variant", "split", "trial", "outer_fold", "idx", "y", "p"]

        If grouping info is provided:
          + ["group", "group_label", "subject_id"]

        If grouping info is NOT provided:
          + [unit_col] (mirrors idx; if unit_col == "idx" it is still present as idx)

        Notes on dtypes:
          - y is float (so missing labels can be represented as np.nan)
          - p is float

    Raises
    ------
    KeyError:
        If requested models are missing, or required keys for a requested split/variant are missing.

    ValueError:
        If idx/y/p lengths mismatch for any fold/variant/split.

    IndexError:
        If idx values are out of bounds for groups_all when grouping metadata is provided.
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
            y_arr: Optional[np.ndarray],
            p_arr: np.ndarray,
            split_name: str,
            trial: Any,
            outer_fold: Any,
            variant: str,
        ) -> None:
            idx_arr = np.asarray(idx_arr, dtype=int)
            p_arr = np.asarray(p_arr, dtype=float)

            if y_arr is not None:
                y_arr = np.asarray(y_arr)
                if len(idx_arr) != len(y_arr) or len(idx_arr) != len(p_arr):
                    raise ValueError(
                        f"Length mismatch: model={mname}, trial={trial}, outer_fold={outer_fold}, "
                        f"variant={variant}, split={split_name} "
                        f"len(idx)={len(idx_arr)}, len(y)={len(y_arr)}, len(p)={len(p_arr)}"
                    )
            else:
                if len(idx_arr) != len(p_arr):
                    raise ValueError(
                        f"Length mismatch: model={mname}, trial={trial}, outer_fold={outer_fold}, "
                        f"variant={variant}, split={split_name} "
                        f"len(idx)={len(idx_arr)}, len(p)={len(p_arr)} (y is missing)"
                    )

            if have_groups:
                assert groups_all is not None and group_id_to_key is not None

                if idx_arr.max(initial=-1) >= len(groups_all) or idx_arr.min(initial=0) < 0:
                    raise IndexError(
                        f"Some idx values are out of bounds for groups_all (len={len(groups_all)}). "
                        f"idx range: [{idx_arr.min()}, {idx_arr.max()}]"
                    )

                g_arr = groups_all[idx_arr]

                label_strs: List[Optional[str]] = []
                subject_ids: List[Optional[str]] = []
                for g in g_arr:
                    lab, sid = group_id_to_key.get(int(g), (None, None))
                    label_strs.append(lab)
                    subject_ids.append(sid)

                if y_arr is None:
                    # per your rule: if no y, also no group_label
                    for i, g, sid, pp in zip(idx_arr, g_arr, subject_ids, p_arr):
                        rows.append({
                            "model": mname,
                            "variant": variant,
                            "split": split_name,
                            "trial": trial,
                            "outer_fold": outer_fold,
                            "idx": int(i),
                            "group": int(g),
                            "group_label": np.nan,
                            "subject_id": sid,
                            "y": np.nan,
                            "p": float(pp),
                        })
                else:
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
                if y_arr is None:
                    for i, pp in zip(idx_arr, p_arr):
                        rows.append({
                            "model": mname,
                            "variant": variant,
                            "split": split_name,
                            "trial": trial,
                            "outer_fold": outer_fold,
                            "idx": int(i),
                            unit_col: int(i) if unit_col != "idx" else int(i),
                            "y": np.nan,   # CHANGED: no pd.NA
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

            # ---------- train OOF ----------
            if include_train:
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
                        split_name="train",
                        trial=trial,
                        outer_fold=outer_fold,
                        variant=v,
                    )

            # ---------- external ----------
            if include_external:
                # idx: prefer explicit key, else default to 0..n_external-1
                if external_idx_key is not None and external_idx_key in r:
                    idx_ex = np.asarray(r[external_idx_key], dtype=int)
                else:
                    n_ex = int(r.get("n_external", len(r.get(external_prob_key_uncalib, []))))
                    idx_ex = np.arange(n_ex, dtype=int)

                # labels optional
                y_ex = np.asarray(r[external_y_key], dtype=int) if external_y_key in r else None

                for v in variants:
                    if v == "uncalib":
                        key = external_prob_key_uncalib
                    else:
                        key = f"{external_prob_key_prefix_calib}{v}"

                    if key not in r:
                        continue

                    p_ex = np.asarray(r[key], dtype=float)

                    _append_rows(
                        idx_arr=idx_ex,
                        y_arr=y_ex,
                        p_arr=p_ex,
                        split_name="external",
                        trial=trial,
                        outer_fold=outer_fold,
                        variant=v,
                    )

        df_m = pd.DataFrame(rows)

        if not df_m.empty:
            df_m["model"] = df_m["model"].astype(str)
            df_m["variant"] = df_m["variant"].astype(str)
            df_m["split"] = df_m["split"].astype(str)
            df_m["idx"] = df_m["idx"].astype(int)
            df_m["p"] = pd.to_numeric(df_m["p"], errors="coerce").astype(float)

            # CHANGED: robust to np.nan / missing / accidental NA-like
            df_m["y"] = pd.to_numeric(df_m["y"], errors="coerce").astype(float)

            if have_groups:
                df_m["group"] = df_m["group"].astype(int)
                # group_label can be str or nan; leave as object
                # subject_id leave as object

        all_dfs.append(df_m)

    if len(all_dfs) == 0:
        return pd.DataFrame()

    df_all = pd.concat(all_dfs, ignore_index=True)

    sort_cols = ["model", "variant", "split", "trial", "outer_fold", "idx"]
    sort_cols = [c for c in sort_cols if c in df_all.columns]
    if sort_cols:
        df_all = df_all.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    return df_all


def aggregate_predictions_by_idx(
    df_long: pd.DataFrame,
    *,
    model_name: str | Sequence[str] | None = None,
    split: str | Sequence[str] = "test",
    variants: Optional[Sequence[str]] = None,
    agg_stats: Sequence[str] = ("mean", "median", "std", "min", "max"),
    # if True, adds y_label when labels exist
    add_y_label: bool = True,
    # how to handle prevalence (optional output column)
    prevalence: Union[bool, float] = True,  # True=auto when labels exist, False=off, float=use value

    # ---- NEW: ensemble pooling across models ----
    add_ensemble: bool = True,
    ensemble_name: str = "ensemble",
    ensemble_models: Sequence[str] | None = None,  # if None: use all selected models after filtering
) -> pd.DataFrame:
    """
    Aggregate repeated predictions per idx into a single row per (model, variant, split, idx),
    and optionally add an "ensemble" model by pooling predictions across multiple models.

    Designed to consume the output of `build_long_predictions_df`, where df_long may contain
    repeated predictions for the same idx across trials/folds (and possibly across models).

    Supports splits:
      - "test"
      - "train"
      - "external"  (labels may be missing)

    Label handling (important for external):
      - If labels exist for the requested subset, `y` is carried as the first non-missing value
        within each (model, variant, split, idx) group.
      - If labels do NOT exist (all y are NaN), aggregated `y` remains NaN and y_label (if requested)
        is set to NaN.

    NEW: Ensemble pooling
    ---------------------
    If add_ensemble=True, this function ALSO produces rows for `model=ensemble_name` by pooling
    *all predictions* across `ensemble_models` (or all selected models) for each (variant, split, idx).
    In other words, if a patient idx has predictions from logistic_regression and xgboost,
    we treat them as additional replicate predictions and aggregate over the combined set.

    Parameters
    ----------
    df_long:
        Long table with columns at least: ["model","variant","split","idx","y","p"].
        y may be np.nan for unlabeled external.

    model_name:
        Controls which base models to include:
          - None: include all models in df_long
          - str: include only that model
          - Sequence[str]: include only those models

    split:
        Split name or list of splits to include (e.g., "test" or ["test","external"]).

    variants:
        Variants to include; if None, uses all variants present after filtering.

    agg_stats:
        Which stats to compute over repeated probabilities p:
          subset of {"mean","median","std","min","max"}.

    add_y_label:
        If True and labels exist, adds y_label using {0:"0 (neg)", 1:"1 (pos)"}.

    prevalence:
        - True: compute prevalence per (model, split) from unique labeled idx, if labels exist
        - False: do not add prevalence_used
        - float: use provided prevalence value (added per row as prevalence_used)

    add_ensemble:
        If True, append an ensemble "model" that pools predictions across multiple models.

    ensemble_name:
        Name to use in df_agg["model"] for the pooled ensemble rows.

    ensemble_models:
        Which models to pool. If None, pools all models remaining after model_name filtering.

    Returns
    -------
    pd.DataFrame
        Aggregated table with one row per (model, variant, split, idx), including:
          - y (float; NaN if unlabeled)
          - n_preds (count of contributing predictions)
          - p_mean / p_median / p_std / p_min / p_max depending on agg_stats
          - optional y_label (if labels exist and add_y_label=True)
          - optional prevalence_used (if enabled)
    """
    required = {"model", "variant", "split", "idx", "y", "p"}
    missing = required - set(df_long.columns)
    if missing:
        raise KeyError(f"df_long is missing required columns: {sorted(missing)}")

    d = df_long.copy()

    # ---- filter models ----
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

    # ---- filter split(s) ----
    if isinstance(split, str):
        splits = [split]
    else:
        splits = list(split)

    d["split"] = d["split"].astype(str)
    d = d[d["split"].isin(splits)].copy()
    if d.empty:
        raise ValueError(f"No rows found for split(s)={splits} (after model filter).")

    # ---- variants ----
    d["variant"] = d["variant"].astype(str)
    if variants is None:
        variants = sorted(d["variant"].unique().tolist())
    else:
        variants = list(variants)

    d = d[d["variant"].isin(variants)].copy()
    if d.empty:
        raise ValueError(f"No rows found after filtering variants={variants}.")

    # ---- types ----
    d["idx"] = pd.to_numeric(d["idx"], errors="coerce").astype(int)
    d["p"] = pd.to_numeric(d["p"], errors="coerce").astype(float)
    d["y"] = pd.to_numeric(d["y"], errors="coerce").astype(float)  # keep NaN if missing

    # ---- helper: y first non-nan ----
    def _first_non_nan(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce")
        x = x[~x.isna()]
        return float(x.iloc[0]) if len(x) else np.nan

    # ---- base per-model aggregation ----
    grp = d.groupby(["model", "variant", "split", "idx"], as_index=False, observed=False)

    agg_dict = {"y": ("y", _first_non_nan), "n_preds": ("p", "size")}
    if "mean" in agg_stats:   agg_dict["p_mean"]   = ("p", "mean")
    if "median" in agg_stats: agg_dict["p_median"] = ("p", "median")
    if "std" in agg_stats:    agg_dict["p_std"]    = ("p", "std")
    if "min" in agg_stats:    agg_dict["p_min"]    = ("p", "min")
    if "max" in agg_stats:    agg_dict["p_max"]    = ("p", "max")

    df_agg = grp.agg(**agg_dict)

    # ---- ensemble pooling across models ----
    if add_ensemble:
        # determine which models to pool
        if ensemble_models is None:
            pool_models = sorted(d["model"].unique().tolist())
        else:
            pool_models = [m for m in ensemble_models if m in set(d["model"].unique())]

        if len(pool_models) == 0:
            raise ValueError(
                "add_ensemble=True but no models available to pool. "
                "Check ensemble_models / model_name filters."
            )

        d_pool = d[d["model"].isin(pool_models)].copy()

        # group WITHOUT model -> pool predictions across models
        grp_e = d_pool.groupby(["variant", "split", "idx"], as_index=False, observed=False)

        df_e = grp_e.agg(**agg_dict)
        df_e.insert(0, "model", ensemble_name)  # add model column

        # append
        df_agg = pd.concat([df_agg, df_e], ignore_index=True)

    # Ensure p_* columns are float
    for c in df_agg.columns:
        if c.startswith("p_") or c == "p":
            df_agg[c] = pd.to_numeric(df_agg[c], errors="coerce").astype(float)

    # ---- y_label only if labels exist ----
    labels_exist = df_agg["y"].notna().any()
    if add_y_label:
        if labels_exist:
            y_map = {0.0: "0 (neg)", 1.0: "1 (pos)"}
            df_agg["y_label"] = df_agg["y"].map(y_map)
            df_agg["y_label"] = pd.Categorical(
                df_agg["y_label"],
                categories=["0 (neg)", "1 (pos)"],
                ordered=True,
            )
        else:
            df_agg["y_label"] = np.nan

    # ---- prevalence_used ----
    if prevalence is not False:
        if isinstance(prevalence, bool):
            if prevalence is True and labels_exist:
                # prevalence per (model, split) using unique labeled idx
                base = (
                    df_agg[df_agg["y"].notna()]
                    .drop_duplicates(["model", "split", "idx"])[["model", "split", "y"]]
                )
                prev_map = base.groupby(["model", "split"])["y"].mean().to_dict()
                df_agg["prevalence_used"] = [
                    float(prev_map.get((m, s), np.nan)) for m, s in zip(df_agg["model"], df_agg["split"])
                ]
            else:
                df_agg["prevalence_used"] = np.nan
        else:
            prev_val = float(prevalence)
            if not (0.0 <= prev_val <= 1.0):
                raise ValueError(f"prevalence must be in [0,1]; got {prev_val}")
            df_agg["prevalence_used"] = prev_val

    # stable ordering (keep ensemble alongside other models)
    df_agg = df_agg.sort_values(
        ["model", "variant", "split", "idx"],
        kind="mergesort",
    ).reset_index(drop=True)

    return df_agg


def compute_logloss_brier_from_df_agg(
    df_agg: pd.DataFrame,
    *,
    split: str | Sequence[str] = "test",
    pred_col: str = "p_mean",
    variants: Optional[Sequence[str]] = None,
    model_names: str | Sequence[str] | None = None,
    method_alias: Mapping[str, str] | None = None,
    prevalence_col: str | None = "prevalence_used",
    eps: float = 1e-15,
) -> pd.DataFrame:
    """
    Compute Log Loss and Brier score from an aggregated per-idx predictions table (df_agg),
    and also compute prevalence-only baselines for each metric.

    Expected df_agg columns (minimum):
      - model, variant, split, idx, y, <pred_col>
    Optional:
      - prevalence_used (or a user-specified prevalence_col)

    Label handling
    --------------
    Metrics require labels. Rows with y=NaN are ignored. If nothing labeled remains after
    filtering, raises ValueError.

    Baselines
    ---------
    For each (variant, split) subset we compute a baseline prevalence π from:
      1) prevalence_col (if provided and present and non-null), else
      2) π = mean(y) on unique idx in that subset.

    Baseline metrics:
      - baseline_log_loss = -[π log(π) + (1-π) log(1-π)]
      - baseline_brier    = π(1-π)

    Returns
    -------
    pd.DataFrame with columns:
      ["model","model_label","variant","split","n_labeled","prevalence_used",
       "log_loss","brier","baseline_log_loss","baseline_brier"]
    """
    if method_alias is None:
        method_alias = {}

    required = {"model", "variant", "split", "idx", "y", pred_col}
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

    # ---- variant filter ----
    if variants is None:
        variants = sorted(d["variant"].astype(str).unique().tolist())
    else:
        variants = list(variants)
    d = d[d["variant"].isin(variants)].copy()
    if d.empty:
        raise ValueError(f"No rows found after filtering variants={variants}.")

    # types
    d["idx"] = pd.to_numeric(d["idx"], errors="coerce").astype(int)
    d["y"] = pd.to_numeric(d["y"], errors="coerce").astype(float)
    d[pred_col] = pd.to_numeric(d[pred_col], errors="coerce").astype(float)

    # display labels
    d["model_label"] = d["model"].map(lambda m: method_alias.get(str(m), str(m))).astype(str)

    # ---- compute prevalence baseline per (variant, split) ----
    prev_map: dict[tuple[str, str], float] = {}

    for (v, s), sub_vs in d.groupby(["variant", "split"], observed=False):
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

    for (m, mlabel, v, s), sub in d.groupby(["model", "model_label", "variant", "split"], observed=False):
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
                variant=str(v),
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
        ["split", "variant", "model_label"],
        kind="mergesort",
    ).reset_index(drop=True)

    return df_metrics


def plot_logloss_brier_from_df_agg(
    df_agg: pd.DataFrame,
    *,
    split: str | Sequence[str] = "test",
    pred_col: str = "p_mean",
    variants: Optional[Sequence[str]] = None,
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
    show_variant_legend: bool | None = None,  # auto: show if len(variants)>1
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
      If multiple variants are provided, a variant legend is shown unless disabled.

    Returns the metrics dataframe (same as compute_logloss_brier_from_df_agg).
    """

    sns.set(style="whitegrid")

    if method_alias is None:
        method_alias = {}

    df_metrics = compute_logloss_brier_from_df_agg(
        df_agg,
        split=split,
        pred_col=pred_col,
        variants=variants,
        model_names=model_names,
        method_alias=method_alias,
        prevalence_col=prevalence_col,
    )

    # Decide whether to show variant legend
    uniq_variants = sorted(df_metrics["variant"].unique().tolist())
    if show_variant_legend is None:
        show_variant_legend = len(uniq_variants) > 1

    # Prepare palette (by model label)
    model_labels = df_metrics["model_label"].tolist()
    uniq_models = list(dict.fromkeys(model_labels))  # stable order as seen
    if model_palette is None:
        # fallback colors (matplotlib cycle) — user typically supplies this
        model_palette = {m: None for m in uniq_models}

    # stable ordering for x
    model_order = uniq_models

    def _plot(metric_col: str, baseline_col: str, title: str, ylim: tuple[float, float] | None):
        # aggregate over variants? no: keep variant-separated bars if multiple variants
        # but most of your use is variants=["beta"], so it becomes single bar per model.
        plot_df = df_metrics.copy()
        plot_df["model_label"] = pd.Categorical(plot_df["model_label"], categories=model_order, ordered=True)

        # If multiple variants, we plot grouped bars (variant within model).
        # If single variant, no grouping needed.
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(model_order), dtype=float)

        if len(uniq_variants) == 1:
            v = uniq_variants[0]
            sub = plot_df[plot_df["variant"] == v].sort_values("model_label")

            heights = sub[metric_col].to_numpy(dtype=float)

            colors = [model_palette.get(m, None) for m in sub["model_label"].astype(str).tolist()]
            bars = ax.bar(x, heights, color=colors)

            # Baseline: same for all models within (variant, split) (by construction)
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
            # grouped bars by variant (legend for variant is useful)
            width = 0.8 / max(1, len(uniq_variants))
            for j, v in enumerate(uniq_variants):
                sub = plot_df[plot_df["variant"] == v].sort_values("model_label")
                heights = sub[metric_col].to_numpy(dtype=float)
                xj = x - 0.4 + width / 2.0 + j * width

                # color by model, but variants differ by bar position, not color.
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

                # baseline line per variant (usually same across variants if same labels;
                # but we keep it correct in case you pass subsets later)
                base_val = float(sub[baseline_col].iloc[0])
                ax.axhline(base_val, color=baseline_color, lw=baseline_lw, ls=baseline_ls)

            ax.set_xticks(x)
            ax.set_xticklabels(model_order, rotation=x_tick_rotation, fontsize=font_size, fontweight="bold")

            if show_variant_legend:
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
    _plot("log_loss", "baseline_log_loss", f"Log loss across models — split={split_title}", logloss_ylim)
    _plot("brier", "baseline_brier", f"Brier score across models — split={split_title}", brier_ylim)

    return df_metrics


def plot_auroc_auprc_from_df_agg(
    df_agg: pd.DataFrame,
    *,
    split: str = "test",
    pred_col: str = "p_mean",
    prevalence_col: str = "prevalence_used",
    variants: Optional[Sequence[str]] = None,

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
    where each row corresponds to one unit (idx) for a given (model, variant, split) and contains:
      - predicted probability summary (e.g., p_mean) in `pred_col`
      - optional labels in column `y` (may be NaN for unlabeled external)
      - optional prevalence baseline value in `prevalence_col` (often repeated across rows)

    Behavior
    --------
    - If labels are present (at least one non-NaN y), computes AUROC/AUPRC for each (model, variant)
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
        Aggregated table with columns: ["model","variant","split","idx","y", pred_col, prevalence_col(optional)].

    split:
        Which split to evaluate (e.g., "test", "train_oof", "external").

    pred_col:
        Which probability column to evaluate (e.g., "p_mean").

    prevalence_col:
        Column containing prevalence baseline value (used only for AUPRC baseline). If missing or NaN,
        AUPRC baseline is skipped.

    variants:
        Which variants to include. If None, uses all variants in df_agg for that split.

    method_alias:
        Optional mapping model_key -> display label (used on x-axis and for model_palette lookup).

    model_palette:
        Optional mapping display label -> color.

    Returns
    -------
    pd.DataFrame
        Metrics table with one row per (model, variant, split), columns:
          ["model","model_display","variant","split","n","prevalence","auprc","auroc"]
    """

    sns.set(style="whitegrid")
    
    required = {"model", "variant", "split", "idx", "y", pred_col}
    missing = required - set(df_agg.columns)
    if missing:
        raise KeyError(f"df_agg missing required columns: {sorted(missing)}")

    if method_alias is None:
        method_alias = {}

    d = df_agg.copy()
    d = d[d["split"] == split].copy()
    if d.empty:
        raise ValueError(f"No rows found in df_agg for split='{split}'.")

    # variants filter
    if variants is None:
        variants = sorted(d["variant"].astype(str).unique().tolist())
    else:
        variants = list(variants)
    d = d[d["variant"].isin(variants)].copy()
    if d.empty:
        raise ValueError(f"No rows found after filtering variants={variants} for split='{split}'.")

    # types
    d["model"] = d["model"].astype(str)
    d["variant"] = d["variant"].astype(str)
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
    # Compute metrics per (model, variant)
    # -------------------------
    rows = []
    for (m, v), sub in d.groupby(["model", "variant"], observed=False):
        sub_labeled = sub[sub["y"].notna()].copy()

        n = int(sub_labeled["idx"].nunique()) if labels_exist else int(sub["idx"].nunique())

        if not labels_exist or sub_labeled.empty:
            rows.append(
                {
                    "model": m,
                    "model_display": method_alias.get(m, m),
                    "variant": v,
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
                "variant": v,
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
        # Expecting one variant or multiple; plot each variant separately (simple + explicit)
        # Here: we’ll plot a grouped-by-variant bar chart if more than 1 variant.
        plot_df = df_metrics.copy()

        # Order by display label (stable)
        model_order = plot_df["model_display"].unique().tolist()

        variants_order = variants if variants is not None else sorted(plot_df["variant"].unique().tolist())

        # build bar positions
        x = np.arange(len(model_order), dtype=float)
        n_var = len(variants_order)
        width = 0.8 / max(1, n_var)

        fig, ax = plt.subplots(figsize=figsize)

        for j, v in enumerate(variants_order):
            sub = plot_df[plot_df["variant"] == v].copy()
            sub = sub.set_index("model_display").reindex(model_order).reset_index()

            vals = sub[metric_col].to_numpy(dtype=float)

            # colors by model label (NOT by variant)
            if model_palette is not None:
                colors = [model_palette.get(lbl, None) for lbl in sub["model_display"].tolist()]
            else:
                colors = None

            xpos = x + (j - (n_var - 1) / 2.0) * width

            bars = ax.bar(
                xpos,
                vals,
                width=width,
                label=v if n_var > 1 else None,  # only show variant legend if multiple variants
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
        # - Only show legend if multiple variants OR baseline exists
        handles, labels = ax.get_legend_handles_labels()
        keep_H, keep_L = [], []

        # keep variant legend only if we have >1 variant
        if n_var > 1:
            # keep unique variant labels
            seen = set()
            for h, l in zip(handles, labels):
                if l in variants_order and l not in seen:
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
    variant: str | None = None,
    prob_col: str = "p_mean",
    threshold_split: str = "train",
    evaluation_split: str = "test",
    exclude_splits: str | Sequence[str] | None = None,
    n_grid: int = 101,
    mode: Literal["train_threshold", "test_threshold", "split_best", "mean_train_threshold"] = "train_threshold",
    # ---- labels / aliasing ----
    method_alias: Mapping[str, str] | None = None,
    split_display_names: Mapping[str, str] | None = None,
    # ---- styling ----
    figsize: tuple[float, float] = (9.0, 5.0),
    font_size: float = 12.0,
    legend_loc: str = "best",
    x_tick_rotation: int = 0,
    split_palette: Mapping[str, str] | None = None,   # e.g. {"train": "...", "test": "...", "external": "..."}
    bar_width: float = 0.36,
    capsize: float = 5.0,
    # ---- baseline ----
    show_baseline: bool = True,
    baseline_value: float = 0.50,
    baseline_color: str = "#D5F713",
    baseline_lw: float = 1.5,
    baseline_ls: str = "--",
    # ---- annotation ----
    annotate_mean_sd: bool = True,
    annotate_decimals: int = 3,
    annotate_font_size: float | None = None,
    annotate_offset: float = 0.015,
    # ---- y limits ----
    ylim: tuple[float, float] | None = None,
    # ---- console threshold summary ----
    print_threshold_summary: bool = True,
) -> pd.DataFrame:
    """
    Plot balanced accuracy from an aggregated predictions table (`df_agg`) and return a
    per-model summary table.

    For each selected model, the function uses one probability column (for example `p_mean`)
    and computes balanced accuracy on two splits:
    - `threshold_split`: split used to choose the threshold
    - `evaluation_split`: split used for evaluation

    The plot can show one or both split bars. If `exclude_splits` is used, those splits are
    hidden at plotting time only; threshold selection and metric calculation are still done
    as specified by `threshold_split`, `evaluation_split`, and `mode`.

    Parameters
    ----------
    df_agg:
        Aggregated predictions table with columns such as `model`, `variant`, `split`, `y`,
        and a probability column like `p_mean`.

    model_names:
        Model name, list of model names, or None to include all models in `df_agg`.

    variant:
        Probability variant to use (for example `"beta"`). If None, the function expects only
        one variant to be present after filtering.

    prob_col:
        Name of the probability column used for thresholding and balanced-accuracy calculation.

    threshold_split:
        Split used to select the threshold, usually `"train"`.

    evaluation_split:
        Split used for evaluation, usually `"test"` or `"external"`.

    exclude_splits:
        Split name or list of split names to hide from the plot. This affects plotting only.

    n_grid:
        Number of thresholds searched on the grid from 0 to 1.

    mode:
        Thresholding strategy:
        - `"train_threshold"`: choose threshold on `threshold_split`
        - `"test_threshold"`: choose threshold on `evaluation_split`
        - `"split_best"`: choose best threshold separately for each split
        - `"mean_train_threshold"`: same intent as the fold-level version; for aggregated data
        this behaves like `"train_threshold"`

    method_alias:
        Optional mapping from internal model names to display labels.

    split_display_names:
        Optional mapping from split names (for example `"train"`, `"test"`, `"external"`)
        to legend/display labels.

    figsize:
        Figure size in inches.

    font_size:
        Base font size for title, labels, ticks, and legend.

    legend_loc:
        Matplotlib legend location string.

    x_tick_rotation:
        Rotation angle for x-axis tick labels.

    split_palette:
        Optional mapping from split name to bar color.

    bar_width:
        Width used for grouped bars.

    capsize:
        Error-bar cap size.

    show_baseline:
        Whether to draw a horizontal baseline reference line.

    baseline_value:
        Y-value of the baseline reference line.

    baseline_color:
        Color of the baseline line.

    baseline_lw:
        Line width of the baseline.

    baseline_ls:
        Line style of the baseline.

    annotate_mean_sd:
        Whether to write `mean ± SD` above each plotted bar.

    annotate_decimals:
        Number of decimals used in bar annotations.

    annotate_font_size:
        Font size for annotations. If None, derived from `font_size`.

    annotate_offset:
        Vertical offset added above each bar annotation.

    ylim:
        Optional y-axis limits as `(ymin, ymax)`. If None, limits are chosen automatically.

    print_threshold_summary:
        Whether to print the selected threshold(s) to the console.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per model, including the chosen threshold and balanced
        accuracy for `threshold_split` and `evaluation_split`.
    """
    sns.set(style="whitegrid")

    required = {"model", "variant", "split", "idx", "y", prob_col}
    missing = required - set(df_agg.columns)
    if missing:
        raise KeyError(f"df_agg is missing required columns: {sorted(missing)}")

    if mode not in {"train_threshold", "test_threshold", "split_best", "mean_train_threshold"}:
        raise ValueError(
            "mode must be 'train_threshold', 'test_threshold', 'split_best', or 'mean_train_threshold'."
        )

    if method_alias is None:
        method_alias = {}

    if split_display_names is None:
        split_display_names = {
            "train": "Train",
            "test": "Test",
            "external": "External",
        }

    if split_palette is None:
        split_palette = {
            "train": "#1587F8",
            "test": "#F14949",
            "external": "#2E9B4E",
        }

    if threshold_split not in split_palette:
        raise ValueError(
            f"threshold_split={threshold_split!r} is missing from split_palette. "
            f"Available keys: {sorted(split_palette.keys())}"
        )
    if evaluation_split not in split_palette:
        raise ValueError(
            f"evaluation_split={evaluation_split!r} is missing from split_palette. "
            f"Available keys: {sorted(split_palette.keys())}"
        )

    d = df_agg.copy()
    d["model"] = d["model"].astype(str)
    d["variant"] = d["variant"].astype(str)
    d["split"] = d["split"].astype(str)
    d["y"] = pd.to_numeric(d["y"], errors="coerce").astype(float)
    d[prob_col] = pd.to_numeric(d[prob_col], errors="coerce").astype(float)

    # -------------------------
    # Filter model(s)
    # -------------------------
    if model_names is None:
        selected = sorted(d["model"].unique().tolist())
    elif isinstance(model_names, str):
        selected = [model_names]
    else:
        selected = list(model_names)

    missing_models = [m for m in selected if m not in set(d["model"].unique())]
    if missing_models:
        raise KeyError(
            f"Model(s) not found in df_agg: {missing_models}. "
            f"Available: {sorted(d['model'].unique().tolist())}"
        )

    d = d[d["model"].isin(selected)].copy()
    if d.empty:
        raise ValueError("No rows remain after model filtering.")

    # -------------------------
    # Filter variant
    # -------------------------
    if variant is not None:
        d = d[d["variant"] == variant].copy()
        if d.empty:
            raise ValueError(f"No rows found for variant={variant!r}.")
    else:
        variants_present = sorted(d["variant"].unique().tolist())
        if len(variants_present) > 1:
            raise ValueError(
                "Multiple variants are present in df_agg after filtering. "
                f"Please specify `variant`. Available: {variants_present}"
            )

    # -------------------------
    # Labels
    # -------------------------
    model_labels = [method_alias.get(m, m) for m in selected]
    dupes = sorted({x for x in model_labels if model_labels.count(x) > 1})
    if dupes:
        raise ValueError(f"method_alias causes duplicate model labels: {dupes}. Make aliases unique.")

    threshold_split_label = split_display_names.get(threshold_split, str(threshold_split))
    evaluation_split_label = split_display_names.get(evaluation_split, str(evaluation_split))

    grid = np.linspace(0.0, 1.0, int(n_grid))

    # -------------------------
    # Helpers
    # -------------------------
    def _get_xy(model: str, split_name: str) -> tuple[np.ndarray, np.ndarray]:
        sub = d[(d["model"] == model) & (d["split"] == split_name)].copy()

        # keep only labeled rows
        sub = sub[sub["y"].notna()].copy()
        if sub.empty:
            raise ValueError(f"No labeled rows for model={model!r}, split={split_name!r}.")

        y = sub["y"].to_numpy(dtype=float)
        p = sub[prob_col].to_numpy(dtype=float)

        uniq = set(np.unique(y[~np.isnan(y)]).tolist())
        if not uniq.issubset({0.0, 1.0}):
            raise ValueError(
                f"Non-binary labels found for model={model!r}, split={split_name!r}: {sorted(uniq)}"
            )

        return y.astype(int), p.astype(float)

    def _best_ba_and_t(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
        ba = np.array([balanced_accuracy_score(y, (s >= t).astype(int)) for t in grid], dtype=float)
        j = int(np.argmax(ba))
        return float(ba[j]), float(grid[j])

    # -------------------------
    # Compute BA per model
    # -------------------------
    threshold_split_vals: list[float] = []
    evaluation_split_vals: list[float] = []
    thresholds: list[float] = []

    for model in selected:
        y_thr, s_thr = _get_xy(model, threshold_split)
        y_eval, s_eval = _get_xy(model, evaluation_split)

        if mode == "split_best":
            ba_thr, _ = _best_ba_and_t(y_thr, s_thr)
            ba_eval, _ = _best_ba_and_t(y_eval, s_eval)
            t_star = np.nan
        else:
            # mean_train_threshold collapses to train_threshold in aggregated data
            if mode in {"train_threshold", "mean_train_threshold"}:
                _, t_star = _best_ba_and_t(y_thr, s_thr)
            else:  # test_threshold
                _, t_star = _best_ba_and_t(y_eval, s_eval)

            ba_thr = balanced_accuracy_score(y_thr, (s_thr >= t_star).astype(int))
            ba_eval = balanced_accuracy_score(y_eval, (s_eval >= t_star).astype(int))

        threshold_split_vals.append(float(ba_thr))
        evaluation_split_vals.append(float(ba_eval))
        thresholds.append(float(t_star) if not np.isnan(t_star) else np.nan)

    # One aggregated BA per model, so SD is zero by construction
    threshold_split_means = np.array(threshold_split_vals, dtype=float)
    evaluation_split_means = np.array(evaluation_split_vals, dtype=float)
    threshold_split_sds = np.zeros_like(threshold_split_means)
    evaluation_split_sds = np.zeros_like(evaluation_split_means)



    # -------------------------
    # Plot
    # -------------------------
    plot_rows = [
        {
            "split": threshold_split,
            "split_label": threshold_split_label,
            "means": threshold_split_means,
            "sds": threshold_split_sds,
            "color": split_palette[threshold_split],
        },
        {
            "split": evaluation_split,
            "split_label": evaluation_split_label,
            "means": evaluation_split_means,
            "sds": evaluation_split_sds,
            "color": split_palette[evaluation_split],
        },
    ]

    # Optional: exclude split bars only at plotting time
    if exclude_splits is not None:
        if isinstance(exclude_splits, str):
            exclude_splits = [exclude_splits]
        else:
            exclude_splits = list(exclude_splits)

        plot_rows = [row for row in plot_rows if row["split"] not in exclude_splits]

        if len(plot_rows) == 0:
            raise ValueError(
                f"No split bars remain to plot after exclude_splits={exclude_splits}."
            )

    x = np.arange(len(model_labels), dtype=float)
    width = float(bar_width)

    n_bars = len(plot_rows)
    if n_bars == 1:
        offsets = [0.0]
        widths = [width * 0.9]
    else:
        offsets = np.linspace(
            -width * (n_bars - 1) / 2.0,
            width * (n_bars - 1) / 2.0,
            n_bars,
        )
        widths = [width] * n_bars

    fig, ax = plt.subplots(figsize=figsize)

    bar_containers = []

    for row, offset, this_width in zip(plot_rows, offsets, widths):
        bars = ax.bar(
            x + offset,
            row["means"],
            this_width,
            yerr=row["sds"],
            capsize=capsize,
            color=row["color"],
            label=row["split_label"],
        )
        bar_containers.append((bars, row["means"], row["sds"]))

    if show_baseline:
        ax.axhline(
            float(baseline_value),
            linestyle=baseline_ls,
            linewidth=baseline_lw,
            color=baseline_color,
            label=f"Baseline = {baseline_value:.2f}",
        )

    ax.set_title(
        "Balanced accuracy from aggregated predictions",
        fontsize=font_size + 1,
        fontweight="bold",
    )
    ax.set_xlabel("Model", fontsize=font_size, fontweight="bold")
    ax.set_ylabel("Balanced accuracy", fontsize=font_size, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=font_size, fontweight="bold", rotation=x_tick_rotation)
    ax.tick_params(axis="y", labelsize=font_size)
    for lab in ax.get_yticklabels():
        lab.set_fontweight("bold")

    if annotate_mean_sd:
        ann_fs = annotate_font_size if annotate_font_size is not None else max(8.0, float(font_size) - 3.0)
        offset = float(annotate_offset)

        def _annotate(bars, means, sds):
            for bar, mean, sd in zip(bars, means, sds):
                x0 = bar.get_x() + bar.get_width() / 2.0
                y0 = float(mean) + float(sd) + offset
                ax.text(
                    x0,
                    y0,
                    f"{mean:.{annotate_decimals}f} ± {sd:.{annotate_decimals}f}",
                    ha="center",
                    va="bottom",
                    fontsize=ann_fs,
                    fontweight="bold",
                )

        for bars, means, sds in bar_containers:
            _annotate(bars, means, sds)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        top = max(
            max(float(np.max(row["means"] + row["sds"])) for row in plot_rows),
            float(baseline_value) if show_baseline else 0.0,
        )
        pad = 0.08 if annotate_mean_sd else 0.05
        ax.set_ylim(0.0, min(1.10, top + pad))

    handles, labels = ax.get_legend_handles_labels()
    handle_map = {lab: h for h, lab in zip(handles, labels)}

    ordered_labels = [row["split_label"] for row in plot_rows]
    if show_baseline:
        ordered_labels.append(f"Baseline = {baseline_value:.2f}")

    ordered_handles = [handle_map[lbl] for lbl in ordered_labels if lbl in handle_map]

    ax.legend(
        ordered_handles,
        ordered_labels,
        loc=legend_loc,
        frameon=True,
        prop={"size": font_size, "weight": "bold"},
        title="",
    )

    fig.tight_layout()
    plt.show()

    # -------------------------
    # Summary output
    # -------------------------
    summary = pd.DataFrame(
        {
            "model": selected,
            "model_label": model_labels,
            "variant": variant if variant is not None else d["variant"].iloc[0],
            "threshold_split": threshold_split,
            "evaluation_split": evaluation_split,
            "threshold_split_ba": threshold_split_means,
            "evaluation_split_ba": evaluation_split_means,
            "threshold": thresholds,
        }
    )

    if print_threshold_summary:
        if mode == "split_best":
            print("Per-model threshold summary:")
            print("  split_best uses independent best thresholds per split, so no single shared threshold is reported.")
        else:
            print("Per-model selected threshold summary:")
            for label, t in zip(model_labels, thresholds):
                print(f"  {label}: {t:.3f}")



    return summary



# def evaluate_external_validation_results(
#     all_results: Mapping[str, Sequence[Mapping[str, Any]]],
#     metrics_to_compute: Optional[list[str]] = None,
#     calibration_methods: Optional[list[str]] = None,
# ) -> dict[str, list[dict[str, Any]]]:
#     """
#     Recompute metrics from stored y/scores for external validation.

#     Expected keys per fold dict (if external validation exists):
#       - "y_external"
#       - "y_external_scores"                      (uncalibrated)
#     Optional calibrated keys (per method):
#       - "calib_external_predictions_<method>"    (e.g. platt, beta)

#     Supported metrics (by name):
#       - "average_precision"
#       - "roc_auc"
#       - "brier_score_loss"
#       - "log_loss"

#     Returns
#     -------
#     eval_results : dict[str, list[dict[str, Any]]]
#         Fold-level entries keyed by model name. Each fold entry includes:
#           - "model_name", "trial", "outer_fold"
#           - "external_prevalence": positive-class prevalence on external set
#           - "n_external" (if present in the fold dict)
#           - metric keys:
#               external_<metric>
#               external_<method>_<metric>   (if calibrated preds exist)
#     """
#     if metrics_to_compute is None:
#         metrics_to_compute = [
#             "average_precision",
#             "roc_auc",
#             "brier_score_loss",
#             "log_loss",
#         ]

#     if calibration_methods is None:
#         calibration_methods = []

#     # Map metric name -> sklearn function
#     metric_fns: dict[str, Any] = {}
#     for m in metrics_to_compute:
#         if m == "average_precision":
#             metric_fns[m] = metrics.average_precision_score
#         elif m == "roc_auc":
#             metric_fns[m] = metrics.roc_auc_score
#         elif m == "brier_score_loss":
#             metric_fns[m] = metrics.brier_score_loss
#         elif m == "log_loss":
#             # wrap so we can safely pass labels=[0,1]
#             metric_fns[m] = lambda y, p: metrics.log_loss(y, p, labels=[0, 1])
#         else:
#             raise ValueError(f"Unsupported metric: {m}")

#     eval_results: dict[str, list[dict[str, Any]]] = {}

#     for model_name, folds in all_results.items():
#         model_entries: list[dict[str, Any]] = []

#         for r in folds:
#             # Skip folds that don't have external validation
#             if "y_external" not in r or "y_external_scores" not in r:
#                 continue

#             y_ext = np.asarray(r["y_external"])
#             y_ext_scores = np.asarray(r["y_external_scores"])

#             external_prevalence = float(np.mean(y_ext))

#             entry: dict[str, Any] = {
#                 "model_name": r.get("model_name", model_name),
#                 "trial": r.get("trial"),
#                 "outer_fold": r.get("outer_fold"),
#                 "external_prevalence": external_prevalence,
#             }

#             if "n_external" in r:
#                 entry["n_external"] = int(r["n_external"])

#             # Uncalibrated external metrics
#             for m_name, scorer in metric_fns.items():
#                 entry[f"external_{m_name}"] = float(scorer(y_ext, y_ext_scores))

#                 # Calibrated external metrics per method (if present)
#                 for method in calibration_methods:
#                     ext_calib_key = f"calib_external_predictions_{method}"
#                     if ext_calib_key in r:
#                         entry[f"external_{method}_{m_name}"] = float(
#                             scorer(y_ext, np.asarray(r[ext_calib_key]))
#                         )

#             model_entries.append(entry)

#         eval_results[model_name] = model_entries

#     return eval_results



# from typing import Any, Mapping, Sequence, Optional
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns


# def plot_external_brier_logloss(
#     external_eval_results: Mapping[str, Sequence[Mapping[str, Any]]],
#     model_names: str | Sequence[str] | None = None,
#     use_calibrated: bool = False,
#     calibration_method: str | None = None,
#     figsize: tuple[float, float] = (9.0, 4.0),
#     font_size: float = 12.0,
#     x_tick_rotation: int = 0,
#     method_alias: Mapping[str, str] | None = None,
#     external_color: str = "#06B850",   # DEFAULT as requested
#     show_prevalence_baseline: bool = True,
#     prevalence: float | None = None,
#     brier_baseline_color: str = "#D5F713",
#     logloss_baseline_color: str = "#D5F713",
#     baseline_lw: float = 1.5,
#     baseline_ls: str = "--",
#     annotate_mean_sd: bool = True,
#     annotate_decimals: int = 3,
#     annotate_font_size: float | None = None,
#     annotate_offset: float = 0.015,
#     brier_ylim: tuple[float, float] | None = None,
#     logloss_ylim: tuple[float, float] | None = None,
# ) -> None:
#     """
#     Plot external validation Brier score and Log loss as two separate bar charts across models,
#     using mean ± SD across outer folds, with optional prevalence baselines and per-bar annotations.

#     Expected keys in each fold entry (from evaluate_external_validation_results):
#       - external_prevalence
#       - external_brier_score_loss, external_log_loss (uncalibrated)
#       - external_<method>_brier_score_loss, external_<method>_log_loss (if calibrated)
#     """

#     if method_alias is None:
#         method_alias = {}

#     # -------------------------
#     # Choose models
#     # -------------------------
#     if model_names is None:
#         model_names = list(external_eval_results.keys())
#     elif isinstance(model_names, str):
#         model_names = [model_names]
#     else:
#         model_names = list(model_names)

#     missing = [m for m in model_names if m not in external_eval_results]
#     if missing:
#         raise KeyError(
#             f"Model(s) not found in external_eval_results: {missing}. "
#             f"Available: {list(external_eval_results.keys())}"
#         )

#     model_labels = [method_alias.get(m, m) for m in model_names]
#     if len(set(model_labels)) != len(model_labels):
#         dupes = pd.Series(model_labels)[pd.Series(model_labels).duplicated(keep=False)].unique().tolist()
#         raise ValueError(
#             f"method_alias causes duplicate model labels {dupes}. "
#             f"Make aliases unique (or omit aliasing for colliding model names)."
#         )

#     # -------------------------
#     # Prevalence baseline (optional)
#     # -------------------------
#     if show_prevalence_baseline:
#         if prevalence is not None:
#             p_mean = float(prevalence)
#         else:
#             prev_vals = [
#                 float(entry["external_prevalence"])
#                 for m in model_names
#                 for entry in external_eval_results[m]
#                 if "external_prevalence" in entry
#             ]
#             if len(prev_vals) == 0:
#                 raise KeyError(
#                     "No 'external_prevalence' values found in external_eval_results entries. "
#                     "Pass prevalence=... explicitly or ensure evaluate_external_validation_results stores it."
#                 )
#             p_mean = float(np.mean(prev_vals))

#         if not (0.0 < p_mean < 1.0):
#             raise ValueError(f"prevalence must be in (0, 1); got {p_mean}")

#         brier_baseline = float(p_mean * (1.0 - p_mean))
#         logloss_baseline = float(-(p_mean * np.log(p_mean) + (1.0 - p_mean) * np.log(1.0 - p_mean)))
#     else:
#         brier_baseline = None
#         logloss_baseline = None

#     # -------------------------
#     # Helper: build tidy DF (single split: External)
#     # -------------------------
#     def _collect(metric_label: str, key: str) -> pd.DataFrame:
#         rows: list[dict[str, Any]] = []
#         for m in model_names:
#             display = method_alias.get(m, m)
#             for f in external_eval_results[m]:
#                 if key not in f:
#                     raise KeyError(
#                         f"Key '{key}' not found for model '{m}'. "
#                         f"Did you compute external metrics (and calibration='{calibration_method}' if applicable)?"
#                     )
#                 rows.append({"model": display, "split": "External", "score": f[key]})
#         df = pd.DataFrame(rows)
#         df["metric"] = metric_label
#         return df

#     # -------------------------
#     # Pick metric keys
#     # -------------------------
#     if not use_calibrated:
#         brier_key = "external_brier_score_loss"
#         ll_key = "external_log_loss"
#         title_suffix = " (external, uncalibrated)"
#     else:
#         if calibration_method is None:
#             raise ValueError("calibration_method must be provided when use_calibrated=True.")
#         brier_key = f"external_{calibration_method}_brier_score_loss"
#         ll_key = f"external_{calibration_method}_log_loss"
#         title_suffix = f" (external, calibrated: {calibration_method})"

#     df_brier = _collect("Brier", brier_key)
#     df_ll = _collect("LogLoss", ll_key)

#     sns.set(style="whitegrid")

#     # -------------------------
#     # Plot helper
#     # -------------------------
#     def _plot_metric(
#         df: pd.DataFrame,
#         y_label: str,
#         baseline_value: float | None,
#         baseline_color: str,
#         baseline_label: str | None,
#         ylim: tuple[float, float] | None = None,
#     ) -> None:
#         plt.figure(figsize=figsize)
#         ax = sns.barplot(
#             data=df,
#             x="model",
#             y="score",
#             estimator=np.mean,
#             errorbar=("sd"),
#             order=model_labels,
#             saturation=1,
#             color=external_color,   # single-color bars
#         )

#         if baseline_value is not None and baseline_label is not None:
#             ax.axhline(
#                 float(baseline_value),
#                 ls=baseline_ls,
#                 lw=baseline_lw,
#                 color=baseline_color,
#                 label=baseline_label,
#             )

#         ax.set_xlabel("Model", fontsize=font_size, fontweight="bold")
#         ax.set_ylabel(y_label, fontsize=font_size, fontweight="bold")
#         ax.set_title(
#             f"{y_label} across models{title_suffix}",
#             fontsize=font_size + 2,
#             fontweight="bold",
#         )

#         ax.tick_params(axis="both", labelsize=font_size)
#         for label in ax.get_xticklabels() + ax.get_yticklabels():
#             label.set_fontweight("bold")
#         ax.tick_params(axis="x", rotation=x_tick_rotation)

#         if ylim is not None:
#             ax.set_ylim(*ylim)

#         # -------------------------
#         # Annotate mean ± SD above each bar (one bar per model)
#         # -------------------------
#         if annotate_mean_sd:
#             summary = (
#                 df.groupby(["model"])["score"]
#                   .agg(mean="mean", sd=lambda x: np.std(x, ddof=1))
#                   .reset_index()
#             )
#             summary["sd"] = summary["sd"].fillna(0.0)

#             stats = {r["model"]: (float(r["mean"]), float(r["sd"])) for _, r in summary.iterrows()}

#             ann_fs = annotate_font_size if annotate_font_size is not None else max(8, float(font_size) - 3)
#             offset = float(annotate_offset)

#             # ax.patches aligns with order=model_labels
#             for model_label, bar in zip(model_labels, ax.patches):
#                 mean, sd = stats[model_label]
#                 x = bar.get_x() + bar.get_width() / 2.0
#                 y = mean + sd + offset
#                 ax.text(
#                     x, y,
#                     f"{mean:.{annotate_decimals}f} ± {sd:.{annotate_decimals}f}",
#                     ha="center", va="bottom",
#                     fontsize=ann_fs, fontweight="bold",
#                 )

#             # expand y-limit if not forced
#             top = max(m + s for (m, s) in stats.values()) + offset + 0.05
#             if ylim is None:
#                 y0, y1 = ax.get_ylim()
#                 ax.set_ylim(y0, max(y1, top))

#         # If baseline is shown, keep legend; otherwise omit
#         if baseline_value is not None and baseline_label is not None:
#             ax.legend(title="", loc="best", prop={"size": font_size, "weight": "bold"})

#         plt.tight_layout()
#         plt.show()

#     _plot_metric(
#         df_brier,
#         "Brier score",
#         brier_baseline,
#         brier_baseline_color,
#         None if brier_baseline is None else f"Baseline = {brier_baseline:.2f}",
#         ylim=brier_ylim,
#     )

#     _plot_metric(
#         df_ll,
#         "Log loss",
#         logloss_baseline,
#         logloss_baseline_color,
#         None if logloss_baseline is None else f"Baseline = {logloss_baseline:.2f}",
#         ylim=logloss_ylim,
#     )



# from typing import Any, Mapping, Sequence, Optional
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns


# def plot_external_auprc_auroc(
#     external_eval_results: Mapping[str, Sequence[Mapping[str, Any]]],
#     model_names: str | Sequence[str] | None = None,
#     use_calibrated: bool = False,
#     calibration_method: str | None = None,
#     figsize: tuple[float, float] = (9.0, 4.0),
#     font_size: float = 12.0,
#     legend_loc: str = "best",
#     x_tick_rotation: int = 0,
#     method_alias: Mapping[str, str] | None = None,
#     external_color: str = "#06B850",   # DEFAULT requested
#     # ---- baseline / prevalence handling ----
#     show_prevalence_baseline: bool = True,
#     prevalence: float | None = None,  # override; else mean of entry["external_prevalence"]
#     auroc_baseline_color: str = "#D5F713",
#     auprc_baseline_color: str = "#D5F713",
#     baseline_lw: float = 1.5,
#     baseline_ls: str = "--",
#     # ---- annotation ----
#     annotate_mean_sd: bool = True,
#     annotate_decimals: int = 3,
#     annotate_font_size: float | None = None,
#     annotate_offset: float = 0.015,
#     # ---- per-metric y-limits ----
#     auprc_ylim: tuple[float, float] | None = None,
#     auroc_ylim: tuple[float, float] | None = None,
# ) -> None:
#     """
#     Plot External AUPRC and AUROC as two separate bar charts across models,
#     using mean ± SD across outer folds.

#     Expected keys in each fold entry (from evaluate_external_validation_results):
#       - external_prevalence
#       - external_average_precision, external_roc_auc (uncalibrated)
#       - external_<method>_average_precision, external_<method>_roc_auc (calibrated)
#     """

#     if method_alias is None:
#         method_alias = {}

#     # -------------------------
#     # Choose models
#     # -------------------------
#     if model_names is None:
#         model_names = list(external_eval_results.keys())
#     elif isinstance(model_names, str):
#         model_names = [model_names]
#     else:
#         model_names = list(model_names)

#     missing = [m for m in model_names if m not in external_eval_results]
#     if missing:
#         raise KeyError(
#             f"Model(s) not found in external_eval_results: {missing}. "
#             f"Available: {list(external_eval_results.keys())}"
#         )

#     model_labels = [method_alias.get(m, m) for m in model_names]
#     if len(set(model_labels)) != len(model_labels):
#         dupes = pd.Series(model_labels)[pd.Series(model_labels).duplicated(keep=False)].unique().tolist()
#         raise ValueError(
#             f"method_alias causes duplicate model labels {dupes}. "
#             f"Make aliases unique (or omit aliasing for colliding model names)."
#         )

#     # -------------------------
#     # Prevalence baseline (AUPRC baseline)
#     # -------------------------
#     p_mean: float | None = None
#     if show_prevalence_baseline:
#         if prevalence is not None:
#             p_mean = float(prevalence)
#         else:
#             prev_vals = [
#                 float(entry["external_prevalence"])
#                 for m in model_names
#                 for entry in external_eval_results[m]
#                 if "external_prevalence" in entry
#             ]
#             if len(prev_vals) == 0:
#                 raise KeyError(
#                     "No 'external_prevalence' values found in external_eval_results entries. "
#                     "Pass prevalence=... explicitly or ensure external evaluator stores it."
#                 )
#             p_mean = float(np.mean(prev_vals))

#     # -------------------------
#     # Build tidy DF per metric (single split: External)
#     # -------------------------
#     def _collect(metric: str, key: str) -> pd.DataFrame:
#         rows: list[dict[str, Any]] = []
#         for m in model_names:
#             display = method_alias.get(m, m)
#             for f in external_eval_results[m]:
#                 if key not in f:
#                     raise KeyError(
#                         f"Key '{key}' not found for model '{m}'. "
#                         f"Did you compute external metrics (and calibration='{calibration_method}' if applicable)?"
#                     )
#                 rows.append({"model": display, "split": "External", "score": f[key]})
#         df = pd.DataFrame(rows)
#         df["metric"] = metric
#         return df

#     # -------------------------
#     # Pick metric keys
#     # -------------------------
#     if not use_calibrated:
#         ap_key = "external_average_precision"
#         roc_key = "external_roc_auc"
#         title_suffix = " (external, uncalibrated)"
#     else:
#         if calibration_method is None:
#             raise ValueError("calibration_method must be provided when use_calibrated=True.")
#         ap_key = f"external_{calibration_method}_average_precision"
#         roc_key = f"external_{calibration_method}_roc_auc"
#         title_suffix = f" (external, calibrated: {calibration_method})"

#     df_ap = _collect("AUPRC", ap_key)
#     df_roc = _collect("AUROC", roc_key)

#     sns.set(style="whitegrid")

#     # -------------------------
#     # Plot helper
#     # -------------------------
#     def _plot_df(
#         df: pd.DataFrame,
#         metric_name: str,
#         ylim: tuple[float, float] | None = None,
#     ) -> None:
#         plt.figure(figsize=figsize)
#         ax = sns.barplot(
#             data=df,
#             x="model",
#             y="score",
#             estimator=np.mean,
#             errorbar=("sd"),
#             order=model_labels,
#             saturation=1,
#             color=external_color,
#         )

#         # Baselines
#         if metric_name == "AUPRC" and show_prevalence_baseline and p_mean is not None:
#             ax.axhline(
#                 float(p_mean),
#                 ls=baseline_ls,
#                 lw=baseline_lw,
#                 color=auprc_baseline_color,
#                 label=f"Baseline = {float(p_mean):.2f}",
#             )

#         if metric_name == "AUROC":
#             ax.axhline(
#                 0.5,
#                 ls=baseline_ls,
#                 lw=baseline_lw,
#                 color=auroc_baseline_color,
#                 label="Baseline = 0.50",
#             )

#         ax.set_xlabel("Model", fontsize=font_size, fontweight="bold")
#         ax.set_ylabel(metric_name, fontsize=font_size, fontweight="bold")
#         ax.set_title(
#             f"{metric_name} across models{title_suffix}",
#             fontsize=font_size + 2,
#             fontweight="bold",
#         )

#         ax.tick_params(axis="both", labelsize=font_size)
#         for label in ax.get_xticklabels() + ax.get_yticklabels():
#             label.set_fontweight("bold")

#         ax.tick_params(axis="x", rotation=x_tick_rotation)

#         if ylim is not None:
#             ax.set_ylim(*ylim)

#         # Annotate mean ± SD (one bar per model)
#         if annotate_mean_sd:
#             summary = (
#                 df.groupby(["model"])["score"]
#                   .agg(mean="mean", sd=lambda x: np.std(x, ddof=1))
#                   .reset_index()
#             )
#             summary["sd"] = summary["sd"].fillna(0.0)
#             stats = {r["model"]: (float(r["mean"]), float(r["sd"])) for _, r in summary.iterrows()}

#             ann_fs = annotate_font_size if annotate_font_size is not None else max(8, float(font_size) - 3)
#             offset = float(annotate_offset)

#             for model_label, bar in zip(model_labels, ax.patches):
#                 mean, sd = stats[model_label]
#                 x = bar.get_x() + bar.get_width() / 2.0
#                 y = mean + sd + offset
#                 ax.text(
#                     x, y,
#                     f"{mean:.{annotate_decimals}f} ± {sd:.{annotate_decimals}f}",
#                     ha="center", va="bottom",
#                     fontsize=ann_fs, fontweight="bold",
#                 )

#             top = max(m + s for (m, s) in stats.values()) + offset + 0.05
#             if ylim is None:
#                 y0, y1 = ax.get_ylim()
#                 ax.set_ylim(y0, max(y1, max(1.05, top)))
#         else:
#             if ylim is None:
#                 ax.set_ylim(0.0, 1.05)

#         ax.legend(title="", loc=legend_loc, prop={"size": font_size, "weight": "bold"})
#         plt.tight_layout()
#         plt.show()

#     _plot_df(df_ap, "AUPRC", ylim=auprc_ylim)
#     _plot_df(df_roc, "AUROC", ylim=auroc_ylim)


# from typing import Any, Mapping, Sequence, Literal
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import balanced_accuracy_score


# def barplot_external_balanced_accuracy(
#     all_results: Mapping[str, Sequence[Mapping[str, Any]]],
#     model_names: str | Sequence[str] | None = None,
#     use_calibrated: bool = False,
#     calibration_method: str | None = None,
#     n_grid: int = 101,
#     mode: Literal["train_threshold", "test_threshold"] = "train_threshold",
#     # ---- labels / aliasing ----
#     method_alias: Mapping[str, str] | None = None,
#     # ---- styling ----
#     figsize: tuple[float, float] = (9.0, 5.0),
#     font_size: float = 12.0,
#     legend_loc: str = "best",
#     x_tick_rotation: int = 0,
#     external_color: str = "#06B850",   # DEFAULT requested
#     bar_width: float = 0.55,
#     capsize: float = 5.0,
#     # ---- baseline ----
#     show_baseline: bool = True,
#     baseline_value: float = 0.50,
#     baseline_color: str = "#D5F713",
#     baseline_lw: float = 1.5,
#     baseline_ls: str = "--",
#     # ---- annotation ----
#     annotate_mean_sd: bool = True,
#     annotate_decimals: int = 3,
#     annotate_font_size: float | None = None,
#     annotate_offset: float = 0.015,
#     # ---- y limits ----
#     ylim: tuple[float, float] | None = None,
#     # ---- console threshold summary ----
#     print_threshold_summary: bool = True,
# ) -> None:
#     """
#     External-only balanced accuracy bar plot.

#     For each model and each outer fold record `r` that contains external predictions:
#       1) Choose a threshold t* on a grid over [0,1] using either:
#            mode="train_threshold": (y_train, train_scores)
#            mode="test_threshold" : (y_test,  test_scores)
#       2) Apply that t* to external scores and compute BA on (y_external, external_scores).
#       3) Aggregate external BA across folds: mean ± SD, plot one bar per model ("External").

#     Score sources:
#       If use_calibrated=False:
#         - threshold selection:
#             train: y_train_scores, test: y_test_scores
#         - external eval:
#             y_external_scores
#       If use_calibrated=True:
#         - threshold selection:
#             train: cv_calib_train_predictions_{method}
#             test : calib_test_predictions_{method}
#         - external eval:
#             calib_external_predictions_{method}
#     """
#     # -------------------------
#     # Validation / defaults
#     # -------------------------
#     if use_calibrated and calibration_method is None:
#         raise ValueError("calibration_method must be provided when use_calibrated=True.")
#     if mode not in {"train_threshold", "test_threshold"}:
#         raise ValueError("mode must be 'train_threshold' or 'test_threshold'.")

#     if method_alias is None:
#         method_alias = {}

#     # -------------------------
#     # Choose models
#     # -------------------------
#     if model_names is None:
#         selected = list(all_results.keys())
#     elif isinstance(model_names, str):
#         selected = [model_names]
#     else:
#         selected = list(model_names)

#     missing = [m for m in selected if m not in all_results]
#     if missing:
#         raise KeyError(f"Model(s) not found in all_results: {missing}. Available: {list(all_results.keys())}")

#     model_labels = [method_alias.get(m, m) for m in selected]
#     if len(set(model_labels)) != len(model_labels):
#         dupes = sorted({x for x in model_labels if model_labels.count(x) > 1})
#         raise ValueError(f"method_alias causes duplicate model labels: {dupes}. Make aliases unique.")

#     grid = np.linspace(0.0, 1.0, int(n_grid))

#     # -------------------------
#     # Helpers
#     # -------------------------
#     def _get_threshold_split_y_scores(r: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
#         """Return (y, scores) for threshold selection split (train or test)."""
#         if mode == "train_threshold":
#             y_key = "y_train"
#             if not use_calibrated:
#                 s_key = "y_train_scores"
#             else:
#                 s_key = f"cv_calib_train_predictions_{calibration_method}"
#         else:  # test_threshold
#             y_key = "y_test"
#             if not use_calibrated:
#                 s_key = "y_test_scores"
#             else:
#                 s_key = f"calib_test_predictions_{calibration_method}"

#         if y_key not in r or s_key not in r:
#             raise KeyError(f"Missing keys '{y_key}'/'{s_key}' in fold record.")
#         return np.asarray(r[y_key]), np.asarray(r[s_key])

#     def _get_external_y_scores(r: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
#         """Return (y_external, external_scores)."""
#         y_key = "y_external"
#         if not use_calibrated:
#             s_key = "y_external_scores"
#         else:
#             s_key = f"calib_external_predictions_{calibration_method}"

#         if y_key not in r or s_key not in r:
#             raise KeyError(f"Missing keys '{y_key}'/'{s_key}' in fold record.")
#         return np.asarray(r[y_key]), np.asarray(r[s_key])

#     def _best_ba_and_t(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
#         ba = np.array([balanced_accuracy_score(y, (s >= t).astype(int)) for t in grid], dtype=float)
#         j = int(np.argmax(ba))
#         return float(ba[j]), float(grid[j])

#     # -------------------------
#     # Compute per-fold external BA for each model
#     # -------------------------
#     ext_vals_per_model: list[np.ndarray] = []
#     tstars_per_model: list[np.ndarray] = []

#     for model in selected:
#         folds = all_results[model]

#         ext_ba: list[float] = []
#         tstars: list[float] = []

#         for r in folds:
#             # need external keys present; if not, skip fold
#             if "y_external" not in r:
#                 continue

#             try:
#                 y_thr, s_thr = _get_threshold_split_y_scores(r)
#                 _, t_star = _best_ba_and_t(y_thr, s_thr)

#                 y_ext, s_ext = _get_external_y_scores(r)
#                 ext_ba.append(balanced_accuracy_score(y_ext, (s_ext >= t_star).astype(int)))
#                 tstars.append(t_star)
#             except KeyError:
#                 continue

#         if len(ext_ba) == 0:
#             raise ValueError(
#                 f"No usable folds with external predictions found for model '{model}'. "
#                 "Check that your fold dicts contain y_external + external score keys."
#             )

#         ext_vals_per_model.append(np.array(ext_ba, dtype=float))
#         tstars_per_model.append(np.array(tstars, dtype=float))

#     ext_means = np.array([v.mean() for v in ext_vals_per_model], dtype=float)
#     ext_sds = np.array([v.std(ddof=1) if v.size > 1 else 0.0 for v in ext_vals_per_model], dtype=float)

#     # -------------------------
#     # Plot
#     # -------------------------
#     x = np.arange(len(model_labels), dtype=float)

#     fig, ax = plt.subplots(figsize=figsize)

#     bars_ext = ax.bar(
#         x,
#         ext_means,
#         bar_width,
#         yerr=ext_sds,
#         capsize=capsize,
#         color=external_color,
#         label="External",
#     )

#     if show_baseline:
#         ax.axhline(
#             float(baseline_value),
#             linestyle=baseline_ls,
#             linewidth=baseline_lw,
#             color=baseline_color,
#             label=f"Baseline = {baseline_value:.2f}",
#         )

#     # title_suffix = " (calibrated)" if use_calibrated else " (uncalibrated)"
#     # if use_calibrated:
#     #     title_suffix = f" (calibrated: {calibration_method})"
#     # title_suffix += f", threshold via {mode.replace('_', ' ')}"
#     title_suffix=''

#     ax.set_title(
#         f"External balanced accuracy across folds{title_suffix}",
#         fontsize=font_size + 1,
#         fontweight="bold",
#     )
#     ax.set_xlabel("Model", fontsize=font_size, fontweight="bold")
#     ax.set_ylabel("Balanced accuracy", fontsize=font_size, fontweight="bold")

#     ax.set_xticks(x)
#     ax.set_xticklabels(model_labels, fontsize=font_size, fontweight="bold", rotation=x_tick_rotation)
#     ax.tick_params(axis="y", labelsize=font_size)
#     for lab in ax.get_yticklabels():
#         lab.set_fontweight("bold")

#     # ---- annotations ----
#     if annotate_mean_sd:
#         ann_fs = annotate_font_size if annotate_font_size is not None else max(8.0, float(font_size) - 3.0)
#         offset = float(annotate_offset)
#         for bar, mean, sd in zip(bars_ext, ext_means, ext_sds):
#             x0 = bar.get_x() + bar.get_width() / 2.0
#             y0 = float(mean) + float(sd) + offset
#             ax.text(
#                 x0,
#                 y0,
#                 f"{mean:.{annotate_decimals}f} ± {sd:.{annotate_decimals}f}",
#                 ha="center",
#                 va="bottom",
#                 fontsize=ann_fs,
#                 fontweight="bold",
#             )

#     # ---- y-lims ----
#     if ylim is not None:
#         ax.set_ylim(*ylim)
#     else:
#         top = max(
#             float(np.max(ext_means + ext_sds)),
#             float(baseline_value) if show_baseline else 0.0,
#         )
#         pad = 0.08 if annotate_mean_sd else 0.05
#         ax.set_ylim(0.0, min(1.10, top + pad))

#     ax.legend(loc=legend_loc, frameon=True, prop={"size": font_size, "weight": "bold"}, title="")
#     fig.tight_layout()
#     plt.show()

#     # -------------------------
#     # Optional: print threshold summary
#     # -------------------------
#     if print_threshold_summary and mode in {"train_threshold", "test_threshold"}:
#         print("Per-model selected threshold summary (mean ± SD across folds):")
#         for label, tarr in zip(model_labels, tstars_per_model):
#             if tarr.size == 0:
#                 print(f"  {label}: (no thresholds computed)")
#                 continue
#             t_mean = float(np.mean(tarr))
#             t_sd = float(np.std(tarr, ddof=1)) if tarr.size > 1 else 0.0
#             print(f"  {label}: {t_mean:.3f} ± {t_sd:.3f}")

# external_eval_results = evaluate_external_validation_results(
#     all_results,
#     metrics_to_compute=["average_precision", "roc_auc", "brier_score_loss", "log_loss"],
#     calibration_methods=["platt", "beta"],
# )

# plot_external_brier_logloss(
#     external_eval_results,
#     model_names=None,
#     use_calibrated=True,
#     calibration_method="beta",
#     method_alias={"logistic_regression": "Logistic regression", "xgboost": "XGBoost"},
#     external_color="#06B850",  # optional, already default
#     figsize=(7, 5),
#     show_prevalence_baseline=True,
#     annotate_font_size=10,
#     logloss_ylim=(0, 1),
# )

# plot_external_auprc_auroc(
#     external_eval_results,
#     model_names=None,
#     use_calibrated=True,
#     calibration_method="beta",
#     method_alias={"logistic_regression": "Logistic regression", "xgboost": "XGBoost"},
#     external_color="#06B850",   # optional; default already
#     figsize=(7, 5),
#     legend_loc="lower right",
#     show_prevalence_baseline=True,
#     annotate_font_size=10,
#     auprc_ylim=(0, 1.0),
#     auroc_ylim=(0, 1.0),
# )

# barplot_external_balanced_accuracy(
#     all_results,
#     model_names=None,
#     use_calibrated=True,
#     calibration_method="beta",
#     mode="train_threshold",
#     method_alias={"logistic_regression": "Logistic regression", "xgboost": "XGBoost"},
#     external_color="#06B850",   # optional; default already
#     show_baseline=True,
#     baseline_color="#D5F713",
#     figsize=(7, 5),
#     legend_loc="lower right",
#     annotate_font_size=10,
#     ylim=(0, 1),
#     print_threshold_summary=True,
# )