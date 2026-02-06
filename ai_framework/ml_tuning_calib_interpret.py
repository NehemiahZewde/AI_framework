# ml_tuning_calib_interpret .py
# ML training calibration and interpretation

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Type, Mapping, Literal, Callable

import numpy as np
import pandas as pd
import optuna
from optuna.trial import Trial

from sklearn.base import ClassifierMixin
# from sklearn.datasets import load_breast_cancer
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_validate
from sklearn.model_selection._split import BaseCrossValidator  # for typing
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from betacal import BetaCalibration

from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
from tqdm.auto import trange
from tqdm.auto import tqdm

from sklearn.utils import column_or_1d
from sklearn.utils.validation import (
    _check_pos_label_consistency,
    check_consistent_length,
)


from sklearn import metrics
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


import pickle
import cloudpickle
import gzip
import json
from datetime import datetime
from pathlib import Path

import re
from .ml_feature_selection import prepare_training_bundle


# ---------------------------------------------------------------------
# Save and load results from training ML model
# ---------------------------------------------------------------------
def save_all_results(
    output_dir: Union[str, Path],
    all_results: Mapping[str, Any],
    *,
    prefix: str = "all_results",
    compress: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """
    Save `all_results` exactly as-is, including sklearn models + SmartCal calibrators,
    even if they contain local/closure-defined objects, using cloudpickle.

    Parameters
    ----------
    output_dir:
        Directory to save into (created if needed).

    all_results:
        Nested results dict (may contain numpy arrays, estimators, calibrators, etc.).

    prefix:
        Base filename (no extension).

    compress:
        If True, gzip the pickle.

    metadata:
        Optional JSON-serializable metadata saved alongside the pickle.

    Returns
    -------
    Path
        The output directory path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pkl_path = out / (f"{prefix}.pkl.gz" if compress else f"{prefix}.pkl")

    if compress:
        with gzip.open(pkl_path, "wb") as f:
            cloudpickle.dump(all_results, f, protocol=cloudpickle.DEFAULT_PROTOCOL)
    else:
        with open(pkl_path, "wb") as f:
            cloudpickle.dump(all_results, f, protocol=cloudpickle.DEFAULT_PROTOCOL)

    if metadata is not None:
        meta_out = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "prefix": prefix,
            "compressed": compress,
            **dict(metadata),
        }
        meta_path = out / f"{prefix}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta_out, f, indent=2)

    print(f"✅ Saved all_results to: {pkl_path.resolve()}")
    return out



def load_all_results(
    output_dir: Union[str, Path],
    *,
    prefix: str = "all_results",
    compress: bool = True,
    load_metadata: bool = False,
    verbose: bool = True,
) -> Any | tuple[Any, Optional[dict[str, Any]]]:
    """
    Loader matching `save_all_results()` (cloudpickle-based).

    If verbose=True, prints the resolved path of the loaded pickle, similar to save.
    """
    out = Path(output_dir)

    pkl_path = out / (f"{prefix}.pkl.gz" if compress else f"{prefix}.pkl")
    if not pkl_path.exists():
        raise FileNotFoundError(f"Could not find {pkl_path.name} in {out.resolve()}")

    if compress:
        with gzip.open(pkl_path, "rb") as f:
            all_results = cloudpickle.load(f)
    else:
        with open(pkl_path, "rb") as f:
            all_results = cloudpickle.load(f)

    if verbose:
        print(f"✅ Loaded all_results from: {pkl_path.resolve()}")

    if not load_metadata:
        return all_results

    meta_path = out / f"{prefix}_meta.json"
    meta: Optional[dict[str, Any]] = None
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)

        if verbose:
            print(f"ℹ️ Loaded metadata from: {meta_path.resolve()}")
    else:
        if verbose:
            print(f"ℹ️ No metadata sidecar found at: {meta_path.resolve()}")

    return all_results, meta





# ---------------------------------------------------------------------
# Objective that works for different models 
# ---------------------------------------------------------------------
def objective_with_args(
    trial: Trial,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_cv: BaseCrossValidator,
    cfg: Dict[str, Any],
    groups_train: Optional[np.ndarray] = None,
) -> float:
    """
    Optuna objective function for nested CV hyperparameter tuning.

    This function:
      1. Builds a model for the given `model_name` using the model's
         parameter configuration in `cfg["models"][model_name]["params"]`.
         The params callable is expected to take an Optuna `trial` and
         return a dict of keyword arguments for the model constructor.
      2. Evaluates the model using inner cross-validation (`inner_cv`)
         on (X_train, y_train) with the metric specified in
         cfg["metric"]["scoring"].
         If `groups_train` is provided, it is passed to `cross_validate`
         so that group-aware splitters (e.g. StratifiedGroupKFold) work.
      3. Stores the per-fold inner train and test scores in the trial's
         `user_attrs` for later inspection.
      4. Returns the mean inner test score, which Optuna will try to
         maximize (or minimize depending on `direction` in the study).

    Parameters
    ----------
    trial : optuna.trial.Trial
        The Optuna trial object used to sample hyperparameters.
    model_name : str
        Name of the model to tune. Must be a key in cfg["models"].
        The corresponding entry must define:
        - "estimator_cls": a scikit-learn–compatible estimator class
        - "params": callable(trial) -> dict of constructor kwargs
    X_train : np.ndarray
        Training features for the outer fold. Inner CV splits will be taken
        from this subset only.
    y_train : np.ndarray
        Training labels for the outer fold.
    inner_cv : BaseCrossValidator
        The inner cross-validation splitter used for model selection.
        Can be StratifiedKFold or StratifiedGroupKFold.
    cfg : dict
        Global configuration dictionary. Must contain:
        - cfg["metric"]["scoring"]: str, the scoring name for cross_validate
        - cfg["models"][model_name]["estimator_cls"]: estimator class
        - cfg["models"][model_name]["params"]: callable(trial) -> dict
    groups_train : np.ndarray or None, default=None
        Group labels for the samples in (X_train, y_train). If provided,
        they are passed to `cross_validate(..., groups=groups_train)` so
        that group-based CV splitters enforce group boundaries.

    Returns
    -------
    float
        Mean inner test score (across inner folds). This is the objective
        value that Optuna uses to compare trials.
    """
    # Get the scoring name (e.g. "average_precision")
    scoring = cfg["metric"]["scoring"]


    # ------------------------------------------------------------------
    # 1. Build model from config-driven parameter space
    # ------------------------------------------------------------------
    if model_name not in cfg["models"]:
        raise ValueError(f"Unknown model_name in objective: {model_name}")

    model_cfg = cfg["models"][model_name]

    if "estimator_cls" not in model_cfg:
        raise KeyError(
            f"cfg['models']['{model_name}'] is missing 'estimator_cls'. "
            "Add it (e.g., LogisticRegression, RandomForestClassifier, XGBClassifier)."
        )

    estimator_cls = model_cfg["estimator_cls"]
    params = model_cfg["params"](trial)

    # Optional safety: drop params that are None (helps avoid passing None into constructors)
    params = {k: v for k, v in params.items() if v is not None}

    model = estimator_cls(**params)


    # ------------------------------------------------------------------
    # 2. Inner cross-validation on the outer-train split
    #    If groups_train is not None, this supports group-aware CV.
    # ------------------------------------------------------------------
    scores = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=inner_cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=True,
        groups=groups_train,
    )

    # Per-fold train and test scores for the chosen metric
    train_scores = scores["train_score"]
    test_scores = scores["test_score"]

    # ------------------------------------------------------------------
    # 3. Store inner CV scores on the trial for later inspection
    # ------------------------------------------------------------------
    trial.set_user_attr("inner_train_scores", train_scores)
    trial.set_user_attr("inner_test_scores", test_scores)

    # ------------------------------------------------------------------
    # 4. Objective value: mean inner test score across folds
    # ------------------------------------------------------------------
    return float(test_scores.mean())



# ---------------------------------------------------------------------
# Helper: build, fit, and evaluate final model on outer train/test
# ---------------------------------------------------------------------
def make_outer_inner_cv(
    model_selection: str,
    n_outer_splits: int,
    n_inner_splits: int,
    outer_trial_idx: int,
) -> Tuple[BaseCrossValidator, BaseCrossValidator]:
    """
    Create outer and inner CV splitters based on a model_selection string.

    Parameters
    ----------
    model_selection : str
        Name of the CV strategy. Supported:
        - "StratifiedKFold"
        - "StratifiedGroupKFold"
    n_outer_splits : int
        Number of folds for the outer CV.
    n_inner_splits : int
        Number of folds for the inner CV (Optuna).
    outer_trial_idx : int
        Index of the outer trial (used for random_state).

    Returns
    -------
    outer_cv : BaseCrossValidator
    inner_cv : BaseCrossValidator
    """
    if model_selection == "StratifiedKFold":
        cv_cls = StratifiedKFold
    elif model_selection == "StratifiedGroupKFold":
        cv_cls = StratifiedGroupKFold
    else:
        raise ValueError(
            f"Unsupported model_selection='{model_selection}'. "
            "Use 'StratifiedKFold' or 'StratifiedGroupKFold'."
        )

    outer_cv = cv_cls(
        n_splits=n_outer_splits,
        shuffle=True,
        random_state=outer_trial_idx,
    )

    inner_cv = cv_cls(
        n_splits=n_inner_splits,
        shuffle=True,
        random_state=outer_trial_idx,
    )

    return outer_cv, inner_cv


def fit_and_evaluate_final_model(
    model_name: str,
    best_params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[ClassifierMixin, Dict[str, float], Dict[str, float], np.ndarray, np.ndarray]:
    """
    Build, fit, and evaluate the final model for one outer CV fold.

    The model is constructed from the config entry `cfg["models"][model_name]` by:
    - taking base params from `params(trial=None)`
    - overwriting them with Optuna’s `best_params`

    This function fits on (X_train, y_train), predicts positive-class probabilities,
    and reports outer-fold AUPRC and ROC AUC on both train and test splits.

    Parameters
    ----------
    model_name : str
        Key in `cfg["models"]`. The config entry must include:
        - "estimator_cls": estimator class (e.g., LogisticRegression, XGBClassifier)
        - "params": callable(trial|None) -> dict of constructor kwargs
    best_params : dict
        Best hyperparameters from Optuna (`study.best_trial.params`).
    X_train, y_train : np.ndarray
        Outer-fold training data.
    X_test, y_test : np.ndarray
        Outer-fold test data.
    cfg : dict
        Global config dictionary.

    Returns
    -------
    final_model : ClassifierMixin
        Fitted classifier instance.
    train_metrics : dict
        Metrics on outer-train split: {"average_precision": ..., "roc_auc": ...}.
    test_metrics : dict
        Metrics on outer-test split: {"average_precision": ..., "roc_auc": ...}.
    y_train_scores : np.ndarray
        Predicted probabilities for the positive class on X_train.
    y_test_scores : np.ndarray
        Predicted probabilities for the positive class on X_test.

    Notes
    -----
    Assumes binary classification and that the estimator implements `predict_proba`.
    """
  
    # ------------------------------------------------------------------
    # 1. Build final model from config + Optuna best params
    # ------------------------------------------------------------------
    if model_name not in cfg["models"]:
        raise ValueError(
            f"Unknown model_name in fit_and_evaluate_final_model: {model_name}"
        )

    model_cfg = cfg["models"][model_name]

    if "estimator_cls" not in model_cfg:
        raise KeyError(
            f"cfg['models']['{model_name}'] is missing 'estimator_cls'. "
            "Add it (e.g., LogisticRegression, RandomForestClassifier, XGBClassifier)."
        )

    estimator_cls = model_cfg["estimator_cls"]

    base_params = model_cfg["params"](None)  # includes fixed params + None placeholders
    final_params = {**base_params, **best_params}

    # Safety: remove any params still None (important with conditional spaces)
    final_params = {k: v for k, v in final_params.items() if v is not None}

    final_model = estimator_cls(**final_params)

    # ------------------------------------------------------------------
    # 2. Fit model on the outer-train split
    # ------------------------------------------------------------------
    final_model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 3. Predicted probabilities for positive class
    # ------------------------------------------------------------------
    y_train_scores = final_model.predict_proba(X_train)[:, 1]
    y_test_scores = final_model.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------------------
    # 4. Compute evaluation metrics
    # ------------------------------------------------------------------
    train_metrics = {
        "average_precision": average_precision_score(y_train, y_train_scores),
        "roc_auc": roc_auc_score(y_train, y_train_scores),
    }
    test_metrics = {
        "average_precision": average_precision_score(y_test, y_test_scores),
        "roc_auc": roc_auc_score(y_test, y_test_scores),
    }

    return final_model, train_metrics, test_metrics, y_train_scores, y_test_scores




# ---------------------------------------------------------------------
# Run nested CV for a single model_name using config
# ---------------------------------------------------------------------
def run_nested_cv_for_model(
    model_name: str,
    bundle: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    x_key: str = "combined_X_raw",
    y_key: str = "combined_y",
    groups_key: Optional[str] = None,   # e.g. "combined_groups" if you want group-aware CV
    model_selection: str = "StratifiedKFold",
) -> List[Dict[str, Any]]:
    """
    Run nested cross-validation with Optuna hyperparameter tuning for ONE model.

    This function supports:
      1) Choosing which dataset "level" to use from `bundle` via keys:
           - x_key (default: "combined_X_raw")
           - y_key (default: "combined_y")
           - groups_key (optional, e.g. "combined_groups")
      2) Per-model feature subsetting (slice columns ONCE per model run) via config:
           - cfg["models"][model_name]["feature_names"] : Optional[List[str]]
               Select these exact feature names (order preserved).
           - cfg["models"][model_name]["n_features"] : Optional[int]
               Select the first K features (prefix mode).
           - Only ONE of feature_names or n_features may be set (or both None => use all).

    Nested CV procedure:
      * Outer loop:
          - Repeated cfg["cv"]["num_trials"] times.
          - Uses a CV splitter (StratifiedKFold or StratifiedGroupKFold) with
            cfg["cv"]["n_outer_splits"] folds.
          - Each outer fold defines an outer train/test split.
      * Inner loop (Optuna tuning):
          - On each outer-train split, run an Optuna study for
            cfg["optuna"]["n_inner_trials"] trials.
          - Each Optuna trial is evaluated with an inner CV of
            cfg["cv"]["n_inner_splits"] folds using objective_with_args(...).
      * Final model per outer fold:
          - Refit a final model with the best hyperparameters on the full
            outer-train split and evaluate on outer-train and outer-test.

    Parameters
    ----------
    model_name:
        Key into cfg["models"].
    bundle:
        Dataset dict containing at least:
          - bundle[x_key]: 2D array (n_samples, n_features)
          - bundle[y_key]: 1D array (n_samples,)
          - bundle["feature_names"]: List[str] of length n_features
        If groups_key is provided, bundle[groups_key] must be length n_samples.
    cfg:
        Global configuration dict with:
          - cfg["cv"]: {"num_trials","n_outer_splits","n_inner_splits"}
          - cfg["optuna"]: {"n_inner_trials","direction", optional "n_jobs"}
          - cfg["models"][model_name]:
              - "params": callable(trial)->dict
              - optional "feature_names": List[str] | None
              - optional "n_features": int | None
              - optional "feature_strict": bool
              - optional "max_print_features": int
    x_key, y_key:
        Keys selecting which X/y arrays to use from bundle.
    groups_key:
        Optional key selecting group ids array from bundle (for group-aware CV).
    model_selection:
        "StratifiedKFold" or "StratifiedGroupKFold".

    Returns
    -------
    results:
        List of dicts, one per outer fold, including:
          - model_name, trial, outer_fold
          - feature_names, n_features
          - inner_train_scores/test_scores + mean/std (from best trial user_attrs)
          - outer_train_metrics, outer_test_metrics
          - best_params, final_model
          - indices + labels + score vectors
    """    
    # Grab this model's configuration block (params search space, feature selection, printing knobs, etc.)
    m_cfg = cfg["models"][model_name]

    # -----------------------------
    # Per-model feature selection knobs (choose ONE approach)
    # -----------------------------
    # Option A: explicitly specify feature names to keep (exact match against bundle["feature_names"])
    keep_features = m_cfg.get("feature_names", None)   # list[str] or None

    # Option B: keep the first K features (prefix mode; uses current column order)
    n_features = m_cfg.get("n_features", None)         # int or None

    # Disallow ambiguous intent: you either pick specific names OR pick top-K by column order
    if keep_features is not None and n_features is not None:
        raise ValueError(f"{model_name}: set only one of 'feature_names' or 'n_features' (or neither).")

    # -----------------------------
    # Pull the correct dataset "level" from bundle (raw vs aggregated/combined)
    # -----------------------------
    # These keys let you decide whether you're running CV on sample-level data or group-aggregated data.
    if x_key not in bundle:
        raise KeyError(f"bundle missing x_key='{x_key}'. Available keys: {list(bundle.keys())[:25]} ...")
    if y_key not in bundle:
        raise KeyError(f"bundle missing y_key='{y_key}'. Available keys: {list(bundle.keys())[:25]} ...")

    # Needed for name-based feature selection and for sanity checks (must match X columns)
    if "feature_names" not in bundle:
        raise KeyError("bundle must contain 'feature_names' for feature selection by name.")

    # Convert to numpy arrays (ensures consistent indexing and shapes)
    X_full = np.asarray(bundle[x_key])   # full feature matrix for this dataset level
    y = np.asarray(bundle[y_key])        # labels aligned row-by-row with X_full

    # -----------------------------
    # Sanity checks: shapes must be consistent before we do any CV splitting
    # -----------------------------
    if X_full.ndim != 2:
        raise ValueError(f"bundle[{x_key}] must be 2D, got shape {X_full.shape}")
    if y.ndim != 1:
        raise ValueError(f"bundle[{y_key}] must be 1D, got shape {y.shape}")

    # Outer CV splitters require X and y to have the same number of rows/samples
    if X_full.shape[0] != len(y):
        raise ValueError(
            f"X/y mismatch for keys ({x_key}, {y_key}): X rows={X_full.shape[0]} vs len(y)={len(y)}"
        )

    # Feature name list must match the number of columns in X
    if X_full.shape[1] != len(bundle["feature_names"]):
        raise ValueError(
            f"Mismatch: X has {X_full.shape[1]} cols but feature_names has {len(bundle['feature_names'])}"
        )

    # -----------------------------
    # Optional: group labels (only needed for StratifiedGroupKFold)
    # -----------------------------
    groups = None
    if groups_key is not None:
        # If user requested group-aware splitting, we need a groups vector aligned with y/X rows
        if groups_key not in bundle:
            raise KeyError(f"bundle missing groups_key='{groups_key}'")
        groups = np.asarray(bundle[groups_key])

        if groups.ndim != 1:
            raise ValueError(f"bundle[{groups_key}] must be 1D, got shape {groups.shape}")

        # Groups must also have one entry per row/sample
        if len(groups) != len(y):
            raise ValueError(
                f"groups/y mismatch for key {groups_key}: len(groups)={len(groups)} vs len(y)={len(y)}"
            )

    # -----------------------------
    # Create a tiny "view bundle" that matches prepare_training_bundle's expected schema:
    # it expects keys "X_raw" and "feature_names"
    # -----------------------------
    view_bundle = {"X_raw": X_full, "feature_names": bundle["feature_names"]}

    # -----------------------------
    # Slice ONCE per model run (column selection only, no data leakage concerns here)
    # - If keep_features is set: select exact named columns.
    # - If n_features is set: select the first K columns.
    # - If neither is set: use all features.
    # -----------------------------
    if keep_features is not None or n_features is not None:
        model_bundle = prepare_training_bundle(
            view_bundle,
            n_features=n_features,
            keep_features=keep_features,
            strict=m_cfg.get("feature_strict", True),  # error on missing names by default
            dedupe=True,                               # de-duplicate requested names (preserve order)
            copy_bundle=True,                          # avoid mutating original bundle
        )
    else:
        model_bundle = view_bundle

    # Final X that will be used for outer & inner CV in this run
    X = model_bundle["X_raw"]
    selected_feature_names = list(model_bundle["feature_names"])

    # Helpful debug print so you can verify the slicing did what you expect
    print(f"[{model_name}] X shape after slicing: {X.shape} (features={len(selected_feature_names)})")

    # Print feature names used (truncate to avoid huge console spam for wide datasets)
    max_show = int(m_cfg.get("max_print_features", 30))
    print(f"[{model_name}] feature_names ({len(selected_feature_names)}):")
    for f in selected_feature_names[:max_show]:
        print(f"  - {f}")
    if len(selected_feature_names) > max_show:
        print(f"  ... (+{len(selected_feature_names) - max_show} more)")

    # If the user selected a group-aware CV strategy, enforce that groups are actually present
    if model_selection == "StratifiedGroupKFold" and groups is None:
        raise ValueError(
            "model_selection='StratifiedGroupKFold' but groups is None. "
            "Provide groups_key (e.g., 'combined_groups') or use 'StratifiedKFold'."
        )

    
    # Enforce that groups are provided when using group-aware splitting
    if model_selection == "StratifiedGroupKFold" and groups is None:
        raise ValueError(
            "model_selection='StratifiedGroupKFold' but groups is None. "
            "Provide a groups array or use 'StratifiedKFold'."
        )


    # Pull out the sub-configs so we don't keep typing cfg["cv"] / cfg["optuna"] everywhere
    cv_cfg = cfg["cv"]           # cross-validation settings (outer/inner splits, repetitions)
    opt_cfg = cfg["optuna"]      # Optuna settings (how many trials, parallelism, etc.)

    # How many times to repeat the entire OUTER CV procedure.
    # If num_trials = 3 and n_outer_splits = 5, you'll run 15 outer folds total.
    NUM_TRIALS = cv_cfg["num_trials"]

    # Number of folds in the OUTER CV loop.
    # Each fold produces one outer test evaluation (generalization estimate).
    n_outer_splits = cv_cfg["n_outer_splits"]

    # Number of folds in the INNER CV loop (inside Optuna objective).
    # Each Optuna trial is evaluated by averaging performance across these inner folds.
    n_inner_splits = cv_cfg["n_inner_splits"]

    # Number of Optuna hyperparameter trials to run PER OUTER FOLD.
    # This is the tuning budget for searching hyperparameters on each outer-train split.
    N_INNER_OPTUNA_TRIALS = opt_cfg["n_inner_trials"]

    # How many parallel Optuna trials to run at once.
    # If missing, default to 1 (fully sequential).
    OPTUNA_N_JOBS = opt_cfg.get("n_jobs", 1)

    # This list will accumulate one dictionary per OUTER fold, including metrics + best params.
    results: List[Dict[str, Any]] = []

    # A running counter across *all* outer folds across all outer repetitions.
    # Used purely for printing progress like "Outer fold 3/15".
    cv_tracker = 0

    # Total number of outer folds we expect to run overall:
    # (num outer repetitions) * (outer folds per repetition).
    total_outer_folds = NUM_TRIALS * n_outer_splits


    # ------------------------------------------------------------------
    # Outer loop: repeated outer cross-validation
    # ------------------------------------------------------------------
    for trial_idx in range(NUM_TRIALS):
        # Build outer & inner CV based on the chosen strategy
        outer_cv, inner_cv = make_outer_inner_cv(
            model_selection=model_selection,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            outer_trial_idx=trial_idx,
        )

        print('='*200)
        print(f"\n=== Model: {model_name} | Trial {trial_idx + 1}/{NUM_TRIALS} ===")
        print('='*200)
        outer_fold_idx = 0

        # Select outer splits depending on whether we have groups or not
        if groups is not None:
            outer_splits = outer_cv.split(X, y, groups)
        else:
            outer_splits = outer_cv.split(X, y)

        # --------------------------------------------------------------
        # Loop over outer folds for this trial
        # --------------------------------------------------------------
        for outer_train_idx, outer_test_idx in outer_splits:
            cv_tracker += 1
            outer_fold_idx += 1
            print(
                f"Outer fold {cv_tracker}/{total_outer_folds} "
                f"(trial {trial_idx}, fold {outer_fold_idx})"
            )

            # Outer train/test split
            X_train, X_test = X[outer_train_idx], X[outer_test_idx]
            y_train, y_test = y[outer_train_idx], y[outer_test_idx]
            groups_train = groups[outer_train_idx] if groups is not None else None

            # ----------------------------------------------------------
            # Inner CV + Optuna hyperparameter tuning on (X_train, y_train)
            # ----------------------------------------------------------
            study = optuna.create_study(direction=opt_cfg["direction"])
            study.optimize(
                lambda tr: objective_with_args(
                    tr,
                    model_name,
                    X_train,
                    y_train,
                    inner_cv,
                    cfg,
                    groups_train=groups_train,
                ),
                n_trials=N_INNER_OPTUNA_TRIALS,
                show_progress_bar=True,
                n_jobs=OPTUNA_N_JOBS,
            )

            best_trial = study.best_trial

            # ----------------------------------------------------------
            # Fit final model on the full outer-train set using best params
            # ----------------------------------------------------------
            final_model,outer_train_metrics,outer_test_metrics, y_train_scores, y_test_scores,= fit_and_evaluate_final_model(model_name=model_name, 
                                                                                                                             best_params=best_trial.params, X_train=X_train,
                                                                                                                             y_train=y_train, X_test=X_test, y_test=y_test, cfg=cfg, )

            # ----------------------------------------------------------
            # Collect all information for this outer fold
            # ----------------------------------------------------------
            results.append(
                {
                    "model_name": model_name,
                    "trial": trial_idx,
                    "outer_fold": outer_fold_idx,

                    # inner CV stats (directly from best_trial)
                    "inner_train_scores": best_trial.user_attrs["inner_train_scores"],
                    "inner_test_scores": best_trial.user_attrs["inner_test_scores"],
                    "inner_train_mean": float(
                        best_trial.user_attrs["inner_train_scores"].mean()
                    ),
                    "inner_train_std": float(
                        best_trial.user_attrs["inner_train_scores"].std()
                    ),
                    "inner_test_mean": float(
                        best_trial.user_attrs["inner_test_scores"].mean()
                    ),
                    "inner_test_std": float(
                        best_trial.user_attrs["inner_test_scores"].std()
                    ),

                    # outer performance
                    "outer_train_metrics": outer_train_metrics,
                    "outer_test_metrics": outer_test_metrics,
                    "best_params": best_trial.params,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "final_model": final_model,

                    # bookkeeping of indices and raw labels/scores
                    "outer_train_idx": outer_train_idx,
                    "outer_test_idx": outer_test_idx,
                    "y_train": y_train,
                    "y_test": y_test,
                    "y_train_scores": y_train_scores,
                    "y_test_scores": y_test_scores,
                }
            )

    return results



# def run_nested_cv_for_model(
#     model_name: str,
#     X: np.ndarray,
#     y: np.ndarray,
#     cfg: Dict[str, Any],
#     groups: Optional[np.ndarray] = None,
#     model_selection: str = "StratifiedKFold",
# ) -> List[Dict[str, Any]]:
#     """
#     Run nested cross-validation with Optuna hyperparameter tuning
#     for a single model specified by `model_name`.

#     The procedure is:

#       * Outer loop:
#           - Repeated `NUM_TRIALS` times.
#           - Each repetition uses a CV splitter (StratifiedKFold or
#             StratifiedGroupKFold, depending on `model_selection`) with
#             `n_outer_splits` folds.
#           - Each outer fold defines an outer train/test split.

#       * Inner loop:
#           - On each outer train split (X_train, y_train), run an Optuna
#             study using `objective_with_args` and an inner CV with
#             `n_inner_splits` folds.
#           - The study searches over hyperparameters defined in the config
#             and selects the best trial based on mean inner test score.

#       * Final model per outer fold:
#           - Rebuild model using the best trial's hyperparameters via
#             `fit_and_evaluate_final_model`.
#           - Fit on the outer train split and evaluate on train and test.
#           - Store inner and outer metrics, best parameters, and indices.

#     Parameters
#     ----------
#     model_name : str
#         Name of the model to run nested CV for. Must be a key in
#         cfg["models"], e.g. "logistic_regression" or "random_forest".
#     X : np.ndarray
#         Full feature matrix, shape (n_samples, n_features).
#     y : np.ndarray
#         Full label vector, shape (n_samples,).
#     cfg : dict
#         Global configuration dictionary. Must contain:
#         - cfg["cv"] with keys: "num_trials", "n_outer_splits", "n_inner_splits"
#         - cfg["optuna"] with keys: "n_inner_trials", "direction"
#         - cfg["models"][model_name]["params"]: callable(trial) -> dict
#     groups : np.ndarray or None, default=None
#         Group labels for the samples, shape (n_samples,). Required if
#         `model_selection == "StratifiedGroupKFold"`, and ignored when
#         using plain "StratifiedKFold".
#     model_selection : str, default="StratifiedKFold"
#         CV strategy to use for both outer and inner loops. Supported:
#         - "StratifiedKFold"       (no grouping)
#         - "StratifiedGroupKFold"  (group-aware splitting)

#     Returns
#     -------
#     results : list of dict
#         One dictionary per outer fold, containing:
#         - "model_name": str
#         - "trial": int (outer repetition index)
#         - "outer_fold": int (fold index within the outer trial)
#         - "inner_train_scores": np.ndarray (per-fold inner train scores)
#         - "inner_test_scores": np.ndarray (per-fold inner test scores)
#         - "inner_train_mean": float
#         - "inner_train_std": float
#         - "inner_test_mean": float
#         - "inner_test_std": float
#         - "outer_train_metrics": dict of metrics on X_train
#         - "outer_test_metrics": dict of metrics on X_test
#         - "best_params": dict of best hyperparameters (Optuna)
#         - "n_train": int, size of outer train set
#         - "n_test": int, size of outer test set
#         - "final_model": fitted model object
#         - "outer_train_idx": np.ndarray of indices used for train
#         - "outer_test_idx": np.ndarray of indices used for test
#         - "y_train": np.ndarray of labels for train
#         - "y_test": np.ndarray of labels for test
#         - "y_train_scores": np.ndarray of scores on train
#         - "y_test_scores": np.ndarray of scores on test
#     """
#     cv_cfg = cfg["cv"]
#     opt_cfg = cfg["optuna"]


#     NUM_TRIALS = cv_cfg["num_trials"]
#     n_outer_splits = cv_cfg["n_outer_splits"]
#     n_inner_splits = cv_cfg["n_inner_splits"]
#     N_INNER_OPTUNA_TRIALS = opt_cfg["n_inner_trials"]
#     OPTUNA_N_JOBS = opt_cfg.get("n_jobs", 1)   # default to sequential if missing
    
#     # Enforce that groups are provided when using group-aware splitting
#     if model_selection == "StratifiedGroupKFold" and groups is None:
#         raise ValueError(
#             "model_selection='StratifiedGroupKFold' but groups is None. "
#             "Provide a groups array or use 'StratifiedKFold'."
#         )

#     results: List[Dict[str, Any]] = []
#     cv_tracker = 0
#     total_outer_folds = NUM_TRIALS * n_outer_splits

#     # ------------------------------------------------------------------
#     # Outer loop: repeated outer cross-validation
#     # ------------------------------------------------------------------
#     for trial_idx in range(NUM_TRIALS):
#         # Build outer & inner CV based on the chosen strategy
#         outer_cv, inner_cv = make_outer_inner_cv(
#             model_selection=model_selection,
#             n_outer_splits=n_outer_splits,
#             n_inner_splits=n_inner_splits,
#             outer_trial_idx=trial_idx,
#         )

#         print('='*200)
#         print(f"\n=== Model: {model_name} | Trial {trial_idx + 1}/{NUM_TRIALS} ===")
#         print('='*200)
#         outer_fold_idx = 0

#         # Select outer splits depending on whether we have groups or not
#         if groups is not None:
#             outer_splits = outer_cv.split(X, y, groups)
#         else:
#             outer_splits = outer_cv.split(X, y)

#         # --------------------------------------------------------------
#         # Loop over outer folds for this trial
#         # --------------------------------------------------------------
#         for outer_train_idx, outer_test_idx in outer_splits:
#             cv_tracker += 1
#             outer_fold_idx += 1
#             print(
#                 f"Outer fold {cv_tracker}/{total_outer_folds} "
#                 f"(trial {trial_idx}, fold {outer_fold_idx})"
#             )

#             # Outer train/test split
#             X_train, X_test = X[outer_train_idx], X[outer_test_idx]
#             y_train, y_test = y[outer_train_idx], y[outer_test_idx]
#             groups_train = groups[outer_train_idx] if groups is not None else None

#             # ----------------------------------------------------------
#             # Inner CV + Optuna hyperparameter tuning on (X_train, y_train)
#             # ----------------------------------------------------------
#             study = optuna.create_study(direction=opt_cfg["direction"])
#             study.optimize(
#                 lambda tr: objective_with_args(
#                     tr,
#                     model_name,
#                     X_train,
#                     y_train,
#                     inner_cv,
#                     cfg,
#                     groups_train=groups_train,
#                 ),
#                 n_trials=N_INNER_OPTUNA_TRIALS,
#                 show_progress_bar=True,
#                 n_jobs=OPTUNA_N_JOBS,
#             )

#             best_trial = study.best_trial

#             # ----------------------------------------------------------
#             # Fit final model on the full outer-train set using best params
#             # ----------------------------------------------------------
#             final_model,outer_train_metrics,outer_test_metrics, y_train_scores, y_test_scores,= fit_and_evaluate_final_model(model_name=model_name, 
#                                                                                                                              best_params=best_trial.params, X_train=X_train,
#                                                                                                                              y_train=y_train, X_test=X_test, y_test=y_test, cfg=cfg, )

#             # ----------------------------------------------------------
#             # Collect all information for this outer fold
#             # ----------------------------------------------------------
#             results.append(
#                 {
#                     "model_name": model_name,
#                     "trial": trial_idx,
#                     "outer_fold": outer_fold_idx,

#                     # inner CV stats (directly from best_trial)
#                     "inner_train_scores": best_trial.user_attrs["inner_train_scores"],
#                     "inner_test_scores": best_trial.user_attrs["inner_test_scores"],
#                     "inner_train_mean": float(
#                         best_trial.user_attrs["inner_train_scores"].mean()
#                     ),
#                     "inner_train_std": float(
#                         best_trial.user_attrs["inner_train_scores"].std()
#                     ),
#                     "inner_test_mean": float(
#                         best_trial.user_attrs["inner_test_scores"].mean()
#                     ),
#                     "inner_test_std": float(
#                         best_trial.user_attrs["inner_test_scores"].std()
#                     ),

#                     # outer performance
#                     "outer_train_metrics": outer_train_metrics,
#                     "outer_test_metrics": outer_test_metrics,
#                     "best_params": best_trial.params,
#                     "n_train": len(y_train),
#                     "n_test": len(y_test),
#                     "final_model": final_model,

#                     # bookkeeping of indices and raw labels/scores
#                     "outer_train_idx": outer_train_idx,
#                     "outer_test_idx": outer_test_idx,
#                     "y_train": y_train,
#                     "y_test": y_test,
#                     "y_train_scores": y_train_scores,
#                     "y_test_scores": y_test_scores,
#                 }
#             )

#     return results


def summarize_cv_split_sizes(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Dict[str, Any],
    groups: Optional[np.ndarray] = None,
    model_selection: str = "StratifiedKFold",
    trial_idx: int = 0,
) -> List[Dict[str, Any]]:
    """
    Summarize outer and inner CV train/test sizes for one nested-CV trial,
    reporting BOTH row-level counts and (when `groups` is provided) unique
    group/patient counts.

    This function mirrors the CV configuration used by your nested CV runner
    (via `make_outer_inner_cv`) but does NOT run Optuna or fit any models.
    It is intended to help you build intuition about how many observations
    end up in each split under standard vs. group-aware nested CV.

    Why group-level reporting matters
    -------------------------------
    When `groups` corresponds to patient IDs and each patient has many rows,
    row counts (e.g., 7,921 train rows) can be misleading about effective
    sample size and generalization. In grouped CV, the unit of generalization
    is typically the patient/group. Therefore, this function reports:

      - Row-level sizes and label counts (same behavior as your original code)
      - Group-level sizes and label counts (unique patients per split)

    Assumptions
    -----------
    If `groups` is provided, this function assumes each group has a SINGLE
    label value in `y` (i.e., all rows from a patient share the same y).
    A consistency check is performed; if any group contains multiple labels,
    a ValueError is raised.

    Parameters
    ----------
    X, y : np.ndarray
        Full dataset features and labels (row-level).
    cfg : dict
        Global configuration dict. Uses:
          - cfg["cv"]["n_outer_splits"]
          - cfg["cv"]["n_inner_splits"]
    groups : np.ndarray or None, default=None
        Group labels (e.g., patient IDs), same length as y.
        Required if model_selection == "StratifiedGroupKFold".
    model_selection : str, default="StratifiedKFold"
        CV strategy passed into `make_outer_inner_cv`.
        Common values:
          - "StratifiedKFold"
          - "StratifiedGroupKFold"
    trial_idx : int, default=0
        Outer trial index (used for random_state / shuffling consistency).

    Returns
    -------
    sizes : list of dict
        One dict per outer fold with:
          Row-level:
            - outer_train_size, outer_test_size
            - inner_train_mean_size, inner_test_mean_size
            - outer_train_label_counts, outer_test_label_counts
            - inner_train_mean_label_counts, inner_test_mean_label_counts

          Group-level (only when groups is not None):
            - outer_train_n_groups, outer_test_n_groups
            - inner_train_mean_n_groups, inner_test_mean_n_groups
            - outer_train_group_label_counts, outer_test_group_label_counts
            - inner_train_mean_group_label_counts, inner_test_mean_group_label_counts

        Notes:
        - Inner values are means across inner folds (to match your original output).
        - Group label counts are counts of UNIQUE groups per label.
    """
    cv_cfg = cfg["cv"]
    n_outer_splits = cv_cfg["n_outer_splits"]
    n_inner_splits = cv_cfg["n_inner_splits"]

    if model_selection == "StratifiedGroupKFold" and groups is None:
        raise ValueError(
            "model_selection='StratifiedGroupKFold' but groups is None. "
            "Provide a groups array or use 'StratifiedKFold'."
        )

    outer_cv, inner_cv = make_outer_inner_cv(
        model_selection=model_selection,
        n_outer_splits=n_outer_splits,
        n_inner_splits=n_inner_splits,
        outer_trial_idx=trial_idx,
    )

    unique_labels = np.unique(y)

    # --- Build group -> label mapping (assumes one label per group) ---
    group_to_label = None
    if groups is not None:
        group_to_label = {}
        # Use first occurrence per group, but verify consistency
        for g in np.unique(groups):
            ys = y[groups == g]
            first = ys[0]
            if not np.all(ys == first):
                raise ValueError(
                    f"Group {g} has multiple labels in y. "
                    "Group-level label tracking assumes one label per group."
                )
            group_to_label[g] = int(first)

    def group_label_counts_from_indices(idxs: np.ndarray) -> Dict[int, int]:
        """Counts of labels at the group/patient level for a subset of rows."""
        if groups is None or group_to_label is None:
            return {}
        g_unique = np.unique(groups[idxs])
        counts = {int(lbl): 0 for lbl in unique_labels}
        for g in g_unique:
            counts[group_to_label[g]] += 1
        return counts

    def n_groups_from_indices(idxs: np.ndarray) -> int:
        if groups is None:
            return 0
        return int(np.unique(groups[idxs]).size)

    outer_splits = outer_cv.split(X, y, groups) if groups is not None else outer_cv.split(X, y)

    sizes: List[Dict[str, Any]] = []

    for outer_fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_splits, start=1):
        X_train, X_test = X[outer_train_idx], X[outer_test_idx]
        y_train, y_test = y[outer_train_idx], y[outer_test_idx]
        groups_train = groups[outer_train_idx] if groups is not None else None

        # ---- ROW-LEVEL totals ----
        outer_train_size = len(y_train)
        outer_test_size = len(y_test)

        outer_train_label_counts = {int(lbl): int((y_train == lbl).sum()) for lbl in unique_labels}
        outer_test_label_counts  = {int(lbl): int((y_test == lbl).sum()) for lbl in unique_labels}

        # ---- PATIENT/GROUP-LEVEL totals ----
        outer_train_n_groups = n_groups_from_indices(outer_train_idx) if groups is not None else None
        outer_test_n_groups  = n_groups_from_indices(outer_test_idx)  if groups is not None else None

        outer_train_group_label_counts = (
            group_label_counts_from_indices(outer_train_idx) if groups is not None else None
        )
        outer_test_group_label_counts = (
            group_label_counts_from_indices(outer_test_idx) if groups is not None else None
        )

        # ---- INNER splits on the outer-train subset ----
        if groups_train is not None:
            inner_splits = inner_cv.split(X_train, y_train, groups_train)
        else:
            inner_splits = inner_cv.split(X_train, y_train)

        inner_train_sizes = []
        inner_test_sizes = []
        inner_train_n_groups = []
        inner_test_n_groups = []

        inner_train_label_counts_acc = {int(lbl): [] for lbl in unique_labels}
        inner_test_label_counts_acc  = {int(lbl): [] for lbl in unique_labels}

        inner_train_group_label_counts_acc = {int(lbl): [] for lbl in unique_labels}
        inner_test_group_label_counts_acc  = {int(lbl): [] for lbl in unique_labels}

        for inner_train_idx, inner_test_idx in inner_splits:
            # indices are relative to X_train / y_train
            y_inner_train = y_train[inner_train_idx]
            y_inner_test  = y_train[inner_test_idx]

            inner_train_sizes.append(len(inner_train_idx))
            inner_test_sizes.append(len(inner_test_idx))

            # row-level label counts
            for lbl in unique_labels:
                lbl_int = int(lbl)
                inner_train_label_counts_acc[lbl_int].append(int((y_inner_train == lbl).sum()))
                inner_test_label_counts_acc[lbl_int].append(int((y_inner_test == lbl).sum()))

            # group-level counts (only when groups are provided)
            if groups_train is not None:
                g_inner_train = np.unique(groups_train[inner_train_idx])
                g_inner_test  = np.unique(groups_train[inner_test_idx])

                inner_train_n_groups.append(int(g_inner_train.size))
                inner_test_n_groups.append(int(g_inner_test.size))

                # group label counts: map group -> label via group_to_label (global mapping)
                train_counts = {int(lbl): 0 for lbl in unique_labels}
                test_counts  = {int(lbl): 0 for lbl in unique_labels}

                for g in g_inner_train:
                    train_counts[group_to_label[g]] += 1
                for g in g_inner_test:
                    test_counts[group_to_label[g]] += 1

                for lbl in unique_labels:
                    lbl_int = int(lbl)
                    inner_train_group_label_counts_acc[lbl_int].append(train_counts[lbl_int])
                    inner_test_group_label_counts_acc[lbl_int].append(test_counts[lbl_int])

        inner_train_mean = float(np.mean(inner_train_sizes))
        inner_test_mean  = float(np.mean(inner_test_sizes))

        inner_train_mean_label_counts = {
            lbl: float(np.mean(counts)) for lbl, counts in inner_train_label_counts_acc.items()
        }
        inner_test_mean_label_counts = {
            lbl: float(np.mean(counts)) for lbl, counts in inner_test_label_counts_acc.items()
        }

        # mean group counts (if groups exist)
        inner_train_mean_n_groups = float(np.mean(inner_train_n_groups)) if inner_train_n_groups else None
        inner_test_mean_n_groups  = float(np.mean(inner_test_n_groups)) if inner_test_n_groups else None

        inner_train_mean_group_label_counts = (
            {lbl: float(np.mean(counts)) for lbl, counts in inner_train_group_label_counts_acc.items()}
            if inner_train_n_groups else None
        )
        inner_test_mean_group_label_counts = (
            {lbl: float(np.mean(counts)) for lbl, counts in inner_test_group_label_counts_acc.items()}
            if inner_test_n_groups else None
        )

        sizes.append(
            {
                "outer_fold": outer_fold_idx,

                # row-level totals
                "outer_train_size": outer_train_size,
                "outer_test_size": outer_test_size,
                "inner_train_mean_size": inner_train_mean,
                "inner_test_mean_size": inner_test_mean,

                "outer_train_label_counts": outer_train_label_counts,
                "outer_test_label_counts": outer_test_label_counts,
                "inner_train_mean_label_counts": inner_train_mean_label_counts,
                "inner_test_mean_label_counts": inner_test_mean_label_counts,

                # group-level totals
                "outer_train_n_groups": outer_train_n_groups,
                "outer_test_n_groups": outer_test_n_groups,
                "outer_train_group_label_counts": outer_train_group_label_counts,
                "outer_test_group_label_counts": outer_test_group_label_counts,

                "inner_train_mean_n_groups": inner_train_mean_n_groups,
                "inner_test_mean_n_groups": inner_test_mean_n_groups,
                "inner_train_mean_group_label_counts": inner_train_mean_group_label_counts,
                "inner_test_mean_group_label_counts": inner_test_mean_group_label_counts,
            }
        )

    return sizes


def plot_outer_counts_and_class_percents(
    sizes,
    figsize=(12, 5),
    font_size=12,
    show_segment_percents=True,
    show_total_labels=True,
    legend_loc="best",
    outer_fold_color=["#005CAB", "#D85128"],   # was class_colors
    inner_fold_color=["#EE927B", "#A9EC82"],   # NEW
    class_names=None,
    count_level="rows",
    title=None,
    folds_show=None,
):
    """
    Plot stacked bar charts for nested CV split composition, showing BOTH:
      - Outer split (Train/Test)
      - Inner split (mean Inner Train/mean Inner Test) used for hyperparameter tuning

    Bar layout
    ----------
    For each outer fold, four stacked bars are shown (left → right):
      1) Outer Train
      2) Outer Test
      3) Inner Train (mean across inner folds)
      4) Inner Test  (mean across inner folds)

    A single "Total" stacked bar is also shown on the far left, computed as:
      Total = (Fold 1 Outer Train) + (Fold 1 Outer Test)

    Color behavior
    --------------
    Colors are assigned by class in sorted class-label order (e.g., [0, 1]).
    - `outer_fold_color` is used for Total + Outer bars
    - `inner_fold_color` is used for Inner bars
    Both must provide at least as many colors as there are classes.

    Parameters
    ----------
    sizes : list[dict]
        Output from `summarize_cv_split_sizes`. Must include the outer and inner
        label-count dictionaries. For example (rows):
          - outer_train_label_counts, outer_test_label_counts
          - inner_train_mean_label_counts, inner_test_mean_label_counts
        And for groups (unique patients):
          - outer_train_group_label_counts, outer_test_group_label_counts
          - inner_train_mean_group_label_counts, inner_test_mean_group_label_counts
    figsize : tuple[int, int], default=(12, 5)
        Figure size (width, height).
    font_size : int, default=12
        Base font size for title, axes labels, ticks, and annotations.
    show_segment_percents : bool, default=True
        If True, annotate each stacked segment with its percent of that bar's total.
    show_total_labels : bool, default=True
        If True, annotate each bar with its total count ("N=...") above the bar.
    legend_loc : str, default="best"
        Location of the legend (matplotlib `loc` argument).
    outer_fold_color : list[str], default=["#005CAB", "#D85128"]
        List of color strings (e.g., hex codes) for the classes used on Total + Outer bars.
        Length must be >= number of classes.
    inner_fold_color : list[str], default=["#EE927B", "#A9EC82"]
        List of color strings for the classes used on Inner bars.
        Length must be >= number of classes.
    class_names : list[str] or None, default=None
        Optional display names for classes in the legend, provided in the same order as
        sorted class labels. If None, legend labels are "Class {label}".
    count_level : {"rows", "groups"}, default="rows"
        Which counts to plot:
          - "rows": uses row-level count dicts (outer_*_label_counts, inner_*_mean_label_counts)
          - "groups": uses unique-group/patient dicts (outer_*_group_label_counts, inner_*_mean_group_label_counts)
    title : str or None, default=None
        Optional title override. If None, a default title is used based on `count_level`.
    folds_show : None | int | list[int] | tuple[int] | set[int], default=None
        Controls which outer folds to display (the "Total" bar is always shown):
          - None: show all folds in `sizes`
          - int k: show first k folds
          - iterable: show only the specified fold numbers (e.g., [1, 3, 7])

    Returns
    -------
    None
        Displays a matplotlib figure.
    """

    if count_level not in ("rows", "groups"):
        raise ValueError("count_level must be 'rows' or 'groups'")

    # ---- filter folds if requested ----
    if folds_show is not None:
        if isinstance(folds_show, int):
            sizes = sizes[:folds_show]
        elif isinstance(folds_show, (list, tuple, set)):
            want = set(int(f) for f in folds_show)
            sizes = [s for s in sizes if int(s["outer_fold"]) in want]
        else:
            raise TypeError("folds_show must be None, an int, or an iterable of fold numbers.")
        if len(sizes) == 0:
            raise ValueError("After applying folds_show, no folds remain to plot.")

    # ---- choose which dicts to read ----
    if count_level == "rows":
        outer_train_key = "outer_train_label_counts"
        outer_test_key  = "outer_test_label_counts"
        inner_train_key = "inner_train_mean_label_counts"
        inner_test_key  = "inner_test_mean_label_counts"
        default_title = "Nested CV (Rows): Outer + Inner(mean) split composition"
    else:
        outer_train_key = "outer_train_group_label_counts"
        outer_test_key  = "outer_test_group_label_counts"
        inner_train_key = "inner_train_mean_group_label_counts"
        inner_test_key  = "inner_test_mean_group_label_counts"
        default_title = "Nested CV (Patients): Outer + Inner(mean) split composition"
        if outer_train_key not in sizes[0] or sizes[0][outer_train_key] is None:
            raise ValueError(
                "Group-level keys not found in sizes. "
                "Make sure summarize_cv_split_sizes computed *_group_label_counts."
            )


    classes = sorted(sizes[0][outer_train_key].keys())
    n_classes = len(classes)

    # Validate colors (lists only; no fallbacks)
    if not isinstance(outer_fold_color, list):
        raise TypeError("outer_fold_color must be a list of color strings.")
    if not isinstance(inner_fold_color, list):
        raise TypeError("inner_fold_color must be a list of color strings.")
    if len(outer_fold_color) < n_classes:
        raise ValueError(f"Need at least {n_classes} colors in outer_fold_color; got {len(outer_fold_color)}.")
    if len(inner_fold_color) < n_classes:
        raise ValueError(f"Need at least {n_classes} colors in inner_fold_color; got {len(inner_fold_color)}.")

    if class_names is not None:
        if not isinstance(class_names, list):
            raise TypeError("class_names must be a list of strings or None.")
        if len(class_names) < n_classes:
            raise ValueError(f"Need at least {n_classes} names in class_names; got {len(class_names)}.")

    outer_class_to_color = {c: outer_fold_color[i] for i, c in enumerate(classes)}
    inner_class_to_color = {c: inner_fold_color[i] for i, c in enumerate(classes)}

    def label_for(i, c):
        return class_names[i] if class_names is not None else f"Class {c}"

    folds = [s["outer_fold"] for s in sizes]
    fold_labels = ["Total"] + [f"Fold {f}" for f in folds]

    def counts_matrix(key):
        return np.array([[s[key].get(c, 0) for c in classes] for s in sizes], dtype=float)

    outer_train_mat = counts_matrix(outer_train_key)
    outer_test_mat  = counts_matrix(outer_test_key)
    inner_train_mat = counts_matrix(inner_train_key)
    inner_test_mat  = counts_matrix(inner_test_key)

    total_counts = np.array(
        [sizes[0][outer_train_key].get(c, 0) + sizes[0][outer_test_key].get(c, 0) for c in classes],
        dtype=float
    )

    x = np.arange(len(fold_labels))

    # Order per fold: Outer Train, Outer Test, Inner Train(mean), Inner Test(mean)
    w = 0.18
    gap = 0.02
    off_outer_train = -(1.5 * w + 1.5 * gap)
    off_outer_test  = -(0.5 * w + 0.5 * gap)
    off_inner_train = +(0.5 * w + 0.5 * gap)
    off_inner_test  = +(1.5 * w + 1.5 * gap)

    fig, ax = plt.subplots(figsize=figsize)

    def annotate_segment(xpos, bottom, height, total):
        if height <= 0 or total <= 0:
            return
        pct = 100.0 * height / total
        ax.text(
            xpos, bottom + height / 2, f"{pct:.1f}%",
            ha="center", va="center",
            fontsize=font_size - 2, fontweight="bold",
        )

    def annotate_total(xpos, total):
        ax.text(
            xpos, total, f"N={int(round(total))}",
            ha="center", va="bottom",
            fontsize=font_size - 2, fontweight="bold",
        )

    def draw_stacked_bar(xpos, counts_vec, color_map):
        bottom = 0.0
        total = float(np.sum(counts_vec))
        for i, c in enumerate(classes):
            h = float(counts_vec[i])
            ax.bar(xpos, h, w, bottom=bottom, color=color_map[c])
            if show_segment_percents:
                annotate_segment(xpos, bottom, h, total)
            bottom += h
        if show_total_labels:
            annotate_total(xpos, total)

    # TOTAL (use OUTER colors)
    draw_stacked_bar(x[0] + off_outer_train, total_counts, outer_class_to_color)

    for fi in range(len(sizes)):
        xi = x[fi + 1]
        draw_stacked_bar(xi + off_outer_train, outer_train_mat[fi], outer_class_to_color)
        draw_stacked_bar(xi + off_outer_test,  outer_test_mat[fi],  outer_class_to_color)
        draw_stacked_bar(xi + off_inner_train, inner_train_mat[fi], inner_class_to_color)
        draw_stacked_bar(xi + off_inner_test,  inner_test_mat[fi],  inner_class_to_color)

    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels, fontsize=font_size, fontweight="bold")
    ax.set_ylabel("Count", fontsize=font_size, fontweight="bold")
    ax.set_title(title or default_title, fontsize=font_size + 2, fontweight="bold")

    # Legend: classes only (use OUTER colors so legend stays consistent)
    legend_handles = [
        Patch(facecolor=outer_class_to_color[c], label=label_for(i, c))
        for i, c in enumerate(classes)
    ]
    ax.legend(handles=legend_handles, loc=legend_loc, fontsize=font_size - 1,
              ncols=min(3, n_classes))

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_metric_over_trials(
    eval_results: Mapping[str, Sequence[Mapping[str, Any]]],
    model_names: str | Sequence[str] | None = None,   # None -> all models
    metric_name: Literal["average_precision", "roc_auc", "brier_score_loss", "log_loss"] = "average_precision",
    use_calibrated: bool = False,
    calibration_method: str | None = None,
    include_uncalib_oof: bool = False,
    split_palette: dict[str, str] | None = None,
    method_alias: Mapping[str, str] | None = None, 
    figsize: tuple[float, float] = (8, 4),
    font_size: int = 12,
    legend_loc: str = "best",
    xtick_step: Optional[int] = None,
    y_lim: Optional[tuple[float, float]] = None,
    # ---- baseline / prevalence line ----
    show_prevalence_baseline: bool = True,
    baseline_color: str = "#D5F713",
    baseline_lw: float = 1.5,
    baseline_ls: str = "--",
) -> None:
    """
    Plot a metric across trials for one or more models with Train vs Test curves
    (mean ± SD across folds per trial).

    Parameters:
    -------------

      eval_results: dict(model_name -> list of fold dicts containing trial + metric keys).
      model_names: model(s) to plot; None = all models in eval_results.
      metric_name: which metric to plot ("average_precision", "roc_auc", "brier_score_loss", "log_loss").
      use_calibrated: if True, use calibrated metric keys for `calibration_method`.
      calibration_method: calibration suffix used to form calibrated keys (e.g. "beta").
      include_uncalib_oof: if True (and not calibrated), Train uses OOF metrics; Test uses outer test.
      split_palette: colors for {"Train": ..., "Test": ...} (defaults to blue/red).
      method_alias: optional mapping model_key -> display name (used in plot title only).
      figsize: matplotlib figure size per model plot.
      font_size: base font size for labels/ticks/title.
      legend_loc: legend location string.
      xtick_step: show every k-th trial on x-axis (None = auto-subsample).
      y_lim: optional (ymin, ymax) for the y-axis.
      show_prevalence_baseline: if True, draw the appropriate baseline (AUROC=0.5; others from prevalence).
            - AUROC: y = 0.5
            - AUPRC: y = p
            - Brier: y = p(1 - p)
            - LogLoss: y = -[p log(p) + (1 - p) log(1 - p)]
        where p is computed as the mean of `entry["prevalence"]` across all folds
        for the model (simple, robust to slight fold-to-fold variation).

      baseline_color/baseline_lw/baseline_ls: baseline line styling.
    """

    # -------------------------
    # Defaults
    # -------------------------
    if split_palette is None:
        split_palette = {"Train": "#1587F8", "Test": "#F14949"}
    for k in ("Train", "Test"):
        if k not in split_palette:
            raise ValueError(f"split_palette must contain '{k}'. Got keys: {list(split_palette.keys())}")

    if method_alias is None:
        method_alias = {}

    if include_uncalib_oof and use_calibrated:
        raise ValueError("include_uncalib_oof=True is only supported when use_calibrated=False.")

    # -------------------------
    # Choose models
    # -------------------------
    if model_names is None:
        selected_models = list(eval_results.keys())
    elif isinstance(model_names, str):
        selected_models = [model_names]
    else:
        selected_models = list(model_names)

    missing_models = [m for m in selected_models if m not in eval_results]
    if missing_models:
        raise KeyError(
            f"Model(s) not found in eval_results: {missing_models}. "
            f"Available: {list(eval_results.keys())}"
        )

    # Display labels (aliasing only affects display)
    model_labels = [method_alias.get(m, m) for m in selected_models]
    if len(set(model_labels)) != len(model_labels):
        dupes = pd.Series(model_labels)[pd.Series(model_labels).duplicated(keep=False)].unique().tolist()
        raise ValueError(
            f"method_alias causes duplicate model labels {dupes}. "
            f"Make aliases unique (or omit aliasing for colliding model names)."
        )

    # Map internal -> display label (for titles)
    display_name = {m: method_alias.get(m, m) for m in selected_models}

    # -------------------------
    # Metric label mapping
    # -------------------------
    metric_label_map: dict[str, str] = {
        "average_precision": "AUPRC",
        "roc_auc": "AUROC",
        "brier_score_loss": "Brier score loss",
        "log_loss": "Log loss",
    }
    metric_label = metric_label_map[metric_name]

    # -------------------------
    # Choose metric keys
    # -------------------------
    if not use_calibrated:
        if include_uncalib_oof:
            train_key = f"cv_uncalib_train_{metric_name}"
            test_key = f"outer_test_{metric_name}"
            title_suffix = " (uncalibrated OOF-train)"
        else:
            train_key = f"outer_train_{metric_name}"
            test_key = f"outer_test_{metric_name}"
            title_suffix = " (uncalibrated)"
    else:
        if calibration_method is None:
            raise ValueError("calibration_method must be provided when use_calibrated=True.")
        train_key = f"cv_calib_train_{calibration_method}_{metric_name}"
        test_key = f"calib_test_{calibration_method}_{metric_name}"
        title_suffix = f" (calibrated: {calibration_method})"

        # sanity check keys exist on first fold of each model
        for m in selected_models:
            if len(eval_results[m]) == 0:
                raise ValueError(f"eval_results['{m}'] is empty.")
            first = eval_results[m][0]
            for k in (train_key, test_key):
                if k not in first:
                    raise KeyError(
                        f"Key '{k}' not found for model '{m}'. "
                        f"Did you compute calibrated metrics for '{calibration_method}'?"
                    )

    sns.set(style="whitegrid")

    # -------------------------
    # Plot one figure per model
    # -------------------------
    for m in selected_models:
        folds = eval_results[m]
        if len(folds) == 0:
            raise ValueError(f"No fold entries for model '{m}'.")

        # ---- compute prevalence baseline (mean across folds) ----
        p_mean: float | None = None
        if show_prevalence_baseline and metric_name in {"average_precision", "brier_score_loss", "log_loss"}:
            prev_vals = [float(r["prevalence"]) for r in folds if "prevalence" in r]
            if len(prev_vals) == 0:
                raise KeyError(
                    f"Missing 'prevalence' in eval_results for model '{m}'. "
                    "Update evaluate_nested_cv_results to store it."
                )
            p_mean = float(np.mean(prev_vals))

        # Determine baseline y-value + label (if enabled)
        baseline_y: float | None = None
        baseline_label: str | None = None

        if show_prevalence_baseline:
            if metric_name == "roc_auc":
                baseline_y = 0.5
                baseline_label = "Baseline = 0.50"
            elif metric_name == "average_precision":
                baseline_y = float(p_mean) if p_mean is not None else None
                if baseline_y is not None:
                    baseline_label = f"Baseline = {baseline_y:.2f}"
            elif metric_name == "brier_score_loss":
                if p_mean is not None:
                    baseline_y = float(p_mean * (1.0 - p_mean))
                    baseline_label = f"Baseline = {baseline_y:.2f}"
            elif metric_name == "log_loss":
                if p_mean is not None:
                    p = float(p_mean)
                    baseline_y = float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))
                    baseline_label = f"Baseline = {baseline_y:.2f}"

        # ---- Build tidy DF: one row per (trial, split) ----
        rows: list[dict[str, Any]] = []
        for r in folds:
            if "trial" not in r:
                raise KeyError(f"Missing key 'trial' in fold dict for model '{m}'.")
            if train_key not in r or test_key not in r:
                raise KeyError(
                    f"Missing required metric keys for model '{m}'. "
                    f"Need '{train_key}' and '{test_key}'."
                )

            trial = r["trial"]
            rows.append({"trial": trial, "split": "Train", "score": r[train_key]})
            rows.append({"trial": trial, "split": "Test",  "score": r[test_key]})

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError(f"No rows to plot for model '{m}' (check eval_results structure).")

        plt.figure(figsize=figsize)
        ax = sns.lineplot(
            data=df,
            x="trial",
            y="score",
            hue="split",
            hue_order=["Train", "Test"],
            estimator=np.mean,
            errorbar=("sd"),
            marker="o",
            palette=split_palette,
        )

        # ---- Baseline line (optional) ----
        if baseline_y is not None:
            ax.axhline(
                float(baseline_y),
                ls=baseline_ls,
                lw=baseline_lw,
                color=baseline_color,
                label=baseline_label,
            )

        # ---- x-ticks: integer trials, with optional subsampling ----
        trials = sorted(df["trial"].unique())
        n_trials = len(trials)

        if xtick_step is None:
            max_labels = 10
            step = max(1, n_trials // max_labels)
        else:
            step = max(1, int(xtick_step))

        shown_trials = trials[::step]
        ax.set_xticks(shown_trials)
        ax.set_xticklabels([str(int(t)) for t in shown_trials])

        # ---- Labels / title / legend ----
        ax.set_xlabel("Trial", fontsize=font_size, fontweight="bold")
        ax.set_ylabel("Score", fontsize=font_size, fontweight="bold")
        ax.set_title(
            f"{metric_label} across trials for {display_name[m]}{title_suffix}",
            fontsize=font_size + 2,
            fontweight="bold",
        )

        ax.tick_params(axis="both", labelsize=font_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        if y_lim is not None:
            ax.set_ylim(*y_lim)

        ax.legend(title="", loc=legend_loc, prop={"size": font_size, "weight": "bold"})
        plt.tight_layout()
        plt.show()



def evaluate_nested_cv_results(
    all_results: Mapping[str, Sequence[Mapping[str, Any]]],
    metrics_to_compute: Optional[list[str]] = None,
    calibration_methods: Optional[list[str]] = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Recompute metrics from stored y/scores for:

      - Uncalibrated outer-train / outer-test scores
      - Uncalibrated OOF train scores
      - Calibrated train/test scores for each calibration method

    Supported metrics (by name):
      - "average_precision"
      - "roc_auc"
      - "brier_score_loss"
      - "log_loss"

    Parameters
    ----------
    all_results : Mapping[str, Sequence[Mapping[str, Any]]]
        {
          "logistic_regression": [ { ... per outer fold ... }, ... ],
          ...
        }
        Each fold dict MUST contain:
          - "y_train", "y_train_scores"
          - "y_test",  "y_test_scores"
        And, if add_cv_oof_predictions has been run:
          - "cv_uncalib_train_predictions"
          - "cv_calib_train_predictions_<method>"
          - "calib_test_predictions_<method>"

    metrics_to_compute : list[str] or None
        If None, defaults to ["average_precision", "roc_auc", "brier_score_loss", "log_loss"].

    calibration_methods : list[str] or None
        List of calibration method names that may be present in all_results,
        e.g. ["platt", "beta"]. If None, defaults to [] (no calibrated metrics).

    Returns
    -------
    eval_results : dict[str, list[dict[str, Any]]]
        Fold-level entries keyed by model name. Each fold entry includes:
          - "model_name", "trial", "outer_fold"
          - "prevalence": positive-class prevalence (computed from y_test)
          - metric keys per the conventions below

        Metric key conventions:
          Uncalibrated final model:
            - outer_train_<metric>, outer_test_<metric>
          Uncalibrated OOF train (if present):
            - cv_uncalib_train_<metric>
          Calibrated (per method, if present):
            - cv_calib_train_<method>_<metric>
            - calib_test_<method>_<metric>
    """
    if metrics_to_compute is None:
        metrics_to_compute = [
            "average_precision",
            "roc_auc",
            "brier_score_loss",
            "log_loss",
        ]

    if calibration_methods is None:
        calibration_methods = []

    # Map metric name -> sklearn function
    metric_fns: dict[str, Any] = {}
    for m in metrics_to_compute:
        if m == "average_precision":
            metric_fns[m] = metrics.average_precision_score
        elif m == "roc_auc":
            metric_fns[m] = metrics.roc_auc_score
        elif m == "brier_score_loss":
            metric_fns[m] = metrics.brier_score_loss
        elif m == "log_loss":
            metric_fns[m] = metrics.log_loss
        else:
            raise ValueError(f"Unsupported metric: {m}")

    eval_results: dict[str, list[dict[str, Any]]] = {}

    for model_name, folds in all_results.items():
        model_entries: list[dict[str, Any]] = []

        for r in folds:
            y_train = np.asarray(r["y_train"])
            y_train_scores = np.asarray(r["y_train_scores"])  # final model, uncalib
            y_test = np.asarray(r["y_test"])
            y_test_scores = np.asarray(r["y_test_scores"])    # final model, uncalib

            # Prevalence (positive rate). With stratified group k-fold, this should be stable.
            prevalence = float(np.mean(y_test))

            entry: dict[str, Any] = {
                "model_name": r.get("model_name", model_name),
                "trial": r["trial"],
                "outer_fold": r["outer_fold"],
                "prevalence": prevalence,
            }

            for m_name, scorer in metric_fns.items():
                # 1) Uncalibrated final model: outer train/test
                entry[f"outer_train_{m_name}"] = float(scorer(y_train, y_train_scores))
                entry[f"outer_test_{m_name}"] = float(scorer(y_test, y_test_scores))

                # 2) Uncalibrated OOF train (if present)
                if "cv_uncalib_train_predictions" in r:
                    entry[f"cv_uncalib_train_{m_name}"] = float(
                        scorer(y_train, np.asarray(r["cv_uncalib_train_predictions"]))
                    )

                # 3) Calibrated metrics per method (if keys exist)
                for method in calibration_methods:
                    train_key = f"cv_calib_train_predictions_{method}"
                    test_key = f"calib_test_predictions_{method}"

                    if train_key in r:
                        entry[f"cv_calib_train_{method}_{m_name}"] = float(
                            scorer(y_train, np.asarray(r[train_key]))
                        )
                    if test_key in r:
                        entry[f"calib_test_{method}_{m_name}"] = float(
                            scorer(y_test, np.asarray(r[test_key]))
                        )

            model_entries.append(entry)

        eval_results[model_name] = model_entries

    return eval_results



def extract_metric_values(
    eval_results,
    metrics_to_compute=None,
    calibration_methods=None,
):
    """
    Extract arrays of metric values for each model, metric, and prediction type.

    Output structure:
    {
      "logistic_regression": {
        "average_precision": {
          "outer_train":        np.array([...]),
          "outer_test":         np.array([...]),
          "cv_uncalib_train":   np.array([...]),
          "cv_calib_train_platt": np.array([...]),
          "calib_test_platt":     np.array([...]),
          "cv_calib_train_beta":  np.array([...]),
          "calib_test_beta":      np.array([...]),
        },
        "roc_auc": { ... },
      },
      "some_other_model": { ... },
    }
    """
    if metrics_to_compute is None:
        metrics_to_compute = ["average_precision", "roc_auc","brier_score_loss", "log_loss"]

    if calibration_methods is None:
        calibration_methods = []

    all_values = {}

    for model_name, folds in eval_results.items():
        model_values = {}

        for m_name in metrics_to_compute:
            metric_dict = {}

            # Base uncalibrated outer train/test
            metric_dict["outer_train"] = np.array(
                [f[f"outer_train_{m_name}"] for f in folds]
            )
            metric_dict["outer_test"] = np.array(
                [f[f"outer_test_{m_name}"] for f in folds]
            )

            # Uncalibrated OOF train (if present)
            key_unc = f"cv_uncalib_train_{m_name}"
            if key_unc in folds[0]:
                metric_dict["cv_uncalib_train"] = np.array(
                    [f[key_unc] for f in folds]
                )

            # Calibrated, per method (if present)
            for method in calibration_methods:
                k_train = f"cv_calib_train_{method}_{m_name}"
                k_test = f"calib_test_{method}_{m_name}"

                if k_train in folds[0]:
                    metric_dict[f"cv_calib_train_{method}"] = np.array(
                        [f[k_train] for f in folds]
                    )
                if k_test in folds[0]:
                    metric_dict[f"calib_test_{method}"] = np.array(
                        [f[k_test] for f in folds]
                    )

            model_values[m_name] = metric_dict

        all_values[model_name] = model_values

    return all_values


def plot_performance_curves(
    all_results: dict[str, list[dict[str, Any]]],
    model_names: Sequence[str] | str | None = None,
    curve: Literal["auroc", "auprc", "bacacc"] = "auroc",
    use_calibrated: bool = False,
    calibration_method: str | None = None,
    split: Literal["train", "test"] = "test",
    figsize: tuple[float, float] = (8, 6),
    font_size: int = 13,
    alpha_folds: float = 0.08,
    lw_folds: float = 2.0,
    lw_mean: float = 2.5,
    show_std_band: bool = True,
    std_band_alpha: float = 0.15,
    n_grid: int = 100,
    # ---- colors ----
    fold_color: str = "gray",
    mean_color: str = "#005CAB",
    band_color: str = "#6699CC",
    refline_color: str = "#D85128",
    # ---- baseline / prevalence line ----
    show_prevalence_baseline: bool = True,
    baseline_color: str = "#D5F713",
    baseline_lw: float = 1.5,
    baseline_ls: str = "--",
    # ---- legend ----
    show_legend: bool = True,
    legend_loc: str = "lower right",
    # ---- NEW: aliasing ----
    method_alias: Mapping[str, str] | None = None,
) -> None:
    """
    Plot final performance curves across outer folds for one or more models from nested-CV results.

    Supported curve types
    ---------------------
    - "auroc": ROC curve (TPR vs FPR) with AUROC computed per fold and summarized.
    - "auprc": Precision–Recall curve (Precision vs Recall) with AUPRC computed per fold and summarized.
    - "bacacc": Balanced Accuracy vs Threshold curve on a fixed threshold grid [0..1].
        * Fold-level scalar summary reported is the best balanced accuracy on the chosen split
            (max over the threshold grid).

    What the plot shows
    -------------------
    - Fold-level curves overlaid as faint lines (variability “cloud”),
    - A mean curve computed by interpolating each fold onto a common x-grid (AUROC/AUPRC),
        or direct averaging on the threshold grid (bacacc),
    - Optional ±1 standard deviation band around the mean curve,
    - Optional baseline reference line (`show_prevalence_baseline=True`):
        * AUROC: diagonal line (random chance)
        * AUPRC: horizontal line at prevalence (mean prevalence across folds)
        * bacacc: horizontal line at 0.50 (random guessing)

    Legend / summary
    ----------------
    - The legend reports the fold-level scalar metric as mean ± SD across folds (e.g., AUROC mean ± SD).
    - The ±1 SD band corresponds to pointwise variability of the curve across folds (on the common grid).

    Score sources
    -------------
    - Uncalibrated scores:
        * Train: "y_train_scores"
        * Test : "y_test_scores"
    - Calibrated scores (if stored in your pipeline):
        * Train: "cv_calib_train_predictions_<method>"
        * Test : "calib_test_predictions_<method>"

    Parameters
    ----------
    all_results:
        Nested-CV results keyed by model name; each model maps to a list of fold dicts.

    model_names:
        Model(s) to plot:
        - None: plot all models in all_results
        - str: plot one model
        - sequence[str]: plot multiple models

    curve:
        "auroc", "auprc", or "bacacc".

    use_calibrated / calibration_method:
        If use_calibrated=True, calibration_method must be provided (e.g., "platt", "beta").

    split:
        "train" or "test" selection for y/score arrays.

    method_alias:
        Optional mapping from internal model keys to display names (title/labels only).

    show_prevalence_baseline:
        If True, draw the baseline reference line for the chosen curve type.

    baseline_color / baseline_lw / baseline_ls:
        Style for the baseline reference line (used for AUPRC and bacacc; AUROC uses this style for the
        diagonal baseline line).

    Returns
    -------
    None
        Displays matplotlib plots (one per model).
    """

    # -------------------------
    # Validate inputs
    # -------------------------
    if use_calibrated and calibration_method is None:
        raise ValueError("calibration_method must be provided when use_calibrated=True.")
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if curve not in {"auroc", "auprc", "bacacc"}:
        raise ValueError("curve must be 'auroc', 'auprc', or 'bacacc'")

    if method_alias is None:
        method_alias = {}

    # -------------------------
    # Choose models
    # -------------------------
    if model_names is None:
        selected = list(all_results.keys())
    elif isinstance(model_names, str):
        selected = [model_names]
    else:
        selected = list(model_names)

    missing = [m for m in selected if m not in all_results]
    if missing:
        raise KeyError(
            f"Model(s) not found in all_results: {missing}. "
            f"Available: {list(all_results.keys())}"
        )

    # sanity check for duplicate display labels (optional but helpful)
    display_labels = [method_alias.get(m, m) for m in selected]
    if len(set(display_labels)) != len(display_labels):
        dupes = (
            [x for x in display_labels if display_labels.count(x) > 1]
        )
        raise ValueError(
            f"method_alias causes duplicate model labels: {sorted(set(dupes))}. "
            "Make aliases unique."
        )

    grid = np.linspace(0.0, 1.0, n_grid)

    # -------------------------
    # Plot one figure per model
    # -------------------------
    for model_name in selected:
        folds = all_results[model_name]
        display_name = method_alias.get(model_name, model_name)

        # -------------------------
        # Choose y/score keys
        # -------------------------
        if not use_calibrated:
            if split == "train":
                score_key, y_key = "y_train_scores", "y_train"
            else:
                score_key, y_key = "y_test_scores", "y_test"
            title_suffix = " (uncalibrated)"
        else:
            if split == "train":
                score_key, y_key = f"cv_calib_train_predictions_{calibration_method}", "y_train"
            else:
                score_key, y_key = f"calib_test_predictions_{calibration_method}", "y_test"
            title_suffix = f" (calibrated: {calibration_method})"

            if folds and score_key not in folds[0]:
                raise KeyError(
                    f"Key '{score_key}' not found for model '{model_name}'. "
                    f"Did you run calibration with method '{calibration_method}'?"
                )

        metric_vals: list[float] = []
        interp_curves: list[np.ndarray] = []
        prevalences: list[float] = []

        plt.figure(figsize=figsize)

        # -------------------------
        # Overlay fold curves
        # -------------------------
        for r in folds:
            if y_key not in r or score_key not in r:
                continue

            y_true = np.asarray(r[y_key])
            y_score = np.asarray(r[score_key])

            if curve == "auroc":
                x, y, _ = roc_curve(y_true, y_score)  # x=fpr, y=tpr
                metric = roc_auc_score(y_true, y_score)

                plt.plot(x, y, alpha=alpha_folds, linewidth=lw_folds, color=fold_color)

                y_i = np.interp(grid, x, y)
                y_i[0] = 0.0
                interp_curves.append(y_i)
                metric_vals.append(float(metric))

            elif curve == "auprc":
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                metric = average_precision_score(y_true, y_score)

                plt.plot(recall, precision, alpha=alpha_folds, linewidth=lw_folds, color=fold_color)

                sort_idx = np.argsort(recall)
                recall_s = recall[sort_idx]
                precision_s = precision[sort_idx]
                y_i = np.interp(grid, recall_s, precision_s)
                interp_curves.append(y_i)

                prevalences.append(float(np.mean(y_true)))
                metric_vals.append(float(metric))

            else:  # curve == "bacacc"
                bacacc_vals = np.array(
                    [balanced_accuracy_score(y_true, (y_score >= t).astype(int)) for t in grid],
                    dtype=float,
                )
                plt.plot(grid, bacacc_vals, alpha=alpha_folds, linewidth=lw_folds, color=fold_color)
                interp_curves.append(bacacc_vals)
                metric_vals.append(float(np.max(bacacc_vals)))

        if len(interp_curves) == 0:
            raise ValueError("No fold curves were computed (missing keys or empty folds).")

        # -------------------------
        # Mean curve + std band
        # -------------------------
        interp_mat = np.vstack(interp_curves)
        mean_curve = interp_mat.mean(axis=0)
        std_curve = interp_mat.std(axis=0)
        metric_arr = np.asarray(metric_vals, dtype=float)

        # scalar SD across folds (this is what we show numerically in the legend)
        metric_sd = float(metric_arr.std(ddof=1)) if metric_arr.size > 1 else 0.0

        # -------------------------
        # Plot mean + band + baseline
        # -------------------------
        if curve == "auroc":
            mean_curve[-1] = 1.0

            plt.plot(
                grid,
                mean_curve,
                linewidth=lw_mean,
                color=mean_color,
                label=f"Mean AUROC = {metric_arr.mean():.3f} ± {metric_sd:.3f}" if show_legend else None,
            )

            if show_std_band:
                plt.fill_between(
                    grid,
                    np.clip(mean_curve - std_curve, 0, 1),
                    np.clip(mean_curve + std_curve, 0, 1),
                    alpha=std_band_alpha,
                    color=band_color,
                    label="±1 SD band" if show_legend else None,
                )

            if show_prevalence_baseline:
                plt.plot(
                    [0, 1],
                    [0, 1],
                    linestyle=baseline_ls,
                    linewidth=baseline_lw,
                    color=baseline_color,
                    label="Baseline" if show_legend else None,
                )

            plt.xlabel("False Positive Rate", fontsize=font_size, fontweight="bold")
            plt.ylabel("True Positive Rate", fontsize=font_size, fontweight="bold")
            title_metric = "AUROC"

        elif curve == "auprc":
            plt.plot(
                grid,
                mean_curve,
                linewidth=lw_mean,
                color=mean_color,
                label=f"Mean AUPRC = {metric_arr.mean():.3f} ± {metric_sd:.3f}" if show_legend else None,
            )

            if show_std_band:
                plt.fill_between(
                    grid,
                    np.clip(mean_curve - std_curve, 0, 1),
                    np.clip(mean_curve + std_curve, 0, 1),
                    alpha=std_band_alpha,
                    color=band_color,
                    label="±1 SD band" if show_legend else None,
                )

            if show_prevalence_baseline:
                baseline = float(np.mean(prevalences)) if prevalences else float(np.mean(folds[0][y_key]))
                plt.hlines(
                    baseline,
                    0,
                    1,
                    linestyles=baseline_ls,
                    linewidth=baseline_lw,
                    color=baseline_color,
                    label=f"Baseline = {baseline:.3f}" if show_legend else None,
                )

            plt.xlabel("Recall", fontsize=font_size, fontweight="bold")
            plt.ylabel("Precision", fontsize=font_size, fontweight="bold")
            title_metric = "AUPRC"

        else:  # bacacc
            plt.plot(
                grid,
                mean_curve,
                linewidth=lw_mean,
                color=mean_color,
                label=f"Mean best bacacc = {metric_arr.mean():.3f} ± {metric_sd:.3f}" if show_legend else None,
            )

            if show_std_band:
                plt.fill_between(
                    grid,
                    np.clip(mean_curve - std_curve, 0, 1),
                    np.clip(mean_curve + std_curve, 0, 1),
                    alpha=std_band_alpha,
                    color=band_color,
                    label="±1 SD band" if show_legend else None,
                )

            if show_prevalence_baseline:
                plt.hlines(
                    0.5,
                    0,
                    1,
                    linestyles=baseline_ls,
                    linewidth=baseline_lw,
                    color=baseline_color,
                    label="Baseline = 0.50" if show_legend else None,
                )

            plt.xlabel("Threshold", fontsize=font_size, fontweight="bold")
            plt.ylabel("Balanced accuracy", fontsize=font_size, fontweight="bold")
            title_metric = "Balanced Accuracy"

        # -------------------------
        # Single-line title (no redundant second line)
        # -------------------------
        plt.title(
            f"{title_metric} curves — {display_name}{title_suffix} [{split}]",
            fontsize=font_size + 2,
            fontweight="bold",
        )

        plt.xticks(fontsize=font_size, fontweight="bold")
        plt.yticks(fontsize=font_size, fontweight="bold")

        if show_legend:
            plt.legend(loc=legend_loc, frameon=True)

        plt.tight_layout()
        plt.show()


def auroc_auprc_curve(
    all_results,
    model_name: str,
    curve: str = "roc",                 # "roc" or "pr"
    use_calibrated: bool = False,
    calibration_method: str | None = None,  # "platt" or "beta"
    split: str = "test",                # "train" or "test"
    figsize=(8, 6),
    font_size: int = 13,
    alpha_folds: float = 0.08,
    lw_folds: float = 2.0,
    lw_mean: float = 2.5,
    show_std_band: bool = True,
    std_band_alpha: float = 0.15,
    n_grid: int = 100,
    # ---- colors ----
    fold_color: str = "gray",
    mean_color: str = "#005CAB",
    band_color: str = "#6699CC",
    refline_color: str = "#D85128",
    # ---- legend ----
    show_legend: bool = True,
    legend_loc: str = "lower right",
):
    """
    Plot ROC (AUROC) or Precision–Recall (AUPRC) curves across outer folds for a given model.

    The plot shows:
      - All fold-level curves overlaid as faint lines (a “cloud” showing variability),
      - A mean curve computed by interpolating each fold’s curve onto a common x-grid,
      - An optional ±1 standard deviation band around the mean curve,
      - A baseline reference line (random-chance diagonal for ROC; prevalence for PR),
        plus an optional legend.

    Supports uncalibrated scores (y_*_scores) or calibrated scores from a chosen
    method (e.g., "platt", "beta") using keys produced by your calibration pipeline:
      - Train calibrated scores: cv_calib_train_predictions_<method>
      - Test calibrated scores : calib_test_predictions_<method>

    Parameters
    ----------
    all_results : dict
        Nested-CV results keyed by model_name, where each fold dict contains y_true and
        score vectors (uncalibrated and/or calibrated).

    model_name : str
        Name of the model key in all_results (e.g., "logistic_regression").

    curve : {"roc", "pr"}, default="roc"
        Which curve to plot: ROC (AUROC) or Precision–Recall (AUPRC).

    use_calibrated : bool, default=False
        If True, plot curves using calibrated predictions for calibration_method.

    calibration_method : str or None, default=None
        Required if use_calibrated=True. Example: "platt" or "beta".

    split : {"train", "test"}, default="test"
        Whether to plot curves from outer-train or outer-test predictions.

    figsize : tuple, default=(8, 6)
        Figure size for matplotlib.

    font_size : int, default=13
        Base font size for title and axis labels.

    alpha_folds : float, default=0.08
        Transparency for individual fold curves.

    lw_folds : float, default=2.0
        Line width for individual fold curves.

    lw_mean : float, default=2.5
        Line width for the mean curve.

    show_std_band : bool, default=True
        Whether to show the mean ± 1 SD shaded band.

    std_band_alpha : float, default=0.15
        Transparency for the SD band.

    n_grid : int, default=100
        Number of points in the common interpolation grid (FPR for ROC, Recall for PR).

    fold_color, mean_color, band_color, refline_color : str
        Colors for fold curves, mean curve, SD band, and baseline reference line.

    show_legend : bool, default=True
        Whether to display a legend.

    legend_loc : str, default="lower right"
        Legend location passed to matplotlib.

    Returns
    -------
    None
        Displays the matplotlib plot.
    """

    if model_name not in all_results:
        raise KeyError(f"Model '{model_name}' not found in all_results.")
    folds = all_results[model_name]

    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if curve not in {"roc", "pr"}:
        raise ValueError("curve must be 'roc' or 'pr'")

    # Choose score key
    if not use_calibrated:
        if split == "train":
            score_key, y_key = "y_train_scores", "y_train"
        else:
            score_key, y_key = "y_test_scores", "y_test"
        title_suffix = " (uncalibrated)"
    else:
        if calibration_method is None:
            raise ValueError("calibration_method must be provided when use_calibrated=True.")
        if split == "train":
            score_key, y_key = f"cv_calib_train_predictions_{calibration_method}", "y_train"
        else:
            score_key, y_key = f"calib_test_predictions_{calibration_method}", "y_test"
        title_suffix = f" (calibrated: {calibration_method})"

        if folds and score_key not in folds[0]:
            raise KeyError(
                f"Key '{score_key}' not found for model '{model_name}'. "
                f"Did you run calibrate_nested_cv_results with calibration_methods including '{calibration_method}'?"
            )

    metric_vals = []
    grid = np.linspace(0, 1, n_grid)
    interp_curves = []
    prevalences = []

    plt.figure(figsize=figsize)

    # overlay fold curves
    for r in folds:
        if y_key not in r or score_key not in r:
            continue

        y_true = r[y_key]
        y_score = r[score_key]

        if curve == "roc":
            x, y, _ = roc_curve(y_true, y_score)  # fpr, tpr
            metric = roc_auc_score(y_true, y_score)

            plt.plot(x, y, alpha=alpha_folds, linewidth=lw_folds, color=fold_color)

            y_i = np.interp(grid, x, y)
            y_i[0] = 0.0
            interp_curves.append(y_i)

        else:
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            metric = average_precision_score(y_true, y_score)

            plt.plot(recall, precision, alpha=alpha_folds, linewidth=lw_folds, color=fold_color)

            sort_idx = np.argsort(recall)
            recall_s = recall[sort_idx]
            precision_s = precision[sort_idx]
            y_i = np.interp(grid, recall_s, precision_s)
            interp_curves.append(y_i)

            prevalences.append(float(np.mean(y_true)))

        metric_vals.append(metric)

    metric_vals = np.asarray(metric_vals)
    if len(interp_curves) == 0:
        raise ValueError("No fold curves were computed (missing keys or empty folds).")

    interp_curves = np.vstack(interp_curves)
    mean_curve = interp_curves.mean(axis=0)
    std_curve = interp_curves.std(axis=0)

    # mean curve + std band + reference line
    if curve == "roc":
        mean_curve[-1] = 1.0

        plt.plot(
            grid, mean_curve,
            linewidth=lw_mean,
            color=mean_color,
            label=f"Mean AUROC = {metric_vals.mean():.3f}" if show_legend else None
        )

        if show_std_band:
            plt.fill_between(
                grid,
                np.clip(mean_curve - std_curve, 0, 1),
                np.clip(mean_curve + std_curve, 0, 1),
                alpha=std_band_alpha,
                color=band_color,
                label="±1 SD" if show_legend else None,
            )

        plt.plot(
            [0, 1], [0, 1],
            linestyle="--",
            linewidth=1,
            color=refline_color,
            label="Baseline (random chance)" if show_legend else None,
        )

        plt.xlabel("False Positive Rate", fontsize=font_size, fontweight="bold")
        plt.ylabel("True Positive Rate", fontsize=font_size, fontweight="bold")
        title_metric = "AUROC"

    else:
        plt.plot(
            grid, mean_curve,
            linewidth=lw_mean,
            color=mean_color,
            label=f"Mean AUPRC = {metric_vals.mean():.3f}" if show_legend else None
        )

        if show_std_band:
            plt.fill_between(
                grid,
                np.clip(mean_curve - std_curve, 0, 1),
                np.clip(mean_curve + std_curve, 0, 1),
                alpha=std_band_alpha,
                color=band_color,
                label="±1 SD" if show_legend else None,
            )

        baseline = float(np.mean(prevalences)) if prevalences else float(np.mean(folds[0][y_key]))
        plt.hlines(
            baseline, 0, 1,
            linestyles="--",
            linewidth=1,
            color=refline_color,
            label=f"Baseline (prevalence = {baseline:.3f})" if show_legend else None,
        )

        plt.xlabel("Recall", fontsize=font_size, fontweight="bold")
        plt.ylabel("Precision", fontsize=font_size, fontweight="bold")
        title_metric = "AUPRC"

    plt.title(
        f"{title_metric} curves — {model_name}{title_suffix} [{split}]\n"
        f"{title_metric} mean={metric_vals.mean():.3f} | min={metric_vals.min():.3f} | max={metric_vals.max():.3f}",
        fontsize=font_size + 2,
        fontweight="bold",
    )

    plt.xticks(fontsize=font_size, fontweight="bold")
    plt.yticks(fontsize=font_size, fontweight="bold")

    if show_legend:
        plt.legend(loc=legend_loc, frameon=True)

    plt.tight_layout()
    plt.show()




def plot_brier_logloss_all_methods(
    eval_results: Mapping[str, Sequence[Mapping[str, Any]]],
    model_names: str | Sequence[str] | None = None,
    calibration_methods: Sequence[str] | None = None,
    include_uncalibrated: bool = True,
    include_uncalib_oof: bool = False,
    sort_methods_by: str = "test",
    figsize: tuple[float, float] | None = None,
    font_size: float = 11.0,
    legend_loc: str = "best",
    # ---- colors for Train/Test bars ----
    split_palette: dict[str, str] | None = None,
    # alias map for method display names
    method_alias: Mapping[str, str] | None = None,
    # ---- prevalence baseline (optional) ----
    show_prevalence_baseline: bool = True,
    prevalence: float | None = None,
    brier_baseline_color: str = "#D5F713",
    logloss_baseline_color: str = "#D5F713",
    baseline_lw: float = 1.5,
    baseline_ls: str = "--",
) -> None:
    """
    Plot calibration-sensitive metrics (Brier score loss and Log loss) as grouped bar charts
    across *calibration methods*, with Train vs Test bars showing mean ± SD over outer folds.

    For each selected model, two figures are produced:
      1) Brier score loss by method (lower is better)
      2) Log loss by method (lower is better)

    Parameters
    ----------
    eval_results:
        Output of `evaluate_nested_cv_results`, shaped like:
        {model_name: [fold_entry_dict, ...], ...}. Each fold entry must contain the metric keys
        used below (uncalibrated and/or calibrated).

    model_names:
        A single model name, a list of model names, or None to plot all models in eval_results.

    calibration_methods:
        List of calibrated method names to include (e.g., list(calibration_config.keys())).
        If None, the function attempts to infer methods from keys like
        "cv_calib_train_<method>_<metric>".

    include_uncalibrated:
        If True, adds an "uncalibrated" baseline method using:
          - outer_train_<metric>
          - outer_test_<metric>

    include_uncalib_oof:
        If True, adds an "uncalib_oof" baseline method using:
          - Train: cv_uncalib_train_<metric>
          - Test : outer_test_<metric>

    sort_methods_by:
        Controls the x-axis order of methods, based on mean score across folds:
          - "test": sort by Test mean (recommended)
          - "train": sort by Train mean
        (For Brier/LogLoss, lower values sort earlier.)

    figsize:
        Figure size (width, height). If None, width auto-scales with number of methods.

    font_size:
        Base font size used for labels/ticks/titles.

    legend_loc:
        Matplotlib legend location string (e.g., "best", "upper right").

    split_palette:
        Dict mapping split labels to colors for bars:
        {"Train": <color>, "Test": <color>}. If None, defaults to
        {"Train": "darkblue", "Test": "darkred"}.

    method_alias:
        Optional dict mapping internal method keys -> short display names for plotting only.
        Example: {"smartcal_plattbinner": "plattbinner"}. Aliases do NOT affect metric lookup.

    show_prevalence_baseline:
        If True, overlays a prevalence baseline line for each metric using prevalence p:
          - Brier baseline   = p(1 - p)
          - Logloss baseline = -[p log(p) + (1-p) log(1-p)]

    prevalence:
        Optional prevalence override (float in (0,1)). If None, prevalence is computed as the
        mean of fold-level `entry["prevalence"]` for the plotted model.

    brier_baseline_color / logloss_baseline_color:
        Line colors for the baseline references.

    baseline_lw / baseline_ls:
        Line width and linestyle for baseline references.

    Returns
    -------
    None
        Displays matplotlib figures.
    """

    # NEW default palette
    if split_palette is None:
        split_palette = {"Train": "darkblue", "Test": "darkred"}

    if method_alias is None:
        method_alias = {}

    # -------------------------
    # Choose models
    # -------------------------
    if model_names is None:
        model_names = list(eval_results.keys())
    elif isinstance(model_names, str):
        model_names = [model_names]
    else:
        model_names = list(model_names)

    missing = [m for m in model_names if m not in eval_results]
    if missing:
        raise KeyError(f"Model(s) not found in eval_results: {missing}. Available: {list(eval_results.keys())}")

    # -------------------------
    # Infer calibration methods if not provided
    # -------------------------
    if calibration_methods is None:
        found = set()
        for m in model_names:
            for entry in eval_results[m][:3]:
                for k in entry.keys():
                    if k.startswith("cv_calib_train_"):
                        for metric in ("average_precision", "roc_auc", "brier_score_loss", "log_loss"):
                            suffix = f"_{metric}"
                            if k.endswith(suffix):
                                method = k[len("cv_calib_train_") : -len(suffix)]
                                found.add(method)
        calibration_methods = sorted(found)

    calibration_methods = list(calibration_methods)

    if sort_methods_by not in {"train", "test"}:
        raise ValueError("sort_methods_by must be 'train' or 'test'")

    # -------------------------
    # Prevalence baseline (optional)
    # -------------------------
    def _get_prevalence_for_model(model: str) -> float | None:
        if not show_prevalence_baseline:
            return None
        if prevalence is not None:
            p = float(prevalence)
        else:
            prev_vals = [float(e["prevalence"]) for e in eval_results[model] if "prevalence" in e]
            if len(prev_vals) == 0:
                raise KeyError(
                    f"No 'prevalence' values found for model '{model}'. "
                    "Either store entry['prevalence'] or pass prevalence=..."
                )
            p = float(np.mean(prev_vals))

        if not (0.0 < p < 1.0):
            raise ValueError(f"prevalence must be in (0, 1); got {p}")
        return p

    # -------------------------
    # Helper: collect tidy DF for a given model + metric
    # -------------------------
    def _collect_model_metric(model: str, metric: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for f in eval_results[model]:
            if include_uncalibrated:
                k_tr = f"outer_train_{metric}"
                k_te = f"outer_test_{metric}"
                if k_tr in f:
                    rows.append({"method": "uncalibrated", "split": "Train", "score": float(f[k_tr])})
                if k_te in f:
                    rows.append({"method": "uncalibrated", "split": "Test", "score": float(f[k_te])})

            if include_uncalib_oof:
                k_oof = f"cv_uncalib_train_{metric}"
                k_te = f"outer_test_{metric}"
                if k_oof in f:
                    rows.append({"method": "uncalib_oof", "split": "Train", "score": float(f[k_oof])})
                if k_te in f:
                    rows.append({"method": "uncalib_oof", "split": "Test", "score": float(f[k_te])})

            for method in calibration_methods:
                k_trm = f"cv_calib_train_{method}_{metric}"
                k_tem = f"calib_test_{method}_{metric}"

                if k_trm in f:
                    rows.append({"method": method, "split": "Train", "score": float(f[k_trm])})
                if k_tem in f:
                    rows.append({"method": method, "split": "Test", "score": float(f[k_tem])})

        df = pd.DataFrame(rows)
        df["model"] = model
        df["metric"] = metric

        # ---- NEW: add display label ----
        df["method_label"] = df["method"].map(lambda m: method_alias.get(m, m))
        return df

    # -------------------------
    # Helper: compute ordering by REAL method key, then map to labels
    # -------------------------
    def _method_order_real(df: pd.DataFrame) -> list[str]:
        df_s = df[df["split"] == ("Test" if sort_methods_by == "test" else "Train")].copy()
        if df_s.empty:
            df_s = df.copy()
        # Brier/LogLoss: lower is better => ascending=True
        means = df_s.groupby("method")["score"].mean().sort_values(ascending=True)
        return means.index.tolist()

    # -------------------------
    # Plot helper
    # -------------------------
    def _plot_metric(
        df: pd.DataFrame,
        title: str,
        y_label: str,
        baseline_value: float | None,
        baseline_color: str,
        baseline_label: str | None,
    ) -> None:
        if df.empty:
            print(f"[WARN] No data to plot for: {title}")
            return

        order_real = _method_order_real(df)
        order_labels = [method_alias.get(m, m) for m in order_real]

        # Guard: aliases must be unique or seaborn x categories collide
        if len(set(order_labels)) != len(order_labels):
            dupes = pd.Series(order_labels)[pd.Series(order_labels).duplicated(keep=False)].unique().tolist()
            raise ValueError(
                f"method_alias causes duplicate labels {dupes}. "
                f"Make aliases unique (or omit aliasing for colliding methods)."
            )

        n_methods = max(1, len(order_labels))
        if figsize is None:
            w = max(10.0, min(26.0, 0.65 * n_methods + 6.0))
            h = 4.8
            _figsize = (w, h)
        else:
            _figsize = figsize

        plt.figure(figsize=_figsize)
        ax = sns.barplot(
            data=df,
            x="method_label",  # <-- plot labels, not raw method keys
            y="score",
            hue="split",
            estimator=np.mean,
            errorbar=("sd"),
            order=order_labels,
            palette=split_palette,
            saturation=1,
        )

        if baseline_value is not None and baseline_label is not None:
            ax.axhline(
                float(baseline_value),
                ls=baseline_ls,
                lw=baseline_lw,
                color=baseline_color,
                label=baseline_label,
            )

        ax.set_xlabel("Calibration method", fontsize=font_size, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=font_size, fontweight="bold")
        ax.set_title(title, fontsize=font_size + 2, fontweight="bold")

        ax.tick_params(axis="both", labelsize=font_size)
        ax.tick_params(axis="x", rotation=25)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        ax.legend(title="", loc=legend_loc, prop={"size": font_size, "weight": "bold"})
        plt.tight_layout()
        plt.show()

    sns.set(style="whitegrid")

    for model in model_names:
        p = _get_prevalence_for_model(model)

        brier_baseline = (p * (1.0 - p)) if (p is not None) else None
        logloss_baseline = (
            float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))) if (p is not None) else None
        )

        df_brier = _collect_model_metric(model, "brier_score_loss")
        df_ll    = _collect_model_metric(model, "log_loss")

        _plot_metric(
            df_brier,
            title=f"{model} — Brier score loss by calibration method (mean ± SD across folds)",
            y_label="Brier score loss (lower is better)",
            baseline_value=brier_baseline,
            baseline_color=brier_baseline_color,
            baseline_label=None if brier_baseline is None else f"Baseline = {brier_baseline:.3f}",
        )

        _plot_metric(
            df_ll,
            title=f"{model} — Log loss by calibration method (mean ± SD across folds)",
            y_label="Log loss (lower is better)",
            baseline_value=logloss_baseline,
            baseline_color=logloss_baseline_color,
            baseline_label=None if logloss_baseline is None else f"Baseline = {logloss_baseline:.3f}",
        )



def plot_brier_logloss(
    eval_results: Mapping[str, Sequence[Mapping[str, Any]]],
    model_names: str | Sequence[str] | None = None,
    use_calibrated: bool = False,
    calibration_method: str | None = None,
    figsize: tuple[float, float] = (9.0, 4.0),
    font_size: float = 12.0,
    legend_loc: str = "best",
    x_tick_rotation: int = 0,  # NEW
    include_uncalib_oof: bool = False,
    method_alias: Mapping[str, str] | None = None,
    split_palette: dict[str, str] | None = None,
    show_prevalence_baseline: bool = True,
    prevalence: float | None = None,
    brier_baseline_color: str = "#D5F713",
    logloss_baseline_color: str = "#D5F713",
    baseline_lw: float = 1.5,
    baseline_ls: str = "--",
    annotate_mean_sd: bool = True,
    annotate_decimals: int = 3,
    annotate_font_size: float | None = None,
    annotate_offset: float = 0.015,
    brier_ylim: tuple[float, float] | None = None,
    logloss_ylim: tuple[float, float] | None = None,
) -> None:
    """
    Plot Brier score and Log loss as two separate grouped bar charts (Train vs Test) across models,
    using mean ± SD across outer folds (optionally calibrated or OOF-train), with optional baselines
    and per-bar "mean ± sd" annotations.

    Parameters:
    --------------
      eval_results: dict(model_name -> list of fold dicts containing metric keys).
      model_names: model(s) to plot; None = all models in eval_results.
      use_calibrated: if True, use calibrated metric keys for `calibration_method`.
      calibration_method: calibration suffix used to form calibrated keys (e.g. "smartcal_beta").
      figsize: matplotlib figure size for each plot.
      font_size: base font size for labels/ticks/title.
      legend_loc: legend location string.
      x_tick_rotation: rotation (deg) for x-axis tick labels.
      include_uncalib_oof: if True (and not calibrated), Train uses OOF metrics; Test uses outer test.
      method_alias: optional mapping model_key -> display name on x-axis.
      split_palette: colors for {"Train": ..., "Test": ...}.
      show_prevalence_baseline: if True, draw Brier/LogLoss baselines derived from prevalence.
      prevalence: optional override for prevalence; else mean of entry["prevalence"] across folds.
      brier_baseline_color/logloss_baseline_color: baseline line colors.
      baseline_lw/baseline_ls: baseline line width / linestyle.
      annotate_mean_sd: if True, add "mean ± sd" text above each bar.
      annotate_decimals: decimals for annotation text.
      annotate_font_size: annotation font size; None = font_size - 3 (min 8).
      annotate_offset: vertical padding above (mean+sd) for annotation (y-units).
      brier_ylim: optional y-axis limits (ymin, ymax) for the Brier-score plot. If None, y-limits are chosen automatically
      logloss_ylim: optional y-axis limits (ymin, ymax) for the Log-loss plot. If None, y-limits are chosen automatically 
    """

    # -------------------------
    # Defaults
    # -------------------------
    if method_alias is None:
        method_alias = {}
    if split_palette is None:
        split_palette = {"Train": "darkblue", "Test": "darkred"}
    for k in ("Train", "Test"):
        if k not in split_palette:
            raise ValueError(f"split_palette must contain '{k}'. Got keys: {list(split_palette.keys())}")

    # -------------------------
    # Choose models
    # -------------------------
    if model_names is None:
        model_names = list(eval_results.keys())
    elif isinstance(model_names, str):
        model_names = [model_names]
    else:
        model_names = list(model_names)

    missing = [m for m in model_names if m not in eval_results]
    if missing:
        raise KeyError(
            f"Model(s) not found in eval_results: {missing}. "
            f"Available: {list(eval_results.keys())}"
        )

    # Create label mapping for x-axis display
    model_labels = [method_alias.get(m, m) for m in model_names]
    if len(set(model_labels)) != len(model_labels):
        dupes = pd.Series(model_labels)[pd.Series(model_labels).duplicated(keep=False)].unique().tolist()
        raise ValueError(
            f"method_alias causes duplicate model labels {dupes}. "
            f"Make aliases unique (or omit aliasing for colliding model names)."
        )

    # -------------------------
    # Prevalence baseline (optional)
    # -------------------------
    p_mean: float | None = None
    if show_prevalence_baseline:
        if prevalence is not None:
            p_mean = float(prevalence)
        else:
            prev_vals = [
                float(entry["prevalence"])
                for m in model_names
                for entry in eval_results[m]
                if "prevalence" in entry
            ]
            if len(prev_vals) == 0:
                raise KeyError(
                    "No 'prevalence' values found in eval_results entries. "
                    "Update evaluate_nested_cv_results to store entry['prevalence'] "
                    "or pass prevalence=... explicitly."
                )
            p_mean = float(np.mean(prev_vals))

        if not (0.0 < p_mean < 1.0):
            raise ValueError(f"prevalence must be in (0, 1); got {p_mean}")

        brier_baseline = float(p_mean * (1.0 - p_mean))
        logloss_baseline = float(-(p_mean * np.log(p_mean) + (1.0 - p_mean) * np.log(1.0 - p_mean)))
    else:
        brier_baseline = None
        logloss_baseline = None

    # -------------------------
    # Helper: build tidy DF
    # -------------------------
    def _collect(metric_label: str, train_key: str, test_key: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for m in model_names:
            display = method_alias.get(m, m)
            for f in eval_results[m]:
                rows.append({"model": display, "split": "Train", "score": f[train_key]})
                rows.append({"model": display, "split": "Test",  "score": f[test_key]})
        df = pd.DataFrame(rows)
        df["metric"] = metric_label
        return df

    # -------------------------
    # Pick metric keys
    # -------------------------
    if include_uncalib_oof and use_calibrated:
        raise ValueError("include_uncalib_oof=True is only supported when use_calibrated=False.")

    if not use_calibrated:
        if include_uncalib_oof:
            df_brier = _collect("Brier", "cv_uncalib_train_brier_score_loss", "outer_test_brier_score_loss")
            df_ll    = _collect("LogLoss", "cv_uncalib_train_log_loss", "outer_test_log_loss")
            title_suffix = " (uncalibrated OOF-train)"
        else:
            df_brier = _collect("Brier", "outer_train_brier_score_loss", "outer_test_brier_score_loss")
            df_ll    = _collect("LogLoss", "outer_train_log_loss", "outer_test_log_loss")
            title_suffix = " (uncalibrated)"
    else:
        if calibration_method is None:
            raise ValueError("calibration_method must be provided when use_calibrated=True.")

        brier_train_key = f"cv_calib_train_{calibration_method}_brier_score_loss"
        brier_test_key  = f"calib_test_{calibration_method}_brier_score_loss"
        ll_train_key    = f"cv_calib_train_{calibration_method}_log_loss"
        ll_test_key     = f"calib_test_{calibration_method}_log_loss"

        for m in model_names:
            if len(eval_results[m]) == 0:
                raise ValueError(f"eval_results['{m}'] is empty.")
            first = eval_results[m][0]
            for k in [brier_train_key, brier_test_key, ll_train_key, ll_test_key]:
                if k not in first:
                    raise KeyError(
                        f"Key '{k}' not found in eval_results for model '{m}'. "
                        f"Did you compute calibrated metrics for '{calibration_method}'?"
                    )

        df_brier = _collect("Brier", brier_train_key, brier_test_key)
        df_ll    = _collect("LogLoss", ll_train_key, ll_test_key)
        title_suffix = f" (calibrated: {calibration_method})"

    sns.set(style="whitegrid")

    # -------------------------
    # Plot helper
    # -------------------------
    def _plot_metric(
        df: pd.DataFrame,
        y_label: str,
        baseline_value: float | None,
        baseline_color: str,
        baseline_label: str | None,
        ylim: tuple[float, float] | None = None, 
    ) -> None:
        plt.figure(figsize=figsize)
        ax = sns.barplot(
            data=df,
            x="model",
            y="score",
            hue="split",
            hue_order=["Train", "Test"],      # NEW: lock ordering for annotation
            estimator=np.mean,
            errorbar=("sd"),
            palette=split_palette,
            order=model_labels,
            saturation=1,
        )

        if baseline_value is not None and baseline_label is not None:
            ax.axhline(
                float(baseline_value),
                ls=baseline_ls,
                lw=baseline_lw,
                color=baseline_color,
                label=baseline_label,
            )

        ax.set_xlabel("Model", fontsize=font_size, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=font_size, fontweight="bold")
        ax.set_title(
            f"{y_label} across models{title_suffix}",
            fontsize=font_size + 2,
            fontweight="bold",
        )

        ax.tick_params(axis="both", labelsize=font_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        # User-controlled x tick rotation (overrides the "len(model_labels) > 3" behavior)
        ax.tick_params(axis="x", rotation=x_tick_rotation)

        if ylim is not None:
            ax.set_ylim(*ylim)
            
        # -------------------------
        # Annotate mean ± SD above each bar
        # -------------------------
        if annotate_mean_sd:
            summary = (
                df.groupby(["model", "split"])["score"]
                  .agg(mean="mean", sd="std")
                  .reset_index()
            )
            summary["sd"] = summary["sd"].fillna(0.0)

            summary["model"] = pd.Categorical(summary["model"], categories=model_labels, ordered=True)
            summary["split"] = pd.Categorical(summary["split"], categories=["Train", "Test"], ordered=True)
            summary = summary.sort_values(["model", "split"]).reset_index(drop=True)

            bars = list(ax.patches)
            if len(bars) != len(summary):
                n = min(len(bars), len(summary))
                bars = bars[:n]
                summary = summary.iloc[:n].reset_index(drop=True)

            ann_fs = annotate_font_size if annotate_font_size is not None else max(8, float(font_size) - 3)
            offset = float(annotate_offset)

            for bar, (_, r) in zip(bars, summary.iterrows()):
                mean = float(r["mean"])
                sd = float(r["sd"])
                x = bar.get_x() + bar.get_width() / 2.0
                y = mean + sd + offset
                ax.text(
                    x,
                    y,
                    f"{mean:.{annotate_decimals}f} ± {sd:.{annotate_decimals}f}",
                    ha="center",
                    va="bottom",
                    fontsize=ann_fs,
                    fontweight="bold",
                )

            # keep labels from clipping
            top = float((summary["mean"] + summary["sd"]).max() + offset + 0.05)
            #ax.set_ylim(0.0, max(ax.get_ylim()[1], top))

            if ylim is None:
                y0, y1 = ax.get_ylim()
                ax.set_ylim(y0, max(y1, top))
                
        ax.legend(title="", loc=legend_loc, prop={"size": font_size, "weight": "bold"})
        plt.tight_layout()
        plt.show()

    _plot_metric(
        df_brier,
        "Brier score",
        brier_baseline,
        brier_baseline_color,
        None if brier_baseline is None else f"Baseline = {brier_baseline:.2f}",
        ylim=brier_ylim,
    )

    _plot_metric(
        df_ll,
        "Log loss",
        logloss_baseline,
        logloss_baseline_color,
        None if logloss_baseline is None else f"Baseline = {logloss_baseline:.2f}",
        ylim=logloss_ylim, 
    )



def plot_auprc_auroc(
    eval_results: Mapping[str, Sequence[Mapping[str, Any]]],
    model_names: str | Sequence[str] | None = None,
    use_calibrated: bool = False,
    calibration_method: str | None = None,
    figsize: tuple[float, float] = (9.0, 4.0),
    font_size: float = 12.0,
    legend_loc: str = "best",
    x_tick_rotation: int = 0,
    # ---- optional additions ----
    include_uncalib_oof: bool = False,
    method_alias: Mapping[str, str] | None = None,          # alias map for *model names*
    split_palette: dict[str, str] | None = None,            # colors for Train/Test bars
    # ---- baseline / prevalence handling ----
    show_prevalence_baseline: bool = True,
    prevalence: float | None = None,  # optional override; if None, compute from eval_results[*]["prevalence"]
    auroc_baseline_color: str = "#D5F713",
    auprc_baseline_color: str = "#D5F713",
    baseline_lw: float = 1.5,
    baseline_ls: str = "--",
    # ---- annotation ----
    annotate_mean_sd: bool = True,
    annotate_decimals: int = 3,
    annotate_font_size: float | None = None,
    annotate_offset: float = 0.015,
    # ---- NEW: per-metric y-limits ----
    auprc_ylim: tuple[float, float] | None = None,
    auroc_ylim: tuple[float, float] | None = None,
) -> None:
    """
    Plot AUPRC and AUROC as two separate grouped bar charts (Train vs Test) across models,
    using mean ± SD across outer folds (optionally calibrated or OOF-train).

    Parameters (brief):
      eval_results: dict(model_name -> list of fold dicts with metric keys).
      model_names: model(s) to plot; None = all keys in eval_results.
      use_calibrated: if True, use calibrated keys for `calibration_method`.
      calibration_method: suffix used to form calibrated metric keys (e.g. "smartcal_beta").
      figsize: matplotlib figure size for each plot.
      font_size: base font size for labels/ticks/title.
      legend_loc: legend location string passed to matplotlib.
      x_tick_rotation: rotation (deg) for x-axis tick labels.
      include_uncalib_oof: if True (and not calibrated), Train uses OOF metrics, Test uses outer test.
      method_alias: optional mapping model_key -> display label on x-axis.
      split_palette: colors for {"Train": ..., "Test": ...}.
      show_prevalence_baseline: if True, draw AUPRC baseline at prevalence (and AUROC at 0.5).
      prevalence: optional override for prevalence baseline; else mean of entry["prevalence"].
      auroc_baseline_color / auprc_baseline_color: baseline line colors.
      baseline_lw / baseline_ls: baseline line width / linestyle.
      annotate_mean_sd: if True, add "mean ± sd" text above each bar.
      annotate_decimals: decimals for annotation text.
      annotate_font_size: annotation font size; None = font_size - 3 (min 8).
      annotate_offset: vertical padding above (mean+sd) for annotation (y-units, ~0–1 scale).
      auprc_ylim: optional y-axis limits (ymin, ymax) for the AUPRC plot.
        If None, y-limits are chosen automatically (and may be expanded to fit annotations).
      auroc_ylim: optional y-axis limits (ymin, ymax) for the AUROC plot.
        If None, y-limits are chosen automatically (and may be expanded to fit annotations).
    """

    # -------------------------
    # Defaults
    # -------------------------
    if method_alias is None:
        method_alias = {}
    if split_palette is None:
        split_palette = {"Train": "darkblue", "Test": "darkred"}
    for k in ("Train", "Test"):
        if k not in split_palette:
            raise ValueError(f"split_palette must contain '{k}'. Got keys: {list(split_palette.keys())}")

    if include_uncalib_oof and use_calibrated:
        raise ValueError("include_uncalib_oof=True is only supported when use_calibrated=False.")

    # -------------------------
    # Choose models
    # -------------------------
    if model_names is None:
        model_names = list(eval_results.keys())
    elif isinstance(model_names, str):
        model_names = [model_names]
    else:
        model_names = list(model_names)

    missing = [m for m in model_names if m not in eval_results]
    if missing:
        raise KeyError(
            f"Model(s) not found in eval_results: {missing}. "
            f"Available: {list(eval_results.keys())}"
        )

    # Display labels for x-axis (aliasing only affects labels)
    model_labels = [method_alias.get(m, m) for m in model_names]
    if len(set(model_labels)) != len(model_labels):
        dupes = pd.Series(model_labels)[pd.Series(model_labels).duplicated(keep=False)].unique().tolist()
        raise ValueError(
            f"method_alias causes duplicate model labels {dupes}. "
            f"Make aliases unique (or omit aliasing for colliding model names)."
        )

    # -------------------------
    # Prevalence baseline (AUPRC only)
    # -------------------------
    p_mean: float | None = None
    if show_prevalence_baseline:
        if prevalence is not None:
            p_mean = float(prevalence)
        else:
            prev_vals = [
                float(entry["prevalence"])
                for m in model_names
                for entry in eval_results[m]
                if "prevalence" in entry
            ]
            if len(prev_vals) == 0:
                raise KeyError(
                    "No 'prevalence' values found in eval_results entries. "
                    "Update evaluate_nested_cv_results to store entry['prevalence'] "
                    "or pass prevalence=... explicitly."
                )
            p_mean = float(np.mean(prev_vals))

    # -------------------------
    # Build one tidy DF per metric
    # -------------------------
    def _collect(metric: str, train_key: str, test_key: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for m in model_names:
            display = method_alias.get(m, m)
            for f in eval_results[m]:
                rows.append({"model": display, "split": "Train", "score": f[train_key]})
                rows.append({"model": display, "split": "Test",  "score": f[test_key]})
        df = pd.DataFrame(rows)
        df["metric"] = metric
        return df

    # -------------------------
    # Pick metric keys
    # -------------------------
    if not use_calibrated:
        if include_uncalib_oof:
            # Train: uncalibrated OOF; Test: outer test
            df_ap  = _collect("AUPRC", "cv_uncalib_train_average_precision", "outer_test_average_precision")
            df_roc = _collect("AUROC", "cv_uncalib_train_roc_auc",          "outer_test_roc_auc")
            title_suffix = " (uncalibrated OOF-train)"
        else:
            # Train/Test: standard outer train/test
            df_ap  = _collect("AUPRC", "outer_train_average_precision", "outer_test_average_precision")
            df_roc = _collect("AUROC", "outer_train_roc_auc",          "outer_test_roc_auc")
            title_suffix = " (uncalibrated)"
    else:
        if calibration_method is None:
            raise ValueError("calibration_method must be provided when use_calibrated=True.")

        ap_train_key  = f"cv_calib_train_{calibration_method}_average_precision"
        ap_test_key   = f"calib_test_{calibration_method}_average_precision"
        roc_train_key = f"cv_calib_train_{calibration_method}_roc_auc"
        roc_test_key  = f"calib_test_{calibration_method}_roc_auc"

        # sanity check on first fold of each model
        for m in model_names:
            if len(eval_results[m]) == 0:
                raise ValueError(f"eval_results['{m}'] is empty.")
            first = eval_results[m][0]
            for k in [ap_train_key, ap_test_key, roc_train_key, roc_test_key]:
                if k not in first:
                    raise KeyError(
                        f"Key '{k}' not found for model '{m}' (calib method '{calibration_method}')."
                    )

        df_ap  = _collect("AUPRC", ap_train_key,  ap_test_key)
        df_roc = _collect("AUROC", roc_train_key, roc_test_key)
        title_suffix = f" (calibrated: {calibration_method})"

    sns.set(style="whitegrid")

    # -------------------------
    # Plot helper
    # -------------------------
    def _plot_df(
        df: pd.DataFrame,
        metric_name: Literal["AUPRC", "AUROC"],
        ylim: tuple[float, float] | None = None,
    ) -> None:
        plt.figure(figsize=figsize)
        ax = sns.barplot(
            data=df,
            x="model",
            y="score",
            hue="split",
            hue_order=["Train", "Test"],      # IMPORTANT: lock ordering for annotation
            estimator=np.mean,
            errorbar=("sd"),                  # mean ± SD
            palette=split_palette,
            order=model_labels,
            saturation=1,
        )

        # Baselines
        if metric_name == "AUPRC" and show_prevalence_baseline and p_mean is not None:
            ax.axhline(
                float(p_mean),
                ls=baseline_ls,
                lw=baseline_lw,
                color=auprc_baseline_color,
                label=f"Baseline = {float(p_mean):.2f}",
            )

        if metric_name == "AUROC":
            ax.axhline(
                0.5,
                ls=baseline_ls,
                lw=baseline_lw,
                color=auroc_baseline_color,
                label="Baseline = 0.50",
            )

        ax.set_xlabel("Model", fontsize=font_size, fontweight="bold")
        ax.set_ylabel(metric_name, fontsize=font_size, fontweight="bold")
        ax.set_title(
            f"{metric_name} across models{title_suffix}",
            fontsize=font_size + 2,
            fontweight="bold",
        )

        ax.tick_params(axis="both", labelsize=font_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        # rotate x tick
        ax.tick_params(axis="x", rotation=x_tick_rotation)

        # Apply user-provided y-limits (if any)
        if ylim is not None:
            ax.set_ylim(*ylim)


        # -------------------------
        # Annotate mean ± SD above each bar 
        # -------------------------
        if annotate_mean_sd:
            # compute mean/sd
            summary = (
                df.groupby(["model", "split"])["score"]
                .agg(mean="mean", sd=lambda x: np.std(x, ddof=1))
                .reset_index()
            )
            summary["sd"] = summary["sd"].fillna(0.0)

            # map (model, split) -> (mean, sd)
            stats = {
                (r["model"], r["split"]): (float(r["mean"]), float(r["sd"]))
                for _, r in summary.iterrows()
            }

            ann_fs = annotate_font_size if annotate_font_size is not None else max(8, float(font_size) - 3)
            offset = float(annotate_offset)

            # containers[0] = Train bars, containers[1] = Test bars (because hue_order is locked)
            for split, container in zip(["Train", "Test"], ax.containers[:2]):
                for model_label, bar in zip(model_labels, container):
                    mean, sd = stats[(model_label, split)]
                    x = bar.get_x() + bar.get_width() / 2.0
                    y = mean + sd + offset
                    ax.text(
                        x, y,
                        f"{mean:.{annotate_decimals}f} ± {sd:.{annotate_decimals}f}",
                        ha="center", va="bottom",
                        fontsize=ann_fs, fontweight="bold",
                    )

            # avoid clipping the text (only if user didn't force ylim)
            top = max(m + s for (m, s) in stats.values()) + offset + 0.05
            if ylim is None:
                y0, y1 = ax.get_ylim()
                ax.set_ylim(y0, max(y1, max(1.05, top)))

        else:
            if ylim is None:
                ax.set_ylim(0.0, 1.05)

        ax.legend(title="", loc=legend_loc, prop={"size": font_size, "weight": "bold"})
        plt.tight_layout()
        plt.show()

    # Two separate plots (with optional separate y-lims)
    _plot_df(df_ap, "AUPRC", ylim=auprc_ylim)
    _plot_df(df_roc, "AUROC", ylim=auroc_ylim)


def plot_patient_auprc_auroc(
    df_pat: pd.DataFrame,
    *,
    model_names: str | Sequence[str] | None = None,
    variants: str | Sequence[str] | None = None,  # e.g. "beta" or ["uncalib","beta"]
    # score column from pooled_patient_risk_summary (e.g. p_median / p_mean / p_softmax / p_q75)
    score_col: str = "p_median",

    figsize: tuple[float, float] = (9.0, 4.0),
    font_size: float = 12.0,
    legend_loc: str = "best",
    x_tick_rotation: int = 0,

    # ---- stylistic parity with plot_auprc_auroc ----
    method_alias: Mapping[str, str] | None = None,          # alias map for *model names*
    split_palette: dict[str, str] | None = None,            # colors for Train/Test bars

    # ---- baseline / prevalence handling ----
    show_prevalence_baseline: bool = True,
    prevalence: float | None = None,  # optional override; if None, compute from df_pat y prevalence
    auroc_baseline_color: str = "#D5F713",
    auprc_baseline_color: str = "#D5F713",
    baseline_lw: float = 1.5,
    baseline_ls: str = "--",

    # ---- annotation ----
    annotate_mean_sd: bool = True,
    annotate_decimals: int = 3,
    annotate_font_size: float | None = None,
    annotate_offset: float = 0.015,

    # ---- per-metric y-limits ----
    auprc_ylim: tuple[float, float] | None = None,
    auroc_ylim: tuple[float, float] | None = None,

    # ---- split selection (mirrors your split semantics) ----
    include_test: bool = True,
    include_train_oof: bool = True,
    split_label_map: Mapping[str, str] | None = None,  # e.g. {"train_oof": "Train", "test": "Test"}
) -> None:
    """
    Patient-level analog of `plot_auprc_auroc`.

    Input `df_pat` is expected to come from pooled_patient_risk_summary and contain patient-level scores.
    For mean ± SD error bars to reflect CV variability, `df_pat` should include per-run identifiers
    (typically: 'trial' and 'outer_fold'), i.e. built with grouping="per_trial_fold".

    If 'trial'/'outer_fold' are absent, the function still works, but SD will be 0 (single score per bar).

    Required columns in df_pat:
      - model, variant, split, y, and `score_col`
      - for run-level SD: trial and outer_fold
    """
    # -------------------------
    # Defaults (match plot_auprc_auroc)
    # -------------------------
    if method_alias is None:
        method_alias = {}
    if split_palette is None:
        split_palette = {"Train": "darkblue", "Test": "darkred"}
    for k in ("Train", "Test"):
        if k not in split_palette:
            raise ValueError(f"split_palette must contain '{k}'. Got keys: {list(split_palette.keys())}")

    if split_label_map is None:
        split_label_map = {"train_oof": "Train", "test": "Test"}

    # -------------------------
    # Basic validation
    # -------------------------
    required = {"model", "variant", "split", "y", score_col}
    missing = required - set(df_pat.columns)
    if missing:
        raise KeyError(f"df_pat missing required columns: {sorted(missing)}")

    # Split selection
    splits: list[str] = []
    if include_train_oof:
        splits.append("train_oof")
    if include_test:
        splits.append("test")
    if len(splits) == 0:
        raise ValueError("No splits selected. Set include_test/include_train_oof to True.")

    d = df_pat[df_pat["split"].isin(splits)].copy()
    if d.empty:
        present = sorted(df_pat["split"].dropna().unique().tolist())
        raise ValueError(f"No rows left after filtering to splits={splits}. Present splits: {present}")

    # Ensure numeric
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
    d["y"] = pd.to_numeric(d["y"], errors="coerce")
    d = d.dropna(subset=[score_col, "y"])

    # Map splits to Train/Test display labels (so plotting matches your original)
    d["split_disp"] = d["split"].map(split_label_map).fillna(d["split"].astype(str))

    # -------------------------
    # Choose models (match plot_auprc_auroc API)
    # -------------------------
    available_models = sorted(d["model"].astype(str).unique().tolist())

    if model_names is None:
        model_names_list = available_models
    elif isinstance(model_names, str):
        model_names_list = [model_names]
    else:
        model_names_list = list(model_names)

    missing_models = [m for m in model_names_list if m not in available_models]
    if missing_models:
        raise KeyError(
            f"Model(s) not found in df_pat: {missing_models}. "
            f"Available: {available_models}"
        )

    # Display labels for x-axis (aliasing only affects labels)
    model_labels = [method_alias.get(m, m) for m in model_names_list]
    if len(set(model_labels)) != len(model_labels):
        dupes = pd.Series(model_labels)[pd.Series(model_labels).duplicated(keep=False)].unique().tolist()
        raise ValueError(
            f"method_alias causes duplicate model labels {dupes}. "
            f"Make aliases unique (or omit aliasing for colliding model names)."
        )

    # Apply model filter + convert model to display label
    d = d[d["model"].isin(model_names_list)].copy()
    d["model_disp"] = d["model"].map(lambda m: method_alias.get(m, m))

    # -------------------------
    # Variant selection (optional)
    # -------------------------
    if variants is None:
        variants_list = sorted(d["variant"].astype(str).unique().tolist())
    elif isinstance(variants, str):
        variants_list = [variants]
    else:
        variants_list = list(variants)

    d = d[d["variant"].isin(variants_list)].copy()
    if d.empty:
        raise ValueError(f"No rows left after filtering to variants={variants_list}.")

    # -------------------------
    # Prevalence baseline (AUPRC only)
    # -------------------------
    p_mean: float | None = None
    if show_prevalence_baseline:
        if prevalence is not None:
            p_mean = float(prevalence)
        else:
            # Compute prevalence from the *filtered* data (consistent with what you're plotting)
            y_vals = d["y"].to_numpy(dtype=float)
            if y_vals.size == 0:
                raise ValueError("Cannot compute prevalence baseline: no y values.")
            p_mean = float(np.mean(y_vals))

    # -------------------------
    # Metric computation helper (patient-level, per run if available)
    # -------------------------
    have_runs = {"trial", "outer_fold"}.issubset(d.columns)

    def _metric_df(metric: Literal["AUPRC", "AUROC"]) -> pd.DataFrame:
        """
        Build a tidy df with rows:
          model_disp, split_disp, score
        where score is computed per-run if possible.
        """
        rows: list[dict[str, Any]] = []

        # group for metric computation:
        # - always separate by model/variant/split
        # - if run columns exist: compute one score per (trial, outer_fold)
        base_keys = ["model_disp", "variant", "split_disp"]
        run_keys = base_keys + (["trial", "outer_fold"] if have_runs else [])

        for keys, gdf in d.groupby(run_keys, sort=False):
            y_true = gdf["y"].to_numpy(dtype=int)
            y_score = gdf[score_col].to_numpy(dtype=float)

            # metric undefined if single class
            if np.unique(y_true).size < 2:
                score = np.nan
            else:
                if metric == "AUROC":
                    score = float(roc_auc_score(y_true, y_score))
                else:
                    score = float(average_precision_score(y_true, y_score))

            row = dict(zip(run_keys, keys if isinstance(keys, tuple) else (keys,)))
            row["score"] = score
            rows.append(row)

        df = pd.DataFrame(rows).dropna(subset=["score"])
        df["metric"] = metric
        return df

    # Compute tidy per-run metric tables
    df_ap = _metric_df("AUPRC")
    df_roc = _metric_df("AUROC")

    # Title suffix (match your style)
    # If you pass multiple variants, you're effectively plotting pooled metrics across those variants;
    # usually you'd plot one variant at a time. We'll include it in the title if single variant.
    # if len(variants_list) == 1:
    #     title_suffix = f" (patient-level; variant: {variants_list[0]}; score={score_col})"
    # else:
    #     title_suffix = f" (patient-level; variants: {', '.join(map(str, variants_list))}; score={score_col})"

    title_suffix = ''
    sns.set(style="whitegrid")

    # -------------------------
    # Plot helper (copied structure from plot_auprc_auroc)
    # -------------------------
    def _plot_df(
        df: pd.DataFrame,
        metric_name: Literal["AUPRC", "AUROC"],
        ylim: tuple[float, float] | None = None,
    ) -> None:
        # IMPORTANT: x-axis should be model only (like your original)
        # If you want separate bars per variant too, we can extend later.
        # For now, assume you're plotting one variant at a time.
        df_plot = df.copy()

        if df_plot["variant"].nunique() > 1:
            # Guardrail: your original plot assumes just model on x-axis.
            # You can still proceed, but you'll be mixing variants.
            # Better: call the function with variants="beta" etc.
            pass

        plt.figure(figsize=figsize)
        ax = sns.barplot(
            data=df_plot,
            x="model_disp",
            y="score",
            hue="split_disp",
            hue_order=["Train", "Test"],      # lock ordering for annotation
            estimator=np.mean,
            errorbar=("sd"),                  # mean ± SD (across runs)
            palette=split_palette,
            order=model_labels,
            saturation=1,
        )

        # Baselines
        if metric_name == "AUPRC" and show_prevalence_baseline and p_mean is not None:
            ax.axhline(
                float(p_mean),
                ls=baseline_ls,
                lw=baseline_lw,
                color=auprc_baseline_color,
                label=f"Baseline = {float(p_mean):.2f}",
            )

        if metric_name == "AUROC":
            ax.axhline(
                0.5,
                ls=baseline_ls,
                lw=baseline_lw,
                color=auroc_baseline_color,
                label="Baseline = 0.50",
            )

        ax.set_xlabel("Model", fontsize=font_size, fontweight="bold")
        ax.set_ylabel(metric_name, fontsize=font_size, fontweight="bold")
        ax.set_title(
            f"{metric_name} across models{title_suffix}",
            fontsize=font_size + 2,
            fontweight="bold",
        )

        ax.tick_params(axis="both", labelsize=font_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        ax.tick_params(axis="x", rotation=x_tick_rotation)

        if ylim is not None:
            ax.set_ylim(*ylim)

        # -------------------------
        # Annotate mean ± SD above each bar
        # -------------------------
        if annotate_mean_sd:
            summary = (
                df_plot.groupby(["model_disp", "split_disp"])["score"]
                .agg(mean="mean", sd=lambda x: np.std(x, ddof=1))
                .reset_index()
            )
            summary["sd"] = summary["sd"].fillna(0.0)

            stats = {
                (r["model_disp"], r["split_disp"]): (float(r["mean"]), float(r["sd"]))
                for _, r in summary.iterrows()
            }

            ann_fs = annotate_font_size if annotate_font_size is not None else max(8, float(font_size) - 3)
            offset = float(annotate_offset)

            # containers[0] = Train bars, containers[1] = Test bars (because hue_order is locked)
            for split, container in zip(["Train", "Test"], ax.containers[:2]):
                for model_label, bar in zip(model_labels, container):
                    # If a (model, split) combo isn't present (e.g. no train_oof), skip safely.
                    if (model_label, split) not in stats:
                        continue
                    mean, sd = stats[(model_label, split)]
                    x = bar.get_x() + bar.get_width() / 2.0
                    y = mean + sd + offset
                    ax.text(
                        x, y,
                        f"{mean:.{annotate_decimals}f} ± {sd:.{annotate_decimals}f}",
                        ha="center", va="bottom",
                        fontsize=ann_fs, fontweight="bold",
                    )

            top = max(m + s for (m, s) in stats.values()) + offset + 0.05 if stats else 1.0
            if ylim is None:
                y0, y1 = ax.get_ylim()
                ax.set_ylim(y0, max(y1, max(1.05, top)))
        else:
            if ylim is None:
                ax.set_ylim(0.0, 1.05)

        ax.legend(title="", loc=legend_loc, prop={"size": font_size, "weight": "bold"})
        plt.tight_layout()
        plt.show()

    _plot_df(df_ap, "AUPRC", ylim=auprc_ylim)
    _plot_df(df_roc, "AUROC", ylim=auroc_ylim)



def barplot_balanced_accuracy(
    all_results: Mapping[str, Sequence[Mapping[str, Any]]],
    model_names: str | Sequence[str] | None = None,
    use_calibrated: bool = False,
    calibration_method: str | None = None,
    n_grid: int = 101,
    mode: Literal["train_threshold", "test_threshold", "split_best"] = "train_threshold",
    # ---- labels / aliasing ----
    method_alias: Mapping[str, str] | None = None,
    # ---- styling ----
    figsize: tuple[float, float] = (9.0, 5.0),
    font_size: float = 12.0,
    legend_loc: str = "best",
    x_tick_rotation: int = 0,
    split_palette: Mapping[str, str] | None = None,  # {"Train": "...", "Test": "..."}
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
) -> None:
    """
    Plot balanced accuracy across outer folds as a grouped bar chart (Train vs Test) for one or
    more models, summarizing performance as mean ± SD across folds.

    This function is intended to work with your nested-CV `all_results` structure where each
    model maps to a list of per-(trial, outer_fold) dictionaries containing y labels and score
    vectors (and optionally calibrated scores).

    What is computed
    ----------------
    For each model and each fold dictionary `r`:
      1) Retrieve (y_train, score_train) and (y_test, score_test) from `r`.
      2) Convert scores to hard predictions using thresholds on a fixed grid in [0, 1].
      3) Compute balanced accuracy:
            BA = (TPR + TNR) / 2
         where TPR is sensitivity and TNR is specificity.
      4) Aggregate fold-level balanced accuracy values across folds and display:
            mean ± SD

    Threshold selection modes
    -------------------------
    mode="train_threshold":
        For each fold, choose a threshold t* that maximizes BA on the TRAIN split for that fold,
        then evaluate BA on both train and test using that same t*.
        This is the recommended “train-chosen threshold” summary.

    mode="test_threshold":
        For each fold, choose a threshold t* that maximizes BA on the TEST split for that fold,
        then evaluate BA on both train and test using that same t*.
        This is optimistic (uses test labels to pick thresholds) and is best used for diagnostic
        or “ceiling” comparisons.

    mode="split_best":
        For each fold, choose thresholds independently for train and test:
            Train bar uses max_t BA_train(t)
            Test  bar uses max_t BA_test(t)
        This is explicitly post hoc (a per-split upper bound), useful for visualization.

    Score sources
    -------------
    If use_calibrated=False:
        Train: r["y_train"], r["y_train_scores"]
        Test : r["y_test"],  r["y_test_scores"]

    If use_calibrated=True:
        `calibration_method` is required and scores are taken from:
        Train: r[f"cv_calib_train_predictions_{calibration_method}"], with y_train
        Test : r[f"calib_test_predictions_{calibration_method}"],     with y_test

    Plot contents
    -------------
    - Two bars per model: Train and Test.
    - Error bars show ±1 SD across folds.
    - Optional horizontal baseline line (default: 0.50) for reference.
    - Optional annotation above each bar showing "mean ± SD".

    Parameters
    ----------
    all_results:
        Nested-CV results keyed by model name; each value is a list of fold dicts.

    model_names:
        Model(s) to plot:
          - None: plot all models in `all_results`
          - str: plot one model
          - sequence[str]: plot multiple models

    use_calibrated / calibration_method:
        If use_calibrated=True, calibration_method must be provided (e.g., "beta") and the
        calibrated prediction keys must exist in the fold dicts.

    n_grid:
        Number of thresholds in the uniform grid over [0, 1].

    mode:
        Threshold selection strategy ("train_threshold", "test_threshold", "split_best").

    method_alias:
        Optional mapping from internal model keys to display labels on the x-axis.

    split_palette:
        Optional colors for bars, e.g. {"Train": "#138CFD", "Test": "#000000"}.

    show_baseline / baseline_value / baseline_color / baseline_lw / baseline_ls:
        Control the horizontal baseline reference line.

    annotate_mean_sd / annotate_decimals / annotate_font_size / annotate_offset:
        Controls for the "mean ± SD" text annotations.

    ylim:
        Optional y-axis limits (ymin, ymax). If None, limits are chosen automatically and may
        expand to avoid clipping annotations.

    print_threshold_summary:
        If True and mode is "train_threshold" or "test_threshold", print mean ± SD of selected
        thresholds (t*) across folds for each model.

    Returns
    -------
    None
        Displays a matplotlib figure.
    """
    # -------------------------
    # Defaults / validation
    # -------------------------
    if use_calibrated and calibration_method is None:
        raise ValueError("calibration_method must be provided when use_calibrated=True.")
    if mode not in {"train_threshold", "test_threshold", "split_best"}:
        raise ValueError("mode must be 'train_threshold', 'test_threshold', or 'split_best'.")

    if method_alias is None:
        method_alias = {}
    if split_palette is None:
        split_palette = {"Train": "darkblue", "Test": "darkred"}
    if "Train" not in split_palette or "Test" not in split_palette:
        raise ValueError("split_palette must contain keys 'Train' and 'Test'.")

    # -------------------------
    # Choose models
    # -------------------------
    if model_names is None:
        selected = list(all_results.keys())
    elif isinstance(model_names, str):
        selected = [model_names]
    else:
        selected = list(model_names)

    missing = [m for m in selected if m not in all_results]
    if missing:
        raise KeyError(
            f"Model(s) not found in all_results: {missing}. Available: {list(all_results.keys())}"
        )

    # display labels + uniqueness check
    model_labels = [method_alias.get(m, m) for m in selected]
    if len(set(model_labels)) != len(model_labels):
        # find duplicates
        seen = set()
        dupes = sorted({x for x in model_labels if (x in seen) or seen.add(x) is None and False})  # type: ignore
        # the trick above is messy; just do a clean count:
        dupes = sorted({x for x in model_labels if model_labels.count(x) > 1})
        raise ValueError(
            f"method_alias causes duplicate model labels: {dupes}. Make aliases unique."
        )

    grid = np.linspace(0.0, 1.0, int(n_grid))

    # -------------------------
    # Helpers
    # -------------------------
    def _get_y_and_scores(r: Mapping[str, Any], split: Literal["train", "test"]) -> tuple[np.ndarray, np.ndarray]:
        if not use_calibrated:
            y_key = "y_train" if split == "train" else "y_test"
            s_key = "y_train_scores" if split == "train" else "y_test_scores"
        else:
            y_key = "y_train" if split == "train" else "y_test"
            s_key = (
                f"cv_calib_train_predictions_{calibration_method}"
                if split == "train"
                else f"calib_test_predictions_{calibration_method}"
            )

        if y_key not in r or s_key not in r:
            raise KeyError(f"Missing keys '{y_key}'/'{s_key}' in fold record.")
        return np.asarray(r[y_key]), np.asarray(r[s_key])

    def _best_ba_and_t(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
        ba = np.array([balanced_accuracy_score(y, (s >= t).astype(int)) for t in grid], dtype=float)
        j = int(np.argmax(ba))
        return float(ba[j]), float(grid[j])

    # -------------------------
    # Compute per-fold BA for each model
    # -------------------------
    train_vals_per_model: list[np.ndarray] = []
    test_vals_per_model: list[np.ndarray] = []
    tstars_per_model: list[np.ndarray] = []

    for model in selected:
        folds = all_results[model]

        train_ba: list[float] = []
        test_ba: list[float] = []
        tstars: list[float] = []

        for r in folds:
            try:
                y_tr, s_tr = _get_y_and_scores(r, "train")
                y_te, s_te = _get_y_and_scores(r, "test")
            except KeyError:
                continue

            if mode == "split_best":
                ba_tr, _ = _best_ba_and_t(y_tr, s_tr)
                ba_te, _ = _best_ba_and_t(y_te, s_te)
                train_ba.append(ba_tr)
                test_ba.append(ba_te)
            else:
                if mode == "train_threshold":
                    _, t_star = _best_ba_and_t(y_tr, s_tr)
                else:  # test_threshold
                    _, t_star = _best_ba_and_t(y_te, s_te)

                tstars.append(t_star)
                train_ba.append(balanced_accuracy_score(y_tr, (s_tr >= t_star).astype(int)))
                test_ba.append(balanced_accuracy_score(y_te, (s_te >= t_star).astype(int)))

        if len(train_ba) == 0:
            raise ValueError(
                f"No usable folds found for model '{model}'. "
                "Check expected score keys for chosen calibration settings."
            )

        train_vals_per_model.append(np.array(train_ba, dtype=float))
        test_vals_per_model.append(np.array(test_ba, dtype=float))
        tstars_per_model.append(np.array(tstars, dtype=float))

    # summarize mean/sd
    train_means = np.array([v.mean() for v in train_vals_per_model], dtype=float)
    test_means = np.array([v.mean() for v in test_vals_per_model], dtype=float)
    train_sds = np.array([v.std(ddof=1) if v.size > 1 else 0.0 for v in train_vals_per_model], dtype=float)
    test_sds = np.array([v.std(ddof=1) if v.size > 1 else 0.0 for v in test_vals_per_model], dtype=float)

    # -------------------------
    # Plot
    # -------------------------


    x = np.arange(len(model_labels), dtype=float)
    width = float(bar_width)

    fig, ax = plt.subplots(figsize=figsize)

    bars_train = ax.bar(
        x - width / 2,
        train_means,
        width,
        yerr=train_sds,
        capsize=capsize,
        color=split_palette["Train"],
        label="Train",
    )
    bars_test = ax.bar(
        x + width / 2,
        test_means,
        width,
        yerr=test_sds,
        capsize=capsize,
        color=split_palette["Test"],
        label="Test",
    )

    baseline_handle = None
    if show_baseline:
        baseline_handle = ax.axhline(
            float(baseline_value),
            linestyle=baseline_ls,
            linewidth=baseline_lw,
            color=baseline_color,
            label=f"Baseline = {baseline_value:.2f}",
        )

    # ---- labels / title (Option A title) ----
    ax.set_title(
        "Balanced accuracy across folds",
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

    # ---- annotations (bold) ----
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

        _annotate(bars_train, train_means, train_sds)
        _annotate(bars_test, test_means, test_sds)

    # ---- y-lims (same as before) ----
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        top = max(
            float(np.max(train_means + train_sds)),
            float(np.max(test_means + test_sds)),
            float(baseline_value) if show_baseline else 0.0,
        )
        pad = 0.08 if annotate_mean_sd else 0.05
        ax.set_ylim(0.0, min(1.10, top + pad))

    # ---- legend order: Train, Test, Baseline ----
    handles, labels = ax.get_legend_handles_labels()

    handle_map = {lab: h for h, lab in zip(handles, labels)}
    ordered_labels = ["Train", "Test"]
    if show_baseline:
        ordered_labels.append(f"Baseline = {baseline_value:.2f}")

    ordered_handles = [handle_map[lbl] for lbl in ordered_labels if lbl in handle_map]

    leg = ax.legend(
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
    # Optional: print threshold summary
    # -------------------------
    if print_threshold_summary and mode in {"train_threshold", "test_threshold"}:
        print("Per-model selected threshold summary (mean ± SD across folds):")
        for label, tarr in zip(model_labels, tstars_per_model):
            if tarr.size == 0:
                print(f"  {label}: (no thresholds computed)")
                continue
            t_mean = float(np.mean(tarr))
            t_sd = float(np.std(tarr, ddof=1)) if tarr.size > 1 else 0.0
            print(f"  {label}: {t_mean:.3f} ± {t_sd:.3f}")






# --------------------------------------------------------------
# Calibration 
# --------------------------------------------------------------
def calibrate_nested_cv_results(
    all_results: Dict[str, List[Dict[str, Any]]],
    bundle: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    x_key: str = "combined_X_raw",
    y_key: str = "combined_y",
    groups_key: Optional[str] = None,
    model_selection: str = "StratifiedKFold",
    n_splits: int = 5,
    calibration_methods: Optional[List[str]] = None,  # e.g. ["platt", "beta"]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Calibrate nested-CV models (per outer fold) using out-of-fold (OOF) probabilities
    computed on the OUTER-TRAIN split, while respecting per-model feature selection.

    Why this function exists
    ------------------------
    After nested CV, each fold dict in `all_results[model_name]` contains:
      - a tuned + fitted estimator under "final_model"
      - "outer_train_idx" and "outer_test_idx" (row indices into the dataset level used)
      - uncalibrated test probabilities under "y_test_scores"

    To calibrate probabilities, we need *training* probabilities that are:
      - out-of-fold (OOF) on the outer-train set (to avoid fitting calibrators on in-fold predictions)
      - produced by a model that sees the exact same feature space that the final model expects

    With your new setup, each model can use its own feature subset (via config). Therefore
    we must:
      1) start from the *superset* X in the bundle (defined by x_key)
      2) slice columns ONCE per model (using prepare_training_bundle and config)
      3) slice rows per fold using outer_train_idx / outer_test_idx

    Behavior / outputs
    ------------------
    For every (model_name, fold_dict) in all_results:
      - Stores OOF uncalibrated train probabilities:
            r["cv_uncalib_train_predictions"]  -> shape (n_train,)

      - For each calibration method requested, fits a calibrator on the OOF train probs and y_train,
        then produces calibrated train + test probabilities:

        If "platt":
            r["calibration_method_platt"] = "platt"
            r["calibrator_platt"] = fitted LogisticRegression
            r["cv_calib_train_predictions_platt"] -> calibrated OOF train probs
            r["calib_test_predictions_platt"] -> calibrated outer-test probs

        If "beta":
            r["calibration_method_beta"] = "beta"
            r["calibrator_beta"] = fitted BetaCalibration
            r["cv_calib_train_predictions_beta"] -> calibrated OOF train probs
            r["calib_test_predictions_beta"] -> calibrated outer-test probs

    Notes
    -----
    - This function does NOT refit the model on the full outer-train split. It only uses
      cross_val_predict(clone(final_model)) to generate OOF train probabilities, and then
      fits calibrators on those probabilities.
    - Test calibration is applied to r["y_test_scores"] (already produced by the fitted final_model
      during nested CV), so we do not need X_test to calibrate.
    - If model_selection == "StratifiedGroupKFold", you must pass groups_key so groups can be
      reconstructed on the outer-train rows for cross_val_predict.
    """
    # ------------------------------------------------------------------
    # 0) Handle calibration method selection + validation
    # ------------------------------------------------------------------
    if calibration_methods is None:
        calibration_methods = ["platt"]  # default behavior

    if len(calibration_methods) == 0:
        raise ValueError("calibration_methods must contain at least one method.")

    # normalize user input
    calibration_methods = [m.lower() for m in calibration_methods]

    # validate supported methods
    supported = {"platt", "beta"}
    unknown = set(calibration_methods) - supported
    if unknown:
        raise ValueError(
            f"Unsupported calibration methods: {unknown}. Supported methods: {supported}."
        )

    # ------------------------------------------------------------------
    # 1) Pull the correct dataset level from the bundle (superset X/y)
    # ------------------------------------------------------------------
    if x_key not in bundle:
        raise KeyError(f"bundle missing x_key='{x_key}'")
    if y_key not in bundle:
        raise KeyError(f"bundle missing y_key='{y_key}'")

    # IMPORTANT:
    # - X_full is the *superset* feature matrix at the dataset level specified by x_key
    # - feature_names_full must correspond to the columns of X_full
    X_full = np.asarray(bundle[x_key])
    y = np.asarray(bundle[y_key])

    if X_full.ndim != 2:
        raise ValueError(f"bundle[{x_key}] must be 2D, got shape {X_full.shape}")
    if y.ndim != 1:
        raise ValueError(f"bundle[{y_key}] must be 1D, got shape {y.shape}")
    if X_full.shape[0] != len(y):
        raise ValueError(
            f"X/y mismatch for keys ({x_key}, {y_key}): X rows={X_full.shape[0]} vs len(y)={len(y)}"
        )

    # Choose which feature-names key to use.
    # If you have separate names for aggregated features, prefer those.
    feature_names_key = "feature_names"
    if str(x_key).startswith("combined_"):
        if "combined_feature_names" in bundle:
            feature_names_key = "combined_feature_names"

    if feature_names_key not in bundle:
        raise KeyError(
            f"bundle missing '{feature_names_key}'. Needed to map names->columns for feature slicing."
        )

    feature_names_full = list(bundle[feature_names_key])
    if X_full.shape[1] != len(feature_names_full):
        raise ValueError(
            f"Mismatch: bundle[{x_key}] has {X_full.shape[1]} cols but "
            f"bundle[{feature_names_key}] has {len(feature_names_full)} names."
        )

    # Optional group labels at the same dataset level (needed only for StratifiedGroupKFold)
    groups_all: Optional[np.ndarray] = None
    if groups_key is not None:
        if groups_key not in bundle:
            raise KeyError(f"bundle missing groups_key='{groups_key}'")
        groups_all = np.asarray(bundle[groups_key])
        if groups_all.ndim != 1:
            raise ValueError(f"bundle[{groups_key}] must be 1D, got shape {groups_all.shape}")
        if len(groups_all) != len(y):
            raise ValueError(
                f"groups/y mismatch for key {groups_key}: len(groups)={len(groups_all)} vs len(y)={len(y)}"
            )

    # ------------------------------------------------------------------
    # 2) Loop over models; for each model slice X_full -> X_model ONCE
    # ------------------------------------------------------------------
    for model_name, folds in all_results.items():
        # Ensure this model exists in config
        if model_name not in cfg["models"]:
            raise KeyError(f"Model '{model_name}' not found in cfg['models'].")

        m_cfg = cfg["models"][model_name]

        # Per-model feature selection knobs (same as training):
        #   - either a list of feature names OR an integer n_features (prefix mode)
        keep_features_cfg = m_cfg.get("feature_names", None)  # list[str] | None
        n_features_cfg = m_cfg.get("n_features", None)        # int | None

        # If your nested CV stored the exact feature names used, prefer those.
        # This protects you if config changes later.
        keep_features_from_results: Optional[List[str]] = None
        if len(folds) > 0 and "feature_names" in folds[0]:
            keep_features_from_results = list(folds[0]["feature_names"])

        if keep_features_from_results is not None:
            keep_features = keep_features_from_results
            n_features_model = None
        else:
            keep_features = keep_features_cfg
            n_features_model = n_features_cfg

        # Do not allow both selection methods at once (ambiguous)
        if keep_features is not None and n_features_model is not None:
            raise ValueError(
                f"{model_name}: set only one of 'feature_names' or 'n_features' (or neither)."
            )

        # Build the "view bundle" expected by prepare_training_bundle
        view_bundle = {"X_raw": X_full, "feature_names": feature_names_full}

        # Slice columns ONCE per model to match the model's trained feature space
        if keep_features is not None or n_features_model is not None:
            mb = prepare_training_bundle(
                view_bundle,
                n_features=n_features_model,
                keep_features=keep_features,
                strict=m_cfg.get("feature_strict", True),
                dedupe=True,
                copy_bundle=True,
            )
        else:
            mb = view_bundle  # no slicing requested

        X_model = np.asarray(mb["X_raw"])                 # shape: (n_samples, D_model)
        feature_names_model = list(mb["feature_names"])   # length: D_model

        # ------------------------------------------------------------------
        # 3) Loop over outer folds for this model and perform calibration
        # ------------------------------------------------------------------
        for r in folds:
            outer_train_idx = r["outer_train_idx"]
            outer_test_idx = r["outer_test_idx"]

            # Outer-train subset (rows only; columns already model-specific)
            X_train = X_model[outer_train_idx]   # shape: (n_train, D_model)
            y_train = y[outer_train_idx]         # shape: (n_train,)

            # NOTE:
            # We don't need X_test to apply calibration, because we calibrate the
            # already-stored uncalibrated test probabilities in r["y_test_scores"].
            # X_test = X_model[outer_test_idx]
            # y_test = y[outer_test_idx]

            # Groups only matter if using StratifiedGroupKFold
            groups_train = None
            if (groups_all is not None) and (model_selection == "StratifiedGroupKFold"):
                groups_train = groups_all[outer_train_idx]

            # Clone the final tuned estimator (unfitted) to generate OOF train probs
            final_model = r["final_model"]
            clf = clone(final_model)

            # Build a "regular CV" splitter (not nested) just to produce OOF predictions
            _, inner_cv = make_outer_inner_cv(
                model_selection=model_selection,
                n_outer_splits=n_splits,   # arbitrary but valid
                n_inner_splits=n_splits,   # K folds for OOF prediction
                outer_trial_idx=r["trial"],  # seed for reproducibility
            )

            # cross_val_predict supports passing groups via keyword
            cv_kwargs = {}
            if groups_train is not None:
                cv_kwargs["groups"] = groups_train

            # OOF predicted probabilities for the positive class on outer-train
            cv_probs_train = cross_val_predict(
                clf,
                X_train,
                y_train,
                cv=inner_cv,
                method="predict_proba",
                **cv_kwargs,
            )
            cv_uncalib_train_predictions = cv_probs_train[:, 1]
            r["cv_uncalib_train_predictions"] = cv_uncalib_train_predictions

            # Uncalibrated outer-test probabilities from nested CV training step
            testset_preds_uncalib = r["y_test_scores"]

            # Store some traceability metadata
            r["calib_x_key"] = x_key
            r["calib_y_key"] = y_key
            r["calib_feature_names_key"] = feature_names_key
            r["calib_feature_names"] = feature_names_model

            # --------------------------------------------------------------
            # 4) Fit calibrators on (OOF train probs, y_train) and apply
            # --------------------------------------------------------------
            if "platt" in calibration_methods:
                # Platt scaling: logistic regression on the 1D score
                calibrator_platt = LogisticRegression(
                    C=np.inf, solver="lbfgs", max_iter=200000
                )
                calibrator_platt.fit(
                    cv_uncalib_train_predictions.reshape(-1, 1),
                    y_train,
                )

                r["calibration_method_platt"] = "platt"
                r["calibrator_platt"] = calibrator_platt

                # Calibrated OOF train probabilities
                r["cv_calib_train_predictions_platt"] = calibrator_platt.predict_proba(
                    cv_uncalib_train_predictions.reshape(-1, 1)
                )[:, 1]

                # Calibrated outer-test probabilities (apply to uncalibrated test scores)
                r["calib_test_predictions_platt"] = calibrator_platt.predict_proba(
                    testset_preds_uncalib.reshape(-1, 1)
                )[:, 1]

            if "beta" in calibration_methods:
                # Beta calibration on the 1D score
                calibrator_beta = BetaCalibration(parameters="abm")
                calibrator_beta.fit(
                    cv_uncalib_train_predictions,
                    y_train,
                )

                r["calibration_method_beta"] = "beta"
                r["calibrator_beta"] = calibrator_beta

                r["cv_calib_train_predictions_beta"] = calibrator_beta.predict(
                    cv_uncalib_train_predictions
                )
                r["calib_test_predictions_beta"] = calibrator_beta.predict(
                    testset_preds_uncalib
                )

    return all_results


# def calibrate_nested_cv_results(
#     all_results: Dict[str, List[Dict[str, Any]]],
#     X: np.ndarray,
#     y: np.ndarray,
#     model_selection: str = "StratifiedKFold",
#     n_splits: int = 5,
#     groups: Optional[np.ndarray] = None,
#     calibration_methods: Optional[List[str]] = None,  # e.g. ["platt", "beta"]
# ) -> Dict[str, List[Dict[str, Any]]]:
#     """
#     For each (model, trial, outer_fold) entry in all_results, reconstruct the
#     outer-train split and compute out-of-fold predicted probabilities via
#     regular cross-validation (no nesting), then calibrate them using one or
#     more calibration methods.

#     Always stores uncalibrated OOF train probabilities as:

#         - "cv_uncalib_train_predictions": np.ndarray of shape (n_train,)

#     For each method in calibration_methods, performs calibration and stores:

#       If "platt" in calibration_methods:
#         - "calibration_method_platt"               : str, "platt"
#         - "calibrator_platt"                       : fitted LogisticRegression
#         - "cv_calib_train_predictions_platt"       : calibrated probs (train, OOF)
#         - "calib_test_predictions_platt"           : calibrated probs (test)

#       If "beta" in calibration_methods:
#         - "calibration_method_beta"                : str, "beta"
#         - "calibrator_beta"                        : fitted BetaCalibration
#         - "cv_calib_train_predictions_beta"        : calibrated probs (train, OOF)
#         - "calib_test_predictions_beta"            : calibrated probs (test)
#     """
#     # Default: only Platt scaling if user doesn’t specify
#     if calibration_methods is None:
#         calibration_methods = ["platt"]

#     if len(calibration_methods) == 0:
#         raise ValueError("calibration_methods must contain at least one method.")

#     # Normalize and validate supported methods
#     calibration_methods = [m.lower() for m in calibration_methods]
#     supported = {"platt", "beta"}
#     unknown = set(calibration_methods) - supported
#     if unknown:
#         raise ValueError(
#             f"Unsupported calibration methods: {unknown}. "
#             f"Supported methods: {supported}."
#         )

#     for model_name, folds in all_results.items():
#         for r in folds:
#             outer_train_idx = r["outer_train_idx"]
#             outer_test_idx = r["outer_test_idx"]

#             # Rebuild train/test subset for this outer fold
#             X_train = X[outer_train_idx]
#             y_train = y[outer_train_idx]

#             X_test = X[outer_test_idx]      # not used now, but handy if needed later
#             y_test = y[outer_test_idx]      # same as above

#             groups_train = None
#             if (groups is not None) and (model_selection == "StratifiedGroupKFold"):
#                 groups_train = groups[outer_train_idx]

#             # Unfitted clone of the tuned final model
#             final_model = r["final_model"]
#             clf = clone(final_model)

#             # Build a regular CV splitter (no nested CV here)
#             _, inner_cv = make_outer_inner_cv(
#                 model_selection=model_selection,
#                 n_outer_splits=n_splits,   # arbitrary but valid
#                 n_inner_splits=n_splits,   # this is our K for regular CV
#                 outer_trial_idx=r["trial"],  # seed for reproducibility
#             )

#             # cross_val_predict with or without groups
#             cv_kwargs = {}
#             if groups_train is not None:
#                 cv_kwargs["groups"] = groups_train

#             cv_probs_train = cross_val_predict(
#                 clf,
#                 X_train,
#                 y_train,
#                 cv=inner_cv,
#                 method="predict_proba",
#                 **cv_kwargs,
#             )

#             # Positive-class OOF probabilities on outer-train (UNCALIBRATED)
#             cv_uncalib_train_predictions = cv_probs_train[:, 1]
#             r["cv_uncalib_train_predictions"] = cv_uncalib_train_predictions

#             # Uncalibrated test scores already stored from nested CV
#             testset_preds_uncalib = r["y_test_scores"]

#             # --------------------------------------------------------------
#             # Apply each requested calibration method
#             # --------------------------------------------------------------
#             if "platt" in calibration_methods:
#                 calibrator_platt = LogisticRegression(C=np.inf,solver="lbfgs", max_iter=200000)
#                 calibrator_platt.fit(
#                     cv_uncalib_train_predictions.reshape(-1, 1),
#                     y_train,
#                 )

#                 cv_calib_train_predictions_platt = calibrator_platt.predict_proba(
#                     cv_uncalib_train_predictions.reshape(-1, 1)
#                 )[:, 1]

#                 calib_test_predictions_platt = calibrator_platt.predict_proba(
#                     testset_preds_uncalib.reshape(-1, 1)
#                 )[:, 1]

#                 r["calibration_method_platt"] = "platt"
#                 r["calibrator_platt"] = calibrator_platt
#                 r["cv_calib_train_predictions_platt"] = (
#                     cv_calib_train_predictions_platt
#                 )
#                 r["calib_test_predictions_platt"] = calib_test_predictions_platt

#             if "beta" in calibration_methods:
#                 calibrator_beta = BetaCalibration(parameters='abm')
#                 calibrator_beta.fit(
#                     cv_uncalib_train_predictions,
#                     y_train,
#                 )

#                 cv_calib_train_predictions_beta = calibrator_beta.predict(
#                     cv_uncalib_train_predictions
#                 )

#                 calib_test_predictions_beta = calibrator_beta.predict(
#                     testset_preds_uncalib
#                 )

#                 r["calibration_method_beta"] = "beta"
#                 r["calibrator_beta"] = calibrator_beta
#                 r["cv_calib_train_predictions_beta"] = (
#                     cv_calib_train_predictions_beta
#                 )
#                 r["calib_test_predictions_beta"] = calib_test_predictions_beta

#     return all_results


# Get slope and intercept of calibration
def _logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute the logit transform in a numerically-stable way.

    The logit is:
        logit(p) = log(p / (1 - p))

    Why we clip:
    ------------
    If p is exactly 0 or 1, logit(p) is -inf or +inf.
    We avoid this by clipping probabilities into [eps, 1-eps].

    Parameters
    ----------
    p:
        Array of predicted probabilities (values intended to be in [0, 1]).
    eps:
        Small value for clipping to avoid infinities in the logit.

    Returns
    -------
    logit_p:
        Array of log-odds values, same shape as p.
    """
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def calibration_slope_intercept_from_probs(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    Compute calibration intercept (alpha) and calibration slope (beta) for binary outcomes.

    We fit the *diagnostic* calibration model:
        P(y=1) = sigmoid(alpha + beta * logit(p_pred))

    Interpretation
    --------------
    - Perfect calibration (in this diagnostic sense): alpha = 0 and beta = 1
    - alpha (intercept): overall under/overprediction ("calibration-in-the-large")
        alpha > 0 -> predicted probabilities tend to be too low (need upward shift in log-odds)
        alpha < 0 -> predicted probabilities tend to be too high (need downward shift in log-odds)
    - beta (slope): how "extreme" the probabilities are
        beta < 1 -> probabilities are too extreme / overconfident (often overfitting)
        beta > 1 -> probabilities are not extreme enough / underconfident

    Parameters
    ----------
    y_true:
        Binary labels (0/1) for each sample.
    p_pred:
        Predicted probabilities for the positive class for each sample.
    eps:
        Clipping value used inside logit(p_pred) to prevent infinities.

    Returns
    -------
    alpha, beta:
        alpha = calibration intercept
        beta  = calibration slope
    """
    # Ensure correct shapes/dtypes
    y_true = np.asarray(y_true).astype(int)
    x = _logit(p_pred, eps=eps).reshape(-1, 1)

    # Fit logistic regression with no regularization (diagnostic fit).
    # If your sklearn version doesn't support penalty="none", use the earlier try/except fallback.
    lr = LogisticRegression(C=np.inf,solver="lbfgs", max_iter=200000) # penalty=None, 
    lr.fit(x, y_true)

    alpha = float(lr.intercept_[0])
    beta  = float(lr.coef_[0, 0])
    return alpha, beta


# -------------------------
# Bootstrap on pooled data
# -------------------------
def bootstrap_alpha_beta_from_pooled(
    y_pool: np.ndarray,
    p_pool: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Bootstrap uncertainty for (alpha, beta) computed on pooled (y, p) pairs.

    Bootstrap procedure (case bootstrap)
    -----------------------------------
    1) Assume pooled pairs (y_i, p_i) are our observed sample of size N.
    2) Repeat n_boot times:
         - sample N indices with replacement
         - refit the calibration diagnostic model
         - store alpha and beta
    3) Summarize the bootstrap distribution with mean/std and percentile CIs.

    Notes
    -----
    - Some bootstrap resamples may be "degenerate" (all y=0 or all y=1),
      in which case logistic regression cannot be fit. Those draws are skipped.
    - n_boot_used may be < n_boot if many degenerate samples occur (rare unless
      the event rate is extremely low/high or N is small).

    Parameters
    ----------
    y_pool:
        Pooled binary labels (0/1), typically concatenated across folds.
    p_pool:
        Pooled predicted probabilities aligned with y_pool.
    n_boot:
        Number of bootstrap resamples to attempt.
    seed:
        Random seed for reproducibility.
    eps:
        Clipping value used inside logit(p) to prevent infinities.

    Returns
    -------
    stats:
        Dictionary with bootstrap summaries:
          - alpha_mean, alpha_std, alpha_ci95
          - beta_mean,  beta_std,  beta_ci95
          - n_boot_used (how many bootstrap fits actually succeeded)
    """
    rng = np.random.default_rng(seed)
    y_pool = np.asarray(y_pool)
    p_pool = np.asarray(p_pool)
    n = len(y_pool)

    alphas: List[float] = []
    betas:  List[float] = []

    for _ in range(n_boot):
        # Sample N rows with replacement
        idx = rng.integers(0, n, size=n)
        y_b = y_pool[idx]
        p_b = p_pool[idx]

        # Skip degenerate samples (all 0s or all 1s)
        if np.unique(y_b).size < 2:
            continue

        a, b = calibration_slope_intercept_from_probs(y_b, p_b, eps=eps)
        alphas.append(a)
        betas.append(b)

    if len(alphas) == 0:
        raise RuntimeError("All bootstrap fits failed (degenerate samples).")

    alphas = np.asarray(alphas, dtype=float)
    betas  = np.asarray(betas, dtype=float)

    return {
        "alpha_mean": float(alphas.mean()),
        "alpha_std":  float(alphas.std(ddof=1)) if len(alphas) > 1 else 0.0,
        "alpha_ci95": (float(np.quantile(alphas, 0.025)), float(np.quantile(alphas, 0.975))),
        "beta_mean":  float(betas.mean()),
        "beta_std":   float(betas.std(ddof=1)) if len(betas) > 1 else 0.0,
        "beta_ci95":  (float(np.quantile(betas, 0.025)), float(np.quantile(betas, 0.975))),
        "n_boot_used": int(len(alphas)),
    }



def pooled_bootstrap_calibration_stats_by_model(
    all_results: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    calibration_methods: Optional[Sequence[str]] = None,
    eps: float = 1e-12,
    n_boot: int = 500,
    seed: int = 42,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Compute pooled calibration intercept (alpha) and slope (beta) for each model and each
    calibration method using:

        P(y=1) = sigmoid(alpha + beta * logit(p_pred))

    This function produces results for TWO evaluation splits per method:

    1) train_oof  (outer-train, but using out-of-fold probabilities)
       - Uncalibrated baseline uses:  cv_uncalib_train_predictions
       - Calibrated methods use:      cv_calib_train_predictions_<method>

    2) outer_test (outer-test probabilities)
       - Uncalibrated baseline uses:  y_test_scores
       - Calibrated methods use:      calib_test_predictions_<method>

    For each split, we:
      - POOL predictions across outer folds (concatenate all folds)
      - FIT alpha/beta once on the pooled data (headline estimate)
      - BOOTSTRAP the pooled data to estimate mean/std and 95% CI for alpha and beta


    How to interpret
    ----------------
    Ideal calibration: β = 1 and α = 0.

    Slope (β): “Are predicted probabilities too extreme or too conservative?”
    - β ≈ 1 : good
    - β < 1 : overconfident / too extreme (high p̂ values are too high; low p̂ too low)
    - β > 1 : underconfident / too conservative (p̂ not extreme enough)

    Intercept (α): “Are probabilities globally biased high or low?” (calibration-in-the-large)
    - α ≈ 0 : good
    - α > 0 : probabilities are generally too low (underpredict positives; shift upward)
    - α < 0 : probabilities are generally too high (overpredict positives; shift downward)

    Notes:
    - α and β are fit jointly; when β ≠ 1, α is conditional on the fitted slope.
    - Adjustments are affine in log-odds: p_cal = sigmoid(α + β * logit(p̂)), so “shifts”
        are not uniform in probability space.
    - In this function, methods are ordered by closeness to the ideal on the chosen split:
        slope by |β_mean − 1| and intercept by |α_mean − 0| for the selected split.      

    Parameters
    ----------
    all_results:
        Dict-like mapping:
          all_results[model_name] -> list of outer-fold result dicts.
        Each outer-fold dict must include:
          - y_train, y_test
          - y_test_scores (uncalibrated outer-test probabilities)
          - cv_uncalib_train_predictions (uncalibrated OOF outer-train probabilities)
        And for each calibrated method <m> (if available):
          - cv_calib_train_predictions_<m>
          - calib_test_predictions_<m>

    calibration_methods:
        Optional list of calibrated method names (the <m> suffix used in the keys above).
        If None, methods are auto-discovered from keys that look like:
          "calib_test_predictions_<m>".
        Note: the uncalibrated baseline ("uncalib") is ALWAYS included.

    eps:
        Small value used to clip probabilities before logit to avoid infinities.

    n_boot:
        Number of bootstrap resamples used to estimate mean/std and 95% CI from pooled data.
        (Some resamples may be skipped if they contain only one class.)

    seed:
        Random seed for bootstrap resampling.

    Returns
    -------
    out:
        Nested dict:
          out[model_name][method] -> flat dict of pooled + bootstrap stats for train_oof and outer_test.

        method includes:
          - "uncalib" (always present)
          - each entry from calibration_methods (or auto-discovered methods)

        Example keys inside out[model][method]:
          - n_folds_train_oof, train_oof_total_sample_count, train_oof_event_rate
          - train_oof_alpha_intercept, train_oof_beta_slope
          - train_oof_alpha_mean, train_oof_alpha_std, train_oof_alpha_ci95
          - train_oof_beta_mean,  train_oof_beta_std,  train_oof_beta_ci95
          - (and the same set for outer_test_*)
    """
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # Outer bar over models
    for model_name, folds in tqdm(all_results.items(), desc="Models", dynamic_ncols=True):
        # -------------------------
        # Decide which calibrated methods to compute
        # -------------------------
        if calibration_methods is None:
            discovered = set()
            for r in folds:
                for k in r.keys():
                    if k.startswith("calib_test_predictions_"):
                        discovered.add(k.replace("calib_test_predictions_", "", 1))
            methods = sorted(discovered)
        else:
            methods = list(calibration_methods)

        # Always include the uncalibrated baseline first
        methods = ["uncalib"] + methods

        out[model_name] = {}

        # Inner bar over methods
        method_bar = tqdm(methods, desc=f"{model_name}: methods", leave=False, dynamic_ncols=True)
        for method in method_bar:
            method_bar.set_postfix_str("pool train_oof")

            # =========================================================
            # 1) POOL TRAIN OOF ACROSS OUTER FOLDS
            # =========================================================
            y_train_all, p_train_all = [], []
            n_folds_train = 0

            for r in folds:
                y_tr = np.asarray(r["y_train"])

                if method == "uncalib":
                    if "cv_uncalib_train_predictions" not in r:
                        continue
                    p_tr = np.asarray(r["cv_uncalib_train_predictions"])
                else:
                    key_tr = f"cv_calib_train_predictions_{method}"
                    if key_tr not in r:
                        continue
                    p_tr = np.asarray(r[key_tr])

                if len(y_tr) != len(p_tr):
                    raise ValueError(
                        f"TRAIN length mismatch in model={model_name}, outer_fold={r.get('outer_fold')}, method={method}: "
                        f"len(y_train)={len(y_tr)} vs len(p_train)={len(p_tr)}"
                    )

                y_train_all.append(y_tr)
                p_train_all.append(p_tr)
                n_folds_train += 1

            if not y_train_all:
                # can't compute train OOF => skip method
                method_bar.set_postfix_str("skip (no train_oof)")
                continue

            y_train_pool = np.concatenate(y_train_all)
            p_train_pool = np.concatenate(p_train_all)

            method_bar.set_postfix_str("fit+boot train_oof")
            train_alpha, train_beta = calibration_slope_intercept_from_probs(
                y_train_pool, p_train_pool, eps=eps
            )
            train_boot = bootstrap_alpha_beta_from_pooled(
                y_train_pool, p_train_pool, n_boot=n_boot, seed=seed, eps=eps
            )

            # =========================================================
            # 2) POOL OUTER TEST ACROSS OUTER FOLDS
            # =========================================================
            method_bar.set_postfix_str("pool outer_test")

            y_test_all, p_test_all = [], []
            n_folds_test = 0

            for r in folds:
                y_te = np.asarray(r["y_test"])

                if method == "uncalib":
                    if "y_test_scores" not in r:
                        continue
                    p_te = np.asarray(r["y_test_scores"])
                else:
                    key_te = f"calib_test_predictions_{method}"
                    if key_te not in r:
                        continue
                    p_te = np.asarray(r[key_te])

                if len(y_te) != len(p_te):
                    raise ValueError(
                        f"TEST length mismatch in model={model_name}, outer_fold={r.get('outer_fold')}, method={method}: "
                        f"len(y_test)={len(y_te)} vs len(p_test)={len(p_te)}"
                    )

                y_test_all.append(y_te)
                p_test_all.append(p_te)
                n_folds_test += 1

            if not y_test_all:
                method_bar.set_postfix_str("skip (no outer_test)")
                continue

            y_test_pool = np.concatenate(y_test_all)
            p_test_pool = np.concatenate(p_test_all)

            method_bar.set_postfix_str("fit+boot outer_test")
            test_alpha, test_beta = calibration_slope_intercept_from_probs(
                y_test_pool, p_test_pool, eps=eps
            )
            test_boot = bootstrap_alpha_beta_from_pooled(
                y_test_pool, p_test_pool, n_boot=n_boot, seed=seed, eps=eps
            )

            # =========================================================
            # 3) SAVE RESULTS
            # =========================================================
            method_bar.set_postfix_str("saving")

            out[model_name][method] = {
                # counts
                "n_folds_train_oof": int(n_folds_train),
                "train_oof_total_sample_count": int(len(y_train_pool)),
                "train_oof_event_rate": float(y_train_pool.mean()),
                "n_folds_outer_test": int(n_folds_test),
                "outer_test_total_sample_count": int(len(y_test_pool)),
                "outer_test_event_rate": float(y_test_pool.mean()),

                # pooled fit
                "train_oof_alpha_intercept": float(train_alpha),
                "train_oof_beta_slope": float(train_beta),
                "outer_test_alpha_intercept": float(test_alpha),
                "outer_test_beta_slope": float(test_beta),

                # bootstrap summary (train oof)
                "train_oof_alpha_mean": train_boot["alpha_mean"],
                "train_oof_alpha_std":  train_boot["alpha_std"],
                "train_oof_beta_mean":  train_boot["beta_mean"],
                "train_oof_beta_std":   train_boot["beta_std"],
                "train_oof_alpha_ci95": train_boot["alpha_ci95"],
                "train_oof_beta_ci95":  train_boot["beta_ci95"],
                "train_oof_n_boot_used": train_boot["n_boot_used"],

                # bootstrap summary (outer test)
                "outer_test_alpha_mean": test_boot["alpha_mean"],
                "outer_test_alpha_std":  test_boot["alpha_std"],
                "outer_test_beta_mean":  test_boot["beta_mean"],
                "outer_test_beta_std":   test_boot["beta_std"],
                "outer_test_alpha_ci95": test_boot["alpha_ci95"],
                "outer_test_beta_ci95":  test_boot["beta_ci95"],
                "outer_test_n_boot_used": test_boot["n_boot_used"],
            }

            method_bar.set_postfix_str("done")

        # close inner bar explicitly (helps some notebook renderers)
        method_bar.close()

    return out



def plot_calibration_param(
    calibration_stats: Mapping[str, Mapping[str, Any]],
    *,
    model_name: str | None = None,
    methods: Sequence[str] | None = None,
    method_alias: Mapping[str, str] | None = None,
    split_palette: Mapping[str, str] | None = None,
    sort_by: str = "test",          # "train" or "test"
    figsize: tuple[float, float] = (16, 5),
    font_size: float = 11.0,
    include_uncalib: bool = True,   # baseline method key is "uncalib"
    legend_loc: str = "best",
    show_ideal_line: bool = True,
    ideal_line_color: str = "#D5F713",
    ideal_line_lw: float = 1.5,
    ideal_line_ls: str = "--",
    # y-limits (optional)
    ylim_beta: tuple[float, float] | None = None,   # slope plot y-lim
    ylim_alpha: tuple[float, float] | None = None,  # intercept plot y-lim
    # NEW: print table of values used in plot
    print_table: bool = False,
    table_decimals: int = 4,
) -> None:
    """
        Plot calibration slope (beta) and intercept (alpha) across methods as Train vs Test
        grouped bar charts using ONLY bootstrap mean ± bootstrap std.

        How to interpret
        ----------------
        Ideal calibration: β = 1 and α = 0.

        Slope (β): “Are predicted probabilities too extreme or too conservative?”
        - β ≈ 1 : good
        - β < 1 : overconfident / too extreme (high p̂ values are too high; low p̂ too low)
        - β > 1 : underconfident / too conservative (p̂ not extreme enough)

        Intercept (α): “Are probabilities globally biased high or low?” (calibration-in-the-large)
        - α ≈ 0 : good
        - α > 0 : probabilities are generally too low (underpredict positives; shift upward)
        - α < 0 : probabilities are generally too high (overpredict positives; shift downward)

        Notes:
        - α and β are fit jointly; when β ≠ 1, α is conditional on the fitted slope.
        - Adjustments are affine in log-odds: p_cal = sigmoid(α + β * logit(p̂)), so “shifts”
            are not uniform in probability space.
        - In this function, methods are ordered by closeness to the ideal on the chosen split:
            slope by |β_mean − 1| and intercept by |α_mean − 0| for the selected split.

            
        IMPORTANT (what Train/Test mean here)
        -------------------------------------
        - "Train" corresponds to pooled OUT-OF-FOLD (OOF) predictions on the outer-train data.
        It reads keys like: train_oof_beta_mean, train_oof_beta_std, etc.
        - "Test" corresponds to pooled OUTER-TEST predictions.
        It reads keys like: outer_test_beta_mean, outer_test_beta_std, etc.

        Expected structure
        ----------------------
        calibration_stats[model_name][method] is a flat dict containing keys like:

        Train (OOF):
            train_oof_alpha_mean, train_oof_alpha_std
            train_oof_beta_mean,  train_oof_beta_std

        Test (outer test):
            outer_test_alpha_mean, outer_test_alpha_std
            outer_test_beta_mean,  outer_test_beta_std

        Parameters
        ----------
        calibration_stats:
            Output of pooled_bootstrap_calibration_stats_by_model().

        model_name:
            If None, plot all models found in calibration_stats.

        methods:
            Optional subset of methods to plot. If None, plot all methods for that model.

        method_alias:
            Optional mapping from method key -> display label on x-axis.

        split_palette:
            Bar colors for Train/Test, e.g. {"Train": "#1587F8", "Test": "#F14949"}.
            If None, defaults are used.

        sort_by:
            Which split drives ordering by closeness to ideal:
            - "train" or "test"

        include_uncalib:
            Whether to include the baseline method named "uncalib".

        ylim_beta / ylim_alpha:
            Optional y-axis limits for slope and intercept plots respectively.

        table_decimals:
            Number of decimals to show in the printed table only.

        Notes
        -----
        Ideal targets:
        - beta (slope) ideal = 1.0
        - alpha (intercept) ideal = 0.0
      """
    if split_palette is None:
        split_palette = {"Train": "#1587F8", "Test": "#F14949"}
    if method_alias is None:
        method_alias = {}

    if sort_by not in ("train", "test"):
        raise ValueError("sort_by must be 'train' or 'test'")
    sort_prefix = "train_oof" if sort_by == "train" else "outer_test"

    # Plot all models if model_name is None
    if model_name is None:
        for m in calibration_stats.keys():
            plot_calibration_param(
                calibration_stats,
                model_name=m,
                methods=methods,
                method_alias=method_alias,
                split_palette=split_palette,
                sort_by=sort_by,
                figsize=figsize,
                font_size=font_size,
                include_uncalib=include_uncalib,
                legend_loc=legend_loc,
                show_ideal_line=show_ideal_line,
                ideal_line_color=ideal_line_color,
                ideal_line_lw=ideal_line_lw,
                ideal_line_ls=ideal_line_ls,
                ylim_beta=ylim_beta,
                ylim_alpha=ylim_alpha,
                print_table=print_table,
                table_decimals=table_decimals,
            )
        return

    model_stats = calibration_stats[model_name]
    methods_to_plot = list(methods) if methods is not None else list(model_stats.keys())
    if not include_uncalib:
        methods_to_plot = [m for m in methods_to_plot if m != "uncalib"]

    def _has_keys(m: str, param_prefix: str) -> bool:
        needed = [
            f"train_oof_{param_prefix}_mean",
            f"train_oof_{param_prefix}_std",
            f"outer_test_{param_prefix}_mean",
            f"outer_test_{param_prefix}_std",
        ]
        return (m in model_stats) and all(k in model_stats[m] for k in needed)

    def _print_table(ms_sorted: list[str], param_prefix: str, ideal: float, title: str) -> None:
        fmt = f"{{:.{table_decimals}f}}"

        header = (
            f"\n{model_name} — {title} (sorted by {sort_by} closeness to ideal={ideal})\n"
            + "-" * 90
        )
        print(header)
        print(f"{'Method':<28} {'Train mean±std':<26} {'Test mean±std':<26} {'sort(|mean-ideal|)':>14}")
        #print("-" * 90)

        for m in ms_sorted:
            disp = method_alias.get(m, m)

            tr_mean = float(model_stats[m][f"train_oof_{param_prefix}_mean"])
            tr_std  = float(model_stats[m][f"train_oof_{param_prefix}_std"])
            te_mean = float(model_stats[m][f"outer_test_{param_prefix}_mean"])
            te_std  = float(model_stats[m][f"outer_test_{param_prefix}_std"])

            sort_mean = float(model_stats[m][f"{sort_prefix}_{param_prefix}_mean"])
            dist = abs(sort_mean - float(ideal))

            train_str = f"{fmt.format(tr_mean)} ± {fmt.format(tr_std)}"
            test_str  = f"{fmt.format(te_mean)} ± {fmt.format(te_std)}"

            print(f"{disp:<28} {train_str:<26} {test_str:<26} {fmt.format(dist):>14}")

        print("-" * 90)

    def _plot_param(param_prefix: str, ideal: float, title_param: str, ylab: str, ylim: tuple[float, float] | None):
        ms = [m for m in methods_to_plot if _has_keys(m, param_prefix)]
        if len(ms) == 0:
            print(f"[plot_calibration_param] No methods found with required keys for {param_prefix} in model={model_name}")
            return

        # Sort by closeness to ideal on chosen split (bootstrap mean)
        ms_sorted = sorted(
            ms,
            key=lambda m: abs(float(model_stats[m][f"{sort_prefix}_{param_prefix}_mean"]) - float(ideal))
        )

        if print_table:
            _print_table(ms_sorted, param_prefix, ideal, title_param)

        labels = [method_alias.get(m, m) for m in ms_sorted]

        train_means = np.array([model_stats[m][f"train_oof_{param_prefix}_mean"] for m in ms_sorted], dtype=float)
        train_stds  = np.array([model_stats[m][f"train_oof_{param_prefix}_std"]  for m in ms_sorted], dtype=float)

        test_means  = np.array([model_stats[m][f"outer_test_{param_prefix}_mean"] for m in ms_sorted], dtype=float)
        test_stds   = np.array([model_stats[m][f"outer_test_{param_prefix}_std"]  for m in ms_sorted], dtype=float)

        x = np.arange(len(ms_sorted))
        w = 0.38

        plt.figure(figsize=figsize)

        train_bars = plt.bar(
            x - w/2, train_means, w, yerr=train_stds, capsize=3,
            label="Train", color=split_palette["Train"]
        )
        test_bars = plt.bar(
            x + w/2, test_means, w, yerr=test_stds, capsize=3,
            label="Test", color=split_palette["Test"]
        )

        ideal_line = None
        if show_ideal_line:
            ideal_line = plt.axhline(
                float(ideal),
                linestyle=ideal_line_ls,
                linewidth=ideal_line_lw,
                color=ideal_line_color,
                label=f"Ideal = {ideal:.1f}",
            )

        plt.title(
            f"{model_name}: {title_param} across calibration methods",
            fontsize=font_size + 2, fontweight="bold"
        )
        plt.ylabel(ylab, fontsize=font_size, fontweight="bold")
        plt.xlabel("Calibration method", fontsize=font_size, fontweight="bold")

        plt.xticks(x, labels, rotation=25, ha="right", fontsize=font_size)
        plt.yticks(fontsize=font_size)

        if ylim is not None:
            plt.ylim(ylim)

        # Force legend order: Train, Test, Ideal
        handles = [train_bars, test_bars]
        legend_labels = ["Train", "Test"]
        if ideal_line is not None:
            handles.append(ideal_line)
            legend_labels.append(f"Ideal = {ideal:.1f}")

        plt.legend(handles, legend_labels, loc=legend_loc)
        plt.tight_layout()
        plt.show()

    _plot_param(param_prefix="beta",  ideal=1.0, title_param="Slope (β)",     ylab="Slope (β)",     ylim=ylim_beta)
    _plot_param(param_prefix="alpha", ideal=0.0, title_param="Intercept (α)", ylab="Intercept (α)", ylim=ylim_alpha)




def calibration_curve_local(
    y_true,
    y_prob,
    *,
    pos_label=None,
    n_bins=5,
    strategy="uniform",
):
    """
    Compute true and predicted probabilities for a calibration curve,
    plus approximate 95% confidence intervals and bin counts.

    Returns
    -------
    prob_true : ndarray of shape (<= n_bins,)
        Fraction of positives in each non-empty bin.

    prob_pred : ndarray of shape (<= n_bins,)
        Mean predicted probability in each non-empty bin.

    lower : ndarray of shape (<= n_bins,)
        Lower bound of approximate 95% CI for prob_true (normal approximation).

    upper : ndarray of shape (<= n_bins,)
        Upper bound of approximate 95% CI for prob_true (normal approximation).

    count : ndarray of shape (<= n_bins,)
        Number of samples in each non-empty bin.
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)
    pos_label = _check_pos_label_consistency(pos_label, y_true)

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            f"Only binary classification is supported. Provided labels {labels}."
        )
    y_true = y_true == pos_label

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.searchsorted(bins[1:-1], y_prob)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    count = bin_total[nonzero]

    # Approximate 95% CI using normal approximation to binomial proportion
    se = np.sqrt(prob_true * (1.0 - prob_true) / count)
    z = 1.96
    lower = prob_true - z * se
    upper = prob_true + z * se

    lower = np.clip(lower, 0.0, 1.0)
    upper = np.clip(upper, 0.0, 1.0)

    return prob_true, prob_pred, lower, upper, count


def extract_calibration_data(
    all_results: Dict[str, List[Dict[str, Any]]],
    model_name: str,
    calibration_methods: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Extract concatenated labels and probabilities for calibration curves
    from `all_results` for a given model, returned in a nested dict
    keyed by model_name.

    For the specified model, this function loops over all outer folds and
    concatenates:

      Train side (outer train subset):
        - y:                    r["y_train"]
        - uncalib:              r["cv_uncalib_train_predictions"]
        - <method>:             r["cv_calib_train_predictions_<method>"]

      Test side (outer test subset):
        - y:                    r["y_test"]
        - uncalib:              r["y_test_scores"]
        - <method>:             r["calib_test_predictions_<method>"]

    Parameters
    ----------
    all_results : dict
        The nested-CV results structure produced by your pipeline, e.g.:
        {
          "logistic_regression": [ { ... per outer fold ... }, ... ],
          "random_forest": [ ... ],
        }

    model_name : str
        Name of the model whose calibration data you want to extract,
        e.g. "logistic_regression". Must be a key in `all_results`.

    calibration_methods : list of str or None, default=None
        List of calibration methods to extract, e.g. ["platt", "beta"].
        If None, the methods are auto-discovered from the keys in the
        first fold by looking for:
          - "cv_calib_train_predictions_<method>"

    Returns
    -------
    calib_data : dict
        Nested dict keyed by model name:

        {
          model_name: {
            "train": {
              "y":        y_train_all,             # shape (n_train_total,)
              "uncalib":  p_train_uncalib_all,     # shape (n_train_total,)
              "platt":    p_train_platt_all,       # if available
              "beta":     p_train_beta_all,        # if available
              ...
            },
            "test": {
              "y":        y_test_all,              # shape (n_test_total,)
              "uncalib":  p_test_uncalib_all,      # shape (n_test_total,)
              "platt":    p_test_platt_all,        # if available
              "beta":     p_test_beta_all,         # if available
              ...
            },
          }
        }
    """
    if model_name not in all_results:
        raise KeyError(f"Model '{model_name}' not found in all_results.")

    folds = all_results[model_name]
    if len(folds) == 0:
        raise ValueError(f"No folds found for model '{model_name}' in all_results.")

    # ------------------------------------------------------------------
    # Auto-discover calibration methods if not provided
    # ------------------------------------------------------------------
    if calibration_methods is None:
        first = folds[0]
        prefix = "cv_calib_train_predictions_"
        detected = []
        for key in first.keys():
            if key.startswith(prefix):
                method = key[len(prefix):]
                detected.append(method)
        calibration_methods = sorted(detected)

    # ------------------------------------------------------------------
    # Collect per-fold data
    # ------------------------------------------------------------------
    y_train_list = []
    p_train_unc_list = []
    p_train_calib = {method: [] for method in calibration_methods}

    y_test_list = []
    p_test_unc_list = []
    p_test_calib = {method: [] for method in calibration_methods}

    for r in folds:
        # --- Train side ---
        y_train_list.append(r["y_train"])
        p_train_unc_list.append(r["cv_uncalib_train_predictions"])

        for method in calibration_methods:
            train_key = f"cv_calib_train_predictions_{method}"
            if train_key in r:
                p_train_calib[method].append(r[train_key])

        # --- Test side ---
        y_test_list.append(r["y_test"])
        p_test_unc_list.append(r["y_test_scores"])

        for method in calibration_methods:
            test_key = f"calib_test_predictions_{method}"
            if test_key in r:
                p_test_calib[method].append(r[test_key])

    # ------------------------------------------------------------------
    # Concatenate across folds
    # ------------------------------------------------------------------
    y_train_all = np.concatenate(y_train_list)
    p_train_uncalib_all = np.concatenate(p_train_unc_list)

    y_test_all = np.concatenate(y_test_list)
    p_test_uncalib_all = np.concatenate(p_test_unc_list)

    train_dict: Dict[str, Any] = {
        "y": y_train_all,
        "uncalib": p_train_uncalib_all,
    }
    test_dict: Dict[str, Any] = {
        "y": y_test_all,
        "uncalib": p_test_uncalib_all,
    }

    for method in calibration_methods:
        if len(p_train_calib[method]) > 0:
            train_dict[method] = np.concatenate(p_train_calib[method])
        else:
            train_dict[method] = None

        if len(p_test_calib[method]) > 0:
            test_dict[method] = np.concatenate(p_test_calib[method])
        else:
            test_dict[method] = None

    calib_data = {
        model_name: {
            "train": train_dict,
            "test": test_dict,
        }
    }
    return calib_data



def calibration_data_prep(
    calib_data,
    n_bins: int = 10,
    strategy: str = "uniform",
):
    """
    For each model and split ('train', 'test') in calib_data, run
    calibration_curve_local on each available probability vector
    (uncalibrated + each calibration method) and store the resulting
    calibration-curve data back into calib_data.

    After this function, for each model and split you get, e.g.:

      calib_data[model]["train_curves"]["uncalib"] = {
          "prob_true": ...,
          "prob_pred": ...,
          "lower":     ...,
          "upper":     ...,
          "count":     ...,
      }
    """
    for model_name, model_dict in calib_data.items():
        for split in ["train", "test"]:
            if split not in model_dict:
                continue

            split_dict = model_dict[split]
            y = split_dict.get("y", None)
            if y is None:
                continue

            # Container for this split's curves
            curves_key = f"{split}_curves"
            if curves_key not in model_dict:
                model_dict[curves_key] = {}
            curves_dict = model_dict[curves_key]

            # Loop over all variants: "uncalib", "platt", "beta", ...
            for variant_name, probs in split_dict.items():
                if variant_name == "y":
                    continue  # labels, not probs
                if probs is None:
                    continue  # method not available / not computed

                prob_true, prob_pred, lower, upper, count = calibration_curve_local(
                    y_true=y,
                    y_prob=probs,
                    n_bins=n_bins,
                    strategy=strategy,
                )

                curves_dict[variant_name] = {
                    "prob_true": prob_true,
                    "prob_pred": prob_pred,
                    "lower": lower,
                    "upper": upper,
                    "count": count,   # NEW
                }

    return calib_data


def plot_calibration_curves(
    calib_data,
    model_name: str,
    methods=None,                 # e.g. ["platt", "beta"]
    include_uncalibrated: bool = True,
    figsize=(6, 6),
    *,
    calibration_stats=None,       # output of pooled_bootstrap_calibration_stats_by_model
    show_alpha_beta: bool = True, # if calibration_stats provided, append α/β to labels
    alpha_beta_decimals: int = 3, # formatting only α/β labels
    # ---- PDP-like styling / aliasing ----
    method_alias: Mapping[str, str] | None = None,
    sns_style: str = "whitegrid",
    font_size: int = 12,
    # colors
    perfect_line_color: str = "black",
    perfect_line_ls: str = "--",
    uncalib_color: str = "#1587F8",
    calibrated_color_map: Mapping[str, str] | None = None,  # optional per-method override
    # band styling
    band_alpha: float = 0.15,
    band_edgecolor: str | None = None,
    # line styling
    lw: float = 2.2,
    marker: str = "o",
    markersize: float = 5,
    capsize: float = 0,
):
    """
    Plot train/test calibration curves for a single model from precomputed binned calibration data.

    Styling matches the PDP plots: line+markers for the mean curve plus a shaded uncertainty band
    (lower/upper), with bold labels/ticks/title and a clean seaborn theme. Axes follow scikit-learn
    conventions: x = mean predicted probability (positive class), y = fraction of positives.

    Parameters (brief):
      calib_data: output of extract_calibration_data + calibration_data_prep (must include {split}_curves).
      model_name: model key inside calib_data.
      methods: calibration methods to include (e.g., ["beta"]); None auto-discovers from train_curves.
      include_uncalibrated: whether to include the "uncalib" curve.
      figsize: figure size for each split plot.
      calibration_stats/show_alpha_beta/alpha_beta_decimals: optionally append α/β mean±std to legend labels.
      method_alias: optional mapping model_key -> display name (title only).
      uncalib_color/calibrated_color_map: colors for uncalibrated and per-method calibrated curves.
      band_alpha/band_edgecolor: uncertainty band appearance.
      lw/marker/markersize: mean-curve appearance.
    """


    if method_alias is None:
        method_alias = {}
    if calibrated_color_map is None:
        calibrated_color_map = {}

    if model_name not in calib_data:
        raise KeyError(f"Model '{model_name}' not found in calib_data.")

    display_model = method_alias.get(model_name, model_name)

    model_dict = calib_data[model_name]
    train_curves = model_dict.get("train_curves", {})
    test_curves  = model_dict.get("test_curves", {})

    # Auto-discover methods from train_curves (excluding 'uncalib')
    if methods is None:
        discovered = [k for k in train_curves.keys() if k != "uncalib"]
        methods = sorted(discovered)

    # Seaborn styling (PDP-like)
    sns.set(style=sns_style)

    # Helper: get α/β text for a given split+method
    def _alpha_beta_suffix(split_name: str, method_key: str) -> str:
        if (calibration_stats is None) or (not show_alpha_beta):
            return ""
        if model_name not in calibration_stats:
            return ""
        if method_key not in calibration_stats[model_name]:
            return ""

        stats_prefix = "train_oof" if split_name == "train" else "outer_test"

        entry = calibration_stats[model_name][method_key]
        a_mean_key = f"{stats_prefix}_alpha_mean"
        a_std_key  = f"{stats_prefix}_alpha_std"
        b_mean_key = f"{stats_prefix}_beta_mean"
        b_std_key  = f"{stats_prefix}_beta_std"

        if not all(k in entry for k in (a_mean_key, a_std_key, b_mean_key, b_std_key)):
            return ""

        a_mean = float(entry[a_mean_key])
        a_std  = float(entry[a_std_key])
        b_mean = float(entry[b_mean_key])
        b_std  = float(entry[b_std_key])

        fmt = f"{{:.{alpha_beta_decimals}f}}"
        return f" (α={fmt.format(a_mean)}±{fmt.format(a_std)}, β={fmt.format(b_mean)}±{fmt.format(b_std)})"

    def _variant_label(split_name: str, variant: str) -> str:
        base = "Uncalibrated" if variant == "uncalib" else f"Calibrated: {variant}"
        return base + _alpha_beta_suffix(split_name, variant)

    def _variant_color(variant: str) -> str:
        if variant == "uncalib":
            return uncalib_color
        return calibrated_color_map.get(variant, "#F14949")  # default calibrated color

    # Helper for one split (train OR test)
    def _plot_split(split_name: str, curves: dict):
        if not curves:
            print(f"No '{split_name}_curves' found for model '{model_name}'. Skipping.")
            return

        # Build ordered list of variants to plot
        variants_to_plot: list[str] = []
        if include_uncalibrated and "uncalib" in curves:
            variants_to_plot.append("uncalib")
        for m in methods:
            if m in curves:
                variants_to_plot.append(m)

        if not variants_to_plot:
            print(
                f"No matching variants to plot for split='{split_name}' "
                f"and methods={methods} (include_uncalibrated={include_uncalibrated})."
            )
            return

        plt.figure(figsize=figsize)
        ax = plt.gca()

        # Perfect calibration line
        ax.plot(
            [0, 1], [0, 1],
            linestyle=perfect_line_ls,
            linewidth=2.0,
            color=perfect_line_color,
            label="Perfectly calibrated",
        )

        # Plot each variant: line+markers + shaded band (lower/upper)
        for variant in variants_to_plot:
            c = curves[variant]
            prob_pred = np.asarray(c["prob_pred"], dtype=float)
            prob_true = np.asarray(c["prob_true"], dtype=float)
            lower = np.asarray(c["lower"], dtype=float)
            upper = np.asarray(c["upper"], dtype=float)

            col = _variant_color(variant)
            label = _variant_label(split_name, variant)

            # band first (subtle)
            ax.fill_between(
                prob_pred,
                lower,
                upper,
                alpha=band_alpha,
                color=col,
                edgecolor=band_edgecolor,
                linewidth=0.0 if band_edgecolor is None else 1.0,
            )

            # mean curve on top (PDP-like)
            ax.plot(
                prob_pred,
                prob_true,
                marker=marker,
                markersize=markersize,
                linewidth=lw,
                color=col,
                label=label,
            )

        # Labels / title (bold, clean)
        ax.set_xlabel("Mean predicted probability (positive class)", fontsize=font_size, fontweight="bold")
        ax.set_ylabel("Fraction of positives (positive class)", fontsize=font_size, fontweight="bold")

        ax.set_title(
            f"Calibration curve — {display_model} ({split_name})",
            fontsize=font_size + 2,
            fontweight="bold",
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

        # Bold ticks
        ax.tick_params(axis="both", labelsize=font_size)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontweight("bold")

        # Legend
        ax.legend(loc="best", frameon=True, prop={"size": font_size})

        plt.tight_layout()
        plt.show()

    # ALWAYS plot train and test separately
    _plot_split("train", train_curves)
    _plot_split("test", test_curves)

def plot_calibration_histogram(
    calib_data: Mapping[str, Any],
    model_name: str,
    split: Literal["train", "test"] = "test",
    variant: str = "uncalib",  # "uncalib" or a calibration method like "beta"
    figsize: tuple[float, float] = (7, 4),
    *,
    font_size: int = 12,
    sns_style: str = "whitegrid",
    method_alias: Mapping[str, str] | None = None,
    # ---- colors (match your other plots) ----
    uncalib_color: str = "#1587F8",
    calibrated_color_map: Mapping[str, str] | None = None,
    # ---- bar styling ----
    bar_alpha: float = 0.85,
    edgecolor: Optional[str] = None,
    linewidth: float = 0.8,
    # ---- normalization ----
    normalize: bool = False,
) -> None:
    """
    Plot a prediction histogram over calibration bins for a given model/split/variant.

    Parameters (brief):
      calib_data: dict-like calibration container produced by calibration_data_prep (must include
        {split}_curves[variant]["prob_pred"] (bin centers) and ["count"] (bin counts)).
      model_name: model key inside calib_data.
      split: "train" or "test" curves to use.
      variant: "uncalib" or a calibration method name (e.g., "beta").
      figsize: matplotlib figure size.
      font_size: base font size for labels/ticks/title.
      sns_style: seaborn style (default "whitegrid").
      method_alias: optional mapping model_key -> display name (title only).
      uncalib_color: bar color for uncalibrated variant.
      calibrated_color_map: optional mapping method -> bar color for calibrated variants.
      bar_alpha/edgecolor/linewidth: bar appearance controls.
      normalize: if True, plot fraction of samples per bin instead of raw counts.
    """
    if method_alias is None:
        method_alias = {}
    if calibrated_color_map is None:
        calibrated_color_map = {}

    if model_name not in calib_data:
        raise KeyError(f"Model '{model_name}' not found in calib_data.")
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    model_dict = calib_data[model_name]
    curves_key = f"{split}_curves"
    curves = model_dict.get(curves_key, {})
    if not curves or variant not in curves:
        available = list(curves.keys()) if isinstance(curves, dict) else []
        raise KeyError(
            f"Variant '{variant}' not found in calib_data['{model_name}']['{curves_key}']. "
            f"Available: {available}"
        )

    c = curves[variant]
    prob_pred = np.asarray(c["prob_pred"], dtype=float)
    count = np.asarray(c["count"], dtype=float)

    if prob_pred.ndim != 1 or count.ndim != 1 or prob_pred.size != count.size:
        raise ValueError("Expected 1D prob_pred and count arrays of the same length.")

    total = float(count.sum())
    if normalize:
        heights = count / total if total > 0 else np.zeros_like(count, dtype=float)
        y_label = "Fraction of samples"
    else:
        heights = count
        y_label = "# samples in bin"

    # Pick color + label
    if variant == "uncalib":
        bar_color = uncalib_color
        variant_label = "Uncalibrated"
    else:
        bar_color = calibrated_color_map.get(variant, "#F14949")
        variant_label = f"Calibrated: {variant}"

    # Reasonable bar width from spacing of bin centers
    x_sorted = np.sort(prob_pred)
    if x_sorted.size >= 2:
        min_dx = float(np.min(np.diff(x_sorted)))
        width = 0.85 * min_dx
    else:
        width = 0.08  # fallback

    sns.set(style=sns_style)
    plt.figure(figsize=figsize)
    ax = plt.gca()

    ax.bar(
        prob_pred,
        heights,
        width=width,
        color=bar_color,
        alpha=bar_alpha,
        edgecolor=edgecolor,
        linewidth=linewidth if edgecolor is not None else 0.0,
        align="center",
    )

    display_model = method_alias.get(model_name, model_name)
    ax.set_xlabel("Mean predicted probability (positive class)", fontsize=font_size, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=font_size, fontweight="bold")
    ax.set_title(
        f"Prediction histogram — {display_model} ({split}, {variant_label})",
        fontsize=font_size + 2,
        fontweight="bold",
    )

    ax.tick_params(axis="both", labelsize=font_size)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")

    ax.set_xlim(0.0, 1.0)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.show()









# def plot_calibration_curves(
#     calib_data,
#     model_name: str,
#     methods=None,                 # e.g. ["platt", "beta"]
#     include_uncalibrated: bool = True,
#     figsize=(6, 6),
#     *,
#     calibration_stats=None,       # output of pooled_bootstrap_calibration_stats_by_model
#     show_alpha_beta: bool = True, # if calibration_stats provided, append α/β to labels
#     alpha_beta_decimals: int = 3, # formatting only α/β  labels
# ):
#     """
#     Plot calibration curves (with error bars) for a given model, using
#     precomputed curves in `calib_data`.

#     IMPORTANT:
#       - Train and test curves are ALWAYS plotted separately.
#       - This function assumes you've already run:
#             extract_calibration_data(...)
#             calibration_data_prep(...)
#         so that calib_data has '<split>_curves' entries.

#     NEW: Optional α/β (calibration intercept/slope) in legend labels
#     ---------------------------------------------------------------
#     If `calibration_stats` is provided (from pooled_bootstrap_calibration_stats_by_model),
#     we will append bootstrap mean±std for:
#       - alpha (intercept) and beta (slope)
#     to each curve label.

#     Mapping between curve split and stats split:
#       - curve split "train" -> stats prefix "train_oof"
#       - curve split "test"  -> stats prefix "outer_test"

#     Parameters
#     ----------
#     calib_data : dict
#         Must contain:
#             calib_data[model_name]["train_curves"][method] and ["test_curves"][method]
#         where each method entry includes:
#             prob_true, prob_pred, lower, upper

#     model_name : str
#         Model key inside calib_data.

#     methods : list[str] or None
#         Calibration methods to plot. If None, auto-discover from train_curves keys
#         excluding "uncalib".

#     include_uncalibrated : bool
#         Whether to plot uncalibrated curve ("uncalib").

#     figsize : tuple
#         Figure size for each of the train/test plots.

#     calibration_stats : dict or None
#         Output from pooled_bootstrap_calibration_stats_by_model.
#         Expected structure:
#             calibration_stats[model_name][method][...flat keys...]
#         containing:
#             train_oof_alpha_mean/std, train_oof_beta_mean/std
#             outer_test_alpha_mean/std, outer_test_beta_mean/std

#     show_alpha_beta : bool
#         If True and calibration_stats is provided, append α/β info to legend labels.

#     alpha_beta_decimals : int
#         Formatting precision for α/β in the legend only.
#     """
#     if model_name not in calib_data:
#         raise KeyError(f"Model '{model_name}' not found in calib_data.")

#     model_dict = calib_data[model_name]
#     train_curves = model_dict.get("train_curves", {})
#     test_curves  = model_dict.get("test_curves", {})

#     # Auto-discover methods from train_curves (excluding 'uncalib')
#     if methods is None:
#         discovered = [k for k in train_curves.keys() if k != "uncalib"]
#         methods = sorted(discovered)

#     # Helper: get α/β text for a given split+method
#     def _alpha_beta_suffix(split_name: str, method_key: str) -> str:
#         if (calibration_stats is None) or (not show_alpha_beta):
#             return ""

#         if model_name not in calibration_stats:
#             return ""

#         if method_key not in calibration_stats[model_name]:
#             return ""

#         # Map curve split -> stats prefix
#         stats_prefix = "train_oof" if split_name == "train" else "outer_test"

#         entry = calibration_stats[model_name][method_key]
#         a_mean_key = f"{stats_prefix}_alpha_mean"
#         a_std_key  = f"{stats_prefix}_alpha_std"
#         b_mean_key = f"{stats_prefix}_beta_mean"
#         b_std_key  = f"{stats_prefix}_beta_std"

#         # If any keys missing, skip suffix
#         if not all(k in entry for k in (a_mean_key, a_std_key, b_mean_key, b_std_key)):
#             return ""

#         a_mean = float(entry[a_mean_key])
#         a_std  = float(entry[a_std_key])
#         b_mean = float(entry[b_mean_key])
#         b_std  = float(entry[b_std_key])

#         fmt = f"{{:.{alpha_beta_decimals}f}}"
#         return f" (α={fmt.format(a_mean)}±{fmt.format(a_std)}, β={fmt.format(b_mean)}±{fmt.format(b_std)})"

#     # Helper for one split (train OR test)
#     def _plot_split(split_name: str, curves: dict):
#         if not curves:
#             print(f"No '{split_name}_curves' found for model '{model_name}'. Skipping.")
#             return

#         plt.figure(figsize=figsize)

#         # Perfect calibration line
#         plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

#         # Build ordered list of variants to plot
#         variants_to_plot = []

#         if include_uncalibrated and "uncalib" in curves:
#             variants_to_plot.append("uncalib")

#         for m in methods:
#             if m in curves:
#                 variants_to_plot.append(m)

#         if not variants_to_plot:
#             print(
#                 f"No matching variants to plot for split='{split_name}' "
#                 f"and methods={methods} (include_uncalibrated={include_uncalibrated})."
#             )
#             plt.close()
#             return

#         # Plot each variant with error bars
#         for variant in variants_to_plot:
#             c = curves[variant]
#             prob_pred = c["prob_pred"]
#             prob_true = c["prob_true"]
#             lower = c["lower"]
#             upper = c["upper"]

#             # yerr as [down, up] distances
#             yerr = [prob_true - lower, upper - prob_true]

#             base_label = "Uncalibrated" if variant == "uncalib" else f"Calibrated: {variant}"
#             label = base_label + _alpha_beta_suffix(split_name, variant)

#             plt.errorbar(
#                 prob_pred,
#                 prob_true,
#                 yerr=yerr,
#                 fmt="o-",
#                 capsize=3,
#                 label=label,
#             )

#         plt.xlabel("Predicted probability")
#         plt.ylabel("Observed frequency")
#         plt.title(f"Calibration curve – {model_name} ({split_name})")
#         plt.xlim(0.0, 1.0)
#         plt.ylim(0.0, 1.0)
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()
#         plt.show()

#     # ALWAYS plot train and test separately
#     _plot_split("train", train_curves)
#     _plot_split("test", test_curves)

# def plot_calibration_histogram(
#     calib_data,
#     model_name: str,
#     split: str = "train",        # "train" or "test"
#     variant: str = "uncalib",    # "uncalib", "platt", "beta", ...
#     normalize: bool = False,     # counts vs fraction of samples
#     figsize=(6, 3),
#     font_size: int = 12,
# ):
#     """
#     Plot a histogram of sample counts per probability bin for a given
#     model / split / variant, using the precomputed calibration bins.

#     This uses the bin centers (`prob_pred`) and bin counts (`count`)
#     stored in `calib_data` by `calibration_data_prep`.

#     Parameters
#     ----------
#     calib_data : dict
#         Nested dict as returned by extract_calibration_data and then
#         augmented by calibration_data_prep, e.g.:

#         calib_data[model_name]["train_curves"][variant] = {
#             "prob_true": ...,
#             "prob_pred": ...,
#             "lower":     ...,
#             "upper":     ...,
#             "count":     ...,
#         }

#     model_name : str
#         Model key inside calib_data, e.g. "logistic_regression".

#     split : {"train", "test"}, default="train"
#         Which split's histogram to plot.

#     variant : str, default="uncalib"
#         Which variant's bins to use:
#           - "uncalib" for uncalibrated probabilities
#           - "platt", "beta", ... for calibrated variants

#     normalize : bool, default=False
#         If False:
#           - y-axis shows raw counts (# samples per bin).
#         If True:
#           - y-axis shows fraction of samples per bin (counts / total).

#     figsize : tuple, default=(6, 3)
#         Figure size passed to plt.figure(figsize=...).

#     font_size : int, default=12
#         Base font size for labels and title.
#     """
#     if model_name not in calib_data:
#         raise KeyError(f"Model '{model_name}' not found in calib_data.")

#     model_dict = calib_data[model_name]

#     curves_key = f"{split}_curves"
#     if curves_key not in model_dict:
#         raise KeyError(
#             f"'{curves_key}' not found for model '{model_name}'. "
#             f"Did you run calibration_data_prep?"
#         )

#     curves = model_dict[curves_key]
#     if variant not in curves:
#         raise KeyError(
#             f"Variant '{variant}' not found in {curves_key} for model '{model_name}'. "
#             f"Available variants: {list(curves.keys())}"
#         )

#     c = curves[variant]
#     prob_pred = np.asarray(c["prob_pred"])
#     count = np.asarray(c["count"])

#     if normalize:
#         total = count.sum()
#         if total > 0:
#             height = count / total
#         else:
#             height = count.astype(float)
#     else:
#         height = count

#     # Sort by bin center just in case
#     order = np.argsort(prob_pred)
#     prob_pred = prob_pred[order]
#     height = height[order]

#     # Approximate bar width: span / number of bins (with a safety fallback)
#     if len(prob_pred) > 1:
#         span = prob_pred.max() - prob_pred.min()
#         width = span / len(prob_pred) if span > 0 else 1.0 / len(prob_pred)
#     else:
#         width = 0.1

#     width *= 0.9  # slight shrink so bars don't touch

#     plt.figure(figsize=figsize)
#     plt.bar(prob_pred, height, width=width, align="center", edgecolor="black")

#     plt.xlabel("Predicted probability (bin centers)", fontsize=font_size)
#     if normalize:
#         plt.ylabel("Fraction of samples", fontsize=font_size)
#     else:
#         plt.ylabel("# samples in bin", fontsize=font_size)

#     label_variant = "Uncalibrated" if variant == "uncalib" else f"Calibrated: {variant}"
#     plt.title(
#         f"Prediction histogram – {model_name} ({split}, {label_variant})",
#         fontsize=font_size + 1,
#     )

#     plt.xlim(0.0, 1.0)
#     plt.grid(axis="y", alpha=0.3)
#     plt.tight_layout()
#     plt.show()






# --------------------------------------------------------------
# Permutation Importance
# --------------------------------------------------------------
def compute_permutation_importance_nested_cv(
    all_results: Dict[str, List[Dict[str, Any]]],
    X: np.ndarray,
    y: np.ndarray,
    scoring: str,
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
    show_progress: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute permutation feature importances for nested-CV results (outer-fold test sets).

    This function iterates over each model and each outer fold stored in `all_results`.
    For every outer fold, it uses the already-fitted `final_model` (trained on that
    fold's outer-train split) and computes permutation importance on the corresponding
    OUTER TEST split (indexed by `outer_test_idx`).

    The computation uses `sklearn.inspection.permutation_importance`, which measures
    feature importance by repeatedly shuffling one feature column at a time and
    observing the change in the chosen `scoring` metric.

    Per-fold stored outputs
    -----------------------
    For each fold dict `r` in `all_results[model_name]`, this function adds:

      - r["permutation_importances"]              : np.ndarray, shape (n_features, n_repeats)
          Raw importance values from sklearn, where each column corresponds to one
          shuffle repeat and each row corresponds to one feature.

      - r["permutation_importance_scoring"]       : str
          The scoring metric name used during permutation importance.

      - r["permutation_importance_n_repeats"]     : int
          Number of shuffle repeats used per feature.

      - r["permutation_importance_random_state"]  : int
          Random seed used for permutation shuffling.

      - r["permutation_importance_n_jobs"]        : int
          Number of parallel jobs used by `permutation_importance`.

    Parameters
    ----------
    all_results:
        Nested-CV results dict structured as:
          {model_name: [fold_dict_0, fold_dict_1, ...]}
        Each fold dict must contain:
          - "final_model"     : fitted estimator
          - "outer_test_idx"  : indices into X/y for the outer test split

    X:
        Full feature matrix used for nested CV, shape (n_samples, n_features).

    y:
        Full label vector used for nested CV, shape (n_samples,).

    scoring:
        Scoring metric name to pass to `permutation_importance`, e.g. "roc_auc",
        "average_precision", etc. This controls how performance degradation is measured.

    n_repeats:
        Number of shuffles performed per feature (more repeats = more stable estimates,
        but higher runtime).

    random_state:
        Random seed for reproducibility of shuffling.

    n_jobs:
        Number of parallel jobs used by `permutation_importance` (-1 uses all cores).

    show_progress:
        If True, shows a tqdm progress bar over folds for each model.

    Returns
    -------
    all_results:
        Same dict as input, mutated in-place with permutation importance outputs stored
        in each fold dict.
    """
    # Loop over each model in the nested CV results.
    for model_name, folds in all_results.items():
        # Optionally show a fold progress bar per model.
        fold_iter = (
            trange(len(folds), desc=f"{model_name}: perm imp folds", leave=False)
            if show_progress
            else range(len(folds))
        )

        # Loop over outer folds for this model.
        for i in fold_iter:
            # Get fold dict.
            r = folds[i]

            # Extract outer test indices for this fold.
            outer_test_idx = r["outer_test_idx"]

            # Slice X/y to the outer test split.
            X_test = X[outer_test_idx]
            y_test = y[outer_test_idx]

            # Use the already-fitted model stored in the fold dict.
            final_model = r["final_model"]

            # Compute permutation importance on the outer test split.
            result = permutation_importance(
                final_model,
                X_test,
                y_test,
                n_repeats=n_repeats,
                random_state=random_state,
                scoring=scoring,
                n_jobs=n_jobs,
            )

            # Store raw importances: shape (n_features, n_repeats).
            r["permutation_importances"] = result.importances

            # Store metadata so you know how it was computed.
            r["permutation_importance_scoring"] = scoring
            r["permutation_importance_n_repeats"] = n_repeats
            r["permutation_importance_random_state"] = random_state
            r["permutation_importance_n_jobs"] = n_jobs

    return all_results

def combine_permutation_importances_nested_cv(
    all_results: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, np.ndarray]:
    """
    Combine permutation importances from all outer folds into a single table
    per model.

    For each model in all_results, this function:

      - Collects r["permutation_importances"] from every outer fold r
        where that key exists.
      - Each r["permutation_importances"] is assumed to have shape:
            (n_features, n_repeats_fold)
      - Concatenates them along axis=1 to form:
            combined_importances.shape == (n_features, sum_f n_repeats_f)

    Returns
    -------
    combined_importances_by_model : dict
        {
          "logistic_regression": np.ndarray of shape (n_features, total_repeats),
          "random_forest":       np.ndarray of shape (n_features, total_repeats),
          ...
        }

        Where total_repeats is the sum of n_repeats across all folds that
        have permutation_importances for that model.

    Notes
    -----
    - This function assumes that the number of features (n_features) is the
      same across folds for a given model.
    - You can then compute per-feature summaries easily, e.g.:

          table = combined_importances_by_model["logistic_regression"]
          mean_per_feature = table.mean(axis=1)
          median_per_feature = np.median(table, axis=1)
          std_per_feature = table.std(axis=1)

      or use the full table for boxplots / violin plots.
    """
    combined_importances_by_model: Dict[str, np.ndarray] = {}

    for model_name, folds in all_results.items():
        per_model_importances = []

        for r in folds:
            if "permutation_importances" not in r:
                continue

            imps = r["permutation_importances"]
            imps = np.asarray(imps)

            per_model_importances.append(imps)

        # If no permutation_importances for this model, skip it
        if not per_model_importances:
            continue

        # Sanity check: all have same n_features
        n_features = per_model_importances[0].shape[0]
        for arr in per_model_importances:
            if arr.shape[0] != n_features:
                raise ValueError(
                    f"Inconsistent n_features for model '{model_name}': "
                    f"got shapes {[a.shape for a in per_model_importances]}"
                )

        # Concatenate along repeats axis (axis=1)
        combined = np.concatenate(per_model_importances, axis=1)
        combined_importances_by_model[model_name] = combined

    return combined_importances_by_model



def run_permutation_importance_pipeline(
    all_results: Dict[str, List[Dict[str, Any]]],
    bundle: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    x_key: str = "combined_X_raw",
    y_key: str = "combined_y",
    scoring: str,
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[
    Dict[str, List[Dict[str, Any]]],
    Dict[str, np.ndarray],
    Dict[str, List[str]],
]:
    """
    Run permutation feature-importance for nested-CV results, supporting per-model feature sets.

    This pipeline mirrors the feature-selection logic used during training:
      - X/y are pulled from `bundle` using `x_key` and `y_key` (e.g., "combined_X_raw"/"combined_y").
      - For each model in `all_results`, X is sliced ONCE using that model's config:
          * cfg["models"][model_name]["feature_names"] (exact-name selection), OR
          * cfg["models"][model_name]["n_features"]   (prefix/top-K selection), OR
          * neither (use all features).
      - The existing nested-CV permutation-importance routines are then applied per model:
          1) compute_permutation_importance_nested_cv(...) on outer test splits
          2) combine_permutation_importances_nested_cv(...) to aggregate fold matrices

    Parameters
    ----------
    all_results:
        Nested-CV results dict structured as:
            {model_name: [fold_dict_0, fold_dict_1, ...]}
        Each fold dict must include at minimum:
            - "final_model"    : fitted estimator for that outer fold
            - "outer_test_idx" : indices into X/y (as selected by x_key/y_key) for outer test split
        This object is mutated in-place to store per-fold permutation importances and metadata.

    bundle:
        Dataset dictionary containing:
            - bundle[x_key]         : 2D numpy array of shape (n_samples, n_features_full)
            - bundle[y_key]         : 1D numpy array of shape (n_samples,)
            - bundle["feature_names"]: list[str] of length n_features_full (column names for bundle[x_key])

    cfg:
        Configuration dictionary that must contain cfg["models"][model_name] for every model in all_results.
        Supported per-model feature-selection keys:
            - "feature_names": Optional[list[str]]
            - "n_features"   : Optional[int]
            - "feature_strict": bool (default True) passed to prepare_training_bundle
        Only one of ("feature_names", "n_features") may be set per model.

    x_key, y_key:
        Keys selecting which dataset level to use from `bundle` (e.g., "combined_X_raw"/"combined_y"
        for group-aggregated datasets, or "X_raw"/"y" for sample-level datasets).
        The `outer_test_idx` stored in fold dicts must refer to indices in these arrays.

    scoring:
        Scoring metric name passed to sklearn.inspection.permutation_importance,
        e.g. "roc_auc" or "average_precision". Measures performance degradation under feature shuffling.

    n_repeats:
        Number of shuffles per feature per fold (higher = more stable, slower runtime).

    random_state:
        Random seed controlling the shuffling for reproducibility.

    n_jobs:
        Number of parallel jobs used by permutation_importance (-1 uses all cores).

    Returns
    -------
    all_results:
        Same dict as input, mutated in-place with per-fold permutation importances stored under keys like:
            - "permutation_importances" : np.ndarray of shape (n_features_model, n_repeats)
        (Exact key names depend on compute_permutation_importance_nested_cv.)

    combined_importances:
        Dict mapping model_name -> np.ndarray of shape (n_features_model, total_repeats),
        where total_repeats is the sum of repeats across all folds for that model.

    model_feature_names:
        Dict mapping model_name -> list[str] of feature names actually used for that model.
        This is intended to be passed to plot_permutation_importances_barplot(feature_names=...).

    Notes
    -----
    - Importance is computed on OUTER TEST splits only (consistent with nested-CV generalization evaluation).
    - This function assumes `compute_permutation_importance_nested_cv` and
      `combine_permutation_importances_nested_cv` accept and operate on the provided X/y and outer_test_idx.
    """

    # ---- pull X/y from the correct dataset level ----
    if x_key not in bundle:
        raise KeyError(f"bundle missing x_key='{x_key}'")
    if y_key not in bundle:
        raise KeyError(f"bundle missing y_key='{y_key}'")
    if "feature_names" not in bundle:
        raise KeyError("bundle must contain 'feature_names' for feature selection")

    X_full = np.asarray(bundle[x_key])
    y = np.asarray(bundle[y_key])
    feature_names_full = list(bundle["feature_names"])

    if X_full.ndim != 2:
        raise ValueError(f"bundle[{x_key}] must be 2D, got shape {X_full.shape}")
    if y.ndim != 1:
        raise ValueError(f"bundle[{y_key}] must be 1D, got shape {y.shape}")
    if X_full.shape[0] != len(y):
        raise ValueError(
            f"X/y mismatch for keys ({x_key}, {y_key}): "
            f"X rows={X_full.shape[0]} vs len(y)={len(y)}"
        )
    if X_full.shape[1] != len(feature_names_full):
        raise ValueError(
            f"Mismatch: X has {X_full.shape[1]} cols but feature_names has {len(feature_names_full)}"
        )

    # Returned so plotting can label each model correctly (since each model may use different features)
    model_feature_names: Dict[str, List[str]] = {}

    # ---- pipeline progress ----
    steps = tqdm(total=2, desc="Permutation importance pipeline", unit="step")

    # ---- Step 1: compute permutation importance (per model) ----
    steps.set_description("Permutation importance: compute per model")

    for model_name in list(all_results.keys()):
        if model_name not in cfg["models"]:
            raise KeyError(f"Model '{model_name}' not found in cfg['models'].")

        m_cfg = cfg["models"][model_name]

        # same knobs as training
        keep_features = m_cfg.get("feature_names", None)  # list[str] | None
        n_features_model = m_cfg.get("n_features", None)  # int | None

        if keep_features is not None and n_features_model is not None:
            raise ValueError(
                f"{model_name}: set only one of 'feature_names' or 'n_features' (or neither)."
            )

        # Build the mini-bundle expected by prepare_training_bundle
        view_bundle = {"X_raw": X_full, "feature_names": feature_names_full}

        # Slice ONCE per model (mimics training exactly)
        if keep_features is not None or n_features_model is not None:
            mb = prepare_training_bundle(
                view_bundle,
                n_features=n_features_model,
                keep_features=keep_features,
                strict=m_cfg.get("feature_strict", True),
                dedupe=True,
                copy_bundle=True,
            )
        else:
            mb = view_bundle

        X_model = np.asarray(mb["X_raw"])
        model_feature_names[model_name] = list(mb["feature_names"])

        # (Optional) stash for traceability in fold dicts
        for r in all_results[model_name]:
            r["pi_feature_names"] = model_feature_names[model_name]
            r["pi_x_key"] = x_key
            r["pi_y_key"] = y_key

        # Run your existing compute function on ONLY this model,
        # using X_model so dimensions match what the model was trained on.
        sub_results = {model_name: all_results[model_name]}
        sub_results = compute_permutation_importance_nested_cv(
            all_results=sub_results,
            X=X_model,
            y=y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=n_jobs,
            show_progress=True,
        )

        # Write back (compute_* mutates in-place, but explicit assignment is clearer)
        all_results[model_name] = sub_results[model_name]

    steps.update(1)

    # ---- Step 2: combine (unchanged) ----
    steps.set_description("Permutation importance: combine folds")
    combined_importances = combine_permutation_importances_nested_cv(all_results)
    steps.update(1)

    steps.close()

    return all_results, combined_importances, model_feature_names


def plot_permutation_importances_barplot(
    combined_importances: Dict[str, np.ndarray],
    model_name: Union[str, Sequence[str], None] = None,
    feature_names: Optional[Union[List[str], Dict[str, List[str]]]] = None,
    top_n: Optional[int] = None,
    scoring: str = "average_precision",
    figsize: tuple[float, float] = (10, 6),
    font_size: int = 12,
    color: str = "#69b3a2",
    x_tick_rotation: float = 90,
    method_alias: Optional[Mapping[str, str]] = None,
) -> None:
    """
    Plot permutation feature importances as barplots for one or more models.

    The input `combined_importances` is typically produced by
    combine_permutation_importances_nested_cv(all_results) (or by run_permutation_importance_pipeline).

    For each selected model, this function:
      - Computes mean importance per feature across all concatenated repeats
      - Sorts features by mean importance (descending)
      - Optionally keeps only the top_n features
      - Builds a tidy DataFrame with one row per (feature, repeat) and plots a barplot
        with error bars representing standard deviation across repeats.

    Parameters
    ----------
    combined_importances:
        Dict mapping model_name -> np.ndarray of shape (n_features, total_repeats).
        Each entry corresponds to a single model's permutation importances, concatenated across folds.

    model_name:
        Which model(s) to plot:
          - None: plot all models found in combined_importances
          - str: plot that single model
          - sequence[str]: plot those models in the given order

    feature_names:
        Feature name labels to use on the x-axis:
          - None: labels default to "f0", "f1", ... per model
          - list[str]: a single shared list of names (must match n_features for the model being plotted)
          - dict[str, list[str]]: per-model feature names (recommended when models use different feature sets)

    top_n:
        If provided, plot only the top_n features by mean importance for each model.

    scoring:
        Label used in the y-axis text (e.g., "AUROC" or "AUPRC"). This does not change computation
        (computation already happened upstream).

    figsize:
        Matplotlib figure size.

    font_size:
        Base font size for plot text.

    color:
        Bar color.

    x_tick_rotation:
        Rotation (degrees) for x-axis tick labels.

    method_alias:
        Optional mapping from model_name (dict key) to a nicer display name for the title.

    Raises
    ------
    KeyError:
        If requested model_name(s) are not present in combined_importances, or if feature_names is a dict
        missing an entry for a selected model.

    ValueError:
        If feature_names length does not match the number of features for a model,
        or if combined_importances arrays are not 2D.

    Notes
    -----
    - If you provide per-model feature names, pass the dict returned by
      run_permutation_importance_pipeline(...)[2] (model_feature_names).
    - Error bars show standard deviation across repeats (concatenated across folds).
    """
    if method_alias is None:
        method_alias = {}

    # choose models
    if model_name is None:
        selected = list(combined_importances.keys())
    elif isinstance(model_name, str):
        selected = [model_name]
    else:
        selected = list(model_name)

    missing = [m for m in selected if m not in combined_importances]
    if missing:
        raise KeyError(
            f"Model(s) not found in combined_importances: {missing}. "
            f"Available: {list(combined_importances.keys())}"
        )

    for m in selected:
        imp = np.asarray(combined_importances[m])  # (n_features, total_repeats)
        if imp.ndim != 2:
            raise ValueError(f"combined_importances['{m}'] must be 2D, got shape {imp.shape}")
        n_features, _total_repeats = imp.shape

        # Feature names (supports per-model dict)
        if feature_names is None:
            fnames = [f"f{i}" for i in range(n_features)]
        elif isinstance(feature_names, dict):
            if m not in feature_names:
                raise KeyError(
                    f"feature_names dict missing key '{m}'. Available: {list(feature_names.keys())}"
                )
            fnames = list(feature_names[m])
            if len(fnames) != n_features:
                raise ValueError(
                    f"len(feature_names['{m}'])={len(fnames)} does not match "
                    f"n_features={n_features} for model '{m}'."
                )
        else:
            fnames = list(feature_names)
            if len(fnames) != n_features:
                raise ValueError(
                    f"len(feature_names)={len(fnames)} does not match "
                    f"n_features={n_features} for model '{m}'."
                )

        # Order features by mean importance (descending)
        means = np.mean(imp, axis=1)
        feature_indices = np.argsort(means)[::-1]
        if top_n is not None:
            top_n_int = int(top_n)
            if top_n_int <= 0:
                raise ValueError("top_n must be > 0 if provided")
            feature_indices = feature_indices[:top_n_int]

        # Build tidy DataFrame: one row per (feature, repeat)
        rows: List[Dict[str, object]] = []
        for idx in feature_indices:
            fname = fnames[idx]
            for val in imp[idx, :]:
                rows.append({"feature": fname, "importance": float(val)})

        df = pd.DataFrame(rows)
        ordered_features = [fnames[i] for i in feature_indices]

        sns.set(style="whitegrid")
        plt.figure(figsize=figsize)
        ax = sns.barplot(
            data=df,
            x="feature",
            y="importance",
            order=ordered_features,
            estimator=np.mean,
            errorbar=("sd"),
            color=color,
            saturation=1,
        )

        y_label = f"Permutation impact on {scoring}"
        display = method_alias.get(m, m)
        title = f"Permutation feature importance (model: {display}, test set)"

        ax.set_xlabel("Feature", fontsize=font_size, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=font_size, fontweight="bold")
        ax.set_title(title, fontsize=font_size + 2, fontweight="bold")

        ax.tick_params(axis="both", labelsize=font_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        ax.tick_params(axis="x", labelrotation=x_tick_rotation)

        plt.tight_layout()
        plt.show()




# def run_permutation_importance_pipeline(
#     all_results: Dict[str, List[Dict[str, Any]]],
#     X: np.ndarray,
#     y: np.ndarray,
#     *,
#     scoring: str,
#     n_repeats: int = 10,
#     random_state: int = 42,
#     n_jobs: int = -1,
# ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, np.ndarray]]:
#     """
#     Run the full permutation-importance workflow as a single pipeline with progress reporting.

#     This pipeline wraps two steps into one call:

#       (1) compute_permutation_importance_nested_cv:
#           Computes permutation feature importance on the OUTER TEST split for each
#           outer fold using the already-fitted `final_model` stored in `all_results`.
#           Results are stored per fold as:
#             - r["permutation_importances"] : np.ndarray (n_features, n_repeats)
#           along with metadata (scoring, n_repeats, random_state, n_jobs).

#       (2) combine_permutation_importances_nested_cv:
#           Combines the fold-level importance matrices into a single table per model
#           by concatenating repeats across folds:
#             combined_importances[model_name].shape == (n_features, total_repeats)

#     A top-level tqdm progress bar is shown to indicate which pipeline stage is running.

#     Parameters
#     ----------
#     all_results:
#         Nested-CV results dict structured as:
#           {model_name: [fold_dict_0, fold_dict_1, ...]}
#         Each fold dict must include:
#           - "final_model"     : fitted estimator for that outer fold
#           - "outer_test_idx"  : indices into X/y for the outer test split
#         This object is mutated in-place by step (1) to store permutation importances.

#     X:
#         Full feature matrix used for nested CV, shape (n_samples, n_features).

#     y:
#         Full label vector used for nested CV, shape (n_samples,).

#     scoring:
#         Scoring metric name passed to `sklearn.inspection.permutation_importance`,
#         e.g. "roc_auc" or "average_precision". This controls how performance
#         degradation is measured when features are shuffled.

#     n_repeats:
#         Number of shuffles per feature per fold (higher = more stable estimates,
#         but slower runtime).

#     random_state:
#         Random seed controlling the shuffling for reproducibility.

#     n_jobs:
#         Number of parallel jobs used by permutation_importance (-1 uses all cores).

#     Returns
#     -------
#     all_results:
#         Same dict as input, mutated in-place with per-fold permutation importances
#         stored under keys like "permutation_importances".

#     combined_importances:
#         Dict mapping model_name -> np.ndarray of shape (n_features, total_repeats),
#         where total_repeats is the sum of repeats across all folds.
#         This output is compatible with `plot_permutation_importances_barplot`.

#     Notes
#     -----
#     - This pipeline computes importance on OUTER TEST splits only (consistent with
#       evaluating generalization in nested CV).
#     - If you later want fold-weighted summaries (equal weight per fold), you can
#       compute per-fold means before combining; this pipeline currently concatenates
#       all repeats across folds.
#     """
#     # Create a simple 2-step progress bar for the pipeline stages.
#     steps = tqdm(total=2, desc="Permutation importance pipeline", unit="step")

#     # ---- Step 1: compute permutation importances per outer fold (outer test split) ----
#     steps.set_description("Permutation importance: compute per fold")
#     all_results = compute_permutation_importance_nested_cv(
#         all_results=all_results,
#         X=X,
#         y=y,
#         scoring=scoring,
#         n_repeats=n_repeats,
#         random_state=random_state,
#         n_jobs=n_jobs,
#         show_progress=True,
#     )
#     steps.update(1)

#     # ---- Step 2: combine fold-level matrices into one table per model ----
#     steps.set_description("Permutation importance: combine folds")
#     combined_importances = combine_permutation_importances_nested_cv(all_results)
#     steps.update(1)

#     # Close the progress bar cleanly.
#     steps.close()

#     # Return both the enriched all_results and the combined importances dict.
#     return all_results, combined_importances

# def plot_permutation_importances_barplot(
#     combined_importances: Dict[str, np.ndarray],
#     model_name: str | Sequence[str] | None = None,
#     feature_names: Optional[List[str]] = None,
#     top_n: Optional[int] = None,
#     scoring: str = "average_precision",
#     figsize=(10, 6),
#     font_size: int = 12,
#     color: str = "#69b3a2",
#     x_tick_rotation: float = 90,
#     method_alias: Mapping[str, str] | None = None,  # NEW
# ):
#     """
#     Barplot of permutation importances for one or more models.

#     Parameters
#     ----------
#     combined_importances : dict
#         Output of combine_permutation_importances_nested_cv(all_results), e.g.:
#         {
#           "logistic_regression": np.ndarray of shape (n_features, total_repeats),
#           "random_forest":       np.ndarray of shape (n_features, total_repeats),
#         }

#     model_name : str | sequence[str] | None
#         Which model(s) to plot. None = all models in combined_importances.

#     feature_names : list[str] or None, default=None
#         Names of features. If None, features will be labeled as "f0", "f1", ...

#     top_n : int or None, default=None
#         If provided, only the top_n features (by mean importance) are plotted.

#     scoring : str, default="average_precision"
#         Name of the scoring metric used in permutation_importance, for labeling.

#     figsize : tuple
#         Size of the matplotlib figure.

#     font_size : int
#         Base font size for labels and ticks.

#     color : str, default="#69b3a2"
#         Color for the bars.

#     x_tick_rotation : float, default=90
#         Rotation angle (in degrees) for x-axis tick labels.

#     method_alias : dict or None
#         Optional mapping model_key -> display name (used in the plot title only).
#     """
#     if method_alias is None:
#         method_alias = {}

#     # choose models
#     if model_name is None:
#         selected = list(combined_importances.keys())
#     elif isinstance(model_name, str):
#         selected = [model_name]
#     else:
#         selected = list(model_name)

#     missing = [m for m in selected if m not in combined_importances]
#     if missing:
#         raise KeyError(
#             f"Model(s) not found in combined_importances: {missing}. "
#             f"Available: {list(combined_importances.keys())}"
#         )

#     for m in selected:
#         imp = np.asarray(combined_importances[m])  # (n_features, total_repeats)
#         n_features, total_repeats = imp.shape

#         # Feature names
#         if feature_names is None:
#             fnames = [f"f{i}" for i in range(n_features)]
#         else:
#             if len(feature_names) != n_features:
#                 raise ValueError(
#                     f"len(feature_names)={len(feature_names)} does not match "
#                     f"n_features={n_features} for model '{m}'."
#                 )
#             fnames = feature_names

#         # Order features by mean importance (descending)
#         means = np.mean(imp, axis=1)
#         feature_indices = np.argsort(means)[::-1]
#         if top_n is not None:
#             feature_indices = feature_indices[:top_n]

#         # Build tidy DataFrame: one row per (feature, repeat)
#         rows = []
#         for idx in feature_indices:
#             fname = fnames[idx]
#             for val in imp[idx, :]:
#                 rows.append({"feature": fname, "importance": val})

#         df = pd.DataFrame(rows)
#         ordered_features = [fnames[i] for i in feature_indices]

#         sns.set(style="whitegrid")
#         plt.figure(figsize=figsize)
#         ax = sns.barplot(
#             data=df,
#             x="feature",
#             y="importance",
#             order=ordered_features,
#             estimator=np.mean,
#             errorbar=("sd"),
#             color=color,
#         )

#         y_label = f"Permutation impact on {scoring}"
#         display = method_alias.get(m, m)
#         title = f"Permutation feature importance (model: {display}, test set)"

#         ax.set_xlabel("Feature", fontsize=font_size, fontweight="bold")
#         ax.set_ylabel(y_label, fontsize=font_size, fontweight="bold")
#         ax.set_title(title, fontsize=font_size + 2, fontweight="bold")

#         ax.tick_params(axis="both", labelsize=font_size)
#         for label in ax.get_xticklabels() + ax.get_yticklabels():
#             label.set_fontweight("bold")

#         ax.tick_params(axis="x", labelrotation=x_tick_rotation)

#         plt.tight_layout()
#         plt.show()

# def compute_permutation_importance_nested_cv(
#     all_results: Dict[str, List[Dict[str, Any]]],
#     X: np.ndarray,
#     y: np.ndarray,
#     scoring: Optional[str] = None,
#     n_repeats: int = 10,
#     random_state: int = 42,
#     n_jobs: int = -1,
# ) -> Dict[str, List[Dict[str, Any]]]:
#     """
#     For each (model, trial, outer_fold) entry in all_results, compute
#     permutation feature importance on the corresponding OUTER TEST set,
#     using the already-fitted final_model for that fold.

#     Stores the full permutation importance matrix per fold:

#         - "permutation_importances": np.ndarray of shape (n_features, n_repeats)

#     Also stores some metadata:

#         - "permutation_importance_scoring": str
#         - "permutation_importance_n_repeats": int
#         - "permutation_importance_n_jobs": int

#     Parameters
#     ----------
#     all_results : dict
#         Nested CV results as produced by run_nested_cv_for_model, e.g.:
#         {
#           "logistic_regression": [ { ... per outer fold ... }, ... ],
#           "random_forest": [ ... ],
#         }

#     X : np.ndarray
#         Full feature matrix used to generate all_results, shape (n_samples, n_features).

#     y : np.ndarray
#         Full label vector, shape (n_samples,).

#     scoring : str or None, default=None
#         Scoring function name to pass to permutation_importance.
#         If None, this will default to config["metric"]["scoring"].

#     n_repeats : int, default=10
#         Number of random shuffles for each feature.

#     random_state : int, default=42
#         Random seed for reproducibility.

#     n_jobs : int, default=-1
#         Number of jobs to run in parallel for permutation_importance.
#         -1 means using all processors.

#     Returns
#     -------
#     all_results : dict
#         Same structure as input, mutated in-place with additional keys per fold:
#         - "permutation_importances"
#         - "permutation_importance_scoring"
#         - "permutation_importance_n_repeats"
#         - "permutation_importance_n_jobs"
#     """
#     # Default scoring: use the same metric as your nested CV config
#     if scoring is None:
#         scoring = config["metric"]["scoring"]  # expects your global `config`

#     for model_name, folds in all_results.items():
#         for r in folds:
#             outer_test_idx = r["outer_test_idx"]

#             # Rebuild test subset for this outer fold
#             X_test = X[outer_test_idx]
#             y_test = y[outer_test_idx]

#             # Use the already-fitted final model for this fold
#             final_model = r["final_model"]

#             # Compute permutation importance on the OUTER TEST set
#             result = permutation_importance(
#                 final_model,
#                 X_test,
#                 y_test,
#                 n_repeats=n_repeats,
#                 random_state=random_state,
#                 scoring=scoring,
#                 n_jobs=n_jobs,
#             )

#             # Store full importance scores (n_features, n_repeats)
#             r["permutation_importances"] = result.importances
#             r["permutation_importance_scoring"] = scoring
#             r["permutation_importance_n_repeats"] = n_repeats

#     return all_results



# --------------------------------------------------------------
# Partial Dependence Display
# --------------------------------------------------------------
def compute_pdp_nested_cv(
    all_results: Dict[str, List[Dict[str, Any]]],
    X: np.ndarray,
    y: np.ndarray,
    grid_resolution: int = 100,
    percentiles: Tuple[float, float] = (0.05, 0.95),
    data_source: str = "test",  # "train", "test", or "both"
    centered: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute Partial Dependence (PDP) curves for each model and each *outer fold*
    in a nested-CV results structure, using the already-fitted `final_model`
    stored per fold.

    This function is designed for nested cross-validation outputs where each
    outer fold dict contains:
      - r["final_model"]       : a fitted sklearn estimator for that outer fold
      - r["outer_train_idx"]   : indices for the outer-train split
      - r["outer_test_idx"]    : indices for the outer-test split

    PDPs are computed using `sklearn.inspection.PartialDependenceDisplay.from_estimator`
    with `kind="average"` (no ICE curves).

    IMPORTANT ABOUT LOW-CARDINALITY FEATURES
    ---------------------------------------
    For binary / low-unique-value features, scikit-learn may return fewer grid
    points than `grid_resolution`. Therefore, different features can produce
    different-length PDP grids. To keep the stored results rectangular
    (2D arrays), this function pads shorter per-feature PDP arrays with `np.nan`.

    Stored outputs (per fold dict `r`)
    ----------------------------------
    If `data_source` includes "train":
      - r["pdp_grid_values_train"]    : np.ndarray, shape (n_features, max_grid_len)
      - r["pdp_average_values_train"] : np.ndarray, shape (n_features, max_grid_len)

    If `data_source` includes "test":
      - r["pdp_grid_values_test"]     : np.ndarray, shape (n_features, max_grid_len)
      - r["pdp_average_values_test"]  : np.ndarray, shape (n_features, max_grid_len)

    Note: `max_grid_len` is the maximum PDP grid length across features for that
    split/fold; rows shorter than `max_grid_len` are padded with `np.nan`.

    Parameters
    ----------
    all_results:
        Dict mapping model_name -> list of fold dicts (outer folds). Each fold dict
        must contain `final_model`, `outer_train_idx`, `outer_test_idx`.

    X:
        Full feature matrix used for nested CV, shape (n_samples, n_features).

    y:
        Label vector (unused here; included for API symmetry), shape (n_samples,).

    grid_resolution:
        Requested number of grid points for continuous features.

    percentiles:
        Percentile range used by sklearn to define the PDP grid (trims extremes).

    data_source:
        Which outer split to compute PDP on: "train", "test", or "both".

    centered:
        Passed to PartialDependenceDisplay.from_estimator; if True, centers PDP.

    Returns
    -------
    all_results:
        Same object as input, mutated in-place with PDP arrays stored per fold.
    """

    # Define which values are allowed for `data_source`.
    allowed_sources = {"train", "test", "both"}

    # Validate that `data_source` is one of the allowed options.
    if data_source not in allowed_sources:
        raise ValueError(
            f"Invalid data_source='{data_source}'. "
            f"Expected one of {allowed_sources}."
        )

    def _compute_pdp_for_split(final_model, X_split):
        """Compute PDP grids + averages for each feature on a given split, padded with NaNs."""

        # Get number of features from the split matrix.
        n_features = X_split.shape[1]

        # Build a list of integer feature indices: [0, 1, ..., n_features-1].
        feature_indices = list(range(n_features))

        # Collect per-feature PDP x-grids (each is 1D, and lengths may differ).
        pdp_grids_list = []

        # Collect per-feature PDP average values (aligned to each feature's grid).
        pdp_values_list = []

        # Loop over features and compute PDP one feature at a time.
        for idx in feature_indices:
            # Create a "throwaway" figure/axes so sklearn can draw without displaying.
            fig, ax = plt.subplots(figsize=(0, 0))

            # Close the figure immediately so nothing shows up in notebooks.
            plt.close(fig)

            # Compute the PDP for this single feature index.
            display = PartialDependenceDisplay.from_estimator(
                final_model,                 # fitted model for this fold
                X_split,                     # data split to compute PDP over
                [idx],                       # compute PDP for one feature
                kind="average",              # average PDP (no ICE curves)
                ax=ax,                       # draw on the hidden axis
                grid_resolution=grid_resolution,  # requested grid resolution
                percentiles=percentiles,     # trim feature range for grid
                centered=centered,           # optionally center the PDP
            )

            # Extract the PDP result for the single requested feature.
            pd_result = display.pd_results[0]

            # Extract the feature grid values (x-axis) and flatten to 1D.
            grid_values = np.ravel(pd_result["grid_values"]).astype(float)

            # Extract the average partial dependence values (y-axis) and flatten to 1D.
            average_values = np.ravel(pd_result["average"]).astype(float)

            # Store the per-feature grid (may be shorter than grid_resolution).
            pdp_grids_list.append(grid_values)

            # Store the per-feature PDP values (same length as grid_values).
            pdp_values_list.append(average_values)

        # Find the maximum PDP grid length across features (for padding).
        max_len = max(len(g) for g in pdp_grids_list)

        # Create a 2D array for grids, padded with NaNs to max_len.
        pdp_grid_array = np.full((len(pdp_grids_list), max_len), np.nan, dtype=float)

        # Create a 2D array for PDP values, padded with NaNs to max_len.
        pdp_value_array = np.full((len(pdp_values_list), max_len), np.nan, dtype=float)

        # Fill each feature row with its real values, leaving the rest as NaN.
        for i, (g, v) in enumerate(zip(pdp_grids_list, pdp_values_list)):
            # Copy grid values into the left part of the row.
            pdp_grid_array[i, : len(g)] = g

            # Copy PDP values into the left part of the row.
            pdp_value_array[i, : len(v)] = v

        # Return padded 2D arrays: shape (n_features, max_len).
        return pdp_grid_array, pdp_value_array

    # Loop over each model entry in the nested-CV results.
    for model_name, folds in all_results.items():
        # Print which model we are processing (helpful for long runs).
        print(f"\n=== Computing PDP for model: {model_name} ===")

        # Loop over outer folds (with a progress bar).
        for i in trange(len(folds), desc=f"{model_name} folds", leave=False):
            # Grab the fold dict for this outer fold.
            r = folds[i]

            # Pull out the already-fitted model for this fold.
            final_model = r["final_model"]

            # Get the indices for the outer-train and outer-test splits.
            outer_train_idx = r["outer_train_idx"]
            outer_test_idx = r["outer_test_idx"]

            # If requested, compute PDP on the outer TRAIN split.
            if data_source in ("train", "both"):
                # Slice the full X to get outer-train data.
                X_train = X[outer_train_idx]

                # Compute padded PDP arrays for this split.
                pdp_grid_train, pdp_avg_train = _compute_pdp_for_split(final_model, X_train)

                # Store the padded PDP grids back into the fold dict.
                r["pdp_grid_values_train"] = pdp_grid_train

                # Store the padded PDP values back into the fold dict.
                r["pdp_average_values_train"] = pdp_avg_train

            # If requested, compute PDP on the outer TEST split.
            if data_source in ("test", "both"):
                # Slice the full X to get outer-test data.
                X_test = X[outer_test_idx]

                # Compute padded PDP arrays for this split.
                pdp_grid_test, pdp_avg_test = _compute_pdp_for_split(final_model, X_test)

                # Store the padded PDP grids back into the fold dict.
                r["pdp_grid_values_test"] = pdp_grid_test

                # Store the padded PDP values back into the fold dict.
                r["pdp_average_values_test"] = pdp_avg_test

    # Return the same nested results dict (mutated in-place).
    return all_results

def aggregate_pdp_nested_cv(
    all_results: Dict[str, List[Dict[str, Any]]],
    source: str = "test",  # "train" or "test"
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate (fold-level) PDP results from `compute_pdp_nested_cv` into a single
    long-format pandas DataFrame per model.

    This function expects that `compute_pdp_nested_cv` has already stored
    PDP arrays inside each fold dict `r`, under keys like:
      - r["pdp_grid_values_test"]     and r["pdp_average_values_test"]
      - r["pdp_grid_values_train"]    and r["pdp_average_values_train"]

    IMPORTANT ABOUT NaN PADDING
    ---------------------------
    When features have low cardinality (e.g., binary), scikit-learn may return
    fewer PDP grid points than `grid_resolution`. In `compute_pdp_nested_cv`,
    we pad shorter per-feature PDP arrays with `np.nan` to produce rectangular
    2D arrays. This aggregator therefore *skips* any (grid_value, pdp_value)
    pairs where either value is NaN.

    Parameters
    ----------
    all_results:
        Dict mapping model_name -> list of outer-fold dicts.

    source:
        Which PDP split to aggregate: "train" or "test".

    feature_names:
        Optional list/sequence of feature names, length must match the number
        of features. If provided, a 'feature_name' column is included.

    Returns
    -------
    aggregated:
        Dict mapping model_name -> pd.DataFrame in long format with columns:
          - model_name
          - trial
          - outer_fold
          - fold_index
          - feature_idx
          - feature_name (optional)
          - grid_index
          - grid_value
          - pdp_value
    """

    # Validate requested source string.
    if source not in {"train", "test"}:
        raise ValueError("source must be 'train' or 'test'")

    # Build the per-fold dict keys that contain PDP grid x-values.
    grid_key = f"pdp_grid_values_{source}"

    # Build the per-fold dict keys that contain PDP average y-values.
    avg_key = f"pdp_average_values_{source}"

    # Create output container: one aggregated DataFrame per model.
    aggregated: Dict[str, pd.DataFrame] = {}

    # Loop over each model name and its list of fold results.
    for model_name, folds in all_results.items():
        # Collect rows as plain dicts, then convert once to a DataFrame.
        rows = []

        # Loop over outer folds for this model.
        for fold_idx, r in enumerate(folds):
            # Skip folds that don't have PDP values computed for this source.
            if grid_key not in r or avg_key not in r:
                continue

            # Read the padded PDP grid arrays (x-values) for this fold/split.
            grid_array = np.asarray(r[grid_key], dtype=float)

            # Read the padded PDP average arrays (y-values) for this fold/split.
            avg_array = np.asarray(r[avg_key], dtype=float)

            # Extract number of features and padded grid length.
            n_features, n_grid_points = grid_array.shape

            # Pull metadata fields (if present) for traceability.
            trial = r.get("trial", None)
            outer_fold = r.get("outer_fold", None)

            # If feature names are provided, ensure they match the PDP feature count.
            if feature_names is not None and len(feature_names) != n_features:
                raise ValueError(
                    f"feature_names has length {len(feature_names)}, "
                    f"but PDP has n_features={n_features}"
                )

            # Loop over features (rows of the padded arrays).
            for feat_idx in range(n_features):
                # Get the padded grid values for this feature.
                grid_vals = grid_array[feat_idx]

                # Get the padded PDP values for this feature.
                pdp_vals = avg_array[feat_idx]

                # Resolve feature name if provided (otherwise None).
                feat_name = feature_names[feat_idx] if feature_names is not None else None

                # Loop over all padded grid positions for this feature.
                for grid_index in range(n_grid_points):
                    # Read the current grid x-value.
                    gv = grid_vals[grid_index]

                    # Read the current PDP y-value.
                    pv = pdp_vals[grid_index]

                    # Skip NaN-padded entries (these are not real PDP points).
                    if not (np.isfinite(gv) and np.isfinite(pv)):
                        continue

                    # Create one long-format record for this (feature, fold, grid point).
                    row = {
                        "model_name": model_name,
                        "trial": trial,
                        "outer_fold": outer_fold,
                        "fold_index": fold_idx,
                        "feature_idx": feat_idx,
                        "grid_index": grid_index,
                        "grid_value": float(gv),
                        "pdp_value": float(pv),
                    }

                    # Add feature_name field if feature_names were provided.
                    if feat_name is not None:
                        row["feature_name"] = feat_name

                    # Append to row list.
                    rows.append(row)

        # If we collected any rows, build the DataFrame from them.
        if rows:
            aggregated[model_name] = pd.DataFrame(rows)
        else:
            # Otherwise, return an empty DataFrame with the expected schema.
            cols = [
                "model_name",
                "trial",
                "outer_fold",
                "fold_index",
                "feature_idx",
                "grid_index",
                "grid_value",
                "pdp_value",
            ]

            # Include feature_name column in schema if requested.
            if feature_names is not None:
                cols.append("feature_name")

            # Create empty DataFrame for this model.
            aggregated[model_name] = pd.DataFrame(columns=cols)

    # Return dict of aggregated DataFrames per model.
    return aggregated

def add_interpolated_mean_pdp_to_agg(
    pdp_agg: Dict[str, pd.DataFrame],
    n_points: int = 100,
    discrete_threshold: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Given aggregated (long-format) PDP points per model (output of
    `aggregate_pdp_nested_cv`), compute a fold-averaged PDP curve (mean ± std)
    for each feature within each model.

    What this function does
    -----------------------
    For each model's aggregated PDP DataFrame:
      1) Group by feature (feature_idx).
      2) Create a "canonical" x-grid for that feature.
      3) For each fold, interpolate that fold's PDP curve onto the canonical grid.
      4) Compute mean and standard deviation across folds at each grid point.
      5) Store the resulting mean/std curve as a new DataFrame in the same dict,
         under the key: f"{model_name}_mean_pdp".

    Discrete vs continuous handling
    -------------------------------
    Some features are binary / low-cardinality. For those, using a dense linspace
    (e.g., 100 points) invents intermediate x-values that do not exist in the
    data. To avoid this, we treat a feature as "discrete" when the number of
    unique grid values observed across folds is <= `discrete_threshold`.

      - If discrete (<= discrete_threshold unique x-values):
          canonical_grid = sorted(unique x-values)
      - If continuous (> discrete_threshold unique x-values):
          canonical_grid = linspace(min(x), max(x), n_points)

    Parameters
    ----------
    pdp_agg:
        Dict mapping model_name -> aggregated PDP DataFrame.
        Each DataFrame is expected to include columns:
          - fold_index
          - feature_idx
          - grid_value
          - pdp_value
        and optionally:
          - feature_name

    n_points:
        Number of points in the canonical grid for continuous features.

    discrete_threshold:
        If a feature has <= this many unique x-values across folds, treat it as
        discrete and use those unique values as the canonical grid.

    Returns
    -------
    pdp_agg:
        Same dict as input, mutated in-place with additional entries:
          - pdp_agg[f"{model_name}_mean_pdp"] = DataFrame with columns:
              - feature_idx
              - grid_value
              - mean_pdp
              - std_pdp
              - feature_name (optional)
    """

    # Loop over a snapshot of items (so we can safely add new keys during iteration).
    for model_name, df in list(pdp_agg.items()):
        # Skip any entries that are already computed mean PDP outputs.
        if model_name.endswith("_mean_pdp"):
            continue

        # If the model's aggregated DataFrame is empty, create an empty mean_pdp entry.
        if df.empty:
            pdp_agg[f"{model_name}_mean_pdp"] = df.copy()
            continue

        # Accumulate output rows for the mean/std PDP curves.
        rows = []

        # Track whether feature_name exists so we preserve it if present.
        has_feature_name = "feature_name" in df.columns

        # Process each feature independently.
        for feature_id, df_feat in df.groupby("feature_idx"):
            # Drop any missing grid/PDP values (safety against NaNs).
            df_feat = df_feat.dropna(subset=["grid_value", "pdp_value"])

            # If nothing remains for this feature, skip it.
            if df_feat.empty:
                continue

            # Capture the feature name once (if provided).
            feat_name = df_feat["feature_name"].iloc[0] if has_feature_name else None

            # Compute the set of unique x-values observed for this feature across folds.
            x_unique_all = np.sort(df_feat["grid_value"].unique())

            # If feature looks discrete (few unique x-values), use those exact values.
            if len(x_unique_all) <= discrete_threshold:
                canonical_grid = x_unique_all
            # Otherwise treat as continuous-ish and build a dense, uniform grid.
            else:
                canonical_grid = np.linspace(x_unique_all.min(), x_unique_all.max(), n_points)

            # List all fold IDs that contributed PDP points for this feature.
            fold_ids = sorted(df_feat["fold_index"].unique())

            # Store fold-wise curves interpolated onto the canonical grid.
            interp_curves = []

            # Build one interpolated curve per fold.
            for fold_id in fold_ids:
                # Select rows for this fold and this feature.
                df_fold = df_feat[df_feat["fold_index"] == fold_id]

                # Extract x (grid values) and y (PDP values) for this fold.
                x = df_fold["grid_value"].values
                y = df_fold["pdp_value"].values

                # Sort by x so interpolation behaves correctly.
                sort_idx = np.argsort(x)
                x_sorted = x[sort_idx]
                y_sorted = y[sort_idx]

                # Interpolate fold PDP onto the canonical grid.
                # Note: For discrete features, canonical_grid is the discrete levels.
                y_interp = np.interp(canonical_grid, x_sorted, y_sorted)

                # Save the interpolated curve for this fold.
                interp_curves.append(y_interp)

            # If for some reason we got no fold curves, skip this feature.
            if not interp_curves:
                continue

            # Stack fold curves into a 2D array: (n_folds, n_grid_points_feature).
            interp_curves = np.vstack(interp_curves)

            # Compute mean PDP across folds at each grid point.
            mean_pdp = interp_curves.mean(axis=0)

            # Compute std PDP across folds at each grid point.
            std_pdp = interp_curves.std(axis=0)

            # Emit long-format rows for this feature's mean/std curve.
            for gv, m, s in zip(canonical_grid, mean_pdp, std_pdp):
                # Build one row at this canonical grid value.
                row = {
                    "feature_idx": feature_id,
                    "grid_value": float(gv),
                    "mean_pdp": float(m),
                    "std_pdp": float(s),
                }

                # Include feature name if available.
                if has_feature_name:
                    row["feature_name"] = feat_name

                # Append row to output list.
                rows.append(row)

        # Build the mean/std DataFrame if we created any rows.
        if rows:
            pdp_agg[f"{model_name}_mean_pdp"] = pd.DataFrame(rows)
        # Otherwise, create an empty DataFrame with the expected columns.
        else:
            base_cols = ["feature_idx", "grid_value", "mean_pdp", "std_pdp"]
            extra_cols = ["feature_name"] if has_feature_name else []
            pdp_agg[f"{model_name}_mean_pdp"] = pd.DataFrame(columns=(base_cols + extra_cols))

    # Return the same dict, now with additional "<model_name>_mean_pdp" entries.
    return pdp_agg


def add_grid_value_raw_to_pdp_agg(
    pdp_agg: Dict[str, pd.DataFrame],
    scaler_bundle: Optional[Dict[str, Any]] = None,
    *,
    preproc_key: str = "preproc",
    scaler_key: str = "scaler",
) -> Dict[str, pd.DataFrame]:
    """
    Add a `grid_value_raw` column to each model's PDP aggregation dataframe.

    This function assumes:
      - PDP grids were computed in the model's *scaled* feature space (z-scores),
        and `grid_value` in each dataframe corresponds to those scaled values.
      - You are using ONE global StandardScaler for all models/folds (the saved one).
      - Each dataframe in `pdp_agg` contains at least these columns:
          - "feature_name" : str
          - "grid_value"   : float

    If `scaler_bundle` is None, this function is a NO-OP and returns `pdp_agg`
    unchanged. This makes it safe to include as an optional pipeline step.

    The function uses the saved scaler statistics (mean_ and scale_) to convert
    each row's `grid_value` from scaled space back to raw units:

        raw = scaled * std + mean

    Parameters
    ----------
    pdp_agg:
        Dictionary mapping model_name -> long-format PDP dataframe.

    scaler_bundle:
        Optional. The loaded object that contains the fitted scaler and feature
        order used during fitting (typically from `load_all_results`).
        If None, return `pdp_agg` unchanged.

        When provided, must contain:
            scaler_bundle[preproc_key][scaler_key] : StandardScaler
            scaler_bundle[preproc_key]["feature_names"] : list[str]

    preproc_key:
        Key for the preprocessing artifacts dict inside `scaler_bundle`
        (default: "preproc").

    scaler_key:
        Key for the StandardScaler inside `scaler_bundle[preproc_key]`
        (default: "scaler").

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with the same keys as `pdp_agg`, where each dataframe has an
        added `grid_value_raw` column when `scaler_bundle` is provided.
        If `scaler_bundle` is None, returns `pdp_agg` unchanged.

    Raises
    ------
    KeyError
        If required keys/columns are missing, or if any `feature_name` cannot
        be mapped to the scaler's fitted feature space.

    TypeError
        If any pdp_agg value is not a pandas DataFrame.
    """
    # If no scaler bundle is provided, do nothing (pipeline-friendly behavior).
    if scaler_bundle is None:
        return pdp_agg

    # 1) Pull out the saved scaler + the feature ordering it was fit on
    if preproc_key not in scaler_bundle:
        raise KeyError(f"scaler_bundle missing key '{preproc_key}'")

    preproc = scaler_bundle[preproc_key]
    if scaler_key not in preproc:
        raise KeyError(f"scaler_bundle['{preproc_key}'] missing key '{scaler_key}'")
    if "feature_names" not in preproc:
        raise KeyError(f"scaler_bundle['{preproc_key}'] missing key 'feature_names'")

    scaler: StandardScaler = preproc[scaler_key]
    full_feature_names = list(preproc["feature_names"])
    name_to_idx = {name: i for i, name in enumerate(full_feature_names)}

    # Convenience views of scaler params
    mu = scaler.mean_   # shape (n_full_features,)
    sd = scaler.scale_  # shape (n_full_features,)

    out: Dict[str, pd.DataFrame] = {}

    # 2) Process each model dataframe
    for model_name, df_in in pdp_agg.items():
        if not isinstance(df_in, pd.DataFrame):
            raise TypeError(f"pdp_agg['{model_name}'] is not a pandas DataFrame")

        # Ensure required columns exist
        required_cols = {"feature_name", "grid_value"}
        missing_cols = required_cols - set(df_in.columns)
        if missing_cols:
            raise KeyError(
                f"pdp_agg['{model_name}'] missing required columns: {sorted(missing_cols)}"
            )

        df = df_in.copy()

        # Map each row’s feature_name -> full index in scaler space
        df["full_idx"] = df["feature_name"].map(name_to_idx)

        # Sanity check: any missing mappings?
        missing_names = df[df["full_idx"].isna()]["feature_name"].unique()
        if len(missing_names) > 0:
            raise KeyError(
                f"[{model_name}] Missing feature_name mappings: {list(missing_names)}"
            )

        # Convert to integer index array for fast numpy indexing
        idx = df["full_idx"].astype(int).to_numpy()

        # Inverse transform the grid values (scaled -> raw units)
        # raw = scaled * std + mean
        df["grid_value_raw"] = df["grid_value"].to_numpy() * sd[idx] + mu[idx]

        # Drop helper column
        df = df.drop(columns=["full_idx"])

        out[model_name] = df

    return out


def run_pdp_pipeline(
    all_results: Dict[str, List[Dict[str, Any]]],
    bundle: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    x_key: str = "combined_X_raw",
    y_key: str = "combined_y",
    data_source: str = "test",
    grid_resolution: int = 100,
    percentiles: Tuple[float, float] = (0.0, 1.0),
    centered: bool = False,
    n_points: int = 20,
    discrete_threshold: int = 10,
    scaler_bundle: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, pd.DataFrame], Dict[str, List[str]]]:
    """
    Run a nested-CV Partial Dependence Plot (PDP) workflow, supporting per-model feature sets.

    This pipeline mirrors the same per-model feature selection used during training and
    permutation importance:
      - X/y are pulled from `bundle` using `x_key` and `y_key` (e.g., group-aggregated arrays).
      - For each model, X is sliced ONCE using that model's config:
          * cfg["models"][model_name]["feature_names"] (exact-name selection), OR
          * cfg["models"][model_name]["n_features"]   (prefix/top-K selection), OR
          * neither (use all features).
      - The existing PDP routines are then applied on a per-model basis so that
        fold indices (outer_train_idx / outer_test_idx) remain valid and the feature
        dimension matches the fitted `final_model`.

    Pipeline stages
    --------------
    (1) compute_pdp_nested_cv (per model):
        Computes per-fold PDP arrays for the requested split(s) (train/test/both) and stores
        them in each fold dict inside `all_results[model_name]`. Low-cardinality features may
        have fewer grid points; your implementation may NaN-pad to allow stacking.

    (2) aggregate_pdp_nested_cv (per model):
        Converts per-fold PDP arrays into a single long-format DataFrame per model.
        Feature naming is handled per-model by passing that model's selected feature list.

    (3) add_interpolated_mean_pdp_to_agg (all models):
        Interpolates each fold's PDP onto a canonical grid per feature and computes mean/std
        across folds, writing results into:
            pdp_agg[f"{model_name}_mean_pdp"].

    (4) add_grid_value_raw_to_pdp_agg (optional, per model):
        If `scaler_bundle` is provided, converts PDP grid x-values back into "raw" units
        and adds a `grid_value_raw` column. (The exact logic depends on your
        add_grid_value_raw_to_pdp_agg implementation.)

    Parameters
    ----------
    all_results:
        Nested-CV results dict structured as:
            {model_name: [fold_dict_0, fold_dict_1, ...]}
        Each fold dict is expected to include:
            - "final_model": fitted estimator for that fold
            - "outer_train_idx": indices into bundle[x_key] / bundle[y_key]
            - "outer_test_idx": indices into bundle[x_key] / bundle[y_key]
        This object is mutated in-place to store PDP arrays/metadata.

    bundle:
        Dataset dictionary containing:
            - bundle[x_key]: 2D numpy array, shape (n_samples, n_features_full)
            - bundle[y_key]: 1D numpy array, shape (n_samples,)
            - bundle["feature_names"]: list[str] of length n_features_full

    cfg:
        Configuration dictionary with per-model feature selection keys under cfg["models"][model_name]:
            - "feature_names": Optional[list[str]]  (exact names)
            - "n_features": Optional[int]           (keep first K columns)
            - "feature_strict": bool (default True) passed to prepare_training_bundle
        Only one of ("feature_names", "n_features") may be set per model.

    x_key, y_key:
        Keys selecting which dataset level to use from the bundle (e.g., "combined_X_raw"/"combined_y"
        for group-level data, or "X_raw"/"y" for sample-level).
        The fold indices in all_results must refer to these arrays.

    data_source:
        Which outer split(s) to compute PDPs on:
            - "train": compute on outer-train split
            - "test": compute on outer-test split
            - "both": compute on both (aggregation defaults to "test" unless your aggregator supports both)

    grid_resolution:
        Requested number of grid points for continuous features for scikit-learn PDP computation.
        Discrete features may yield fewer points.

    percentiles:
        Percentile range (0..1) for PDP grid boundaries. (0.0, 1.0) uses full observed range.

    centered:
        Whether to center PDP curves (passed through to your PDP computation routine).

    n_points:
        Canonical grid size used when interpolating/averaging PDP curves across folds (continuous features).

    discrete_threshold:
        If a feature has <= this many unique values across folds, treat it as discrete when building
        the canonical grid (use observed unique values rather than a dense linspace).

    scaler_bundle:
        Optional object used by add_grid_value_raw_to_pdp_agg to convert grid values back into raw units.
        If None, the raw-value step is skipped.

    Returns
    -------
    all_results:
        Same dict as input, mutated in-place with fold-level PDP arrays stored inside each fold dict.

    pdp_agg:
        Dict of DataFrames containing aggregated PDP points and mean/std tables per model:
            - pdp_agg[model_name]                 : long-format PDP points (across folds)
            - pdp_agg[f"{model_name}_mean_pdp"]   : mean/std PDP curves per feature

    model_feature_names:
        Dict mapping model_name -> list[str] of feature names actually used for that model
        (useful for labeling plots, and for sanity-checking feature selection).

    Notes
    -----
    - The critical requirement is that the X passed into compute_pdp_nested_cv for a given model
      has the same feature dimension/order that the model was trained on.
    - This function assumes outer indices stored in all_results refer to the rows of bundle[x_key]/bundle[y_key].
    """

    # ---- pull X/y from bundle ----
    if x_key not in bundle:
        raise KeyError(f"bundle missing x_key='{x_key}'")
    if y_key not in bundle:
        raise KeyError(f"bundle missing y_key='{y_key}'")
    if "feature_names" not in bundle:
        raise KeyError("bundle must contain 'feature_names' for feature selection")

    X_full = np.asarray(bundle[x_key])
    y = np.asarray(bundle[y_key])
    feature_names_full = list(bundle["feature_names"])

    if X_full.ndim != 2:
        raise ValueError(f"bundle[{x_key}] must be 2D, got shape {X_full.shape}")
    if y.ndim != 1:
        raise ValueError(f"bundle[{y_key}] must be 1D, got shape {y.shape}")
    if X_full.shape[0] != len(y):
        raise ValueError(
            f"X/y mismatch for keys ({x_key}, {y_key}): X rows={X_full.shape[0]} vs len(y)={len(y)}"
        )
    if X_full.shape[1] != len(feature_names_full):
        raise ValueError(
            f"Mismatch: X has {X_full.shape[1]} cols but feature_names has {len(feature_names_full)}"
        )

    model_feature_names: Dict[str, List[str]] = {}

    # 3 base steps + optional raw-grid step
    n_steps = 4 if scaler_bundle is not None else 3
    steps = tqdm(total=n_steps, desc="PDP pipeline", unit="step")

    # ---- Step 1: compute PDP per model (model-specific X) ----
    steps.set_description("PDP pipeline: compute PDP per model")

    for model_name in list(all_results.keys()):
        if model_name not in cfg["models"]:
            raise KeyError(f"Model '{model_name}' not found in cfg['models'].")

        m_cfg = cfg["models"][model_name]

        keep_features = m_cfg.get("feature_names", None)   # list[str] | None
        n_features_model = m_cfg.get("n_features", None)   # int | None

        if keep_features is not None and n_features_model is not None:
            raise ValueError(
                f"{model_name}: set only one of 'feature_names' or 'n_features' (or neither)."
            )

        # Mini-bundle expected by prepare_training_bundle
        view_bundle = {"X_raw": X_full, "feature_names": feature_names_full}

        # Slice ONCE per model (exactly like training / permutation importance)
        if keep_features is not None or n_features_model is not None:
            mb = prepare_training_bundle(
                view_bundle,
                n_features=n_features_model,
                keep_features=keep_features,
                strict=m_cfg.get("feature_strict", True),
                dedupe=True,
                copy_bundle=True,
            )
        else:
            mb = view_bundle

        X_model = np.asarray(mb["X_raw"])
        fnames_model = list(mb["feature_names"])
        model_feature_names[model_name] = fnames_model

        # Optional: stash feature names per fold for traceability
        for r in all_results[model_name]:
            r["pdp_feature_names"] = fnames_model
            r["pdp_x_key"] = x_key
            r["pdp_y_key"] = y_key

        # Run your EXISTING compute function on ONLY this model
        sub_results = {model_name: all_results[model_name]}
        sub_results = compute_pdp_nested_cv(
            all_results=sub_results,
            X=X_model,
            y=y,
            grid_resolution=grid_resolution,
            percentiles=percentiles,
            data_source=data_source,
            centered=centered,
        )
        all_results[model_name] = sub_results[model_name]

    steps.update(1)

    # ---- Step 2: aggregate per model (so each model gets its own feature_names) ----
    steps.set_description("PDP pipeline: aggregate folds")
    agg_source = data_source if data_source in ("train", "test") else "test"

    pdp_agg: Dict[str, pd.DataFrame] = {}
    for model_name in list(all_results.keys()):
        sub_results = {model_name: all_results[model_name]}
        sub_agg = aggregate_pdp_nested_cv(
            all_results=sub_results,
            source=agg_source,
            feature_names=model_feature_names[model_name],  # <-- per-model feature names
        )
        pdp_agg.update(sub_agg)

    steps.update(1)

    # ---- Step 3: mean/std curves across folds ----
    steps.set_description("PDP pipeline: mean/std curves")
    pdp_agg = add_interpolated_mean_pdp_to_agg(
        pdp_agg=pdp_agg,
        n_points=n_points,
        discrete_threshold=discrete_threshold,
    )
    steps.update(1)

    # ---- Step 4: add raw grid values (optional) ----
    # Keep it simple: do it per model so each model's mean_pdp table maps correctly.
    if scaler_bundle is not None:
        steps.set_description("PDP pipeline: add raw grid values")

        for model_name in list(all_results.keys()):
            keys = [model_name, f"{model_name}_mean_pdp"]
            sub_pdp_agg = {k: pdp_agg[k] for k in keys if k in pdp_agg}

            sub_pdp_agg = add_grid_value_raw_to_pdp_agg(
                pdp_agg=sub_pdp_agg,
                scaler_bundle=scaler_bundle,
            )

            # write back
            for k, v in sub_pdp_agg.items():
                pdp_agg[k] = v

        steps.update(1)

    steps.close()
    return all_results, pdp_agg, model_feature_names


def plot_all_mean_pdp_with_std(
    pdp_agg: Mapping[str, Any],
    model_name: Union[str, Sequence[str], None] = None,
    *,
    feature_names: Optional[Union[List[str], Dict[str, List[str]]]] = None,
    x_scale: Literal["scaled", "raw"] = "scaled",
    figsize: tuple[float, float] = (8, 4),
    font_size: int = 12,
    y_lim: Optional[tuple[float, float]] = None,
    line_color: str = "darkblue",
    fill_color: str = "blue",
    sns_style: str = "whitegrid",
    fill_alpha: float = 0.2,
    fill_edgecolor: Optional[str] = None,
    method_alias: Optional[Mapping[str, str]] = None,
) -> None:
    """
    Plot mean PDP ± std for all features, for one or more models.

    This function expects the mean/std PDP tables produced by add_interpolated_mean_pdp_to_agg,
    stored in pdp_agg under keys like:
        pdp_agg[f"{model_name}_mean_pdp"].

    For each selected model, and for each feature within that model, it produces a separate figure
    showing:
        - mean PDP curve across outer folds
        - shaded band of ± 1 standard deviation across folds

    Feature labeling supports both global and per-model feature-name lists, similar to the updated
    permutation-importance plotting:
      - If the mean PDP table contains a "feature_name" column, that is used.
      - Otherwise, `feature_names` may be provided:
          * None -> labels default to "feature_idx={i}"
          * list[str] -> shared names (must align to feature_idx)
          * dict[str, list[str]] -> per-model names (recommended when models use different feature sets)

    Parameters
    ----------
    pdp_agg:
        Mapping containing PDP aggregation tables. Must include keys ending in "_mean_pdp".
        For a given model `m`, expected key:
            pdp_agg[f"{m}_mean_pdp"] -> pandas DataFrame.

        Required columns in each mean PDP DataFrame:
            - "feature_idx" : int feature index within the model
            - "grid_value"  : float x-grid value (scaled space)
            - "mean_pdp"    : float mean predicted response across folds
            - "std_pdp"     : float standard deviation across folds
        Optional columns:
            - "feature_name"     : str, used for labeling if present
            - "grid_value_raw"   : float, required if x_scale="raw"

    model_name:
        Which model(s) to plot:
          - None: plot all models found via "*_mean_pdp" keys
          - str: plot that single model
          - sequence[str]: plot those models

    feature_names:
        Optional feature names for labeling:
          - None: use "feature_name" column if present, else fallback to indices
          - list[str]: shared list (only appropriate if all models have identical feature sets)
          - dict[str, list[str]]: per-model names (recommended when models differ)

    x_scale:
        Which x-axis values to plot:
          - "scaled": uses "grid_value"
          - "raw": uses "grid_value_raw" (requires that add_grid_value_raw_to_pdp_agg was run)

    figsize:
        Figure size for each feature plot.

    font_size:
        Base font size for labels and ticks.

    y_lim:
        Optional y-axis limits as (ymin, ymax).

    line_color, fill_color:
        Colors for the PDP mean line and the ±std shaded band.

    sns_style:
        Seaborn style passed to sns.set(...).

    fill_alpha:
        Transparency for the shaded ±std region.

    fill_edgecolor:
        Optional edge color for the shaded region.

    method_alias:
        Optional mapping model_key -> display name used in the plot title.

    Raises
    ------
    KeyError:
        If requested models are missing, or if x_scale="raw" but "grid_value_raw" is absent.

    ValueError / TypeError:
        If pdp_agg entries are missing required columns or are not DataFrames.

    Notes
    -----
    - This plots one feature per figure to keep each PDP readable.
    - If you want a multi-panel grid layout, you can build on the same df grouping logic.
    """

    if method_alias is None:
        method_alias = {}

    # -------------------------
    # Decide which models to plot
    # -------------------------
    suffix = "_mean_pdp"
    available_models = sorted(
        [k[: -len(suffix)] for k in pdp_agg.keys() if isinstance(k, str) and k.endswith(suffix)]
    )

    if model_name is None:
        selected_models = available_models
        if not selected_models:
            raise KeyError(
                "No '*_mean_pdp' keys found in pdp_agg. "
                "Make sure you called add_interpolated_mean_pdp_to_agg first."
            )
    elif isinstance(model_name, str):
        selected_models = [model_name]
    else:
        selected_models = list(model_name)

    missing = [m for m in selected_models if f"{m}{suffix}" not in pdp_agg]
    if missing:
        raise KeyError(
            f"Model(s) not found in pdp_agg mean PDP keys: {missing}. "
            f"Available: {available_models}"
        )

    # Choose x column + xlabel based on requested x_scale
    if x_scale == "raw":
        x_col = "grid_value_raw"
        x_label = "Feature value (raw)"
    else:
        x_col = "grid_value"
        x_label = "Feature value (scaled)"

    sns.set(style=sns_style)

    # -------------------------
    # Plot: for each model, for each feature -> one figure
    # -------------------------
    for m in selected_models:
        mean_key = f"{m}{suffix}"
        df = pdp_agg[mean_key]

        if df is None or (hasattr(df, "empty") and df.empty):
            raise ValueError(f"No mean PDP data available for model '{m}' (key: '{mean_key}').")

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"pdp_agg['{mean_key}'] must be a pandas DataFrame; got {type(df)}")

        required_cols = {"feature_idx", "grid_value", "mean_pdp", "std_pdp"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise KeyError(
                f"Missing required columns in pdp_agg['{mean_key}']: {sorted(missing_cols)}. "
                f"Found columns: {list(df.columns)}"
            )

        if x_scale == "raw" and x_col not in df.columns:
            raise KeyError(
                f"You requested x_scale='raw' but '{x_col}' is missing in pdp_agg['{mean_key}']. "
                f"Run add_grid_value_raw_to_pdp_agg(...) on the mean-PDP tables first."
            )

        model_display = method_alias.get(m, m)

        # Determine a fallback feature name list for this model (if df lacks feature_name)
        fnames_model: Optional[List[str]] = None
        if "feature_name" not in df.columns:
            if feature_names is None:
                fnames_model = None
            elif isinstance(feature_names, dict):
                if m not in feature_names:
                    raise KeyError(
                        f"feature_names dict missing key '{m}'. Available: {list(feature_names.keys())}"
                    )
                fnames_model = list(feature_names[m])
            else:
                fnames_model = list(feature_names)

        # Validate fallback length if provided
        if fnames_model is not None:
            n_feat_in_df = int(df["feature_idx"].max()) + 1
            # Note: this assumes feature_idx are 0..(n-1). If not, you can relax this.
            if len(fnames_model) < n_feat_in_df:
                raise ValueError(
                    f"feature_names for model '{m}' has length {len(fnames_model)} "
                    f"but mean PDP table implies at least {n_feat_in_df} features."
                )

        for feature_id, df_feat in df.groupby("feature_idx"):
            df_feat = df_feat.sort_values(x_col)

            # Feature label logic (prefer df column; fallback to provided names; else idx)
            if "feature_name" in df.columns:
                feat_label = str(df_feat["feature_name"].iloc[0])
            elif fnames_model is not None and int(feature_id) < len(fnames_model):
                feat_label = fnames_model[int(feature_id)]
            else:
                feat_label = f"feature_idx={feature_id}"

            plt.figure(figsize=figsize)

            ax = sns.lineplot(
                data=df_feat,
                x=x_col,
                y="mean_pdp",
                marker="o",
                color=line_color, 
            )

            x = df_feat[x_col].to_numpy()
            mean = df_feat["mean_pdp"].to_numpy()
            std = df_feat["std_pdp"].to_numpy()

            ax.fill_between(
                x,
                mean - std,
                mean + std,
                alpha=fill_alpha,
                color=fill_color,
                edgecolor=fill_edgecolor,
            )

            ax.set_xlabel(x_label, fontsize=font_size, fontweight="bold")
            ax.set_ylabel("Predicted probability", fontsize=font_size, fontweight="bold")
            ax.set_title(
                f"Mean PDP ± std for {feat_label}\nModel: {model_display}",
                fontsize=font_size + 2,
                fontweight="bold",
            )

            if y_lim is not None:
                ax.set_ylim(*y_lim)

            ax.tick_params(axis="both", labelsize=font_size)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight("bold")

            plt.tight_layout()
            plt.show()


# def run_pdp_pipeline(
#     all_results: Dict[str, List[Dict[str, Any]]],
#     X: np.ndarray,
#     y: np.ndarray,
#     feature_names: Optional[Sequence[str]] = None,
#     *,
#     data_source: str = "test",
#     grid_resolution: int = 100,
#     percentiles: Tuple[float, float] = (0.0, 1.0),
#     centered: bool = False,
#     n_points: int = 20,
#     discrete_threshold: int = 10,
#     scaler_bundle: Optional[Dict[str, Any]] = None,
# ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, pd.DataFrame]]:
#     """
#     Run the full PDP workflow as a single pipeline with progress reporting.

#     This pipeline wraps up to four steps into one call:

#       (1) compute_pdp_nested_cv:
#           Computes per-fold partial dependence (PDP) curves for each model and
#           stores the results back into `all_results` under keys like:
#             - "pdp_grid_values_test" / "pdp_average_values_test"
#             - "pdp_grid_values_train" / "pdp_average_values_train"
#           Note: PDP arrays are NaN-padded per feature so that low-cardinality
#           features (e.g., binary) can have fewer grid points without breaking
#           array stacking.

#       (2) aggregate_pdp_nested_cv:
#           Converts the per-fold (padded) PDP arrays into a single long-format
#           DataFrame per model, skipping any NaN-padded entries.

#       (3) add_interpolated_mean_pdp_to_agg:
#           For each model and each feature, creates a canonical x-grid and
#           interpolates each fold's PDP curve onto it, then computes mean/std
#           across folds, storing results as:
#             pdp_agg[f"{model_name}_mean_pdp"].

#       (4) add_grid_value_raw_to_pdp_agg (optional):
#           If `scaler_bundle` is provided, converts standardized/scaled PDP
#           x-values (grid_value) back into raw feature units and appends a
#           `grid_value_raw` column to each DataFrame in `pdp_agg`.

#     A top-level tqdm progress bar is shown to indicate which pipeline stage
#     is running.

#     Parameters
#     ----------
#     all_results:
#         Nested-CV results dict with structure:
#           {model_name: [fold_dict_0, fold_dict_1, ...]}
#         Each fold dict must include:
#           - "final_model": fitted estimator for that fold
#           - "outer_train_idx": indices into X for outer-train split
#           - "outer_test_idx": indices into X for outer-test split
#         This object is mutated in-place by step (1) to store PDP arrays.

#     X:
#         Full feature matrix used for the nested CV, shape (n_samples, n_features).
#         PDP computations slice this matrix using outer_train_idx / outer_test_idx.

#     y:
#         Label vector, shape (n_samples,). Included for API symmetry with other
#         code paths; not used directly for PDP computations.

#     feature_names:
#         Optional list/sequence of feature names (length must equal n_features).
#         If provided, aggregations include a "feature_name" column.

#     data_source:
#         Which outer split to compute PDPs on in step (1):
#           - "train": compute PDP on outer-train split only
#           - "test":  compute PDP on outer-test split only
#           - "both":  compute PDP on both outer-train and outer-test splits
#         Note: step (2) aggregates a single source at a time; if you pass "both",
#         this pipeline will aggregate "test" by default unless you extend it.

#     grid_resolution:
#         Requested number of grid points for continuous features in
#         `PartialDependenceDisplay.from_estimator`. Low-cardinality features may
#         produce fewer grid points (handled via NaN padding).

#     percentiles:
#         Percentile range used by scikit-learn to define PDP grid boundaries
#         (e.g., (0.05, 0.95) trims extremes). Use (0.0, 1.0) to include full range.

#     centered:
#         Passed to `PartialDependenceDisplay.from_estimator`. If True, centers
#         the PDP curves.

#     n_points:
#         Number of points used for the canonical grid when a feature is treated
#         as continuous in step (3). Larger values produce smoother mean curves.

#     discrete_threshold:
#         If a feature has <= this many unique x-values across folds, treat it
#         as discrete in step (3) and use the observed unique values as the
#         canonical grid (instead of creating a dense linspace).

#     scaler_bundle:
#         Optional object (as returned by your `load_all_results`) containing the
#         scaler needed to invert standardized/scaled feature values.
#         If provided, step (4) runs and adds `grid_value_raw` to each DataFrame
#         in `pdp_agg`. If None, step (4) is skipped.

#     Returns
#     -------
#     all_results:
#         Same nested-CV results dict as input, mutated in-place with PDP arrays
#         added per fold.

#     pdp_agg:
#         Dict mapping:
#           - pdp_agg[model_name] = long-format PDP points across folds
#           - pdp_agg[f"{model_name}_mean_pdp"] = mean/std PDP curves per feature
#         If scaler_bundle is provided, each DataFrame also includes:
#           - grid_value_raw (inverse-transformed grid values)

#     Notes
#     -----
#     - Recommended pattern: load scaler_bundle outside this pipeline (I/O),
#       then pass the in-memory object into `run_pdp_pipeline`.
#     """

#     # Decide how many pipeline stages we will run (3 base + optional raw-grid stage).
#     n_steps = 4 if scaler_bundle is not None else 3

#     # Create a progress bar for the pipeline stages.
#     steps = tqdm(total=n_steps, desc="PDP pipeline", unit="step")

#     # ---- Step 1: compute per-fold PDP arrays and store them in all_results ----
#     steps.set_description("PDP pipeline: compute PDP per fold")
#     all_results = compute_pdp_nested_cv(
#         all_results=all_results,
#         X=X,
#         y=y,
#         grid_resolution=grid_resolution,
#         percentiles=percentiles,
#         data_source=data_source,
#         centered=centered,
#     )
#     steps.update(1)

#     # ---- Step 2: aggregate per-fold PDP arrays into long-format DataFrames ----
#     steps.set_description("PDP pipeline: aggregate folds")
#     agg_source = data_source if data_source in ("train", "test") else "test"
#     pdp_agg = aggregate_pdp_nested_cv(
#         all_results=all_results,
#         source=agg_source,
#         feature_names=feature_names,
#     )
#     steps.update(1)

#     # ---- Step 3: compute mean/std PDP curves across folds per feature ----
#     steps.set_description("PDP pipeline: mean/std curves")
#     pdp_agg = add_interpolated_mean_pdp_to_agg(
#         pdp_agg=pdp_agg,
#         n_points=n_points,
#         discrete_threshold=discrete_threshold,
#     )
#     steps.update(1)

#     # ---- Step 4 (optional): add inverse-transformed raw grid values ----
#     if scaler_bundle is not None:
#         steps.set_description("PDP pipeline: add raw grid values")
#         pdp_agg = add_grid_value_raw_to_pdp_agg(
#             pdp_agg=pdp_agg,
#             scaler_bundle=scaler_bundle,
#         )
#         steps.update(1)

#     # Close the progress bar cleanly.
#     steps.close()

#     # Return both the enriched all_results and the aggregated DataFrames.
#     return all_results, pdp_agg

# def plot_all_mean_pdp_with_std(
#     pdp_agg: Mapping[str, Any],
#     model_name: str | Sequence[str] | None = None,  # None -> all models found in pdp_agg
#     *,
#     x_scale: Literal["scaled", "raw"] = "scaled",
#     figsize: tuple[float, float] = (8, 4),
#     font_size: int = 12,
#     y_lim: Optional[tuple[float, float]] = None,
#     line_color: str = "darkblue",
#     fill_color: str = "blue",
#     sns_style: str = "whitegrid",
#     fill_alpha: float = 0.2,
#     fill_edgecolor: Optional[str] = None,
#     method_alias: Mapping[str, str] | None = None,  # NEW
# ) -> None:
#     """
#     Plot mean PDP ± std for ALL features, for one or more models.

#     (Same behavior as before; `method_alias` only affects the displayed model name in the title.)
#     """

#     if method_alias is None:
#         method_alias = {}

#     # -------------------------
#     # Decide which models to plot
#     # -------------------------
#     suffix = "_mean_pdp"
#     available_models = sorted(
#         [k[: -len(suffix)] for k in pdp_agg.keys() if isinstance(k, str) and k.endswith(suffix)]
#     )

#     if model_name is None:
#         selected_models = available_models
#         if not selected_models:
#             raise KeyError(
#                 "No '*_mean_pdp' keys found in pdp_agg. "
#                 "Make sure you called add_interpolated_mean_pdp_to_agg first."
#             )
#     elif isinstance(model_name, str):
#         selected_models = [model_name]
#     else:
#         selected_models = list(model_name)

#     # Validate requested models exist
#     missing = [m for m in selected_models if f"{m}{suffix}" not in pdp_agg]
#     if missing:
#         raise KeyError(
#             f"Model(s) not found in pdp_agg mean PDP keys: {missing}. "
#             f"Available: {available_models}"
#         )

#     # Choose x column + xlabel based on requested x_scale
#     if x_scale == "raw":
#         x_col = "grid_value_raw"
#         x_label = "Feature value (raw)"
#     else:
#         x_col = "grid_value"
#         x_label = "Feature value (scaled)"

#     sns.set(style=sns_style)

#     # -------------------------
#     # Plot: for each model, for each feature -> one figure
#     # -------------------------
#     for m in selected_models:
#         mean_key = f"{m}{suffix}"
#         df = pdp_agg[mean_key]

#         if df is None or (hasattr(df, "empty") and df.empty):
#             raise ValueError(f"No mean PDP data available for model '{m}' (key: '{mean_key}').")

#         if not isinstance(df, pd.DataFrame):
#             raise TypeError(f"pdp_agg['{mean_key}'] must be a pandas DataFrame; got {type(df)}")

#         required_cols = {"feature_idx", "grid_value", "mean_pdp", "std_pdp"}
#         missing_cols = required_cols - set(df.columns)
#         if missing_cols:
#             raise KeyError(
#                 f"Missing required columns in pdp_agg['{mean_key}']: {sorted(missing_cols)}. "
#                 f"Found columns: {list(df.columns)}"
#             )

#         if x_scale == "raw" and x_col not in df.columns:
#             raise KeyError(
#                 f"You requested x_scale='raw' but '{x_col}' is missing in pdp_agg['{mean_key}']. "
#                 f"Run add_grid_value_raw_to_pdp_agg(...) on the mean-PDP tables first."
#             )

#         has_feature_name = "feature_name" in df.columns
#         model_display = method_alias.get(m, m)

#         for feature_id, df_feat in df.groupby("feature_idx"):
#             df_feat = df_feat.sort_values(x_col)

#             if has_feature_name:
#                 feat_label = str(df_feat["feature_name"].iloc[0])
#             else:
#                 feat_label = f"feature_idx={feature_id}"

#             plt.figure(figsize=figsize)

#             ax = sns.lineplot(
#                 data=df_feat,
#                 x=x_col,
#                 y="mean_pdp",
#                 marker="o",
#                 color=line_color,
#             )

#             x = df_feat[x_col].to_numpy()
#             mean = df_feat["mean_pdp"].to_numpy()
#             std = df_feat["std_pdp"].to_numpy()

#             ax.fill_between(
#                 x,
#                 mean - std,
#                 mean + std,
#                 alpha=fill_alpha,
#                 color=fill_color,
#                 edgecolor=fill_edgecolor,
#             )

#             ax.set_xlabel(x_label, fontsize=font_size, fontweight="bold")
#             ax.set_ylabel("Predicted probability", fontsize=font_size, fontweight="bold")
#             ax.set_title(
#                 f"Mean PDP ± std for {feat_label}\nModel: {model_display}",
#                 fontsize=font_size + 2,
#                 fontweight="bold",
#             )

#             if y_lim is not None:
#                 ax.set_ylim(*y_lim)

#             ax.tick_params(axis="both", labelsize=font_size)
#             for label in ax.get_xticklabels() + ax.get_yticklabels():
#                 label.set_fontweight("bold")

#             plt.tight_layout()
#             plt.show()


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



# def build_long_predictions_df(
#     all_results: Mapping[str, Sequence[Mapping[str, Any]]],
#     *,
#     model_name: str,
#     groups_all: Optional[np.ndarray] = None,
#     group_id_to_key: Optional[Mapping[int, Tuple[str, str]]] = None,  # group -> (label_str, subject_id)
#     methods: Optional[Sequence[str]] = None,
#     include_uncalibrated: bool = True,
#     include_test: bool = True,
#     include_train_oof: bool = False,
#     # NEW: what to use as unit id when groups aren't available
#     unit_col: str = "idx",  # in non-grouped setting, unit id is just idx
# ) -> pd.DataFrame:
#     """
#     Build a long table with one row per (idx, fold, variant, split).

#     If groups_all and group_id_to_key are provided, adds:
#       - group (patient/group id)
#       - group_label (e.g., "ASD"/"TD")
#       - subject_id (e.g., NDAR...)

#     Otherwise, omits those fields and uses unit_col (default "idx") as the unit identifier.

#     Output columns (always):
#       ["model","variant","split","trial","outer_fold","idx","y","p"]

#     Output columns (only if grouping info provided):
#       + ["group","group_label","subject_id"]
#     """
#     if model_name not in all_results:
#         raise KeyError(f"Model '{model_name}' not found in all_results.")

#     folds = all_results[model_name]

#     have_groups = (groups_all is not None) and (group_id_to_key is not None)
#     if have_groups:
#         groups_all = np.asarray(groups_all)

#     # Discover calibration methods if not provided
#     if methods is None:
#         discovered = set()
#         for r in folds:
#             for k in r.keys():
#                 if k.startswith("calib_test_predictions_"):
#                     discovered.add(k.replace("calib_test_predictions_", "", 1))
#         methods_list = sorted(discovered)
#     else:
#         methods_list = list(methods)

#     variants: List[str] = []
#     if include_uncalibrated:
#         variants.append("uncalib")
#     variants.extend(methods_list)

#     rows: List[Dict[str, Any]] = []

#     def _append_rows(
#         *,
#         idx_arr: np.ndarray,
#         y_arr: np.ndarray,
#         p_arr: np.ndarray,
#         split_name: str,
#         trial: Any,
#         outer_fold: Any,
#         variant: str,
#     ) -> None:
#         idx_arr = np.asarray(idx_arr, dtype=int)
#         y_arr = np.asarray(y_arr, dtype=int)
#         p_arr = np.asarray(p_arr, dtype=float)

#         if len(idx_arr) != len(y_arr) or len(idx_arr) != len(p_arr):
#             raise ValueError(
#                 f"Length mismatch: trial={trial}, outer_fold={outer_fold}, variant={variant}, split={split_name} "
#                 f"len(idx)={len(idx_arr)}, len(y)={len(y_arr)}, len(p)={len(p_arr)}"
#             )

#         if have_groups:
#             assert groups_all is not None and group_id_to_key is not None

#             if idx_arr.max(initial=-1) >= len(groups_all) or idx_arr.min(initial=0) < 0:
#                 raise IndexError(
#                     f"Some idx values are out of bounds for groups_all (len={len(groups_all)}). "
#                     f"idx range: [{idx_arr.min()}, {idx_arr.max()}]"
#                 )

#             g_arr = groups_all[idx_arr]

#             # lookup label_str and subject_id per group
#             label_strs = []
#             subject_ids = []
#             for g in g_arr:
#                 lab, sid = group_id_to_key.get(int(g), (None, None))
#                 label_strs.append(lab)
#                 subject_ids.append(sid)

#             for i, g, lab, sid, yy, pp in zip(idx_arr, g_arr, label_strs, subject_ids, y_arr, p_arr):
#                 rows.append({
#                     "model": model_name,
#                     "variant": variant,
#                     "split": split_name,
#                     "trial": trial,
#                     "outer_fold": outer_fold,
#                     "idx": int(i),
#                     "group": int(g),
#                     "group_label": lab,
#                     "subject_id": sid,
#                     "y": int(yy),
#                     "p": float(pp),
#                 })
#         else:
#             # Non-grouped case: unit is just idx (or whatever you pass as unit_col).
#             # We do NOT invent group_label/subject_id here because the function doesn't know them.
#             for i, yy, pp in zip(idx_arr, y_arr, p_arr):
#                 rows.append({
#                     "model": model_name,
#                     "variant": variant,
#                     "split": split_name,
#                     "trial": trial,
#                     "outer_fold": outer_fold,
#                     "idx": int(i),
#                     unit_col: int(i) if unit_col != "idx" else int(i),  # keep explicit for clarity
#                     "y": int(yy),
#                     "p": float(pp),
#                 })

#     for r in folds:
#         trial = r.get("trial", None)
#         outer_fold = r.get("outer_fold", None)

#         # ---------- outer test ----------
#         if include_test:
#             idx = np.asarray(r["outer_test_idx"], dtype=int)
#             y = np.asarray(r["y_test"], dtype=int)

#             for v in variants:
#                 key = "y_test_scores" if v == "uncalib" else f"calib_test_predictions_{v}"
#                 if key not in r:
#                     continue
#                 p = np.asarray(r[key], dtype=float)

#                 _append_rows(
#                     idx_arr=idx,
#                     y_arr=y,
#                     p_arr=p,
#                     split_name="test",
#                     trial=trial,
#                     outer_fold=outer_fold,
#                     variant=v,
#                 )

#         # ---------- train OOF (optional) ----------
#         if include_train_oof:
#             idx_tr = np.asarray(r["outer_train_idx"], dtype=int)
#             y_tr = np.asarray(r["y_train"], dtype=int)

#             for v in variants:
#                 key = "cv_uncalib_train_predictions" if v == "uncalib" else f"cv_calib_train_predictions_{v}"
#                 if key not in r:
#                     continue
#                 p_tr = np.asarray(r[key], dtype=float)

#                 _append_rows(
#                     idx_arr=idx_tr,
#                     y_arr=y_tr,
#                     p_arr=p_tr,
#                     split_name="train_oof",
#                     trial=trial,
#                     outer_fold=outer_fold,
#                     variant=v,
#                 )

#     df = pd.DataFrame(rows)

#     # dtypes
#     if not df.empty:
#         df["model"] = df["model"].astype(str)
#         df["variant"] = df["variant"].astype(str)
#         df["split"] = df["split"].astype(str)
#         df["idx"] = df["idx"].astype(int)
#         df["y"] = df["y"].astype(int)
#         df["p"] = df["p"].astype(float)

#         if have_groups:
#             df["group"] = df["group"].astype(int)

#     return df

# def pooled_patient_risk_summary(
#     df_long: pd.DataFrame,
#     *,
#     agg: Literal["mean", "median", "max", "quantile", "softmax"] = "mean",
#     quantile: float = 0.75,
#     beta: float = 5.0,
#     eps: float = 1e-6,
#     lower_q: float = 0.05,
#     upper_q: float = 0.95,
#     ddof: int = 0,
#     grouping: Literal["all_trials", "per_trial_fold"] = "all_trials",
#     unit_col: Optional[str] = "group",
#     # guardrail (optional)
#     min_expected_rows_per_unit: Optional[int] = None,
# ) -> pd.DataFrame:
#     """
#     Aggregate window-/row-level predicted probabilities to patient-level summaries.

#     Context / why this exists
#     -------------------------
#     In your EEG setting, each patient (a.k.a. *group*) contributes many rows/windows that all share
#     the same clinical label. During repeated/nested cross-validation, a patient may appear in the outer-test
#     set multiple times (across trials/seeds and outer folds). Therefore, for a given patient you can end up
#     with many out-of-sample predicted probabilities:
#       - variability across EEG windows within the patient
#       - variability across CV repetitions / train-test splits (different fitted models)

#     This function is designed to support BOTH common aggregation choices with the SAME code path:

#     Parameters
#     ----------
#     df_long:
#         Long table with at least columns:
#         ["model","variant","split","trial","outer_fold","group","subject_id","group_label","y","p"].
#         Each row is typically a window/epoch prediction for a patient/group.
#         Notes:
#         - trial/outer_fold must be present if grouping="per_trial_fold".
#         - p should be numeric probabilities in [0, 1]. Non-numeric values are coerced to NaN and dropped
#             within each group.

#     grouping:
#         Controls what constitutes a “patient aggregation unit” (i.e., the groupby key).
#         - "all_trials":      ["model","variant","split","group"]
#             Pools window-level predictions for a patient across ALL CV runs (trial × outer_fold).
#             Interpretation: for each patient, pool *all* out-of-sample window-level predictions across all runs,
#             then compute a single patient-level center + spread.

#         - "per_trial_fold":  ["model","variant","split","trial","outer_fold","group"]
#             Computes one patient summary per run (trial × outer_fold), isolating within-run window variability.
#             Interpretation: compute one patient-level summary per CV run (trial × outer_fold), using only that run’s
#             window-level predictions for the patient. This isolates within-patient window heterogeneity per run.

#     agg:
#         Aggregation used to produce a single patient-level probability summary from the (optionally winsorized)
#         window probabilities.
#         - "mean": mean of p_used within group
#         - "median": median of p_used within group
#         - "max": max of p_used within group (max pooling)
#         - "quantile": quantile(p_used, quantile) within group (quantile pooling)
#         - "softmax": softmax-pooled weighted mean of probabilities within group
#             weights = softmax(beta * logit(p_used))
#             p_softmax = sum_i w_i * p_used_i
#             (Output remains in [0, 1] because it is a convex combination of probabilities.)

#     quantile:
#         Quantile used when agg="quantile". Must be in [0, 1]. Default 0.75.
#         Example: quantile=0.90 returns the 90th percentile of the within-group (winsorized) probabilities.

#     beta:
#         Softmax sharpness used when agg="softmax". Must be > 0. Default 5.0.
#         Interpretation:
#         - beta -> 0: weights become nearly uniform (approaches mean pooling)
#         - larger beta: weights concentrate on the highest-evidence windows (more max-like)

#     eps:
#         Numerical stability clip used when agg="softmax". Default 1e-6.
#         Probabilities are clipped to [eps, 1-eps] before computing logit(p) to avoid infinities.

#     lower_q, upper_q:
#         Winsorization quantile cutoffs in [0, 1], with lower_q < upper_q.
#         These are applied within each group defined by `grouping`.
#         - lower_q == 0.0 disables LOWER capping (no lower winsorization)
#         - upper_q == 1.0 disables UPPER capping (no upper winsorization)
#         If a side is disabled, its reported cap value (p_cap_low or p_cap_high) is set to NaN to avoid
#         implying a cap was applied.
#         Setting lower_q=0.0 and upper_q=1.0 disables winsorization entirely (p_used == p).

#     ddof:
#         Degrees of freedom used for standard deviation of p_used within each group (np.std).
#         Default 0. (ddof=1 gives the sample standard deviation.)

#     unit_col:
#         Column that identifies the “unit” you are aggregating to.
#         - EEG windowed setting: unit_col="group" (patient id for GroupKFold)
#         - Single-row-per-patient setting: unit_col="idx" (unique patient row id)
#         If unit_col is None, the function will attempt to infer one from ["group","subject_id","idx"].

#     min_expected_rows_per_unit:
#         Optional guardrail. If set (e.g., 2 or 10), the function raises if median rows per unit is below this,
#         which helps catch mistakes like using a per-row unique id in a windowed dataset.
#         Leave as None for the future "one row per patient" case where n_rows_per_unit == 1 is expected.        
#     Returns
#     -------
#     pd.DataFrame
#         One row per unique key defined by `grouping`, containing:
#         - the grouping key columns (e.g., model/variant/split/group, plus trial/outer_fold if per_trial_fold),
#         - subject_id, group_label, y (taken as the first value within the group),
#         - n_windows: number of non-NaN window probabilities used,
#         - p_mean / p_median / p_max / p_qXX / p_softmax: the chosen patient-level probability summary,
#         - p_total_std: std of p_used within group (interpreted according to the chosen grouping),
#         - lower_q, upper_q and realized cap values p_cap_low / p_cap_high,
#         - for softmax: beta, eps
#         - for quantile: quantile

#     """
#     # -------------------------
#     # Infer unit_col if needed
#     # -------------------------
#     if unit_col is None:
#         for cand in ("group", "subject_id", "idx"):
#             if cand in df_long.columns:
#                 unit_col = cand
#                 break
#         if unit_col is None:
#             raise KeyError("Could not infer unit_col. Please pass unit_col='group' or 'idx' (or another id column).")

#     if unit_col not in df_long.columns:
#         raise KeyError(f"unit_col='{unit_col}' not found in df_long columns.")

#     # -------------------------
#     # Grouping schemes (built from unit_col)
#     # -------------------------
#     GROUPING_SCHEMES: Dict[str, list[str]] = {
#         "all_trials": ["model", "variant", "split", unit_col],
#         "per_trial_fold": ["model", "variant", "split", "trial", "outer_fold", unit_col],
#     }
#     if grouping not in GROUPING_SCHEMES:
#         raise ValueError(f"Unknown grouping='{grouping}'. Choose one of {list(GROUPING_SCHEMES)}")

#     group_cols = GROUPING_SCHEMES[grouping]

#     # -------------------------
#     # Validation
#     # -------------------------
#     if agg not in ("mean", "median", "max", "quantile", "softmax"):
#         raise ValueError("agg must be one of {'mean','median','max','quantile','softmax'}")

#     if agg == "quantile" and not (0.0 <= quantile <= 1.0):
#         raise ValueError(f"quantile must be in [0,1] for quantile pooling. Got {quantile}")

#     if agg == "softmax":
#         if beta <= 0:
#             raise ValueError(f"beta must be > 0 for softmax pooling. Got {beta}")
#         if eps <= 0 or eps >= 0.1:
#             raise ValueError(f"eps should be small and positive (e.g. 1e-6). Got {eps}")

#     if not (0.0 <= lower_q <= 1.0 and 0.0 <= upper_q <= 1.0 and lower_q < upper_q):
#         raise ValueError(f"Require 0 <= lower_q < upper_q <= 1. Got {lower_q}, {upper_q}")

#     required_base = {"model", "variant", "split", "subject_id", "group_label", "y", "p", unit_col}
#     missing = required_base - set(df_long.columns)
#     if missing:
#         raise KeyError(f"df_long missing required columns: {sorted(missing)}")

#     need_cols = set(group_cols) - set(df_long.columns)
#     if need_cols:
#         raise KeyError(f"grouping='{grouping}' requires missing columns: {sorted(need_cols)}")

#     d = df_long.copy()
#     d["p"] = pd.to_numeric(d["p"], errors="coerce")

#     # -------------------------
#     # Optional guardrail
#     # -------------------------
#     if min_expected_rows_per_unit is not None:
#         # compute within the filtered dataframe d (not grouped by run)
#         counts = d.groupby([unit_col], sort=False).size()
#         if len(counts) > 0:
#             med = float(np.median(counts.to_numpy()))
#             if med < float(min_expected_rows_per_unit):
#                 raise ValueError(
#                     f"Guardrail: median rows per unit ({unit_col}) is {med:.1f}, "
#                     f"below min_expected_rows_per_unit={min_expected_rows_per_unit}. "
#                     f"Did you accidentally set unit_col to a per-row unique id?"
#                 )

#     # -------------------------
#     # Output column naming
#     # -------------------------
#     if agg == "quantile":
#         q_tag = int(round(quantile * 100))
#         center_col = f"p_q{q_tag}"
#     elif agg == "softmax":
#         center_col = "p_softmax"
#     else:
#         center_col = {"mean": "p_mean", "median": "p_median", "max": "p_max"}[agg]

#     out_rows: list[dict] = []

#     # Independent-sided winsorization flags
#     apply_low = (lower_q > 0.0)
#     apply_high = (upper_q < 1.0)

#     for keys, gdf in d.groupby(group_cols, sort=False):
#         p = gdf["p"].to_numpy(dtype=float)
#         p = p[~np.isnan(p)]
#         if p.size == 0:
#             continue

#         # Winsorize (optional, per side)
#         if not apply_low and not apply_high:
#             lo = np.nan
#             hi = np.nan
#             p_used = p
#         else:
#             lo = float(np.quantile(p, lower_q)) if apply_low else np.nan
#             hi = float(np.quantile(p, upper_q)) if apply_high else np.nan
#             lo_clip = lo if apply_low else -np.inf
#             hi_clip = hi if apply_high else np.inf
#             p_used = np.clip(p, lo_clip, hi_clip)

#         # Aggregate
#         if agg == "mean":
#             p_center = float(np.mean(p_used))
#         elif agg == "median":
#             p_center = float(np.median(p_used))
#         elif agg == "max":
#             p_center = float(np.max(p_used))
#         elif agg == "quantile":
#             p_center = float(np.quantile(p_used, quantile))
#         else:  # "softmax"
#             p_clip = np.clip(p_used, eps, 1.0 - eps)
#             s = np.log(p_clip) - np.log1p(-p_clip)  # logit(p)
#             t = beta * s
#             t = t - np.max(t)
#             w = np.exp(t)
#             w_sum = np.sum(w)
#             if not np.isfinite(w_sum) or w_sum == 0.0:
#                 p_center = float(np.mean(p_used))
#             else:
#                 w = w / w_sum
#                 p_center = float(np.sum(w * p_used))

#         # Spread (on winsorized values, regardless of agg)
#         p_std = float(np.std(p_used, ddof=ddof))

#         row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
#         row.update(
#             {
#                 "grouping": grouping,
#                 "unit_col": unit_col,
#                 "subject_id": gdf["subject_id"].iloc[0],
#                 "group_label": gdf["group_label"].iloc[0],
#                 "y": int(gdf["y"].iloc[0]),
#                 "n_windows": int(p.size),
#                 center_col: p_center,
#                 "p_total_std": p_std,
#                 "lower_q": float(lower_q),
#                 "upper_q": float(upper_q),
#                 "p_cap_low": lo,
#                 "p_cap_high": hi,
#             }
#         )

#         if agg == "quantile":
#             row["quantile"] = float(quantile)
#         if agg == "softmax":
#             row["beta"] = float(beta)
#             row["eps"] = float(eps)

#         out_rows.append(row)

#     return pd.DataFrame(out_rows)




def build_long_predictions_df(
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
    Build a long-form predictions table with one row per predicted example, optionally across
    multiple models.

    This function converts `all_results` (a dict keyed by model name, where each value is a
    sequence of per-fold result dictionaries) into a single tidy/long pandas DataFrame suitable
    for downstream aggregation (e.g., patient pooling) and plotting.

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
    # grouping=None -> "all_trials" (minimal change)
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
    # Validation (minimal: subject_id/group_label optional)
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

    return pd.DataFrame(out_rows)



# def pooled_patient_risk_summary(
#     df_long: pd.DataFrame,
#     *,
#     agg: Literal["mean", "median", "max", "quantile", "softmax"] = "mean",
#     quantile: float = 0.75,
#     beta: float = 5.0,
#     eps: float = 1e-6,
#     lower_q: float = 0.05,
#     upper_q: float = 0.95,
#     ddof: int = 0,
#     grouping: Literal["all_trials", "per_trial_fold"] = "all_trials",
#     unit_col: Optional[str] = "group",
#     splits: Optional[Sequence[str]] = None,
#     include_test: bool = True,
#     include_train_oof: bool = False,
# ) -> pd.DataFrame:
#     """
#     Aggregate window-/row-level predicted probabilities to patient-level summaries.

#     Context / why this exists
#     -------------------------
#     In your EEG setting, each patient (a.k.a. *group*) contributes many rows/windows that all share
#     the same clinical label. During repeated/nested cross-validation, a patient may appear in the outer-test
#     set multiple times (across trials/seeds and outer folds). Therefore, for a given patient you can end up
#     with many out-of-sample predicted probabilities:
#       - variability across EEG windows within the patient
#       - variability across CV repetitions / train-test splits (different fitted models)

#     This function is designed to support BOTH common aggregation choices with the SAME code path:

#     Parameters
#     ----------
#     df_long:
#         Long table with at least columns:
#         ["model","variant","split","trial","outer_fold","group","subject_id","group_label","y","p"].
#         Each row is typically a window/epoch prediction for a patient/group.
#         Notes:
#         - trial/outer_fold must be present if grouping="per_trial_fold".
#         - p should be numeric probabilities in [0, 1]. Non-numeric values are coerced to NaN and dropped
#             within each group.

#     grouping:
#         Controls what constitutes a “patient aggregation unit” (i.e., the groupby key).
#         - "all_trials":      ["model","variant","split","group"]
#             Pools window-level predictions for a patient across ALL CV runs (trial × outer_fold).
#             Interpretation: for each patient, pool *all* out-of-sample window-level predictions across all runs,
#             then compute a single patient-level center + spread.

#         - "per_trial_fold":  ["model","variant","split","trial","outer_fold","group"]
#             Computes one patient summary per run (trial × outer_fold), isolating within-run window variability.
#             Interpretation: compute one patient-level summary per CV run (trial × outer_fold), using only that run’s
#             window-level predictions for the patient. This isolates within-patient window heterogeneity per run.

#     agg:
#         Aggregation used to produce a single patient-level probability summary from the (optionally winsorized)
#         window probabilities.
#         - "mean": mean of p_used within group
#         - "median": median of p_used within group
#         - "max": max of p_used within group (max pooling)
#         - "quantile": quantile(p_used, quantile) within group (quantile pooling)
#         - "softmax": softmax-pooled weighted mean of probabilities within group
#             weights = softmax(beta * logit(p_used))
#             p_softmax = sum_i w_i * p_used_i
#             (Output remains in [0, 1] because it is a convex combination of probabilities.)

#     quantile:
#         Quantile used when agg="quantile". Must be in [0, 1]. Default 0.75.
#         Example: quantile=0.90 returns the 90th percentile of the within-group (winsorized) probabilities.

#     beta:
#         Softmax sharpness used when agg="softmax". Must be > 0. Default 5.0.
#         Interpretation:
#         - beta -> 0: weights become nearly uniform (approaches mean pooling)
#         - larger beta: weights concentrate on the highest-evidence windows (more max-like)

#     eps:
#         Numerical stability clip used when agg="softmax". Default 1e-6.
#         Probabilities are clipped to [eps, 1-eps] before computing logit(p) to avoid infinities.

#     lower_q, upper_q:
#         Winsorization quantile cutoffs in [0, 1], with lower_q < upper_q.
#         These are applied within each group defined by `grouping`.
#         - lower_q == 0.0 disables LOWER capping (no lower winsorization)
#         - upper_q == 1.0 disables UPPER capping (no upper winsorization)
#         If a side is disabled, its reported cap value (p_cap_low or p_cap_high) is set to NaN to avoid
#         implying a cap was applied.
#         Setting lower_q=0.0 and upper_q=1.0 disables winsorization entirely (p_used == p).

#     ddof:
#         Degrees of freedom used for standard deviation of p_used within each group (np.std).
#         Default 0. (ddof=1 gives the sample standard deviation.)

#     unit_col:
#         Column that identifies the “unit” you are aggregating to.
#         - EEG windowed setting: unit_col="group" (patient id for GroupKFold)
#         - Single-row-per-patient setting: unit_col="idx" (unique patient row id)
#         If unit_col is None, the function will attempt to infer one from ["group","subject_id","idx"].
    
#     Split filtering    
#     If `splits` is provided, only those splits are included.
#     Otherwise, splits are selected based on include_test/include_train_oof:
#       - include_test=True      -> include split == "test"
#       - include_train_oof=True -> include split == "train_oof" 
      
#     Returns
#     -------
#     pd.DataFrame
#         One row per unique key defined by `grouping`, containing:
#         - the grouping key columns (e.g., model/variant/split/group, plus trial/outer_fold if per_trial_fold),
#         - subject_id, group_label, y (taken as the first value within the group),
#         - n_windows: number of non-NaN window probabilities used,
#         - p_mean / p_median / p_max / p_qXX / p_softmax: the chosen patient-level probability summary,
#         - p_total_std: std of p_used within group (interpreted according to the chosen grouping),
#         - lower_q, upper_q and realized cap values p_cap_low / p_cap_high,
#         - for softmax: beta, eps
#         - for quantile: quantile

#     """
#     # -------------------------
#     # Split filtering (NEW)
#     # -------------------------
#     if "split" not in df_long.columns:
#         raise KeyError("df_long must contain a 'split' column for split filtering.")

#     if splits is not None:
#         splits_list = list(splits)
#         if len(splits_list) == 0:
#             raise ValueError("If provided, splits must be a non-empty list/sequence of split names.")
#     else:
#         splits_list = []
#         if include_test:
#             splits_list.append("test")
#         if include_train_oof:
#             splits_list.append("train_oof")

#         if len(splits_list) == 0:
#             raise ValueError(
#                 "No splits selected. Set include_test/include_train_oof to True, "
#                 "or pass splits=['test', 'train_oof', ...]."
#             )

#     # Filter early so everything downstream (guardrails, grouping) applies to the chosen split(s)
#     d = df_long[df_long["split"].isin(splits_list)].copy()

#     # Optional: fail fast if you filtered everything away
#     if d.empty:
#         present = sorted(df_long["split"].dropna().unique().tolist())
#         raise ValueError(
#             f"After filtering, no rows remain for splits={splits_list}. "
#             f"Splits present in df_long: {present}"
#         )

#     # -------------------------
#     # Infer unit_col if needed
#     # -------------------------
#     if unit_col is None:
#         for cand in ("group", "subject_id", "idx"):
#             if cand in d.columns:
#                 unit_col = cand
#                 break
#         if unit_col is None:
#             raise KeyError("Could not infer unit_col. Please pass unit_col='group' or 'idx' (or another id column).")

#     if unit_col not in d.columns:
#         raise KeyError(f"unit_col='{unit_col}' not found in df_long columns.")

#     # -------------------------
#     # Grouping schemes
#     # -------------------------
#     GROUPING_SCHEMES = {
#         "all_trials": ["model", "variant", "split", unit_col],
#         "per_trial_fold": ["model", "variant", "split", "trial", "outer_fold", unit_col],
#     }
#     if grouping not in GROUPING_SCHEMES:
#         raise ValueError(f"Unknown grouping='{grouping}'. Choose one of {list(GROUPING_SCHEMES)}")

#     group_cols = GROUPING_SCHEMES[grouping]

#     # -------------------------
#     # Validation (use `d`, not df_long)
#     # -------------------------
#     required_base = {"model", "variant", "split", "subject_id", "group_label", "y", "p", unit_col}
#     missing = required_base - set(d.columns)
#     if missing:
#         raise KeyError(f"df_long missing required columns: {sorted(missing)}")

#     need_cols = set(group_cols) - set(d.columns)
#     if need_cols:
#         raise KeyError(f"grouping='{grouping}' requires missing columns: {sorted(need_cols)}")

#     d["p"] = pd.to_numeric(d["p"], errors="coerce")



#     # -------------------------
#     # Output column naming
#     # -------------------------
#     if agg == "quantile":
#         q_tag = int(round(quantile * 100))
#         center_col = f"p_q{q_tag}"
#     elif agg == "softmax":
#         center_col = "p_softmax"
#     else:
#         center_col = {"mean": "p_mean", "median": "p_median", "max": "p_max"}[agg]

#     out_rows: list[dict] = []

#     # Independent-sided winsorization flags
#     apply_low = (lower_q > 0.0)
#     apply_high = (upper_q < 1.0)

#     for keys, gdf in d.groupby(group_cols, sort=False):
#         p = gdf["p"].to_numpy(dtype=float)
#         p = p[~np.isnan(p)]
#         if p.size == 0:
#             continue

#         # Winsorize (optional, per side)
#         if not apply_low and not apply_high:
#             lo = np.nan
#             hi = np.nan
#             p_used = p
#         else:
#             lo = float(np.quantile(p, lower_q)) if apply_low else np.nan
#             hi = float(np.quantile(p, upper_q)) if apply_high else np.nan
#             lo_clip = lo if apply_low else -np.inf
#             hi_clip = hi if apply_high else np.inf
#             p_used = np.clip(p, lo_clip, hi_clip)

#         # Aggregate
#         if agg == "mean":
#             p_center = float(np.mean(p_used))
#         elif agg == "median":
#             p_center = float(np.median(p_used))
#         elif agg == "max":
#             p_center = float(np.max(p_used))
#         elif agg == "quantile":
#             p_center = float(np.quantile(p_used, quantile))
#         else:  # "softmax"
#             p_clip = np.clip(p_used, eps, 1.0 - eps)
#             s = np.log(p_clip) - np.log1p(-p_clip)  # logit(p)
#             t = beta * s
#             t = t - np.max(t)
#             w = np.exp(t)
#             w_sum = np.sum(w)
#             if not np.isfinite(w_sum) or w_sum == 0.0:
#                 p_center = float(np.mean(p_used))
#             else:
#                 w = w / w_sum
#                 p_center = float(np.sum(w * p_used))

#         # Spread (on winsorized values, regardless of agg)
#         p_std = float(np.std(p_used, ddof=ddof))

#         row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
#         row.update(
#             {
#                 "grouping": grouping,
#                 "unit_col": unit_col,
#                 "subject_id": gdf["subject_id"].iloc[0],
#                 "group_label": gdf["group_label"].iloc[0],
#                 "y": int(gdf["y"].iloc[0]),
#                 "n_windows": int(p.size),
#                 center_col: p_center,
#                 "p_total_std": p_std,
#                 "lower_q": float(lower_q),
#                 "upper_q": float(upper_q),
#                 "p_cap_low": lo,
#                 "p_cap_high": hi,
#             }
#         )

#         if agg == "quantile":
#             row["quantile"] = float(quantile)
#         if agg == "softmax":
#             row["beta"] = float(beta)
#             row["eps"] = float(eps)

#         out_rows.append(row)

#     return pd.DataFrame(out_rows)





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


