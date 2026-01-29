# ml_feature_selection.py
# ML feature selection 

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Type, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from phik.phik import phik_from_array
from feature_engine.selection import MRMR
from skrebate import ReliefF, SURF, SURFstar, MultiSURF, MultiSURFstar
from tqdm.auto import trange
from tqdm.auto import tqdm  
from sklearn.datasets import make_classification
from feature_engine.selection import DropConstantFeatures
from joblib import Parallel, delayed
import os
from numpy.typing import NDArray
Bundle = Dict[str, Any]
from copy import deepcopy



def prepare_training_bundle(
    bundle: Bundle,
    n_features: Optional[int] = None,
    keep_features: Optional[Sequence[str]] = None,
    *,
    strict: bool = True,
    dedupe: bool = True,
    copy_bundle: bool = True,
) -> Bundle:
    """
    Return a training-ready bundle with flexible feature selection.

    Selection precedence:
      1) keep_features (exact names)
      2) n_features (first k)
      3) if both None -> return all features (no reduction)

    Args:
      n_features: keep first n feature columns (prefix mode)
      keep_features: keep these feature names (order preserved)
      strict: if True, error on missing keep_features; else drop missing
      dedupe: if True, de-duplicate keep_features while preserving order
      copy_bundle: if True, return shallow copy; if False, may return original bundle when unchanged
    """
    if "X_raw" not in bundle or "feature_names" not in bundle:
        raise KeyError("bundle must contain 'X_raw' and 'feature_names'")

    X: NDArray[np.floating] = bundle["X_raw"]
    feature_names: List[str] = list(bundle["feature_names"])

    if X.ndim != 2:
        raise ValueError(f"X_raw must be 2D, got shape {X.shape}")
    if X.shape[1] != len(feature_names):
        raise ValueError(f"Mismatch: X has {X.shape[1]} cols but feature_names has {len(feature_names)}")

    # If nothing requested: return as-is (or shallow copy)
    if keep_features is None and n_features is None:
        return dict(bundle) if copy_bundle else bundle

    # Avoid ambiguous intent
    if keep_features is not None and n_features is not None:
        raise ValueError("Provide either keep_features OR n_features, not both.")

    out = dict(bundle)  # shallow copy

    # ---- selection by names ----
    if keep_features is not None:
        if len(keep_features) == 0:
            raise ValueError("keep_features must be non-empty")

        if dedupe:
            seen = set()
            keep_features = [n for n in keep_features if not (n in seen or seen.add(n))]

        name_to_idx = {n: i for i, n in enumerate(feature_names)}

        missing = [n for n in keep_features if n not in name_to_idx]
        if missing and strict:
            raise KeyError(f"Requested features not found: {missing[:10]}{'...' if len(missing) > 10 else ''}")

        idxs = [name_to_idx[n] for n in keep_features if n in name_to_idx]
        if len(idxs) == 0:
            raise ValueError("No features selected (all requested features missing).")

        out["X_raw"] = X[:, idxs]
        out["feature_names"] = [feature_names[i] for i in idxs]
        return out

    # ---- selection by prefix (n_features) ----
    if n_features < 0:
        raise ValueError("n_features must be >= 0")

    k = min(n_features, X.shape[1])
    out["X_raw"] = X[:, :k]
    out["feature_names"] = feature_names[:k]
    return out




def drop_constant_features(
    X_raw: np.ndarray,
    feature_names: List[str],
    *,
    mode: str = "both",              # "global" | "local" | "both"
    groups: Optional[np.ndarray] = None,
    tol_global: float = 1.0,
    tol_local: float = 1.0,
    missing_values: str = "raise",
    local_frac: float = 1.0,
    min_group_samples: int = 1,
) -> Dict[str, Any]:
    """
    Drop constant / quasi-constant features using Feature-engine's DropConstantFeatures.

    Parameters
    ----------
    X_raw : np.ndarray
        Shape (n_samples, n_features).
    feature_names : list[str]
        Length n_features; aligned to X_raw columns.
    mode : {"global","local","both"}
        - "global": drop features constant across ALL samples (uses tol_global).
        - "local":  drop features constant within groups in >= local_frac fraction of groups (uses tol_local).
        - "both":   run global first, then local on the remaining features.
    groups : np.ndarray | None
        Required if mode includes "local". Shape (n_samples,).
    tol_global : float
        Passed to DropConstantFeatures in global step. In (0, 1].
    tol_local : float
        Passed to DropConstantFeatures in per-group local step. In (0, 1].
    missing_values : {"raise","ignore","include"}
        Passed to DropConstantFeatures.
    local_frac : float
        In [0, 1]. Local aggregation threshold:
        drop feature if constant in >= local_frac fraction of considered groups.
    min_group_samples : int
        Only groups with at least this many rows are considered in local step.

    Returns (dict)
    --------------
    {
      "X_reduced": np.ndarray,
      "kept_feature_names": list[str],
      "dropped_features": list[str],      # total dropped (global + local)
      "dropped_global": list[str],
      "dropped_local": list[str],
      "n_features_in": int,
      "n_features_out": int,
      "mode": str,
      "groups_used": int | None,          # only for local/both
    }
    """
    # ---- validation ----
    if not isinstance(X_raw, np.ndarray) or X_raw.ndim != 2:
        raise ValueError("X_raw must be a 2D numpy array (n_samples, n_features).")

    n_samples, n_features = X_raw.shape

    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match X_raw columns ({n_features})."
        )

    mode = str(mode).lower().strip()
    if mode not in {"global", "local", "both"}:
        raise ValueError("mode must be one of {'global','local','both'}.")

    if not (0 < tol_global <= 1.0):
        raise ValueError("tol_global must be in (0, 1].")

    if not (0 < tol_local <= 1.0):
        raise ValueError("tol_local must be in (0, 1].")

    if not (0.0 <= local_frac <= 1.0):
        raise ValueError("local_frac must be in [0, 1].")

    if min_group_samples < 1:
        raise ValueError("min_group_samples must be >= 1.")

    needs_local = mode in {"local", "both"}
    if needs_local and groups is None:
        raise ValueError("groups must be provided when mode is 'local' or 'both'.")

    if needs_local:
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != n_samples:
            raise ValueError("groups must be a 1D array with length equal to X_raw.shape[0].")

    # Keep dtype stable (pandas/feature-engine may upcast)
    orig_dtype = X_raw.dtype

    # Work in a DataFrame so we preserve feature names through feature-engine
    X_df = pd.DataFrame(X_raw, columns=feature_names)

    dropped_global: List[str] = []
    dropped_local: List[str] = []
    groups_used: Optional[int] = None

    # ---- global step ----
    if mode in {"global", "both"}:
        dcf = DropConstantFeatures(tol=tol_global, missing_values=missing_values)
        X_df = dcf.fit_transform(X_df)
        dropped_global = list(dcf.features_to_drop_)

    # ---- local step ----
    if needs_local:
        const_counts = pd.Series(0, index=X_df.columns, dtype=int)
        groups_used = 0

        for g in np.unique(groups):
            in_group = (groups == g)
            if in_group.sum() < min_group_samples:
                continue

            groups_used += 1
            dcf_g = DropConstantFeatures(tol=tol_local, missing_values=missing_values)
            dcf_g.fit(X_df.loc[in_group, :])

            dropped_in_g = list(dcf_g.features_to_drop_)
            if dropped_in_g:
                const_counts.loc[dropped_in_g] += 1

        if groups_used > 0:
            frac_constant = const_counts / groups_used
            drop_mask = frac_constant >= float(local_frac)
            dropped_local = const_counts.index[drop_mask].tolist()
            X_df = X_df.loc[:, ~drop_mask]
        else:
            # No eligible groups -> local does nothing
            dropped_local = []

    # ---- assemble results ----
    kept_feature_names = X_df.columns.tolist()
    X_reduced = X_df.to_numpy(dtype=orig_dtype, copy=True)

    # total dropped (ordered, unique)
    dropped_total: List[str] = []
    seen = set()
    for name in dropped_global + dropped_local:
        if name not in seen:
            dropped_total.append(name)
            seen.add(name)

    return {
        "X_reduced": X_reduced,
        "kept_feature_names": kept_feature_names,
        "dropped_features": dropped_total,
        "dropped_global": dropped_global,
        "dropped_local": dropped_local,
        "n_features_in": n_features,
        "n_features_out": len(kept_feature_names),
        "mode": mode,
        "groups_used": groups_used,
    }


def drop_features_by_name(
    X: np.ndarray,
    feature_names: Sequence[str],
    drop_names: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Drop columns from X based on feature names.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Original feature matrix.

    feature_names : sequence of str, length = n_features
        Names of the columns in X, in order.

    drop_names : sequence of str
        Names of features to drop. Any name not found is simply ignored.

    Returns
    -------
    X_new : np.ndarray
        X with the specified columns removed.

    feature_names_new : list of str
        Updated feature names, aligned with columns of X_new.
    """
    feature_names = list(feature_names)
    drop_set = set(drop_names)

    # indices to KEEP (i.e., whose name is not in drop_set)
    keep_indices = [i for i, name in enumerate(feature_names) if name not in drop_set]

    X_new = X[:, keep_indices]
    feature_names_new = [feature_names[i] for i in keep_indices]

    return X_new, feature_names_new
    
    
def sample_one_row_per_group(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly pick ONE row per group.

    Parameters
    ----------
    X : array (n_samples, n_features)
    y : array (n_samples,)
    groups : array (n_samples,)
        Group ID for each sample (e.g. patient ID).
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    X_sub : array (n_groups, n_features)
    y_sub : array (n_groups,)
    groups_sub : array (n_groups,)
        Unique group IDs (one per row in X_sub).
    indices : array (n_groups,)
        Original row indices chosen from X.
    """
    rng = np.random.default_rng(random_state)

    unique_groups = np.unique(groups)
    chosen_indices = []

    for g in unique_groups:
        # all row indices belonging to group g
        idx_g = np.where(groups == g)[0]
        # randomly pick ONE of those indices
        chosen_idx = rng.choice(idx_g)
        chosen_indices.append(chosen_idx)

    chosen_indices = np.array(chosen_indices)

    X_sub = X[chosen_indices, :]
    y_sub = y[chosen_indices]
    groups_sub = groups[chosen_indices]

    return X_sub, y_sub, groups_sub, chosen_indices





def run_feature_selection_pipeline(
    X_raw: np.ndarray,
    feature_names: List[str],
    *,
    groups: Optional[np.ndarray] = None,
    constant_cfg: Optional[Mapping[str, Any]] = None,
    collinearity_cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run feature selection as a small pipeline with grouped configuration dicts.

    This pipeline supports two optional stages:
      (1) Drop constant / quasi-constant features (Feature-engine DropConstantFeatures)
      (2) Drop pairwise-collinear features (correlation threshold pruning)

    Each stage is controlled via a config dict that includes an `enable` flag:
      - constant_cfg["enable"]      -> enable/disable constant-feature dropping
      - collinearity_cfg["enable"]  -> enable/disable collinearity pruning

    If a config is None, the stage is treated as disabled.

    Parameters
    ----------
    X_raw:
        Input feature matrix, shape (n_samples, n_features).

    feature_names:
        List of feature names aligned with columns of X_raw.

    groups:
        Optional grouping vector, shape (n_samples,). Required when:
          - constant_cfg has mode "local" or "both"
          - you want group-aware bootstrapped correlation in collinearity stage

    constant_cfg:
        Dict controlling `drop_constant_features`. If None, stage is disabled.
        Keys (defaults shown):
          - enable=True
          - mode="both"              # "global" | "local" | "both"
          - tol_global=1.0
          - tol_local=1.0
          - missing_values="raise"
          - local_frac=1.0
          - min_group_samples=1

    collinearity_cfg:
        Dict controlling `remove_pairwise_collinear_features`. If None, stage is disabled.
        Keys (defaults shown):
          - enable=True
          - methods_config={"corr": {"method": "spearman", "min_periods": 1, "numeric_only": False}}
          - N=1000
          - threshold=0.8
          - random_state=42
          - parallelize=False
          - n_jobs=-1
          - backend="loky"

    Returns
    -------
    out:
        {
          "X": np.ndarray,
          "feature_names_selected": list[str],
          "logs": {
              "constant_drop": dict | None,
              "collinearity": dict | None
          }
        }
    """

    # -------------------------
    # 0) Normalize configs
    # -------------------------

    if constant_cfg is None:
        constant_cfg_used = {"enable": False}
    else:
        constant_cfg_used = {
            "enable": True,
            "mode": "both",
            "tol_global": 1.0,
            "tol_local": 1.0,
            "missing_values": "raise",
            "local_frac": 1.0,
            "min_group_samples": 1,
            **dict(constant_cfg),
        }

    if collinearity_cfg is None:
        collinearity_cfg_used = {"enable": False}
    else:
        collinearity_cfg_used = {
            "enable": True,
            # UPDATED DEFAULT:
            "methods_config": {"corr": {"method": "spearman", "min_periods": 1, "numeric_only": False}},
            "N": 1000,
            "threshold": 0.8,
            "random_state": 42,
            "parallelize": False,
            "n_jobs": -1,
            "backend": "loky",
            **dict(collinearity_cfg),
        }

    constant_enabled = bool(constant_cfg_used.get("enable", False))
    collinearity_enabled = bool(collinearity_cfg_used.get("enable", False))

    # -------------------------
    # 1) Progress bar setup
    # -------------------------

    n_stages = int(constant_enabled) + int(collinearity_enabled)
    stages = tqdm(total=max(n_stages, 1), desc="Feature selection pipeline", unit="stage")

    # -------------------------
    # 2) Initialize working state
    # -------------------------

    X_current = X_raw
    names_current = list(feature_names)

    logs: Dict[str, Any] = {"constant_drop": None, "collinearity": None}

    # -------------------------
    # 3) Stage 1: Drop constants
    # -------------------------
    if constant_enabled:
        stages.set_description("Feature selection: drop constant features")

        const_drop_results = drop_constant_features(
            X_raw=X_current,
            feature_names=names_current,
            mode=constant_cfg_used["mode"],
            groups=groups,
            tol_global=constant_cfg_used["tol_global"],
            tol_local=constant_cfg_used["tol_local"],
            missing_values=constant_cfg_used["missing_values"],
            local_frac=constant_cfg_used["local_frac"],
            min_group_samples=constant_cfg_used["min_group_samples"],
        )

        X_current = const_drop_results["X_reduced"]
        names_current = const_drop_results["kept_feature_names"]
        logs["constant_drop"] = const_drop_results

        stages.update(1)

    # -------------------------
    # 4) Stage 2: Drop collinear
    # -------------------------
    if collinearity_enabled:
        stages.set_description("Feature selection: drop collinear features")

        X_pruned, kept_names, removed_info = remove_pairwise_collinear_features(
            X=X_current,
            groups=groups,
            methods_config=collinearity_cfg_used["methods_config"],
            N=collinearity_cfg_used["N"],
            feature_names=names_current,
            threshold=collinearity_cfg_used["threshold"],
            random_state=collinearity_cfg_used["random_state"],
            parallelize=collinearity_cfg_used["parallelize"],
            n_jobs=collinearity_cfg_used["n_jobs"],
            backend=collinearity_cfg_used["backend"],
        )

        X_current = X_pruned
        names_current = kept_names

        logs["collinearity"] = {
            "removed_info": removed_info,
            **{k: v for k, v in collinearity_cfg_used.items() if k != "enable"},
        }

        stages.update(1)

    stages.close()

    return {
        "X": X_current,
        "feature_names_selected": names_current,
        "logs": logs,
    }





def compute_vif_for_matrix(
    X: np.ndarray,
    feature_names: Sequence[str],
    vif_cap: float = 1e6,
) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for all features in X.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Design matrix containing only numeric columns.
    feature_names : Sequence[str]
        Names for each column of X. Length must equal n_features.
    vif_cap : float, default=1e6
        Any VIF larger than this (including inf) will be clipped to vif_cap.

    Returns
    -------
    vif_df : pd.DataFrame
        DataFrame with columns ['feature', 'VIF', 'tolerance'], sorted by
        VIF descending and excluding the intercept.
    """

    feature_names_arr = np.asarray(feature_names)


    # Wrap as DataFrame so statsmodels can add a constant and we preserve names
    df = pd.DataFrame(X, columns=feature_names_arr)

    # Drop rows with NA (VIF can't handle missing values)
    df = df.dropna(axis=0)

    # Add intercept column
    X_const = sm.add_constant(df, has_constant="add")

    vif_data: List[Tuple[str, float]] = []
    for i in range(X_const.shape[1]):
        # variance_inflation_factor can emit inf or NaN in degenerate cases
        with np.errstate(divide="ignore", invalid="ignore"):
            vif_val = variance_inflation_factor(X_const.values, i)

        # Clip huge/infinite values for numeric stability
        if vif_val > vif_cap or not np.isfinite(vif_val):
            vif_val = float(vif_cap)

        vif_data.append((str(X_const.columns[i]), float(vif_val)))

    vif_df = pd.DataFrame(vif_data, columns=["feature", "VIF"])

    # Drop the intercept row
    vif_df = vif_df[vif_df["feature"] != "const"].copy()

    # Tolerance is 1 / VIF
    vif_df["tolerance"] = 1.0 / vif_df["VIF"]

    # Sort from largest to smallest VIF (most problematic first)
    vif_df = vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)

    return vif_df


def compute_corr_matrix(
    X: np.ndarray,
    corr_cfg: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """
    Compute a feature-by-feature *absolute* correlation matrix using
    pandas.DataFrame.corr.

    This is intended for collinearity detection, where we care about the
    *strength* of the linear (or rank) relationship, not its direction.
    For example, correlations of +0.9 and -0.9 are treated as equally
    problematic and both map to 0.9 in the returned matrix.

    The result always:
      - uses absolute values of correlation (sign is discarded)
      - has NaNs replaced by 0
      - has a zero diagonal

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix (numeric).
    corr_cfg : Mapping[str, Any] or None
        Optional config dict passed to pandas.DataFrame.corr. Recognized keys:
          - "method": str, default "spearman"
                One of {"pearson", "spearman", "kendall"}.
          - "min_periods": int, default 1
          - "numeric_only": bool, default False

    Returns
    -------
    corr_abs : np.ndarray of shape (n_features, n_features)
        Absolute correlation matrix with zero diagonal. This is typically used
        with a threshold on |corr|, e.g. remove pairs with corr_abs >= 0.8.
    """
    df = pd.DataFrame(X)

    # Unpack configuration with defaults
    cfg = dict(corr_cfg) if corr_cfg is not None else {}
    method: str = cfg.get("method", "spearman")
    min_periods: int = cfg.get("min_periods", 1)
    numeric_only: bool = cfg.get("numeric_only", False)

    corr = df.corr(
        method=method,
        min_periods=min_periods,
        numeric_only=numeric_only,
    ).to_numpy()

    # Replace NaNs with 0, take absolute values (so +/-0.9 are treated the same),
    # and zero out the diagonal
    corr = np.nan_to_num(corr, nan=0.0)
    corr = np.abs(corr)
    np.fill_diagonal(corr, 0.0)

    return corr


def _prepare_collinearity_inputs(
    X: np.ndarray,
    groups: Optional[np.ndarray] = None,
    N: int = 20,
    feature_names: Optional[Sequence[str]] = None,
    threshold: float = 0.8,
) -> bool:
    """
    Validate inputs for remove_collinear_features.

    This function performs only validation (no mutation or defaulting)
    and returns True if all checks pass, otherwise raises an error.
    """
    # ---- X checks ----
    if not isinstance(X, np.ndarray):
        raise TypeError(
            f"X must be a numpy.ndarray, got {type(X).__name__}. "
            "Convert before calling remove_collinear_features."
        )
    if X.ndim != 2:
        raise ValueError(
            f"X must be 2D (n_samples, n_features); got shape {X.shape}."
        )

    n_samples, n_features = X.shape

    # ---- groups checks ----
    if groups is not None and not isinstance(groups, np.ndarray):
        raise TypeError(
            f"groups must be a numpy.ndarray when provided, got {type(groups).__name__}."
        )

    if groups is not None:
        if groups.ndim != 1:
            raise ValueError(
                f"groups must be 1D (n_samples,); got shape {groups.shape}."
            )
        if groups.shape[0] != n_samples:
            raise ValueError(
                f"groups length ({groups.shape[0]}) does not match "
                f"number of rows in X ({n_samples})."
            )

        unique_groups = np.unique(groups)
        if unique_groups.size < 2:
            raise ValueError(
                "groups must contain at least 2 unique group IDs for "
                "group-aware collinearity handling."
            )

        if N < 1:
            raise ValueError(
                f"N must be >= 1 when groups is provided; got {N}."
            )

    # ---- feature_names checks ----
    if feature_names is not None:
        feature_names_arr = np.asarray(feature_names)
        if feature_names_arr.ndim != 1 or feature_names_arr.size != n_features:
            raise ValueError(
                "feature_names must be 1D and have length equal to "
                f"number of columns in X ({n_features})."
            )

    # ---- threshold ----
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"threshold must be between 0 and 1; got {threshold}."
        )

    return True


def _validate_target_array(
    y: np.ndarray,
    n_samples: int,
    name: str = "y",
) -> None:
    """
    Validate a 1D target array against the expected number of samples.

    Parameters
    ----------
    y : np.ndarray
        Target array to validate.
    n_samples : int
        Expected number of samples (rows in X).
    name : str, default="y"
        Name used in error messages.

    Raises
    ------
    TypeError
        If y is not a numpy.ndarray.
    ValueError
        If y is not 1D or if its length does not match n_samples.
    """
    if not isinstance(y, np.ndarray):
        raise TypeError(
            f"{name} must be a numpy.ndarray, got {type(y).__name__}. "
            "Convert to numpy before calling."
        )

    if y.ndim != 1:
        raise ValueError(
            f"{name} must be 1D (n_samples,); got shape {y.shape}."
        )

    if y.shape[0] != n_samples:
        raise ValueError(
            f"{name} length ({y.shape[0]}) does not match number of rows in X ({n_samples})."
        )





def remove_pairwise_collinear_features(
    X: np.ndarray,
    groups: Optional[np.ndarray] = None,
    methods_config: Optional[Mapping[str, Mapping[str, Any]]] = None,
    N: int = 100,
    feature_names: Optional[Sequence[str]] = None,
    threshold: float = 0.8,
    random_state: Optional[int] = 42,

    # parallelization controls
    parallelize: bool = False,
    n_jobs: int = -1,
    backend: str = "loky",
) -> Tuple[np.ndarray, List[str], Dict[int, Dict[str, Any]]]:
    """
    Remove pairwise collinear features based on an absolute correlation threshold.

    This function computes a feature-by-feature absolute correlation matrix
    (optionally in a group-aware fashion, averaged over subsamples), then
    greedily removes later columns in highly correlated pairs while preserving
    the original column order as priority.

    Column priority
    ---------------
    - The current column order in X is treated as priority:
      earlier columns are considered "more important" and are kept when
      conflicts arise (i.e., when a pair has |corr| >= threshold, the
      later column is removed).

    Group-aware behavior
    --------------------
    - If `groups` is None:
          * A single correlation matrix is computed on the full X.
    - If `groups` is provided:
          * A group-aware correlation matrix is built by averaging N
            correlation matrices. For each of the N subsamples, exactly
            one row per group is sampled, a correlation matrix is computed,
            and all N are averaged.

    Parallelization (group-aware mode only)
    ---------------------------------------
    - If `groups` is not None and `parallelize=True`, the N subsample
      correlation computations are run in parallel using joblib:
        Parallel(n_jobs=n_jobs, backend=backend)

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix. MUST be a numpy array.

    groups : np.ndarray of shape (n_samples,), optional
        Group IDs. If provided, used for group-aware processing.
        If None: a single correlation matrix is computed on the full data.

    methods_config : mapping or None, optional
        Configuration for the correlation stage.

        If None, defaults to correlation-only using spearman:
            methods_config = {"corr": {}}

        Recognized top-level keys:
          - "corr": dict with options for the correlation matrix, passed
            through to `compute_corr_matrix` via its `corr_cfg` argument:
                * "method"       (str, default "spearman")
                      One of {"pearson", "spearman", "kendall"}.
                * "min_periods"  (int, default 1)
                * "numeric_only" (bool, default False)

        Any other top-level keys will raise a ValueError.

    N : int, default=100
        Number of group-aware subsamples to average over when `groups`
        is not None. For each subsample, exactly one row per group is
        selected and a correlation matrix is computed; the final matrix
        is the average over N such matrices.

    feature_names : sequence of str or None, optional
        Names for the columns of X. If provided, must have length
        n_features. Used for outputs and the removal log.
        If None, generic names 'f0', 'f1', ... are used.

    threshold : float, default=0.8
        Absolute correlation threshold above which a feature is considered
        pairwise collinear and is removed. That is, if |corr(i, j)| >= threshold,
        then feature j is removed (assuming i < j).

    random_state : int or None, optional, default=42
        Seed for RNG used in group-aware subsampling. When `parallelize=True`,
        seeds are generated up-front for reproducibility.

    parallelize : bool, default=False
        If True and `groups` is not None, compute the N correlation matrices
        in parallel with joblib.

    n_jobs : int, default=-1
        Passed to joblib.Parallel. -1 uses all available cores.

    backend : str, default="loky"
        Passed to joblib.Parallel. "loky" uses process-based parallelism.

    Returns
    -------
    X_pruned : np.ndarray of shape (n_samples, n_features_kept)
        X after correlation-based pairwise collinearity removal.

    kept_feature_names : list of str
        Names of the kept features (after correlation pruning).

    removed_info : dict[int, dict]
        Mapping from removed feature index (w.r.t. original X) to a dict with:
          - 'removed_feature_index' : int
          - 'removed_feature_name'  : str
          - 'reason'                : str (always "corr" for this function)
          - 'kept_feature_index'    : int (index of the feature that was kept)
          - 'kept_feature_name'     : str
          - 'corr_value'            : float (absolute correlation at removal time)
    """
    # -------------------------------------------------------------
    # 0. Validation (raises on error)
    # -------------------------------------------------------------
    _prepare_collinearity_inputs(
        X=X,
        groups=groups,
        N=N,
        feature_names=feature_names,
        threshold=threshold,
    )

    n_samples, n_features = X.shape

    # -------------------------------------------------------------
    # 1. Methods config: correlation ("corr") only
    # -------------------------------------------------------------
    if methods_config is None:
        methods_config = {"corr": {}}
    else:
        methods_config = dict(methods_config)

    allowed_methods = {"corr"}
    unknown_methods = set(methods_config.keys()) - allowed_methods
    if unknown_methods:
        raise ValueError(
            f"Unknown methods in methods_config: {unknown_methods}. "
            f"Allowed: {allowed_methods}."
        )

    corr_cfg = methods_config.get("corr", None)

    # ---- RNG ----
    rng = np.random.default_rng(random_state)

    # -------------------------------------------------------------
    # 2. Feature names
    # -------------------------------------------------------------
    if feature_names is not None:
        feature_names_arr = np.asarray(feature_names, dtype=str)
    else:
        feature_names_arr = np.array([f"f{i}" for i in range(n_features)], dtype=str)

    original_indices = np.arange(n_features)
    X_current = X.copy()
    feature_names_current = feature_names_arr.copy()

    removed_info: Dict[int, Dict[str, Any]] = {}

    # -------------------------------------------------------------
    # 3. Correlation ("corr") stage
    # -------------------------------------------------------------
    if corr_cfg is not None:
        if groups is None:
            corr_abs = compute_corr_matrix(X_current, corr_cfg)

        else:
            n_features_current = X_current.shape[1]
            y_dummy = np.zeros(n_samples, dtype=float)

            if parallelize:
                # IMPORTANT CHANGE:
                # Do NOT build corr_list of length N (that explodes memory for large N).
                # Instead, split seeds into a few chunks, sum correlations per chunk
                # in parallel, then sum the chunk-sums in the main process.

                print('='*100)
                print(">>>> Parallelization active")
                print('='*100)
                seeds = rng.integers(0, 1_000_000, size=N, dtype=np.int64)

                # Estimate number of workers similar to joblib semantics
                cpu = os.cpu_count() or 1
                if n_jobs == -1:
                    n_workers = cpu
                elif n_jobs < -1:
                    n_workers = max(1, cpu + 1 + n_jobs)  # e.g., -2 => cpu-1
                else:
                    n_workers = max(1, int(n_jobs))

                # Keep number of chunks small (≈ number of workers) to bound memory
                n_chunks = min(N, n_workers)
                seed_chunks = np.array_split(seeds, n_chunks)

                p = n_features_current

                def _chunk_sum(seed_chunk: np.ndarray) -> Tuple[np.ndarray, int]:
                    # local accumulation for this chunk
                    corr_sum_local = np.zeros((p, p), dtype=np.float32)
                    for seed_b in seed_chunk:
                        X_sub, _, _, _ = sample_one_row_per_group(
                            X_current, y_dummy, groups, random_state=int(seed_b)
                        )
                        corr_b = compute_corr_matrix(X_sub, corr_cfg).astype(np.float32, copy=False)
                        corr_sum_local += corr_b
                    return corr_sum_local, int(len(seed_chunk))

                partials = Parallel(n_jobs=n_jobs, backend=backend)(
                    delayed(_chunk_sum)(chunk) for chunk in seed_chunks
                )

                corr_sum = np.zeros((p, p), dtype=np.float32)
                count = 0
                for corr_part, c in partials:
                    corr_sum += corr_part
                    count += c

                corr_abs = corr_sum / float(count)

            else:
                corr_sum = np.zeros((n_features_current, n_features_current), dtype=float)

                for _ in trange(N, desc="Bootstrap iterations"):
                    seed_b = int(rng.integers(0, 1_000_000))

                    X_sub, _, _, _ = sample_one_row_per_group(
                        X_current, y_dummy, groups, random_state=seed_b
                    )

                    corr_b = compute_corr_matrix(X_sub, corr_cfg)
                    corr_sum += corr_b

                corr_abs = corr_sum / float(N)

        # Greedy removal using current column order as priority
        n_features_current = X_current.shape[1]
        kept_mask = np.ones(n_features_current, dtype=bool)

        for i in range(n_features_current):
            if not kept_mask[i]:
                continue
            for j in range(i + 1, n_features_current):
                if kept_mask[j] and corr_abs[i, j] >= threshold:
                    kept_mask[j] = False

                    orig_j = int(original_indices[j])
                    orig_i = int(original_indices[i])

                    if orig_j not in removed_info:
                        removed_info[orig_j] = {
                            "removed_feature_index": orig_j,
                            "removed_feature_name": str(feature_names_current[j]),
                            "reason": "corr",
                            "kept_feature_index": orig_i,
                            "kept_feature_name": str(feature_names_current[i]),
                            "corr_value": float(corr_abs[i, j]),
                        }

        X_current = X_current[:, kept_mask]
        feature_names_current = feature_names_current[kept_mask]
        original_indices = original_indices[kept_mask]

    # -------------------------------------------------------------
    # 4. Final outputs
    # -------------------------------------------------------------
    X_pruned = X_current
    kept_feature_names = feature_names_current.astype(str).tolist()

    return X_pruned, kept_feature_names, removed_info


def remove_multicollinearity_features(
    X: np.ndarray,
    groups: Optional[np.ndarray] = None,
    methods_config: Optional[Mapping[str, Mapping[str, Any]]] = None,
    N: int = 100,
    feature_names: Optional[Sequence[str]] = None,
    random_state: Optional[int] = 42,
    # Parallelization controls (grouped mode only)
    parallelize: bool = False,
    n_jobs: int = -1,
    backend: str = "loky",
) -> Dict[str, Any]:
    """
    Remove multicollinear features using iterative Variance Inflation Factor (VIF) elimination.

    What this does
    --------------
    This function performs *stepwise* VIF-based feature pruning:

      1) Compute VIF values for all currently-retained features.
      2) If the worst (largest) VIF exceeds a user-defined threshold,
         remove that single feature.
      3) Repeat until:
           - all remaining features have VIF <= threshold, OR
           - only one feature remains.

    What `N` means in grouped mode (common confusion)
    -------------------------------------------------
    `N` does NOT mean "run the whole algorithm N times".

    If `groups` is provided, each *elimination step* estimates a group-aware VIF
    vector by averaging over `N` bootstrap-like subsamples:

      - For each of N subsamples:
          * sample exactly 1 row per group
          * compute VIFs on that subsample
      - average the N VIF vectors -> avg_vif
      - drop ONE feature with the largest avg_vif (if above threshold)
      - repeat on the reduced feature set

    So total bootstrap/VIF work is roughly:
        (# elimination steps) * (N bootstraps per step).

    Group-aware behavior
    --------------------
    - If `groups` is None:
        Standard VIF is computed using the full dataset X at each elimination step.

    - If `groups` is provided:
        VIF is computed in a group-aware way (average VIF across N subsamples per step).

    Progress reporting
    ------------------
    - Shows an outer progress bar for elimination steps.
    - The progress bar description explicitly states whether `groups` is None or not.

    Parallelization (grouped mode only)
    -----------------------------------
    - If `groups` is not None and `parallelize=True`, the N bootstrap iterations
      *within each elimination step* are parallelized using joblib:
        Parallel(n_jobs=n_jobs, backend=backend)

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.

    groups : np.ndarray of shape (n_samples,), optional
        Group IDs. If provided, grouped bootstrapping is used.

    methods_config : mapping or None, optional
        If None, defaults to {"vif": {}}.
        Supported:
          - "vif": dict with:
              * "threshold" (float, default 5.0)
              * "cap"       (float, default 1e6)

    N : int, default=100
        Number of bootstrap subsamples per elimination step (grouped mode only).

    feature_names : sequence[str] or None
        Names for columns of X; if None, uses f0..f{p-1}.

    random_state : int or None, default=42
        RNG seed for grouped subsampling.

    parallelize : bool, default=False
        If True and groups is not None, parallelize bootstrap iterations per step.

    n_jobs : int, default=-1
        Passed to joblib.Parallel.

    backend : str, default="loky"
        Passed to joblib.Parallel. "loky" is process-based.

    Returns
    -------
    out : dict
        {
          "X": np.ndarray,
              The pruned feature matrix (same n_samples, fewer columns).

          "feature_names_selected": list[str],
              Names of retained features (aligned to columns of "X").

          "logs": dict
              Metadata and removal information, including:
                - "removed_features": dict[int, dict]
                    Keyed by original feature index with:
                      * removed_feature_index
                      * removed_feature_name
                      * reason: "vif" or "vif_grouped"
                      * vif_value: VIF / avg VIF at removal time
                - configuration + summary counts (threshold, cap, N, etc.)
        }
    """
    # -------------------------------------------------------------
    # 0) Validate inputs
    # -------------------------------------------------------------
    _prepare_collinearity_inputs(
        X=X,
        groups=groups,
        N=N,
        feature_names=feature_names,
        threshold=0.0,  # dummy (VIF has its own threshold)
    )

    n_samples, n_features = X.shape

    # -------------------------------------------------------------
    # 1) Parse methods_config (VIF-only)
    # -------------------------------------------------------------
    if methods_config is None:
        methods_config = {"vif": {}}
    else:
        methods_config = dict(methods_config)

    allowed_methods = {"vif"}
    unknown_methods = set(methods_config.keys()) - allowed_methods
    if unknown_methods:
        raise ValueError(
            f"Unknown methods in methods_config: {unknown_methods}. "
            f"Allowed: {allowed_methods}."
        )

    vif_cfg = dict(methods_config.get("vif", {}))
    vif_threshold: float = float(vif_cfg.get("threshold", 5.0))
    vif_cap: float = float(vif_cfg.get("cap", 1e6))

    # RNG (used primarily for grouped-mode seed generation)
    rng = np.random.default_rng(random_state)

    # -------------------------------------------------------------
    # 2) Prepare feature names + bookkeeping for iterative removal
    # -------------------------------------------------------------
    if feature_names is not None:
        feature_names_arr = np.asarray(feature_names, dtype=str)
    else:
        feature_names_arr = np.array([f"f{i}" for i in range(n_features)], dtype=str)

    # Track mapping from current columns back to original indices
    original_indices = np.arange(n_features)
    X_current = X.copy()
    feature_names_current = feature_names_arr.copy()

    # Removal log keyed by original column index
    removed_info: Dict[int, Dict[str, Any]] = {}

    # -------------------------------------------------------------
    # 3) VIF elimination (two modes: groups=None vs grouped)
    # -------------------------------------------------------------
    if X_current.shape[1] > 1:

        # =========================================================
        # A) groups is None: standard full-data VIF each step
        # =========================================================
        if groups is None:
            mode_str = "groups=None (full data)"
            step = 0

            with tqdm(desc=f"VIF elimination steps | {mode_str}", unit="step") as pbar:
                while X_current.shape[1] > 1:
                    # Compute VIFs on FULL data
                    vif_df = compute_vif_for_matrix(
                        X_current,
                        feature_names_current,
                        vif_cap=vif_cap,
                    )

                    # Assumes compute_vif_for_matrix sorts descending by VIF
                    max_row = vif_df.iloc[0]
                    max_vif = float(max_row["VIF"])
                    drop_name = str(max_row["feature"])
                    n_features_left = int(X_current.shape[1])

                    # Stop if worst VIF is acceptable
                    if max_vif <= vif_threshold:
                        pbar.set_postfix_str(
                            f"done | n_features={n_features_left} | max_vif={max_vif:.3f} <= {vif_threshold}"
                        )
                        break

                    # Identify which column to drop
                    drop_idx_local = int(np.where(feature_names_current == drop_name)[0][0])
                    orig_idx = int(original_indices[drop_idx_local])

                    # Log removal
                    if orig_idx not in removed_info:
                        removed_info[orig_idx] = {
                            "removed_feature_index": orig_idx,
                            "removed_feature_name": drop_name,
                            "reason": "vif",
                            "vif_value": max_vif,
                        }

                    # Update progress display (before dropping)
                    step += 1
                    pbar.set_postfix_str(
                        f"step={step} | drop={drop_name} | vif={max_vif:.3f} | n_features={n_features_left}"
                    )

                    # Drop column
                    keep_local = np.ones(n_features_left, dtype=bool)
                    keep_local[drop_idx_local] = False

                    X_current = X_current[:, keep_local]
                    feature_names_current = feature_names_current[keep_local]
                    original_indices = original_indices[keep_local]

                    pbar.update(1)

        # =========================================================
        # B) groups provided: group-aware avg VIF each step
        # =========================================================
        else:
            n_groups = int(len(np.unique(groups)))
            par_str = f"parallel={parallelize} ({backend})" if parallelize else "parallel=False"
            mode_str = f"groups!=None (group-aware, N={N}, n_groups={n_groups}, {par_str})"

            # Dummy y for sample_one_row_per_group signature
            y_dummy = np.zeros(n_samples, dtype=float)

            step = 0
            with tqdm(desc=f"VIF elimination steps | {mode_str}", unit="step") as pbar:
                while X_current.shape[1] > 1:
                    n_features_left = int(X_current.shape[1])

                    # Pre-generate seeds so sequential and parallel are reproducible
                    seeds = rng.integers(0, 1_000_000, size=N, dtype=np.int64)

                    def _one_boot(seed_n: int) -> np.ndarray:
                        # 1) sample one row per group
                        X_sub, _, _, _ = sample_one_row_per_group(
                            X_current, y_dummy, groups, random_state=int(seed_n)
                        )

                        # 2) compute VIFs on this subsample
                        vif_df_b = compute_vif_for_matrix(
                            X_sub,
                            feature_names_current,
                            vif_cap=vif_cap,
                        )

                        # 3) align VIF vector to current feature order
                        vif_map = dict(zip(vif_df_b["feature"], vif_df_b["VIF"]))
                        return np.array(
                            [float(vif_map[name]) for name in feature_names_current],
                            dtype=float,
                        )

                    # ---- Compute avg_vif either in parallel or sequentially ----
                    if parallelize:
                        boot_vecs = Parallel(n_jobs=n_jobs, backend=backend)(
                            delayed(_one_boot)(int(s)) for s in seeds
                        )
                        # Average across bootstraps (shape: (N, p) -> (p,))
                        avg_vif = np.mean(np.vstack(boot_vecs), axis=0)
                    else:
                        avg_vif = np.zeros(n_features_left, dtype=float)
                        for s in seeds:
                            avg_vif += _one_boot(int(s))
                        avg_vif /= float(N)

                    # Pick worst feature by average VIF
                    j_star = int(np.argmax(avg_vif))
                    max_avg_vif = float(avg_vif[j_star])
                    drop_name = str(feature_names_current[j_star])

                    # Stop condition
                    if max_avg_vif <= vif_threshold:
                        pbar.set_postfix_str(
                            f"done | n_features={n_features_left} | max_avg_vif={max_avg_vif:.3f} <= {vif_threshold}"
                        )
                        break

                    # Log removal using original index
                    orig_idx = int(original_indices[j_star])
                    if orig_idx not in removed_info:
                        removed_info[orig_idx] = {
                            "removed_feature_index": orig_idx,
                            "removed_feature_name": drop_name,
                            "reason": "vif_grouped",
                            "vif_value": max_avg_vif,
                        }

                    # Update progress display (before dropping)
                    step += 1
                    pbar.set_postfix_str(
                        f"step={step} | drop={drop_name} | avg_vif={max_avg_vif:.3f} | n_features={n_features_left}"
                    )

                    # Drop column
                    keep_local = np.ones(n_features_left, dtype=bool)
                    keep_local[j_star] = False

                    X_current = X_current[:, keep_local]
                    feature_names_current = feature_names_current[keep_local]
                    original_indices = original_indices[keep_local]

                    pbar.update(1)

    # -------------------------------------------------------------
    # 4) Pack outputs into a single dict
    # -------------------------------------------------------------
    kept_feature_names = feature_names_current.astype(str).tolist()

    logs: Dict[str, Any] = {
        "removed_features": removed_info,
        "mode": "grouped" if groups is not None else "ungrouped",
        "vif_threshold": vif_threshold,
        "vif_cap": vif_cap,
        "N": int(N),
        "random_state": random_state,
        "parallelize": bool(parallelize),
        "n_jobs": int(n_jobs),
        "backend": str(backend),
        "n_features_start": int(n_features),
        "n_features_selected": int(X_current.shape[1]),
        "n_features_removed": int(n_features - X_current.shape[1]),
    }

    return {
        "X": X_current,
        "feature_names_selected": kept_feature_names,
        "logs": logs,
    }

# def remove_multicollinearity_features(
#     X: np.ndarray,
#     groups: Optional[np.ndarray] = None,
#     methods_config: Optional[Mapping[str, Mapping[str, Any]]] = None,
#     N: int = 100,
#     feature_names: Optional[Sequence[str]] = None,
#     random_state: Optional[int] = 42,
# ) -> Tuple[np.ndarray, List[str], Dict[int, Dict[str, Any]]]:
#     """
#     Remove multicollinear features using Variance Inflation Factor (VIF).

#     This function performs iterative VIF-based elimination:

#       - At each step, it computes VIFs for all remaining features.
#       - If the largest VIF exceeds a specified threshold, the corresponding
#         feature is removed.
#       - The process repeats until all remaining features have VIF below
#         the threshold or only one feature is left.

#     Group-aware behavior
#     --------------------
#     - If `groups` is None:
#           * Standard VIF is computed on the full X at each iteration.
#     - If `groups` is provided:
#           * A group-aware VIF vector is built by averaging N VIF vectors.
#             For each of the N subsamples, exactly one row per group is
#             sampled, VIFs are computed, and all N vectors are averaged.

#     Parameters
#     ----------
#     X : np.ndarray of shape (n_samples, n_features)
#         Feature matrix. MUST be a numpy array.

#     groups : np.ndarray of shape (n_samples,), optional
#         Group IDs. If provided, used for group-aware processing.
#         If None: VIF is computed on the full data at each iteration.

#     methods_config : mapping or None, optional
#         Configuration for the VIF stage.

#         If None, defaults to:
#             methods_config = {"vif": {}}

#         Recognized top-level keys:
#           - "vif": dict with options for the VIF stage:
#                 * "threshold"  (float, default 10.0)
#                       VIF cutoff above which a feature is considered
#                       problematic and subject to removal.
#                 * "cap"        (float, default 1e6)
#                       Any VIF larger than this (including inf) will be
#                       clipped to this value in the computation.

#         Any other top-level keys will raise a ValueError.

#     N : int, default=100
#         Number of group-aware subsamples to average over when `groups`
#         is not None. For each subsample, exactly one row per group is
#         selected and a VIF vector is computed; the final VIF vector at
#         each iteration is the average over N such vectors.

#     feature_names : sequence of str or None, optional
#         Names for the columns of X. If provided, must have length
#         n_features. Used for output names and the removal log.
#         If None, generic names 'f0', 'f1', ... are used.

#     random_state : int or None, optional
#         Seed for RNG used in group-aware subsampling.

#     Returns
#     -------
#     X_pruned : np.ndarray of shape (n_samples, n_features_kept)
#         X after VIF-based multicollinearity removal.

#     kept_feature_names : list of str
#         Names of the kept features (after VIF pruning).

#     removed_info : dict[int, dict]
#         Mapping from removed feature index (w.r.t. original X) to a dict with:
#           - 'removed_feature_index' : int
#           - 'removed_feature_name'  : str
#           - 'reason'                : str
#                 * "vif"         (standard VIF on full data)
#                 * "vif_grouped" (group-aware averaged VIF)
#           - 'vif_value'            : float
#                 The VIF (or averaged VIF, in grouped mode) at the time of removal.
#     """
#     # -------------------------------------------------------------
#     # 0. Validation (raises on error)
#     # -------------------------------------------------------------
#     # We reuse the same validation helper; the 'threshold' argument is
#     # only used for bounds checking (0 <= threshold <= 1), so we pass
#     # a dummy 0.0 here since VIF has its own threshold in methods_config.
#     _prepare_collinearity_inputs(
#         X=X,
#         groups=groups,
#         N=N,
#         feature_names=feature_names,
#         threshold=0.0,
#     )

#     n_samples, n_features = X.shape

#     # -------------------------------------------------------------
#     # 1. Methods config: VIF ("vif") only
#     # -------------------------------------------------------------
#     if methods_config is None:
#         # default: VIF-only with default thresholds
#         methods_config = {"vif": {}}
#     else:
#         # make a shallow copy to avoid mutating caller's dict
#         methods_config = dict(methods_config)

#     allowed_methods = {"vif"}
#     unknown_methods = set(methods_config.keys()) - allowed_methods
#     if unknown_methods:
#         raise ValueError(
#             f"Unknown methods in methods_config: {unknown_methods}. "
#             f"Allowed: {allowed_methods}."
#         )

#     vif_cfg = dict(methods_config.get("vif", {}))
#     vif_threshold: float = float(vif_cfg.get("threshold", 5.0))
#     vif_cap: float = float(vif_cfg.get("cap", 1e6))

#     # ---- RNG ----
#     rng = np.random.default_rng(random_state)

#     # -------------------------------------------------------------
#     # 2. Feature names: apply defaults and normalize to np.ndarray
#     # -------------------------------------------------------------
#     if feature_names is not None:
#         feature_names_arr = np.asarray(feature_names, dtype=str)
#     else:
#         feature_names_arr = np.array(
#             [f"f{i}" for i in range(n_features)], dtype=str
#         )

#     # Track mapping from current columns back to original indices
#     original_indices = np.arange(n_features)
#     X_current = X.copy()
#     feature_names_current = feature_names_arr.copy()

#     # Removal log: keys are original column indices
#     removed_info: Dict[int, Dict[str, Any]] = {}

#     # -------------------------------------------------------------
#     # 3. VIF stage (non-grouped or group-aware)
#     # -------------------------------------------------------------
#     if X_current.shape[1] > 1:
#         if groups is None:
#             # ---- Simple VIF (no groups): standard iterative removal ----
#             while True:
#                 vif_df = compute_vif_for_matrix(
#                     X_current,
#                     feature_names_current,
#                     vif_cap=vif_cap,
#                 )
#                 max_row = vif_df.iloc[0]  # sorted descending by VIF
#                 max_vif = float(max_row["VIF"])

#                 # Stop if all VIFs are acceptable
#                 if max_vif <= vif_threshold:
#                     break

#                 drop_name = str(max_row["feature"])
#                 # location of this feature in current arrays
#                 drop_idx_local = int(
#                     np.where(feature_names_current == drop_name)[0][0]
#                 )
#                 orig_idx = int(original_indices[drop_idx_local])

#                 if orig_idx not in removed_info:
#                     removed_info[orig_idx] = {
#                         "removed_feature_index": orig_idx,
#                         "removed_feature_name": drop_name,
#                         "reason": "vif",
#                         "vif_value": max_vif,
#                     }

#                 keep_local = np.ones(X_current.shape[1], dtype=bool)
#                 keep_local[drop_idx_local] = False

#                 X_current = X_current[:, keep_local]
#                 feature_names_current = feature_names_current[keep_local]
#                 original_indices = original_indices[keep_local]

#                 if X_current.shape[1] <= 1:
#                     break
#         else:
#             # ---- Group-aware VIF: N subsamples per elimination step ----
#             y_dummy = np.zeros(n_samples, dtype=float)

#             while X_current.shape[1] > 1:
#                 p_curr = X_current.shape[1]
#                 avg_vif = np.zeros(p_curr, dtype=float)

#                 # Inner loop: average VIF across N subsamples
#                 for n in trange(N, desc="Bootstrap iterations (VIF)"):
#                     seed_b = int(rng.integers(0, 1_000_000))
#                     X_sub, _, _, _ = sample_one_row_per_group(
#                         X_current, y_dummy, groups, random_state=seed_b
#                     )

#                     vif_df_b = compute_vif_for_matrix(
#                         X_sub,
#                         feature_names_current,
#                         vif_cap=vif_cap,
#                     )

#                     # Map VIFs back to current feature order
#                     vif_map = dict(zip(vif_df_b["feature"], vif_df_b["VIF"]))
#                     vif_vec = np.array(
#                         [float(vif_map[name]) for name in feature_names_current],
#                         dtype=float,
#                     )
#                     avg_vif += vif_vec

#                 # Average across N subsamples
#                 avg_vif /= float(N)

#                 # Select the feature with the worst (largest) average VIF
#                 j_star = int(np.argmax(avg_vif))
#                 max_avg_vif = float(avg_vif[j_star])

#                 # If all are under threshold, stop
#                 if max_avg_vif <= vif_threshold:
#                     break

#                 drop_name = str(feature_names_current[j_star])
#                 orig_idx = int(original_indices[j_star])

#                 if orig_idx not in removed_info:
#                     removed_info[orig_idx] = {
#                         "removed_feature_index": orig_idx,
#                         "removed_feature_name": drop_name,
#                         "reason": "vif_grouped",
#                         "vif_value": max_avg_vif,
#                     }

#                 keep_local = np.ones(X_current.shape[1], dtype=bool)
#                 keep_local[j_star] = False

#                 X_current = X_current[:, keep_local]
#                 feature_names_current = feature_names_current[keep_local]
#                 original_indices = original_indices[keep_local]

#                 if X_current.shape[1] <= 1:
#                     break

#     # -------------------------------------------------------------
#     # 4. Final outputs
#     # -------------------------------------------------------------
#     X_pruned = X_current
#     kept_feature_names = feature_names_current.astype(str).tolist()

#     return X_pruned, kept_feature_names, removed_info


def plot_aggregated_feature_ranking_barplot(
    agg_results: Dict[str, pd.DataFrame],
    top_k: int = 10,
    figsize: Tuple[float, float] = (10.0, 4.0),
    title: str = "Top Aggregated Features",
) -> None:
    """
    Plot a styled barplot of aggregated feature rankings.

    Parameters
    ----------
    agg_results : dict
        Output from aggregate_feature_rankings, expected to be:
            {"final_feat_ranking": pd.DataFrame(...)}
        The DataFrame must have at least:
            - 'feature'
            - 'agg_mean_rank'
        Optionally:
            - 'feature_name'

    top_k : int, default=10
        Number of top features to display (based on lowest agg_mean_rank).

    figsize : tuple, default=(10, 4)
        Matplotlib figure size.

    title : str, default="Top Aggregated Features"
        Plot title.
    """
    if "final_feat_ranking" not in agg_results:
        raise KeyError(
            "agg_results must contain key 'final_feat_ranking'. "
            f"Available keys: {list(agg_results.keys())}"
        )

    df = agg_results["final_feat_ranking"].copy()

    required_cols = {"feature", "agg_mean_rank"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Ranking DataFrame is missing required columns: {sorted(missing)}"
        )

    # Use feature_name if present, otherwise fall back to index as string
    if "feature_name" not in df.columns:
        df["feature_name"] = df["feature"].astype(str)

    # Sort by agg_mean_rank (best first) and keep top_k
    df = df.sort_values("agg_mean_rank", ascending=True).head(top_k)

    # Preserve this order in the plot
    df["feature_name"] = pd.Categorical(
        df["feature_name"],
        categories=df["feature_name"],
        ordered=True,
    )

    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=df,
        x="feature_name",
        y="agg_mean_rank",
        ax=ax,
        color="brown",      # bar fill color
        edgecolor="black",    # bar border color
        linewidth=1.2,        # border thickness
    )

    ax.set_xlabel("Feature")
    ax.set_ylabel("Aggregated mean rank (lower is better)")
    ax.set_title(title)

    # Light y-grid for readability
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Rotate x labels and right-align them
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Clean up spines a bit
    sns.despine(ax=ax, left=False, bottom=False)

    plt.tight_layout()
    plt.show()


    


def compute_phik_per_feature(X_sub: np.ndarray, y_sub: np.ndarray) -> np.ndarray:
    """
    Compute PhiK correlation between each feature column in X_sub and the target y_sub.

    Parameters
    ----------
    X_sub : np.ndarray of shape (n_samples_sub, n_features)
        Subsampled feature matrix.
    y_sub : np.ndarray of shape (n_samples_sub,)
        Target labels for the subsampled rows.

    Returns
    -------
    np.ndarray of shape (n_features,)
        PhiK score for each feature (one value per column in X_sub).
    """
    # Unpack the shape for clarity: number of samples and number of features
    n_samples_sub, n_features = X_sub.shape

    # Allocate an array to store one PhiK score per feature
    phik_scores = np.zeros(n_features, dtype=float)

    # Loop over each feature column in X_sub
    for j in range(n_features):
        # Extract j-th feature as a 1D array of length n_samples_sub
        x_col = X_sub[:, j]

        # Compute PhiK between this feature and the target.
        # num_vars=['x'] tells phik_from_array which variable is numeric;
        # y_sub is treated as the other variable by default.
        phi_k = phik_from_array(x_col, y_sub, num_vars=['x'])

        # Store the PhiK score in the corresponding position
        phik_scores[j] = phi_k

    # Return the vector of PhiK scores, aligned with feature indices 0..n_features-1
    return phik_scores


def compute_feature_scores(
    method_name: str,
    X_sub: np.ndarray,
    y_sub: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    """
    Compute a 1D score per feature for a given feature selection method.

    This function is the central "dispatcher" for all feature selection methods
    used in the bootstrap pipeline. It takes a method name, a subsampled
    (X_sub, y_sub), and a config dict with method-specific parameters, and
    returns a 1D numpy array of scores (one score per feature).

    Parameters
    ----------
    method_name : str
        Name/label of the method.
        Examples:
          - "phik"
          - "mrmr_FCQ"
          - "mrmr_MIQ"
        The label is used for routing:
          - any name starting with "phik" -> PhiK per feature
          - any name starting with "mrmr" -> MRMR (any variant: FCQ, FCD, MID, MIQ)

    X_sub : array-like of shape (n_samples_sub, n_features)
        Subsampled feature matrix (e.g. one row per group for a given bootstrap).
        Each column is a feature; each row is one sample in this bootstrap iteration.

    y_sub : array-like of shape (n_samples_sub,)
        Labels corresponding to the rows of X_sub.

    config : dict
        Method-specific configuration. Typical keys:
          - For MRMR:
              {
                "mrmr_method": "MIQ",   # 'MID', 'MIQ', 'FCD', or 'FCQ'
                "regression": False,    # True for regression targets, False for classification
                "discrete_features": [... optional list of bools ...]
              }
        If "discrete_features" is not provided for MRMR, we will assume that
        all features are continuous and set it to [False] * n_features.

    Returns
    -------
    scores : np.ndarray of shape (n_features,)
        One score per feature, aligned with the columns of X_sub.
        The calling code will convert these scores into ranks.
    """

    # Convert X_sub to a numpy array to guarantee consistent shape operations.
    X_sub = np.asarray(X_sub)

    # Convert y_sub to a numpy array to guarantee consistent shape operations.
    y_sub = np.asarray(y_sub)

    # Extract the number of features (columns) from X_sub.
    n_features = X_sub.shape[1]

    # Normalize the method name to lowercase for routing logic (case-insensitive).
    lower_name = method_name.lower()

    # -------------------------------------------------------------------------
    # Handle PHIK-based feature scoring
    # -------------------------------------------------------------------------
    # If the method name starts with "phik", we use PhiK correlation
    # to compute one score per feature.
    if lower_name.startswith("phik"):
        # Compute PhiK per feature vs y_sub.
        raw_scores = compute_phik_per_feature(X_sub, y_sub)

    # -------------------------------------------------------------------------
    # Handle MRMR-based feature scoring (all MRMR variants)
    # -------------------------------------------------------------------------
    # If the method name starts with "mrmr", we use the MRMR selector
    # from feature_engine, with the specific "method" and other parameters
    # provided via the config dict.
    elif lower_name.startswith("mrmr"):
        # Read the MRMR variant ("FCQ", "FCD", "MID", "MIQ") from config;
        # default to "FCQ" if not provided.
        mrmr_method = config.get("mrmr_method", "FCQ")

        # Read the regression flag from config; default is False (i.e. classification).
        regression = config.get("regression", False)

        # Extract discrete_features from config if provided; otherwise, assume
        # that all features are continuous (False for every feature).
        discrete_features = config.get("discrete_features", None)
        if discrete_features is None:
            # Assume all continuous if user did not specify discrete_features.
            discrete_features = [False] * n_features

        # Build a list of synthetic feature names for the DataFrame columns.
        feature_names = [f"f{i}" for i in range(n_features)]

        # Wrap X_sub into a DataFrame so that MRMR can use the column names.
        X_df = pd.DataFrame(X_sub, columns=feature_names)

        # Instantiate the MRMR selector with:
        #  - method = chosen MRMR variant ("FCQ", "FCD", "MID", "MIQ")
        #  - regression flag (True/False)
        #  - discrete_features to differentiate continuous vs discrete variables
        #  - max_features = n_features so we get a relevance score for *every* feature.
        sel = MRMR(
            method=mrmr_method,
            regression=regression,
            discrete_features=discrete_features,
            max_features=n_features,
        )

        # Fit the MRMR selector on the subsampled data.
        sel.fit(X_df, y_sub)

        # Create a Series with:
        #  - index = variable names used by MRMR
        #  - values = relevance scores produced by MRMR
        relevance_ser = pd.Series(
            sel.relevance_,
            index=sel.variables_,
            name="relevance"
        )

        # Reindex the relevance Series to the original feature_names order
        # so that the scores align with the columns in X_sub.
        relevance_ser = relevance_ser.reindex(feature_names)

        # Extract the underlying numpy array of relevance scores.
        raw_scores = relevance_ser.values


    # -------------------------------------------------------------------------
    # Handle scikit-rebate (skrebate) feature scoring
    # -------------------------------------------------------------------------
    # Supported: ReliefF, SURF, SURFstar, MultiSURF, MultiSURFstar
    elif lower_name in {"relieff", "surf", "surfstar", "multisurf", "multisurfstar"}:
        # Copy params from methods_config
        params = dict(config)

        # Pick the correct estimator class
        if lower_name == "relieff":
            selector = ReliefF(**params)
        elif lower_name == "surf":
            selector = SURF(**params)
        elif lower_name == "surfstar":
            selector = SURFstar(**params)
        elif lower_name == "multisurf":
            selector = MultiSURF(**params)
        elif lower_name == "multisurfstar":
            selector = MultiSURFstar(**params)
        else:
            raise ValueError(f"Unsupported skrebate method: {method_name}")

        # Fit and pull per-feature importance scores
        selector.fit(X_sub, y_sub)
        raw_scores = selector.feature_importances_
        
 
    # -------------------------------------------------------------------------
    # Handle unknown method names
    # -------------------------------------------------------------------------
    # If we reach this else branch, the method_name does not match any
    # of the supported prefixes ("phik", "mrmr"), so we raise an error.
    else:
        # Raise an informative error to signal that this method must be added.
        raise ValueError(
            f"Unknown method '{method_name}'. "
            f"Add it to compute_feature_scores()."
        )

    # -------------------------------------------------------------------------
    # Standardize and validate the output scores
    # -------------------------------------------------------------------------
    # Convert raw_scores to a 1D numpy array to ensure consistent shape.
    scores = np.asarray(raw_scores).ravel()

    # Check that the number of scores matches the number of features.
    # This is critical: we must have exactly one score per feature.
    if scores.shape[0] != n_features:
        # Raise an informative error indicating the mismatch.
        raise ValueError(
            f"Method '{method_name}' returned {scores.shape[0]} scores, "
            f"but X_sub has {n_features} features."
        )

    # Return the validated 1D scores array to the caller.
    return scores

def build_feature_name_map(
    feature_names: Union[Dict[int, str], Sequence[str], np.ndarray]
) -> Dict[int, str]:
    """
    Given a list/array of feature names, return a dict mapping:
        index -> feature_name

    If feature_names is already a dict, it's assumed to be that mapping
    and is returned as-is.
    """
    if isinstance(feature_names, dict):
        return feature_names
    return {i: name for i, name in enumerate(feature_names)}





def aggregate_feature_rankings(
    results: Dict[str, pd.DataFrame],
    top_k: Optional[int] = None,
    feature_names: Optional[Union[Sequence[str], np.ndarray, Dict[int, str]]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate feature rankings from multiple methods into a single final ranking
    by computing the overall mean rank per feature across methods, then
    optionally returning only the top_k features.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        Output from bootstrap_feature_mean_rank.
        Keys are like "<method_name>_feat_ranking".
        Each DataFrame must have at least:
            - 'feature'   : feature index
            - 'mean_rank' : mean rank for that method
        Typically each DataFrame contains ALL features.

    top_k : int or None, optional
        If None:
            Return ALL features in the final ranking.
        If int:
            After aggregating and sorting, keep only the first `top_k` rows
            (best features).

    feature_names : list/array or dict, optional
        If list/array:
            Assumed to be ordered so that index i corresponds to feature_names[i].
        If dict:
            Assumed to map integer feature index -> feature name.
        If None:
            No feature_name column is added to the final result.

    Returns
    -------
    final_result : dict
        {
          "final_feat_ranking": pd.DataFrame
        }

        The DataFrame has columns:
            - 'feature'         : feature index
            - 'n_methods'       : how many methods contributed a mean_rank
            - 'agg_mean_rank'   : average of mean_rank across methods
            - 'feature_name'    : (optional) human-readable feature name
        Sorted by 'agg_mean_rank' ascending (best features at the top),
        then truncated to top_k if specified.
    """

    # ------------------------------------------------------------------
    # Collect per-feature mean_rank values across methods
    # ------------------------------------------------------------------
    # feature_index -> list of mean_rank values (one per method)
    feature_ranks = {}

    for key, df in results.items():
        # Use ALL rows in each method's DataFrame
        for _, row in df.iterrows():
            feat = int(row["feature"])
            mr   = float(row["mean_rank"])

            if feat not in feature_ranks:
                feature_ranks[feat] = []
            feature_ranks[feat].append(mr)

    # ------------------------------------------------------------------
    # Build an aggregated DataFrame from the collected ranks
    # ------------------------------------------------------------------
    records = []

    for feat, ranks_list in feature_ranks.items():
        n_methods = len(ranks_list)
        agg_mean_rank = float(np.mean(ranks_list))

        records.append(
            {
                "feature": feat,
                "n_methods": n_methods,
                "agg_mean_rank": agg_mean_rank,
            }
        )

    df_final = pd.DataFrame(records)

    # Sort by overall mean rank (best = smallest), tie-break by feature index
    df_final = df_final.sort_values(
        by=["agg_mean_rank", "feature"],
        ascending=[True, True],
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Optionally truncate to top_k
    # ------------------------------------------------------------------
    if top_k is not None:
        df_final = df_final.head(top_k)

    # ------------------------------------------------------------------
    # Optionally attach a human-readable feature_name column
    # ------------------------------------------------------------------
    if feature_names is not None:
        feature_name_map = build_feature_name_map(feature_names)
        df_final["feature_name"] = df_final["feature"].map(feature_name_map)

    return {"final_feat_ranking": df_final}



def construct_X_from_ranked_features(
    X: Union[np.ndarray, pd.DataFrame],
    agg_results: Dict[str, pd.DataFrame],
    feature_names: Optional[Union[Sequence[str], np.ndarray]] = None,
    ranking_key: str = "final_feat_ranking",
) -> Tuple[Union[np.ndarray, pd.DataFrame], List[str], np.ndarray]:
    
    """
    Construct a reduced feature matrix X based on a ranked feature list.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or pd.DataFrame
        Original feature matrix.

    agg_results : dict
        Output from aggregate_feature_rankings, e.g.:
            {
              "final_feat_ranking": pd.DataFrame(...)
            }
        The DataFrame must have at least columns:
            - 'feature'       : integer column indices
            - 'n_methods'     : number of methods contributing to this feature's rank
            - 'agg_mean_rank' : aggregated mean rank across methods
        Optionally:
            - 'feature_name'  : human-readable feature name

    feature_names : list/array or None, optional
        Original feature names aligned with columns of X.
        If None and X is a DataFrame, X.columns are used.
        If None and X is a numpy array, generic names like "f0", "f1", ...
        are created for the selected features.

    ranking_key : str, default="final_feat_ranking"
        Key in agg_results that contains the ranking DataFrame.

    Returns
    -------
    X : same type as X
        X restricted to the ranked features, in the same order as in the
        ranking DataFrame (best feature first).

    selected_feature_names : list of str
        Names of the selected features, aligned with X columns.

    selected_feature_indices : np.ndarray of int
        Original column indices (0..n_features-1) of the selected features.
    """

    # ------------------------------------------------------------------
    # 1. Validate and standardize X
    # ------------------------------------------------------------------
    is_dataframe = isinstance(X, pd.DataFrame)

    if is_dataframe:
        n_samples, n_features = X.shape
        if n_features == 0:
            raise ValueError("X has zero features (no columns).")
        if n_samples == 0:
            raise ValueError("X has zero samples (no rows).")
    else:
        X = np.asarray(X)
        if X.ndim != 2:
            raise TypeError(
                f"X must be a 2D array or DataFrame; got array with shape {X.shape}."
            )
        n_samples, n_features = X.shape
        if n_features == 0:
            raise ValueError("X has zero features (no columns).")
        if n_samples == 0:
            raise ValueError("X has zero samples (no rows).")

    # ------------------------------------------------------------------
    # 2. Validate agg_results and ranking DataFrame
    # ------------------------------------------------------------------
    if not isinstance(agg_results, dict):
        raise TypeError(
            f"agg_results must be a dict-like object from aggregate_feature_rankings; "
            f"got {type(agg_results).__name__}."
        )

    if ranking_key not in agg_results:
        raise KeyError(
            f"agg_results does not contain key '{ranking_key}'. "
            f"Available keys: {list(agg_results.keys())}"
        )

    df_rank = agg_results[ranking_key]

    if not isinstance(df_rank, pd.DataFrame):
        raise TypeError(
            f"agg_results['{ranking_key}'] must be a pandas DataFrame; "
            f"got {type(df_rank).__name__}."
        )

    if df_rank.shape[0] == 0:
        raise ValueError(
            f"Ranking DataFrame agg_results['{ranking_key}'] is empty (no ranked features)."
        )

    # --- NEW: check expected columns ---
    required_cols = {"feature", "n_methods", "agg_mean_rank"}
    optional_cols = {"feature_name"}

    missing_required = required_cols - set(df_rank.columns)
    if missing_required:
        raise ValueError(
            f"Ranking DataFrame agg_results['{ranking_key}'] is missing required "
            f"columns: {sorted(missing_required)}. "
            f"Expected at least: {sorted(required_cols)}."
        )

    # (We don’t *require* feature_name, but we can note if it's missing.)
    # If you want to be strict, you could also enforce optional_cols ⊆ columns.

    feat_vals = df_rank["feature"].values

    # Check for missing values
    if pd.isna(feat_vals).any():
        raise ValueError(
            "'feature' column in ranking DataFrame contains missing values."
        )

    # Attempt to cast to integer indices safely
    try:
        indices_int = feat_vals.astype(int)
    except (TypeError, ValueError):
        raise ValueError(
            "'feature' column in ranking DataFrame must contain integer-like values."
        )

    # If original values are floats, ensure they are integer-valued
    if np.issubdtype(feat_vals.dtype, np.floating):
        if not np.allclose(feat_vals, indices_int, rtol=0, atol=0):
            raise ValueError(
                "'feature' column contains non-integer values that cannot be safely "
                "interpreted as column indices."
            )

    selected_feature_indices = indices_int

    if selected_feature_indices.size == 0:
        raise ValueError("No feature indices found in ranking DataFrame.")

    # Check for duplicate indices
    if len(np.unique(selected_feature_indices)) != len(selected_feature_indices):
        raise ValueError(
            "Ranking DataFrame contains duplicate feature indices in 'feature' column."
        )

    # Bounds check
    if selected_feature_indices.min() < 0 or selected_feature_indices.max() >= n_features:
        raise ValueError(
            f"Ranking refers to feature index outside valid range [0, {n_features - 1}]. "
            f"Got indices in range [{selected_feature_indices.min()}, "
            f"{selected_feature_indices.max()}]."
        )

    # ------------------------------------------------------------------
    # 3. Validate / derive feature_names
    # ------------------------------------------------------------------
    if feature_names is not None:
        feature_names_arr = np.asarray(feature_names)
        if feature_names_arr.ndim != 1:
            raise ValueError(
                "feature_names must be a 1D list/array of names aligned with columns of X."
            )
        if feature_names_arr.size != n_features:
            raise ValueError(
                f"feature_names length ({feature_names_arr.size}) does not match "
                f"number of columns in X ({n_features})."
            )
    else:
        if is_dataframe:
            feature_names_arr = np.asarray(X.columns)
        else:
            feature_names_arr = None  # will generate generic names

    # ------------------------------------------------------------------
    # 4. Subset X by ranked indices, preserving ranking order
    # ------------------------------------------------------------------
    if is_dataframe:
        X_selected = X.iloc[:, selected_feature_indices]

        if feature_names_arr is not None:
            selected_feature_names = [
                feature_names_arr[i] for i in selected_feature_indices
            ]
        else:
            selected_feature_names = list(X_selected.columns)

    else:
        X_selected = X[:, selected_feature_indices]

        if feature_names_arr is not None:
            selected_feature_names = [
                feature_names_arr[i] for i in selected_feature_indices
            ]
        else:
            selected_feature_names = [f"f{i}" for i in selected_feature_indices]

    return X_selected, selected_feature_names, selected_feature_indices


def ranks_from_scores(
    scores: Union[Sequence[float], np.ndarray],
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Take a 1D score vector, return a DataFrame sorted by |score|.

    Parameters
    ----------
    scores : array-like, shape (n_features,)
        Raw scores from some feature selection method.
    ascending : bool, default=False
        Passed to sort_values on |score|:
        - False: largest |score| is rank 0 (typical)
        - True : smallest |score| is rank 0

    Returns
    -------
    df : pd.DataFrame with columns:
        - 'feature'        : feature index (0 .. n_features-1)
        - 'score'          : raw score
        - 'score_for_rank' : |score|
        - 'rank'           : 0 = best, 1 = second, ...
    """
    scores = np.asarray(scores)
    score_for_rank = np.abs(scores)  # ALWAYS abs

    df = pd.DataFrame({
        "feature": np.arange(len(scores)),
        "score": scores,
        "score_for_rank": score_for_rank
    })

    df = df.sort_values("score_for_rank", ascending=ascending).reset_index(drop=True)
    df["rank"] = np.arange(len(df))

    return df








def compute_feature_mean_ranks(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    methods_config: Optional[Dict[str, Dict[str, Any]]] = None,
    N: int = 100,
    ascending: bool = True,
    random_state: Optional[int] = 42,

    # NEW: parallelization controls (grouped mode only)
    parallelize: bool = False,
    n_jobs: int = -1,
    backend: str = "loky",
) -> Dict[str, pd.DataFrame]:
    """
    Generic feature ranking with optional group-aware bootstrapping.

    Modes
    -----
    1) Grouped mode (bootstrapping):
       - If `groups` is a numpy array with >= 2 unique values, we assume multiple
         groups (e.g. patient IDs).
       - For each of N iterations:
           * sample one row per group via `sample_one_row_per_group(X, y, groups, ...)`
           * for each method in `methods_config`:
               - compute per-feature scores with `compute_feature_scores`
               - convert scores -> ranks via `ranks_from_scores`
       - Aggregate mean rank per feature per method across bootstraps.

    2) Ungrouped mode (no bootstrapping):
       - If `groups` is None, we treat rows as independent and skip bootstrapping.
       - We use the full dataset once:
           * X_sub = X, y_sub = y
           * for each method:
               - compute scores once
               - convert to ranks once
       - The returned `mean_rank` per feature is just the rank from that single run.

    Parallelization (grouped mode only)
    ----------------------------------
    - If `groups` is not None and `parallelize=True`, bootstrap iterations are
      parallelized with joblib:
        Parallel(n_jobs=n_jobs, backend=backend)

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix. MUST be a numpy array (not a DataFrame).

    y : np.ndarray of shape (n_samples,)
        Target labels. MUST be a numpy array.

    groups : np.ndarray of shape (n_samples,), optional
        Group IDs (e.g. patient IDs). If provided and has >= 2 unique values,
        we do group-aware bootstrapping. If None, we do a single pass.

    methods_config : dict[str, dict], optional
        Keys are method labels (used in output), e.g.
            {
              "phik": {},
              "mrmr_MID": {"mrmr_method": "MID", "regression": False},
              "reliefF": {...},
            }
        Each value is a config dict passed into `compute_feature_scores`.
        If None, defaults to {"phik": {}}.

    N : int, default=100
        Number of bootstrap iterations (only used if `groups` is not None).

    ascending : bool, default=True
        Sorting direction for final mean_rank per method:
        - True  -> smaller mean_rank first (best features at top)
        - False -> larger mean_rank first.

    random_state : int or None, default=42
        Seed for the RNG (used only in grouped bootstrapping). Seeds are generated
        up-front for reproducibility (including in parallel mode).

    parallelize : bool, default=False
        If True and `groups` is not None, parallelize bootstrap iterations.

    n_jobs : int, default=-1
        Passed to joblib.Parallel. -1 uses all available cores.

    backend : str, default="loky"
        Passed to joblib.Parallel. "loky" uses process-based parallelism.

    Returns
    -------
    results : dict[str, pd.DataFrame]
        For each method label (e.g. 'phik', 'mrmr_FCD'), we return
        `label + '_feat_ranking'` as the key, mapped to a DataFrame:
            - 'feature'   : feature index
            - 'mean_rank' : average rank across bootstraps (or single run)
        DataFrames are sorted by 'mean_rank' using `ascending`.
    """
    # -------------------------------------------------------------
    # 0. Type & shape checks
    # -------------------------------------------------------------
    _prepare_collinearity_inputs(
        X=X,
        groups=groups,
        N=N,
        feature_names=None,
        threshold=0.0,  # dummy
    )

    n_samples, n_features = X.shape
    _validate_target_array(y, n_samples=n_samples, name="y")

    # -------------------------------------------------------------
    # 1. Methods config & rank matrix initialization
    # -------------------------------------------------------------
    if methods_config is None:
        methods_config = {"phik": {}}
    else:
        methods_config = dict(methods_config)

    grouped_mode = groups is not None
    rng = np.random.default_rng(random_state)

    rows_per_method = N if grouped_mode else 1
    rank_mats = {
        method_name: np.zeros((rows_per_method, n_features), dtype=float)
        for method_name in methods_config.keys()
    }

    # -------------------------------------------------------------
    # 2. Main loop: grouped bootstrapping vs single full-data run
    # -------------------------------------------------------------
    if grouped_mode:
        # Pre-generate seeds so sequential and parallel are reproducible
        seeds = rng.integers(0, 1_000_000, size=N, dtype=np.int64)

        def _one_boot(seed_n: int) -> Dict[str, np.ndarray]:
            # sample one row per group
            X_sub, y_sub, _, _ = sample_one_row_per_group(
                X, y, groups, random_state=int(seed_n)
            )

            out: Dict[str, np.ndarray] = {}
            for method_name, config in methods_config.items():
                scores = compute_feature_scores(method_name, X_sub, y_sub, config)

                # largest |score| => rank 0
                df_rank = ranks_from_scores(scores, ascending=False)

                ranks_this = np.zeros(n_features, dtype=float)
                ranks_this[df_rank["feature"].values] = df_rank["rank"].values
                out[method_name] = ranks_this

            return out

        if parallelize:
            print('='*100)
            print(">>>> Parallelization active")
            print('='*100)
            boot_results = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(_one_boot)(s) for s in seeds
            )
            for n, out in enumerate(boot_results):
                for method_name, ranks_this in out.items():
                    rank_mats[method_name][n] = ranks_this
        else:
            for n in trange(N, desc="Bootstrap iterations"):
                out = _one_boot(seeds[n])
                for method_name, ranks_this in out.items():
                    rank_mats[method_name][n] = ranks_this

    # OLD CODE TO BE REMOVED
    # else:
    #     # Ungrouped mode: one run on full data
    #     X_sub = X
    #     y_sub = y

    #     for method_name, config in methods_config.items():
    #         scores = compute_feature_scores(method_name, X_sub, y_sub, config)
    #         df_rank = ranks_from_scores(scores, ascending=False)

    #         ranks_this = np.zeros(n_features, dtype=float)
    #         ranks_this[df_rank["feature"].values] = df_rank["rank"].values

    #         rank_mats[method_name][0] = ranks_this


    else:
        # Ungrouped mode: one run on full data (no bootstrapping)
        X_sub = X
        y_sub = y

        method_items = list(methods_config.items())

        pbar = tqdm(method_items, desc="Methods (ungrouped)")
        for method_name, config in pbar:
            # Show the current method name on the progress bar
            pbar.set_postfix_str(method_name)

            scores = compute_feature_scores(method_name, X_sub, y_sub, config)
            df_rank = ranks_from_scores(scores, ascending=False)

            ranks_this = np.zeros(n_features, dtype=float)
            ranks_this[df_rank["feature"].values] = df_rank["rank"].values

            rank_mats[method_name][0] = ranks_this



    # -------------------------------------------------------------
    # 3. Aggregate results
    # -------------------------------------------------------------
    results: Dict[str, pd.DataFrame] = {}
    for method_name, mat in rank_mats.items():
        mean_rank = mat.mean(axis=0)

        df_summary = pd.DataFrame({
            "feature": np.arange(n_features),
            "mean_rank": mean_rank
        }).sort_values(
            by="mean_rank",
            ascending=ascending
        ).reset_index(drop=True)

        results[f"{method_name}_feat_ranking"] = df_summary

    return results




# -------------------------------------------------------------
# Testing feature selection methods
# -------------------------------------------------------------
# Use code below for drop cosntant features testing
def make_synthetic_constant_dataset(
    n_groups: int = 4,
    samples_per_group: int = 30,
    n_noise_features: int = 3,
    seed: int = 7,
):
    """
    Synthetic dataset to test BOTH global + local constant-feature dropping.

    Returns
    -------
    X_raw : (n_samples, n_features) float32
    y          : (n_samples,) int32  (simple binary labels by group)
    groups     : (n_samples,) int32
    feature_names : list[str]

    Feature design (useful expectations):
      - global_const_zeros: globally constant -> should drop in global mode (tol_global=1.0)
      - global_const_pi:    globally constant -> should drop in global mode (tol_global=1.0)

      - group_const_value:  constant *within every group*, but different across groups
                           -> NOT dropped globally, but dropped locally when local_frac=1.0

      - const_in_half_groups: constant within ~half the groups, varying in the others
                           -> dropped locally when local_frac <= ~0.5

      - const_in_one_group: constant only in group 0
                           -> dropped locally when local_frac <= 1/n_groups

      - noise_*: varying everywhere -> should not be dropped
    """
    rng = np.random.default_rng(seed)

    n_samples = n_groups * samples_per_group
    groups = np.repeat(np.arange(n_groups, dtype=np.int32), samples_per_group)

    # Optional labels (just to mimic your pipeline shape)
    # e.g., alternating 0/1 by group
    y = (groups % 2).astype(np.int32)

    # ---- Build features ----
    # 1) Globally constant features
    global_const_zeros = np.zeros(n_samples, dtype=np.float32)
    global_const_pi = np.full(n_samples, np.pi, dtype=np.float32)

    # 2) Constant within each group, but different across groups (NOT globally constant)
    #    e.g., group 0 all 0.1, group 1 all 0.2, ...
    group_const_value = np.zeros(n_samples, dtype=np.float32)
    for g in range(n_groups):
        group_const_value[groups == g] = np.float32(0.1 * (g + 1))

    # 3) Constant within about half the groups, varying in the others
    const_in_half_groups = np.zeros(n_samples, dtype=np.float32)
    half = max(1, n_groups // 2)
    for g in range(n_groups):
        mask = groups == g
        if g < half:
            # constant within these groups
            const_in_half_groups[mask] = np.float32(42.0)
        else:
            # varying within these groups
            const_in_half_groups[mask] = rng.normal(loc=0.0, scale=1.0, size=mask.sum()).astype(np.float32)

    # 4) Constant within only one group (group 0), varying elsewhere
    const_in_one_group = rng.normal(loc=0.0, scale=1.0, size=n_samples).astype(np.float32)
    const_in_one_group[groups == 0] = np.float32(-7.0)

    # 5) Regular varying features (noise)
    noise = rng.normal(size=(n_samples, n_noise_features)).astype(np.float32)

    # Stack into matrix
    X = np.column_stack([
        global_const_zeros,
        global_const_pi,
        group_const_value,
        const_in_half_groups,
        const_in_one_group,
        noise
    ]).astype(np.float32)

    feature_names = (
        [
            "global_const_zeros",
            "global_const_pi",
            "group_const_value",
            "const_in_half_groups",
            "const_in_one_group",
        ]
        + [f"noise_{i}" for i in range(n_noise_features)]
    )

    return X, y, groups, feature_names





def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow for most keys, deep-merge nested dicts."""
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _materialize_methods_config(
    methods_registry: Dict[str, Dict[str, Any]],
    method_names: Sequence[str],
    n_features_current: int,
    method_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build a stage-specific methods_config dict from the global registry,
    applying optional per-method overrides and resolving "ALL" placeholders.
    """
    methods_config: Dict[str, Dict[str, Any]] = {}
    method_overrides = method_overrides or {}

    for name in method_names:
        if name not in methods_registry:
            raise KeyError(
                f"Method '{name}' not found in cfg['methods'] registry. "
                f"Available: {sorted(methods_registry.keys())}"
            )

        params = deepcopy(methods_registry[name])
        if name in method_overrides:
            params = _deep_merge(params, method_overrides[name])

        # Resolve placeholders (your Relief-family methods need this)
        if params.get("n_features_to_select") == "ALL":
            params["n_features_to_select"] = int(n_features_current)

        methods_config[name] = params

    return methods_config


def run_rank_select_pipeline(
    *,
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    groups: Optional[np.ndarray],
    feature_names: Optional[Union[Sequence[str], np.ndarray]],
    cfg: Dict[str, Any],
) -> Tuple[Union[np.ndarray, pd.DataFrame], List[str], List[Dict[str, Any]]]:
    """
    Runs a multi-stage rank→aggregate→select pipeline driven by `cfg`.

    cfg format (matches what you defined):
      - cfg["defaults"] : dict of args for compute_feature_mean_ranks (optional)
      - cfg["methods"]  : registry of method_name -> params (required)
      - cfg["stages"]   : list of stages (required), each with:
            name, method_names, N, top_k
        optional per-stage:
            random_state, parallelize, n_jobs, backend, ascending
            method_overrides: {method_name: {param: value, ...}}

    Returns a single dict with:
      - "X", "feature_names_selected"
      - "history" (ordered list of stage outputs)
      - "by_stage" (dict keyed by stage name)
    """
    if "methods" not in cfg or "stages" not in cfg:
        raise KeyError("cfg must contain keys: 'methods' and 'stages'.")

    defaults = deepcopy(cfg.get("defaults", {}))
    methods_registry = cfg["methods"]
    stages = cfg["stages"]

    if feature_names is None:
        names_current = [str(i) for i in range(X.shape[1])]
    else:
        names_current = [str(f) for f in list(feature_names)]

    X_current = X
    history: List[Dict[str, Any]] = []
    by_stage: Dict[str, Dict[str, Any]] = {}

    for stage_idx, stage in enumerate(stages):
        if "method_names" not in stage or "N" not in stage or "top_k" not in stage:
            raise KeyError(
                f"Stage {stage_idx} missing required keys: 'method_names', 'N', 'top_k'. "
                f"Got: {list(stage.keys())}"
            )

        stage_name = stage.get("name", f"stage_{stage_idx}")
        stage_cfg = _deep_merge(defaults, stage)

        print( ">>> Stage = ", stage_name)

        methods_config = _materialize_methods_config(
            methods_registry=methods_registry,
            method_names=stage_cfg["method_names"],
            n_features_current=int(X_current.shape[1]),
            method_overrides=stage_cfg.get("method_overrides"),
        )

        # 1) bootstrap mean ranks
        results = compute_feature_mean_ranks(
            X=np.asarray(X_current) if isinstance(X_current, pd.DataFrame) else X_current,
            y=y,
            groups=groups,
            methods_config=methods_config,
            N=int(stage_cfg["N"]),
            ascending=bool(stage_cfg.get("ascending", True)),
            random_state=stage_cfg.get("random_state", 42),
            parallelize=bool(stage_cfg.get("parallelize", False)),
            n_jobs=int(stage_cfg.get("n_jobs", -1)),
            backend=str(stage_cfg.get("backend", "loky")),
        )

        # 2) aggregate across methods
        agg_results = aggregate_feature_rankings(
            results=results,
            top_k=int(stage_cfg["top_k"]),
            feature_names=names_current,
        )

        # 3) subset X to selected features
        X_next, names_next, selected_idx = construct_X_from_ranked_features(
            X=X_current,
            agg_results=agg_results,
            feature_names=names_current,
            ranking_key=stage_cfg.get("ranking_key", "final_feat_ranking"),
        )

        stage_out = {
            "stage": stage_name,
            "N": int(stage_cfg["N"]),
            "top_k": int(stage_cfg["top_k"]),
            "method_names": list(stage_cfg["method_names"]),
            "results": results,
            "agg_results": agg_results,
            "selected_idx": selected_idx,
            "kept_feature_names": names_next,
            "n_features_in": int(X_current.shape[1]),
            "n_features_out": int(X_next.shape[1]),
        }

        history.append(stage_out)
        by_stage[stage_name] = stage_out

        X_current = X_next
        names_current = names_next

    return {
        "X": X_current,
        "feature_names_selected": names_current,
        "history": history,
        "by_stage": by_stage,
    }
    






def make_grouped_classification(
    n_groups: int = 20,
    samples_per_group: int = 15,
    n_features: int = 50,
    n_informative: int = 5,
    n_redundant: int = 5,
    n_repeated: int = 0,
    n_classes: int = 2,
    class_sep: float = 1.0,
    flip_y: float = 0.01,
    random_state: Optional[int] = 42,
    shuffle_samples: bool = False,
    # feature-name options
    make_feature_names: bool = True,
    feature_name_prefix: str = "Feat_",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """
    Generate a grouped synthetic classification dataset.

    Returns
    -------
    X : (n_samples, n_features) ndarray
        Feature matrix.
    y : (n_samples,) ndarray
        Class labels.
    groups : (n_samples,) ndarray
        Group IDs for each sample: 0..n_groups-1 (repeated samples_per_group times).
    feature_names : list of str
        Names of each feature, e.g. ["Feat_01", "Feat_02", ...].
    meta : dict
        Dictionary with metadata:
        - "informative_indices": ndarray of informative feature indices
        - "redundant_indices": ndarray of redundant feature indices
        - "params": dict of parameters actually used
    """
    n_samples = n_groups * samples_per_group

    if n_informative + n_redundant + n_repeated > n_features:
        raise ValueError(
            "n_informative + n_redundant + n_repeated cannot exceed n_features"
        )

    # We keep shuffle=False so that informative/redundant features stay at low indices
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=n_classes,
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=random_state,
        shuffle=False,  # preserves feature index structure
    )

    # Groups: [0,0,...,0, 1,1,...,1, ..., n_groups-1,...]
    groups = np.repeat(np.arange(n_groups), samples_per_group)

    # Ground-truth feature indices
    informative_indices = np.arange(n_informative)
    redundant_indices = np.arange(n_informative, n_informative + n_redundant)

    # Optional shuffling of samples (NOT features)
    if shuffle_samples:
        rng = np.random.RandomState(random_state) if random_state is not None else np.random
        perm = rng.permutation(n_samples)
        X = X[perm]
        y = y[perm]
        groups = groups[perm]

    # Feature names
    if make_feature_names:
        # Zero-pad based on number of features: Feat_01, Feat_02, ..., Feat_50
        width = len(str(n_features))
        feature_names = [
            f"{feature_name_prefix}{i + 1:0{width}d}"
            for i in range(n_features)
        ]
    else:
        feature_names = [str(i) for i in range(n_features)]

    meta = {
        "informative_indices": informative_indices,
        "redundant_indices": redundant_indices,
        "params": dict(
            n_groups=n_groups,
            samples_per_group=samples_per_group,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            n_classes=n_classes,
            class_sep=class_sep,
            flip_y=flip_y,
            random_state=random_state,
            shuffle_samples=shuffle_samples,
            make_feature_names=make_feature_names,
            feature_name_prefix=feature_name_prefix,
        ),
    }

    return X, y, groups, feature_names, meta


