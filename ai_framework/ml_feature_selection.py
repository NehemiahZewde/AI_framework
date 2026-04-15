# ml_feature_selection.py
# ML feature selection 

from __future__ import annotations
from typing import Any, DefaultDict, Dict, Hashable, List, Optional, Sequence, Tuple, Union, Literal, Mapping
from collections import defaultdict
from math import ceil
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm.auto import tqdm
from copy import deepcopy
from sklearn.datasets import make_classification, make_regression
from joblib import Parallel, delayed
from tqdm.auto import trange
import os
from feature_engine.selection import MRMR
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor


RankingMetric = Literal[
    "auto",
    "mean_normalized_rank",
    "mean_importance",
]


RankingMetric = Literal[
    "auto",
    "mean_normalized_rank",
    "mean_importance",
]

# ============================================================
# Shared data sampling function 
# ============================================================
def sample_one_row_per_group(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly sample exactly one row from each group.

    This helper is used for group-mode ranking to create a derived dataset with
    one representative observation per group. For each unique group ID in
    `groups`, the function finds all matching row indices and randomly selects
    one of them. The selected rows are returned in the order of the sorted unique
    group IDs produced by `np.unique(groups)`.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Full feature matrix.
    y : np.ndarray of shape (n_samples,)
        Target vector aligned row-wise with `X`.
    groups : np.ndarray of shape (n_samples,)
        Group identifier for each row, such as a subject ID or patient ID.
    random_state : Optional[int], default=42
        Seed for reproducible random row selection within each group.

    Returns
    -------
    X_sub : np.ndarray of shape (n_groups, n_features)
        Feature matrix containing one sampled row per unique group.
    y_sub : np.ndarray of shape (n_groups,)
        Target values corresponding to the sampled rows.
    groups_sub : np.ndarray of shape (n_groups,)
        Group IDs corresponding to the sampled rows. This contains each unique
        group exactly once.
    indices : np.ndarray of shape (n_groups,)
        Original row indices from `X` that were selected.

    Notes
    -----
    - Sampling is performed independently within each group.
    - The number of returned rows equals the number of unique values in `groups`.
    - This function does not validate input shapes; callers should ensure that
    `X`, `y`, and `groups` are aligned before calling.
    """
    rng = np.random.default_rng(random_state)

    unique_groups = np.unique(groups)
    chosen_indices = []

    for g in unique_groups:
        idx_g = np.where(groups == g)[0]
        chosen_idx = rng.choice(idx_g)
        chosen_indices.append(chosen_idx)

    chosen_indices = np.array(chosen_indices)

    X_sub = X[chosen_indices, :]
    y_sub = y[chosen_indices]
    groups_sub = groups[chosen_indices]

    return X_sub, y_sub, groups_sub, chosen_indices


# from numpy.typing import NDArray
# Bundle = Dict[str, Any]
# def prepare_training_bundle(
#     bundle: Bundle,
#     n_features: Optional[int] = None,
#     keep_features: Optional[Sequence[str]] = None,
#     *,
#     strict: bool = True,
#     dedupe: bool = True,
#     copy_bundle: bool = True,
# ) -> Bundle:
#     """
#     Return a training-ready bundle with flexible feature selection.

#     Selection precedence:
#       1) keep_features (exact names)
#       2) n_features (first k)
#       3) if both None -> return all features (no reduction)

#     Args:
#       n_features: keep first n feature columns (prefix mode)
#       keep_features: keep these feature names (order preserved)
#       strict: if True, error on missing keep_features; else drop missing
#       dedupe: if True, de-duplicate keep_features while preserving order
#       copy_bundle: if True, return shallow copy; if False, may return original bundle when unchanged
#     """
#     if "X_raw" not in bundle or "feature_names" not in bundle:
#         raise KeyError("bundle must contain 'X_raw' and 'feature_names'")

#     X: NDArray[np.floating] = bundle["X_raw"]
#     feature_names: List[str] = list(bundle["feature_names"])

#     if X.ndim != 2:
#         raise ValueError(f"X_raw must be 2D, got shape {X.shape}")
#     if X.shape[1] != len(feature_names):
#         raise ValueError(f"Mismatch: X has {X.shape[1]} cols but feature_names has {len(feature_names)}")

#     # If nothing requested: return as-is (or shallow copy)
#     if keep_features is None and n_features is None:
#         return dict(bundle) if copy_bundle else bundle

#     # Avoid ambiguous intent
#     if keep_features is not None and n_features is not None:
#         raise ValueError("Provide either keep_features OR n_features, not both.")

#     out = dict(bundle)  # shallow copy

#     # ---- selection by names ----
#     if keep_features is not None:
#         if len(keep_features) == 0:
#             raise ValueError("keep_features must be non-empty")

#         if dedupe:
#             seen = set()
#             keep_features = [n for n in keep_features if not (n in seen or seen.add(n))]

#         name_to_idx = {n: i for i, n in enumerate(feature_names)}

#         missing = [n for n in keep_features if n not in name_to_idx]
#         if missing and strict:
#             raise KeyError(f"Requested features not found: {missing[:10]}{'...' if len(missing) > 10 else ''}")

#         idxs = [name_to_idx[n] for n in keep_features if n in name_to_idx]
#         if len(idxs) == 0:
#             raise ValueError("No features selected (all requested features missing).")

#         out["X_raw"] = X[:, idxs]
#         out["feature_names"] = [feature_names[i] for i in idxs]
#         return out

#     # ---- selection by prefix (n_features) ----
#     if n_features < 0:
#         raise ValueError("n_features must be >= 0")

#     k = min(n_features, X.shape[1])
#     out["X_raw"] = X[:, :k]
#     out["feature_names"] = feature_names[:k]
#     return out


# ============================================================
# Shared validation helpers
# ============================================================

def _validate_X_y(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate supervised-learning inputs and return normalized numpy arrays."""
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X must be a 2D numpy array; got shape {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be a 1D numpy array; got shape {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of rows. "
            f"Got X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}."
        )

    return X, y


def _validate_feature_names(
    feature_names: Sequence[str],
    n_features: int,
    *,
    require_unique: bool = False,
) -> List[str]:
    """Validate feature-name count and optionally require uniqueness."""
    feature_names_list = [str(f) for f in feature_names]

    if len(feature_names_list) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names_list)}) must match "
            f"number of columns in X ({n_features})."
        )

    if require_unique and len(set(feature_names_list)) != len(feature_names_list):
        raise ValueError("feature_names must be unique.")

    return feature_names_list


def _validate_groups(
    groups: Optional[np.ndarray],
    n_samples: int,
    *,
    required: bool = False,
) -> Optional[np.ndarray]:
    """Validate optional group labels and enforce presence when required."""
    if required and groups is None:
        raise ValueError("groups must be provided when group_mode is True.")

    if groups is None:
        return None

    groups_array = np.asarray(groups)
    if groups_array.ndim != 1 or len(groups_array) != n_samples:
        raise ValueError(
            "groups must be a 1D array with length equal to the number of rows in X."
        )

    return groups_array


def _validate_subset_sizes(
    subset_sizes: Sequence[int],
    n_features: int,
) -> List[int]:
    """Validate subset sizes and return sorted unique sizes within valid bounds."""
    subset_sizes_list = [int(size) for size in subset_sizes]

    if len(subset_sizes_list) == 0:
        raise ValueError("subset_sizes must be a non-empty sequence of integers.")

    invalid_subset_sizes = sorted(
        {size for size in subset_sizes_list if not (1 <= size <= n_features)}
    )
    if invalid_subset_sizes:
        raise ValueError(
            "Invalid subset_sizes detected. Each subset size must be between 1 and "
            f"the number of columns in X ({n_features}). Invalid values: {invalid_subset_sizes}"
        )

    return sorted(set(subset_sizes_list))


def _validate_ranking_metric(
    ranking_metric: str,
) -> str:
    """Validate the configured final ranking metric."""
    allowed_ranking_metrics = {
        "auto",
        "mean_normalized_rank",
        "mean_importance",
    }
    if ranking_metric not in allowed_ranking_metrics:
        raise ValueError(
            f"ranking_metric must be one of {sorted(allowed_ranking_metrics)}"
        )
    return ranking_metric


def _resolve_effective_ranking_metric(
    ranking_metric: str,
    subset_sizes: Sequence[int],
) -> str:
    """Resolve 'auto' ranking_metric to the effective metric used for sorting."""
    if ranking_metric == "auto":
        if all(size == 1 for size in subset_sizes):
            return "mean_importance"
        return "mean_normalized_rank"
    return ranking_metric


def _validate_models_dict(
    model_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate the model registry used by the ranking pipeline."""
    if not isinstance(model_dict, dict) or len(model_dict) == 0:
        raise ValueError("cfg['models'] must be a non-empty dict of model_name -> estimator.")
    return model_dict


def _validate_original_feature_indices(
    original_feature_indices: Optional[np.ndarray],
    n_features: int,
) -> np.ndarray:
    """Validate or initialize mapping from current columns to original feature indices."""
    if original_feature_indices is None:
        out = np.arange(n_features, dtype=int)
    else:
        out = np.asarray(original_feature_indices, dtype=int)

    if out.ndim != 1:
        raise ValueError(
            f"original_feature_indices must be 1D; got shape {out.shape}."
        )
    if len(out) != n_features:
        raise ValueError(
            "original_feature_indices length must match the number of columns in X. "
            f"Got len(original_feature_indices)={len(out)} and n_features={n_features}."
        )
    if len(np.unique(out)) != len(out):
        raise ValueError("original_feature_indices must contain unique values.")

    return out


def _validate_stage_selection_output(
    selected_idx_local: np.ndarray,
    names_next: Sequence[str],
    n_current_features: int,
    stage_name: str,
) -> np.ndarray:
    """Validate stage output indices before passing selected features forward."""
    selected_idx_local = np.asarray(selected_idx_local, dtype=int)

    if selected_idx_local.ndim != 1:
        raise ValueError(
            f"Stage '{stage_name}' returned selected_feature_indices_local with "
            f"invalid shape {selected_idx_local.shape}; expected 1D."
        )

    if len(selected_idx_local) != len(names_next):
        raise ValueError(
            f"Stage '{stage_name}' returned inconsistent selection sizes: "
            f"{len(selected_idx_local)} indices vs {len(names_next)} names."
        )

    if np.any(selected_idx_local < 0) or np.any(selected_idx_local >= n_current_features):
        raise ValueError(
            f"Stage '{stage_name}' returned out-of-bounds local indices: "
            f"{selected_idx_local.tolist()}"
        )

    return selected_idx_local


def _validate_cv_targets(
    y: np.ndarray,
    n_splits: int,
    *,
    task_type: str,
    context: str = "CV",
) -> None:
    """Validate that the target supports the requested cross-validation scheme."""
    y_array = np.asarray(y)

    if y_array.ndim != 1:
        raise ValueError(f"{context}: y must be 1D; got shape {y_array.shape}.")

    if task_type == "classification":
        unique_classes, class_counts = np.unique(y_array, return_counts=True)

        if len(unique_classes) < 2:
            raise ValueError(
                f"{context}: need at least 2 classes for classification CV, "
                f"but found {len(unique_classes)} class."
            )

        min_class_count = int(class_counts.min())
        if n_splits > min_class_count:
            class_count_str = ", ".join(
                f"class {cls}: {cnt}"
                for cls, cnt in zip(unique_classes.tolist(), class_counts.tolist())
            )
            raise ValueError(
                f"{context}: n_splits={n_splits} cannot be greater than the number of "
                f"members in the smallest class ({min_class_count}). "
                f"Class counts: {class_count_str}."
            )

    elif task_type == "regression":
        if n_splits > len(y_array):
            raise ValueError(
                f"{context}: n_splits={n_splits} cannot exceed number of samples ({len(y_array)})."
            )


def _validate_group_target_consistency(
    y: np.ndarray,
    groups: np.ndarray,
) -> None:
    """Ensure each group maps to exactly one target value in group_mode."""
    y_array = np.asarray(y)
    groups_array = np.asarray(groups)

    if y_array.ndim != 1:
        raise ValueError(f"y must be 1D; got shape {y_array.shape}.")
    if groups_array.ndim != 1:
        raise ValueError(f"groups must be 1D; got shape {groups_array.shape}.")
    if len(y_array) != len(groups_array):
        raise ValueError(
            f"y and groups must have the same length. "
            f"Got len(y)={len(y_array)} and len(groups)={len(groups_array)}."
        )

    inconsistent_groups: List[Dict[str, Any]] = []

    for g in np.unique(groups_array):
        targets_in_group = np.unique(y_array[groups_array == g])
        if len(targets_in_group) > 1:
            inconsistent_groups.append(
                {
                    "group": g,
                    "targets": targets_in_group.tolist(),
                }
            )

    if inconsistent_groups:
        preview = ", ".join(
            f"group {item['group']}: targets={item['targets']}"
            for item in inconsistent_groups[:5]
        )
        extra = ""
        if len(inconsistent_groups) > 5:
            extra = f" ... and {len(inconsistent_groups) - 5} more inconsistent groups"
        raise ValueError(
            "group_mode=True requires each group to have a single consistent target value, "
            f"but inconsistent groups were found: {preview}{extra}."
        )


def _validate_pipeline_cfg(
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Validate top-level pipeline config and return defaults, models, and stages."""
    if "models" not in cfg:
        raise KeyError("cfg must contain key 'models'.")
    if "stages" not in cfg:
        raise KeyError("cfg must contain key 'stages'.")

    defaults = deepcopy(cfg.get("defaults", {}))
    models_registry = _validate_models_dict(cfg["models"])
    stages = cfg["stages"]

    if not isinstance(stages, list) or len(stages) == 0:
        raise ValueError("cfg['stages'] must be a non-empty list of stage configs.")

    validated_stages: List[Dict[str, Any]] = []
    stage_names: List[str] = []

    for idx, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError(f"Each stage must be a dict. Got {type(stage).__name__}.")

        stage_name = str(stage.get("name", f"stage_{idx}"))

        if "top_k" not in stage:
            raise KeyError(f"Stage '{stage_name}' is missing required key 'top_k'.")
        if "subset_sizes" not in stage:
            raise KeyError(f"Stage '{stage_name}' is missing required key 'subset_sizes'.")

        validated_stages.append(stage)
        stage_names.append(stage_name)

    duplicate_stage_names = sorted(
        {name for name in stage_names if stage_names.count(name) > 1}
    )
    if duplicate_stage_names:
        raise ValueError(
            "Stage names must be unique. Duplicate stage names found: "
            f"{duplicate_stage_names}"
        )

    return defaults, models_registry, validated_stages


def _validate_task_type(task_type: str) -> str:
    """Validate the pipeline task type."""
    allowed_task_types = {"classification", "regression"}
    if task_type not in allowed_task_types:
        raise ValueError(
            f"task_type must be one of {sorted(allowed_task_types)}; got {task_type!r}."
        )
    return task_type


def _validate_scoring(scoring: Any) -> Any:
    """Validate scoring as a sklearn scoring string, callable, or None."""
    if scoring is None:
        return scoring
    if isinstance(scoring, str) or callable(scoring):
        return scoring
    raise ValueError(
        "scoring must be a valid sklearn scoring string, callable scorer, or None."
    )


def _validate_collinearity_inputs(
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


# ============================================================
# Pairwise collinearity pruning (correlation threshold)
# ============================================================

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
) -> Dict[str, Any]:
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
    dict
        A dictionary with keys:
        - 'X_raw' : np.ndarray of shape (n_samples, n_features_kept)
            X after correlation-based pairwise collinearity removal.
        - 'feature_names' : list of str
            Names of the kept features (after correlation pruning).
        - 'removed_info' : dict[int, dict]
            Mapping from removed feature index (w.r.t. original X) to a dict with:
                * 'removed_feature_index' : int
                * 'removed_feature_name'  : str
                * 'reason'                : str (always "corr" for this function)
                * 'kept_feature_index'    : int (index of the feature that was kept)
                * 'kept_feature_name'     : str
                * 'corr_value'            : float (absolute correlation at removal time)
    
    """
    # -------------------------------------------------------------
    # 0. Validation (raises on error)
    # -------------------------------------------------------------
    _validate_collinearity_inputs(
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

    # 3) Final return block
    return {
        "X": X_pruned,
        "feature_names": kept_feature_names,
        "removed_info": removed_info,
    }



# ============================================================
# Filter feature selection
# ============================================================
def run_coarse_mrmr_selection(
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    feature_names: Optional[Sequence[str]] = None,
    N: int = 5,
    top_k: int = 15,
    random_state: int = 42,
    regression: bool = False,
    variant: str = "FCD",
    discrete_unique_threshold: int = 10,
    parallelize: bool = False,
    n_jobs: int = -1,
    backend: str = "loky",
) -> Dict[str, Any]:
    """
    Run a coarse feature selection step using a single MRMR variant.

    This function performs a simplified MRMR-based coarse feature selection
    stage. When `groups` is provided, it repeatedly samples one row per group,
    ranks features using the selected MRMR variant, and averages feature ranks
    across iterations. When `groups` is None, it runs once on the full dataset.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame of shape (n_samples, n_features)
        Input feature matrix.

    y : np.ndarray of shape (n_samples,)
        Target vector aligned row-wise with `X`.

    groups : Optional[np.ndarray] of shape (n_samples,), default=None
        Optional group identifiers. If provided, group-aware repeated sampling
        is used.

    feature_names : Optional[Sequence[str]], default=None
        Optional names aligned to the columns of `X`.

    N : int, default=5
        Number of repeated one-row-per-group bootstrap iterations when
        `groups` is provided.

    top_k : int, default=15
        Number of top-ranked features to keep.

    random_state : int, default=42
        Seed for reproducibility.

    regression : bool, default=False
        Passed directly to MRMR. Use True for regression, False for
        classification.

    variant : str, default="FCD"
        MRMR variant passed directly to `MRMR(method=...)`.

    discrete_unique_threshold : int, default=10
        A feature is treated as discrete if its number of unique values is
        less than or equal to this threshold.

    parallelize : bool, default=False
        If True and `groups` is provided, parallelize the N repeated MRMR runs.

    n_jobs : int, default=-1
        Number of workers for joblib.Parallel. -1 uses all available cores.

    backend : str, default="loky"
        Backend passed to joblib.Parallel.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing selected features and ranking outputs.
    """

    # ============================================================
    # 1. Standardize inputs
    # ============================================================
    is_df = isinstance(X, pd.DataFrame)
    X_np = X.to_numpy() if is_df else np.asarray(X)
    y = np.asarray(y)

    # ============================================================
    # 2. Validate core inputs
    # ============================================================
    if X_np.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {X_np.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D; got shape {y.shape}.")
    if X_np.shape[0] != y.shape[0]:
        raise ValueError(
            f"X rows ({X_np.shape[0]}) must match y length ({y.shape[0]})."
        )
    if N < 1:
        raise ValueError(f"N must be >= 1; got {N}.")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1; got {top_k}.")
    if discrete_unique_threshold < 1:
        raise ValueError(
            f"discrete_unique_threshold must be >= 1; got {discrete_unique_threshold}."
        )

    # ============================================================
    # 3. Read dimensions and normalize top_k
    # ============================================================
    n_samples, n_features = X_np.shape
    top_k = min(top_k, n_features)

    # ============================================================
    # 4. Validate optional groups input
    # ============================================================
    if groups is not None:
        groups = np.asarray(groups)
        if groups.ndim != 1:
            raise ValueError(f"groups must be 1D; got shape {groups.shape}.")
        if groups.shape[0] != n_samples:
            raise ValueError(
                f"groups length ({groups.shape[0]}) must match X rows ({n_samples})."
            )
        if np.unique(groups).size < 2:
            raise ValueError("groups must contain at least 2 unique values.")

    # ============================================================
    # 5. Validate optional feature names
    # ============================================================
    if feature_names is not None and len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match "
            f"number of features ({n_features})."
        )

    # ============================================================
    # 6. Initialize bookkeeping objects
    # ============================================================
    rng = np.random.default_rng(random_state)
    feature_cols = [f"f{i}" for i in range(n_features)]

    # ============================================================
    # 7. Infer discrete vs continuous features from full X
    # ============================================================
    n_unique_per_feature = [np.unique(X_np[:, j]).size for j in range(n_features)]
    discrete_features = [
        n_unique <= discrete_unique_threshold
        for n_unique in n_unique_per_feature
    ]

    # ============================================================
    # 8. Allocate rank matrix
    # ============================================================
    n_iters = N if groups is not None else 1
    rank_mat = np.zeros((n_iters, n_features), dtype=float)

    # ============================================================
    # 9. Define one grouped iteration
    # ============================================================
    def _one_grouped_iteration(seed_i: int) -> np.ndarray:
        X_sub, y_sub, _, _ = sample_one_row_per_group(
            X=X_np,
            y=y,
            groups=groups,
            random_state=int(seed_i),
        )

        selector = MRMR(
            method=variant,
            regression=regression,
            discrete_features=discrete_features,
            max_features=n_features,
        )
        selector.fit(pd.DataFrame(X_sub, columns=feature_cols), y_sub)

        scores = (
            pd.Series(selector.relevance_, index=selector.variables_)
            .reindex(feature_cols)
            .to_numpy()
        )

        order = np.argsort(-np.abs(scores))
        ranks = np.empty(n_features, dtype=float)
        ranks[order] = np.arange(n_features)
        return ranks

    # ============================================================
    # 10. Group-aware repeated ranking
    # ============================================================
    if groups is not None:
        seeds = rng.integers(0, 1_000_000, size=N, dtype=np.int64)

        if parallelize:
            print('='*100)
            print(">>>> Parallelization active")
            print('='*100)

            rank_rows = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(_one_grouped_iteration)(seed_i) for seed_i in seeds
            )
            for i, ranks in enumerate(rank_rows):
                rank_mat[i] = ranks
        else:
            for i in trange(N, desc=f"MRMR-{variant} bootstraps"):
                rank_mat[i] = _one_grouped_iteration(int(seeds[i]))

    # ============================================================
    # 11. Single-pass ranking on full data
    # ============================================================
    else:
        selector = MRMR(
            method=variant,
            regression=regression,
            discrete_features=discrete_features,
            max_features=n_features,
        )
        selector.fit(pd.DataFrame(X_np, columns=feature_cols), y)

        scores = (
            pd.Series(selector.relevance_, index=selector.variables_)
            .reindex(feature_cols)
            .to_numpy()
        )

        order = np.argsort(-np.abs(scores))
        ranks = np.empty(n_features, dtype=float)
        ranks[order] = np.arange(n_features)
        rank_mat[0] = ranks

    # ============================================================
    # 12. Aggregate ranks across iterations
    # ============================================================
    mean_rank = rank_mat.mean(axis=0)

    ranking = (
        pd.DataFrame({
            "feature": np.arange(n_features),
            "mean_rank": mean_rank,
            "is_discrete": discrete_features,
            "n_unique": n_unique_per_feature,
        })
        .sort_values(["mean_rank", "feature"], ascending=[True, True])
        .reset_index(drop=True)
    )

    # ============================================================
    # 13. Keep top-k features
    # ============================================================
    top_k_ranking = ranking.head(top_k).copy()
    selected_idx = top_k_ranking["feature"].to_numpy(dtype=int)

    # ============================================================
    # 14. Resolve selected feature names
    # ============================================================
    if feature_names is None:
        if is_df:
            selected_names = [str(X.columns[i]) for i in selected_idx]
        else:
            selected_names = [f"f{i}" for i in selected_idx]
    else:
        selected_names = [str(feature_names[i]) for i in selected_idx]

    # ============================================================
    # 15. Subset X to selected features
    # ============================================================
    X_selected = X.iloc[:, selected_idx] if is_df else X_np[:, selected_idx]
    top_k_ranking["feature_name"] = selected_names

    # ============================================================
    # 16. Return outputs
    # ============================================================
    return {
        "X": X_selected,
        "feature_indices": selected_idx,
        "feature_names": selected_names,
        "ranking": ranking,
        "top_k_ranking": top_k_ranking,
        "variant": variant,
        "discrete_unique_threshold": discrete_unique_threshold,
    }

# ============================================================
# Permutation based feature selection
# ============================================================

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    Values from `override` take precedence over `base`.
    """
    out = deepcopy(base)
    for k, v in override.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def single_dataset_permutation_ranking(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    cfg: Dict[str, Any],
    *,
    seed_offset: int = 0,
    stage_name: Optional[str] = None,
    stage_index: Optional[int] = None,
    n_stages: Optional[int] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[int, pd.DataFrame]]]:
    """
    Run balanced permutation-based feature ranking on one fixed dataset.

    This function performs ranking on a single, already-defined dataset and does
    not do any group resampling. It evaluates each model in `cfg["models"]`
    independently by repeatedly sampling balanced feature subsets, fitting the
    model under cross-validation, and computing permutation importance on the
    validation folds.

    For each subset size:
    - features are sampled repeatedly with a balancing scheme so under-sampled
      features are prioritized
    - permutation importance is computed within each CV fold
    - per-feature summaries are computed:
        - mean_rank
        - mean_normalized_rank
        - mean_importance
        - times_sampled
        - n_observations

    Final per-feature rankings across subset sizes are then aggregated using
    weighted averaging, where the weight for each subset size is the number of
    observations contributing to that feature's summary at that subset size.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix for the dataset to rank.
    y : np.ndarray of shape (n_samples,)
        Target vector.
    feature_names : Sequence[str]
        Names of the columns in `X`. Must match `X.shape[1]` and be unique.
    cfg : Dict[str, Any]
        Ranking configuration. Expected keys include:
        {
            "task_type": "classification",            # "classification" or "regression"
            "models": {
                "model_name": estimator,
                ...
            },
            "scoring": "roc_auc",                     # sklearn scoring string or callable used in permutation_importance
            "subset_sizes": [5, 10],                 # feature subset sizes evaluated during ranking
            "n_splits": 5,                           # number of CV folds used in the selected splitter
            "n_repeats": 10,                         # number of feature permutations per fold inside permutation_importance
            "target_feature_appearances": 20,        # target number of times each feature should appear across sampled subsets
            "random_state": 42,                      # base random seed for reproducibility
            "ranking_metric": "auto",                # final ranking rule: auto, mean_normalized_rank, or mean_importance
        }
    seed_offset : int, default=0
        Offset added to the base random state so repeated outer calls can remain
        reproducible while still varying randomness.
    stage_name : Optional[str], default=None
        Human-readable stage name from the outer pipeline, used only for tqdm
        progress display.
    stage_index : Optional[int], default=None
        Zero-based stage index from the outer pipeline, used only for tqdm
        progress display.
    n_stages : Optional[int], default=None
        Total number of stages in the outer pipeline, used only for tqdm
        progress display.

    Returns
    -------
    final_ranking_by_model : Dict[str, pd.DataFrame]
        Mapping from model name to a final ranking table. Each table contains one
        row per feature and includes weighted summary metrics across subset sizes,
        such as:
        - feature
        - mean_normalized_rank_across_sizes
        - mean_importance_across_sizes
        - n_subset_sizes_used
        - total_n_observations_across_sizes

    detailed_results_by_model : Dict[str, Dict[int, pd.DataFrame]]
        Mapping from model name to per-subset-size ranking summaries.
        The inner dictionary maps:
            subset_size -> pd.DataFrame
        where each DataFrame contains per-feature summaries for that subset size.

    Notes
    -----
    - This function does not perform feature selection directly; it produces
      rankings only.
    - If `ranking_metric == "auto"`, the final ranking uses:
        - "mean_importance" when all subset sizes are 1
        - "mean_normalized_rank" otherwise
    - `subset_sizes` are validated against the current number of features in `X`.
    """
    # ============================================================
    # 1. Read config
    # ============================================================
    # Read and validate the model registry that will be evaluated on this fixed
    # dataset. Each model is ranked independently.
    model_dict = cfg.get("models", None)
    if model_dict is None:
        raise KeyError("cfg must contain a 'models' key.")
    model_dict = _validate_models_dict(model_dict)

    # Read the task type so cross-validation and default scoring can be matched
    # to either classification or regression behavior.
    task_type = _validate_task_type(cfg.get("task_type", "classification"))

    # Choose a sensible default scoring rule based on task type, while still
    # allowing the caller to override it with any valid sklearn scorer.
    default_scoring_by_task = {
        "classification": "roc_auc",
        "regression": "neg_mean_squared_error",
    }
    scoring = _validate_scoring(
        cfg.get("scoring", default_scoring_by_task[task_type])
    )

    # Read the remaining ranking controls that govern subset sampling, CV,
    # permutation importance repetition, and final sorting behavior.
    subset_sizes = cfg.get("subset_sizes", [10, 15, 20])
    n_splits = int(cfg.get("n_splits", 5))
    n_repeats = int(cfg.get("n_repeats", 10))
    target_feature_appearances = int(cfg.get("target_feature_appearances", 20))
    random_state = int(cfg.get("random_state", 42))
    ranking_metric = _validate_ranking_metric(cfg.get("ranking_metric", "auto"))
    stage_name = str(cfg.get("name", "feature_ranking"))

    # ============================================================
    # 2. Input validation and setup
    # ============================================================
    # Validate the supervised-learning inputs and record the shape of the fixed
    # dataset being ranked.
    X, y = _validate_X_y(X, y)
    n_samples, n_features = X.shape

    # Validate feature names against the columns of `X` and require uniqueness so
    # all downstream feature lookups by name remain unambiguous.
    feature_names_list = _validate_feature_names(
        feature_names,
        n_features,
        require_unique=True,
    )

    # Validate and normalize the subset sizes requested for ranking on the
    # current dataset.
    valid_subset_sizes = _validate_subset_sizes(subset_sizes, n_features)

    # Resolve the effective ranking metric once up front so the final ranking
    # tables can be sorted consistently after all aggregation is complete.
    effective_ranking_metric = _resolve_effective_ranking_metric(
        ranking_metric,
        valid_subset_sizes,
    )

    # Validate that the target vector can support the requested number of CV
    # folds for the chosen task type.
    _validate_cv_targets(
        y,
        n_splits,
        task_type=task_type,
        context="single_dataset_permutation_ranking",
    )

    # Materialize the feature matrix as a DataFrame so feature subsets can be
    # selected by name while preserving readable column labels.
    X_df = pd.DataFrame(X, columns=feature_names_list)

    # ============================================================
    # 3. Run per model
    # ============================================================
    # Initialize the two top-level outputs:
    # - a final aggregated ranking table per model
    # - a detailed per-subset-size summary per model
    final_ranking_by_model: Dict[str, pd.DataFrame] = {}
    detailed_results_by_model: Dict[str, Dict[int, pd.DataFrame]] = {}

    # Rank features independently for each model template in the registry.
    for model_name, model_template in model_dict.items():

        # Seed a per-model RNG that drives feature-subset shuffling and
        # permutation-importance randomness.
        rng = np.random.RandomState(random_state + seed_offset)

        # Choose the cross-validation splitter based on task type:
        # - StratifiedKFold for classification
        # - KFold for regression
        if task_type == "classification":
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state + seed_offset,
            )
        else:
            cv = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state + seed_offset,
            )

        # Store the subset-size-specific summary tables for this model.
        detailed_results: Dict[int, pd.DataFrame] = {}

        # Collect per-feature summaries across subset sizes so a final
        # cross-subset-size ranking can be built after all subset sizes are done.
        overall_records: DefaultDict[Hashable, List[Dict[str, float]]] = defaultdict(list)


        # Create a progress bar over subset sizes so the outer bar reflects the
        # ranking plan for the current stage configuration.
        subset_progress = tqdm(
            valid_subset_sizes,
            total=len(valid_subset_sizes),
            desc=f"Feature selection stage={stage_name} | model={model_name}",
            unit="subset",
            leave=False,
        )

        # Evaluate the model separately at each requested subset size.
        for subset_size in subset_progress:
            # Compute how many subset-sampling runs are needed so, on average,
            # each feature appears roughly `target_feature_appearances` times.
            n_runs: int = ceil((target_feature_appearances * n_features) / subset_size)

            # Track how many times each feature has been sampled at this subset
            # size so under-sampled features can be prioritized.
            feature_counts: Dict[Hashable, int] = {
                feature: 0 for feature in feature_names_list
            }

            # Collect per-feature metric observations across runs and CV folds.
            feature_rank_records: DefaultDict[Hashable, List[float]] = defaultdict(list)
            feature_norm_rank_records: DefaultDict[Hashable, List[float]] = defaultdict(list)
            feature_importance_records: DefaultDict[Hashable, List[float]] = defaultdict(list)

            # Progress bar over repeated subset-sampling runs for this model and
            # subset size.
            run_progress_desc = f"{model_name} | subset={subset_size}"
            
            run_progress = tqdm(
                range(n_runs),
                total=n_runs,
                desc=run_progress_desc,
                unit="run",
                leave=False,
            )

            # Repeat balanced subset sampling enough times to build a stable
            # feature-level ranking summary.
            for _ in run_progress:
                run_progress.set_postfix(cv_folds=n_splits)

                # Start from a shuffled copy of all feature names so ties in
                # sample counts are broken randomly but reproducibly.
                shuffled_feature_names: List[Hashable] = feature_names_list.copy()
                rng.shuffle(shuffled_feature_names)

                # Sort features by how often they have already been sampled so
                # less frequently used features are prioritized.
                shuffled_feature_names.sort(
                    key=lambda feature: feature_counts[feature]
                )

                # Take the first `subset_size` features after balancing.
                selected_features: List[Hashable] = shuffled_feature_names[:subset_size]

                # Update sampling counts for the features used in this run.
                for feature in selected_features:
                    feature_counts[feature] += 1

                # Slice the DataFrame down to the currently selected features.
                X_subset: pd.DataFrame = X_df[selected_features]

                # Create the train/validation splits for this sampled subset.
                split_iter = cv.split(X_subset, y)

                # Evaluate permutation importance on each validation fold.
                for train_idx, valid_idx in split_iter:
                    # Build the train and validation matrices for this fold.
                    X_train: pd.DataFrame = X_subset.iloc[train_idx]
                    X_valid: pd.DataFrame = X_subset.iloc[valid_idx]

                    # Slice the aligned targets for this fold.
                    y_train = y[train_idx]
                    y_valid = y[valid_idx]

                    # Clone the model template so every fold/run starts from a
                    # fresh unfitted estimator.
                    fitted_model = clone(model_template)
                    fitted_model.fit(X_train, y_train)

                    # Compute permutation importance on the held-out validation
                    # fold using the requested scoring rule.
                    permutation_result = permutation_importance(
                        fitted_model,
                        X_valid,
                        y_valid,
                        scoring=scoring,
                        n_repeats=n_repeats,
                        random_state=rng.randint(0, 1_000_000),
                        n_jobs=-1,
                    )

                    # Convert the mean importances into a Series indexed by
                    # selected feature name for easier downstream lookup.
                    feature_importances = pd.Series(
                        permutation_result.importances_mean,
                        index=selected_features,
                    )

                    # Rank features within the current sampled subset from most
                    # important to least important.
                    feature_ranks = feature_importances.rank(
                        ascending=False,
                        method="average",
                    )

                    # Normalize ranks onto a 0-to-1-like scale where 1 is best,
                    # so results from different subset sizes remain comparable.
                    if subset_size == 1:
                        normalized_feature_ranks = pd.Series(
                            data=1.0,
                            index=selected_features,
                            dtype=float,
                        )
                    else:
                        normalized_feature_ranks = 1 - (
                            (feature_ranks - 1) / (subset_size - 1)
                        )

                    # Append this fold's observations to the per-feature record
                    # collections for later aggregation.
                    for feature in selected_features:
                        feature_importance_records[feature].append(
                            float(feature_importances[feature])
                        )
                        feature_rank_records[feature].append(
                            float(feature_ranks[feature])
                        )
                        feature_norm_rank_records[feature].append(
                            float(normalized_feature_ranks[feature])
                        )

            # Build the summary table for this subset size after all runs/folds
            # have been accumulated.
            subset_summary_rows: List[Dict[str, Any]] = []

            # Compute one aggregated row per feature that was observed at this
            # subset size.
            for feature in feature_names_list:
                if not feature_importance_records[feature]:
                    continue

                # The number of recorded fold-level observations contributing to
                # this feature's metrics at this subset size.
                n_observations = len(feature_importance_records[feature])

                # Average the raw rank, normalized rank, and importance across
                # all observations for this feature.
                mean_rank = float(np.mean(feature_rank_records[feature]))
                mean_normalized_rank = float(np.mean(feature_norm_rank_records[feature]))
                mean_importance = float(np.mean(feature_importance_records[feature]))

                # Append the per-feature summary row for this subset size.
                subset_summary_rows.append(
                    {
                        "feature": feature,
                        "subset_size": subset_size,
                        "times_sampled": feature_counts[feature],
                        "n_observations": n_observations,
                        "mean_rank": mean_rank,
                        "mean_normalized_rank": mean_normalized_rank,
                        "mean_importance": mean_importance,
                    }
                )

                # Also append a compact record used later to aggregate rankings
                # across subset sizes.
                overall_records[feature].append(
                    {
                        "subset_size": float(subset_size),
                        "mean_normalized_rank": mean_normalized_rank,
                        "mean_importance": mean_importance,
                        "n_observations": float(n_observations),
                    }
                )

            # Convert the subset-size summary rows into a DataFrame.
            subset_summary_df = pd.DataFrame(subset_summary_rows)

            # Sort the subset-size summary from strongest to weakest features.
            subset_summary_df = subset_summary_df.sort_values(
                by=["mean_normalized_rank", "mean_importance"],
                ascending=[False, False],
            ).reset_index(drop=True)

            # Store the sorted summary table under this subset size.
            detailed_results[subset_size] = subset_summary_df

        # Build the final cross-subset-size ranking table for this model.
        final_rows: List[Dict[str, Any]] = []

        # Collapse each feature's subset-size-specific records into one final row.
        for feature, records in overall_records.items():
            # Weight subset-size contributions by the number of observations
            # supporting each subset-size summary.
            weights = np.array(
                [record["n_observations"] for record in records],
                dtype=float,
            )

            # Guard against invalid weighting inputs before computing averages.
            if np.any(weights < 0):
                raise ValueError(f"Negative n_observations encountered for feature '{feature}'.")
            if np.all(weights == 0):
                raise ValueError(f"All n_observations are zero for feature '{feature}'.")

            # Compute the weighted average normalized rank across subset sizes.
            weighted_mean_normalized_rank = float(
                np.average(
                    [record["mean_normalized_rank"] for record in records],
                    weights=weights,
                )
            )

            # Compute the weighted average importance across subset sizes.
            weighted_mean_importance = float(
                np.average(
                    [record["mean_importance"] for record in records],
                    weights=weights,
                )
            )

            # Append the final aggregated ranking row for this feature.
            final_rows.append(
                {
                    "feature": feature,
                    "mean_normalized_rank_across_sizes": weighted_mean_normalized_rank,
                    "mean_importance_across_sizes": weighted_mean_importance,
                    "n_subset_sizes_used": len(records),
                    "total_n_observations_across_sizes": int(weights.sum()),
                }
            )

        # Convert the aggregated rows into the final ranking DataFrame.
        final_ranking = pd.DataFrame(final_rows)

        # Sort the final ranking according to the resolved ranking rule.
        if effective_ranking_metric == "mean_normalized_rank":
            final_ranking = final_ranking.sort_values(
                by=[
                    "mean_normalized_rank_across_sizes",
                    "mean_importance_across_sizes",
                ],
                ascending=[False, False],
            ).reset_index(drop=True)
        elif effective_ranking_metric == "mean_importance":
            final_ranking = final_ranking.sort_values(
                by=[
                    "mean_importance_across_sizes",
                    "mean_normalized_rank_across_sizes",
                ],
                ascending=[False, False],
            ).reset_index(drop=True)
        else:
            raise ValueError(
                f"Unsupported effective_ranking_metric: {effective_ranking_metric}"
            )

        # Store the final ranking table and the detailed subset-size summaries
        # for this model.
        final_ranking_by_model[model_name] = final_ranking
        detailed_results_by_model[model_name] = detailed_results

    # Return both the final model-level ranking tables and the more detailed
    # subset-size-specific summaries.
    return final_ranking_by_model, detailed_results_by_model
    
def balanced_permutation_rank_select_stage(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    feature_names: Sequence[str],
    cfg: Dict[str, Any],
    original_feature_indices: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Run a single-stage balanced rank-and-select procedure.

    This is the single-stage engine used by the multi-stage pipeline. It first
    ranks features using `single_dataset_permutation_ranking(...)`, then selects the top
    `top_k` features per model and returns the reduced design matrix along with
    feature names and indices.

    The engine supports two execution modes:

    1) Non-group mode (`group_mode=False`)
    - ranking is run once on the full dataset

    2) Group mode (`group_mode=True`)
    - one row per group is repeatedly sampled using
        `sample_one_row_per_group(...)`
    - ranking is run on each sampled dataset
    - final rankings and detailed results are aggregated across group iterations

    This function is stage-local: it operates on the current `X` passed into it.
    However, if `original_feature_indices` is provided, the returned
    `selected_feature_indices` are mapped back to the original dataset column
    indices.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Current feature matrix for this stage.
    y : np.ndarray of shape (n_samples,)
        Target vector.
    groups : Optional[np.ndarray] of shape (n_samples,)
        Group identifiers for each row. Required when `cfg["group_mode"]` is True.
    feature_names : Sequence[str]
        Feature names corresponding to the columns of `X`. Must match
        `X.shape[1]` and be unique within the current stage.
    cfg : Dict[str, Any]
        Single-stage configuration. Expected keys include:
        {
            "models": {
                "model_name": estimator,
                ...
            },
            "scoring": "roc_auc",
            "subset_sizes": [5, 10],
            "n_splits": 5,
            "n_repeats": 10,
            "target_feature_appearances": 20,
            "random_state": 42,
            "ranking_metric": "auto" | "mean_normalized_rank" | "mean_importance",
            "group_mode": False,
            "group_iterations": 10,
            "top_k": 10,
        }
    original_feature_indices : Optional[np.ndarray] of shape (n_features,), default=None
        Original dataset column indices corresponding to the columns of `X`.
        If None, defaults to `np.arange(X.shape[1])`, meaning the current columns
        are assumed to already be in original dataset order.

    Returns
    -------
    out : Dict[str, Any]
        Dictionary with keys:

        "final_ranking_by_model" : Dict[str, pd.DataFrame]
            Final per-model ranking tables with stable schema across group and
            non-group modes. Columns:
            - feature
            - mean_normalized_rank
            - mean_importance
            - n_subset_sizes_used
            - total_n_observations
            - group_iterations_used

        "detailed_results_by_model" : Dict[str, Dict[int, pd.DataFrame]]
            Per-model, per-subset-size detailed summaries with stable schema
            across group and non-group modes. Each table contains:
            - feature
            - subset_size
            - times_sampled
            - n_observations
            - mean_rank
            - mean_normalized_rank
            - mean_importance
            - group_iterations_used

        "selected_by_model" : Dict[str, Dict[str, Any]]
            Per-model selected outputs:
            {
                "X": np.ndarray,
                "selected_feature_names": List[str],
                "selected_feature_indices": np.ndarray,
                "selected_feature_indices_local": np.ndarray,
            }

            where:
            - `selected_feature_indices` are ORIGINAL dataset indices
            - `selected_feature_indices_local` are indices relative to the input
              `X` passed into this engine call

    Notes
    -----
    - This function performs a single ranking-and-selection stage only.
    - Multi-stage orchestration should be handled by
      `balanced_permutation_rank_select_pipeline(...)`.
    - In group mode, this function validates that each group has a single
      consistent target value before resampling.
    - `top_k` must be between 1 and the number of current features.
    """
    # ============================================================
    # 1. Read config
    # ============================================================
    # Read the stage-level execution settings that determine whether we run the
    # ranking once or aggregate across repeated one-row-per-group samples.
    group_mode = bool(cfg.get("group_mode", False))
    group_iterations = int(cfg.get("group_iterations", 10))
    random_state = int(cfg.get("random_state", 42))
    top_k = cfg.get("top_k", None)

    # `top_k` is required because this engine always performs both ranking and
    # selection within the current stage.
    if top_k is None:
        raise KeyError("cfg must contain 'top_k' for single-stage feature selection.")

    # Convert `top_k` to an integer explicitly so later indexing logic operates
    # on a normalized numeric type.
    top_k = int(top_k)

    # Reject invalid selection requests before running any ranking work.
    if top_k < 1:
        raise ValueError("cfg['top_k'] must be >= 1.")

    # ============================================================
    # 2. Input validation
    # ============================================================
    # Validate the supervised-learning inputs and record the size of the current
    # stage-local feature space.
    X, y = _validate_X_y(X, y)
    n_samples, n_features = X.shape

    # Validate that the provided feature names align with the current columns of
    # `X`, and require uniqueness so feature-name-based lookups remain safe.
    feature_names_list = _validate_feature_names(
        feature_names,
        n_features,
        require_unique=True,
    )

    # Validate or initialize the mapping from current-stage columns back to the
    # original dataset columns. This is what preserves original feature indices
    # even after repeated stage-wise reductions.
    original_feature_indices = _validate_original_feature_indices(
        original_feature_indices,
        n_features,
    )

    # Validate optional group IDs. When group mode is enabled, groups become
    # required because the engine will repeatedly sample one row per group.
    groups = _validate_groups(
        groups,
        n_samples,
        required=group_mode,
    )

    # In group mode, enforce the assumption that each group corresponds to one
    # stable target value. Without this invariant, one-row-per-group sampling
    # would create ambiguous group-level labels/targets.
    if group_mode:
        _validate_group_target_consistency(y, groups)

    # Group-mode aggregation must run at least once when enabled.
    if group_iterations < 1:
        raise ValueError("group_iterations must be >= 1.")

    # Do not allow the stage to request more features than currently exist in
    # the stage-local design matrix.
    if top_k > n_features:
        raise ValueError(
            f"cfg['top_k'] ({top_k}) cannot exceed number of features ({n_features})."
        )

    # Build a feature-name -> local-column-index lookup so selected feature names
    # from the ranking output can be converted back into array indices.
    feature_name_to_idx = {name: idx for idx, name in enumerate(feature_names_list)}

    # ============================================================
    # 3. Non-group mode: run once
    # ============================================================
    if not group_mode:
        # In non-group mode, perform ranking one time on the full current-stage
        # dataset without any group-based resampling.
        final_ranking_by_model, detailed_results_by_model = single_dataset_permutation_ranking(
            X=X,
            y=y,
            feature_names=feature_names_list,
            cfg=cfg,
            seed_offset=0,
        )

        # The single-dataset ranking helper may expose legacy "across_sizes"
        # column names. Normalize them here so this engine always returns one
        # canonical final-ranking schema.
        normalized_final_ranking_by_model: Dict[str, pd.DataFrame] = {}

        # Normalize each model's final ranking table independently.
        for model_name, df_rank in final_ranking_by_model.items():
            # Rename legacy summary columns into the canonical output names used
            # by this engine and by the group-mode aggregation path.
            df_norm = df_rank.rename(
                columns={
                    "mean_normalized_rank_across_sizes": "mean_normalized_rank",
                    "mean_importance_across_sizes": "mean_importance",
                    "total_n_observations_across_sizes": "total_n_observations",
                }
            ).copy()

            # Non-group mode always corresponds to exactly one ranking run, so
            # annotate that explicitly when the column is absent.
            if "group_iterations_used" not in df_norm.columns:
                df_norm["group_iterations_used"] = 1

            # Keep only the canonical final-ranking columns and present them in a
            # fixed order for downstream consumers.
            df_norm = df_norm[
                [
                    "feature",
                    "mean_normalized_rank",
                    "mean_importance",
                    "n_subset_sizes_used",
                    "total_n_observations",
                    "group_iterations_used",
                ]
            ].reset_index(drop=True)

            # Store the normalized final ranking table for this model.
            normalized_final_ranking_by_model[model_name] = df_norm

        # Apply the same normalization pattern to the per-subset detailed tables
        # so the detailed outputs also follow one canonical schema.
        normalized_detailed_results_by_model: Dict[str, Dict[int, pd.DataFrame]] = {}

        # Normalize each model's nested subset-size -> DataFrame mapping.
        for model_name, detail_dict in detailed_results_by_model.items():
            normalized_detail_dict: Dict[int, pd.DataFrame] = {}

            # Normalize each subset-size-specific detailed table separately.
            for subset_size, df_subset in detail_dict.items():
                # Work on a copy so the raw output is not mutated in place.
                df_norm = df_subset.copy()

                # In non-group mode, each detailed table also comes from exactly
                # one ranking run, so mark that explicitly if needed.
                if "group_iterations_used" not in df_norm.columns:
                    df_norm["group_iterations_used"] = 1

                # Keep only the canonical detailed-result columns in a consistent
                # order across models and subset sizes.
                df_norm = df_norm[
                    [
                        "feature",
                        "subset_size",
                        "times_sampled",
                        "n_observations",
                        "mean_rank",
                        "mean_normalized_rank",
                        "mean_importance",
                        "group_iterations_used",
                    ]
                ].reset_index(drop=True)

                # Store the normalized detailed table under its subset size.
                normalized_detail_dict[int(subset_size)] = df_norm

            # Store the normalized detailed-results mapping for this model.
            normalized_detailed_results_by_model[model_name] = normalized_detail_dict

        # Replace the raw single-dataset outputs with their normalized versions so
        # the remainder of the function works with one stable schema.
        final_ranking_by_model = normalized_final_ranking_by_model
        detailed_results_by_model = normalized_detailed_results_by_model

    else:
        # ========================================================
        # 4. Group mode: repeated sample_one_row_per_group + aggregate
        # ========================================================
        # Seed a reproducible generator that will create one independent random
        # seed per group-resampling iteration.
        rng_group = np.random.default_rng(random_state)

        # Pre-draw all per-iteration seeds so the group-mode run is reproducible
        # given the base random state.
        group_seeds = rng_group.integers(
            0,
            1_000_000,
            size=group_iterations,
            dtype=np.int64,
        )

        # Collect one final-ranking table per model per iteration.
        all_rankings_by_model: Dict[str, List[pd.DataFrame]] = defaultdict(list)

        # Collect one detailed-results bundle per model per iteration.
        all_details_by_model: Dict[str, List[Dict[int, pd.DataFrame]]] = defaultdict(list)

        # Repeat the one-row-per-group sampling process `group_iterations` times.
        for iter_idx in tqdm(
            range(group_iterations),
            total=group_iterations,
            desc="Group bootstrap iterations",
            unit="iter",
        ):
            # Use the iteration-specific seed so each grouped resample differs
            # while remaining reproducible overall.
            seed_n = int(group_seeds[iter_idx])

            # Create a derived dataset containing exactly one sampled row per
            # group from the full stage-local dataset.
            X_sub, y_sub, _, _ = sample_one_row_per_group(
                X,
                y,
                groups,
                random_state=seed_n,
            )

            # Run the standard single-dataset ranking procedure on this derived
            # group-level dataset.
            rankings_run, details_run = single_dataset_permutation_ranking(
                X=X_sub,
                y=y_sub,
                feature_names=feature_names_list,
                cfg=cfg,
                seed_offset=seed_n,
            )

            # Normalize each iteration's final ranking output immediately so the
            # later aggregation code can assume one stable column schema.
            for model_name, df_rank in rankings_run.items():
                df_rank_norm = df_rank.rename(
                    columns={
                        "mean_normalized_rank_across_sizes": "mean_normalized_rank",
                        "mean_importance_across_sizes": "mean_importance",
                        "total_n_observations_across_sizes": "total_n_observations",
                    }
                ).copy()

                # Each stored iteration corresponds to one group-resampled ranking
                # run, so annotate that count when missing.
                if "group_iterations_used" not in df_rank_norm.columns:
                    df_rank_norm["group_iterations_used"] = 1

                # Append the normalized ranking table to this model's list of
                # per-iteration ranking outputs.
                all_rankings_by_model[model_name].append(df_rank_norm)

            # Normalize each iteration's detailed outputs as well so detailed
            # aggregation can work with a stable schema.
            for model_name, detail_dict in details_run.items():
                normalized_detail_dict: Dict[int, pd.DataFrame] = {}

                # Normalize each subset-size-specific detailed table separately.
                for subset_size, df_subset in detail_dict.items():
                    df_subset_norm = df_subset.copy()

                    # Mark that this detailed table comes from one group-resampled
                    # ranking iteration when the column is absent.
                    if "group_iterations_used" not in df_subset_norm.columns:
                        df_subset_norm["group_iterations_used"] = 1

                    # Store the normalized detailed table under its subset size.
                    normalized_detail_dict[int(subset_size)] = df_subset_norm

                # Append the normalized detailed-results bundle for this model.
                all_details_by_model[model_name].append(normalized_detail_dict)

        # --------------------------------------------------------
        # 5. Aggregate final rankings across group iterations
        # --------------------------------------------------------
        # Recompute the effective ranking metric so the aggregated outputs are
        # sorted using the same ranking rule as the single-dataset path.
        ranking_metric = _validate_ranking_metric(cfg.get("ranking_metric", "auto"))
        subset_sizes = cfg.get("subset_sizes", [10, 15, 20])
        subset_sizes_for_metric = _validate_subset_sizes(subset_sizes, n_features)
        effective_ranking_metric = _resolve_effective_ranking_metric(
            ranking_metric,
            subset_sizes_for_metric,
        )

        # Initialize the final containers that will hold the aggregated outputs
        # returned by this engine.
        final_ranking_by_model = {}
        detailed_results_by_model = {}

        # Aggregate final ranking tables model by model.
        for model_name, rank_list in all_rankings_by_model.items():
            # Collect all per-feature metrics across group iterations before
            # collapsing them into a single aggregated final-ranking table.
            feature_records: DefaultDict[Hashable, Dict[str, List[float]]] = defaultdict(
                lambda: {
                    "mean_normalized_rank": [],
                    "mean_importance": [],
                    "n_subset_sizes_used": [],
                    "total_n_observations": [],
                }
            )

            # Append each iteration's per-feature values into the aggregation
            # record for that feature.
            for df_rank in rank_list:
                for _, row in df_rank.iterrows():
                    feat = row["feature"]
                    feature_records[feat]["mean_normalized_rank"].append(
                        float(row["mean_normalized_rank"])
                    )
                    feature_records[feat]["mean_importance"].append(
                        float(row["mean_importance"])
                    )
                    feature_records[feat]["n_subset_sizes_used"].append(
                        float(row["n_subset_sizes_used"])
                    )
                    feature_records[feat]["total_n_observations"].append(
                        float(row["total_n_observations"])
                    )

            final_rows: List[Dict[str, Any]] = []

            # Collapse each feature's per-iteration values into one aggregated row.
            for feat, vals in feature_records.items():
                # Weight each iteration by its number of contributing observations
                # so higher-information runs contribute more to the final estimate.
                weights = np.asarray(vals["total_n_observations"], dtype=float)

                # Guard against invalid observation counts before computing
                # weighted averages.
                if np.any(weights < 0):
                    raise ValueError(
                        f"Negative total_n_observations encountered for feature '{feat}'."
                    )
                if np.all(weights == 0):
                    raise ValueError(
                        f"All total_n_observations are zero for feature '{feat}'."
                    )

                # Compute the aggregated final-ranking metrics for this feature.
                final_rows.append(
                    {
                        "feature": feat,
                        "mean_normalized_rank": float(
                            np.average(vals["mean_normalized_rank"], weights=weights)
                        ),
                        "mean_importance": float(
                            np.average(vals["mean_importance"], weights=weights)
                        ),
                        "n_subset_sizes_used": float(
                            np.average(vals["n_subset_sizes_used"], weights=weights)
                        ),
                        "total_n_observations": int(weights.sum()),
                        "group_iterations_used": len(vals["mean_normalized_rank"]),
                    }
                )

            # Convert the aggregated rows into a final-ranking DataFrame.
            final_df = pd.DataFrame(final_rows)

            # Sort the aggregated ranking table using the same primary ranking
            # rule used in the single-dataset path.
            if effective_ranking_metric == "mean_normalized_rank":
                final_df = final_df.sort_values(
                    by=[
                        "mean_normalized_rank",
                        "mean_importance",
                    ],
                    ascending=[False, False],
                ).reset_index(drop=True)
            elif effective_ranking_metric == "mean_importance":
                final_df = final_df.sort_values(
                    by=[
                        "mean_importance",
                        "mean_normalized_rank",
                    ],
                    ascending=[False, False],
                ).reset_index(drop=True)
            else:
                raise ValueError(
                    f"Unsupported effective_ranking_metric: {effective_ranking_metric}"
                )

            # Store the aggregated final-ranking table for this model.
            final_ranking_by_model[model_name] = final_df

        # --------------------------------------------------------
        # 6. Aggregate detailed per-subset-size results across group iterations
        # --------------------------------------------------------
        # Aggregate the detailed outputs separately for each model.
        for model_name, details_list in all_details_by_model.items():
            # Group all detailed tables by subset size so each subset size gets
            # its own aggregated detailed summary.
            subset_size_to_tables: DefaultDict[int, List[pd.DataFrame]] = defaultdict(list)

            # Collect all tables for the same subset size into one list.
            for detail_dict in details_list:
                for subset_size, df_subset in detail_dict.items():
                    subset_size_to_tables[int(subset_size)].append(df_subset)

            aggregated_detail_dict: Dict[int, pd.DataFrame] = {}

            # Aggregate one detailed summary table per subset size.
            for subset_size, df_list in subset_size_to_tables.items():
                # Collect the detailed per-feature metrics across all group
                # iterations for this subset size.
                feature_records: DefaultDict[Hashable, Dict[str, List[float]]] = defaultdict(
                    lambda: {
                        "times_sampled": [],
                        "n_observations": [],
                        "mean_rank": [],
                        "mean_normalized_rank": [],
                        "mean_importance": [],
                    }
                )

                # Append each iteration's detailed values into the aggregation
                # record for that feature.
                for df_subset in df_list:
                    for _, row in df_subset.iterrows():
                        feat = row["feature"]
                        feature_records[feat]["times_sampled"].append(float(row["times_sampled"]))
                        feature_records[feat]["n_observations"].append(float(row["n_observations"]))
                        feature_records[feat]["mean_rank"].append(float(row["mean_rank"]))
                        feature_records[feat]["mean_normalized_rank"].append(
                            float(row["mean_normalized_rank"])
                        )
                        feature_records[feat]["mean_importance"].append(
                            float(row["mean_importance"])
                        )

                agg_rows: List[Dict[str, Any]] = []

                # Collapse each feature's detailed values into one aggregated row
                # for this subset size.
                for feat, vals in feature_records.items():
                    agg_rows.append(
                        {
                            "feature": feat,
                            "subset_size": subset_size,
                            "times_sampled": float(np.mean(vals["times_sampled"])),
                            "n_observations": float(np.mean(vals["n_observations"])),
                            "mean_rank": float(np.mean(vals["mean_rank"])),
                            "mean_normalized_rank": float(np.mean(vals["mean_normalized_rank"])),
                            "mean_importance": float(np.mean(vals["mean_importance"])),
                            "group_iterations_used": len(vals["mean_rank"]),
                        }
                    )

                # Convert the aggregated rows into a detailed DataFrame and sort
                # it from strongest to weakest features.
                agg_df = pd.DataFrame(agg_rows).sort_values(
                    by=[
                        "mean_normalized_rank",
                        "mean_importance",
                    ],
                    ascending=[False, False],
                ).reset_index(drop=True)

                # Store the aggregated detailed table for this subset size.
                aggregated_detail_dict[subset_size] = agg_df

            # Store the full subset-size -> aggregated detailed table mapping for
            # this model.
            detailed_results_by_model[model_name] = aggregated_detail_dict

    # ============================================================
    # 7. Select top-k per model
    # ============================================================
    # Build the final per-model selected-feature outputs from the normalized or
    # aggregated ranking tables.
    selected_by_model: Dict[str, Dict[str, Any]] = {}

    # Select features independently for each model.
    for model_name, df_rank in final_ranking_by_model.items():
        # Keep only the highest-ranked `top_k` features from the final ranking.
        df_top = df_rank.head(top_k).copy().reset_index(drop=True)

        # Read the selected feature names in ranked order.
        selected_feature_names = df_top["feature"].astype(str).tolist()

        # Convert the selected feature names into current-stage column indices.
        selected_feature_indices_local = np.array(
            [feature_name_to_idx[name] for name in selected_feature_names],
            dtype=int,
        )

        # Map the local stage indices back to the original dataset column indices.
        selected_feature_indices = original_feature_indices[selected_feature_indices_local]

        # Slice the current-stage design matrix down to only the selected columns.
        X_selected = X[:, selected_feature_indices_local]

        # Return both the reduced matrix and both index systems so callers can use
        # the selection either in the current stage space or the original space.
        selected_by_model[model_name] = {
            "X": X_selected,
            "selected_feature_names": selected_feature_names,
            "selected_feature_indices": selected_feature_indices,
            "selected_feature_indices_local": selected_feature_indices_local,
        }

    # ============================================================
    # 8. Return
    # ============================================================
    # Return the ranking outputs, detailed summaries, and top-k selections for
    # each model evaluated in this stage.
    return {
        "final_ranking_by_model": final_ranking_by_model,
        "detailed_results_by_model": detailed_results_by_model,
        "selected_by_model": selected_by_model,
    }

def balanced_permutation_rank_select_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    feature_names: Sequence[str],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run a multi-stage balanced feature rank-and-select pipeline.

    This function orchestrates the full pipeline separately for each model in
    `cfg["models"]`. Each stage calls `balanced_permutation_rank_select_stage(...)`
    using a stage-specific config containing exactly one model. The selected
    feature matrix and feature names from one stage become the inputs to the next
    stage for that same model.

    A key property of this function is that it preserves original dataset feature
    indices across stages. Even though each stage operates on a reduced feature
    matrix, the final returned `selected_feature_indices` are always relative to
    the original input `X`.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Full input feature matrix.
    y : 
    groups : Optional[np.ndarray] of shape (n_samples,)
        Optional group identifiers aligned row-wise with `X` and `y`.
        These are passed through to each stage and are only required when a
        stage enables `group_mode=True`.
    feature_names : Sequence[str]
        Names of the columns in `X`. Must match `X.shape[1]` and be unique.
    cfg : Dict[str, Any]
        Pipeline configuration dictionary. Expected structure:

        {
            "defaults": {
                "task_type": "classification",       # "classification" or "regression"
                "scoring": "roc_auc",                # metric used by permutation_importance on validation folds
                "n_splits": 5,                       # number of CV folds used by the selected CV splitter
                "n_repeats": 10,                     # number of permutation shuffles per feature
                "target_feature_appearances": 20,    # target number of times each feature should appear across sampled subsets
                "random_state": 42,                  # base random seed for reproducibility
                "ranking_metric": "auto",            # final ranking rule: auto, mean_normalized_rank, or mean_importance
                "group_mode": False,                 # if True, repeatedly sample one row per group before ranking
                "group_iterations": 10,              # number of one-row-per-group resampling iterations when group_mode=True
            },
            "models": {
                "logistic_regression": LogisticRegression(...),
                "XGBoost": XGBClassifier(...),
            },
            "stages": [
                {
                    "name": "coarse_rank_select",    # stage label used in output/history
                    "top_k": 10,                     # number of highest-ranked features to keep after this stage
                    "subset_sizes": [5, 10],         # feature subset sizes tested during ranking in this stage
                },
                {
                    "name": "final_rank_select",     # second-stage label used in output/history
                    "top_k": 5,                      # number of highest-ranked features to keep after this stage
                    "subset_sizes": [5, 10],         # feature subset sizes tested during ranking in this stage
                },
            ],
        }

        Notes on config:
        - `defaults` contains pipeline-wide settings that are merged into each stage
        - `models` maps model names to sklearn-compatible estimators
        - `stages` defines the ordered sequence of rank-and-select steps
        - `top_k` is required for every stage
        - `subset_sizes` is required for every stage
        - each stage config is merged over `cfg["defaults"]`, so stage-level keys
          can override defaults for that stage only

    Returns
    -------
    out : Dict[str, Any]
        Dictionary with key:

        "final_by_model" : Dict[str, Any]
            Per-model final pipeline outputs:
            {
                model_name: {
                    "X": np.ndarray,
                    "feature_names_selected": List[str],
                    "selected_feature_indices": np.ndarray,
                    "history": List[Dict[str, Any]],
                    "by_stage": Dict[str, Dict[str, Any]],
                },
                ...
            }

            where:
            - `X` is the final reduced feature matrix for that model
            - `feature_names_selected` are the final selected feature names
            - `selected_feature_indices` are ORIGINAL dataset column indices
            - `history` records outputs for each stage in order
            - `by_stage` maps stage name to its recorded output
        "y" : np.ndarray of shape (n_samples,)
            Target vector.
    Stage History
    -------------
    Each stage record contains:
    - stage
    - top_k
    - n_features_in
    - n_features_out
    - cfg_used
    - final_ranking
    - detailed_results
    - selected_feature_names
    - selected_feature_indices
    - selected_feature_indices_local
    - X_selected

    Notes
    -----
    - The pipeline is run independently for each model from start to finish.
    - Stages do not share rankings across models.
    - Original feature-index tracking is preserved even after repeated reductions.
    - If a later stage uses subset sizes larger than the number of remaining
      features, validation will raise an error.
    """
    # ------------------------------------------------------------------
    # 1. Validate top-level inputs
    # ------------------------------------------------------------------
    # Validate the core inputs and capture the starting feature-space shape.
    X, y = _validate_X_y(X, y)
    n_samples, n_features = X.shape

    # Validate feature names against the input matrix and validate optional
    # group IDs if they were provided.
    feature_names_list = _validate_feature_names(
        feature_names,
        n_features,
        require_unique=True,
    )
    groups = _validate_groups(groups, n_samples, required=False)

    # Validate and unpack the top-level pipeline config into reusable pieces:
    # shared defaults, the model registry, and the ordered stage definitions.
    defaults, models_registry, stages = _validate_pipeline_cfg(cfg)

    # ------------------------------------------------------------------
    # 2. Run full pipeline separately for each model
    # ------------------------------------------------------------------
    final_by_model: Dict[str, Any] = {}

    for model_name, model_estimator in models_registry.items():
        # Start each model from the full feature set. These variables are updated
        # stage by stage as the feature space is reduced.
        X_current = X.copy()
        names_current = list(feature_names_list)
        original_indices_current = np.arange(n_features, dtype=int)

        # `history` keeps the ordered stage outputs, while `by_stage` provides
        # direct lookup by stage name.
        history: List[Dict[str, Any]] = []
        by_stage: Dict[str, Dict[str, Any]] = {}

        for stage_idx, stage in enumerate(stages):
            # Resolve the stage name and merge stage-specific overrides on top of
            # the shared defaults to create the config used for this stage only.
            stage_name = stage.get("name", f"stage_{stage_idx}")
            stage_cfg = _deep_merge(defaults, stage)

            # Run the stage using exactly one model so each model gets its own
            # independent rank-and-select path through the pipeline.
            stage_cfg["models"] = {model_name: model_estimator}

            engine_out = balanced_permutation_rank_select_stage(
                X=X_current,
                y=y,
                groups=groups,
                feature_names=names_current,
                cfg=stage_cfg,
                original_feature_indices=original_indices_current,
            )

            # Extract the ranking output and the selected features for this model
            # from the stage engine result.
            selected_model_out = engine_out["selected_by_model"][model_name]
            ranking_model_out = engine_out["final_ranking_by_model"][model_name]
            detail_model_out = engine_out["detailed_results_by_model"][model_name]

            X_next = selected_model_out["X"]
            names_next = list(selected_model_out["selected_feature_names"])

            # Validate that the returned local indices are aligned with the stage
            # output before recording and propagating them to the next stage.
            selected_idx_local = _validate_stage_selection_output(
                selected_model_out["selected_feature_indices_local"],
                names_next,
                len(original_indices_current),
                stage_name,
            )
            selected_idx_original = selected_model_out["selected_feature_indices"]

            # Record a complete snapshot of this stage so the caller can inspect
            # rankings, selected features, and the exact config that was used.
            stage_out = {
                "stage": stage_name,
                "top_k": int(stage_cfg["top_k"]),
                "n_features_in": int(X_current.shape[1]),
                "n_features_out": int(X_next.shape[1]),
                "cfg_used": deepcopy(stage_cfg),
                "final_ranking": ranking_model_out,
                "detailed_results": detail_model_out,
                "selected_feature_names": names_next,
                "selected_feature_indices": np.asarray(selected_idx_original, dtype=int).copy(),
                "selected_feature_indices_local": selected_idx_local.copy(),
                "X_selected": X_next,
            }

            history.append(stage_out)
            by_stage[stage_name] = stage_out

            # Feed the selected feature matrix forward so the next stage operates
            # only on the features retained by the current stage.
            X_current = X_next
            names_current = names_next
            original_indices_current = np.asarray(selected_idx_original, dtype=int).copy()

        # Store the final reduced matrix and the full stage history for this model.
        final_by_model[model_name] = {
            "X": X_current,
            "feature_names_selected": names_current,
            "selected_feature_indices": original_indices_current.copy(),
            "history": history,
            "by_stage": by_stage,
        }

    return {
        "final_by_model": final_by_model,
        "y":y,
    }


# ============================================================
# Synthetic data: Test function used for feature selection 
# ============================================================

def make_synthetic_feature_selection_dataset(
    *,
    task_type: str = "classification",
    use_groups: bool = False,
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 10,
    n_redundant: int = 0,
    n_repeated: int = 0,
    n_classes: int = 2,
    shuffle: bool = False,
    random_state: int = 42,
    n_groups: Optional[int] = None,
    group_noise_std: float = 0.05,
    regression_noise: float = 0.0,
    effective_rank: Optional[int] = None,
    tail_strength: float = 0.5,
    bias: float = 0.0,
    add_collinear_features: bool = False,
    n_collinear: int = 0,
    collinearity_strength: float = 0.95,
    collinear_source: str = "informative",   # "informative" | "noise" | "any"
    collinear_sign: str = "positive",        # "positive" | "negative" | "mixed"
    collinear_noise_std: float = 0.05,
) -> Dict[str, Any]:
    """
    Create a synthetic dataset for testing feature selection.

    Supports both classification and regression, with optional grouped data and
    optional explicit pairwise-collinear feature injection.

    Modes
    -----
    1) Ungrouped mode
    - classification: standard `make_classification` dataset
    - regression: standard `make_regression` dataset

    2) Grouped mode
    - create `n_groups` base samples
    - repeat each base sample evenly so the total number of rows equals
        `n_samples`
    - add Gaussian noise to the features within each group
    - keep the target fixed within each group

    3) Optional collinearity injection
    - overwrite selected columns with noisy copies of source columns
    - control the number, strength, sign, and source pool of induced
        pairwise-collinear features
    - useful for testing `remove_pairwise_collinear_features(...)`

    Parameters
    ----------
    task_type : str, default="classification"
        Either "classification" or "regression".

    use_groups : bool, default=False
        Whether to generate grouped data.

    n_samples : int, default=1000
        Total number of rows in the returned dataset.

    n_features : int, default=20
        Total number of features.

    n_informative : int, default=10
        Number of informative features passed to the sklearn generator.

    n_redundant : int, default=0
        Number of redundant features passed to `make_classification`.
        Ignored for regression generation.

    n_repeated : int, default=0
        Number of repeated features passed to `make_classification`.
        Ignored for regression generation.

    n_classes : int, default=2
        Number of target classes for classification.
        Ignored when `task_type="regression"`.

    shuffle : bool, default=False
        Whether to shuffle features in the sklearn generator.

    random_state : int, default=42
        Random seed for reproducibility.

    n_groups : Optional[int], default=None
        Number of groups to generate when `use_groups=True`.
        If None and `use_groups=True`, defaults to 50.

    group_noise_std : float, default=0.05
        Standard deviation of Gaussian noise added to each replicated feature row
        within a group.

    regression_noise : float, default=0.0
        Standard deviation of the Gaussian noise applied to the regression target
        by `make_regression`. Ignored for classification.

    effective_rank : Optional[int], default=None
        Effective rank passed to `make_regression`. Ignored for classification.

    tail_strength : float, default=0.5
        Tail strength passed to `make_regression`. Ignored for classification.

    bias : float, default=0.0
        Bias term passed to `make_regression`. Ignored for classification.

    add_collinear_features : bool, default=False
        If True, inject explicit pairwise-collinear features after the base dataset
        has been generated.

    n_collinear : int, default=0
        Number of columns to overwrite with noisy copies of source columns when
        `add_collinear_features=True`.

    collinearity_strength : float, default=0.95
        Multiplicative strength of the copied source feature when constructing a
        synthetic collinear target feature. Larger values generally produce stronger
        absolute correlations.

    collinear_source : str, default="informative"
        Source pool used when selecting columns to copy from.
        One of:
        - "informative": source columns are drawn only from informative features
        - "noise": source columns are drawn only from non-informative features
        - "any": source columns are drawn from any feature

    collinear_sign : str, default="positive"
        Sign of the copied relationship used for synthetic collinearity.
        One of:
        - "positive": all injected pairs are positively correlated
        - "negative": all injected pairs are negatively correlated
        - "mixed": each injected pair randomly uses either +1 or -1

    collinear_noise_std : float, default=0.05
        Standard deviation multiplier for additive Gaussian noise used when creating
        synthetic collinear target columns. This is scaled relative to the source
        feature's standard deviation.

    Returns
    -------
    out : Dict[str, Any]
        {
            "X": np.ndarray,
            "y": np.ndarray,
            "groups": np.ndarray | None,
            "feature_names": list[str],
            "true_informative": set[str],
            "collinearity_info": {
                "enabled": bool,
                "n_collinear_requested": int,
                "n_collinear_applied": int,
                "collinearity_strength": float,
                "collinear_source": str,
                "collinear_sign": str,
                "collinear_noise_std": float,
                "pairs": list[dict],
            },
        }

    Notes
    -----
    - In grouped mode, `n_samples` must be divisible by `n_groups` so that
    each group has the same number of rows.
    - Example: if `n_samples=500` and `n_groups=5`, then each group will have
    `500 // 5 = 100` rows.
    - In grouped mode, targets are held constant within each group so the output
    is compatible with one-row-per-group resampling.
    - Collinearity is injected after base dataset generation so correlation-based
    feature pruning can be tested directly and predictably.
    - Injected collinear target columns preferentially copy from earlier columns
    when possible, which aligns with the current column-order priority rule used
    in `remove_pairwise_collinear_features(...)`.
    """
    if task_type not in {"classification", "regression"}:
        raise ValueError(
            f"task_type must be 'classification' or 'regression'; got {task_type!r}."
        )

    if n_informative < 1:
        raise ValueError(f"n_informative must be >= 1; got {n_informative}.")
    if n_informative > n_features:
        raise ValueError(
            f"n_informative ({n_informative}) cannot exceed n_features ({n_features})."
        )

    if not (0.0 <= collinearity_strength <= 1.0):
        raise ValueError(
            f"collinearity_strength must be between 0 and 1; got {collinearity_strength}."
        )

    allowed_collinear_sources = {"informative", "noise", "any"}
    if collinear_source not in allowed_collinear_sources:
        raise ValueError(
            f"collinear_source must be one of {sorted(allowed_collinear_sources)}; "
            f"got {collinear_source!r}."
        )

    allowed_collinear_signs = {"positive", "negative", "mixed"}
    if collinear_sign not in allowed_collinear_signs:
        raise ValueError(
            f"collinear_sign must be one of {sorted(allowed_collinear_signs)}; "
            f"got {collinear_sign!r}."
        )

    if n_collinear < 0:
        raise ValueError(f"n_collinear must be >= 0; got {n_collinear}.")
    if collinear_noise_std < 0:
        raise ValueError(
            f"collinear_noise_std must be >= 0; got {collinear_noise_std}."
        )

    rng = np.random.default_rng(random_state)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    true_informative = {f"feature_{i}" for i in range(n_informative)}

    def _generate_base_dataset(n_rows: int) -> Tuple[np.ndarray, np.ndarray]:
        if task_type == "classification":
            X_base, y_base = make_classification(
                n_samples=n_rows,
                n_features=n_features,
                n_informative=n_informative,
                n_redundant=n_redundant,
                n_repeated=n_repeated,
                n_classes=n_classes,
                shuffle=shuffle,
                random_state=random_state,
            )
            return np.asarray(X_base, dtype=float), np.asarray(y_base)

        X_base, y_base = make_regression(
            n_samples=n_rows,
            n_features=n_features,
            n_informative=n_informative,
            noise=regression_noise,
            effective_rank=effective_rank,
            tail_strength=tail_strength,
            bias=bias,
            shuffle=shuffle,
            random_state=random_state,
        )
        return np.asarray(X_base, dtype=float), np.asarray(y_base, dtype=float)

    def _inject_collinear_features(X_in: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Overwrite selected columns with noisy copies of source columns to induce
        explicit pairwise collinearity.
        """
        X_out = X_in.copy()

        informative_idx = np.arange(n_informative, dtype=int)
        noise_idx = np.arange(n_informative, n_features, dtype=int)
        all_idx = np.arange(n_features, dtype=int)

        if collinear_source == "informative":
            source_pool = informative_idx
        elif collinear_source == "noise":
            source_pool = noise_idx
        else:
            source_pool = all_idx

        if len(source_pool) == 0:
            raise ValueError(
                f"collinear_source={collinear_source!r} produced an empty source pool. "
                "Adjust n_informative or n_features."
            )

        # To preserve earlier-column priority for remove_pairwise_collinear_features,
        # overwrite later columns using earlier columns as sources whenever possible.
        candidate_targets = np.arange(1, n_features, dtype=int)

        if len(candidate_targets) == 0:
            raise ValueError("Need at least 2 features to inject collinearity.")

        n_collinear_effective = min(n_collinear, len(candidate_targets))
        if add_collinear_features and n_collinear > len(candidate_targets):
            raise ValueError(
                f"n_collinear ({n_collinear}) cannot exceed the number of eligible "
                f"target columns ({len(candidate_targets)})."
            )

        target_indices = rng.choice(
            candidate_targets,
            size=n_collinear_effective,
            replace=False,
        )

        collinear_pairs: List[Dict[str, Any]] = []

        for target_idx in sorted(target_indices.tolist()):
            valid_source_pool = source_pool[source_pool != target_idx]

            # Strong preference for earlier columns so pruning keeps the earlier feature
            earlier_sources = valid_source_pool[valid_source_pool < target_idx]
            if len(earlier_sources) > 0:
                source_idx = int(rng.choice(earlier_sources))
            else:
                source_idx = int(rng.choice(valid_source_pool))

            if collinear_sign == "positive":
                sign = 1.0
            elif collinear_sign == "negative":
                sign = -1.0
            else:
                sign = float(rng.choice([-1.0, 1.0]))

            source_values = X_out[:, source_idx]
            source_std = float(np.std(source_values))

            # Scale copy noise relative to the source feature scale so
            # collinearity_strength behaves more predictably.
            noise = rng.normal(
                loc=0.0,
                scale=collinear_noise_std * max(source_std, 1e-12),
                size=X_out.shape[0],
            )

            X_out[:, target_idx] = (
                sign * collinearity_strength * source_values + noise
            )

            collinear_pairs.append(
                {
                    "source_feature_index": source_idx,
                    "source_feature_name": feature_names[source_idx],
                    "target_feature_index": target_idx,
                    "target_feature_name": feature_names[target_idx],
                    "sign": sign,
                }
            )

        return X_out, {
            "enabled": add_collinear_features,
            "n_collinear_requested": n_collinear,
            "n_collinear_applied": n_collinear_effective,
            "collinearity_strength": collinearity_strength,
            "collinear_source": collinear_source,
            "collinear_sign": collinear_sign,
            "collinear_noise_std": collinear_noise_std,
            "pairs": collinear_pairs,
        }

    if not use_groups:
        X, y = _generate_base_dataset(n_samples)
        groups = None

    else:
        if n_groups is None:
            n_groups = 50

        n_groups = int(n_groups)

        if n_groups < 1:
            raise ValueError("When use_groups=True, n_groups must be >= 1.")

        if n_samples % n_groups != 0:
            raise ValueError(
                f"When use_groups=True, n_samples ({n_samples}) must be divisible "
                f"by n_groups ({n_groups})."
            )

        rows_per_group = n_samples // n_groups
        X_group, y_group = _generate_base_dataset(n_groups)

        X_rows = []
        y_rows = []
        group_ids = []

        for g in range(n_groups):
            base_x = X_group[g]
            base_y = y_group[g]

            for _ in range(rows_per_group):
                x_rep = base_x + rng.normal(0.0, group_noise_std, size=n_features)
                X_rows.append(x_rep)
                y_rows.append(base_y)
                group_ids.append(g)

        X = np.asarray(X_rows, dtype=float)
        y = np.asarray(y_rows)
        groups = np.asarray(group_ids)

    # Inject explicit pairwise-collinear structure after the base dataset
    # has been generated so collinearity pruning can be tested directly.
    if add_collinear_features and n_collinear > 0:
        X, collinearity_info = _inject_collinear_features(X)
    else:
        collinearity_info = {
            "enabled": False,
            "n_collinear_requested": n_collinear,
            "n_collinear_applied": 0,
            "collinearity_strength": collinearity_strength,
            "collinear_source": collinear_source,
            "collinear_sign": collinear_sign,
            "collinear_noise_std": collinear_noise_std,
            "pairs": [],
        }

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "feature_names": feature_names,
        "true_informative": true_informative,
        "collinearity_info": collinearity_info,
    }


# ============================================================
# Example
# ============================================================

# # Classification test data
# data = make_synthetic_feature_selection_dataset(
#     task_type="classification",
#     use_groups=False,
#     n_samples=500,
#     n_features=20,
#     n_informative=10,
#     random_state=42,
#     add_collinear_features=True,
#     n_collinear=4,
#     collinearity_strength=0.98,
#     collinear_source="informative",
#     collinear_sign="mixed",
#     collinear_noise_std=0.02,
# )

# # Regression test data
# data = make_synthetic_feature_selection_dataset(
#     task_type="regression",
#     use_groups=False,
#     n_samples=500,
#     n_features=20,
#     n_informative=10,
#     shuffle=False,
#     random_state=42,
#     regression_noise=2.0,
# )

# collinear_pruned = remove_pairwise_collinear_features(
#     X=data["X"],              # Feature matrix (n_samples, n_features) to run selection on (here: scaled features)
#     groups=data["groups"],    # Group IDs per row (used for "local" constant checks + group-aware correlation bootstrapping)
#     methods_config={          # How to compute correlations for collinearity checks
#         "corr": {
#             "method": "spearman", # Correlation type ("spearman" recommended for monotonic/rank relationships)
#             "min_periods": 1,     # Minimum observations required per pairwise correlation
#             "numeric_only": False, # Use all columns (X is numeric anyway; kept for consistency)
#         }
#     },
#     N=10,                                   # If groups is provided: number of bootstrap subsamples (one row per group) to average corr matrices over
#     feature_names=data["feature_names"],    # List of feature names aligned to X_raw columns
#     threshold=0.8,                          # Drop feature j if |corr(i, j)| >= threshold (keeping earlier feature i, dropping later j)
#     # parallelization controls
#     random_state=42,      # Seed for RNG used in group-aware subsampling. When `parallelize=True`, seeds are generated up-front for reproducibility.
#     parallelize=True,     # If True (and groups provided), parallelize the N bootstrap correlation computations
#     n_jobs=-1,            # Number of parallel workers (-1 = use all CPU cores)
#     backend="loky",       #  Passed to joblib.Parallel. "loky" uses process-based parallelism.
# )



# RANK_SELECT_PIPELINE_CFG = {
#     "defaults": {
#         "task_type": "classification",       # "classification" or "regression"
#         "scoring": "roc_auc",                # metric used by permutation_importance to score validation performance
#         "n_splits": 5,                       # number of CV folds used in the selected splitter
#         "n_repeats": 10,                     # number of permutation shuffles per feature inside permutation_importance
#         "target_feature_appearances": 20,    # target number of times each feature should appear across sampled subsets
#         "random_state": 42,                  # base random seed for reproducibility
#         "ranking_metric": "auto",            # final ranking rule: auto, mean_normalized_rank, or mean_importance
#         "group_mode": False,                 # if True, repeatedly sample one row per group before ranking
#         "group_iterations": 10,              # number of one-row-per-group resampling iterations when group_mode=True
#     },
#     "models": {
#         "logistic_regression": LogisticRegression(max_iter=20000),   # model to rank features with
#         #"XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),  # second model to rank features with
#     },
#     "stages": [
#         {
#             "name": "coarse_rank_select",    # stage label used in output/history
#             "top_k": 10,                     # number of highest-ranked features to keep after this stage
#             "subset_sizes": [5, 10],         # feature subset sizes tested during ranking in this stage
#         },
#         {
#             "name": "final_rank_select",     # second-stage label used in output/history
#             "top_k": 5,                      # number of highest-ranked features to keep after this stage
#             "subset_sizes": [5, 10],         # feature subset sizes tested during ranking in this stage
#         },
#     ],
# }

# out = balanced_permutation_rank_select_pipeline(
#     X=collinear_pruned['X_collinear_pruned'],
#     y=data["y"],
#     groups=data["groups"],
#     feature_names=collinear_pruned['feature_names'],
#     cfg=RANK_SELECT_PIPELINE_CFG,
# )






from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_validate
from sklearn.model_selection._split import BaseCrossValidator  # for typing

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Type, Mapping


from typing import Any, Dict, List, Optional, Sequence
import numpy as np



from typing import Any, Dict, List, Optional, Sequence
import numpy as np


# ---------------------------------------------------------------------
# Nested cross-validation feature selection 
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


def build_nested_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Dict[str, Any],
    groups: Optional[np.ndarray] = None,
    model_selection: str = "StratifiedKFold",
) -> List[Dict[str, Any]]:
    """
    Build and return a reusable **nested cross-validation split plan**.

    This function creates a deterministic “plan” of indices that defines:
      - an **outer CV** split (train vs test), repeated `num_trials` times, and
      - for each outer-train split, an **inner CV** split (train vs validation)
        generated strictly within the outer-train portion.

    The key idea is that the returned split plan can be reused across multiple
    downstream steps (e.g., feature selection, ranking, VIF pruning, model training),
    ensuring that every pipeline stage uses the exact same train/test partitions.

    Parameters
    ----------
    X : np.ndarray
        Full feature matrix of shape (n_samples, n_features). Used only to drive
        the CV splitters (i.e., shapes); the returned indices always refer to rows
        of this original X.
    y : np.ndarray
        Target vector of shape (n_samples,). Used for stratification (and for
        group-aware stratification where supported).
    cfg : Dict[str, Any]
        Configuration dictionary containing CV settings under `cfg["cv"]`:
          - "num_trials": int
              Number of repeated outer-CV runs (each run uses a different random seed).
          - "n_outer_splits": int
              Number of folds in the outer CV.
          - "n_inner_splits": int
              Number of folds in the inner CV.
    groups : Optional[np.ndarray], default=None
        Optional group labels of shape (n_samples,). Required if
        `model_selection="StratifiedGroupKFold"`. When provided, both outer and
        inner splits are group-aware (no group leakage between train and test/val).
    model_selection : str, default="StratifiedKFold"
        Strategy name used to construct the outer and inner CV splitters via
        `make_outer_inner_cv(...)`. Supported options:
          - "StratifiedKFold"
          - "StratifiedGroupKFold" (requires `groups`)

    Returns
    -------
    split_plan : List[Dict[str, Any]]
        A list of dictionaries, one per **outer fold** across all trials
        (length = num_trials * n_outer_splits). Each dictionary contains:

        Outer-fold metadata
        - "trial": int
            Outer repetition index (0-based).
        - "outer_fold": int
            Outer fold index within the trial (1-based in this implementation).
        - "model_selection": str
            CV strategy used.

        Outer-fold indices (GLOBAL indices into X/y)
        - "outer_train_idx": np.ndarray
            1D integer array of row indices used for outer training.
        - "outer_test_idx": np.ndarray
            1D integer array of row indices used for outer testing.

        Inner splits (GLOBAL indices, all subsets of outer_train_idx)
        - "inner_splits": List[Dict[str, Any]]
            List of length `n_inner_splits`. Each item contains:
              - "inner_fold": int
                  Inner fold index (1-based).
              - "inner_train_idx": np.ndarray
                  1D integer array of GLOBAL row indices for inner training.
              - "inner_val_idx": np.ndarray
                  1D integer array of GLOBAL row indices for inner validation.

        Convenience label slices (aligned to the outer split)
        - "y_outer_train": np.ndarray
            y values for outer_train_idx (same order as outer_train_idx).
        - "y_outer_test": np.ndarray
            y values for outer_test_idx (same order as outer_test_idx).

    Notes
    -----
    - Inner CV splits are generated **only** from the outer-train portion.
      The outer-test data never influences inner splitting.
    - The indices stored in the plan are **GLOBAL indices** into the original X/y.
      This makes it easy to slice any derived matrices later (scaled X, pruned X, etc.)
      as long as rows correspond to the original sample ordering.
    - Group-aware behavior:
        If `groups` is provided and `model_selection="StratifiedGroupKFold"`,
        each group will appear in only one side of a split (no leakage).
    - This function assumes you already have `make_outer_inner_cv(...)` implemented
      and that it returns `(outer_cv, inner_cv)` objects exposing a scikit-learn-like
      `.split(X, y[, groups])` generator.
    """

    cv_cfg = cfg["cv"]
    NUM_TRIALS = cv_cfg["num_trials"]
    n_outer_splits = cv_cfg["n_outer_splits"]
    n_inner_splits = cv_cfg["n_inner_splits"]

    if model_selection == "StratifiedGroupKFold" and groups is None:
        raise ValueError(
            "model_selection='StratifiedGroupKFold' but groups is None. "
            "Provide a groups array or use 'StratifiedKFold'."
        )

    split_plan: List[Dict[str, Any]] = []
    cv_tracker = 0
    total_outer_folds = NUM_TRIALS * n_outer_splits

    for trial_idx in range(NUM_TRIALS):
        outer_cv, inner_cv = make_outer_inner_cv(
            model_selection=model_selection,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            outer_trial_idx=trial_idx,
        )

        # Outer splits (global indices)
        if groups is not None:
            outer_splits = outer_cv.split(X, y, groups)
        else:
            outer_splits = outer_cv.split(X, y)

        outer_fold_idx = 0
        for outer_train_idx, outer_test_idx in outer_splits:
            cv_tracker += 1
            outer_fold_idx += 1

            print(
                f"Outer fold {cv_tracker}/{total_outer_folds} "
                f"(trial {trial_idx}, fold {outer_fold_idx})"
            )

            # Build inner splits on the OUTER-TRAIN subset
            X_train = X[outer_train_idx]
            y_train = y[outer_train_idx]
            groups_train = groups[outer_train_idx] if groups is not None else None

            if groups_train is not None:
                inner_iter = inner_cv.split(X_train, y_train, groups_train)
            else:
                inner_iter = inner_cv.split(X_train, y_train)

            inner_splits: List[Dict[str, Any]] = []
            inner_fold_idx = 0

            for inner_train_local, inner_val_local in inner_iter:
                inner_fold_idx += 1

                # Map local indices (into X_train) back to GLOBAL indices (into X)
                inner_train_idx = outer_train_idx[inner_train_local]
                inner_val_idx = outer_train_idx[inner_val_local]

                inner_splits.append(
                    {
                        "inner_fold": inner_fold_idx,
                        "inner_train_idx": inner_train_idx,
                        "inner_val_idx": inner_val_idx,
                    }
                )

            split_plan.append(
                {
                    "trial": trial_idx,
                    "outer_fold": outer_fold_idx,
                    "model_selection": model_selection,
                    "outer_train_idx": outer_train_idx,
                    "outer_test_idx": outer_test_idx,
                    "inner_splits": inner_splits,
                    "y_outer_train": y_train,
                    "y_outer_test": y[outer_test_idx],
                }
            )

    return split_plan









def nested_cv_balanced_rank_select(
    nested_splits: List[Dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    rank_select_cfg: Dict[str, Any],
    groups: Optional[np.ndarray] = None,
    rank_select_out_key: str = "rank_select_out",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Attach balanced permutation rank-select outputs to each OUTER fold
    in a nested CV split plan.

    For each fold in `nested_splits`, this function:

      1) slices the OUTER-TRAIN portion of X / y / groups
      2) runs `balanced_permutation_rank_select_pipeline(...)` on that
         outer-train subset only
      3) stores the full pipeline output back into the fold dictionary

    This function does NOT use the inner splits directly. The inner CV used by
    the rank-select pipeline is controlled internally by `rank_select_cfg`
    (for example via cfg["defaults"]["n_splits"]).

    Parameters
    ----------
    nested_splits : List[Dict[str, Any]]
        Nested CV split plan. Each fold dict must contain:
        - "outer_train_idx"
        and may contain:
        - "trial"
        - "outer_fold"

    X : np.ndarray of shape (n_samples, n_features)
        Full feature matrix. Each fold slices rows using outer_train_idx.

    y : np.ndarray of shape (n_samples,)
        Full target vector aligned row-wise with X.

    feature_names : Sequence[str]
        Feature names aligned to the columns of X.

    rank_select_cfg : Dict[str, Any]
        Configuration dictionary passed directly into
        `balanced_permutation_rank_select_pipeline(...)`.

    groups : Optional[np.ndarray] of shape (n_samples,), default=None
        Optional group labels aligned row-wise with X. If provided, each fold
        slices the outer-train rows and passes them into the rank-select pipeline.

    rank_select_out_key : str, default="rank_select_out"
        Key used to store the rank-select pipeline output inside each fold dict.

    verbose : bool, default=True
        If True, print per-fold progress messages.

    Returns
    -------
    nested_splits : List[Dict[str, Any]]
        The same list object, mutated in-place with rank-select outputs attached.

    Stored Output
    -------------
    For each fold, this function stores:

        fold[rank_select_out_key] = rank_select_out

    where `rank_select_out` is the full output of
    `balanced_permutation_rank_select_pipeline(...)`, typically including:
    - "final_by_model"
    - "y"

    Notes
    -----
    - Feature selection is run using OUTER-TRAIN data only.
    - This preserves nested-CV discipline: outer-test rows are not used during
      feature selection for that fold.
    - The function stores the full per-fold output so downstream steps can
      inspect:
        - selected feature names
        - selected feature indices
        - stage histories
        - by-stage results
        - outputs for multiple models
    """
    # -----------------------------
    # 1) Basic validation
    # -----------------------------
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D; got shape {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of rows; got {X.shape[0]} and {y.shape[0]}."
        )

    feature_names_list = list(feature_names)
    if len(feature_names_list) != X.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names_list)}) must match "
            f"X.shape[1] ({X.shape[1]})."
        )

    if groups is not None:
        groups = np.asarray(groups)
        if groups.ndim != 1:
            raise ValueError(f"groups must be 1D; got shape {groups.shape}.")
        if len(groups) != X.shape[0]:
            raise ValueError(
                f"groups length ({len(groups)}) must match X.shape[0] ({X.shape[0]})."
            )

    if not isinstance(rank_select_cfg, dict):
        raise TypeError("rank_select_cfg must be a dictionary.")

    if "defaults" not in rank_select_cfg:
        raise KeyError("rank_select_cfg must contain a 'defaults' key.")
    if "models" not in rank_select_cfg:
        raise KeyError("rank_select_cfg must contain a 'models' key.")
    if "stages" not in rank_select_cfg:
        raise KeyError("rank_select_cfg must contain a 'stages' key.")

    # -----------------------------
    # 2) Per outer fold
    # -----------------------------
    for i, fold in enumerate(nested_splits):
        if "outer_train_idx" not in fold:
            raise KeyError(f"Fold {i} is missing required key 'outer_train_idx'.")

        trial = fold.get("trial", None)
        outer_fold = fold.get("outer_fold", None)

        outer_train_idx = np.asarray(fold["outer_train_idx"], dtype=int)

        X_train = X[outer_train_idx]
        y_train = y[outer_train_idx]
        groups_train = groups[outer_train_idx] if groups is not None else None

        if verbose:
            print(
                f"[RANK_SELECT] trial={trial} outer_fold={outer_fold} | "
                f"X_train={X_train.shape} | n_models={len(rank_select_cfg['models'])}"
            )

        rank_select_out = balanced_permutation_rank_select_pipeline(
            X=X_train,
            y=y_train,
            groups=groups_train,
            feature_names=feature_names_list,
            cfg=rank_select_cfg,
        )

        fold[rank_select_out_key] = rank_select_out

        if verbose:
            final_by_model = rank_select_out["final_by_model"]
            summary_parts = []

            for model_name, model_out in final_by_model.items():
                n_selected = len(model_out["feature_names_selected"])
                summary_parts.append(f"{model_name}: {n_selected} feats")

            summary_text = " | ".join(summary_parts)
            print(
                f"[DONE] trial={trial} outer_fold={outer_fold} | {summary_text}"
            )

    return nested_splits


