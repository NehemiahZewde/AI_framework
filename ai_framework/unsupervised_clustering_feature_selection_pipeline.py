# ============================================================
# Unsupervised clustering-based feature selection
# ============================================================
#
# Add this section to the same module/file that already contains your shared
# supervised feature-selection helpers, including:
#   - _deep_merge
#   - _validate_feature_names
#   - _validate_groups
#   - _validate_subset_sizes
#   - _validate_ranking_metric
#   - _resolve_effective_ranking_metric
#   - _validate_models_dict
#   - _validate_original_feature_indices
#   - _validate_stage_selection_output
#   - _validate_pipeline_cfg
#   - sample_one_row_per_group
#
# This code intentionally does NOT use y, ROC AUC, supervised CV, validation_fraction,
# or sklearn's supervised permutation_importance.

from __future__ import annotations
from typing import Any, DefaultDict, Dict, Hashable, List, Optional, Sequence, Tuple, Literal, Mapping, Iterable, Set
from collections import defaultdict
from copy import deepcopy
from math import ceil

import numpy as np
import pandas as pd
from sklearn.base import clone
from tqdm.auto import tqdm



ClusteringScoring = Literal[
    # Unsupervised / geometry-based clustering scores
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
    "inertia",

    # Label-informed clustering scores
    "adjusted_rand",
    "normalized_mutual_info",
    "v_measure",
    "homogeneity",
    "completeness",
]

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from itertools import combinations

from scipy.stats import kendalltau, spearmanr


MissingFeaturePolicy = Literal[
    "intersection",
    "union_fill_zero",
    "union_worst_rank",
]

ScoreMissingPolicy = Literal[
    "intersection",
    "union_fill_zero",
    "union_keep_na",
]


from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    homogeneity_score,
    completeness_score,
)


# ============================================================
# Shared validation helpers
# ============================================================
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


# ============================================================
# Clustering validation / scoring helpers
# ============================================================

def _validate_X_only(X: np.ndarray) -> np.ndarray:
    """Validate unsupervised-learning feature matrix and return normalized numpy array."""
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(f"X must be a 2D numpy array; got shape {X.shape}.")
    if X.shape[0] < 2:
        raise ValueError("X must contain at least 2 rows for clustering.")
    if X.shape[1] < 1:
        raise ValueError("X must contain at least 1 feature for clustering.")

    return X


def _requires_y_for_clustering_score(scoring: str) -> bool:
    """
    Return True if the clustering score requires target labels y.

    Some clustering scores evaluate only geometric cluster structure in X-space.
    Others compare the discovered cluster labels to known target labels. This
    helper centralizes that distinction so downstream functions can decide when
    y is required.

    Parameters
    ----------
    scoring : str
        Clustering scoring metric.

    Returns
    -------
    bool
        True if scoring requires y, otherwise False.

    Examples
    --------
    - "silhouette" -> False
    - "adjusted_rand" -> True
    """
    scoring = _validate_clustering_scoring(scoring)

    label_informed_scores = {
        "adjusted_rand",
        "normalized_mutual_info",
        "v_measure",
        "homogeneity",
        "completeness",
    }

    return scoring in label_informed_scores


def _validate_y_for_clustering_score(
    y: Optional[np.ndarray],
    *,
    n_samples: int,
    scoring: str,
) -> Optional[np.ndarray]:
    """
    Validate target labels for label-informed clustering scores.

    For unsupervised clustering scores, y is optional and returned as None when
    not provided. For label-informed clustering scores, y is required because
    the score compares cluster assignments against the true labels.

    Parameters
    ----------
    y : Optional[np.ndarray]
        Target labels aligned row-wise with X. Required when scoring is one of:
        - "adjusted_rand"
        - "normalized_mutual_info"
        - "v_measure"
        - "homogeneity"
        - "completeness"

    n_samples : int
        Number of rows in the feature matrix X that will be clustered/scored.

    scoring : str
        Clustering scoring metric.

    Returns
    -------
    Optional[np.ndarray]
        Validated 1D y array if provided, otherwise None.

    Raises
    ------
    ValueError
        If y is required but missing, has the wrong shape, has the wrong length,
        or contains fewer than two unique labels.
    """
    scoring = _validate_clustering_scoring(scoring)

    # If this metric does not use y, allow y to be omitted.
    if not _requires_y_for_clustering_score(scoring):
        if y is None:
            return None

        y_array = np.asarray(y)

        if y_array.ndim != 1:
            raise ValueError(f"y must be 1D when provided; got shape {y_array.shape}.")
        if len(y_array) != n_samples:
            raise ValueError(
                f"y length ({len(y_array)}) must match number of rows in X ({n_samples})."
            )

        return y_array

    # Label-informed scores require y.
    if y is None:
        raise ValueError(
            f"scoring={scoring!r} requires y because it is a label-informed "
            "clustering metric."
        )

    y_array = np.asarray(y)

    if y_array.ndim != 1:
        raise ValueError(f"y must be 1D for scoring={scoring!r}; got shape {y_array.shape}.")

    if len(y_array) != n_samples:
        raise ValueError(
            f"y length ({len(y_array)}) must match number of rows in X ({n_samples}) "
            f"for scoring={scoring!r}."
        )

    unique_y = np.unique(y_array)

    if len(unique_y) < 2:
        raise ValueError(
            f"scoring={scoring!r} requires y to contain at least 2 unique labels; "
            f"found {len(unique_y)}."
        )

    return y_array


def _validate_clustering_scoring(scoring: str) -> str:
    """
    Validate clustering scoring metric.

    Supported metrics are split into two families:

    1) Unsupervised / geometry-based metrics
       These evaluate cluster structure using only X and the cluster labels.

       - "silhouette"
       - "calinski_harabasz"
       - "davies_bouldin"
       - "inertia"

    2) Label-informed clustering metrics
       These evaluate how well the cluster assignments recover the provided
       target labels y.

       - "adjusted_rand"
       - "normalized_mutual_info"
       - "v_measure"
       - "homogeneity"
       - "completeness"

    Notes
    -----
    This validator only checks that the scoring string is known. It does not
    check whether y was provided. Use _validate_y_for_clustering_score(...) for
    that.
    """
    allowed = {
        # Unsupervised / geometry-based scores
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "inertia",

        # Label-informed scores
        "adjusted_rand",
        "normalized_mutual_info",
        "v_measure",
        "homogeneity",
        "completeness",
    }

    if scoring not in allowed:
        raise ValueError(
            f"clustering scoring must be one of {sorted(allowed)}; got {scoring!r}."
        )

    return scoring



def _validate_clustering_estimator(
    estimator: Any,
    *,
    model_name: str,
) -> None:
    """Validate that a clustering estimator can be fit and scored in the simplified pipeline."""
    has_fit_predict = hasattr(estimator, "fit_predict")
    has_fit_and_predict = hasattr(estimator, "fit") and hasattr(estimator, "predict")

    if not (has_fit_predict or has_fit_and_predict):
        raise ValueError(
            f"Clustering model {model_name!r} must implement fit_predict(...) "
            f"or fit(...) + predict(...)."
        )


def _validate_clustering_row_subsampling(
    *,
    enabled: bool,
    train_fraction: float,
    n_samples: int,
) -> None:
    """Validate optional row subsampling for clustering ranking runs."""
    if not (0.0 < train_fraction <= 1.0):
        raise ValueError(
            f"row_subsampling['train_fraction'] must be in (0, 1]; got {train_fraction}."
        )

    if enabled:
        n_rows = int(np.floor(n_samples * train_fraction))
        if n_rows < 2:
            raise ValueError(
                "Row subsampling leaves too few rows for clustering ranking. "
                f"n_samples={n_samples}, train_fraction={train_fraction}, rows_kept={n_rows}."
            )


def _sample_rows_unsupervised(
    n_samples: int,
    *,
    train_fraction: float,
    random_state: Optional[int],
) -> np.ndarray:
    """Sample rows for one unsupervised ranking run."""
    if not (0.0 < train_fraction <= 1.0):
        raise ValueError(f"train_fraction must be in (0, 1]; got {train_fraction}.")

    all_indices = np.arange(n_samples)

    if train_fraction == 1.0:
        return all_indices

    rng = np.random.default_rng(random_state)
    n_keep = int(np.floor(n_samples * train_fraction))
    n_keep = max(2, min(n_keep, n_samples))

    return np.sort(rng.choice(all_indices, size=n_keep, replace=False))


def _fit_predict_clusters(
    clusterer: Any,
    X: np.ndarray,
) -> Tuple[Any, np.ndarray]:
    """Clone the clusterer, fit it on X, and return fitted model plus labels."""
    fitted = clone(clusterer)

    if hasattr(fitted, "fit_predict"):
        labels = fitted.fit_predict(X)
    else:
        fitted.fit(X)
        if not hasattr(fitted, "predict"):
            raise ValueError(
                "Clusterer must implement fit_predict(...) or fit(...) + predict(...)."
            )
        labels = fitted.predict(X)

    return fitted, np.asarray(labels)


def _score_clusters(
    fitted_clusterer: Any,
    X: np.ndarray,
    labels: np.ndarray,
    scoring: str,
    y: Optional[np.ndarray] = None,
) -> float:
    """
    Score clustering quality so that higher is always better.

    Supported unsupervised / geometry-based scores:
    - silhouette: higher is better
    - calinski_harabasz: higher is better
    - davies_bouldin: lower is better, returned as negative DB so higher is better
    - inertia: lower is better for KMeans-like estimators, returned as negative inertia

    Supported label-informed scores:
    - adjusted_rand: compares cluster labels against y, adjusted for chance
    - normalized_mutual_info: mutual information between cluster labels and y
    - v_measure: harmonic mean of homogeneity and completeness
    - homogeneity: each cluster contains only members of a single class
    - completeness: all members of a class are assigned to the same cluster

    Parameters
    ----------
    fitted_clusterer : Any
        Fitted clustering estimator. Required for scoring="inertia" because the
        estimator must expose inertia_.

    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix used for clustering/scoring.

    labels : np.ndarray of shape (n_samples,)
        Cluster labels produced by the clustering estimator.

    scoring : str
        Clustering scoring metric.

    y : Optional[np.ndarray] of shape (n_samples,), default=None
        True target labels. Required for label-informed scoring metrics.

    Returns
    -------
    float
        Clustering score where higher is always better.

    Notes
    -----
    Label-informed scores do not change how the clustering model is fit. They
    only change how the resulting cluster assignments are evaluated.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    scoring = _validate_clustering_scoring(scoring)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {X.shape}.")

    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D; got shape {labels.shape}.")

    if len(labels) != X.shape[0]:
        raise ValueError(
            f"labels length ({len(labels)}) must match number of rows in X ({X.shape[0]})."
        )

    # Validate y only when needed, while still checking shape if y is provided.
    y_array = _validate_y_for_clustering_score(
        y,
        n_samples=X.shape[0],
        scoring=scoring,
    )

    unique_labels = np.unique(labels)
    n_clusters_found = len(unique_labels)

    # Geometry-based scores require at least 2 clusters and fewer clusters than rows.
    if scoring in {"silhouette", "calinski_harabasz", "davies_bouldin"}:
        if n_clusters_found < 2 or n_clusters_found >= X.shape[0]:
            return np.nan

    if scoring == "silhouette":
        return float(silhouette_score(X, labels))

    if scoring == "calinski_harabasz":
        return float(calinski_harabasz_score(X, labels))

    if scoring == "davies_bouldin":
        return -float(davies_bouldin_score(X, labels))

    if scoring == "inertia":
        if not hasattr(fitted_clusterer, "inertia_"):
            raise ValueError("scoring='inertia' requires the fitted clusterer to expose inertia_.")
        return -float(fitted_clusterer.inertia_)

    # Label-informed scores compare discovered cluster labels to the known labels y.
    # For Goal B, these are the main scores of interest.
    if scoring == "adjusted_rand":
        return float(adjusted_rand_score(y_array, labels))

    if scoring == "normalized_mutual_info":
        return float(normalized_mutual_info_score(y_array, labels))

    if scoring == "v_measure":
        return float(v_measure_score(y_array, labels))

    if scoring == "homogeneity":
        return float(homogeneity_score(y_array, labels))

    if scoring == "completeness":
        return float(completeness_score(y_array, labels))

    raise ValueError(f"Unsupported clustering scoring metric: {scoring!r}.")

# def _score_clusters(
#     fitted_clusterer: Any,
#     X: np.ndarray,
#     labels: np.ndarray,
#     scoring: str,
# ) -> float:
#     """
#     Score clustering quality so that higher is always better.

#     Supported scores:
#     - silhouette: higher is better
#     - calinski_harabasz: higher is better
#     - davies_bouldin: lower is better, returned as negative DB so higher is better
#     - inertia: lower is better for KMeans-like estimators, returned as negative inertia
#     """
#     X = np.asarray(X)
#     labels = np.asarray(labels)
#     scoring = _validate_clustering_scoring(scoring)

#     unique_labels = np.unique(labels)
#     n_clusters_found = len(unique_labels)

#     if scoring in {"silhouette", "calinski_harabasz", "davies_bouldin"}:
#         if n_clusters_found < 2 or n_clusters_found >= X.shape[0]:
#             return np.nan

#     if scoring == "silhouette":
#         return float(silhouette_score(X, labels))

#     if scoring == "calinski_harabasz":
#         return float(calinski_harabasz_score(X, labels))

#     if scoring == "davies_bouldin":
#         return -float(davies_bouldin_score(X, labels))

#     if scoring == "inertia":
#         if not hasattr(fitted_clusterer, "inertia_"):
#             raise ValueError("scoring='inertia' requires the fitted clusterer to expose inertia_.")
#         return -float(fitted_clusterer.inertia_)

#     raise ValueError(f"Unsupported clustering scoring metric: {scoring!r}.")


# ============================================================
# Clustering permutation importance
# ============================================================
def clustering_permutation_importance(
    clusterer: Any,
    X: np.ndarray,
    feature_names: Sequence[str],
    *,
    scoring: str = "silhouette",
    y: Optional[np.ndarray] = None,
    n_repeats: int = 10,
    random_state: Optional[int] = 42,
) -> pd.Series:
    """
    Compute same-sample permutation importance for clustering feature selection.

    This function supports two scoring modes:

    1) Unsupervised / geometry-based clustering scoring
       - Fit the clustering model on X.
       - Score the cluster structure using X and the discovered cluster labels.
       - Examples: silhouette, calinski_harabasz, davies_bouldin, inertia.

    2) Label-informed clustering scoring
       - Fit the clustering model on X without using y.
       - Score the discovered cluster labels against y.
       - Examples: adjusted_rand, normalized_mutual_info, v_measure,
         homogeneity, completeness.

    Procedure
    ---------
    1) Fit the clustering model on the sampled rows and selected features.
    2) Score clustering quality on that same sampled dataset.
    3) Shuffle one feature at a time.
    4) Refit the clustering model on the shuffled dataset.
    5) Re-score clustering quality.

    importance(feature_j) = baseline_cluster_score - shuffled_refit_cluster_score

    Large positive importance means shuffling the feature damages clustering quality.
    Near-zero importance means shuffling the feature barely changes clustering quality.
    Negative importance means shuffling the feature improves clustering quality, which may indicate noise.

    Parameters
    ----------
    clusterer : Any
        Sklearn-compatible clustering estimator. Must implement fit_predict(...)
        or fit(...) + predict(...).

    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix used for clustering.

    feature_names : Sequence[str]
        Names of the columns in X.

    scoring : str, default="silhouette"
        Clustering scoring metric. Can be unsupervised or label-informed.

    y : Optional[np.ndarray] of shape (n_samples,), default=None
        Target labels aligned row-wise with X. Required only when scoring is
        label-informed, for example "adjusted_rand", "normalized_mutual_info",
        or "v_measure".

    n_repeats : int, default=10
        Number of independent shuffles per feature.

    random_state : Optional[int], default=42
        Random seed for reproducible feature shuffling.

    Returns
    -------
    pd.Series
        Feature importance values indexed by feature name.

    Notes
    -----
    Even when scoring uses y, the clustering estimator is still fit without y.
    The labels are used only to evaluate the resulting cluster assignments.
    """
    scoring = _validate_clustering_scoring(scoring)

    X = np.asarray(X)
    feature_names_list = list(feature_names)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {X.shape}.")

    if len(feature_names_list) != X.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names_list)}) must match "
            f"number of columns ({X.shape[1]})."
        )

    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1; got {n_repeats}.")

    # Validate y once up front. For unsupervised scores, this returns None unless
    # y was provided. For label-informed scores, y is required.
    y_array = _validate_y_for_clustering_score(
        y,
        n_samples=X.shape[0],
        scoring=scoring,
    )

    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------
    # Baseline clustering score
    # ------------------------------------------------------------
    fitted_base, labels_base = _fit_predict_clusters(clusterer, X)

    base_score = _score_clusters(
        fitted_clusterer=fitted_base,
        X=X,
        labels=labels_base,
        scoring=scoring,
        y=y_array,
    )

    if not np.isfinite(base_score):
        raise ValueError(
            "Baseline clustering score is invalid. This usually means the clustering "
            "produced too few valid clusters for the selected scoring metric. Try reducing "
            "n_clusters, increasing row_subsampling['train_fraction'], or using a different "
            "scoring metric. For label-informed scoring, also check that y has valid class labels."
        )

    importances: List[float] = []

    # ------------------------------------------------------------
    # Feature-wise permutation importance
    # ------------------------------------------------------------
    for j, feature_name in enumerate(feature_names_list):
        drops: List[float] = []

        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])

            fitted_perm, labels_perm = _fit_predict_clusters(clusterer, X_perm)

            perm_score = _score_clusters(
                fitted_clusterer=fitted_perm,
                X=X_perm,
                labels=labels_perm,
                scoring=scoring,
                y=y_array,
            )

            if np.isfinite(perm_score):
                drops.append(float(base_score - perm_score))

        if not drops:
            raise ValueError(
                f"No valid clustering score drops were computed for feature {feature_name!r}. "
                "This usually means the clustering score was invalid after feature shuffling. "
                "Try reducing n_clusters, increasing row_subsampling['train_fraction'], or using "
                "a different scoring metric."
            )

        importances.append(float(np.mean(drops)))

    return pd.Series(importances, index=feature_names_list, dtype=float)


# def clustering_permutation_importance(
#     clusterer: Any,
#     X: np.ndarray,
#     feature_names: Sequence[str],
#     *,
#     scoring: str = "silhouette",
#     n_repeats: int = 10,
#     random_state: Optional[int] = 42,
# ) -> pd.Series:
#     """
#     Compute unsupervised same-sample permutation importance for clustering.

#     This is the simplified clustering analogue of the supervised pipeline's
#     row-subsampled ranking logic:

#     1) Fit the clustering model on the sampled rows and selected features.
#     2) Score clustering quality on that same sampled dataset.
#     3) Shuffle one feature at a time.
#     4) Refit the clustering model on the shuffled dataset.
#     5) Re-score clustering quality.

#     importance(feature_j) = baseline_cluster_score - shuffled_refit_cluster_score

#     Large positive importance means shuffling the feature damages clustering quality.
#     Near-zero importance means shuffling the feature barely changes clustering quality.
#     Negative importance means shuffling the feature improves clustering quality, which may indicate noise.
#     """
#     scoring = _validate_clustering_scoring(scoring)

#     X = np.asarray(X)
#     feature_names_list = list(feature_names)

#     if X.ndim != 2:
#         raise ValueError(f"X must be 2D; got shape {X.shape}.")
#     if len(feature_names_list) != X.shape[1]:
#         raise ValueError(
#             f"feature_names length ({len(feature_names_list)}) must match "
#             f"number of columns ({X.shape[1]})."
#         )
#     if n_repeats < 1:
#         raise ValueError(f"n_repeats must be >= 1; got {n_repeats}.")

#     rng = np.random.default_rng(random_state)

#     fitted_base, labels_base = _fit_predict_clusters(clusterer, X)
#     base_score = _score_clusters(
#         fitted_clusterer=fitted_base,
#         X=X,
#         labels=labels_base,
#         scoring=scoring,
#     )

#     if not np.isfinite(base_score):
#         raise ValueError(
#             "Baseline clustering score is invalid. This usually means the clustering "
#             "produced too few valid clusters for the selected scoring metric. Try reducing "
#             "n_clusters, increasing row_subsampling['train_fraction'], or using a different "
#             "scoring metric such as 'calinski_harabasz'."
#         )

#     importances: List[float] = []

#     for j, feature_name in enumerate(feature_names_list):
#         drops: List[float] = []

#         for _ in range(n_repeats):
#             X_perm = X.copy()
#             X_perm[:, j] = rng.permutation(X_perm[:, j])

#             fitted_perm, labels_perm = _fit_predict_clusters(clusterer, X_perm)
#             perm_score = _score_clusters(
#                 fitted_clusterer=fitted_perm,
#                 X=X_perm,
#                 labels=labels_perm,
#                 scoring=scoring,
#             )

#             if np.isfinite(perm_score):
#                 drops.append(float(base_score - perm_score))

#         if not drops:
#             raise ValueError(
#                 f"No valid clustering score drops were computed for feature {feature_name!r}. "
#                 "This usually means the clustering score was invalid after feature shuffling. "
#                 "Try reducing n_clusters, increasing row_subsampling['train_fraction'], or using "
#                 "a different scoring metric such as 'calinski_harabasz'."
#             )

#         importances.append(float(np.mean(drops)))

#     return pd.Series(importances, index=feature_names_list, dtype=float)


# ============================================================
# Single-dataset clustering ranking
# ============================================================
def single_dataset_clustering_ranking(
    X: np.ndarray,
    feature_names: Sequence[str],
    cfg: Dict[str, Any],
    *,
    y: Optional[np.ndarray] = None,
    seed_offset: int = 0,
    stage_name: Optional[str] = None,
    stage_index: Optional[int] = None,
    n_stages: Optional[int] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[int, pd.DataFrame]]]:
    """
    Run balanced clustering-based feature ranking on one fixed dataset.

    This is the clustering sibling of single_dataset_permutation_ranking(...).

    The clustering model is always fit without y. However, the cluster scoring
    step can be either:

    1) Unsupervised / geometry-based
       The score uses X and the discovered cluster labels.
       Examples:
       - silhouette
       - calinski_harabasz
       - davies_bouldin
       - inertia

    2) Label-informed
       The score compares discovered cluster labels against y.
       Examples:
       - adjusted_rand
       - normalized_mutual_info
       - v_measure
       - homogeneity
       - completeness

    For each feature-subset run:
    - optionally sample rows from the full dataset
    - sample a balanced subset of features
    - fit/score clustering on the sampled rows
    - shuffle one feature at a time, refit, and measure score drop

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix for the dataset to rank.

    feature_names : Sequence[str]
        Names of the columns in X. Must match X.shape[1] and be unique.

    cfg : Dict[str, Any]
        Ranking configuration. Expected keys include:
        {
            "models": {
                "model_name": clusterer,
                ...
            },
            "scoring": "silhouette" | "adjusted_rand" | ...,
            "subset_sizes": [5, 10],
            "n_repeats": 10,
            "target_feature_appearances": 20,
            "random_state": 42,
            "ranking_metric": "auto",
            "row_subsampling": {
                "enabled": False,
                "train_fraction": 1.0,
            },
        }

    y : Optional[np.ndarray] of shape (n_samples,), default=None
        Target labels aligned row-wise with X. Required only when scoring is
        label-informed.

    seed_offset : int, default=0
        Offset added to the base random state so repeated outer calls can remain
        reproducible while still varying randomness.

    stage_name : Optional[str], default=None
        Human-readable stage name from the outer pipeline, used only for tqdm
        progress display.

    stage_index : Optional[int], default=None
        Zero-based stage index from the outer pipeline, reserved for compatibility.

    n_stages : Optional[int], default=None
        Total number of stages in the outer pipeline, reserved for compatibility.

    Returns
    -------
    final_ranking_by_model : Dict[str, pd.DataFrame]
        Mapping from model name to a final ranking table. Each table contains one
        row per feature and includes weighted summary metrics across subset sizes.

    detailed_results_by_model : Dict[str, Dict[int, pd.DataFrame]]
        Mapping from model name to per-subset-size ranking summaries.

    Notes
    -----
    If row subsampling is enabled and scoring uses y, both X and y are sliced to
    the same sampled rows before clustering and scoring.
    """
    # ============================================================
    # 1. Read and validate config
    # ============================================================
    model_dict = cfg.get("models", None)
    if model_dict is None:
        raise KeyError("cfg must contain a 'models' key.")
    model_dict = _validate_models_dict(model_dict)

    scoring = _validate_clustering_scoring(cfg.get("scoring", "silhouette"))

    for model_name, clusterer_template in model_dict.items():
        _validate_clustering_estimator(
            clusterer_template,
            model_name=model_name,
        )

    subset_sizes = cfg.get("subset_sizes", [10, 15, 20])
    n_repeats = int(cfg.get("n_repeats", 10))
    target_feature_appearances = int(cfg.get("target_feature_appearances", 20))
    random_state = int(cfg.get("random_state", 42))
    ranking_metric = _validate_ranking_metric(cfg.get("ranking_metric", "auto"))
    stage_name = str(cfg.get("name", stage_name or "clustering_feature_ranking"))

    row_subsampling_cfg = dict(cfg.get("row_subsampling", {}))
    row_subsampling_enabled = bool(row_subsampling_cfg.get("enabled", False))
    row_subsample_train_fraction = float(row_subsampling_cfg.get("train_fraction", 1.0))

    # ============================================================
    # 2. Input validation
    # ============================================================
    X = _validate_X_only(X)
    n_samples, n_features = X.shape

    feature_names_list = _validate_feature_names(
        feature_names,
        n_features,
        require_unique=True,
    )

    # Validate y once at the full-dataset level. If scoring is unsupervised, y is optional.
    # If scoring is label-informed, y is required and must align with X.
    y_array = _validate_y_for_clustering_score(
        y,
        n_samples=n_samples,
        scoring=scoring,
    )

    valid_subset_sizes = _validate_subset_sizes(subset_sizes, n_features)

    effective_ranking_metric = _resolve_effective_ranking_metric(
        ranking_metric,
        valid_subset_sizes,
    )

    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1; got {n_repeats}.")

    if target_feature_appearances < 1:
        raise ValueError(
            f"target_feature_appearances must be >= 1; got {target_feature_appearances}."
        )

    _validate_clustering_row_subsampling(
        enabled=row_subsampling_enabled,
        train_fraction=row_subsample_train_fraction,
        n_samples=n_samples,
    )

    if not row_subsampling_enabled:
        row_subsample_train_fraction = 1.0

    X_df = pd.DataFrame(X, columns=feature_names_list)

    # ============================================================
    # 3. Run per clustering model
    # ============================================================
    final_ranking_by_model: Dict[str, pd.DataFrame] = {}
    detailed_results_by_model: Dict[str, Dict[int, pd.DataFrame]] = {}

    for model_name, clusterer_template in model_dict.items():
        rng = np.random.RandomState(random_state + seed_offset)

        detailed_results: Dict[int, pd.DataFrame] = {}
        overall_records: DefaultDict[Hashable, List[Dict[str, float]]] = defaultdict(list)

        subset_progress = tqdm(
            valid_subset_sizes,
            total=len(valid_subset_sizes),
            desc=f"Clustering feature selection stage={stage_name} | model={model_name}",
            unit="subset",
            leave=False,
        )

        for subset_size in subset_progress:
            n_runs: int = ceil((target_feature_appearances * n_features) / subset_size)

            feature_counts: Dict[Hashable, int] = {
                feature: 0 for feature in feature_names_list
            }

            feature_rank_records: DefaultDict[Hashable, List[float]] = defaultdict(list)
            feature_norm_rank_records: DefaultDict[Hashable, List[float]] = defaultdict(list)
            feature_importance_records: DefaultDict[Hashable, List[float]] = defaultdict(list)

            run_progress = tqdm(
                range(n_runs),
                total=n_runs,
                desc=f"{model_name} | subset={subset_size}",
                unit="run",
                leave=False,
            )

            for _ in run_progress:
                run_progress.set_postfix(
                    row_subsample="on" if row_subsampling_enabled else "off",
                    row_fraction=row_subsample_train_fraction,
                    scoring=scoring,
                    uses_y="yes" if _requires_y_for_clustering_score(scoring) else "no",
                )

                # ------------------------------------------------------------
                # Balanced feature-subset sampling
                # ------------------------------------------------------------
                shuffled_feature_names: List[Hashable] = feature_names_list.copy()
                rng.shuffle(shuffled_feature_names)

                # Prioritize features that have appeared fewer times so far.
                shuffled_feature_names.sort(key=lambda feature: feature_counts[feature])

                selected_features: List[Hashable] = shuffled_feature_names[:subset_size]

                for feature in selected_features:
                    feature_counts[feature] += 1

                X_subset_full: pd.DataFrame = X_df[selected_features]

                # ------------------------------------------------------------
                # Optional row subsampling
                # ------------------------------------------------------------
                # If scoring uses y, y must be sliced using the exact same rows
                # as X. If scoring is unsupervised, y_subset remains None unless
                # the user provided y anyway.
                if row_subsampling_enabled:
                    row_idx = _sample_rows_unsupervised(
                        n_samples=n_samples,
                        train_fraction=row_subsample_train_fraction,
                        random_state=int(rng.randint(0, 1_000_000)),
                    )

                    X_subset = X_subset_full.iloc[row_idx]

                    if y_array is not None:
                        y_subset = y_array[row_idx]
                    else:
                        y_subset = None

                else:
                    X_subset = X_subset_full
                    y_subset = y_array

                X_subset_np = X_subset.to_numpy()

                # ------------------------------------------------------------
                # Clustering permutation importance
                # ------------------------------------------------------------
                feature_importances = clustering_permutation_importance(
                    clusterer=clusterer_template,
                    X=X_subset_np,
                    feature_names=selected_features,
                    scoring=scoring,
                    y=y_subset,
                    n_repeats=n_repeats,
                    random_state=int(rng.randint(0, 1_000_000)),
                )

                # Rank features within the current sampled subset from most
                # important to least important.
                feature_ranks = feature_importances.rank(
                    ascending=False,
                    method="average",
                )

                # Normalize ranks so subset sizes are comparable.
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

            # ========================================================
            # 4. Aggregate this subset size
            # ========================================================
            subset_summary_rows: List[Dict[str, Any]] = []

            for feature in feature_names_list:
                if not feature_importance_records[feature]:
                    continue

                n_observations = len(feature_importance_records[feature])
                mean_rank = float(np.mean(feature_rank_records[feature]))
                mean_normalized_rank = float(np.mean(feature_norm_rank_records[feature]))
                mean_importance = float(np.mean(feature_importance_records[feature]))

                subset_summary_rows.append(
                    {
                        "feature": feature,
                        "subset_size": subset_size,
                        "times_sampled": feature_counts[feature],
                        "n_observations": n_observations,
                        "mean_rank": mean_rank,
                        "mean_normalized_rank": mean_normalized_rank,
                        "mean_importance": mean_importance,
                        "scoring": scoring,
                        "label_informed_scoring": _requires_y_for_clustering_score(scoring),
                        "row_subsampling_enabled": row_subsampling_enabled,
                        "row_subsample_train_fraction": row_subsample_train_fraction,
                    }
                )

                overall_records[feature].append(
                    {
                        "subset_size": float(subset_size),
                        "mean_normalized_rank": mean_normalized_rank,
                        "mean_importance": mean_importance,
                        "n_observations": float(n_observations),
                    }
                )

            subset_summary_df = pd.DataFrame(subset_summary_rows)

            if subset_summary_df.empty:
                raise RuntimeError(
                    f"No clustering ranking records were produced for subset_size={subset_size}."
                )

            subset_summary_df = subset_summary_df.sort_values(
                by=["mean_normalized_rank", "mean_importance"],
                ascending=[False, False],
            ).reset_index(drop=True)

            detailed_results[subset_size] = subset_summary_df

        # ============================================================
        # 5. Aggregate across subset sizes
        # ============================================================
        final_rows: List[Dict[str, Any]] = []

        for feature, records in overall_records.items():
            weights = np.array(
                [record["n_observations"] for record in records],
                dtype=float,
            )

            if np.any(weights < 0):
                raise ValueError(f"Negative n_observations encountered for feature '{feature}'.")

            if np.all(weights == 0):
                raise ValueError(f"All n_observations are zero for feature '{feature}'.")

            final_rows.append(
                {
                    "feature": feature,
                    "mean_normalized_rank_across_sizes": float(
                        np.average(
                            [record["mean_normalized_rank"] for record in records],
                            weights=weights,
                        )
                    ),
                    "mean_importance_across_sizes": float(
                        np.average(
                            [record["mean_importance"] for record in records],
                            weights=weights,
                        )
                    ),
                    "n_subset_sizes_used": len(records),
                    "total_n_observations_across_sizes": int(weights.sum()),
                    "scoring": scoring,
                    "label_informed_scoring": _requires_y_for_clustering_score(scoring),
                    "row_subsampling_enabled": row_subsampling_enabled,
                    "row_subsample_train_fraction": row_subsample_train_fraction,
                }
            )

        final_ranking = pd.DataFrame(final_rows)

        if final_ranking.empty:
            raise RuntimeError(
                f"No final clustering ranking records were produced for model={model_name!r}."
            )

        if effective_ranking_metric == "mean_normalized_rank":
            final_ranking = final_ranking.sort_values(
                by=["mean_normalized_rank_across_sizes", "mean_importance_across_sizes"],
                ascending=[False, False],
            ).reset_index(drop=True)

        elif effective_ranking_metric == "mean_importance":
            final_ranking = final_ranking.sort_values(
                by=["mean_importance_across_sizes", "mean_normalized_rank_across_sizes"],
                ascending=[False, False],
            ).reset_index(drop=True)

        else:
            raise ValueError(f"Unsupported effective_ranking_metric: {effective_ranking_metric}")

        final_ranking_by_model[model_name] = final_ranking
        detailed_results_by_model[model_name] = detailed_results

    return final_ranking_by_model, detailed_results_by_model


# def single_dataset_clustering_ranking(
#     X: np.ndarray,
#     feature_names: Sequence[str],
#     cfg: Dict[str, Any],
#     *,
#     seed_offset: int = 0,
#     stage_name: Optional[str] = None,
#     stage_index: Optional[int] = None,
#     n_stages: Optional[int] = None,
# ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[int, pd.DataFrame]]]:
#     """
#     Run balanced clustering-based feature ranking on one fixed dataset.

#     This is the unsupervised sibling of single_dataset_permutation_ranking(...).
#     It does not use y, supervised CV, validation_fraction, or supervised scoring.

#     For each feature-subset run:
#     - optionally sample rows from the full dataset
#     - sample a balanced subset of features
#     - fit/score clustering on the sampled rows
#     - shuffle one feature at a time, refit, and measure score drop
#     """
#     model_dict = cfg.get("models", None)
#     if model_dict is None:
#         raise KeyError("cfg must contain a 'models' key.")
#     model_dict = _validate_models_dict(model_dict)

#     scoring = _validate_clustering_scoring(cfg.get("scoring", "silhouette"))

#     for model_name, clusterer_template in model_dict.items():
#         _validate_clustering_estimator(
#             clusterer_template,
#             model_name=model_name,
#         )

#     subset_sizes = cfg.get("subset_sizes", [10, 15, 20])
#     n_repeats = int(cfg.get("n_repeats", 10))
#     target_feature_appearances = int(cfg.get("target_feature_appearances", 20))
#     random_state = int(cfg.get("random_state", 42))
#     ranking_metric = _validate_ranking_metric(cfg.get("ranking_metric", "auto"))
#     stage_name = str(cfg.get("name", stage_name or "clustering_feature_ranking"))

#     row_subsampling_cfg = dict(cfg.get("row_subsampling", {}))
#     row_subsampling_enabled = bool(row_subsampling_cfg.get("enabled", False))
#     row_subsample_train_fraction = float(row_subsampling_cfg.get("train_fraction", 1.0))

#     X = _validate_X_only(X)
#     n_samples, n_features = X.shape

#     feature_names_list = _validate_feature_names(
#         feature_names,
#         n_features,
#         require_unique=True,
#     )

#     valid_subset_sizes = _validate_subset_sizes(subset_sizes, n_features)
#     effective_ranking_metric = _resolve_effective_ranking_metric(
#         ranking_metric,
#         valid_subset_sizes,
#     )

#     if n_repeats < 1:
#         raise ValueError(f"n_repeats must be >= 1; got {n_repeats}.")
#     if target_feature_appearances < 1:
#         raise ValueError(
#             f"target_feature_appearances must be >= 1; got {target_feature_appearances}."
#         )

#     _validate_clustering_row_subsampling(
#         enabled=row_subsampling_enabled,
#         train_fraction=row_subsample_train_fraction,
#         n_samples=n_samples,
#     )

#     if not row_subsampling_enabled:
#         row_subsample_train_fraction = 1.0

#     X_df = pd.DataFrame(X, columns=feature_names_list)

#     final_ranking_by_model: Dict[str, pd.DataFrame] = {}
#     detailed_results_by_model: Dict[str, Dict[int, pd.DataFrame]] = {}

#     for model_name, clusterer_template in model_dict.items():
#         rng = np.random.RandomState(random_state + seed_offset)

#         detailed_results: Dict[int, pd.DataFrame] = {}
#         overall_records: DefaultDict[Hashable, List[Dict[str, float]]] = defaultdict(list)

#         subset_progress = tqdm(
#             valid_subset_sizes,
#             total=len(valid_subset_sizes),
#             desc=f"Clustering feature selection stage={stage_name} | model={model_name}",
#             unit="subset",
#             leave=False,
#         )

#         for subset_size in subset_progress:
#             n_runs: int = ceil((target_feature_appearances * n_features) / subset_size)

#             feature_counts: Dict[Hashable, int] = {
#                 feature: 0 for feature in feature_names_list
#             }

#             feature_rank_records: DefaultDict[Hashable, List[float]] = defaultdict(list)
#             feature_norm_rank_records: DefaultDict[Hashable, List[float]] = defaultdict(list)
#             feature_importance_records: DefaultDict[Hashable, List[float]] = defaultdict(list)

#             run_progress = tqdm(
#                 range(n_runs),
#                 total=n_runs,
#                 desc=f"{model_name} | subset={subset_size}",
#                 unit="run",
#                 leave=False,
#             )

#             for _ in run_progress:
#                 run_progress.set_postfix(
#                     row_subsample="on" if row_subsampling_enabled else "off",
#                     row_fraction=row_subsample_train_fraction,
#                     scoring=scoring,
#                 )

#                 shuffled_feature_names: List[Hashable] = feature_names_list.copy()
#                 rng.shuffle(shuffled_feature_names)
#                 shuffled_feature_names.sort(key=lambda feature: feature_counts[feature])
#                 selected_features: List[Hashable] = shuffled_feature_names[:subset_size]

#                 for feature in selected_features:
#                     feature_counts[feature] += 1

#                 X_subset_full: pd.DataFrame = X_df[selected_features]

#                 if row_subsampling_enabled:
#                     row_idx = _sample_rows_unsupervised(
#                         n_samples=n_samples,
#                         train_fraction=row_subsample_train_fraction,
#                         random_state=int(rng.randint(0, 1_000_000)),
#                     )
#                     X_subset = X_subset_full.iloc[row_idx]
#                 else:
#                     X_subset = X_subset_full

#                 X_subset_np = X_subset.to_numpy()

#                 feature_importances = clustering_permutation_importance(
#                     clusterer=clusterer_template,
#                     X=X_subset_np,
#                     feature_names=selected_features,
#                     scoring=scoring,
#                     n_repeats=n_repeats,
#                     random_state=int(rng.randint(0, 1_000_000)),
#                 )

#                 feature_ranks = feature_importances.rank(
#                     ascending=False,
#                     method="average",
#                 )

#                 if subset_size == 1:
#                     normalized_feature_ranks = pd.Series(
#                         data=1.0,
#                         index=selected_features,
#                         dtype=float,
#                     )
#                 else:
#                     normalized_feature_ranks = 1 - (
#                         (feature_ranks - 1) / (subset_size - 1)
#                     )

#                 for feature in selected_features:
#                     feature_importance_records[feature].append(
#                         float(feature_importances[feature])
#                     )
#                     feature_rank_records[feature].append(
#                         float(feature_ranks[feature])
#                     )
#                     feature_norm_rank_records[feature].append(
#                         float(normalized_feature_ranks[feature])
#                     )

#             subset_summary_rows: List[Dict[str, Any]] = []

#             for feature in feature_names_list:
#                 if not feature_importance_records[feature]:
#                     continue

#                 n_observations = len(feature_importance_records[feature])
#                 mean_rank = float(np.mean(feature_rank_records[feature]))
#                 mean_normalized_rank = float(np.mean(feature_norm_rank_records[feature]))
#                 mean_importance = float(np.mean(feature_importance_records[feature]))

#                 subset_summary_rows.append(
#                     {
#                         "feature": feature,
#                         "subset_size": subset_size,
#                         "times_sampled": feature_counts[feature],
#                         "n_observations": n_observations,
#                         "mean_rank": mean_rank,
#                         "mean_normalized_rank": mean_normalized_rank,
#                         "mean_importance": mean_importance,
#                         "scoring": scoring,
#                         "row_subsampling_enabled": row_subsampling_enabled,
#                         "row_subsample_train_fraction": row_subsample_train_fraction,
#                     }
#                 )

#                 overall_records[feature].append(
#                     {
#                         "subset_size": float(subset_size),
#                         "mean_normalized_rank": mean_normalized_rank,
#                         "mean_importance": mean_importance,
#                         "n_observations": float(n_observations),
#                     }
#                 )

#             subset_summary_df = pd.DataFrame(subset_summary_rows)

#             subset_summary_df = subset_summary_df.sort_values(
#                 by=["mean_normalized_rank", "mean_importance"],
#                 ascending=[False, False],
#             ).reset_index(drop=True)

#             detailed_results[subset_size] = subset_summary_df

#         final_rows: List[Dict[str, Any]] = []

#         for feature, records in overall_records.items():
#             weights = np.array([record["n_observations"] for record in records], dtype=float)

#             if np.any(weights < 0):
#                 raise ValueError(f"Negative n_observations encountered for feature '{feature}'.")
#             if np.all(weights == 0):
#                 raise ValueError(f"All n_observations are zero for feature '{feature}'.")

#             final_rows.append(
#                 {
#                     "feature": feature,
#                     "mean_normalized_rank_across_sizes": float(
#                         np.average(
#                             [record["mean_normalized_rank"] for record in records],
#                             weights=weights,
#                         )
#                     ),
#                     "mean_importance_across_sizes": float(
#                         np.average(
#                             [record["mean_importance"] for record in records],
#                             weights=weights,
#                         )
#                     ),
#                     "n_subset_sizes_used": len(records),
#                     "total_n_observations_across_sizes": int(weights.sum()),
#                     "scoring": scoring,
#                     "row_subsampling_enabled": row_subsampling_enabled,
#                     "row_subsample_train_fraction": row_subsample_train_fraction,
#                 }
#             )

#         final_ranking = pd.DataFrame(final_rows)

#         if effective_ranking_metric == "mean_normalized_rank":
#             final_ranking = final_ranking.sort_values(
#                 by=["mean_normalized_rank_across_sizes", "mean_importance_across_sizes"],
#                 ascending=[False, False],
#             ).reset_index(drop=True)
#         elif effective_ranking_metric == "mean_importance":
#             final_ranking = final_ranking.sort_values(
#                 by=["mean_importance_across_sizes", "mean_normalized_rank_across_sizes"],
#                 ascending=[False, False],
#             ).reset_index(drop=True)
#         else:
#             raise ValueError(f"Unsupported effective_ranking_metric: {effective_ranking_metric}")

#         final_ranking_by_model[model_name] = final_ranking
#         detailed_results_by_model[model_name] = detailed_results

#     return final_ranking_by_model, detailed_results_by_model


# ============================================================
# Clustering rank-select stage
# ============================================================
def balanced_clustering_rank_select_stage(
    X: np.ndarray,
    groups: Optional[np.ndarray],
    feature_names: Sequence[str],
    cfg: Dict[str, Any],
    y: Optional[np.ndarray] = None,
    original_feature_indices: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Run a single clustering-based rank-and-select stage.

    This mirrors balanced_permutation_rank_select_stage(...), but uses clustering
    quality instead of supervised prediction quality.

    The clustering model is always fit without y. However, the scoring step can
    optionally use y when cfg["scoring"] is a label-informed clustering metric,
    such as:
    - "adjusted_rand"
    - "normalized_mutual_info"
    - "v_measure"
    - "homogeneity"
    - "completeness"

    Execution modes
    ---------------
    1) Non-group mode:
       Ranking is run once on the full current-stage dataset.

    2) Group mode:
       One row per group is repeatedly sampled. Ranking is run on each sampled
       dataset, and rankings are aggregated across group iterations.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Current feature matrix for this stage.

    groups : Optional[np.ndarray] of shape (n_samples,)
        Optional group identifiers. Required when group_mode=True.

    feature_names : Sequence[str]
        Feature names corresponding to the columns of X.

    cfg : Dict[str, Any]
        Single-stage clustering rank-select configuration.

    y : Optional[np.ndarray] of shape (n_samples,), default=None
        Target labels aligned row-wise with X. Required only when cfg["scoring"]
        is label-informed.

    original_feature_indices : Optional[np.ndarray] of shape (n_features,), default=None
        Original dataset column indices corresponding to the current-stage columns.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing final rankings, detailed results, and selected
        features per clustering model.
    """
    # ============================================================
    # 1. Read config
    # ============================================================
    group_mode = bool(cfg.get("group_mode", False))
    group_iterations = int(cfg.get("group_iterations", 10))
    random_state = int(cfg.get("random_state", 42))
    top_k = cfg.get("top_k", None)

    row_subsampling_cfg = dict(cfg.get("row_subsampling", {}))
    row_subsampling_enabled = bool(row_subsampling_cfg.get("enabled", False))
    row_subsample_train_fraction = float(row_subsampling_cfg.get("train_fraction", 1.0))

    if not row_subsampling_enabled:
        row_subsample_train_fraction = 1.0

    scoring = _validate_clustering_scoring(cfg.get("scoring", "silhouette"))
    label_informed_scoring = _requires_y_for_clustering_score(scoring)

    if top_k is None:
        raise KeyError("cfg must contain 'top_k' for clustering feature selection.")

    top_k = int(top_k)

    if top_k < 1:
        raise ValueError("cfg['top_k'] must be >= 1.")

    # ============================================================
    # 2. Input validation
    # ============================================================
    X = _validate_X_only(X)
    n_samples, n_features = X.shape

    feature_names_list = _validate_feature_names(
        feature_names,
        n_features,
        require_unique=True,
    )

    original_feature_indices = _validate_original_feature_indices(
        original_feature_indices,
        n_features,
    )

    groups = _validate_groups(
        groups,
        n_samples,
        required=group_mode,
    )

    # Validate y at the stage level so errors appear early and clearly.
    # If scoring is unsupervised, y remains optional.
    y_array = _validate_y_for_clustering_score(
        y,
        n_samples=n_samples,
        scoring=scoring,
    )

    if group_iterations < 1:
        raise ValueError("group_iterations must be >= 1.")

    if top_k > n_features:
        raise ValueError(
            f"cfg['top_k'] ({top_k}) cannot exceed number of features ({n_features})."
        )

    feature_name_to_idx = {name: idx for idx, name in enumerate(feature_names_list)}

    # ============================================================
    # 3. Non-group mode
    # ============================================================
    if not group_mode:
        final_ranking_by_model, detailed_results_by_model = single_dataset_clustering_ranking(
            X=X,
            feature_names=feature_names_list,
            cfg=cfg,
            y=y_array,
            seed_offset=0,
        )

        normalized_final_ranking_by_model: Dict[str, pd.DataFrame] = {}

        for model_name, df_rank in final_ranking_by_model.items():
            df_norm = df_rank.rename(
                columns={
                    "mean_normalized_rank_across_sizes": "mean_normalized_rank",
                    "mean_importance_across_sizes": "mean_importance",
                    "total_n_observations_across_sizes": "total_n_observations",
                }
            ).copy()

            if "group_iterations_used" not in df_norm.columns:
                df_norm["group_iterations_used"] = 1

            if "label_informed_scoring" not in df_norm.columns:
                df_norm["label_informed_scoring"] = label_informed_scoring

            df_norm = df_norm[
                [
                    "feature",
                    "mean_normalized_rank",
                    "mean_importance",
                    "n_subset_sizes_used",
                    "total_n_observations",
                    "group_iterations_used",
                    "scoring",
                    "label_informed_scoring",
                    "row_subsampling_enabled",
                    "row_subsample_train_fraction",
                ]
            ].reset_index(drop=True)

            normalized_final_ranking_by_model[model_name] = df_norm

        normalized_detailed_results_by_model: Dict[str, Dict[int, pd.DataFrame]] = {}

        for model_name, detail_dict in detailed_results_by_model.items():
            normalized_detail_dict: Dict[int, pd.DataFrame] = {}

            for subset_size, df_subset in detail_dict.items():
                df_norm = df_subset.copy()

                if "group_iterations_used" not in df_norm.columns:
                    df_norm["group_iterations_used"] = 1

                if "label_informed_scoring" not in df_norm.columns:
                    df_norm["label_informed_scoring"] = label_informed_scoring

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
                        "scoring",
                        "label_informed_scoring",
                        "row_subsampling_enabled",
                        "row_subsample_train_fraction",
                    ]
                ].reset_index(drop=True)

                normalized_detail_dict[int(subset_size)] = df_norm

            normalized_detailed_results_by_model[model_name] = normalized_detail_dict

        final_ranking_by_model = normalized_final_ranking_by_model
        detailed_results_by_model = normalized_detailed_results_by_model

    # ============================================================
    # 4. Group mode
    # ============================================================
    else:
        rng_group = np.random.default_rng(random_state)
        group_seeds = rng_group.integers(
            0,
            1_000_000,
            size=group_iterations,
            dtype=np.int64,
        )

        all_rankings_by_model: Dict[str, List[pd.DataFrame]] = defaultdict(list)
        all_details_by_model: Dict[str, List[Dict[int, pd.DataFrame]]] = defaultdict(list)

        # sample_one_row_per_group requires a y argument. If the chosen scoring
        # does not use y, pass a dummy vector. If scoring uses y, pass the real y.
        y_for_group_sampling = (
            y_array
            if y_array is not None
            else np.zeros(n_samples, dtype=float)
        )

        for iter_idx in tqdm(
            range(group_iterations),
            total=group_iterations,
            desc="Clustering group bootstrap iterations",
            unit="iter",
        ):
            seed_n = int(group_seeds[iter_idx])

            X_sub, y_sub, _, _ = sample_one_row_per_group(
                X=X,
                y=y_for_group_sampling,
                groups=groups,
                random_state=seed_n,
            )

            # If scoring is unsupervised, keep y_sub as None downstream.
            # If scoring is label-informed, y_sub is the sampled real target.
            y_sub_for_scoring = y_sub if label_informed_scoring else None

            rankings_run, details_run = single_dataset_clustering_ranking(
                X=X_sub,
                feature_names=feature_names_list,
                cfg=cfg,
                y=y_sub_for_scoring,
                seed_offset=seed_n,
            )

            for model_name, df_rank in rankings_run.items():
                df_rank_norm = df_rank.rename(
                    columns={
                        "mean_normalized_rank_across_sizes": "mean_normalized_rank",
                        "mean_importance_across_sizes": "mean_importance",
                        "total_n_observations_across_sizes": "total_n_observations",
                    }
                ).copy()

                if "group_iterations_used" not in df_rank_norm.columns:
                    df_rank_norm["group_iterations_used"] = 1

                if "label_informed_scoring" not in df_rank_norm.columns:
                    df_rank_norm["label_informed_scoring"] = label_informed_scoring

                all_rankings_by_model[model_name].append(df_rank_norm)

            for model_name, detail_dict in details_run.items():
                normalized_detail_dict: Dict[int, pd.DataFrame] = {}

                for subset_size, df_subset in detail_dict.items():
                    df_subset_norm = df_subset.copy()

                    if "group_iterations_used" not in df_subset_norm.columns:
                        df_subset_norm["group_iterations_used"] = 1

                    if "label_informed_scoring" not in df_subset_norm.columns:
                        df_subset_norm["label_informed_scoring"] = label_informed_scoring

                    normalized_detail_dict[int(subset_size)] = df_subset_norm

                all_details_by_model[model_name].append(normalized_detail_dict)

        ranking_metric = _validate_ranking_metric(cfg.get("ranking_metric", "auto"))
        subset_sizes = cfg.get("subset_sizes", [10, 15, 20])
        subset_sizes_for_metric = _validate_subset_sizes(subset_sizes, n_features)

        effective_ranking_metric = _resolve_effective_ranking_metric(
            ranking_metric,
            subset_sizes_for_metric,
        )

        final_ranking_by_model = {}
        detailed_results_by_model = {}

        # --------------------------------------------------------
        # 4a. Aggregate final rankings across group iterations
        # --------------------------------------------------------
        for model_name, rank_list in all_rankings_by_model.items():
            feature_records: DefaultDict[Hashable, Dict[str, List[float]]] = defaultdict(
                lambda: {
                    "mean_normalized_rank": [],
                    "mean_importance": [],
                    "n_subset_sizes_used": [],
                    "total_n_observations": [],
                }
            )

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

            for feat, vals in feature_records.items():
                weights = np.asarray(vals["total_n_observations"], dtype=float)

                if np.any(weights < 0):
                    raise ValueError(
                        f"Negative total_n_observations encountered for feature '{feat}'."
                    )

                if np.all(weights == 0):
                    raise ValueError(
                        f"All total_n_observations are zero for feature '{feat}'."
                    )

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
                        "scoring": scoring,
                        "label_informed_scoring": label_informed_scoring,
                        "row_subsampling_enabled": row_subsampling_enabled,
                        "row_subsample_train_fraction": row_subsample_train_fraction,
                    }
                )

            final_df = pd.DataFrame(final_rows)

            if final_df.empty:
                raise RuntimeError(
                    f"No aggregated group-mode ranking records were produced for model={model_name!r}."
                )

            if effective_ranking_metric == "mean_normalized_rank":
                final_df = final_df.sort_values(
                    by=["mean_normalized_rank", "mean_importance"],
                    ascending=[False, False],
                ).reset_index(drop=True)

            elif effective_ranking_metric == "mean_importance":
                final_df = final_df.sort_values(
                    by=["mean_importance", "mean_normalized_rank"],
                    ascending=[False, False],
                ).reset_index(drop=True)

            else:
                raise ValueError(f"Unsupported effective_ranking_metric: {effective_ranking_metric}")

            final_ranking_by_model[model_name] = final_df

        # --------------------------------------------------------
        # 4b. Aggregate detailed per-subset-size results
        # --------------------------------------------------------
        for model_name, details_list in all_details_by_model.items():
            subset_size_to_tables: DefaultDict[int, List[pd.DataFrame]] = defaultdict(list)

            for detail_dict in details_list:
                for subset_size, df_subset in detail_dict.items():
                    subset_size_to_tables[int(subset_size)].append(df_subset)

            aggregated_detail_dict: Dict[int, pd.DataFrame] = {}

            for subset_size, df_list in subset_size_to_tables.items():
                feature_records: DefaultDict[Hashable, Dict[str, List[float]]] = defaultdict(
                    lambda: {
                        "times_sampled": [],
                        "n_observations": [],
                        "mean_rank": [],
                        "mean_normalized_rank": [],
                        "mean_importance": [],
                    }
                )

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
                            "scoring": scoring,
                            "label_informed_scoring": label_informed_scoring,
                            "row_subsampling_enabled": row_subsampling_enabled,
                            "row_subsample_train_fraction": row_subsample_train_fraction,
                        }
                    )

                agg_df = pd.DataFrame(agg_rows)

                if agg_df.empty:
                    raise RuntimeError(
                        f"No aggregated detailed records were produced for subset_size={subset_size}."
                    )

                agg_df = agg_df.sort_values(
                    by=["mean_normalized_rank", "mean_importance"],
                    ascending=[False, False],
                ).reset_index(drop=True)

                aggregated_detail_dict[subset_size] = agg_df

            detailed_results_by_model[model_name] = aggregated_detail_dict

    # ============================================================
    # 5. Select top-k per model
    # ============================================================
    selected_by_model: Dict[str, Dict[str, Any]] = {}

    for model_name, df_rank in final_ranking_by_model.items():
        df_top = df_rank.head(top_k).copy().reset_index(drop=True)

        selected_feature_names = df_top["feature"].astype(str).tolist()

        selected_feature_indices_local = np.array(
            [feature_name_to_idx[name] for name in selected_feature_names],
            dtype=int,
        )

        selected_feature_indices = original_feature_indices[selected_feature_indices_local]
        X_selected = X[:, selected_feature_indices_local]

        selected_by_model[model_name] = {
            "X": X_selected,
            "selected_feature_names": selected_feature_names,
            "selected_feature_indices": selected_feature_indices,
            "selected_feature_indices_local": selected_feature_indices_local,
        }

    return {
        "final_ranking_by_model": final_ranking_by_model,
        "detailed_results_by_model": detailed_results_by_model,
        "selected_by_model": selected_by_model,
    }


# def balanced_clustering_rank_select_stage(
#     X: np.ndarray,
#     groups: Optional[np.ndarray],
#     feature_names: Sequence[str],
#     cfg: Dict[str, Any],
#     original_feature_indices: Optional[np.ndarray] = None,
# ) -> Dict[str, Any]:
#     """
#     Run a single unsupervised clustering-based rank-and-select stage.

#     This mirrors balanced_permutation_rank_select_stage(...), but uses clustering
#     quality instead of supervised prediction quality and does not accept y.
#     """
#     group_mode = bool(cfg.get("group_mode", False))
#     group_iterations = int(cfg.get("group_iterations", 10))
#     random_state = int(cfg.get("random_state", 42))
#     top_k = cfg.get("top_k", None)

#     row_subsampling_cfg = dict(cfg.get("row_subsampling", {}))
#     row_subsampling_enabled = bool(row_subsampling_cfg.get("enabled", False))
#     row_subsample_train_fraction = float(row_subsampling_cfg.get("train_fraction", 1.0))
#     if not row_subsampling_enabled:
#         row_subsample_train_fraction = 1.0

#     scoring = _validate_clustering_scoring(cfg.get("scoring", "silhouette"))

#     if top_k is None:
#         raise KeyError("cfg must contain 'top_k' for clustering feature selection.")
#     top_k = int(top_k)
#     if top_k < 1:
#         raise ValueError("cfg['top_k'] must be >= 1.")

#     X = _validate_X_only(X)
#     n_samples, n_features = X.shape

#     feature_names_list = _validate_feature_names(
#         feature_names,
#         n_features,
#         require_unique=True,
#     )

#     original_feature_indices = _validate_original_feature_indices(
#         original_feature_indices,
#         n_features,
#     )

#     groups = _validate_groups(groups, n_samples, required=group_mode)

#     if group_iterations < 1:
#         raise ValueError("group_iterations must be >= 1.")

#     if top_k > n_features:
#         raise ValueError(
#             f"cfg['top_k'] ({top_k}) cannot exceed number of features ({n_features})."
#         )

#     feature_name_to_idx = {name: idx for idx, name in enumerate(feature_names_list)}

#     if not group_mode:
#         final_ranking_by_model, detailed_results_by_model = single_dataset_clustering_ranking(
#             X=X,
#             feature_names=feature_names_list,
#             cfg=cfg,
#             seed_offset=0,
#         )

#         normalized_final_ranking_by_model: Dict[str, pd.DataFrame] = {}

#         for model_name, df_rank in final_ranking_by_model.items():
#             df_norm = df_rank.rename(
#                 columns={
#                     "mean_normalized_rank_across_sizes": "mean_normalized_rank",
#                     "mean_importance_across_sizes": "mean_importance",
#                     "total_n_observations_across_sizes": "total_n_observations",
#                 }
#             ).copy()

#             if "group_iterations_used" not in df_norm.columns:
#                 df_norm["group_iterations_used"] = 1

#             df_norm = df_norm[
#                 [
#                     "feature",
#                     "mean_normalized_rank",
#                     "mean_importance",
#                     "n_subset_sizes_used",
#                     "total_n_observations",
#                     "group_iterations_used",
#                     "scoring",
#                     "row_subsampling_enabled",
#                     "row_subsample_train_fraction",
#                 ]
#             ].reset_index(drop=True)

#             normalized_final_ranking_by_model[model_name] = df_norm

#         normalized_detailed_results_by_model: Dict[str, Dict[int, pd.DataFrame]] = {}

#         for model_name, detail_dict in detailed_results_by_model.items():
#             normalized_detail_dict: Dict[int, pd.DataFrame] = {}

#             for subset_size, df_subset in detail_dict.items():
#                 df_norm = df_subset.copy()

#                 if "group_iterations_used" not in df_norm.columns:
#                     df_norm["group_iterations_used"] = 1

#                 df_norm = df_norm[
#                     [
#                         "feature",
#                         "subset_size",
#                         "times_sampled",
#                         "n_observations",
#                         "mean_rank",
#                         "mean_normalized_rank",
#                         "mean_importance",
#                         "group_iterations_used",
#                         "scoring",
#                         "row_subsampling_enabled",
#                         "row_subsample_train_fraction",
#                     ]
#                 ].reset_index(drop=True)

#                 normalized_detail_dict[int(subset_size)] = df_norm

#             normalized_detailed_results_by_model[model_name] = normalized_detail_dict

#         final_ranking_by_model = normalized_final_ranking_by_model
#         detailed_results_by_model = normalized_detailed_results_by_model

#     else:
#         rng_group = np.random.default_rng(random_state)
#         group_seeds = rng_group.integers(0, 1_000_000, size=group_iterations, dtype=np.int64)

#         all_rankings_by_model: Dict[str, List[pd.DataFrame]] = defaultdict(list)
#         all_details_by_model: Dict[str, List[Dict[int, pd.DataFrame]]] = defaultdict(list)

#         y_dummy = np.zeros(n_samples, dtype=float)

#         for iter_idx in tqdm(
#             range(group_iterations),
#             total=group_iterations,
#             desc="Clustering group bootstrap iterations",
#             unit="iter",
#         ):
#             seed_n = int(group_seeds[iter_idx])

#             X_sub, _, _, _ = sample_one_row_per_group(
#                 X=X,
#                 y=y_dummy,
#                 groups=groups,
#                 random_state=seed_n,
#             )

#             rankings_run, details_run = single_dataset_clustering_ranking(
#                 X=X_sub,
#                 feature_names=feature_names_list,
#                 cfg=cfg,
#                 seed_offset=seed_n,
#             )

#             for model_name, df_rank in rankings_run.items():
#                 df_rank_norm = df_rank.rename(
#                     columns={
#                         "mean_normalized_rank_across_sizes": "mean_normalized_rank",
#                         "mean_importance_across_sizes": "mean_importance",
#                         "total_n_observations_across_sizes": "total_n_observations",
#                     }
#                 ).copy()

#                 if "group_iterations_used" not in df_rank_norm.columns:
#                     df_rank_norm["group_iterations_used"] = 1

#                 all_rankings_by_model[model_name].append(df_rank_norm)

#             for model_name, detail_dict in details_run.items():
#                 normalized_detail_dict: Dict[int, pd.DataFrame] = {}

#                 for subset_size, df_subset in detail_dict.items():
#                     df_subset_norm = df_subset.copy()

#                     if "group_iterations_used" not in df_subset_norm.columns:
#                         df_subset_norm["group_iterations_used"] = 1

#                     normalized_detail_dict[int(subset_size)] = df_subset_norm

#                 all_details_by_model[model_name].append(normalized_detail_dict)

#         ranking_metric = _validate_ranking_metric(cfg.get("ranking_metric", "auto"))
#         subset_sizes = cfg.get("subset_sizes", [10, 15, 20])
#         subset_sizes_for_metric = _validate_subset_sizes(subset_sizes, n_features)
#         effective_ranking_metric = _resolve_effective_ranking_metric(
#             ranking_metric,
#             subset_sizes_for_metric,
#         )

#         final_ranking_by_model = {}
#         detailed_results_by_model = {}

#         for model_name, rank_list in all_rankings_by_model.items():
#             feature_records: DefaultDict[Hashable, Dict[str, List[float]]] = defaultdict(
#                 lambda: {
#                     "mean_normalized_rank": [],
#                     "mean_importance": [],
#                     "n_subset_sizes_used": [],
#                     "total_n_observations": [],
#                 }
#             )

#             for df_rank in rank_list:
#                 for _, row in df_rank.iterrows():
#                     feat = row["feature"]
#                     feature_records[feat]["mean_normalized_rank"].append(
#                         float(row["mean_normalized_rank"])
#                     )
#                     feature_records[feat]["mean_importance"].append(
#                         float(row["mean_importance"])
#                     )
#                     feature_records[feat]["n_subset_sizes_used"].append(
#                         float(row["n_subset_sizes_used"])
#                     )
#                     feature_records[feat]["total_n_observations"].append(
#                         float(row["total_n_observations"])
#                     )

#             final_rows: List[Dict[str, Any]] = []

#             for feat, vals in feature_records.items():
#                 weights = np.asarray(vals["total_n_observations"], dtype=float)

#                 if np.any(weights < 0):
#                     raise ValueError(
#                         f"Negative total_n_observations encountered for feature '{feat}'."
#                     )
#                 if np.all(weights == 0):
#                     raise ValueError(
#                         f"All total_n_observations are zero for feature '{feat}'."
#                     )

#                 final_rows.append(
#                     {
#                         "feature": feat,
#                         "mean_normalized_rank": float(
#                             np.average(vals["mean_normalized_rank"], weights=weights)
#                         ),
#                         "mean_importance": float(
#                             np.average(vals["mean_importance"], weights=weights)
#                         ),
#                         "n_subset_sizes_used": float(
#                             np.average(vals["n_subset_sizes_used"], weights=weights)
#                         ),
#                         "total_n_observations": int(weights.sum()),
#                         "group_iterations_used": len(vals["mean_normalized_rank"]),
#                         "scoring": scoring,
#                         "row_subsampling_enabled": row_subsampling_enabled,
#                         "row_subsample_train_fraction": row_subsample_train_fraction,
#                     }
#                 )

#             final_df = pd.DataFrame(final_rows)

#             if effective_ranking_metric == "mean_normalized_rank":
#                 final_df = final_df.sort_values(
#                     by=["mean_normalized_rank", "mean_importance"],
#                     ascending=[False, False],
#                 ).reset_index(drop=True)
#             elif effective_ranking_metric == "mean_importance":
#                 final_df = final_df.sort_values(
#                     by=["mean_importance", "mean_normalized_rank"],
#                     ascending=[False, False],
#                 ).reset_index(drop=True)
#             else:
#                 raise ValueError(f"Unsupported effective_ranking_metric: {effective_ranking_metric}")

#             final_ranking_by_model[model_name] = final_df

#         for model_name, details_list in all_details_by_model.items():
#             subset_size_to_tables: DefaultDict[int, List[pd.DataFrame]] = defaultdict(list)

#             for detail_dict in details_list:
#                 for subset_size, df_subset in detail_dict.items():
#                     subset_size_to_tables[int(subset_size)].append(df_subset)

#             aggregated_detail_dict: Dict[int, pd.DataFrame] = {}

#             for subset_size, df_list in subset_size_to_tables.items():
#                 feature_records: DefaultDict[Hashable, Dict[str, List[float]]] = defaultdict(
#                     lambda: {
#                         "times_sampled": [],
#                         "n_observations": [],
#                         "mean_rank": [],
#                         "mean_normalized_rank": [],
#                         "mean_importance": [],
#                     }
#                 )

#                 for df_subset in df_list:
#                     for _, row in df_subset.iterrows():
#                         feat = row["feature"]
#                         feature_records[feat]["times_sampled"].append(float(row["times_sampled"]))
#                         feature_records[feat]["n_observations"].append(float(row["n_observations"]))
#                         feature_records[feat]["mean_rank"].append(float(row["mean_rank"]))
#                         feature_records[feat]["mean_normalized_rank"].append(
#                             float(row["mean_normalized_rank"])
#                         )
#                         feature_records[feat]["mean_importance"].append(
#                             float(row["mean_importance"])
#                         )

#                 agg_rows: List[Dict[str, Any]] = []

#                 for feat, vals in feature_records.items():
#                     agg_rows.append(
#                         {
#                             "feature": feat,
#                             "subset_size": subset_size,
#                             "times_sampled": float(np.mean(vals["times_sampled"])),
#                             "n_observations": float(np.mean(vals["n_observations"])),
#                             "mean_rank": float(np.mean(vals["mean_rank"])),
#                             "mean_normalized_rank": float(np.mean(vals["mean_normalized_rank"])),
#                             "mean_importance": float(np.mean(vals["mean_importance"])),
#                             "group_iterations_used": len(vals["mean_rank"]),
#                             "scoring": scoring,
#                             "row_subsampling_enabled": row_subsampling_enabled,
#                             "row_subsample_train_fraction": row_subsample_train_fraction,
#                         }
#                     )

#                 agg_df = pd.DataFrame(agg_rows).sort_values(
#                     by=["mean_normalized_rank", "mean_importance"],
#                     ascending=[False, False],
#                 ).reset_index(drop=True)

#                 aggregated_detail_dict[subset_size] = agg_df

#             detailed_results_by_model[model_name] = aggregated_detail_dict

#     selected_by_model: Dict[str, Dict[str, Any]] = {}

#     for model_name, df_rank in final_ranking_by_model.items():
#         df_top = df_rank.head(top_k).copy().reset_index(drop=True)

#         selected_feature_names = df_top["feature"].astype(str).tolist()

#         selected_feature_indices_local = np.array(
#             [feature_name_to_idx[name] for name in selected_feature_names],
#             dtype=int,
#         )

#         selected_feature_indices = original_feature_indices[selected_feature_indices_local]
#         X_selected = X[:, selected_feature_indices_local]

#         selected_by_model[model_name] = {
#             "X": X_selected,
#             "selected_feature_names": selected_feature_names,
#             "selected_feature_indices": selected_feature_indices,
#             "selected_feature_indices_local": selected_feature_indices_local,
#         }

#     return {
#         "final_ranking_by_model": final_ranking_by_model,
#         "detailed_results_by_model": detailed_results_by_model,
#         "selected_by_model": selected_by_model,
#     }


# ============================================================
# Full clustering rank-select pipeline
# ============================================================
def balanced_clustering_rank_select_pipeline(
    X: np.ndarray,
    groups: Optional[np.ndarray],
    feature_names: Sequence[str],
    cfg: Dict[str, Any],
    y: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Run a multi-stage clustering feature rank-and-select pipeline.

    This is the clustering sibling of balanced_permutation_rank_select_pipeline(...).
    It runs independently for each clustering model in cfg["models"], carries
    selected features from stage to stage, and preserves original dataset feature
    indices.

    The clustering estimator itself is always fit without y. However, the scoring
    function can optionally use y when cfg["scoring"] is label-informed.

    Supported scoring modes
    -----------------------
    Unsupervised / geometry-based:
    - "silhouette"
    - "calinski_harabasz"
    - "davies_bouldin"
    - "inertia"

    Label-informed:
    - "adjusted_rand"
    - "normalized_mutual_info"
    - "v_measure"
    - "homogeneity"
    - "completeness"

    Interpretation
    --------------
    If scoring is unsupervised, features are ranked by how much clustering
    geometry/structure degrades when each feature is permuted.

    If scoring is label-informed, features are ranked by how much agreement
    between cluster assignments and y degrades when each feature is permuted.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Full input feature matrix.

    groups : Optional[np.ndarray] of shape (n_samples,)
        Optional group identifiers. Required only if a stage enables group_mode.

    feature_names : Sequence[str]
        Names of the columns in X. Must match X.shape[1] and be unique.

    cfg : Dict[str, Any]
        Pipeline configuration dictionary.

    y : Optional[np.ndarray] of shape (n_samples,), default=None
        Target labels aligned row-wise with X. Required only when cfg["scoring"]
        or any stage-level scoring override is label-informed.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing final outputs by clustering model.

        {
            "final_by_model": {
                model_name: {
                    "X": final reduced feature matrix,
                    "feature_names_selected": final selected feature names,
                    "selected_feature_indices": original dataset feature indices,
                    "history": ordered list of stage outputs,
                    "by_stage": stage-name lookup,
                }
            },
            "y": y or None,
        }
    """
    # ============================================================
    # 1. Validate top-level inputs
    # ============================================================
    X = _validate_X_only(X)
    n_samples, n_features = X.shape

    feature_names_list = _validate_feature_names(
        feature_names,
        n_features,
        require_unique=True,
    )

    groups = _validate_groups(
        groups,
        n_samples,
        required=False,
    )

    defaults, models_registry, stages = _validate_pipeline_cfg(cfg)

    # Validate y against all scoring metrics that may appear after defaults/stage merge.
    # This catches missing y early when any stage uses label-informed scoring.
    y_array: Optional[np.ndarray]

    if y is None:
        y_array = None
    else:
        y_array = np.asarray(y)

        if y_array.ndim != 1:
            raise ValueError(f"y must be 1D when provided; got shape {y_array.shape}.")

        if len(y_array) != n_samples:
            raise ValueError(
                f"y length ({len(y_array)}) must match number of rows in X ({n_samples})."
            )

    for stage_idx, stage in enumerate(stages):
        stage_cfg_for_validation = _deep_merge(defaults, stage)
        stage_scoring = _validate_clustering_scoring(
            stage_cfg_for_validation.get("scoring", "silhouette")
        )

        _validate_y_for_clustering_score(
            y_array,
            n_samples=n_samples,
            scoring=stage_scoring,
        )

    # ============================================================
    # 2. Run full pipeline separately for each clustering model
    # ============================================================
    final_by_model: Dict[str, Any] = {}

    for model_name, model_estimator in models_registry.items():
        X_current = X.copy()
        names_current = list(feature_names_list)
        original_indices_current = np.arange(n_features, dtype=int)

        history: List[Dict[str, Any]] = []
        by_stage: Dict[str, Dict[str, Any]] = {}

        for stage_idx, stage in enumerate(stages):
            stage_name = stage.get("name", f"stage_{stage_idx}")

            # Merge stage-specific overrides on top of defaults.
            stage_cfg = _deep_merge(defaults, stage)
            stage_cfg["models"] = {model_name: model_estimator}

            stage_scoring = _validate_clustering_scoring(
                stage_cfg.get("scoring", "silhouette")
            )
            label_informed_scoring = _requires_y_for_clustering_score(stage_scoring)

            engine_out = balanced_clustering_rank_select_stage(
                X=X_current,
                y=y_array,
                groups=groups,
                feature_names=names_current,
                cfg=stage_cfg,
                original_feature_indices=original_indices_current,
            )

            selected_model_out = engine_out["selected_by_model"][model_name]
            ranking_model_out = engine_out["final_ranking_by_model"][model_name]
            detail_model_out = engine_out["detailed_results_by_model"][model_name]

            X_next = selected_model_out["X"]
            names_next = list(selected_model_out["selected_feature_names"])

            selected_idx_local = _validate_stage_selection_output(
                selected_model_out["selected_feature_indices_local"],
                names_next,
                len(original_indices_current),
                stage_name,
            )

            selected_idx_original = selected_model_out["selected_feature_indices"]

            row_subsampling_cfg = dict(stage_cfg.get("row_subsampling", {}))
            row_subsampling_enabled = bool(row_subsampling_cfg.get("enabled", False))
            row_subsample_train_fraction = float(row_subsampling_cfg.get("train_fraction", 1.0))

            if not row_subsampling_enabled:
                row_subsample_train_fraction = 1.0

            stage_out = {
                "stage": stage_name,
                "top_k": int(stage_cfg["top_k"]),
                "n_features_in": int(X_current.shape[1]),
                "n_features_out": int(X_next.shape[1]),
                "cfg_used": deepcopy(stage_cfg),

                # Scoring metadata
                "scoring": stage_scoring,
                "label_informed_scoring": label_informed_scoring,

                # Row-subsampling metadata
                "row_subsampling_enabled": row_subsampling_enabled,
                "row_subsample_train_fraction": row_subsample_train_fraction,

                # Ranking/selection outputs
                "final_ranking": ranking_model_out,
                "detailed_results": detail_model_out,
                "selected_feature_names": names_next,
                "selected_feature_indices": np.asarray(selected_idx_original, dtype=int).copy(),
                "selected_feature_indices_local": selected_idx_local.copy(),
                "X_selected": X_next,
            }

            history.append(stage_out)
            by_stage[stage_name] = stage_out

            # Feed selected features forward to the next stage.
            X_current = X_next
            names_current = names_next
            original_indices_current = np.asarray(selected_idx_original, dtype=int).copy()

        final_by_model[model_name] = {
            "X": X_current,
            "feature_names_selected": names_current,
            "selected_feature_indices": original_indices_current.copy(),
            "history": history,
            "by_stage": by_stage,
        }

    return {
        "final_by_model": final_by_model,
        "y": y_array,
    }


# def balanced_clustering_rank_select_pipeline(
#     X: np.ndarray,
#     groups: Optional[np.ndarray],
#     feature_names: Sequence[str],
#     cfg: Dict[str, Any],
# ) -> Dict[str, Any]:
#     """
#     Run a multi-stage unsupervised clustering feature rank-and-select pipeline.

#     This is the unsupervised sibling of balanced_permutation_rank_select_pipeline(...).
#     It runs independently for each clustering model in cfg['models'], carries selected
#     features from stage to stage, and preserves original dataset feature indices.

#     Interpretation
#     --------------
#     Features are ranked by how much clustering quality drops when each feature is
#     permuted and the clustering model is refit. This is unsupervised. It does not
#     mean a feature predicts a target; it means a feature supports the cluster structure
#     under the chosen clustering algorithm and scoring metric.
#     """
#     X = _validate_X_only(X)
#     n_samples, n_features = X.shape

#     feature_names_list = _validate_feature_names(
#         feature_names,
#         n_features,
#         require_unique=True,
#     )

#     groups = _validate_groups(groups, n_samples, required=False)

#     defaults, models_registry, stages = _validate_pipeline_cfg(cfg)

#     final_by_model: Dict[str, Any] = {}

#     for model_name, model_estimator in models_registry.items():
#         X_current = X.copy()
#         names_current = list(feature_names_list)
#         original_indices_current = np.arange(n_features, dtype=int)

#         history: List[Dict[str, Any]] = []
#         by_stage: Dict[str, Dict[str, Any]] = {}

#         for stage_idx, stage in enumerate(stages):
#             stage_name = stage.get("name", f"stage_{stage_idx}")
#             stage_cfg = _deep_merge(defaults, stage)
#             stage_cfg["models"] = {model_name: model_estimator}

#             engine_out = balanced_clustering_rank_select_stage(
#                 X=X_current,
#                 groups=groups,
#                 feature_names=names_current,
#                 cfg=stage_cfg,
#                 original_feature_indices=original_indices_current,
#             )

#             selected_model_out = engine_out["selected_by_model"][model_name]
#             ranking_model_out = engine_out["final_ranking_by_model"][model_name]
#             detail_model_out = engine_out["detailed_results_by_model"][model_name]

#             X_next = selected_model_out["X"]
#             names_next = list(selected_model_out["selected_feature_names"])

#             selected_idx_local = _validate_stage_selection_output(
#                 selected_model_out["selected_feature_indices_local"],
#                 names_next,
#                 len(original_indices_current),
#                 stage_name,
#             )
#             selected_idx_original = selected_model_out["selected_feature_indices"]

#             row_subsampling_cfg = dict(stage_cfg.get("row_subsampling", {}))
#             row_subsampling_enabled = bool(row_subsampling_cfg.get("enabled", False))
#             row_subsample_train_fraction = float(row_subsampling_cfg.get("train_fraction", 1.0))
#             if not row_subsampling_enabled:
#                 row_subsample_train_fraction = 1.0

#             stage_out = {
#                 "stage": stage_name,
#                 "top_k": int(stage_cfg["top_k"]),
#                 "n_features_in": int(X_current.shape[1]),
#                 "n_features_out": int(X_next.shape[1]),
#                 "cfg_used": deepcopy(stage_cfg),
#                 "scoring": stage_cfg.get("scoring", "silhouette"),
#                 "row_subsampling_enabled": row_subsampling_enabled,
#                 "row_subsample_train_fraction": row_subsample_train_fraction,
#                 "final_ranking": ranking_model_out,
#                 "detailed_results": detail_model_out,
#                 "selected_feature_names": names_next,
#                 "selected_feature_indices": np.asarray(selected_idx_original, dtype=int).copy(),
#                 "selected_feature_indices_local": selected_idx_local.copy(),
#                 "X_selected": X_next,
#             }

#             history.append(stage_out)
#             by_stage[stage_name] = stage_out

#             X_current = X_next
#             names_current = names_next
#             original_indices_current = np.asarray(selected_idx_original, dtype=int).copy()

#         final_by_model[model_name] = {
#             "X": X_current,
#             "feature_names_selected": names_current,
#             "selected_feature_indices": original_indices_current.copy(),
#             "history": history,
#             "by_stage": by_stage,
#         }

#     return {
#         "final_by_model": final_by_model,
#     }


# ============================================================
# Example usage
# ============================================================

# from sklearn.cluster import KMeans
#
# CLUSTER_RANK_SELECT_PIPELINE_CFG = {
#     "defaults": {
#         "scoring": "silhouette",              # cluster-quality metric; higher is better
#                                               # options: "silhouette", "calinski_harabasz",
#                                               # "davies_bouldin" (-1 * score), "inertia" (-1 * score)
#         "n_repeats": 25,                      # number of shuffles per selected feature
#         "target_feature_appearances": 5,      # target appearances per feature across sampled subsets
#         "random_state": 42,                   # reproducibility seed
#         "ranking_metric": "auto",             # "auto", "mean_normalized_rank", or "mean_importance"
#         "group_mode": False,                  # if True, repeatedly sample one row per group first
#         "group_iterations": 5,                # used only when group_mode=True
#         "row_subsampling": {
#             "enabled": True,                  # if True, each feature-subset run samples rows first
#             "train_fraction": 0.9,            # fraction of full rows kept for each run; rest discarded
#         },
#     },
#     "models": {
#         "kmeans_5": KMeans(n_clusters=5, random_state=42, n_init="auto"),
#     },
#     "stages": [
#         {
#             "name": "coarse_cluster_rank_select",
#             "top_k": 20,
#             "subset_sizes": [15],
#             "n_repeats": 25,
#             "target_feature_appearances": 5,
#         },
#         {
#             "name": "final_cluster_rank_select",
#             "top_k": 15,
#             "subset_sizes": [15],
#             "n_repeats": 25,
#             "target_feature_appearances": 50,
#         },
#     ],
# }
#
# cluster_features = balanced_clustering_rank_select_pipeline(
#     X=data["X"],
#     groups=None,
#     feature_names=data["feature_names"],
#     cfg=CLUSTER_RANK_SELECT_PIPELINE_CFG,
# )
#
# selected = cluster_features["final_by_model"]["kmeans_5"]["feature_names_selected"]




# ============================================================
# A. Selected-set stability
# ============================================================
#
# Purpose:
#   Compare selected feature SETS across timepoints.
#
# This section answers:
#   - Which selected features are shared across timepoints?
#   - Which selected features are baseline-specific, week6-specific, etc.?
#   - Which selected features are stable across all timepoints?
#   - How similar are selected feature sets pairwise?
#
# This section does NOT use feature ranks or feature scores.
# Ranking stability and score stability should be handled separately.

# ============================================================
# A1. Extract selected features from pipeline output
# ============================================================

def get_selected_features(
    pipeline_output: Mapping,
    *,
    model_name: str = "kmeans_5",
) -> List[str]:
    """
    Extract the final selected feature names from one clustering feature-selection output.

    Parameters
    ----------
    pipeline_output : Mapping
        Output from `balanced_clustering_rank_select_pipeline(...)`.

    model_name : str, default="kmeans_5"
        Name of the clustering model inside `pipeline_output["final_by_model"]`.

    Returns
    -------
    List[str]
        Final selected feature names for the requested model.

    Raises
    ------
    KeyError
        If the expected model or selected-feature path is not present.
    """
    return list(
        pipeline_output["final_by_model"][model_name]["feature_names_selected"]
    )


# ============================================================
# A2. Validation helpers for selected-feature dictionaries
# ============================================================

def validate_selected_by_timepoint(
    selected_by_timepoint: Mapping[str, Sequence[str]],
    *,
    timepoints: Optional[Sequence[str]] = None,
    require_unique_within_timepoint: bool = True,
) -> List[str]:
    """
    Validate selected-feature inputs and return the timepoint order to use.

    Parameters
    ----------
    selected_by_timepoint : Mapping[str, Sequence[str]]
        Dictionary mapping timepoint name to selected feature names.

        Example:
            {
                "baseline": ["feature_1", "feature_2"],
                "week6": ["feature_1", "feature_3"],
                "month6": ["feature_2", "feature_3"],
            }

    timepoints : Optional[Sequence[str]], default=None
        Explicit timepoint order. If None, the insertion order of
        `selected_by_timepoint` is used.

    require_unique_within_timepoint : bool, default=True
        If True, raise an error if a timepoint contains duplicate selected features.

    Returns
    -------
    List[str]
        Validated timepoint order.

    Raises
    ------
    ValueError
        If inputs are empty, timepoints are missing, or duplicates are found.
    """
    if not selected_by_timepoint:
        raise ValueError("selected_by_timepoint must be a non-empty mapping.")

    if timepoints is None:
        timepoint_order = list(selected_by_timepoint.keys())
    else:
        timepoint_order = list(timepoints)

    if len(timepoint_order) < 2:
        raise ValueError("At least two timepoints are required for stability analysis.")

    missing_timepoints = [
        tp for tp in timepoint_order
        if tp not in selected_by_timepoint
    ]
    if missing_timepoints:
        raise ValueError(
            f"These requested timepoints are missing from selected_by_timepoint: "
            f"{missing_timepoints}"
        )

    for tp in timepoint_order:
        features = list(selected_by_timepoint[tp])

        if len(features) == 0:
            raise ValueError(f"Timepoint {tp!r} has no selected features.")

        if require_unique_within_timepoint:
            duplicated = sorted({
                feature for feature in features
                if features.count(feature) > 1
            })
            if duplicated:
                raise ValueError(
                    f"Timepoint {tp!r} contains duplicate selected features: "
                    f"{duplicated}"
                )

    return timepoint_order


def selected_sets_by_timepoint(
    selected_by_timepoint: Mapping[str, Sequence[str]],
    *,
    timepoints: Optional[Sequence[str]] = None,
) -> Dict[str, Set[str]]:
    """
    Convert selected-feature lists into selected-feature sets.

    Parameters
    ----------
    selected_by_timepoint : Mapping[str, Sequence[str]]
        Dictionary mapping timepoint name to selected feature names.

    timepoints : Optional[Sequence[str]], default=None
        Explicit timepoint order. If None, insertion order is used.

    Returns
    -------
    Dict[str, Set[str]]
        Dictionary mapping each timepoint to a set of selected features.
    """
    timepoint_order = validate_selected_by_timepoint(
        selected_by_timepoint,
        timepoints=timepoints,
    )

    return {
        tp: set(map(str, selected_by_timepoint[tp]))
        for tp in timepoint_order
    }


# ============================================================
# A3. Pairwise selected-set similarity metrics
# ============================================================

def jaccard_similarity(
    features_a: Iterable[str],
    features_b: Iterable[str],
) -> float:
    """
    Compute Jaccard similarity between two selected-feature sets.

    Jaccard similarity is:

        |A intersection B| / |A union B|

    Parameters
    ----------
    features_a : Iterable[str]
        First selected-feature collection.

    features_b : Iterable[str]
        Second selected-feature collection.

    Returns
    -------
    float
        Jaccard similarity. Returns np.nan if both sets are empty.
    """
    set_a = set(features_a)
    set_b = set(features_b)

    union = set_a | set_b

    if len(union) == 0:
        return float(np.nan)

    return len(set_a & set_b) / len(union)


def overlap_fraction_min_denominator(
    features_a: Iterable[str],
    features_b: Iterable[str],
) -> float:
    """
    Compute overlap fraction using the smaller selected set as denominator.

    This is:

        |A intersection B| / min(|A|, |B|)

    This metric answers:
        "What fraction of the smaller selected set is recovered in the other set?"

    Parameters
    ----------
    features_a : Iterable[str]
        First selected-feature collection.

    features_b : Iterable[str]
        Second selected-feature collection.

    Returns
    -------
    float
        Overlap fraction. Returns np.nan if either set is empty.
    """
    set_a = set(features_a)
    set_b = set(features_b)

    denominator = min(len(set_a), len(set_b))

    if denominator == 0:
        return float(np.nan)

    return len(set_a & set_b) / denominator


def compare_selected_feature_sets(
    selected_by_timepoint: Mapping[str, Sequence[str]],
    *,
    timepoints: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Compare selected-feature sets pairwise across timepoints.

    Parameters
    ----------
    selected_by_timepoint : Mapping[str, Sequence[str]]
        Dictionary mapping timepoint name to selected feature names.

    timepoints : Optional[Sequence[str]], default=None
        Explicit timepoint order. If None, insertion order is used.

    Returns
    -------
    pd.DataFrame
        One row per pairwise timepoint comparison.

        Columns include:
            - comparison
            - timepoint_a
            - timepoint_b
            - n_selected_a
            - n_selected_b
            - n_overlap
            - n_union
            - overlap_fraction
            - jaccard
            - overlap_features
            - only_timepoint_a_features
            - only_timepoint_b_features
    """
    timepoint_order = validate_selected_by_timepoint(
        selected_by_timepoint,
        timepoints=timepoints,
    )

    selected_sets = selected_sets_by_timepoint(
        selected_by_timepoint,
        timepoints=timepoint_order,
    )

    rows = []

    for tp_a, tp_b in combinations(timepoint_order, 2):
        set_a = selected_sets[tp_a]
        set_b = selected_sets[tp_b]

        overlap = set_a & set_b
        union = set_a | set_b

        rows.append(
            {
                "comparison": f"{tp_a}_vs_{tp_b}",
                "timepoint_a": tp_a,
                "timepoint_b": tp_b,
                "n_selected_a": len(set_a),
                "n_selected_b": len(set_b),
                "n_overlap": len(overlap),
                "n_union": len(union),
                "overlap_fraction": overlap_fraction_min_denominator(set_a, set_b),
                "jaccard": jaccard_similarity(set_a, set_b),
                "overlap_features": sorted(overlap),
                "only_timepoint_a_features": sorted(set_a - set_b),
                "only_timepoint_b_features": sorted(set_b - set_a),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# A4. Feature-level presence and pattern labels
# ============================================================

def label_three_timepoint_pattern(
    present_timepoints: Set[str],
    *,
    baseline_label: str = "baseline",
    mid_label: str = "week6",
    final_label: str = "month6",
) -> str:
    """
    Assign an interpretable longitudinal pattern label for exactly three timepoints.

    Parameters
    ----------
    present_timepoints : Set[str]
        Set of timepoints where the feature was selected.

    baseline_label : str, default="baseline"
        Name of the first timepoint.

    mid_label : str, default="week6"
        Name of the middle timepoint.

    final_label : str, default="month6"
        Name of the final timepoint.

    Returns
    -------
    str
        Interpretable pattern label.

    Notes
    -----
    These labels are intentionally descriptive rather than statistical.
    They describe selected-feature presence only.
    """
    baseline = baseline_label
    mid = mid_label
    final = final_label

    if present_timepoints == {baseline, mid, final}:
        return "stable_all_timepoints"

    if present_timepoints == {baseline, mid}:
        return "early_driver_fades_by_final"

    if present_timepoints == {baseline, final}:
        return "baseline_and_final_not_mid"

    if present_timepoints == {mid, final}:
        return "post_baseline_emergent"

    if present_timepoints == {baseline}:
        return "baseline_specific"

    if present_timepoints == {mid}:
        return f"{mid}_specific"

    if present_timepoints == {final}:
        return f"{final}_specific"

    return "unclassified"


def label_general_presence_pattern(
    *,
    n_timepoints_selected: int,
    n_timepoints_total: int,
) -> str:
    """
    Assign a general stability label based on feature-selection frequency.

    Parameters
    ----------
    n_timepoints_selected : int
        Number of timepoints where the feature was selected.

    n_timepoints_total : int
        Total number of timepoints considered.

    Returns
    -------
    str
        General stability label.
    """
    if n_timepoints_selected == n_timepoints_total:
        return "stable_all_timepoints"

    if n_timepoints_selected == n_timepoints_total - 1:
        return "mostly_stable"

    if n_timepoints_selected == 1:
        return "timepoint_specific"

    return "partially_stable"


def build_feature_presence_map(
    selected_by_timepoint: Mapping[str, Sequence[str]],
    *,
    timepoints: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build a feature-level table showing selected-feature presence across timepoints.

    Parameters
    ----------
    selected_by_timepoint : Mapping[str, Sequence[str]]
        Dictionary mapping timepoint name to selected feature names.

    timepoints : Optional[Sequence[str]], default=None
        Explicit timepoint order. If None, insertion order is used.

    Returns
    -------
    pd.DataFrame
        One row per feature selected in at least one timepoint.

        Columns include:
            - feature
            - present_timepoints
            - n_timepoints_selected
            - n_timepoints_total
            - selection_frequency
            - selected_<timepoint> columns
            - stability_label
            - pattern_label

    Notes
    -----
    This function only evaluates selected-feature presence/absence.
    It does not use ranking scores or feature importance values.
    """
    timepoint_order = validate_selected_by_timepoint(
        selected_by_timepoint,
        timepoints=timepoints,
    )

    selected_sets = selected_sets_by_timepoint(
        selected_by_timepoint,
        timepoints=timepoint_order,
    )

    all_selected_features = sorted(set().union(*selected_sets.values()))
    n_timepoints_total = len(timepoint_order)

    use_three_timepoint_labels = n_timepoints_total == 3

    rows = []

    for feature in all_selected_features:
        present = [
            tp for tp in timepoint_order
            if feature in selected_sets[tp]
        ]

        present_set = set(present)
        n_present = len(present)

        row = {
            "feature": feature,
            "present_timepoints": ", ".join(present),
            "n_timepoints_selected": n_present,
            "n_timepoints_total": n_timepoints_total,
            "selection_frequency": n_present / n_timepoints_total,
        }

        for tp in timepoint_order:
            row[f"selected_{tp}"] = feature in selected_sets[tp]

        row["stability_label"] = label_general_presence_pattern(
            n_timepoints_selected=n_present,
            n_timepoints_total=n_timepoints_total,
        )

        if use_three_timepoint_labels:
            row["pattern_label"] = label_three_timepoint_pattern(
                present_set,
                baseline_label=timepoint_order[0],
                mid_label=timepoint_order[1],
                final_label=timepoint_order[2],
            )
        else:
            row["pattern_label"] = row["stability_label"]

        rows.append(row)

    out = pd.DataFrame(rows)

    out = out.sort_values(
        by=[
            "n_timepoints_selected",
            "selection_frequency",
            "feature",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return out


# ============================================================
# A5. Optional selected-set stability summary
# ============================================================

def summarize_selected_set_stability(
    selected_by_timepoint: Mapping[str, Sequence[str]],
    *,
    timepoints: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run the full selected-set stability analysis.

    Parameters
    ----------
    selected_by_timepoint : Mapping[str, Sequence[str]]
        Dictionary mapping timepoint name to selected feature names.

    timepoints : Optional[Sequence[str]], default=None
        Explicit timepoint order. If None, insertion order is used.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing:

            "pairwise_overlap"
                Pairwise selected-set overlap summary.

            "feature_presence"
                Feature-level selected-presence map.

            "pattern_counts"
                Counts of features by pattern label.

    Notes
    -----
    This wrapper is convenient for Section A only.
    It does not compute ranking correlations, score stability, ICC, or plots.
    """
    timepoint_order = validate_selected_by_timepoint(
        selected_by_timepoint,
        timepoints=timepoints,
    )

    pairwise_overlap = compare_selected_feature_sets(
        selected_by_timepoint,
        timepoints=timepoint_order,
    )

    feature_presence = build_feature_presence_map(
        selected_by_timepoint,
        timepoints=timepoint_order,
    )

    pattern_counts = (
        feature_presence
        .groupby(["pattern_label", "stability_label"], dropna=False)
        .size()
        .reset_index(name="n_features")
        .sort_values(
            by=["n_features", "pattern_label"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    return {
        "pairwise_overlap": pairwise_overlap,
        "feature_presence": feature_presence,
        "pattern_counts": pattern_counts,
    }

# ============================================================
# A6. Stable selected features by selection frequency
# ============================================================

def get_stable_features_by_frequency(
    feature_presence: pd.DataFrame,
    *,
    min_selection_frequency: float = 0.5,
) -> List[str]:
    """
    Return features selected in at least a requested fraction of timepoints.

    This function uses the selected-feature presence table from Section A.

    Parameters
    ----------
    feature_presence : pd.DataFrame
        Output from build_feature_presence_map(...) or
        selected_set_results["feature_presence"].

        Must contain:
            - feature
            - selection_frequency

    min_selection_frequency : float, default=0.5
        Minimum fraction of timepoints where a feature must be selected.

        Examples with 3 timepoints:
            1.0  -> selected in 3/3 timepoints
            0.67 -> selected in at least 2/3 timepoints
            0.5  -> selected in at least 2/3 timepoints
            0.33 -> selected in at least 1/3 timepoints
            0.0  -> no frequency filtering among features selected at least once

    Returns
    -------
    List[str]
        Feature names passing the selection-frequency threshold.
    """
    if not (0.0 <= min_selection_frequency <= 1.0):
        raise ValueError("min_selection_frequency must be between 0.0 and 1.0.")

    required_cols = {"feature", "selection_frequency"}
    missing_cols = sorted(required_cols - set(feature_presence.columns))

    if missing_cols:
        raise KeyError(
            f"feature_presence is missing required columns: {missing_cols}"
        )

    stable_features = (
        feature_presence
        .loc[
            feature_presence["selection_frequency"] >= min_selection_frequency,
            "feature",
        ]
        .astype(str)
        .tolist()
    )

    return stable_features

# ============================================================
# B. Ranking stability
# ============================================================
#
# Purpose:
#   Compare feature RANKINGS and ranking SCORES across timepoints.
#
# This section answers:
#   - Are feature ranking scores correlated across timepoints?
#   - Which features moved up or down in rank?
#   - Which features changed most in score?
#   - Which features are the largest overall movers?
#
# This section assumes each timepoint has a final_ranking dataframe
# from your clustering feature-selection pipeline.
#
# Expected ranking dataframe columns:
#   - feature
#   - mean_normalized_rank
#   - mean_importance
#
# Most commonly used score_col:
#   - "mean_normalized_rank"
#
# Notes:
#   - Higher score is assumed to mean better / more important.
#   - Rank 1 means best feature.
#   - Negative rank_change means the feature moved UP at timepoint B.
#   - Positive rank_change means the feature moved DOWN at timepoint B.



# ============================================================
# B1. Extract final rankings from pipeline output
# ============================================================

def get_final_ranking_table(
    pipeline_output: Mapping,
    *,
    model_name: str = "kmeans_5",
) -> pd.DataFrame:
    """
    Extract the final-stage ranking table from one clustering feature-selection output.

    Parameters
    ----------
    pipeline_output : Mapping
        Output from `balanced_clustering_rank_select_pipeline(...)`.

    model_name : str, default="kmeans_5"
        Model name inside `pipeline_output["final_by_model"]`.

    Returns
    -------
    pd.DataFrame
        Final ranking dataframe from the last pipeline stage.

    Raises
    ------
    KeyError
        If the expected path is missing from the pipeline output.
    """
    # The pipeline stores each stage in history.
    # The final ranking table lives in the last stage.
    ranking_df = pipeline_output["final_by_model"][model_name]["history"][-1]["final_ranking"]

    # Return a copy so downstream edits do not mutate the original output.
    return ranking_df.copy()

# ============================================================
# B1b. Extract final selected ranking table from pipeline output
# ============================================================

def get_final_selected_ranking_table(
    pipeline_output: Mapping,
    *,
    model_name: str = "kmeans_5",
) -> pd.DataFrame:
    """
    Extract the final-stage ranking table restricted to the final selected features.

    This differs from get_final_ranking_table(...), which returns the full
    final-stage ranking table. In your pipeline, the final-stage ranking may
    contain more rows than the final selected top_k features.

    Parameters
    ----------
    pipeline_output : Mapping
        Output from balanced_clustering_rank_select_pipeline(...).

    model_name : str, default="kmeans_5"
        Model name inside pipeline_output["final_by_model"].

    Returns
    -------
    pd.DataFrame
        Final-stage ranking dataframe restricted to the final selected features.
    """
    selected_features = set(
        map(
            str,
            pipeline_output["final_by_model"][model_name]["feature_names_selected"],
        )
    )

    final_ranking = (
        pipeline_output["final_by_model"][model_name]["history"][-1]["final_ranking"]
        .copy()
    )

    out = (
        final_ranking
        .loc[final_ranking["feature"].astype(str).isin(selected_features)]
        .copy()
        .reset_index(drop=True)
    )

    return out

# ============================================================
# B1c. Build final selected ranking tables across timepoints
# ============================================================

def get_final_selected_ranking_by_timepoint(
    pipeline_output_by_timepoint: Mapping[str, Mapping],
    *,
    model_name: str = "kmeans_5",
    timepoints: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Build ranking_by_timepoint using only the final selected features from each
    pipeline output.

    Use this instead of get_final_ranking_table(...) when ranking stability
    should focus only on the final selected top_k features.

    Parameters
    ----------
    pipeline_output_by_timepoint : Mapping[str, Mapping]
        Dictionary mapping timepoint name to pipeline output.

        Example:
            {
                "baseline": cluster_features,
                "week6": cluster_features2,
                "month6": cluster_features3,
            }

    model_name : str, default="kmeans_5"
        Model name inside each pipeline output.

    timepoints : Optional[Sequence[str]], default=None
        Explicit timepoint order. If None, dictionary insertion order is used.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Timepoint -> final selected ranking dataframe.
    """
    if not pipeline_output_by_timepoint:
        raise ValueError("pipeline_output_by_timepoint must be a non-empty mapping.")

    if timepoints is None:
        timepoint_order = list(pipeline_output_by_timepoint.keys())
    else:
        timepoint_order = list(timepoints)

    missing_timepoints = [
        tp for tp in timepoint_order
        if tp not in pipeline_output_by_timepoint
    ]

    if missing_timepoints:
        raise KeyError(
            f"These requested timepoints are missing from pipeline_output_by_timepoint: "
            f"{missing_timepoints}"
        )

    return {
        tp: get_final_selected_ranking_table(
            pipeline_output_by_timepoint[tp],
            model_name=model_name,
        )
        for tp in timepoint_order
    }

# ============================================================
# B1d. Filter ranking tables to stable selected features
# ============================================================

def filter_ranking_to_features(
    ranking_df: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    """
    Filter one ranking dataframe to a requested feature list.

    Parameters
    ----------
    ranking_df : pd.DataFrame
        Ranking table containing at least a "feature" column.

    features : Sequence[str]
        Features to keep.

    Returns
    -------
    pd.DataFrame
        Filtered ranking table.
    """
    if "feature" not in ranking_df.columns:
        raise KeyError("ranking_df must contain a 'feature' column.")

    feature_set = set(map(str, features))

    out = (
        ranking_df
        .loc[ranking_df["feature"].astype(str).isin(feature_set)]
        .copy()
        .reset_index(drop=True)
    )

    return out


def filter_ranking_by_timepoint_to_features(
    ranking_by_timepoint: Mapping[str, pd.DataFrame],
    *,
    features: Sequence[str],
    timepoints: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Filter each timepoint ranking table to the same requested feature set.

    Parameters
    ----------
    ranking_by_timepoint : Mapping[str, pd.DataFrame]
        Dictionary mapping timepoint name to ranking dataframe.

    features : Sequence[str]
        Features to keep.

    timepoints : Optional[Sequence[str]], default=None
        Explicit timepoint order. If None, dictionary insertion order is used.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Filtered ranking tables by timepoint.
    """
    if not ranking_by_timepoint:
        raise ValueError("ranking_by_timepoint must be a non-empty mapping.")

    if timepoints is None:
        timepoint_order = list(ranking_by_timepoint.keys())
    else:
        timepoint_order = list(timepoints)

    missing_timepoints = [
        tp for tp in timepoint_order
        if tp not in ranking_by_timepoint
    ]

    if missing_timepoints:
        raise KeyError(
            f"These requested timepoints are missing from ranking_by_timepoint: "
            f"{missing_timepoints}"
        )

    return {
        tp: filter_ranking_to_features(
            ranking_by_timepoint[tp],
            features=features,
        )
        for tp in timepoint_order
    }
    
# ============================================================
# B2. Validation and score extraction helpers
# ============================================================

def validate_ranking_by_timepoint(
    ranking_by_timepoint: Mapping[str, pd.DataFrame],
    *,
    timepoints: Optional[Sequence[str]] = None,
    score_col: str = "mean_normalized_rank",
) -> List[str]:
    """
    Validate ranking tables and return the timepoint order to use.

    Parameters
    ----------
    ranking_by_timepoint : Mapping[str, pd.DataFrame]
        Dictionary mapping timepoint name to final ranking dataframe.

    timepoints : Optional[Sequence[str]], default=None
        Explicit timepoint order. If None, insertion order of the dictionary is used.

    score_col : str, default="mean_normalized_rank"
        Ranking score column to use for comparisons.

    Returns
    -------
    List[str]
        Validated timepoint order.

    Raises
    ------
    ValueError
        If fewer than two timepoints are provided.

    KeyError
        If a requested timepoint is missing or a ranking table is missing required columns.
    """
    if not ranking_by_timepoint:
        raise ValueError("ranking_by_timepoint must be a non-empty mapping.")

    # Use explicit timepoint order when provided.
    # Otherwise, preserve dictionary insertion order.
    if timepoints is None:
        timepoint_order = list(ranking_by_timepoint.keys())
    else:
        timepoint_order = list(timepoints)

    if len(timepoint_order) < 2:
        raise ValueError("At least two timepoints are required for ranking stability analysis.")

    # Confirm all requested timepoints exist.
    missing_timepoints = [
        tp for tp in timepoint_order
        if tp not in ranking_by_timepoint
    ]
    if missing_timepoints:
        raise KeyError(
            f"These requested timepoints are missing from ranking_by_timepoint: "
            f"{missing_timepoints}"
        )

    # Check each ranking table for required columns.
    for tp in timepoint_order:
        ranking_df = ranking_by_timepoint[tp]

        required_cols = {"feature", score_col}
        missing_cols = sorted(required_cols - set(ranking_df.columns))

        if missing_cols:
            raise KeyError(
                f"Ranking table for timepoint {tp!r} is missing columns: {missing_cols}. "
                f"Available columns: {list(ranking_df.columns)}"
            )

        if ranking_df.empty:
            raise ValueError(f"Ranking table for timepoint {tp!r} is empty.")

        # Feature names should be unique in a final ranking table.
        duplicated_features = (
            ranking_df["feature"]
            .astype(str)
            .duplicated()
        )

        if duplicated_features.any():
            duplicates = (
                ranking_df.loc[duplicated_features, "feature"]
                .astype(str)
                .unique()
                .tolist()
            )
            raise ValueError(
                f"Ranking table for timepoint {tp!r} contains duplicate features: "
                f"{duplicates}"
            )

    return timepoint_order


def ranking_score_series(
    ranking_df: pd.DataFrame,
    *,
    score_col: str = "mean_normalized_rank",
) -> pd.Series:
    """
    Convert a ranking dataframe into a feature-indexed score series.

    Parameters
    ----------
    ranking_df : pd.DataFrame
        Final ranking dataframe.

    score_col : str, default="mean_normalized_rank"
        Score column to extract.

    Returns
    -------
    pd.Series
        Index is feature name. Values are feature scores.

    Notes
    -----
    Higher score is assumed to mean better / more important.
    """
    # Keep only the feature and score columns.
    # Feature names are converted to strings for consistency.
    out = (
        ranking_df[["feature", score_col]]
        .copy()
        .assign(feature=lambda df: df["feature"].astype(str))
        .set_index("feature")[score_col]
        .astype(float)
    )

    return out


def align_pairwise_scores(
    ranking_a: pd.DataFrame,
    ranking_b: pd.DataFrame,
    *,
    score_col: str = "mean_normalized_rank",
    missing_policy: MissingFeaturePolicy = "intersection",
) -> pd.DataFrame:
    """
    Align two ranking tables into one feature-level score table.

    Parameters
    ----------
    ranking_a : pd.DataFrame
        Ranking table for timepoint A.

    ranking_b : pd.DataFrame
        Ranking table for timepoint B.

    score_col : str, default="mean_normalized_rank"
        Ranking score column to compare.

    missing_policy : {"intersection", "union_fill_zero", "union_worst_rank"},
        default="intersection"
        How to handle features that appear in one ranking table but not the other.

        "intersection":
            Keep only features present in both rankings.

        "union_fill_zero":
            Keep all features from either ranking.
            Missing scores are filled with 0.0.
            This is often reasonable for mean_normalized_rank because 0 is the
            natural low end of the normalized-rank scale.

        "union_worst_rank":
            Keep all features from either ranking.
            Missing features are assigned worse-than-observed rank later.
            Scores remain NaN in this aligned table and are handled during ranking.

    Returns
    -------
    pd.DataFrame
        Feature-level aligned scores.

        Columns:
            - feature
            - score_a
            - score_b
            - present_a
            - present_b
    """
    # Convert each ranking table into a feature -> score series.
    scores_a = ranking_score_series(ranking_a, score_col=score_col)
    scores_b = ranking_score_series(ranking_b, score_col=score_col)

    if missing_policy == "intersection":
        # Only evaluate features that appear in both rankings.
        features = sorted(set(scores_a.index) & set(scores_b.index))

    elif missing_policy in {"union_fill_zero", "union_worst_rank"}:
        # Evaluate all features appearing in either ranking.
        features = sorted(set(scores_a.index) | set(scores_b.index))

    else:
        raise ValueError(
            "missing_policy must be one of: "
            "'intersection', 'union_fill_zero', 'union_worst_rank'."
        )

    # Build one aligned table.
    # reindex introduces NaN for missing features.
    aligned = pd.DataFrame(
        {
            "feature": features,
            "score_a": scores_a.reindex(features).to_numpy(dtype=float),
            "score_b": scores_b.reindex(features).to_numpy(dtype=float),
        }
    )

    # Track whether the feature was actually present in each timepoint.
    # This is useful for interpretation when using union-based policies.
    aligned["present_a"] = aligned["feature"].isin(scores_a.index)
    aligned["present_b"] = aligned["feature"].isin(scores_b.index)

    if missing_policy == "union_fill_zero":
        # Fill missing values with 0.
        # This treats missing features as very low scoring.
        aligned[["score_a", "score_b"]] = aligned[["score_a", "score_b"]].fillna(0.0)

    return aligned


# ============================================================
# B3. Pairwise rank-change and score-change tables
# ============================================================

def pairwise_rank_score_change_table(
    ranking_a: pd.DataFrame,
    ranking_b: pd.DataFrame,
    *,
    timepoint_a: str,
    timepoint_b: str,
    score_col: str = "mean_normalized_rank",
    missing_policy: MissingFeaturePolicy = "intersection",
) -> pd.DataFrame:
    """
    Build a feature-level table of score and rank changes between two timepoints.

    Parameters
    ----------
    ranking_a : pd.DataFrame
        Ranking table for timepoint A.

    ranking_b : pd.DataFrame
        Ranking table for timepoint B.

    timepoint_a : str
        Name of timepoint A.

    timepoint_b : str
        Name of timepoint B.

    score_col : str, default="mean_normalized_rank"
        Ranking score column to compare.

    missing_policy : {"intersection", "union_fill_zero", "union_worst_rank"},
        default="intersection"
        How to handle features missing from one ranking table.

    Returns
    -------
    pd.DataFrame
        One row per compared feature.

        Main columns:
            - feature
            - present_<timepoint_a>
            - present_<timepoint_b>
            - <score_col>_<timepoint_a>
            - <score_col>_<timepoint_b>
            - score_change
            - abs_score_change
            - rank_<timepoint_a>
            - rank_<timepoint_b>
            - rank_change
            - abs_rank_change
            - rank_direction
            - score_direction

    Notes
    -----
    Rank convention:
        rank 1 = best feature.

    rank_change:
        rank_b - rank_a

    Interpretation:
        negative rank_change -> feature moved up at timepoint B
        positive rank_change -> feature moved down at timepoint B
        zero rank_change     -> same rank position
    """
    aligned = align_pairwise_scores(
        ranking_a,
        ranking_b,
        score_col=score_col,
        missing_policy=missing_policy,
    )

    # If no features are available after alignment, return an empty table
    # with the expected columns.
    if aligned.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                f"present_{timepoint_a}",
                f"present_{timepoint_b}",
                f"{score_col}_{timepoint_a}",
                f"{score_col}_{timepoint_b}",
                "score_change",
                "abs_score_change",
                f"rank_{timepoint_a}",
                f"rank_{timepoint_b}",
                "rank_change",
                "abs_rank_change",
                "rank_direction",
                "score_direction",
                "missing_policy",
            ]
        )

    # Copy aligned scores so we can create ranks.
    score_a = aligned["score_a"].copy()
    score_b = aligned["score_b"].copy()

    if missing_policy == "union_worst_rank":
        # For rank computation only, assign missing features worse-than-observed scores.
        # We do NOT overwrite the displayed scores; missing displayed scores stay NaN.
        #
        # This preserves the information that the feature was absent while still
        # allowing rank movement to be computed.
        min_a = score_a.min(skipna=True)
        min_b = score_b.min(skipna=True)

        # If all scores are missing on one side, fallback to 0.
        # This should rarely happen if inputs passed validation.
        fill_a = 0.0 if pd.isna(min_a) else min_a - 1.0
        fill_b = 0.0 if pd.isna(min_b) else min_b - 1.0

        rank_score_a = score_a.fillna(fill_a)
        rank_score_b = score_b.fillna(fill_b)

    else:
        # For intersection and union_fill_zero, scores are already rankable.
        rank_score_a = score_a
        rank_score_b = score_b

    # Higher score means better feature.
    # Therefore rank in descending order.
    ranks_a = rank_score_a.rank(ascending=False, method="average")
    ranks_b = rank_score_b.rank(ascending=False, method="average")

    out = aligned.copy()

    # Rename score and presence columns to include timepoint names.
    out = out.rename(
        columns={
            "score_a": f"{score_col}_{timepoint_a}",
            "score_b": f"{score_col}_{timepoint_b}",
            "present_a": f"present_{timepoint_a}",
            "present_b": f"present_{timepoint_b}",
        }
    )

    # Score movement: positive means score increased at timepoint B.
    out["score_change"] = (
        out[f"{score_col}_{timepoint_b}"]
        - out[f"{score_col}_{timepoint_a}"]
    )
    out["abs_score_change"] = out["score_change"].abs()

    # Rank movement: negative means feature moved up at timepoint B.
    out[f"rank_{timepoint_a}"] = ranks_a.to_numpy(dtype=float)
    out[f"rank_{timepoint_b}"] = ranks_b.to_numpy(dtype=float)

    out["rank_change"] = (
        out[f"rank_{timepoint_b}"]
        - out[f"rank_{timepoint_a}"]
    )
    out["abs_rank_change"] = out["rank_change"].abs()

    # Human-readable rank direction.
    out["rank_direction"] = np.select(
        [
            out["rank_change"] < 0,
            out["rank_change"] > 0,
            out["rank_change"] == 0,
        ],
        [
            f"moved_up_at_{timepoint_b}",
            f"moved_down_at_{timepoint_b}",
            "unchanged",
        ],
        default="unknown",
    )

    # Human-readable score direction.
    out["score_direction"] = np.select(
        [
            out["score_change"] > 0,
            out["score_change"] < 0,
            out["score_change"] == 0,
        ],
        [
            f"score_increased_at_{timepoint_b}",
            f"score_decreased_at_{timepoint_b}",
            "unchanged",
        ],
        default="unknown",
    )

    out["missing_policy"] = missing_policy

    # Put largest rank movers first by default.
    out = out.sort_values(
        by=["abs_rank_change", "abs_score_change", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return out


def pairwise_score_change_table(
    rank_score_change_df: pd.DataFrame,
    *,
    timepoint_a: str,
    timepoint_b: str,
    score_col: str = "mean_normalized_rank",
) -> pd.DataFrame:
    """
    Extract a score-focused change table from a combined rank/score table.

    Parameters
    ----------
    rank_score_change_df : pd.DataFrame
        Output from `pairwise_rank_score_change_table(...)`.

    timepoint_a : str
        Name of timepoint A.

    timepoint_b : str
        Name of timepoint B.

    score_col : str, default="mean_normalized_rank"
        Score column used in the combined table.

    Returns
    -------
    pd.DataFrame
        Score-change focused table sorted by absolute score change.
    """
    score_cols = [
        "feature",
        f"present_{timepoint_a}",
        f"present_{timepoint_b}",
        f"{score_col}_{timepoint_a}",
        f"{score_col}_{timepoint_b}",
        "score_change",
        "abs_score_change",
        "score_direction",
        "missing_policy",
    ]

    missing_cols = [
        col for col in score_cols
        if col not in rank_score_change_df.columns
    ]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    # Keep only score-related columns and sort by score movement.
    out = (
        rank_score_change_df[score_cols]
        .copy()
        .sort_values(
            by=["abs_score_change", "feature"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    return out


def pairwise_rank_change_table(
    rank_score_change_df: pd.DataFrame,
    *,
    timepoint_a: str,
    timepoint_b: str,
) -> pd.DataFrame:
    """
    Extract a rank-focused change table from a combined rank/score table.

    Parameters
    ----------
    rank_score_change_df : pd.DataFrame
        Output from `pairwise_rank_score_change_table(...)`.

    timepoint_a : str
        Name of timepoint A.

    timepoint_b : str
        Name of timepoint B.

    Returns
    -------
    pd.DataFrame
        Rank-change focused table sorted by absolute rank change.
    """
    rank_cols = [
        "feature",
        f"present_{timepoint_a}",
        f"present_{timepoint_b}",
        f"rank_{timepoint_a}",
        f"rank_{timepoint_b}",
        "rank_change",
        "abs_rank_change",
        "rank_direction",
        "missing_policy",
    ]

    missing_cols = [
        col for col in rank_cols
        if col not in rank_score_change_df.columns
    ]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    # Keep only rank-related columns and sort by rank movement.
    out = (
        rank_score_change_df[rank_cols]
        .copy()
        .sort_values(
            by=["abs_rank_change", "feature"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    return out


# ============================================================
# B4. Pairwise ranking correlations
# ============================================================

def compute_pairwise_ranking_correlation(
    rank_score_change_df: pd.DataFrame,
    *,
    timepoint_a: str,
    timepoint_b: str,
    score_col: str = "mean_normalized_rank",
) -> Dict[str, float]:
    """
    Compute Spearman and Kendall correlation for two aligned score vectors.

    Parameters
    ----------
    rank_score_change_df : pd.DataFrame
        Output from `pairwise_rank_score_change_table(...)`.

    timepoint_a : str
        Name of timepoint A.

    timepoint_b : str
        Name of timepoint B.

    score_col : str, default="mean_normalized_rank"
        Score column used for the comparison.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
            - n_features_compared
            - spearman_corr
            - spearman_pvalue
            - kendall_corr
            - kendall_pvalue

    Notes
    -----
    P-values should be treated as exploratory because features may be correlated.
    The correlations themselves are usually the main descriptive outputs.
    """
    col_a = f"{score_col}_{timepoint_a}"
    col_b = f"{score_col}_{timepoint_b}"

    required_cols = [col_a, col_b]
    missing_cols = [
        col for col in required_cols
        if col not in rank_score_change_df.columns
    ]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    # Drop rows with missing displayed scores.
    # This matters for missing_policy="union_worst_rank", where absent scores remain NaN.
    corr_df = rank_score_change_df[[col_a, col_b]].dropna(axis=0, how="any")

    n_features = len(corr_df)

    if n_features < 2:
        return {
            "n_features_compared": float(n_features),
            "spearman_corr": np.nan,
            "spearman_pvalue": np.nan,
            "kendall_corr": np.nan,
            "kendall_pvalue": np.nan,
        }

    values_a = corr_df[col_a].to_numpy(dtype=float)
    values_b = corr_df[col_b].to_numpy(dtype=float)

    # If either vector is constant, scipy may return nan.
    # That is okay; we preserve nan because correlation is not meaningful.
    spearman = spearmanr(values_a, values_b)
    kendall = kendalltau(values_a, values_b)

    return {
        "n_features_compared": float(n_features),
        "spearman_corr": float(spearman.statistic),
        "spearman_pvalue": float(spearman.pvalue),
        "kendall_corr": float(kendall.statistic),
        "kendall_pvalue": float(kendall.pvalue),
    }


# ============================================================
# B5. Top movers
# ============================================================

def top_ranking_movers(
    rank_score_change_df: pd.DataFrame,
    *,
    top_n: int = 10,
    sort_by: Literal[
        "abs_rank_change",
        "abs_score_change",
        "combined_movement",
    ] = "abs_rank_change",
) -> pd.DataFrame:
    """
    Return the top moving features between two timepoints.

    Parameters
    ----------
    rank_score_change_df : pd.DataFrame
        Output from `pairwise_rank_score_change_table(...)`.

    top_n : int, default=10
        Number of top movers to return.

    sort_by : {"abs_rank_change", "abs_score_change", "combined_movement"},
        default="abs_rank_change"
        Sorting rule for top movers.

        "abs_rank_change":
            Features with largest absolute rank movement.

        "abs_score_change":
            Features with largest absolute score movement.

        "combined_movement":
            Average of normalized absolute rank movement and normalized absolute
            score movement. This is useful when you want features that changed
            substantially on both scales.

    Returns
    -------
    pd.DataFrame
        Top moving features.

    Notes
    -----
    This function does not plot anything.
    Plotting should stay separate.
    """
    if top_n < 1:
        raise ValueError("top_n must be >= 1.")

    df = rank_score_change_df.copy()

    required_cols = {
        "feature",
        "abs_rank_change",
        "abs_score_change",
        "rank_change",
        "score_change",
    }
    missing_cols = sorted(required_cols - set(df.columns))

    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    if sort_by == "combined_movement":
        # Normalize rank movement and score movement to [0, 1]-like scales.
        # This prevents rank movement from dominating simply because rank units
        # are larger than score units.
        max_rank_change = df["abs_rank_change"].max()
        max_score_change = df["abs_score_change"].max()

        if max_rank_change == 0 or pd.isna(max_rank_change):
            df["rank_movement_scaled"] = 0.0
        else:
            df["rank_movement_scaled"] = df["abs_rank_change"] / max_rank_change

        if max_score_change == 0 or pd.isna(max_score_change):
            df["score_movement_scaled"] = 0.0
        else:
            df["score_movement_scaled"] = df["abs_score_change"] / max_score_change

        # Combined score: equal weight to rank movement and score movement.
        df["combined_movement"] = (
            df["rank_movement_scaled"]
            + df["score_movement_scaled"]
        ) / 2.0

        sort_col = "combined_movement"

    elif sort_by in {"abs_rank_change", "abs_score_change"}:
        sort_col = sort_by

    else:
        raise ValueError(
            "sort_by must be one of: "
            "'abs_rank_change', 'abs_score_change', 'combined_movement'."
        )

    out = (
        df.sort_values(
            by=[sort_col, "feature"],
            ascending=[False, True],
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return out


# ============================================================
# B6. Full ranking-stability wrapper
# ============================================================

def summarize_ranking_stability(
    ranking_by_timepoint: Mapping[str, pd.DataFrame],
    *,
    timepoints: Optional[Sequence[str]] = None,
    score_col: str = "mean_normalized_rank",
    missing_policy: MissingFeaturePolicy = "intersection",
    top_n_movers: int = 10,
) -> Dict[str, object]:
    """
    Run the full Section B ranking-stability analysis.

    Parameters
    ----------
    ranking_by_timepoint : Mapping[str, pd.DataFrame]
        Dictionary mapping timepoint name to final ranking dataframe.

    timepoints : Optional[Sequence[str]], default=None
        Explicit timepoint order. If None, insertion order is used.

    score_col : str, default="mean_normalized_rank"
        Ranking score column to compare.

    missing_policy : {"intersection", "union_fill_zero", "union_worst_rank"},
        default="intersection"
        How to handle features missing from one ranking table.

    top_n_movers : int, default=10
        Number of top movers to return per pairwise comparison.

    Returns
    -------
    Dict[str, object]
        Dictionary containing:

            "ranking_correlation_summary" : pd.DataFrame
                Pairwise Spearman/Kendall summary.

            "pairwise_rank_score_change_tables" : Dict[str, pd.DataFrame]
                Full rank + score change table for each comparison.

            "pairwise_rank_change_tables" : Dict[str, pd.DataFrame]
                Rank-focused table for each comparison.

            "pairwise_score_change_tables" : Dict[str, pd.DataFrame]
                Score-focused table for each comparison.

            "top_rank_movers" : Dict[str, pd.DataFrame]
                Top movers sorted by absolute rank change.

            "top_score_movers" : Dict[str, pd.DataFrame]
                Top movers sorted by absolute score change.

            "top_combined_movers" : Dict[str, pd.DataFrame]
                Top movers sorted by combined rank and score movement.
    """
    timepoint_order = validate_ranking_by_timepoint(
        ranking_by_timepoint,
        timepoints=timepoints,
        score_col=score_col,
    )

    summary_rows = []

    pairwise_rank_score_change_tables: Dict[str, pd.DataFrame] = {}
    pairwise_rank_change_tables: Dict[str, pd.DataFrame] = {}
    pairwise_score_change_tables: Dict[str, pd.DataFrame] = {}

    top_rank_movers_by_comparison: Dict[str, pd.DataFrame] = {}
    top_score_movers_by_comparison: Dict[str, pd.DataFrame] = {}
    top_combined_movers_by_comparison: Dict[str, pd.DataFrame] = {}

    for tp_a, tp_b in combinations(timepoint_order, 2):
        comparison = f"{tp_a}_vs_{tp_b}"

        # Build one combined feature-level table for this pair.
        rank_score_change_df = pairwise_rank_score_change_table(
            ranking_by_timepoint[tp_a],
            ranking_by_timepoint[tp_b],
            timepoint_a=tp_a,
            timepoint_b=tp_b,
            score_col=score_col,
            missing_policy=missing_policy,
        )

        pairwise_rank_score_change_tables[comparison] = rank_score_change_df

        # Split the combined table into rank-focused and score-focused views.
        pairwise_rank_change_tables[comparison] = pairwise_rank_change_table(
            rank_score_change_df,
            timepoint_a=tp_a,
            timepoint_b=tp_b,
        )

        pairwise_score_change_tables[comparison] = pairwise_score_change_table(
            rank_score_change_df,
            timepoint_a=tp_a,
            timepoint_b=tp_b,
            score_col=score_col,
        )

        # Compute Spearman/Kendall correlations for this pair.
        corr = compute_pairwise_ranking_correlation(
            rank_score_change_df,
            timepoint_a=tp_a,
            timepoint_b=tp_b,
            score_col=score_col,
        )

        summary_rows.append(
            {
                "comparison": comparison,
                "timepoint_a": tp_a,
                "timepoint_b": tp_b,
                "score_col": score_col,
                "missing_policy": missing_policy,
                **corr,
            }
        )

        # Extract top movers using three definitions.
        top_rank_movers_by_comparison[comparison] = top_ranking_movers(
            rank_score_change_df,
            top_n=top_n_movers,
            sort_by="abs_rank_change",
        )

        top_score_movers_by_comparison[comparison] = top_ranking_movers(
            rank_score_change_df,
            top_n=top_n_movers,
            sort_by="abs_score_change",
        )

        top_combined_movers_by_comparison[comparison] = top_ranking_movers(
            rank_score_change_df,
            top_n=top_n_movers,
            sort_by="combined_movement",
        )

    ranking_correlation_summary = pd.DataFrame(summary_rows)

    return {
        "ranking_correlation_summary": ranking_correlation_summary,
        "pairwise_rank_score_change_tables": pairwise_rank_score_change_tables,
        "pairwise_rank_change_tables": pairwise_rank_change_tables,
        "pairwise_score_change_tables": pairwise_score_change_tables,
        "top_rank_movers": top_rank_movers_by_comparison,
        "top_score_movers": top_score_movers_by_comparison,
        "top_combined_movers": top_combined_movers_by_comparison,
    }


# ============================================================
# C. Score stability
# ============================================================
#
# Purpose:
#   Evaluate stability of feature SCORES across timepoints.
#
# This section answers:
#   - Are feature scores globally stable across timepoints?
#   - Which features have low vs high score variance?
#   - Which features have the largest score range?
#   - Which features have the largest baseline-to-follow-up changes?
#   - What is the ICC-style stability of feature scores?
#
# This section uses ranking score values, usually:
#   - mean_normalized_rank
#   - mean_importance
#
# Recommended default:
#   score_col="mean_normalized_rank"


# ============================================================
# C1. Validation and score extraction
# ============================================================

def validate_score_stability_inputs(
    ranking_by_timepoint: Mapping[str, pd.DataFrame],
    *,
    timepoints: Optional[Sequence[str]] = None,
    score_col: str = "mean_normalized_rank",
) -> list[str]:
    """
    Validate ranking tables for score-stability analysis.

    Parameters
    ----------
    ranking_by_timepoint : Mapping[str, pd.DataFrame]
        Dictionary mapping timepoint name to final ranking dataframe.

    timepoints : Optional[Sequence[str]], default=None
        Explicit timepoint order. If None, insertion order of
        `ranking_by_timepoint` is used.

    score_col : str, default="mean_normalized_rank"
        Score column to use for score stability.

    Returns
    -------
    list[str]
        Validated timepoint order.

    Raises
    ------
    ValueError
        If the input is empty or fewer than two timepoints are provided.

    KeyError
        If required columns are missing.
    """
    if not ranking_by_timepoint:
        raise ValueError("ranking_by_timepoint must be a non-empty mapping.")

    # Use user-specified timepoint order when provided.
    # Otherwise, preserve dictionary insertion order.
    if timepoints is None:
        timepoint_order = list(ranking_by_timepoint.keys())
    else:
        timepoint_order = list(timepoints)

    if len(timepoint_order) < 2:
        raise ValueError("At least two timepoints are required for score stability.")

    # Make sure every requested timepoint exists.
    missing_timepoints = [
        tp for tp in timepoint_order
        if tp not in ranking_by_timepoint
    ]
    if missing_timepoints:
        raise KeyError(
            f"Missing timepoints in ranking_by_timepoint: {missing_timepoints}"
        )

    # Validate each ranking table.
    for tp in timepoint_order:
        ranking_df = ranking_by_timepoint[tp]

        required_cols = {"feature", score_col}
        missing_cols = sorted(required_cols - set(ranking_df.columns))

        if missing_cols:
            raise KeyError(
                f"Ranking table for {tp!r} is missing columns: {missing_cols}. "
                f"Available columns: {list(ranking_df.columns)}"
            )

        if ranking_df.empty:
            raise ValueError(f"Ranking table for {tp!r} is empty.")

        # Duplicate feature rows would make score alignment ambiguous.
        duplicated_features = ranking_df["feature"].astype(str).duplicated()

        if duplicated_features.any():
            duplicates = (
                ranking_df.loc[duplicated_features, "feature"]
                .astype(str)
                .unique()
                .tolist()
            )
            raise ValueError(
                f"Ranking table for {tp!r} contains duplicate features: {duplicates}"
            )

    return timepoint_order


def extract_feature_score_series(
    ranking_df: pd.DataFrame,
    *,
    score_col: str = "mean_normalized_rank",
) -> pd.Series:
    """
    Extract a feature-indexed score series from one ranking table.

    Parameters
    ----------
    ranking_df : pd.DataFrame
        Final ranking dataframe.

    score_col : str, default="mean_normalized_rank"
        Score column to extract.

    Returns
    -------
    pd.Series
        Index is feature name. Values are score values.
    """
    # Convert feature names to strings for consistent alignment.
    # Convert scores to float so downstream numeric summaries are reliable.
    scores = (
        ranking_df[["feature", score_col]]
        .copy()
        .assign(feature=lambda df: df["feature"].astype(str))
        .set_index("feature")[score_col]
        .astype(float)
    )

    return scores


# ============================================================
# C2. Build feature x timepoint score matrix
# ============================================================

def build_feature_score_matrix(
    ranking_by_timepoint: Mapping[str, pd.DataFrame],
    *,
    timepoints: Optional[Sequence[str]] = None,
    score_col: str = "mean_normalized_rank",
    missing_policy: ScoreMissingPolicy = "intersection",
) -> pd.DataFrame:
    """
    Build a feature x timepoint score matrix.

    Parameters
    ----------
    ranking_by_timepoint : Mapping[str, pd.DataFrame]
        Dictionary mapping timepoint name to final ranking dataframe.

    timepoints : Optional[Sequence[str]], default=None
        Explicit timepoint order. If None, insertion order is used.

    score_col : str, default="mean_normalized_rank"
        Score column to use.

    missing_policy : {"intersection", "union_fill_zero", "union_keep_na"},
        default="intersection"
        How to handle features that are missing from one or more timepoints.

        "intersection":
            Keep only features that appear in every timepoint ranking table.

        "union_fill_zero":
            Keep features that appear in any timepoint.
            Fill missing scores with 0.0.
            This is reasonable for mean_normalized_rank because 0 is the
            lowest normalized-rank value.

        "union_keep_na":
            Keep features that appear in any timepoint.
            Leave missing scores as NaN.
            Per-feature summaries will use skip-NaN logic where appropriate.
            ICC will require complete rows unless handled separately.

    Returns
    -------
    pd.DataFrame
        Rows are features. Columns are timepoints. Values are feature scores.
    """
    timepoint_order = validate_score_stability_inputs(
        ranking_by_timepoint,
        timepoints=timepoints,
        score_col=score_col,
    )

    score_series_by_timepoint: Dict[str, pd.Series] = {}

    for tp in timepoint_order:
        # Extract one feature -> score series per timepoint.
        score_series_by_timepoint[tp] = extract_feature_score_series(
            ranking_by_timepoint[tp],
            score_col=score_col,
        )

    # Concatenate all timepoint score series into a feature x timepoint matrix.
    score_matrix = pd.concat(score_series_by_timepoint, axis=1)

    # Reorder columns explicitly to match the requested timepoint order.
    score_matrix = score_matrix.loc[:, timepoint_order]

    if missing_policy == "intersection":
        # Keep only features with scores at all timepoints.
        score_matrix = score_matrix.dropna(axis=0, how="any")

    elif missing_policy == "union_fill_zero":
        # Treat missing features as score 0.
        # This is often appropriate for mean_normalized_rank but should be
        # stated clearly in reporting.
        score_matrix = score_matrix.fillna(0.0)

    elif missing_policy == "union_keep_na":
        # Keep NaNs as explicit missingness.
        # Useful for inspection and per-feature summaries.
        pass

    else:
        raise ValueError(
            "missing_policy must be one of: "
            "'intersection', 'union_fill_zero', 'union_keep_na'."
        )

    return score_matrix


# ============================================================
# C3. Per-feature score stability summaries
# ============================================================

def summarize_per_feature_score_stability(
    feature_score_matrix: pd.DataFrame,
    *,
    timepoints: Optional[Sequence[str]] = None,
    score_col: str = "mean_normalized_rank",
    coefficient_variation_epsilon: float = 1e-12,
) -> pd.DataFrame:
    """
    Compute per-feature score-stability summaries across timepoints.

    Parameters
    ----------
    feature_score_matrix : pd.DataFrame
        Feature x timepoint matrix from `build_feature_score_matrix(...)`.

    timepoints : Optional[Sequence[str]], default=None
        Timepoint columns to summarize. If None, all columns are used.

    score_col : str, default="mean_normalized_rank"
        Name of the score being summarized. Used only for metadata.

    coefficient_variation_epsilon : float, default=1e-12
        Small value used to avoid division by zero when computing coefficient
        of variation.

    Returns
    -------
    pd.DataFrame
        One row per feature with score-stability summaries.

        Columns include:
            - feature
            - score_<timepoint> for each timepoint
            - n_timepoints_observed
            - mean_score
            - std_score
            - variance_score
            - min_score
            - max_score
            - score_range
            - coefficient_of_variation
            - max_abs_step_change
            - first_to_last_change
            - abs_first_to_last_change
            - score_col
    """
    if feature_score_matrix.empty:
        raise ValueError("feature_score_matrix is empty.")

    if timepoints is None:
        timepoint_order = list(feature_score_matrix.columns)
    else:
        timepoint_order = list(timepoints)

    missing_cols = [
        tp for tp in timepoint_order
        if tp not in feature_score_matrix.columns
    ]
    if missing_cols:
        raise KeyError(
            f"These timepoints are missing from feature_score_matrix: {missing_cols}"
        )

    # Work on a copy in the requested timepoint order.
    matrix = feature_score_matrix.loc[:, timepoint_order].copy()

    rows = []

    for feature, row in matrix.iterrows():
        # Convert the row to numeric values.
        values = row.to_numpy(dtype=float)

        # Count how many non-missing scores are available for this feature.
        observed_mask = np.isfinite(values)
        observed_values = values[observed_mask]

        if len(observed_values) == 0:
            # This should be rare, but keeps the function safe under union_keep_na.
            mean_score = np.nan
            std_score = np.nan
            variance_score = np.nan
            min_score = np.nan
            max_score = np.nan
            score_range = np.nan
            coefficient_of_variation = np.nan
        else:
            mean_score = float(np.nanmean(values))
            std_score = float(np.nanstd(values, ddof=1)) if len(observed_values) > 1 else 0.0
            variance_score = float(np.nanvar(values, ddof=1)) if len(observed_values) > 1 else 0.0
            min_score = float(np.nanmin(values))
            max_score = float(np.nanmax(values))
            score_range = max_score - min_score

            # CV = std / mean.
            # Use abs(mean) in denominator because scores are usually non-negative,
            # but this keeps the function more general.
            denominator = max(abs(mean_score), coefficient_variation_epsilon)
            coefficient_of_variation = std_score / denominator

        # Step changes are adjacent timepoint changes:
        # baseline -> week6, week6 -> month6, etc.
        step_changes = np.diff(values)

        if np.all(~np.isfinite(step_changes)):
            max_abs_step_change = np.nan
        else:
            max_abs_step_change = float(np.nanmax(np.abs(step_changes)))

        # First-to-last change summarizes the total directional change over time.
        first_value = values[0]
        last_value = values[-1]

        if np.isfinite(first_value) and np.isfinite(last_value):
            first_to_last_change = float(last_value - first_value)
            abs_first_to_last_change = abs(first_to_last_change)
        else:
            first_to_last_change = np.nan
            abs_first_to_last_change = np.nan

        out_row = {
            "feature": str(feature),
            "n_timepoints_observed": int(np.sum(observed_mask)),
            "n_timepoints_total": len(timepoint_order),
            "mean_score": mean_score,
            "std_score": std_score,
            "variance_score": variance_score,
            "min_score": min_score,
            "max_score": max_score,
            "score_range": score_range,
            "coefficient_of_variation": coefficient_of_variation,
            "max_abs_step_change": max_abs_step_change,
            "first_to_last_change": first_to_last_change,
            "abs_first_to_last_change": abs_first_to_last_change,
            "score_col": score_col,
        }

        # Add raw score columns with readable names.
        for tp, value in zip(timepoint_order, values):
            out_row[f"score_{tp}"] = value

        rows.append(out_row)

    out = pd.DataFrame(rows)

    # Sort by most score-unstable features first.
    out = out.sort_values(
        by=["score_range", "std_score", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return out


# ============================================================
# C4. ICC-style feature-score stability
# ============================================================

def interpret_icc_value(icc: float) -> str:
    """
    Interpret an ICC value using a common rough reliability guide.

    Parameters
    ----------
    icc : float
        ICC value.

    Returns
    -------
    str
        Interpretation label.
    """
    if pd.isna(icc):
        return "not_available"

    if icc < 0.50:
        return "poor_stability"

    if icc < 0.75:
        return "moderate_stability"

    if icc < 0.90:
        return "good_stability"

    return "excellent_stability"


def compute_icc_score_stability(
    feature_score_matrix: pd.DataFrame,
    *,
    score_col: str = "mean_normalized_rank",
    require_complete_rows: bool = True,
) -> pd.DataFrame:
    """
    Compute ICC-style agreement and consistency for feature scores.

    Parameters
    ----------
    feature_score_matrix : pd.DataFrame
        Feature x timepoint score matrix.

    score_col : str, default="mean_normalized_rank"
        Name of the score used. Stored as metadata in the output.

    require_complete_rows : bool, default=True
        If True, drop features with missing scores before computing ICC.

        ICC formulas require a complete feature x timepoint matrix.
        Therefore, if `feature_score_matrix` contains NaNs, rows with NaNs
        are dropped when this is True.

    Returns
    -------
    pd.DataFrame
        ICC-style summary with two rows:

            - agreement
            - consistency

        Columns include:
            - icc_type
            - icc
            - interpretation
            - n_features
            - n_timepoints
            - score_col

    Notes
    -----
    This is an ICC-style descriptive reliability summary where:

        rows = features
        columns = timepoints

    Agreement ICC penalizes systematic shifts in scores across timepoints.
    Consistency ICC focuses more on whether relative feature differences are preserved.

    With only three timepoints, treat this as descriptive rather than definitive.
    """
    if feature_score_matrix.empty:
        raise ValueError("feature_score_matrix is empty.")

    matrix = feature_score_matrix.copy()

    if require_complete_rows:
        # ICC requires complete observations across all timepoints.
        # Drop features missing any score.
        matrix = matrix.dropna(axis=0, how="any")

    if matrix.isna().any().any():
        raise ValueError(
            "feature_score_matrix contains NaNs. "
            "Use require_complete_rows=True or fill missing values before ICC."
        )

    X = matrix.to_numpy(dtype=float)

    if X.ndim != 2:
        raise ValueError("feature_score_matrix must be 2D.")

    n_features, n_timepoints = X.shape

    if n_features < 2:
        raise ValueError("Need at least 2 features to compute ICC.")

    if n_timepoints < 2:
        raise ValueError("Need at least 2 timepoints to compute ICC.")

    # Grand mean across all feature-timepoint values.
    grand_mean = float(np.mean(X))

    # Mean score per feature.
    feature_means = np.mean(X, axis=1)

    # Mean score per timepoint.
    timepoint_means = np.mean(X, axis=0)

    # Sum of squares for feature effects.
    # This captures how much features differ from one another on average.
    ss_feature = n_timepoints * np.sum((feature_means - grand_mean) ** 2)

    # Sum of squares for timepoint effects.
    # This captures systematic shifts in scores between timepoints.
    ss_timepoint = n_features * np.sum((timepoint_means - grand_mean) ** 2)

    # Residual/error sum of squares after removing feature and timepoint means.
    residual = (
        X
        - feature_means[:, None]
        - timepoint_means[None, :]
        + grand_mean
    )
    ss_error = np.sum(residual ** 2)

    # Degrees of freedom.
    df_feature = n_features - 1
    df_timepoint = n_timepoints - 1
    df_error = df_feature * df_timepoint

    # Mean squares.
    ms_feature = ss_feature / df_feature
    ms_timepoint = ss_timepoint / df_timepoint
    ms_error = ss_error / df_error

    # ICC agreement:
    # Penalizes systematic timepoint shifts.
    icc_agreement = (
        (ms_feature - ms_error)
        / (
            ms_feature
            + (n_timepoints - 1) * ms_error
            + (n_timepoints * (ms_timepoint - ms_error) / n_features)
        )
    )

    # ICC consistency:
    # Does not penalize timepoint mean shifts in the same way.
    # Focuses more on whether relative feature differences are preserved.
    icc_consistency = (
        (ms_feature - ms_error)
        / (ms_feature + (n_timepoints - 1) * ms_error)
    )

    out = pd.DataFrame(
        [
            {
                "icc_type": "agreement",
                "icc": float(icc_agreement),
                "interpretation": interpret_icc_value(float(icc_agreement)),
                "n_features": n_features,
                "n_timepoints": n_timepoints,
                "score_col": score_col,
                "require_complete_rows": require_complete_rows,
            },
            {
                "icc_type": "consistency",
                "icc": float(icc_consistency),
                "interpretation": interpret_icc_value(float(icc_consistency)),
                "n_features": n_features,
                "n_timepoints": n_timepoints,
                "score_col": score_col,
                "require_complete_rows": require_complete_rows,
            },
        ]
    )

    return out


# ============================================================
# C5. Full score-stability wrapper
# ============================================================

def summarize_score_stability(
    ranking_by_timepoint: Mapping[str, pd.DataFrame],
    *,
    timepoints: Optional[Sequence[str]] = None,
    score_col: str = "mean_normalized_rank",
    missing_policy: ScoreMissingPolicy = "intersection",
) -> Dict[str, pd.DataFrame]:
    """
    Run the full Section C score-stability analysis.

    Parameters
    ----------
    ranking_by_timepoint : Mapping[str, pd.DataFrame]
        Dictionary mapping timepoint name to final ranking dataframe.

    timepoints : Optional[Sequence[str]], default=None
        Explicit timepoint order. If None, insertion order is used.

    score_col : str, default="mean_normalized_rank"
        Score column to summarize.

    missing_policy : {"intersection", "union_fill_zero", "union_keep_na"},
        default="intersection"
        How to handle missing features across timepoints.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing:

            "feature_score_matrix"
                Feature x timepoint score matrix.

            "per_feature_score_stability"
                Feature-level score variance/range/CV table.

            "icc_score_stability"
                ICC-style global score-stability summary.
    """
    # Build the core score matrix.
    feature_score_matrix = build_feature_score_matrix(
        ranking_by_timepoint,
        timepoints=timepoints,
        score_col=score_col,
        missing_policy=missing_policy,
    )

    # Summarize score variability per feature.
    per_feature_score_stability = summarize_per_feature_score_stability(
        feature_score_matrix,
        timepoints=timepoints,
        score_col=score_col,
    )

    # ICC requires complete data.
    # If missing_policy="union_keep_na", rows with missing scores are dropped here.
    icc_score_stability = compute_icc_score_stability(
        feature_score_matrix,
        score_col=score_col,
        require_complete_rows=True,
    )

    return {
        "feature_score_matrix": feature_score_matrix,
        "per_feature_score_stability": per_feature_score_stability,
        "icc_score_stability": icc_score_stability,
    }





# ============================================================
# Plot section: Feature stability visualization
# ============================================================
#
# These plotting functions assume you have already computed:
#
# Section A:
#   selected_set_results = summarize_selected_set_stability(...)
#   feature_presence = selected_set_results["feature_presence"]
#   selected_overlap_summary = selected_set_results["pairwise_overlap"]
#
# Section B:
#   ranking_stability_results = summarize_ranking_stability(...)
#   ranking_correlation_summary = ranking_stability_results["ranking_correlation_summary"]
#
# Section C:
#   score_stability_results = summarize_score_stability(...)
#   feature_score_matrix = score_stability_results["feature_score_matrix"]
#   per_feature_score_stability = score_stability_results["per_feature_score_stability"]
#   icc_score_stability = score_stability_results["icc_score_stability"]
#
# Default plotting rule:
#   Only plot features selected in at least 2 timepoints.
#
# This keeps plots readable and focused on repeated signals.


# ============================================================
# Shared plotting style helper
# ============================================================



def _apply_feature_stability_axis_style(
    ax: plt.Axes,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    font_size: float = 12.0,
    x_tick_rotation: int = 0,
    legend_loc: Optional[str] = None,
) -> None:
    """
    Apply consistent axis styling for feature-stability plots.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis to style.

    title : str
        Axis title.

    xlabel : str
        X-axis label.

    ylabel : str
        Y-axis label.

    font_size : float, default=12.0
        Base font size.

    x_tick_rotation : int, default=0
        Rotation angle for x-axis tick labels.

    legend_loc : Optional[str], default=None
        If provided, show legend at this location.
    """
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

    ax.tick_params(axis="both", labelsize=font_size)
    ax.tick_params(axis="x", rotation=x_tick_rotation)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    if legend_loc is not None:
        ax.legend(
            title="",
            loc=legend_loc,
            prop={"size": font_size, "weight": "bold"},
        )


def _validate_palette(
    palette: Optional[Mapping[str, str]],
    *,
    required_keys: Sequence[str],
    palette_name: str = "palette",
) -> Mapping[str, str]:
    """
    Validate that a palette contains required keys.

    Parameters
    ----------
    palette : Optional[Mapping[str, str]]
        User-provided color mapping.

    required_keys : Sequence[str]
        Required palette keys.

    palette_name : str, default="palette"
        Name used in error messages.

    Returns
    -------
    Mapping[str, str]
        Validated palette.
    """
    if palette is None:
        return {}

    missing = [
        key for key in required_keys
        if key not in palette
    ]

    if missing:
        raise ValueError(
            f"{palette_name} must contain keys {list(required_keys)}. "
            f"Missing: {missing}. Got keys: {list(palette.keys())}"
        )

    return palette


# ============================================================
# P0. Shared plot filtering helper
# ============================================================

def filter_features_for_plot(
    feature_presence: pd.DataFrame,
    *,
    min_timepoints_selected: int = 2,
    features: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Select features to include in plots based on timepoint-selection frequency.

    Parameters
    ----------
    feature_presence : pd.DataFrame
        Output from `build_feature_presence_map(...)` or
        `selected_set_results["feature_presence"]`.

        Must contain:
            - feature
            - n_timepoints_selected

    min_timepoints_selected : int, default=2
        Minimum number of timepoints where a feature must be selected to appear
        in plots.

        Example:
            2 -> plot features selected in at least 2 timepoints.
            3 -> plot only features selected in all 3 timepoints.

    features : Optional[Sequence[str]], default=None
        Optional manual feature list. If provided, the final plot list is the
        intersection of this list and the min-timepoint filter.

    Returns
    -------
    List[str]
        Filtered feature names to use in plots.
    """
    required_cols = {"feature", "n_timepoints_selected"}
    missing_cols = sorted(required_cols - set(feature_presence.columns))

    if missing_cols:
        raise KeyError(
            f"feature_presence is missing required columns: {missing_cols}"
        )

    if min_timepoints_selected < 1:
        raise ValueError("min_timepoints_selected must be >= 1.")

    df = feature_presence.copy()

    # Keep only features selected in at least the requested number of timepoints.
    df = df[df["n_timepoints_selected"] >= min_timepoints_selected]

    # If the user provided a manual feature list, intersect with it.
    if features is not None:
        requested_features = set(map(str, features))
        df = df[df["feature"].astype(str).isin(requested_features)]

    # Keep the order already defined in feature_presence.
    # Usually this is sorted by n_timepoints_selected and feature name.
    plot_features = df["feature"].astype(str).tolist()

    return plot_features


# ============================================================
# P1. Feature presence heatmap
# ============================================================

def plot_feature_presence_heatmap(
    feature_presence: pd.DataFrame,
    *,
    timepoints: Sequence[str] = ("baseline", "week6", "month6"),
    min_timepoints_selected: int = 2,
    features: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    font_size: float = 12.0,
    x_tick_rotation: float = 35,
    cmap_name: str = "Blues",
    show_colorbar: bool = True,
    colorbar_label: str = "Selection status",
    colorbar_ticklabels: Tuple[str, str] = ("No", "Yes"),
    show_row_separators: bool = True,
    row_separator_color: str = "white",
    row_separator_linewidth: float = 0.8,
    show_selection_count_in_ylabel: bool = True,
    sort_rows: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a clean binary heatmap showing whether each feature was selected
    at each timepoint.

    Parameters
    ----------
    feature_presence : pd.DataFrame
        Output from Section A feature-presence analysis.

        Expected columns:
            - feature
            - n_timepoints_selected
            - selected_<timepoint> for each timepoint

    timepoints : Sequence[str], default=("baseline", "week6", "month6")
        Timepoint order to show on the x-axis.

    min_timepoints_selected : int, default=2
        Only plot features selected in at least this many timepoints.

    features : Optional[Sequence[str]], default=None
        Optional manual feature list.

    title : Optional[str], default=None
        Plot title.

    figsize : Optional[Tuple[float, float]], default=None
        Optional matplotlib figure size.

    font_size : float, default=12.0
        Base font size.

    x_tick_rotation : float, default=35
        Rotation angle for x-axis timepoint labels.

    cmap_name : str, default="Blues"
        Matplotlib colormap name used to encode selection status.

    show_colorbar : bool, default=True
        If True, show a colorbar indicating No/Yes selection.

    colorbar_label : str, default="Selection status"
        Label for the colorbar.

    colorbar_ticklabels : Tuple[str, str], default=("No", "Yes")
        Labels for binary colorbar ticks at 0 and 1.

    show_row_separators : bool, default=True
        If True, draw horizontal separators between feature rows.

        Vertical separators are intentionally not drawn because they can make
        each timepoint column appear split into multiple sections.

    row_separator_color : str, default="white"
        Color for horizontal row separators.

    row_separator_linewidth : float, default=0.8
        Line width for horizontal row separators.

    show_selection_count_in_ylabel : bool, default=True
        If True, append selected count to feature labels, e.g. feature_1 (3/3).

    sort_rows : bool, default=True
        If True, sort rows by:
            1. number of selected timepoints descending
            2. binary presence pattern
            3. feature name

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axis.
    """
    # ------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------
    selected_cols = [f"selected_{tp}" for tp in timepoints]

    required_cols = {"feature", "n_timepoints_selected", *selected_cols}
    missing_cols = sorted(required_cols - set(feature_presence.columns))

    if missing_cols:
        raise KeyError(
            f"feature_presence is missing required columns: {missing_cols}"
        )

    if min_timepoints_selected < 1:
        raise ValueError("min_timepoints_selected must be >= 1.")

    # ------------------------------------------------------------
    # Filter features
    # ------------------------------------------------------------
    plot_features = filter_features_for_plot(
        feature_presence,
        min_timepoints_selected=min_timepoints_selected,
        features=features,
    )

    if len(plot_features) == 0:
        raise ValueError(
            "No features passed the plot filter. "
            "Try lowering min_timepoints_selected."
        )

    df = feature_presence.copy()
    df["feature"] = df["feature"].astype(str)

    df = df[df["feature"].isin(plot_features)].copy()

    if df.empty:
        raise ValueError(
            "Filtered features were not found in feature_presence."
        )

    # ------------------------------------------------------------
    # Sort rows for readability
    # ------------------------------------------------------------
    if sort_rows:
        df["_pattern"] = (
            df[selected_cols]
            .astype(int)
            .astype(str)
            .agg("".join, axis=1)
        )

        df = df.sort_values(
            by=["n_timepoints_selected", "_pattern", "feature"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Build binary matrix
    # ------------------------------------------------------------
    matrix = df[selected_cols].astype(bool).astype(int).to_numpy()

    if show_selection_count_in_ylabel:
        y_labels = [
            f"{row['feature']} ({int(row['n_timepoints_selected'])}/{len(timepoints)})"
            for _, row in df.iterrows()
        ]
    else:
        y_labels = df["feature"].astype(str).tolist()

    n_features = len(y_labels)
    n_timepoints = len(timepoints)

    # ------------------------------------------------------------
    # Figure sizing
    # ------------------------------------------------------------
    if figsize is None:
        figsize = (
            max(6.5, 1.8 * n_timepoints + 2.5),
            max(3.5, 0.42 * n_features + 1.5),
        )

    fig, ax = plt.subplots(figsize=figsize)

    # ------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------
    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError as exc:
        raise ValueError(
            f"Invalid cmap_name={cmap_name!r}. "
            "Use a valid matplotlib colormap name such as "
            "'Blues', 'Purples', 'Reds', or 'Greens'."
        ) from exc

    im = ax.imshow(
        matrix,
        cmap=cmap,
        norm=Normalize(vmin=0, vmax=1),
        aspect="auto",
        interpolation="nearest",
    )

    # ------------------------------------------------------------
    # Axis ticks and labels
    # ------------------------------------------------------------
    ax.set_xticks(np.arange(n_timepoints))
    ax.set_xticklabels(
        timepoints,
        rotation=x_tick_rotation,
        ha="right" if x_tick_rotation != 0 else "center",
    )

    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(y_labels)

    if title is None:
        title = (
            "Selected Feature Presence Across Timepoints "
            f"(min selected timepoints = {min_timepoints_selected})"
        )

    _apply_feature_stability_axis_style(
        ax,
        title=title,
        xlabel="Timepoint",
        ylabel="Feature",
        font_size=font_size,
        x_tick_rotation=x_tick_rotation,
    )

    # Re-apply horizontal alignment after style helper.
    x_tick_ha = "right" if x_tick_rotation != 0 else "center"
    for label in ax.get_xticklabels():
        label.set_ha(x_tick_ha)

    # ------------------------------------------------------------
    # Remove vertical gridlines / column split appearance
    # ------------------------------------------------------------
    ax.grid(False)

    # Optional horizontal row separators only.
    # These help separate features without visually splitting timepoint columns.
    if show_row_separators:
        for y in np.arange(0.5, n_features, 1.0):
            ax.axhline(
                y,
                color=row_separator_color,
                linewidth=row_separator_linewidth,
            )

    # Remove spines for a cleaner heatmap.
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ------------------------------------------------------------
    # Optional colorbar
    # ------------------------------------------------------------
    if show_colorbar:
        cbar = fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )

        cbar.set_label(
            colorbar_label,
            fontsize=font_size,
            fontweight="bold",
        )

        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(list(colorbar_ticklabels))

        cbar.ax.tick_params(labelsize=font_size - 1)

        for label in cbar.ax.get_yticklabels():
            label.set_fontweight("bold")

    fig.tight_layout()

    return fig, ax


# ============================================================
# Plot 2: Pairwise selected-feature overlap
# ============================================================

def plot_pairwise_selected_overlap(
    selected_overlap_summary: pd.DataFrame,
    *,
    metric_col: str = "jaccard",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 4.5),
    font_size: float = 12.0,
    x_tick_rotation: int = 30,
    bar_color: str = "darkblue",
    annotate: bool = True,
    annotate_decimals: int = 2,
    annotate_font_size: Optional[float] = None,
    annotate_offset: float = 0.02,
    ylim: Optional[Tuple[float, float]] = (0.0, 1.05),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot pairwise selected-feature overlap across timepoints.

    Parameters
    ----------
    selected_overlap_summary : pd.DataFrame
        Output from Section A pairwise overlap analysis.

    metric_col : str, default="jaccard"
        Metric to plot on the y-axis.
        Common options:
            - "jaccard"
            - "overlap_fraction"

    title : Optional[str], default=None
        Plot title.

    figsize : Tuple[float, float], default=(8.0, 4.5)
        Matplotlib figure size.

    font_size : float, default=12.0
        Base font size for labels, title, and ticks.

    x_tick_rotation : int, default=30
        Rotation angle for x-axis tick labels.

    bar_color : str, default="darkblue"
        Bar color.

    annotate : bool, default=True
        If True, annotate each bar with metric value and overlap count.

    annotate_decimals : int, default=2
        Number of decimals for metric-value annotation.

    annotate_font_size : Optional[float], default=None
        Annotation font size. If None, uses max(8, font_size - 3).

    annotate_offset : float, default=0.02
        Vertical offset above each bar for annotation.

    ylim : Optional[Tuple[float, float]], default=(0.0, 1.05)
        Optional y-axis limits.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axis.
    """
    df = selected_overlap_summary.copy()

    required_cols = {"comparison", metric_col, "n_overlap", "n_union"}
    missing_cols = sorted(required_cols - set(df.columns))

    if missing_cols:
        raise KeyError(
            f"selected_overlap_summary is missing required columns: {missing_cols}"
        )

    # Sort from highest to lowest overlap for easier reading.
    df = df.sort_values(metric_col, ascending=False).reset_index(drop=True)

    if title is None:
        title = f"Pairwise Selected-Feature Overlap ({metric_col})"

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        df["comparison"],
        df[metric_col],
        color=bar_color,
    )

    if ylim is not None:
        ax.set_ylim(*ylim)

    _apply_feature_stability_axis_style(
        ax,
        title=title,
        xlabel="Comparison",
        ylabel=metric_col,
        font_size=font_size,
        x_tick_rotation=x_tick_rotation,
    )

    if annotate:
        ann_fs = annotate_font_size if annotate_font_size is not None else max(8, font_size - 3)

        for bar, (_, row) in zip(bars, df.iterrows()):
            metric_value = float(row[metric_col])
            label = (
                f"{metric_value:.{annotate_decimals}f}\n"
                f"{int(row['n_overlap'])} / {int(row['n_union'])}"
            )

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                metric_value + annotate_offset,
                label,
                ha="center",
                va="bottom",
                fontsize=ann_fs,
                fontweight="bold",
            )

    fig.tight_layout()

    return fig, ax

# ============================================================
# P3. Pairwise Spearman/Kendall ranking correlations
# ============================================================

def plot_pairwise_ranking_correlations(
    ranking_correlation_summary: pd.DataFrame,
    *,
    title: str = "Pairwise Ranking Correlations Across Timepoints",
    figsize: Tuple[float, float] = (8.5, 4.5),
    font_size: float = 12.0,
    x_tick_rotation: float = 30,
    x_tick_ha: Optional[str] = None,
    spearman_color: str = "darkblue",
    kendall_color: str = "darkred",
    legend_loc: str = "best",
    annotate: bool = True,
    annotate_decimals: int = 2,
    annotate_font_size: Optional[float] = None,
    annotate_offset: float = 0.03,
    ylim: Optional[Tuple[float, float]] = None,
    show_zero_line: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot pairwise Spearman and Kendall ranking correlations.

    Parameters
    ----------
    ranking_correlation_summary : pd.DataFrame
        Output from Section B ranking-stability summary.

        Expected columns:
            - comparison
            - spearman_corr
            - kendall_corr

    title : str, default="Pairwise Ranking Correlations Across Timepoints"
        Plot title.

    figsize : Tuple[float, float], default=(8.5, 4.5)
        Matplotlib figure size.

    font_size : float, default=12.0
        Base font size for title, labels, ticks, legend, and annotations.

    x_tick_rotation : float, default=30
        Rotation angle for x-axis tick labels.

    x_tick_ha : Optional[str], default=None
        Horizontal alignment for x-axis tick labels.

        If None:
            - "center" when x_tick_rotation == 0
            - "right" otherwise

    spearman_color : str, default="darkblue"
        Bar color for Spearman correlation.

    kendall_color : str, default="darkred"
        Bar color for Kendall correlation.

    legend_loc : str, default="best"
        Legend location.

    annotate : bool, default=True
        If True, annotate bars with numeric correlation values.

    annotate_decimals : int, default=2
        Number of decimals for bar annotations.

    annotate_font_size : Optional[float], default=None
        Annotation font size. If None, uses max(8, font_size - 3).

    annotate_offset : float, default=0.03
        Vertical offset for annotations.

    ylim : Optional[Tuple[float, float]], default=None
        Optional y-axis limits. If None, matplotlib chooses limits automatically.

    show_zero_line : bool, default=True
        If True, draw a horizontal reference line at correlation = 0.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axis.
    """
    df = ranking_correlation_summary.copy()

    required_cols = {"comparison", "spearman_corr", "kendall_corr"}
    missing_cols = sorted(required_cols - set(df.columns))

    if missing_cols:
        raise KeyError(
            f"ranking_correlation_summary is missing columns: {missing_cols}"
        )

    # Auto-align tick labels:
    # If labels are not rotated, center them under the grouped bars.
    # If labels are rotated, right-align them for readability.
    if x_tick_ha is None:
        x_tick_ha = "center" if float(x_tick_rotation) == 0 else "right"

    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    spearman_bars = ax.bar(
        x - width / 2,
        df["spearman_corr"],
        width,
        label="Spearman",
        color=spearman_color,
    )

    kendall_bars = ax.bar(
        x + width / 2,
        df["kendall_corr"],
        width,
        label="Kendall",
        color=kendall_color,
    )

    if ylim is not None:
        ax.set_ylim(*ylim)

    if show_zero_line:
        ax.axhline(0.0, linewidth=1.2, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(
        df["comparison"].astype(str),
        rotation=x_tick_rotation,
        ha=x_tick_ha,
    )

    _apply_feature_stability_axis_style(
        ax,
        title=title,
        xlabel="Comparison",
        ylabel="Correlation",
        font_size=font_size,
        x_tick_rotation=x_tick_rotation,
        legend_loc=legend_loc,
    )

    # Re-apply horizontal alignment after the shared style helper,
    # because tick_params can modify tick appearance.
    for label in ax.get_xticklabels():
        label.set_ha(x_tick_ha)

    if annotate:
        ann_fs = (
            annotate_font_size
            if annotate_font_size is not None
            else max(8, font_size - 3)
        )

        for bar, value in zip(spearman_bars, df["spearman_corr"].to_numpy(dtype=float)):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + annotate_offset if value >= 0 else value - annotate_offset,
                f"{value:.{annotate_decimals}f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=ann_fs,
                fontweight="bold",
            )

        for bar, value in zip(kendall_bars, df["kendall_corr"].to_numpy(dtype=float)):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + annotate_offset if value >= 0 else value - annotate_offset,
                f"{value:.{annotate_decimals}f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=ann_fs,
                fontweight="bold",
            )

    fig.tight_layout()

    return fig, ax


# ============================================================
# P4. Feature score trajectories
# ============================================================

def plot_feature_rank_trajectories(
    ranking_by_timepoint: Mapping[str, pd.DataFrame],
    feature_presence: pd.DataFrame,
    *,
    timepoints: Sequence[str] = ("baseline", "week6", "month6"),
    min_timepoints_selected: int = 2,
    features: Optional[Sequence[str]] = None,
    top_n: Optional[int] = 8,
    sort_by: str = "mean_rank",
    score_col: str = "mean_normalized_rank",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (9.5, 5.0),
    font_size: float = 12.0,
    line_palette: str = "tab10",
    ylabel: str = "Rank among selected features (1 = best)",
    label_lines: bool = False,
    show_legend: bool = True,
    legend_loc: str = "center left",
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = (1.02, 0.5),
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """
    Plot feature rank trajectories across timepoints.

    This plot shows where repeatedly selected features rank within each
    timepoint's selected-feature ranking table.

    Parameters
    ----------
    ranking_by_timepoint : Mapping[str, pd.DataFrame]
        Dictionary mapping timepoint name to ranking dataframe.

        Each dataframe must contain:
            - feature
            - score_col

        Recommended input:
            stable_ranking_by_timepoint

        This ensures the plot is focused on final selected top-k features,
        optionally filtered to features passing the selected-frequency threshold.

    feature_presence : pd.DataFrame
        Feature-presence table from Section A.

        Expected columns:
            - feature
            - n_timepoints_selected

    timepoints : Sequence[str], default=("baseline", "week6", "month6")
        Timepoint order to plot on the x-axis.

    min_timepoints_selected : int, default=2
        Only include features selected in at least this many timepoints.

    features : Optional[Sequence[str]], default=None
        Optional manual feature list.

        If provided, the plotted features are the intersection of:
            - features selected in at least `min_timepoints_selected`
            - this manual list

    top_n : Optional[int], default=8
        If provided, plot only the top N features after filtering and sorting.

        If None, plot all filtered features.

    sort_by : str, default="mean_rank"
        Rule for selecting/sorting plotted features.

        Options:
            - "mean_rank": features with best average rank
            - "rank_range": features with largest rank movement
            - "n_timepoints_selected": features selected most often
            - "feature": alphabetical

    score_col : str, default="mean_normalized_rank"
        Score column used to compute ranks within each timepoint.

        Higher score is treated as better, so rank 1 is assigned to the
        highest score.

    title : Optional[str], default=None
        Plot title.

    figsize : Tuple[float, float], default=(9.5, 5.0)
        Matplotlib figure size.

    font_size : float, default=12.0
        Base font size.

    line_palette : str, default="tab10"
        Matplotlib colormap name used for feature lines.

    ylabel : str, default="Rank among selected features (1 = best)"
        Y-axis label.

    label_lines : bool, default=False
        If True, label each line at its final timepoint.

    show_legend : bool, default=True
        If True, show legend.

    legend_loc : str, default="center left"
        Legend location.

    legend_bbox_to_anchor : Optional[Tuple[float, float]], default=(1.02, 0.5)
        Optional legend bbox anchor.

    Returns
    -------
    Tuple[pd.DataFrame, plt.Figure, plt.Axes]
        rank_matrix :
            Feature x timepoint rank matrix used for plotting.

        fig, ax :
            Matplotlib figure and axis.

    Notes
    -----
    Rank convention:

        rank 1 = best feature

    The y-axis is inverted so better ranks appear higher on the plot.

    Missing values can occur when a feature passes the repeated-selection filter
    but is not present in every timepoint. Those missing points are left as NaN,
    so matplotlib will break the line at missing timepoints.
    """
    # ------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------
    if not ranking_by_timepoint:
        raise ValueError("ranking_by_timepoint must be a non-empty mapping.")

    timepoint_order = list(timepoints)

    missing_timepoints = [
        tp for tp in timepoint_order
        if tp not in ranking_by_timepoint
    ]

    if missing_timepoints:
        raise KeyError(
            f"These timepoints are missing from ranking_by_timepoint: "
            f"{missing_timepoints}"
        )

    for tp in timepoint_order:
        ranking_df = ranking_by_timepoint[tp]

        required_cols = {"feature", score_col}
        missing_cols = sorted(required_cols - set(ranking_df.columns))

        if missing_cols:
            raise KeyError(
                f"Ranking table for {tp!r} is missing columns: {missing_cols}"
            )

    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be >= 1 when provided.")

    allowed_sort_by = {
        "mean_rank",
        "rank_range",
        "n_timepoints_selected",
        "feature",
    }

    if sort_by not in allowed_sort_by:
        raise ValueError(
            f"sort_by must be one of {sorted(allowed_sort_by)}; got {sort_by!r}."
        )

    # ------------------------------------------------------------
    # Select repeatedly selected features using Section A logic
    # ------------------------------------------------------------
    plot_features = filter_features_for_plot(
        feature_presence,
        min_timepoints_selected=min_timepoints_selected,
        features=features,
    )

    if len(plot_features) == 0:
        raise ValueError(
            "No features passed the plot filter. "
            "Try lowering min_timepoints_selected."
        )

    plot_features = list(map(str, plot_features))

    # ------------------------------------------------------------
    # Build feature x timepoint rank matrix
    # ------------------------------------------------------------
    rank_series_by_timepoint = {}

    for tp in timepoint_order:
        ranking_df = ranking_by_timepoint[tp].copy()
        ranking_df["feature"] = ranking_df["feature"].astype(str)

        ranking_df = ranking_df.sort_values(
            by=[score_col, "feature"],
            ascending=[False, True],
        ).reset_index(drop=True)

        ranking_df["rank"] = ranking_df[score_col].rank(
            ascending=False,
            method="average",
        )

        rank_series_by_timepoint[tp] = (
            ranking_df
            .set_index("feature")["rank"]
            .astype(float)
        )

    rank_matrix = pd.concat(rank_series_by_timepoint, axis=1)
    rank_matrix = rank_matrix.reindex(plot_features)
    rank_matrix = rank_matrix.loc[:, timepoint_order]

    # Drop features that do not appear in any ranking table after alignment.
    rank_matrix = rank_matrix.dropna(axis=0, how="all")

    if rank_matrix.empty:
        raise ValueError(
            "No repeated selected features were found in ranking_by_timepoint."
        )

    # ------------------------------------------------------------
    # Add sorting metadata
    # ------------------------------------------------------------
    summary = pd.DataFrame(index=rank_matrix.index)
    summary["mean_rank"] = rank_matrix.mean(axis=1, skipna=True)
    summary["rank_range"] = (
        rank_matrix.max(axis=1, skipna=True)
        - rank_matrix.min(axis=1, skipna=True)
    )

    presence_lookup = (
        feature_presence
        .copy()
        .assign(feature=lambda df: df["feature"].astype(str))
        .set_index("feature")["n_timepoints_selected"]
    )

    summary["n_timepoints_selected"] = presence_lookup.reindex(rank_matrix.index)

    if sort_by == "mean_rank":
        ordered_features = (
            summary
            .sort_values(by=["mean_rank", "rank_range"], ascending=[True, False])
            .index
            .tolist()
        )

    elif sort_by == "rank_range":
        ordered_features = (
            summary
            .sort_values(by=["rank_range", "mean_rank"], ascending=[False, True])
            .index
            .tolist()
        )

    elif sort_by == "n_timepoints_selected":
        ordered_features = (
            summary
            .sort_values(
                by=["n_timepoints_selected", "mean_rank"],
                ascending=[False, True],
            )
            .index
            .tolist()
        )

    else:
        ordered_features = sorted(rank_matrix.index.astype(str).tolist())

    if top_n is not None:
        ordered_features = ordered_features[:top_n]

    rank_matrix = rank_matrix.loc[ordered_features]

    # ------------------------------------------------------------
    # Plot rank trajectories
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.get_cmap(line_palette)
    colors = [
        cmap(i % cmap.N)
        for i in range(len(rank_matrix))
    ]

    x = np.arange(len(timepoint_order))

    for color, (feature, row) in zip(colors, rank_matrix.iterrows()):
        y = row.to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=feature,
            color=color,
        )

        if label_lines:
            finite_idx = np.where(np.isfinite(y))[0]

            if len(finite_idx) > 0:
                last_idx = finite_idx[-1]

                ax.text(
                    x[last_idx] + 0.03,
                    y[last_idx],
                    str(feature),
                    va="center",
                    fontsize=max(8, font_size - 3),
                    fontweight="bold",
                    color=color,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(timepoint_order)

    ax.invert_yaxis()

    if title is None:
        title = (
            f"Top {len(rank_matrix)} Feature Rank Trajectories "
            f"by {sort_by} "
            f"(min selected timepoints = {min_timepoints_selected})"
        )

    _apply_feature_stability_axis_style(
        ax,
        title=title,
        xlabel="Timepoint",
        ylabel=ylabel,
        font_size=font_size,
        x_tick_rotation=0,
    )

    if show_legend:
        if legend_bbox_to_anchor is None:
            ax.legend(
                loc=legend_loc,
                prop={"size": max(8, font_size - 2), "weight": "bold"},
            )
        else:
            ax.legend(
                loc=legend_loc,
                bbox_to_anchor=legend_bbox_to_anchor,
                prop={"size": max(8, font_size - 2), "weight": "bold"},
            )

    fig.tight_layout()

    return rank_matrix, fig, ax

# ============================================================
# P5. Top score-unstable features across all timepoints
# ============================================================


def plot_repeated_feature_rank_changes(
    rank_score_change_df: pd.DataFrame,
    feature_presence: pd.DataFrame,
    *,
    min_timepoints_selected: int = 2,
    comparison_name: Optional[str] = None,
    features: Optional[Sequence[str]] = None,
    top_n: Optional[int] = None,
    sort_by: str = "abs_rank_change",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (9.0, 5.0),
    font_size: float = 12.0,
    positive_color: str = "darkred",
    negative_color: str = "darkblue",
    zero_color: str = "gray",
    annotate: bool = True,
    annotate_decimals: int = 1,
    annotate_font_size: Optional[float] = None,
    annotate_offset_fraction: float = 0.02,
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """
    Plot signed rank changes for repeatedly selected final top-k features.

    This plot is intended to replace top-pairwise-mover plots when the goal is
    to focus only on features that repeatedly appeared in the final selected
    feature sets from the feature-selection pipeline.

    The filtering logic is:

        1) Start with a pairwise rank/score change table from Section B.
        2) Use the Section A feature-presence table to identify features selected
           in at least `min_timepoints_selected` timepoints.
        3) Optionally intersect with a manually provided `features` list.
        4) Plot signed rank changes only for those repeated selected features.

    Parameters
    ----------
    rank_score_change_df : pd.DataFrame
        Pairwise rank/score change table from Section B.

        Expected columns:
            - feature
            - rank_change
            - abs_rank_change

        Usually one table from:
            ranking_stability_results["pairwise_rank_score_change_tables"][comparison]

        This table may contain more features than the final selected top-k
        features, depending on how the ranking-stability inputs were built.
        This function filters it using `feature_presence`.

    feature_presence : pd.DataFrame
        Feature-presence table from Section A.

        Expected columns:
            - feature
            - n_timepoints_selected

        Usually:
            selected_set_results["feature_presence"]

        This table should be built from the final selected feature sets, such as
        those extracted with:
            get_selected_features(...)

    min_timepoints_selected : int, default=2
        Only include features selected in at least this many timepoints.

        Example with three timepoints:
            3 -> only features selected in all three timepoints
            2 -> features selected in at least two of three timepoints
            1 -> features selected in at least one timepoint

        This makes the plot align with the feature-presence heatmap.

    comparison_name : Optional[str], default=None
        Comparison label used in the plot title, such as:
            - "baseline_vs_week6"
            - "baseline_vs_month6"
            - "week6_vs_month6"

    features : Optional[Sequence[str]], default=None
        Optional manual feature list.

        If provided, the final plotted features are the intersection of:
            - features selected in at least `min_timepoints_selected`
            - this manual feature list

    top_n : Optional[int], default=None
        If provided, show only the top N features after filtering and sorting.

        If None, show all filtered repeated features.

    sort_by : str, default="abs_rank_change"
        Column used to sort features before optional top-N filtering.

        Common options:
            - "abs_rank_change"
            - "rank_change"
            - "abs_score_change"
            - "score_change"

        Columns must exist in `rank_score_change_df`.

    title : Optional[str], default=None
        Plot title.

        If None, a title is generated from `comparison_name` and
        `min_timepoints_selected`.

    figsize : Tuple[float, float], default=(9.0, 5.0)
        Matplotlib figure size.

    font_size : float, default=12.0
        Base font size.

    positive_color : str, default="darkred"
        Color for positive rank changes.

        Positive `rank_change` means the feature moved down at the second
        timepoint, because rank 1 is best.

    negative_color : str, default="darkblue"
        Color for negative rank changes.

        Negative `rank_change` means the feature moved up at the second
        timepoint, because rank 1 is best.

    zero_color : str, default="gray"
        Color for zero rank change.

    annotate : bool, default=True
        If True, annotate bars with signed rank-change values.

    annotate_decimals : int, default=1
        Number of decimal places used for rank-change annotations.

    annotate_font_size : Optional[float], default=None
        Annotation font size.

        If None, uses:
            max(8, font_size - 3)

    annotate_offset_fraction : float, default=0.02
        Annotation offset as a fraction of the x-axis range.

    Returns
    -------
    Tuple[pd.DataFrame, plt.Figure, plt.Axes]
        filtered_rank_changes :
            Rank-change table after repeated-feature filtering, sorting, and
            optional top-N filtering.

        fig, ax :
            Matplotlib figure and axis.

    Notes
    -----
    Rank convention:

        rank 1 = best feature

    Signed rank change is computed upstream as:

        rank_change = rank_timepoint_b - rank_timepoint_a

    Therefore:

        negative rank_change -> feature moved up at timepoint B
        positive rank_change -> feature moved down at timepoint B
        zero rank_change     -> feature stayed at the same rank

    This plot is intended to build from the heatmap:

        first identify repeatedly selected features,
        then inspect how their ranks changed between a pair of timepoints.

    Important
    ---------
    This function does not itself decide what the "final selected top-k" features
    are. It relies on `feature_presence` being built from final selected feature
    sets. Therefore, make sure `feature_presence` comes from:

        selected_by_timepoint = {
            tp: get_selected_features(pipeline_output_by_timepoint[tp], ...)
            for tp in TIMEPOINTS
        }

        selected_set_results = summarize_selected_set_stability(...)
        feature_presence = selected_set_results["feature_presence"]
    """
    # ------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------
    required_rank_cols = {"feature", "rank_change", "abs_rank_change"}
    missing_rank_cols = sorted(required_rank_cols - set(rank_score_change_df.columns))

    if missing_rank_cols:
        raise KeyError(
            f"rank_score_change_df is missing required columns: {missing_rank_cols}"
        )

    if sort_by not in rank_score_change_df.columns:
        raise KeyError(
            f"sort_by={sort_by!r} not found in rank_score_change_df. "
            f"Available columns: {list(rank_score_change_df.columns)}"
        )

    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be >= 1 when provided.")

    # ------------------------------------------------------------
    # Filter to repeatedly selected final top-k features
    # ------------------------------------------------------------
    repeated_features = filter_features_for_plot(
        feature_presence,
        min_timepoints_selected=min_timepoints_selected,
        features=features,
    )

    if len(repeated_features) == 0:
        raise ValueError(
            "No features passed the repeated-feature filter. "
            "Try lowering min_timepoints_selected."
        )

    repeated_features = set(map(str, repeated_features))

    df = rank_score_change_df.copy()
    df["feature"] = df["feature"].astype(str)

    df = df[df["feature"].isin(repeated_features)].copy()

    if df.empty:
        raise ValueError(
            "None of the repeatedly selected features were found in rank_score_change_df."
        )

    # ------------------------------------------------------------
    # Sort and optionally keep top N
    # ------------------------------------------------------------
    ascending = False if sort_by.startswith("abs_") else True

    df = df.sort_values(
        by=[sort_by, "feature"],
        ascending=[ascending, True],
    ).reset_index(drop=True)

    if top_n is not None:
        df = df.head(top_n).copy().reset_index(drop=True)

    # ------------------------------------------------------------
    # Plot signed rank change
    # ------------------------------------------------------------
    plot_df = df.iloc[::-1].copy()

    values = plot_df["rank_change"].to_numpy(dtype=float)

    colors = [
        positive_color if value > 0 else negative_color if value < 0 else zero_color
        for value in values
    ]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        plot_df["feature"],
        values,
        color=colors,
    )

    ax.axvline(0, linewidth=1.5, color="black")

    # Add padding so annotations do not get clipped.
    if len(values) > 0:
        max_abs_value = np.nanmax(np.abs(values))

        if np.isfinite(max_abs_value) and max_abs_value > 0:
            pad = max_abs_value * 0.20
            ax.set_xlim(-max_abs_value - pad, max_abs_value + pad)

    if title is None:
        if comparison_name is None:
            title = (
                "Rank Changes for Repeatedly Selected Features "
                f"(min selected timepoints = {min_timepoints_selected})"
            )
        else:
            title = (
                f"Rank Changes for Repeatedly Selected Features: {comparison_name} "
                f"(min selected timepoints = {min_timepoints_selected})"
            )

    _apply_feature_stability_axis_style(
        ax,
        title=title,
        xlabel="Signed rank change (negative = moved up, positive = moved down)",
        ylabel="Feature",
        font_size=font_size,
    )

    # ------------------------------------------------------------
    # Annotate bars
    # ------------------------------------------------------------
    if annotate:
        ann_fs = (
            annotate_font_size
            if annotate_font_size is not None
            else max(8, font_size - 3)
        )

        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min
        offset = annotate_offset_fraction * x_range

        for bar, value in zip(bars, values):
            y = bar.get_y() + bar.get_height() / 2.0

            if value >= 0:
                x_text = value + offset
                ha = "left"
            else:
                x_text = value - offset
                ha = "right"

            ax.text(
                x_text,
                y,
                f"{value:.{annotate_decimals}f}",
                va="center",
                ha=ha,
                fontsize=ann_fs,
                fontweight="bold",
            )

    fig.tight_layout()

    return df, fig, ax




def top_score_unstable_features_plot_table(
    per_feature_score_stability: pd.DataFrame,
    feature_presence: pd.DataFrame,
    *,
    min_timepoints_selected: int = 2,
    top_n: int = 10,
    instability_col: str = "score_range",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """
    Show the features with the largest score instability across all timepoints.

    Parameters
    ----------
    per_feature_score_stability : pd.DataFrame
        Feature-level score stability table from Section C.

        Expected columns include:
            - feature
            - mean_score
            - score_range
            - std_score
            - first_to_last_change
            - abs_first_to_last_change

    feature_presence : pd.DataFrame
        Feature-presence table from Section A.
        Used to filter features by number of selected timepoints.

    min_timepoints_selected : int, default=2
        Only include features selected in at least this many timepoints.

    top_n : int, default=10
        Number of top unstable features to show.

    instability_col : str, default="score_range"
        Column used to define instability.

        Common options:
            - "score_range"
            - "std_score"
            - "abs_first_to_last_change"
            - "max_abs_step_change"

    title : Optional[str], default=None
        Plot title.

    figsize : Tuple[float, float], default=(8, 5)
        Matplotlib figure size.

    Returns
    -------
    Tuple[pd.DataFrame, plt.Figure, plt.Axes]
        top_features :
            Table of top unstable features.

        fig, ax :
            Matplotlib figure and axis.

    Notes
    -----
    This plot is intended to replace a more abstract mean-score-vs-variability
    scatter plot when the goal is readability.
    """
    required_cols = {
        "feature",
        "mean_score",
        instability_col,
    }
    missing_cols = sorted(required_cols - set(per_feature_score_stability.columns))

    if missing_cols:
        raise KeyError(
            f"per_feature_score_stability is missing columns: {missing_cols}"
        )

    if top_n < 1:
        raise ValueError("top_n must be >= 1.")

    # Reuse the shared plot filter:
    # default is features selected in at least 2 timepoints.
    plot_features = filter_features_for_plot(
        feature_presence,
        min_timepoints_selected=min_timepoints_selected,
    )

    if len(plot_features) == 0:
        raise ValueError(
            "No features passed the plot filter. "
            "Try lowering min_timepoints_selected."
        )

    df = per_feature_score_stability.copy()
    df["feature"] = df["feature"].astype(str)

    # Keep only repeatedly selected features.
    df = df[df["feature"].isin(plot_features)].copy()

    if df.empty:
        raise ValueError(
            "Filtered features were not found in per_feature_score_stability."
        )

    # Sort by largest instability.
    top_features = (
        df.sort_values(
            by=[instability_col, "feature"],
            ascending=[False, True],
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    if title is None:
        title = (
            f"Top {top_n} Score-Unstable Features "
            f"({instability_col}; min selected timepoints = {min_timepoints_selected})"
        )

    fig, ax = plt.subplots(figsize=figsize)

    # Reverse order so the largest mover appears at the top.
    plot_df = top_features.iloc[::-1]

    ax.barh(plot_df["feature"], plot_df[instability_col])

    ax.set_title(title)
    ax.set_xlabel(instability_col)
    ax.set_ylabel("Feature")

    # Add numeric labels at the end of each bar.
    for _, row in plot_df.iterrows():
        ax.text(
            row[instability_col],
            row["feature"],
            f"{row[instability_col]:.3f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    fig.tight_layout()

    return top_features, fig, ax


# ============================================================
# Plot 5: Top pairwise rank/score movers
# ============================================================

def top_pairwise_movers_plot_table(
    rank_score_change_df: pd.DataFrame,
    *,
    comparison_name: Optional[str] = None,
    top_n: int = 10,
    sort_by: str = "abs_rank_change",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (9.0, 5.0),
    font_size: float = 12.0,
    positive_color: str = "darkred",
    negative_color: str = "darkblue",
    zero_color: str = "gray",
    annotate: bool = True,
    annotate_decimals: int = 2,
    annotate_font_size: Optional[float] = None,
    annotate_offset_fraction: float = 0.02,
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """
    Show the features with the largest pairwise rank or score movement.

    Parameters
    ----------
    rank_score_change_df : pd.DataFrame
        One dataframe from Section B:

            ranking_stability_results["pairwise_rank_score_change_tables"][comparison]

        or:

            ranking_stability_results["pairwise_rank_score_change_tables"][comparison]

        Expected columns include:
            - feature
            - rank_change
            - abs_rank_change
            - score_change
            - abs_score_change

    comparison_name : Optional[str], default=None
        Optional comparison label, such as:
            - "baseline_vs_week6"
            - "baseline_vs_month6"
            - "week6_vs_month6"

    top_n : int, default=10
        Number of top movers to show.

    sort_by : str, default="abs_rank_change"
        Column used to define top movers.

        Common options:
            - "abs_rank_change"
            - "abs_score_change"

    title : Optional[str], default=None
        Plot title.

    figsize : Tuple[float, float], default=(9.0, 5.0)
        Matplotlib figure size.

    font_size : float, default=12.0
        Base font size.

    positive_color : str, default="darkred"
        Bar color for positive movement.

    negative_color : str, default="darkblue"
        Bar color for negative movement.

    zero_color : str, default="gray"
        Bar color for zero movement.

    annotate : bool, default=True
        If True, annotate each bar with the signed movement value.

    annotate_decimals : int, default=2
        Number of decimals for score-change annotation.

    annotate_font_size : Optional[float], default=None
        Annotation font size. If None, uses max(8, font_size - 3).

    annotate_offset_fraction : float, default=0.02
        Offset as a fraction of x-axis range for annotation placement.

    Returns
    -------
    Tuple[pd.DataFrame, plt.Figure, plt.Axes]
        top_movers :
            Table of top moving features.

        fig, ax :
            Matplotlib figure and axis.

    Notes
    -----
    Negative rank_change means the feature moved up at the second timepoint.
    Positive rank_change means the feature moved down at the second timepoint.
    """
    required_cols = {
        "feature",
        "rank_change",
        "abs_rank_change",
        "score_change",
        "abs_score_change",
    }
    missing_cols = sorted(required_cols - set(rank_score_change_df.columns))

    if missing_cols:
        raise KeyError(
            f"rank_score_change_df is missing required columns: {missing_cols}"
        )

    if sort_by not in rank_score_change_df.columns:
        raise KeyError(
            f"sort_by={sort_by!r} not found in rank_score_change_df. "
            f"Available columns: {list(rank_score_change_df.columns)}"
        )

    if top_n < 1:
        raise ValueError("top_n must be >= 1.")

    df = rank_score_change_df.copy()
    df["feature"] = df["feature"].astype(str)

    # Select the top movers by the requested absolute movement metric.
    top_movers = (
        df.sort_values(
            by=[sort_by, "feature"],
            ascending=[False, True],
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    # Decide which signed value to plot.
    if sort_by == "abs_rank_change":
        x_col = "rank_change"
        xlabel = "Rank Change"
        value_format = "{:.1f}"
    elif sort_by == "abs_score_change":
        x_col = "score_change"
        xlabel = "Score Change"
        value_format = f"{{:.{annotate_decimals}f}}"
    else:
        x_col = sort_by
        xlabel = sort_by
        value_format = f"{{:.{annotate_decimals}f}}"

    if title is None:
        if comparison_name is None:
            title = f"Top {top_n} Feature Movers ({sort_by})"
        else:
            title = f"Top {top_n} Feature Movers: {comparison_name} ({sort_by})"

    # Reverse order so the largest mover appears at the top.
    plot_df = top_movers.iloc[::-1].copy()

    values = plot_df[x_col].to_numpy(dtype=float)

    colors = [
        positive_color if value > 0 else negative_color if value < 0 else zero_color
        for value in values
    ]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        plot_df["feature"],
        values,
        color=colors,
    )

    ax.axvline(0, linewidth=1.5, color="black")

    _apply_feature_stability_axis_style(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel="Feature",
        font_size=font_size,
    )

    if annotate:
        ann_fs = annotate_font_size if annotate_font_size is not None else max(8, font_size - 3)

        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min
        offset = annotate_offset_fraction * x_range

        for bar, value in zip(bars, values):
            y = bar.get_y() + bar.get_height() / 2.0

            if value >= 0:
                x = value + offset
                ha = "left"
            else:
                x = value - offset
                ha = "right"

            ax.text(
                x,
                y,
                value_format.format(value),
                va="center",
                ha=ha,
                fontsize=ann_fs,
                fontweight="bold",
            )

    fig.tight_layout()

    return top_movers, fig, ax


# ============================================================
# Plot 6: ICC score stability
# ============================================================

def plot_icc_score_stability(
    icc_score_stability: pd.DataFrame,
    *,
    title: str = "ICC Score Stability Across Timepoints",
    figsize: Tuple[float, float] = (7.0, 4.5),
    font_size: float = 12.0,
    bar_color: str = "darkblue",
    annotate: bool = True,
    annotate_decimals: int = 2,
    annotate_font_size: Optional[float] = None,
    annotate_offset: float = 0.03,
    ylim: Optional[Tuple[float, float]] = None,
    show_zero_line: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot ICC agreement and ICC consistency.

    Parameters
    ----------
    icc_score_stability : pd.DataFrame
        Output from Section C ICC score-stability analysis.

        Expected columns:
            - icc_type
            - icc

    title : str, default="ICC Score Stability Across Timepoints"
        Plot title.

    figsize : Tuple[float, float], default=(7.0, 4.5)
        Matplotlib figure size.

    font_size : float, default=12.0
        Base font size.

    bar_color : str, default="darkblue"
        Bar color.

    annotate : bool, default=True
        If True, annotate bars with ICC values.

    annotate_decimals : int, default=2
        Number of decimals for ICC annotation.

    annotate_font_size : Optional[float], default=None
        Annotation font size. If None, uses max(8, font_size - 3).

    annotate_offset : float, default=0.03
        Vertical offset for annotation.

    ylim : Optional[Tuple[float, float]], default=None
        Optional y-axis limits. If None, matplotlib chooses limits automatically.

    show_zero_line : bool, default=True
        If True, draw a horizontal reference line at ICC = 0.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axis.

    Notes
    -----
    This plot intentionally does not include ICC interpretation labels or
    threshold reference lines because those cutoffs are context-dependent.
    """
    df = icc_score_stability.copy()

    required_cols = {"icc_type", "icc"}
    missing_cols = sorted(required_cols - set(df.columns))

    if missing_cols:
        raise KeyError(
            f"icc_score_stability is missing required columns: {missing_cols}"
        )

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        df["icc_type"],
        df["icc"],
        color=bar_color,
    )

    # Let matplotlib choose y-limits unless the user provides them.
    if ylim is not None:
        ax.set_ylim(*ylim)

    # A zero line is objective and helps orient negative/near-zero ICC values.
    if show_zero_line:
        ax.axhline(0.0, linewidth=1.2, color="black")

    _apply_feature_stability_axis_style(
        ax,
        title=title,
        xlabel="ICC type",
        ylabel="ICC",
        font_size=font_size,
    )

    if annotate:
        ann_fs = (
            annotate_font_size
            if annotate_font_size is not None
            else max(8, font_size - 3)
        )

        for bar, (_, row) in zip(bars, df.iterrows()):
            icc_value = float(row["icc"])

            label = f"{icc_value:.{annotate_decimals}f}"

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                icc_value + annotate_offset if icc_value >= 0 else icc_value - annotate_offset,
                label,
                ha="center",
                va="bottom" if icc_value >= 0 else "top",
                fontsize=ann_fs,
                fontweight="bold",
            )

    fig.tight_layout()

    return fig, ax



# ============================================================
# Membership consistency using reference-timepoint clustering
# ============================================================

from typing import Any, Dict, Mapping, Optional, Sequence, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone


# ============================================================
# 1. Validation helpers
# ============================================================

def _validate_timepoints_present(
    mapping: Mapping[str, Any],
    *,
    timepoints: Sequence[str],
    mapping_name: str,
) -> List[str]:
    """Validate that all requested timepoints exist in a mapping."""
    timepoint_order = list(timepoints)

    missing = [tp for tp in timepoint_order if tp not in mapping]
    if missing:
        raise KeyError(
            f"{mapping_name} is missing these requested timepoints: {missing}"
        )

    return timepoint_order


def _validate_clusterer_can_predict(
    clusterer: Any,
    *,
    model_name: str,
) -> None:
    """
    Reference-model membership consistency requires predict(...).

    Supported:
        KMeans and other estimators with fit(...) + predict(...).

    Not supported:
        Standard AgglomerativeClustering, because it does not provide predict(...).
    """
    if not hasattr(clusterer, "fit"):
        raise ValueError(f"Model {model_name!r} must implement fit(...).")

    if not hasattr(clusterer, "predict"):
        raise ValueError(
            f"Model {model_name!r} must implement predict(...). "
            "Reference-model membership consistency requires assigning clusters "
            "to non-reference timepoints using the same fitted model."
        )


# ============================================================
# 2. Reference feature selection
# ============================================================
def get_reference_top_features(
    ranking_by_timepoint: Mapping[str, pd.DataFrame],
    *,
    reference_timepoint: str,
    top_k_features: Optional[int] = None,
    feature_col: str = "feature",
) -> List[str]:
    """
    Get the top-k features from the reference timepoint ranking table.

    If top_k_features=None, all features in the reference ranking table are used.
    """
    if reference_timepoint not in ranking_by_timepoint:
        raise KeyError(
            f"reference_timepoint={reference_timepoint!r} is missing from ranking_by_timepoint."
        )

    ranking_df = ranking_by_timepoint[reference_timepoint]

    if ranking_df.empty:
        raise ValueError(
            f"Ranking table for reference_timepoint={reference_timepoint!r} is empty."
        )

    if feature_col not in ranking_df.columns:
        raise KeyError(
            f"Ranking table for reference_timepoint={reference_timepoint!r} "
            f"is missing feature column {feature_col!r}."
        )

    features = ranking_df[feature_col].astype(str).tolist()

    if top_k_features is not None:
        top_k_features = int(top_k_features)

        if top_k_features < 1:
            raise ValueError("top_k_features must be >= 1 when provided.")

        if top_k_features > len(features):
            raise ValueError(
                f"top_k_features={top_k_features} exceeds number of available "
                f"features for {reference_timepoint!r}: {len(features)}."
            )

        features = features[:top_k_features]

    if len(features) == 0:
        raise ValueError(f"No reference features were selected for {reference_timepoint!r}.")

    if len(set(features)) != len(features):
        duplicated = sorted({f for f in features if features.count(f) > 1})
        raise ValueError(f"Reference features contain duplicates: {duplicated}")

    return features


def build_reference_feature_matrices(
    X_df_by_timepoint: Mapping[str, pd.DataFrame],
    *,
    reference_features: Sequence[str],
    timepoints: Sequence[str],
) -> Dict[str, np.ndarray]:
    """
    For every timepoint, subset columns to the same reference feature list.

    This is the key step:
        baseline-reference model uses baseline top-k features,
        and those exact features are extracted from baseline, week6, and month6.
    """
    timepoint_order = _validate_timepoints_present(
        X_df_by_timepoint,
        timepoints=timepoints,
        mapping_name="X_df_by_timepoint",
    )

    reference_features = [str(f) for f in reference_features]

    if len(reference_features) == 0:
        raise ValueError("reference_features must be non-empty.")

    X_by_timepoint: Dict[str, np.ndarray] = {}

    for tp in timepoint_order:
        X_df = X_df_by_timepoint[tp]

        if not isinstance(X_df, pd.DataFrame):
            raise TypeError(
                f"X_df_by_timepoint[{tp!r}] must be a pandas DataFrame so features "
                "can be selected by column name."
            )

        X_df_str_cols = X_df.copy()
        X_df_str_cols.columns = X_df_str_cols.columns.astype(str)

        missing_features = [
            f for f in reference_features
            if f not in X_df_str_cols.columns
        ]

        if missing_features:
            raise KeyError(
                f"Timepoint {tp!r} is missing these reference features: "
                f"{missing_features}"
            )

        X_by_timepoint[tp] = X_df_str_cols.loc[:, reference_features].to_numpy()

    return X_by_timepoint


# ============================================================
# 3. Fit reference model and assign clusters
# ============================================================
def assign_clusters_from_reference_model(
    X_by_timepoint: Mapping[str, np.ndarray],
    patient_ids_by_timepoint: Mapping[str, Sequence],
    *,
    models: Mapping[str, Any],
    reference_timepoint: str,
    timepoints: Sequence[str],
) -> Dict[str, Any]:
    """
    Fit each clustering model on one reference timepoint, then assign clusters
    to all timepoints using that same fitted reference model.
    """
    timepoint_order = _validate_timepoints_present(
        X_by_timepoint,
        timepoints=timepoints,
        mapping_name="X_by_timepoint",
    )

    _validate_timepoints_present(
        patient_ids_by_timepoint,
        timepoints=timepoint_order,
        mapping_name="patient_ids_by_timepoint",
    )

    if reference_timepoint not in timepoint_order:
        raise ValueError(
            f"reference_timepoint={reference_timepoint!r} must be one of {timepoint_order}."
        )

    if not isinstance(models, Mapping) or len(models) == 0:
        raise ValueError("models must be a non-empty mapping of model_name -> estimator.")

    X_reference = np.asarray(X_by_timepoint[reference_timepoint])

    if X_reference.ndim != 2:
        raise ValueError(
            f"X_by_timepoint[{reference_timepoint!r}] must be 2D; "
            f"got shape {X_reference.shape}."
        )

    out: Dict[str, Any] = {}

    for model_name, model_template in models.items():
        model_name = str(model_name)

        _validate_clusterer_can_predict(
            model_template,
            model_name=model_name,
        )

        fitted_model = clone(model_template)
        fitted_model.fit(X_reference)

        assignments_by_timepoint: Dict[str, pd.DataFrame] = {}

        for tp in timepoint_order:
            X_tp = np.asarray(X_by_timepoint[tp])
            patient_ids = list(patient_ids_by_timepoint[tp])

            if X_tp.ndim != 2:
                raise ValueError(
                    f"X_by_timepoint[{tp!r}] must be 2D; got shape {X_tp.shape}."
                )

            if X_tp.shape[1] != X_reference.shape[1]:
                raise ValueError(
                    f"Timepoint {tp!r} has {X_tp.shape[1]} features, but reference "
                    f"timepoint {reference_timepoint!r} has {X_reference.shape[1]} features."
                )

            if len(patient_ids) != X_tp.shape[0]:
                raise ValueError(
                    f"patient_ids_by_timepoint[{tp!r}] length ({len(patient_ids)}) "
                    f"must match number of rows in X ({X_tp.shape[0]})."
                )

            labels = fitted_model.predict(X_tp)

            assignments_by_timepoint[tp] = pd.DataFrame(
                {
                    "patient_id": patient_ids,
                    "cluster": labels.astype(int),
                }
            )

        out[model_name] = {
            "reference_timepoint": reference_timepoint,
            "fitted_model": fitted_model,
            "assignments_by_timepoint": assignments_by_timepoint,
        }

    return out


# ============================================================
# 4. Build patient-level trajectories
# ============================================================
def build_membership_trajectory_table(
    assignments_by_timepoint: Mapping[str, pd.DataFrame],
    *,
    timepoints: Sequence[str],
    patient_id_col: str = "patient_id",
    cluster_col: str = "cluster",
    require_complete_cases: bool = True,
) -> pd.DataFrame:
    """
    Merge cluster assignments across timepoints into one patient-level table.

    Output:
        patient_id | baseline_cluster | week6_cluster | month6_cluster | trajectory
    """
    timepoint_order = _validate_timepoints_present(
        assignments_by_timepoint,
        timepoints=timepoints,
        mapping_name="assignments_by_timepoint",
    )

    merged: Optional[pd.DataFrame] = None

    for tp in timepoint_order:
        df = assignments_by_timepoint[tp].copy()

        required_cols = {patient_id_col, cluster_col}
        missing_cols = sorted(required_cols - set(df.columns))
        if missing_cols:
            raise KeyError(
                f"Assignments for timepoint {tp!r} are missing columns: {missing_cols}"
            )

        df = df[[patient_id_col, cluster_col]].copy()
        df = df.rename(columns={cluster_col: f"{tp}_cluster"})

        if df[patient_id_col].duplicated().any():
            duplicated_ids = df.loc[
                df[patient_id_col].duplicated(),
                patient_id_col,
            ].tolist()
            raise ValueError(
                f"Assignments for timepoint {tp!r} contain duplicated patient IDs. "
                f"Examples: {duplicated_ids[:10]}"
            )

        if merged is None:
            merged = df
        else:
            how = "inner" if require_complete_cases else "outer"
            merged = merged.merge(df, on=patient_id_col, how=how)

    if merged is None or merged.empty:
        raise ValueError("No patient-level membership trajectories could be built.")

    cluster_cols = [f"{tp}_cluster" for tp in timepoint_order]

    if require_complete_cases:
        merged = merged.dropna(subset=cluster_cols).reset_index(drop=True)

    def _make_trajectory(row: pd.Series) -> str:
        vals = []
        for col in cluster_cols:
            val = row[col]
            vals.append("NA" if pd.isna(val) else str(int(val)))
        return "->".join(vals)

    merged["trajectory"] = merged.apply(_make_trajectory, axis=1)

    return merged.reset_index(drop=True)


# ============================================================
# 5. Summarize membership consistency
# ============================================================
def summarize_membership_consistency(
    trajectory_df: pd.DataFrame,
    *,
    timepoints: Sequence[str],
    patient_id_col: str = "patient_id",
) -> Dict[str, pd.DataFrame]:
    """
    Summarize patient-level cluster membership consistency.
    """
    timepoint_order = list(timepoints)
    cluster_cols = [f"{tp}_cluster" for tp in timepoint_order]

    missing_cols = sorted(
        {patient_id_col, "trajectory", *cluster_cols} - set(trajectory_df.columns)
    )
    if missing_cols:
        raise KeyError(f"trajectory_df is missing required columns: {missing_cols}")

    if trajectory_df.empty:
        raise ValueError("trajectory_df is empty.")

    pairwise_rows = []

    for i in range(len(timepoint_order)):
        for j in range(i + 1, len(timepoint_order)):
            tp_a = timepoint_order[i]
            tp_b = timepoint_order[j]

            col_a = f"{tp_a}_cluster"
            col_b = f"{tp_b}_cluster"

            valid = trajectory_df[[col_a, col_b]].dropna()
            n_patients = int(valid.shape[0])

            if n_patients == 0:
                n_same = 0
                consistency = np.nan
            else:
                n_same = int((valid[col_a].astype(int) == valid[col_b].astype(int)).sum())
                consistency = n_same / n_patients

            pairwise_rows.append(
                {
                    "comparison": f"{tp_a}_vs_{tp_b}",
                    "timepoint_a": tp_a,
                    "timepoint_b": tp_b,
                    "n_patients": n_patients,
                    "n_same_cluster": n_same,
                    "membership_consistency": consistency,
                }
            )

    pairwise_consistency = pd.DataFrame(pairwise_rows)

    complete = trajectory_df.dropna(subset=cluster_cols).copy()
    n_complete = int(complete.shape[0])

    if n_complete == 0:
        n_all_same = 0
        all_timepoint_consistency_value = np.nan
    else:
        first_col = cluster_cols[0]
        all_same_mask = np.ones(n_complete, dtype=bool)

        for col in cluster_cols[1:]:
            all_same_mask &= (
                complete[first_col].astype(int).to_numpy()
                == complete[col].astype(int).to_numpy()
            )

        n_all_same = int(all_same_mask.sum())
        all_timepoint_consistency_value = n_all_same / n_complete

    all_timepoint_consistency = pd.DataFrame(
        [
            {
                "timepoints": "->".join(timepoint_order),
                "n_patients": n_complete,
                "n_same_cluster_all_timepoints": n_all_same,
                "membership_consistency_all_timepoints": all_timepoint_consistency_value,
            }
        ]
    )

    trajectory_counts = (
        trajectory_df
        .groupby("trajectory", dropna=False)
        .size()
        .reset_index(name="n_patients")
        .sort_values(["n_patients", "trajectory"], ascending=[False, True])
        .reset_index(drop=True)
    )

    patient_level = trajectory_df[[patient_id_col, *cluster_cols, "trajectory"]].copy()
    patient_level["same_cluster_all_timepoints"] = (
        patient_level[cluster_cols]
        .nunique(axis=1, dropna=True)
        .eq(1)
    )

    return {
        "pairwise_consistency": pairwise_consistency,
        "all_timepoint_consistency": all_timepoint_consistency,
        "trajectory_counts": trajectory_counts,
        "patient_level_consistency": patient_level,
    }


# ============================================================
# 6. Full wrapper
# ============================================================
def run_reference_membership_consistency(
    X_df_by_timepoint: Mapping[str, pd.DataFrame],
    patient_ids_by_timepoint: Mapping[str, Sequence],
    ranking_by_timepoint: Mapping[str, pd.DataFrame],
    *,
    models: Mapping[str, Any],
    timepoints: Sequence[str],
    reference_timepoints: Optional[Sequence[str]] = None,
    top_k_features: Optional[int] = None,
    feature_col: str = "feature",
    require_complete_cases: bool = True,
) -> Dict[str, Any]:
    """
    Run reference-model membership consistency.

    For each reference timepoint:
        1. Take that reference timepoint's top-k features.
        2. Extract those same features from all timepoints.
        3. Fit each model on the reference timepoint.
        4. Predict cluster assignments for all timepoints using that fitted model.
        5. Build patient trajectories.
        6. Summarize membership consistency.
    """
    timepoint_order = list(timepoints)

    _validate_timepoints_present(
        X_df_by_timepoint,
        timepoints=timepoint_order,
        mapping_name="X_df_by_timepoint",
    )
    _validate_timepoints_present(
        patient_ids_by_timepoint,
        timepoints=timepoint_order,
        mapping_name="patient_ids_by_timepoint",
    )
    _validate_timepoints_present(
        ranking_by_timepoint,
        timepoints=timepoint_order,
        mapping_name="ranking_by_timepoint",
    )

    if reference_timepoints is None:
        reference_timepoint_order = timepoint_order
    else:
        reference_timepoint_order = list(reference_timepoints)

    for ref_tp in reference_timepoint_order:
        if ref_tp not in timepoint_order:
            raise ValueError(
                f"reference_timepoint {ref_tp!r} must be one of {timepoint_order}."
            )

    out: Dict[str, Any] = {}

    for reference_timepoint in reference_timepoint_order:
        reference_features = get_reference_top_features(
            ranking_by_timepoint,
            reference_timepoint=reference_timepoint,
            top_k_features=top_k_features,
            feature_col=feature_col,
        )

        X_ref_feature_by_timepoint = build_reference_feature_matrices(
            X_df_by_timepoint,
            reference_features=reference_features,
            timepoints=timepoint_order,
        )

        assigned_by_model = assign_clusters_from_reference_model(
            X_ref_feature_by_timepoint,
            patient_ids_by_timepoint,
            models=models,
            reference_timepoint=reference_timepoint,
            timepoints=timepoint_order,
        )

        by_model: Dict[str, Any] = {}

        for model_name, model_out in assigned_by_model.items():
            trajectory_table = build_membership_trajectory_table(
                model_out["assignments_by_timepoint"],
                timepoints=timepoint_order,
                patient_id_col="patient_id",
                cluster_col="cluster",
                require_complete_cases=require_complete_cases,
            )

            summary = summarize_membership_consistency(
                trajectory_table,
                timepoints=timepoint_order,
                patient_id_col="patient_id",
            )

            by_model[model_name] = {
                "assignments_by_timepoint": model_out["assignments_by_timepoint"],
                "trajectory_table": trajectory_table,
                "summary": summary,
                "fitted_model": model_out["fitted_model"],
            }

        out[f"{reference_timepoint}_reference"] = {
            "reference_timepoint": reference_timepoint,
            "reference_features": reference_features,
            "top_k_features": top_k_features,
            "by_model": by_model,
        }

    return out


# ============================================================
# 7. Transition table for plotting
# ============================================================
def compute_membership_transition_long_table(
    trajectory_table: pd.DataFrame,
    *,
    timepoint_a: str,
    timepoint_b: str,
    normalize: str = "row",
) -> pd.DataFrame:
    """
    Build a long transition table for one source -> target comparison.

    For normalize='row':
        Within each source cluster, destination proportions sum to 1.
    """
    col_a = f"{timepoint_a}_cluster"
    col_b = f"{timepoint_b}_cluster"

    missing_cols = sorted({col_a, col_b} - set(trajectory_table.columns))
    if missing_cols:
        raise KeyError(f"trajectory_table is missing required columns: {missing_cols}")

    if normalize not in {"row", "column", "all", "none"}:
        raise ValueError("normalize must be one of {'row', 'column', 'all', 'none'}.")

    df = trajectory_table[[col_a, col_b]].dropna().copy()

    if df.empty:
        raise ValueError(
            f"No complete cluster assignments available for {timepoint_a}_vs_{timepoint_b}."
        )

    df[col_a] = df[col_a].astype(int)
    df[col_b] = df[col_b].astype(int)

    clusters = sorted(set(df[col_a].unique()).union(set(df[col_b].unique())))

    counts = pd.crosstab(df[col_a], df[col_b], dropna=False)
    counts = counts.reindex(index=clusters, columns=clusters, fill_value=0)

    rows = []

    for source_cluster in clusters:
        for target_cluster in clusters:
            n = int(counts.loc[source_cluster, target_cluster])

            if normalize == "row":
                denom = int(counts.loc[source_cluster, :].sum())
            elif normalize == "column":
                denom = int(counts.loc[:, target_cluster].sum())
            elif normalize == "all":
                denom = int(counts.to_numpy().sum())
            else:
                denom = 1

            if normalize == "none":
                proportion = float(n)
            else:
                proportion = n / denom if denom > 0 else np.nan

            rows.append(
                {
                    "source_cluster": int(source_cluster),
                    "target_cluster": int(target_cluster),
                    "n_patients": n,
                    "proportion": proportion,
                    "denominator": denom,
                    "comparison": f"{timepoint_a}_to_{timepoint_b}",
                    "timepoint_a": timepoint_a,
                    "timepoint_b": timepoint_b,
                    "normalize": normalize,
                }
            )

    return pd.DataFrame(rows)


def compute_membership_transition_long_table_multi(
    trajectory_table: pd.DataFrame,
    *,
    timepoint_a: str,
    timepoint_b_list: Sequence[str],
    normalize: str = "row",
) -> pd.DataFrame:
    """
    Build one long transition table from one source timepoint to multiple targets.
    """
    if len(timepoint_b_list) == 0:
        raise ValueError("timepoint_b_list must contain at least one target timepoint.")

    all_rows = []

    for timepoint_b in timepoint_b_list:
        one_df = compute_membership_transition_long_table(
            trajectory_table=trajectory_table,
            timepoint_a=timepoint_a,
            timepoint_b=timepoint_b,
            normalize=normalize,
        )

        one_df["target_timepoint"] = timepoint_b
        all_rows.append(one_df)

    return pd.concat(all_rows, axis=0, ignore_index=True)


# ============================================================
# 8. Final grouped barplot
# ============================================================

def plot_membership_transition_grouped_barplot_multi(
    trajectory_table: pd.DataFrame,
    *,
    timepoint_a: str,
    timepoint_b_list: Sequence[str],
    normalize: str = "row",
    figsize: Tuple[float, float] = (10.5, 5.5),
    font_size: float = 12.0,
    ylim: Tuple[float, float] = (0.0, 1.0),
    annotate: bool = True,
    show_counts: bool = True,
    timepoint_palette: Optional[Mapping[str, str]] = None,
    source_cluster_label_prefix: str = "Source Cluster",
    destination_cluster_label_prefix: str = "Destination Cluster",
    timepoint_alias: Optional[Mapping[str, str]] = None,
    legend_loc: Optional[str] = None,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """
    Plot membership transitions from one source timepoint to multiple target timepoints.

    Design:
        x-axis major groups = source clusters at timepoint_a
        x-axis subgroups    = destination clusters
        color               = target timepoint
        y-axis              = percent of source cluster

    Recommended:
        normalize='row'

    Interpretation:
        Among patients in source cluster X at timepoint_a,
        what percent are assigned to each destination cluster at week6/month6/etc.?
    """
    plot_df = compute_membership_transition_long_table_multi(
        trajectory_table=trajectory_table,
        timepoint_a=timepoint_a,
        timepoint_b_list=timepoint_b_list,
        normalize=normalize,
    )

    source_clusters = sorted(plot_df["source_cluster"].unique().tolist())
    target_clusters = sorted(plot_df["target_cluster"].unique().tolist())
    target_timepoints = list(timepoint_b_list)

    n_source = len(source_clusters)
    n_dest = len(target_clusters)
    n_time = len(target_timepoints)

    if n_source == 0 or n_dest == 0 or n_time == 0:
        raise ValueError(
            "No source clusters, destination clusters, or target timepoints to plot."
        )

    if timepoint_alias is None:
        timepoint_alias = {}

    def _timepoint_label(tp: str) -> str:
        return timepoint_alias.get(tp, tp)

    if timepoint_palette is None:
        default_colors = [
            "#1587F8",
            "#F14949",
            "#2CA02C",
            "#9467BD",
            "#FF7F0E",
            "#8C564B",
        ]
        timepoint_palette = {
            tp: default_colors[i % len(default_colors)]
            for i, tp in enumerate(target_timepoints)
        }
    else:
        missing = [tp for tp in target_timepoints if tp not in timepoint_palette]
        if missing:
            raise ValueError(f"timepoint_palette is missing colors for timepoints: {missing}")

    fig, ax = plt.subplots(figsize=figsize)

    source_gap = 0.8
    dest_gap = 0.24
    bar_width = 0.28

    subgroup_width = n_time * bar_width
    source_group_width = n_dest * subgroup_width + (n_dest - 1) * dest_gap

    source_start_positions = {}
    current_x = 0.0

    for source_cluster in source_clusters:
        source_start_positions[source_cluster] = current_x
        current_x += source_group_width + source_gap

    x_tick_positions = []
    x_tick_labels = []

    for source_cluster in source_clusters:
        source_start = source_start_positions[source_cluster]

        for d_idx, dest_cluster in enumerate(target_clusters):
            dest_start = source_start + d_idx * (subgroup_width + dest_gap)

            dest_center = dest_start + (subgroup_width - bar_width) / 2
            x_tick_positions.append(dest_center)
            x_tick_labels.append(f"{destination_cluster_label_prefix} {dest_cluster}")

            for t_idx, target_tp in enumerate(target_timepoints):
                row_df = plot_df.loc[
                    (plot_df["source_cluster"] == source_cluster)
                    & (plot_df["target_cluster"] == dest_cluster)
                    & (plot_df["target_timepoint"] == target_tp)
                ]

                if row_df.empty:
                    value = np.nan
                    n_patients = 0
                else:
                    row = row_df.iloc[0]
                    value = float(row["proportion"])
                    n_patients = int(row["n_patients"])

                x = dest_start + t_idx * bar_width

                ax.bar(
                    x,
                    value,
                    width=bar_width,
                    color=timepoint_palette[target_tp],
                    edgecolor="black",
                    linewidth=0.8,
                    label=(
                        _timepoint_label(target_tp)
                        if (
                            source_cluster == source_clusters[0]
                            and dest_cluster == target_clusters[0]
                        )
                        else None
                    ),
                )

                if annotate:
                    if pd.isna(value):
                        label = "NA"
                        y_text = 0.0
                    else:
                        if normalize == "none":
                            label = f"{int(value)}"
                        else:
                            label = f"{value:.0%}"
                            if show_counts:
                                label += f"\nn={n_patients}"
                        y_text = value

                    ax.text(
                        x,
                        y_text + 0.015,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=max(font_size - 4, 7),
                        fontweight="bold",
                    )

    for source_cluster in source_clusters:
        source_start = source_start_positions[source_cluster]
        source_center = source_start + (source_group_width - bar_width) / 2

        ax.text(
            source_center,
            -0.12 if normalize != "none" else -0.08,
            f"{timepoint_a} {source_cluster_label_prefix} {source_cluster}",
            ha="center",
            va="top",
            fontsize=font_size,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )

        if source_cluster != source_clusters[-1]:
            sep_x = source_start + source_group_width + source_gap / 2 - bar_width / 2
            ax.axvline(sep_x, color="gray", linewidth=0.8, alpha=0.35)

    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(
        x_tick_labels,
        fontsize=max(font_size - 1, 8),
        fontweight="bold",
    )

    if normalize == "none":
        ax.set_ylabel("Number of patients", fontsize=font_size, fontweight="bold")
    else:
        ax.set_ylabel("Percent of source cluster", fontsize=font_size, fontweight="bold")
        ax.set_ylim(*ylim)

    ax.set_xlabel(
        "Destination cluster within each source cluster",
        fontsize=font_size,
        fontweight="bold",
        labelpad=34,
    )

    if title is None:
        title = f"{timepoint_a} membership transitions to {', '.join(target_timepoints)}"

    ax.set_title(title, fontsize=font_size + 2, fontweight="bold")

    ax.tick_params(axis="both", labelsize=font_size)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.grid(axis="y", alpha=0.25)

    if legend_loc is None:
        legend_loc = "upper right"

    legend_kwargs = {
        "title": "Target timepoint",
        "fontsize": max(font_size - 1, 8),
        "title_fontsize": font_size,
        "frameon": True,
        "loc": legend_loc,
    }

    if legend_bbox_to_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor

    ax.legend(**legend_kwargs)

    fig.tight_layout()

    return plot_df, fig, ax