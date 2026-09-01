
"""
clinical_response_ga_clustering.py

Single-file implementation for clinical-response-guided GA feature selection.

This file contains everything needed for the current workflow:

1. Add clinical/treatment columns to an existing synthetic feature dataset.
2. Validate the generated synthetic clinical-response signal.
3. Run a teammate-style clinical-response GA on a SINGLE dataframe.

Core idea
---------
The dataframe is one row per subject/sample and contains:

    feature_0 ... feature_p
    treatment_arm
    vi3_v1_soc_ss
    vi3_v7_soc_ss
    age                         optional
    hidden_subtype / label       optional, validation only

The GA uses only feature columns for clustering. Treatment and outcome columns are
used after clustering to score whether discovered clusters show different treatment
effects.

The clinical-response fitness is:

    fitness =
        ws * bootstrap_stability
      + wc * clinical_meaningfulness
      + wq * cluster_quality
      - wf * feature_count_penalty

The clinical meaningfulness score, Sclinical, follows the teammate-style objective:
clusters should show heterogeneous treatment effects.
"""

from __future__ import annotations


# =============================================================================
# Imports
# =============================================================================

import os
import time
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

Number = Union[int, float]


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch, Patch
from matplotlib.path import Path
import math

from matplotlib.lines import Line2D
from sklearn.decomposition import PCA


from scipy import stats

# =============================================================================
# Synthetic clinical-response dataframe config
# =============================================================================

@dataclass
class SyntheticClinicalResponseConfig:
    """
    Configuration for adding clinical/treatment columns to an existing synthetic
    feature-selection dataset.

    The expected input data dictionary comes from a feature generator and contains:

        data["X"]
        data["y"]
        data["feature_names"]

    This config controls how treatment assignment and outcome scores are generated.

    Assumption
    ----------
    Higher follow-up score is better. Therefore, a positive treatment effect
    increases the follow-up score for treated subjects.
    """

    # Output column names.
    treatment_col: str = "treatment_arm"
    baseline_col: str = "vi3_v1_soc_ss"
    followup_col: str = "vi3_v7_soc_ss"
    hidden_subtype_col: str = "hidden_subtype"
    label_col: str = "label"
    age_col: str = "age"

    # Treatment labels matching the teammate code convention.
    treated_label: str = "Treatment"
    control_label: str = "Placebo"
    treatment_probability: float = 0.5

    # Baseline score generation.
    baseline_mean: float = 65.0
    baseline_sd: float = 10.0
    baseline_range: Tuple[float, float] = (20.0, 105.0)

    # Optional baseline shift by hidden subtype/class.
    baseline_shift_by_subtype: Optional[Mapping[Any, float]] = None

    # Natural non-treatment change from baseline to follow-up.
    natural_change_mean: float = 0.0
    natural_change_sd: float = 3.0

    # Treatment effect by hidden subtype/class.
    # If None:
    #   binary y: class 0 -> 0, class 1 -> +8
    #   multiclass y: effects linearly spaced from -4 to +8
    treatment_effect_by_subtype: Optional[Mapping[Any, float]] = None

    # Extra outcome noise at follow-up.
    followup_noise_sd: float = 5.0

    # Optional age generation.
    add_age: bool = True
    age_mean: float = 10.0
    age_sd: float = 2.0
    age_range: Tuple[float, float] = (5.0, 18.0)

    # Optional post-hoc label column.
    add_label: bool = True

    # Reproducibility.
    random_state: int = 42


# =============================================================================
# Synthetic clinical-response dataframe helpers
# =============================================================================

def _validate_probability(name: str, value: float) -> None:
    """
    Validate that a probability is in [0, 1].

    Parameters
    ----------
    name:
        Name used in the error message.

    value:
        Probability value to validate.
    """
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1; got {value}.")


def _default_treatment_effects(y: np.ndarray) -> Dict[Any, float]:
    """
    Create default treatment effects by hidden subtype/class.

    Binary default:
        lower sorted class -> 0.0
        higher sorted class -> +8.0

    Multiclass default:
        effects linearly spaced from -4.0 to +8.0.

    Parameters
    ----------
    y:
        Hidden subtype/class array.

    Returns
    -------
    dict
        Mapping from class/subtype value to treatment effect.
    """
    classes = sorted(pd.Series(y).dropna().unique().tolist())

    if len(classes) == 0:
        raise ValueError("Cannot infer treatment effects because y has no valid classes.")

    if len(classes) == 1:
        return {classes[0]: 4.0}

    if len(classes) == 2:
        return {classes[0]: 0.0, classes[1]: 8.0}

    effects = np.linspace(-4.0, 8.0, num=len(classes))
    return {cls: float(effect) for cls, effect in zip(classes, effects)}


def make_synthetic_clinical_response_dataframe(
    data: Dict[str, Any],
    *,
    config: Optional[SyntheticClinicalResponseConfig] = None,
) -> Dict[str, Any]:
    """
    Add treatment and clinical outcome columns to an existing synthetic dataset.

    This keeps the data as a SINGLE dataframe.

    Parameters
    ----------
    data:
        Dictionary with required keys:
            - "X": numeric feature matrix, shape (n_samples, n_features)
            - "y": hidden subtype/class labels, shape (n_samples,)
            - "feature_names": list of feature names, length n_features

        Optional keys are passed through:
            - "true_informative"
            - "collinearity_info"
            - "groups"

    config:
        Configuration controlling treatment assignment and outcome generation.

    Returns
    -------
    dict
        {
            "df": pd.DataFrame,
            "feature_cols": list[str],
            "treatment_col": str,
            "baseline_col": str,
            "followup_col": str,
            "age_col": str | None,
            "hidden_subtype_col": str,
            "label_col": str | None,
            "true_informative": set[str] | None,
            "collinearity_info": dict | None,
            "groups": Any,
            "treatment_effect_by_subtype": dict,
            "config": dict,
        }

    Notes
    -----
    Follow-up is generated as:

        followup =
            baseline
          + natural_change
          + I(treated) * treatment_effect_by_hidden_subtype
          + noise
    """
    if config is None:
        config = SyntheticClinicalResponseConfig()

    required = ["X", "y", "feature_names"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"data is missing required keys: {missing}")

    _validate_probability("treatment_probability", config.treatment_probability)

    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"])
    feature_cols = list(data["feature_names"])

    if X.ndim != 2:
        raise ValueError(f"data['X'] must be 2D; got shape {X.shape}.")
    if len(y) != X.shape[0]:
        raise ValueError(
            f"len(data['y']) must equal number of rows in X; "
            f"got len(y)={len(y)} and X.shape[0]={X.shape[0]}."
        )
    if len(feature_cols) != X.shape[1]:
        raise ValueError(
            f"len(data['feature_names']) must equal number of columns in X; "
            f"got {len(feature_cols)} names and X.shape[1]={X.shape[1]}."
        )
    if len(set(feature_cols)) != len(feature_cols):
        raise ValueError("feature_names must be unique.")

    rng = np.random.default_rng(config.random_state)
    n_samples = X.shape[0]
    score_low, score_high = config.baseline_range

    # -------------------------------------------------------------------------
    # Build feature dataframe.
    # -------------------------------------------------------------------------
    df = pd.DataFrame(X, columns=feature_cols)

    # Hidden subtype is used for validation only, not as a clustering feature.
    df[config.hidden_subtype_col] = y

    if config.add_label:
        df[config.label_col] = y
        label_col_out: Optional[str] = config.label_col
    else:
        label_col_out = None

    # -------------------------------------------------------------------------
    # Treatment assignment.
    # -------------------------------------------------------------------------
    is_treated = rng.binomial(1, config.treatment_probability, size=n_samples).astype(int)
    df[config.treatment_col] = np.where(
        is_treated == 1,
        config.treated_label,
        config.control_label,
    )

    # -------------------------------------------------------------------------
    # Baseline score.
    # -------------------------------------------------------------------------
    baseline = rng.normal(config.baseline_mean, config.baseline_sd, size=n_samples)

    if config.baseline_shift_by_subtype is not None:
        shifts = (
            pd.Series(y)
            .map(dict(config.baseline_shift_by_subtype))
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        baseline = baseline + shifts

    baseline = np.clip(baseline, score_low, score_high)

    # -------------------------------------------------------------------------
    # Follow-up score with subtype-specific treatment response.
    # -------------------------------------------------------------------------
    if config.treatment_effect_by_subtype is None:
        treatment_effect_map = _default_treatment_effects(y)
    else:
        treatment_effect_map = dict(config.treatment_effect_by_subtype)

    treatment_effect = (
        pd.Series(y)
        .map(treatment_effect_map)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    natural_change = rng.normal(
        config.natural_change_mean,
        config.natural_change_sd,
        size=n_samples,
    )
    followup_noise = rng.normal(0.0, config.followup_noise_sd, size=n_samples)

    followup = baseline + natural_change + is_treated * treatment_effect + followup_noise
    followup = np.clip(followup, score_low, score_high)

    df[config.baseline_col] = baseline
    df[config.followup_col] = followup

    # -------------------------------------------------------------------------
    # Optional age column.
    # -------------------------------------------------------------------------
    if config.add_age:
        age_low, age_high = config.age_range
        age = rng.normal(config.age_mean, config.age_sd, size=n_samples)
        df[config.age_col] = np.clip(age, age_low, age_high)
        age_col_out: Optional[str] = config.age_col
    else:
        age_col_out = None

    return {
        "df": df,
        "feature_cols": feature_cols,
        "treatment_col": config.treatment_col,
        "baseline_col": config.baseline_col,
        "followup_col": config.followup_col,
        "age_col": age_col_out,
        "hidden_subtype_col": config.hidden_subtype_col,
        "label_col": label_col_out,
        "true_informative": data.get("true_informative"),
        "collinearity_info": data.get("collinearity_info"),
        "groups": data.get("groups"),
        "treatment_effect_by_subtype": treatment_effect_map,
        "config": asdict(config),
    }


def summarize_synthetic_clinical_response(
    df: pd.DataFrame,
    *,
    treatment_col: str = "treatment_arm",
    baseline_col: str = "vi3_v1_soc_ss",
    followup_col: str = "vi3_v7_soc_ss",
    hidden_subtype_col: str = "hidden_subtype",
) -> pd.DataFrame:
    """
    Summarize synthetic clinical response by hidden subtype and treatment arm.

    Parameters
    ----------
    df:
        Single dataframe containing treatment, baseline, follow-up, and hidden subtype.

    treatment_col:
        Treatment column name.

    baseline_col:
        Baseline outcome column name.

    followup_col:
        Follow-up outcome column name.

    hidden_subtype_col:
        Synthetic hidden subtype column name.

    Returns
    -------
    pd.DataFrame
        Summary table with counts, baseline mean, follow-up mean, and change mean.
    """
    d = df.copy()
    d["change"] = (
        pd.to_numeric(d[followup_col], errors="coerce")
        - pd.to_numeric(d[baseline_col], errors="coerce")
    )

    return (
        d.groupby([hidden_subtype_col, treatment_col], dropna=False)
        .agg(
            n=(followup_col, "size"),
            baseline_mean=(baseline_col, "mean"),
            followup_mean=(followup_col, "mean"),
            change_mean=("change", "mean"),
            change_sd=("change", "std"),
        )
        .reset_index()
    )

def summarize_ga_search_config_simple(
    *,
    n_total_features,
    min_features,
    max_features,
    num_generations,
    sol_per_pop,
    num_parents_mating,
    keep_parents,
    mutation_percent_genes,
):

    avg_selected = (min_features + max_features) / 2
    expected_mutated_genes = n_total_features * mutation_percent_genes / 100
    total_masks_evaluated = num_generations * sol_per_pop

    total_possible_masks = sum(
        math.comb(n_total_features, k)
        for k in range(min_features, max_features + 1)
    )

    parent_pct = 100 * num_parents_mating / sol_per_pop
    keep_pct = 100 * keep_parents / sol_per_pop
    selected_min_pct = 100 * min_features / n_total_features
    selected_max_pct = 100 * max_features / n_total_features

    summary_df = pd.DataFrame([
        {
            "Concept": "Feature space",
            "Value": f"{n_total_features} candidate features",
            "Interpretation": "Total number of features the GA can choose from.",
        },
        {
            "Concept": "Mask size constraint",
            "Value": f"{min_features}–{max_features} selected features",
            "Interpretation": (
                f"Each candidate solution uses only {selected_min_pct:.1f}%–"
                f"{selected_max_pct:.1f}% of all features, so this is a very sparse search."
            ),
        },
        {
            "Concept": "Possible feature masks",
            "Value": f"{total_possible_masks:,.0f}",
            "Interpretation": (
                "Number of possible feature subsets within the allowed mask-size range. "
                "This is far too large to search exhaustively."
            ),
        },
        {
            "Concept": "GA evaluation budget",
            "Value": f"{total_masks_evaluated:,} masks",
            "Interpretation": (
                f"The GA evaluates {num_generations} generations × "
                f"{sol_per_pop} masks per generation."
            ),
        },
        {
            "Concept": "Parent selection",
            "Value": f"{num_parents_mating} parents ({parent_pct:.0f}% of population)",
            "Interpretation": (
                "Controls how many candidate masks are used to create the next generation."
            ),
        },
        {
            "Concept": "Elite carryover",
            "Value": f"{keep_parents} masks ({keep_pct:.0f}% of population)",
            "Interpretation": (
                "Keeps the best masks unchanged so strong solutions are not lost."
            ),
        },
        {
            "Concept": "Mutation scale",
            "Value": (
                f"{mutation_percent_genes}% ≈ "
                f"{expected_mutated_genes:.1f} genes per offspring"
            ),
            "Interpretation": (
                f"Because valid masks only select about {avg_selected:.1f} features on average, "
                "this mutation rate makes small local edits instead of replacing most of the mask."
            ),
        },
    ])

    notes = [
        (
            f"The GA is searching a huge combinatorial space "
            f"({total_possible_masks:,.0f} possible masks) using only "
            f"{total_masks_evaluated:,} evaluated masks."
        ),
        (
            f"The feature masks are very sparse: only {min_features}–{max_features} "
            f"features are selected out of {n_total_features}."
        ),
        (
            f"A {mutation_percent_genes}% mutation rate changes about "
            f"{expected_mutated_genes:.1f} genes per offspring, which is appropriate "
            f"for sparse masks of size {min_features}–{max_features}."
        ),
        (
            f"Parent selection uses {parent_pct:.0f}% of the population, and elite carryover "
            f"preserves {keep_pct:.0f}% of masks. These are reasonable starting values."
        ),
    ]

    return summary_df, notes

# =============================================================================
# Clinical-response GA config
# =============================================================================

@dataclass
class ClinicalResponseGAFSConfig:
    """
    Configuration for clinical-response-guided GA feature selection.

    The defaults mirror the teammate-style objective:

        stability + clinical meaningfulness + cluster quality - feature penalty.
    """

    # Clustering.
    k: int = 2
    n_init: int = 20
    random_seed: int = 42

    # Optional clustering-model dictionary.
    # If models is None, the code uses KMeans(n_clusters=k, random_state=random_seed, n_init=n_init).
    # If models is provided, each value must implement fit_predict(X_scaled), or fit(X_scaled)
    # followed by labels_ or predict(X_scaled). The first model in the dictionary is used by
    # default unless active_model_name is set.
    models: Optional[Dict[str, Any]] = None
    active_model_name: Optional[str] = None

    # Clinical endpoint columns.
    # These are kept for backward compatibility. New code can instead use
    # clinical_config below to keep treatment/outcome settings grouped.
    baseline_col: str = "vi3_v1_soc_ss"
    followup_col: str = "vi3_v7_soc_ss"

    # ------------------------------------------------------------------
    # Feature role configuration.
    # ------------------------------------------------------------------
    # feature_config defines which columns are used to CREATE clusters.
    # These are the candidate features that the GA selects from.
    #
    # Example:
    # feature_config={
    #     "clustering_feature_cols": ["feature_0", "feature_1", ...]
    # }
    feature_config: Optional[Dict[str, Any]] = None

    # Optional validation-only columns for synthetic/debug workflows.
    # These columns are NOT used for clustering or scoring.
    #
    # Example:
    # validation_config={
    #     "hidden_subtype_col": "hidden_subtype",
    #     "label_col": "label",
    # }
    validation_config: Optional[Dict[str, Any]] = None

    # Optional logging/plotting configuration.
    #
    # If metrics_to_show is None, logs and plots automatically use the active
    # fitness_components for the current fitness_preset.
    #
    # Example:
    # logging_config={
    #     "metrics_to_show": [
    #         "bootstrap_ari_norm",
    #         "silhouette_norm",
    #         "feature_penalty_norm",
    #     ],
    #     "metric_display_names": {
    #         "bootstrap_ari_norm": "ARI",
    #         "silhouette_norm": "sil",
    #         "feature_penalty_norm": "feat_pen",
    #     },
    # }
    logging_config: Optional[Dict[str, Any]] = None

    # Optional grouped clinical configuration kept for backward compatibility.
    # New code should prefer:
    # cfg.fitness_preset_config[preset]["scoring_columns"].
    clinical_config: Optional[Dict[str, Any]] = None

    # GA parameters.
    num_generations: int = 10
    sol_per_pop: int = 20
    num_parents_mating: int = 6
    keep_parents: int = 2
    keep_elitism: int = 1
    parent_selection_type: str = "sss"
    crossover_type: str = "uniform"
    mutation_type: str = "random"
    mutation_percent_genes: int = 10

    # ------------------------------------------------------------------
    # Legacy fitness weights.
    # ------------------------------------------------------------------
    # These are kept for backward compatibility. If fitness_components is None,
    # these weights are converted into the default weighted-sum fitness:
    #
    #   ws * stability_norm
    # + wc * sclin_norm
    # + wq * cluster_norm
    # - wf * feature_penalty_norm
    ws: float = 0.20  # Stability weight.
    wc: float = 0.50  # Clinical meaningfulness weight.
    wq: float = 0.20  # Cluster quality weight.
    wf: float = 0.10  # Feature-count penalty weight.

    # ------------------------------------------------------------------
    # Dynamic fitness configuration.
    # ------------------------------------------------------------------
    # fitness_function_name controls HOW metrics are combined.
    #
    # Supported built-ins:
    #   "weighted_sum"     : weighted sum of configured components.
    #   "stability_only"   : stability_norm - feature penalty if available.
    #   "clinical_only"    : sclin_norm - feature penalty if available.
    #   "gated_clinical"   : clinical score only if stability exceeds a threshold.
    #
    # fitness_components controls WHICH metrics enter the weighted_sum function.
    # If None, the legacy ws/wc/wq/wf weights are used.
    #
    # Example:
    # fitness_components={
    #     "stability_norm": {"weight": 1.0, "direction": "maximize"},
    #     "feature_penalty_norm": {"weight": 0.1, "direction": "minimize"},
    # }
    fitness_function_name: str = "weighted_sum"

    # Optional named fitness function / preset.
    # Used when fitness_function_name="weighted_sum".
    # The preset selects the function. fitness_components supplies the weights.
    #
    # Available presets:
    #   "balanced_clinical"         : stability + Sclinical + cluster quality - feature penalty
    #   "unsupervised_clustering"   : stability + cluster quality - feature penalty
    #   "label_guided_clustering"   : stability + cluster quality + label alignment - feature penalty
    #   "stability_only"            : stability - feature penalty
    #   "cluster_quality_only"      : silhouette-derived quality - feature penalty
    fitness_preset: Optional[str] = "balanced_clinical"

    # Explicit component dictionary. Kept for backward compatibility.
    # New code should prefer fitness_preset_config below.
    fitness_components: Optional[Dict[str, Dict[str, Any]]] = None

    # Nested configuration for each named fitness preset/function.
    #
    # Example:
    # fitness_preset="balanced_clinical"
    # fitness_preset_config={
    #     "balanced_clinical": {
    #         "fitness_components": {
    #             "stability_norm": {"weight": 0.20, "direction": "maximize"},
    #             "sclin_norm": {"weight": 0.50, "direction": "maximize"},
    #             "cluster_norm": {"weight": 0.20, "direction": "maximize"},
    #             "feature_penalty_norm": {"weight": 0.10, "direction": "minimize"},
    #         },
    #         "sclinical": {
    #             "w1_spread": 1.0,
    #             "w2_opposite_sign_bonus": 5.0,
    #             "w3_precision": 1.0,
    #             "w4_small_cluster_penalty": 5.0,
    #             "w5_arm_imbalance_penalty": 2.0,
    #             "scale": 30.0,
    #         },
    #         "feature_selection": {
    #             "min_features": 3,
    #             "max_features": 10,
    #             "feature_fraction_penalty_power": 1.0,
    #         },
    #         "clinical_effect": {
    #             "min_cluster_total_n": 12,
    #             "min_arm_n_per_cluster": 3,
    #         },
    #     }
    # }
    fitness_preset_config: Optional[Dict[str, Dict[str, Any]]] = None

    custom_fitness_function: Optional[Callable[[Dict[str, float]], float]] = None
    gated_metric: str = "stability_norm"
    gated_threshold: float = 0.75
    gated_primary_metric: str = "sclin_norm"
    gated_penalty_metric: str = "feature_penalty_norm"
    gated_penalty_weight: float = 0.10

    # Sclinical subweights.
    sclin_w1_spread: float = 1.0
    sclin_w2_opposite_sign_bonus: float = 5.0
    sclin_w3_precision: float = 1.0
    sclin_w4_small_cluster_penalty: float = 5.0
    sclin_w5_arm_imbalance_penalty: float = 2.0
    sclin_scale: float = 30.0

    # Bootstrap stability.
    n_bootstrap: int = 10
    bootstrap_random_seed: int = 123

    # Feature subset constraints.
    min_features: int = 3
    max_features: int = 10
    feature_fraction_penalty_power: float = 1.0

    # Clinical effect constraints.
    min_cluster_total_n: int = 12
    min_arm_n_per_cluster: int = 3

    # General.
    eps: float = 1e-6
    use_cache: bool = True



def _safe_config_value(value: Any) -> Any:
    """
    Convert config values into result-safe / serialization-friendly objects.

    This is especially important for longitudinal configs because users may put
    full pandas dataframes under fitness_preset_config[preset]["timepoint_config"]["timepoint_dfs"].
    Those dataframes should not be copied into result["config"] or written to disk.
    """
    if isinstance(value, pd.DataFrame):
        return {
            "__type__": "DataFrame",
            "shape": list(value.shape),
            "columns": list(map(str, value.columns)),
        }

    if isinstance(value, np.ndarray):
        return {
            "__type__": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, Mapping):
        return {str(k): _safe_config_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_safe_config_value(v) for v in value]

    if callable(value):
        return getattr(value, "__name__", repr(value))

    # sklearn models and other complex objects are represented compactly.
    module = getattr(value.__class__, "__module__", "")
    if module.startswith("sklearn") or module.startswith("xgboost") or module.startswith("lightgbm"):
        return repr(value)

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def safe_config_dict(cfg: ClinicalResponseGAFSConfig) -> Dict[str, Any]:
    """
    Convert config to a result-safe dictionary.

    This avoids serialization issues when config contains sklearn models, custom
    callables, or runtime dataframes inside longitudinal timepoint_config.
    """
    out: Dict[str, Any] = {field.name: getattr(cfg, field.name) for field in fields(cfg)}

    if cfg.models is not None:
        out["models"] = {name: repr(model) for name, model in cfg.models.items()}

    if cfg.custom_fitness_function is not None:
        out["custom_fitness_function"] = getattr(
            cfg.custom_fitness_function,
            "__name__",
            repr(cfg.custom_fitness_function),
        )

    return _safe_config_value(out)



def fitness_preset_requires_clinical_scoring(preset_name: Optional[str]) -> bool:
    """
    Return whether the active fitness preset requires clinical/treatment columns.

    Presets that include Sclinical need:
        treatment_col, baseline_col, followup_col, treated_label, control_label

    Purely unsupervised presets do not need clinical_config.
    """
    name = str(preset_name or "balanced_clinical").lower()
    return name in {"balanced_clinical"}



def resolve_feature_config(
    cfg: ClinicalResponseGAFSConfig,
    *,
    feature_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Resolve clustering feature columns.

    These are the only columns used to CREATE clusters.

    Resolution order
    ----------------
    1. cfg.feature_config["clustering_feature_cols"]
    2. feature_cols argument, kept for backward compatibility

    Parameters
    ----------
    cfg:
        GA configuration.

    feature_cols:
        Optional fallback list of candidate clustering features.

    Returns
    -------
    list[str]
        Candidate clustering feature columns.

    Raises
    ------
    ValueError
        If no clustering feature columns are provided.
    """
    feature_cfg = dict(cfg.feature_config or {})

    resolved = feature_cfg.get("clustering_feature_cols", feature_cols)

    # Longitudinal convenience: when the active preset provides
    # timepoint_feature_cols, infer the GA chromosome columns from the
    # reference timepoint feature list. This lets users keep all timepoint
    # information in one config block.
    if resolved is None:
        try:
            preset_cfg = get_active_fitness_preset_config(cfg)
            timepoint_cfg = dict(preset_cfg.get("timepoint_config", {}) or {})
            tp_feature_cols = timepoint_cfg.get("timepoint_feature_cols", None)
            reference_tp = timepoint_cfg.get("reference_timepoint", None)

            if tp_feature_cols is not None:
                tp_feature_cols = dict(tp_feature_cols)
                if reference_tp is None:
                    reference_tp = "baseline" if "baseline" in tp_feature_cols else next(iter(tp_feature_cols.keys()))
                if reference_tp in tp_feature_cols:
                    resolved = tp_feature_cols[reference_tp]
        except Exception:
            resolved = None

    if resolved is None:
        raise ValueError(
            "No clustering feature columns were provided. Set "
            "cfg.feature_config={'clustering_feature_cols': [...]}, pass feature_cols=..., "
            "or provide timepoint_config['timepoint_feature_cols'][reference_timepoint] "
            "for longitudinal presets."
        )

    resolved = list(resolved)

    if len(resolved) == 0:
        raise ValueError("clustering_feature_cols cannot be empty.")

    if len(set(resolved)) != len(resolved):
        raise ValueError("clustering_feature_cols contains duplicate names.")

    return resolved




def resolve_single_timepoint_dataframe(
    cfg: ClinicalResponseGAFSConfig,
    *,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Resolve the dataframe for the single-timepoint GA workflow.

    Resolution order
    ----------------
    1. Explicit ``df`` argument passed to ``make_clinical_response_ga`` or
       ``evaluate_mask_clinical_response``.
    2. ``cfg.fitness_preset_config[cfg.fitness_preset]["data_config"]["df"]``.
    3. ``cfg.fitness_preset_config[cfg.fitness_preset]["input_config"]["df"]``
       for naming compatibility.

    Returns
    -------
    pd.DataFrame
        Resolved single-timepoint dataframe.
    """
    if df is not None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame; got {type(df)!r}.")
        return df

    preset_name = cfg.fitness_preset or "balanced_clinical"
    preset_cfg = get_active_fitness_preset_config(cfg)

    data_cfg = dict(preset_cfg.get("data_config", {}) or {})
    input_cfg = dict(preset_cfg.get("input_config", {}) or {})

    resolved_df = data_cfg.get("df", input_cfg.get("df", None))

    if resolved_df is None:
        raise ValueError(
            "No dataframe was provided for the single-timepoint GA. Pass df=... to "
            "make_clinical_response_ga(...), or put it under "
            f"fitness_preset_config[{preset_name!r}]['data_config']['df']."
        )

    if not isinstance(resolved_df, pd.DataFrame):
        raise TypeError(
            "Resolved single-timepoint dataframe must be a pandas DataFrame; "
            f"got {type(resolved_df)!r}."
        )

    return resolved_df


def resolve_clinical_config(
    cfg: ClinicalResponseGAFSConfig,
    *,
    treatment_col: Optional[str] = None,
    age_col: Optional[str] = None,
    treated_label: Optional[str] = None,
    control_label: Optional[str] = None,
    require_clinical: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Resolve clinical/treatment configuration for the active fitness preset.

    Clinical configuration is only required for presets that use clinical
    treatment-response scoring, such as "balanced_clinical".

    Resolution order
    ----------------
    1. cfg.fitness_preset_config[cfg.fitness_preset]["clinical_config"]
    2. cfg.clinical_config, kept for backward compatibility
    3. loose function arguments
    4. cfg.baseline_col / cfg.followup_col fallback fields

    Parameters
    ----------
    cfg:
        Clinical-response GA configuration.

    treatment_col:
        Optional fallback treatment column name.

    age_col:
        Optional fallback age column name.

    treated_label:
        Optional fallback treated label.

    control_label:
        Optional fallback control label.

    require_clinical:
        If None, inferred from the active fitness preset.

    Returns
    -------
    dict
        Resolved clinical config. For non-clinical presets, values may be None.
    """
    preset_name = cfg.fitness_preset or "balanced_clinical"
    if require_clinical is None:
        require_clinical = fitness_preset_requires_clinical_scoring(preset_name)

    preset_cfg = get_active_fitness_preset_config(cfg)

    # New preferred location: scoring_columns under the active preset.
    preset_clinical = dict(preset_cfg.get("scoring_columns", {}) or {})

    # Backward compatibility with older naming.
    if not preset_clinical and "clinical_config" in preset_cfg:
        preset_clinical = dict(preset_cfg.get("clinical_config", {}) or {})

    # Backward compatibility with the previous top-level clinical_config.
    top_level_clinical = dict(cfg.clinical_config or {})

    resolved = {
        "treatment_col": (
            preset_clinical.get("treatment_col")
            or top_level_clinical.get("treatment_col")
            or treatment_col
        ),
        "baseline_col": (
            preset_clinical.get("baseline_col")
            or top_level_clinical.get("baseline_col")
            or cfg.baseline_col
        ),
        "followup_col": (
            preset_clinical.get("followup_col")
            or top_level_clinical.get("followup_col")
            or cfg.followup_col
        ),
        "age_col": (
            preset_clinical.get("age_col")
            if "age_col" in preset_clinical
            else top_level_clinical.get("age_col", age_col)
        ),
        "treated_label": (
            preset_clinical.get("treated_label")
            or top_level_clinical.get("treated_label")
            or treated_label
            or "Treatment"
        ),
        "control_label": (
            preset_clinical.get("control_label")
            or top_level_clinical.get("control_label")
            or control_label
            or "Placebo"
        ),
    }

    if require_clinical:
        required_keys = ["treatment_col", "baseline_col", "followup_col"]
        missing = [key for key in required_keys if resolved.get(key) in {None, ""}]
        if missing:
            raise ValueError(
                f"fitness_preset={preset_name!r} requires scoring_columns, but these "
                f"required fields are missing: {missing}. Put scoring columns under "
                f"fitness_preset_config[{preset_name!r}]['scoring_columns']."
            )

        # Keep older internal code paths working.
        cfg.baseline_col = resolved["baseline_col"]
        cfg.followup_col = resolved["followup_col"]

    return resolved

# =============================================================================
# Treatment and dataframe utilities
# =============================================================================

def standardize_treatment_arm(
    values: pd.Series,
    *,
    treated_label: str = "Treatment",
    control_label: str = "Placebo",
) -> pd.Series:
    """
    Standardize treatment coding to teammate-style treatment labels.

    Accepted treated values include:
        1, "1", "treated", "treatment", "active", "drug", treated_label

    Accepted control values include:
        0, "0", "control", "placebo", "untreated"

    Parameters
    ----------
    values:
        Treatment values.

    treated_label:
        Output label for treated subjects.

    control_label:
        Output label for control/placebo subjects.

    Returns
    -------
    pd.Series
        Standardized treatment labels.
    """
    def map_one(x: Any) -> Any:
        if pd.isna(x):
            return np.nan

        s = str(x).strip()
        sl = s.lower()

        treated_values = {
            "1", "1.0", "treated", "treatment", "active", "drug",
            treated_label.lower()
        }
        control_values = {
            "0", "0.0", "control", "placebo", "untreated",
            control_label.lower()
        }

        if sl in treated_values:
            return treated_label
        if sl in control_values:
            return control_label

        raise ValueError(
            f"Unrecognized treatment value {x!r}. Expected 0/1, treated/control, "
            f"or the configured treatment/control labels {treated_label!r}/{control_label!r}."
        )

    return values.map(map_one)


def validate_clinical_response_inputs(
    *,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    treatment_col: str,
    baseline_col: str,
    followup_col: str,
    age_col: Optional[str] = None,
) -> None:
    """
    Validate that the dataframe contains all columns required for the GA.

    Parameters
    ----------
    df:
        Single dataframe.

    feature_cols:
        Feature columns used for clustering.

    treatment_col:
        Treatment assignment column.

    baseline_col:
        Baseline outcome column.

    followup_col:
        Follow-up outcome column.

    age_col:
        Optional age column.

    Raises
    ------
    ValueError
        If required columns are missing or feature columns are invalid.
    """
    if len(feature_cols) == 0:
        raise ValueError("feature_cols cannot be empty.")

    if len(set(feature_cols)) != len(feature_cols):
        raise ValueError("feature_cols contains duplicate names.")

    required_cols = list(feature_cols) + [treatment_col, baseline_col, followup_col]
    if age_col is not None:
        required_cols.append(age_col)

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    forbidden_as_features = {treatment_col, baseline_col, followup_col}
    if age_col is not None:
        forbidden_as_features.add(age_col)

    bad_features = [col for col in feature_cols if col in forbidden_as_features]
    if bad_features:
        raise ValueError(
            "These non-feature columns were included in feature_cols: "
            f"{bad_features}"
        )


# =============================================================================
# Feature mask helpers
# =============================================================================

def _repair_mask(
    mask: Sequence[Any],
    cfg: ClinicalResponseGAFSConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Convert a candidate solution to a valid binary mask.

    The repaired mask respects cfg.min_features and cfg.max_features.
    """
    arr = np.asarray(mask, dtype=float)
    binary = (arr >= 0.5).astype(int)

    n_selected = int(binary.sum())

    if n_selected < cfg.min_features:
        zero_idx = np.where(binary == 0)[0]
        need = min(cfg.min_features - n_selected, len(zero_idx))
        if need > 0:
            add_idx = rng.choice(zero_idx, size=need, replace=False)
            binary[add_idx] = 1

    n_selected = int(binary.sum())

    if n_selected > cfg.max_features:
        one_idx = np.where(binary == 1)[0]
        drop_n = n_selected - cfg.max_features
        drop_idx = rng.choice(one_idx, size=drop_n, replace=False)
        binary[drop_idx] = 0

    return binary


def _selected_columns(feature_cols: Sequence[str], mask: Sequence[int]) -> List[str]:
    """Return selected feature column names for a binary mask."""
    return [col for col, selected in zip(feature_cols, mask) if int(selected) == 1]


def make_sparse_initial_population(
    n_features: int,
    *,
    sol_per_pop: int,
    min_features: int,
    max_features: int,
    random_seed: int,
) -> np.ndarray:
    """
    Create a sparse binary initial GA population.

    Parameters
    ----------
    n_features:
        Total number of candidate features.

    sol_per_pop:
        Number of GA solutions in the population.

    min_features:
        Minimum number of selected features per solution.

    max_features:
        Maximum number of selected features per solution.

    random_seed:
        Reproducibility seed.

    Returns
    -------
    np.ndarray
        Binary population matrix of shape (sol_per_pop, n_features).
    """
    rng = np.random.default_rng(random_seed)
    population = np.zeros((sol_per_pop, n_features), dtype=int)

    max_features = min(max_features, n_features)
    min_features = min(min_features, max_features)

    for row_idx in range(sol_per_pop):
        n_selected = int(rng.integers(min_features, max_features + 1))
        selected_idx = rng.choice(np.arange(n_features), size=n_selected, replace=False)
        population[row_idx, selected_idx] = 1

    return population


# =============================================================================
# Clustering and stability helpers
# =============================================================================

def _get_active_cluster_model(cfg: ClinicalResponseGAFSConfig) -> Tuple[str, Any]:
    """
    Return the active clustering model from the config.

    If cfg.models is None, this constructs a default KMeans model using cfg.k,
    cfg.random_seed, and cfg.n_init.

    If cfg.models is provided, the active model is:
        - cfg.active_model_name, when provided
        - otherwise the first model in cfg.models

    Returns
    -------
    tuple
        (model_name, unfitted_model)
    """
    if cfg.models is None:
        return (
            f"kmeans_{cfg.k}",
            KMeans(
                n_clusters=cfg.k,
                random_state=cfg.random_seed,
                n_init=cfg.n_init,
            ),
        )

    if len(cfg.models) == 0:
        raise ValueError("cfg.models was provided but is empty.")

    if cfg.active_model_name is None:
        model_name = next(iter(cfg.models.keys()))
    else:
        model_name = cfg.active_model_name

    if model_name not in cfg.models:
        raise ValueError(
            f"active_model_name={model_name!r} not found in cfg.models. "
            f"Available models: {list(cfg.models.keys())}"
        )

    return model_name, cfg.models[model_name]


def _fit_cluster_labels(
    X: np.ndarray,
    *,
    cfg: ClinicalResponseGAFSConfig,
) -> Tuple[np.ndarray, np.ndarray, Any, StandardScaler, str]:
    """
    Standardize X, fit the configured clustering model, and return labels.

    The selected feature matrix is always standardized before clustering. This is
    separate from optional clinical-column scaling.

    Supported model behavior
    ------------------------
    The model may implement:
        - fit_predict(X_scaled)
        - or fit(X_scaled) with labels_
        - or fit(X_scaled) followed by predict(X_scaled)

    Parameters
    ----------
    X:
        Selected feature matrix.

    cfg:
        Clinical-response GA configuration containing the clustering model settings.

    Returns
    -------
    tuple
        (labels, X_scaled, fitted_model, fitted_scaler, model_name)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_name, model_template = _get_active_cluster_model(cfg)
    model = deepcopy(model_template)

    if hasattr(model, "fit_predict"):
        labels = model.fit_predict(X_scaled)
    else:
        model.fit(X_scaled)
        if hasattr(model, "labels_"):
            labels = model.labels_
        elif hasattr(model, "predict"):
            labels = model.predict(X_scaled)
        else:
            raise ValueError(
                f"Clustering model {model_name!r} must implement fit_predict, "
                "or fit with labels_, or fit plus predict."
            )

    labels = np.asarray(labels)

    if labels.shape[0] != X.shape[0]:
        raise ValueError(
            f"Clustering model {model_name!r} returned {labels.shape[0]} labels "
            f"for {X.shape[0]} rows."
        )

    return labels, X_scaled, model, scaler, model_name


def compute_cluster_quality(X_scaled: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute cluster quality metrics.

    Currently this returns silhouette only because it is the metric used by the
    teammate-style fitness.

    Parameters
    ----------
    X_scaled:
        Scaled selected feature matrix.

    labels:
        Cluster labels.

    Returns
    -------
    dict
        {"silhouette": value}
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return {"silhouette": np.nan}

    counts = pd.Series(labels).value_counts()
    if (counts < 2).any():
        return {"silhouette": np.nan}

    try:
        sil = float(silhouette_score(X_scaled, labels))
    except Exception:
        sil = np.nan

    return {"silhouette": sil}


def compute_bootstrap_stability(
    X: np.ndarray,
    ref_labels: np.ndarray,
    *,
    cfg: ClinicalResponseGAFSConfig,
) -> Dict[str, float]:
    """
    Compute bootstrap clustering stability using ARI.

    For each bootstrap run:
        1. Resample rows with replacement.
        2. Fit scaler and KMeans on the bootstrap sample.
        3. Predict cluster labels for all original rows.
        4. Compare bootstrap labels to reference labels using ARI.

    ARI is label-permutation invariant, so cluster IDs do not need to match.

    Parameters
    ----------
    X:
        Selected feature matrix.

    ref_labels:
        Reference clustering labels from the full data.

    cfg:
        Clinical-response GA configuration containing bootstrap and model settings.

    Returns
    -------
    dict
        {"ari_mean": mean ARI, "ari_sd": standard deviation of ARI}
    """
    rng = np.random.default_rng(cfg.bootstrap_random_seed)
    n_samples = X.shape[0]

    if cfg.n_bootstrap <= 0:
        return {"ari_mean": np.nan, "ari_sd": np.nan}

    ari_values: List[float] = []

    for bootstrap_idx in range(cfg.n_bootstrap):
        sample_idx = rng.integers(0, n_samples, size=n_samples)
        X_boot = X[sample_idx]

        try:
            scaler_b = StandardScaler()
            X_boot_scaled = scaler_b.fit_transform(X_boot)

            model_name, model_template = _get_active_cluster_model(cfg)
            model_b = deepcopy(model_template)

            # If the model exposes random_state, perturb it across bootstraps.
            if hasattr(model_b, "random_state"):
                try:
                    setattr(model_b, "random_state", cfg.bootstrap_random_seed + bootstrap_idx)
                except Exception:
                    pass

            if hasattr(model_b, "fit"):
                model_b.fit(X_boot_scaled)
            else:
                raise ValueError(f"Clustering model {model_name!r} does not implement fit().")

            X_all_scaled = scaler_b.transform(X)

            if hasattr(model_b, "predict"):
                predicted_labels = model_b.predict(X_all_scaled)
            elif hasattr(model_b, "labels_"):
                # Models without predict cannot assign all original rows unless the bootstrap
                # sample equals the original data. Mark this bootstrap as invalid.
                predicted_labels = np.full(X.shape[0], np.nan)
            else:
                predicted_labels = np.full(X.shape[0], np.nan)

            if pd.Series(predicted_labels).isna().any():
                ari_values.append(np.nan)
            else:
                ari_values.append(float(adjusted_rand_score(ref_labels, predicted_labels)))
        except Exception:
            ari_values.append(np.nan)

    valid = pd.Series(ari_values, dtype=float).dropna()

    if len(valid) == 0:
        return {"ari_mean": np.nan, "ari_sd": np.nan}

    return {
        "ari_mean": float(valid.mean()),
        "ari_sd": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
    }


# =============================================================================
# Clinical treatment-effect helpers
# =============================================================================

def compute_ancova_adjusted_effect(
    df_cluster: pd.DataFrame,
    *,
    treatment_col: str,
    baseline_col: str,
    followup_col: str,
    age_col: Optional[str] = None,
    treated_label: str = "Treatment",
    control_label: str = "Placebo",
    min_arm_n: int = 3,
) -> Dict[str, Any]:
    """
    Estimate adjusted treatment effect inside one cluster.

    Model
    -----
    If age is unavailable or insufficient:

        followup ~ treatment_binary + baseline

    If age is available and sufficiently populated:

        followup ~ treatment_binary + baseline + age

    Parameters
    ----------
    df_cluster:
        Dataframe containing one discovered cluster.

    treatment_col:
        Treatment assignment column.

    baseline_col:
        Baseline outcome column.

    followup_col:
        Follow-up outcome column.

    age_col:
        Optional age column.

    treated_label:
        Standardized treatment label.

    control_label:
        Standardized control/placebo label.

    min_arm_n:
        Minimum number of treated and control subjects required.

    Returns
    -------
    dict
        Treatment effect information:
            beta, se, p_value, confidence interval, cluster arm counts, model formula.
    """
    import statsmodels.api as sm

    needed_cols = [treatment_col, baseline_col, followup_col]
    if age_col is not None and age_col in df_cluster.columns:
        needed_cols.append(age_col)

    d = df_cluster[needed_cols].copy()
    d[treatment_col] = standardize_treatment_arm(
        d[treatment_col],
        treated_label=treated_label,
        control_label=control_label,
    )

    d[baseline_col] = pd.to_numeric(d[baseline_col], errors="coerce")
    d[followup_col] = pd.to_numeric(d[followup_col], errors="coerce")

    if age_col is not None and age_col in d.columns:
        d[age_col] = pd.to_numeric(d[age_col], errors="coerce")

    d = d.dropna(subset=[treatment_col, baseline_col, followup_col]).copy()

    n_total = int(len(d))
    n_treated = int((d[treatment_col] == treated_label).sum())
    n_control = int((d[treatment_col] == control_label).sum())

    result: Dict[str, Any] = {
        "n_total": n_total,
        "n_treated": n_treated,
        "n_control": n_control,
        "beta": np.nan,
        "se": np.nan,
        "p_value": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "model_formula": None,
        "used_age": False,
    }

    if n_treated < min_arm_n or n_control < min_arm_n:
        return result

    if d[followup_col].nunique(dropna=True) < 2:
        return result

    d["treatment_binary"] = (d[treatment_col] == treated_label).astype(float)

    y = d[followup_col].astype(float)
    X_cols: Dict[str, pd.Series] = {
        "treatment_binary": d["treatment_binary"].astype(float),
        "baseline": d[baseline_col].astype(float),
    }

    used_age = False

    if age_col is not None and age_col in d.columns:
        d_with_age = d.dropna(subset=[age_col]).copy()

        enough_age_rows = len(d_with_age) >= max(8, min_arm_n * 2)
        enough_treated = (d_with_age[treatment_col] == treated_label).sum() >= min_arm_n
        enough_control = (d_with_age[treatment_col] == control_label).sum() >= min_arm_n

        if enough_age_rows and enough_treated and enough_control:
            y = d_with_age[followup_col].astype(float)
            X_cols = {
                "treatment_binary": d_with_age["treatment_binary"].astype(float),
                "baseline": d_with_age[baseline_col].astype(float),
                "age": d_with_age[age_col].astype(float),
            }
            used_age = True

    X_design = pd.DataFrame(X_cols)
    X_design = sm.add_constant(X_design, has_constant="add")

    try:
        model = sm.OLS(y, X_design).fit()

        beta = float(model.params["treatment_binary"])
        se = float(model.bse["treatment_binary"])
        p_value = float(model.pvalues["treatment_binary"])
        ci = model.conf_int().loc["treatment_binary"]

        result.update(
            {
                "beta": beta,
                "se": se,
                "p_value": p_value,
                "ci_low": float(ci[0]),
                "ci_high": float(ci[1]),
                "model_formula": (
                    f"{followup_col} ~ treatment + baseline + age"
                    if used_age
                    else f"{followup_col} ~ treatment + baseline"
                ),
                "used_age": used_age,
            }
        )
    except Exception:
        # Return NaN effect values if the model fails for this cluster.
        pass

    return result


def compute_subtype_treatment_effects(
    df: pd.DataFrame,
    labels: np.ndarray,
    *,
    treatment_col: str,
    baseline_col: str,
    followup_col: str,
    age_col: Optional[str],
    cfg: ClinicalResponseGAFSConfig,
    treated_label: str = "Treatment",
    control_label: str = "Placebo",
) -> pd.DataFrame:
    """
    Compute treatment effects separately inside each discovered cluster.

    Parameters
    ----------
    df:
        Dataframe used for clustering and clinical scoring.

    labels:
        KMeans cluster labels aligned to df rows.

    treatment_col:
        Treatment assignment column.

    baseline_col:
        Baseline outcome column.

    followup_col:
        Follow-up outcome column.

    age_col:
        Optional age column.

    cfg:
        Clinical-response GA configuration.

    treated_label:
        Standardized treatment label.

    control_label:
        Standardized control/placebo label.

    Returns
    -------
    pd.DataFrame
        One row per cluster with treatment effect estimates and arm counts.
    """
    d = df.copy()
    d["_cluster"] = np.asarray(labels)

    rows: List[Dict[str, Any]] = []

    for cluster_label in sorted(pd.Series(labels).dropna().unique()):
        sub = d[d["_cluster"] == cluster_label].copy()

        effect = compute_ancova_adjusted_effect(
            sub,
            treatment_col=treatment_col,
            baseline_col=baseline_col,
            followup_col=followup_col,
            age_col=age_col,
            treated_label=treated_label,
            control_label=control_label,
            min_arm_n=cfg.min_arm_n_per_cluster,
        )
        effect["cluster"] = int(cluster_label)
        rows.append(effect)

    return pd.DataFrame(rows)



def compute_label_alignment(
    y_true: Sequence[Any],
    cluster_labels: Sequence[Any],
    *,
    metric: str = "ari_nmi",
) -> Dict[str, float]:
    """
    Compute agreement between discovered clusters and an external label column.

    Parameters
    ----------
    y_true:
        External labels, e.g. 0/1 class labels from synthetic data.

    cluster_labels:
        Cluster assignments from the clustering model.

    metric:
        Which label-alignment score to use:
            "ari"      : adjusted rand index only
            "nmi"      : normalized mutual information only
            "ari_nmi"  : average of clipped ARI and NMI

    Returns
    -------
    dict
        label_alignment_raw:
            The selected label alignment score.

        label_alignment_norm:
            Normalized score used by the fitness function.

        label_ari_raw:
            Adjusted Rand Index between y_true and cluster labels.

        label_nmi_raw:
            Normalized Mutual Information between y_true and cluster labels.
    """
    y = np.asarray(y_true)
    labels = np.asarray(cluster_labels)

    if y.shape[0] != labels.shape[0]:
        raise ValueError(
            f"y_true has {y.shape[0]} rows but cluster_labels has {labels.shape[0]} rows."
        )

    ari = float(adjusted_rand_score(y, labels))
    nmi = float(normalized_mutual_info_score(y, labels))

    # ARI can be negative. For fitness, use a 0..1 clipped version.
    ari_norm = float(np.clip(ari, 0.0, 1.0))
    nmi_norm = float(np.clip(nmi, 0.0, 1.0))

    metric_name = str(metric).lower()

    if metric_name == "ari":
        score = ari_norm
    elif metric_name == "nmi":
        score = nmi_norm
    elif metric_name == "ari_nmi":
        score = 0.5 * ari_norm + 0.5 * nmi_norm
    else:
        raise ValueError(
            f"Unknown label alignment metric={metric!r}. "
            "Use 'ari', 'nmi', or 'ari_nmi'."
        )

    return {
        "label_alignment_raw": float(score),
        "label_alignment_norm": float(score),
        "label_ari_raw": ari,
        "label_nmi_raw": nmi,
        "metric": metric_name,
    }


def compute_sclinical(
    effects_df: pd.DataFrame,
    *,
    cfg: ClinicalResponseGAFSConfig,
) -> Dict[str, float]:
    """
    Compute teammate-style clinical meaningfulness score.

    Formula
    -------
    Sclinical_raw =
        w1 * treatment-effect spread
      + w2 * opposite-direction bonus
      + w3 * average precision
      - w4 * small-cluster penalty
      - w5 * treatment-arm imbalance penalty

    where:
        spread = max(beta_k) - min(beta_k)
        opposite-direction bonus = 1 if at least one beta is positive and one negative
        precision = mean(abs(beta_k) / (SE_k + eps))

    The raw score is squashed to [0, 1]:

        Sclinical_norm = (tanh(Sclinical_raw / scale) + 1) / 2

    Parameters
    ----------
    effects_df:
        Cluster-specific treatment effects.

    cfg:
        Clinical-response GA configuration.

    Returns
    -------
    dict
        Raw score, normalized score, and components.
    """
    if effects_df is None or len(effects_df) == 0:
        return {
            "sclin_raw": np.nan,
            "sclin_norm": 0.0,
            "spread": np.nan,
            "opposite_sign_bonus": 0.0,
            "precision_mean": np.nan,
            "small_cluster_penalty": np.nan,
            "arm_imbalance_penalty": np.nan,
            "n_valid_effects": 0,
        }

    d = effects_df.copy()
    valid = d.dropna(subset=["beta", "se"]).copy()

    small_cluster_penalty = float((d["n_total"] < cfg.min_cluster_total_n).mean())

    imbalance_values: List[float] = []
    for _, row in d.iterrows():
        n_treated = row.get("n_treated", np.nan)
        n_control = row.get("n_control", np.nan)
        n_total = row.get("n_total", np.nan)

        if pd.isna(n_treated) or pd.isna(n_control) or pd.isna(n_total) or n_total <= 0:
            imbalance_values.append(1.0)
            continue

        # 0 means perfectly balanced; 1 means all one arm.
        imbalance = abs(float(n_treated) - float(n_control)) / max(float(n_total), cfg.eps)
        imbalance_values.append(imbalance)

    arm_imbalance_penalty = (
        float(np.nanmean(imbalance_values)) if len(imbalance_values) > 0 else 1.0
    )

    if len(valid) == 0:
        sclin_raw = (
            -cfg.sclin_w4_small_cluster_penalty * small_cluster_penalty
            -cfg.sclin_w5_arm_imbalance_penalty * arm_imbalance_penalty
        )
        sclin_norm = float((np.tanh(sclin_raw / cfg.sclin_scale) + 1.0) / 2.0)

        return {
            "sclin_raw": float(sclin_raw),
            "sclin_norm": sclin_norm,
            "spread": np.nan,
            "opposite_sign_bonus": 0.0,
            "precision_mean": np.nan,
            "small_cluster_penalty": small_cluster_penalty,
            "arm_imbalance_penalty": arm_imbalance_penalty,
            "n_valid_effects": 0,
        }

    betas = valid["beta"].to_numpy(dtype=float)
    ses = valid["se"].to_numpy(dtype=float)

    spread = float(np.nanmax(betas) - np.nanmin(betas))
    opposite_sign_bonus = float((np.nanmax(betas) > 0.0) and (np.nanmin(betas) < 0.0))
    precision_mean = float(np.nanmean(np.abs(betas) / (ses + cfg.eps)))

    sclin_raw = (
        cfg.sclin_w1_spread * spread
        + cfg.sclin_w2_opposite_sign_bonus * opposite_sign_bonus
        + cfg.sclin_w3_precision * precision_mean
        - cfg.sclin_w4_small_cluster_penalty * small_cluster_penalty
        - cfg.sclin_w5_arm_imbalance_penalty * arm_imbalance_penalty
    )

    sclin_norm = float((np.tanh(sclin_raw / cfg.sclin_scale) + 1.0) / 2.0)

    return {
        "sclin_raw": float(sclin_raw),
        "sclin_norm": sclin_norm,
        "spread": spread,
        "opposite_sign_bonus": opposite_sign_bonus,
        "precision_mean": precision_mean,
        "small_cluster_penalty": small_cluster_penalty,
        "arm_imbalance_penalty": arm_imbalance_penalty,
        "n_valid_effects": int(len(valid)),
    }




# =============================================================================
# Dynamic fitness helpers
# =============================================================================


def get_active_fitness_preset_config(
    cfg: ClinicalResponseGAFSConfig,
) -> Dict[str, Any]:
    """
    Return the nested config for the active fitness preset.

    This is the main Stage 2 organization point:

        fitness_preset
            selects the standalone fitness function

        fitness_preset_config[fitness_preset]
            stores all parameters for that function

    If no nested config is provided, an empty dictionary is returned and the
    code falls back to legacy/default config fields.
    """
    preset_name = cfg.fitness_preset or "balanced_clinical"

    if cfg.fitness_preset_config is None:
        return {}

    if preset_name not in cfg.fitness_preset_config:
        raise ValueError(
            f"fitness_preset={preset_name!r} but cfg.fitness_preset_config "
            f"does not contain that key. Available keys: "
            f"{list(cfg.fitness_preset_config.keys())}"
        )

    return dict(cfg.fitness_preset_config[preset_name] or {})


def apply_active_fitness_preset_config(
    cfg: ClinicalResponseGAFSConfig,
) -> None:
    """
    Apply the active nested fitness preset configuration to cfg fields.

    This keeps downstream code simple while letting the notebook organize all
    preset-specific parameters inside cfg.fitness_preset_config.

    Supported nested sections
    -------------------------
    "sclinical":
        w1_spread, w2_opposite_sign_bonus, w3_precision,
        w4_small_cluster_penalty, w5_arm_imbalance_penalty, scale

    "feature_selection":
        min_features, max_features, feature_fraction_penalty_power

    "clinical_effect":
        min_cluster_total_n, min_arm_n_per_cluster
    """
    preset_cfg = get_active_fitness_preset_config(cfg)

    sclinical = dict(preset_cfg.get("sclinical", {}) or {})
    cfg.sclin_w1_spread = sclinical.get("w1_spread", cfg.sclin_w1_spread)
    cfg.sclin_w2_opposite_sign_bonus = sclinical.get(
        "w2_opposite_sign_bonus",
        cfg.sclin_w2_opposite_sign_bonus,
    )
    cfg.sclin_w3_precision = sclinical.get("w3_precision", cfg.sclin_w3_precision)
    cfg.sclin_w4_small_cluster_penalty = sclinical.get(
        "w4_small_cluster_penalty",
        cfg.sclin_w4_small_cluster_penalty,
    )
    cfg.sclin_w5_arm_imbalance_penalty = sclinical.get(
        "w5_arm_imbalance_penalty",
        cfg.sclin_w5_arm_imbalance_penalty,
    )
    cfg.sclin_scale = sclinical.get("scale", cfg.sclin_scale)

    feature_selection = dict(preset_cfg.get("feature_selection", {}) or {})
    cfg.min_features = feature_selection.get("min_features", cfg.min_features)
    cfg.max_features = feature_selection.get("max_features", cfg.max_features)
    cfg.feature_fraction_penalty_power = feature_selection.get(
        "feature_fraction_penalty_power",
        cfg.feature_fraction_penalty_power,
    )

    clinical_effect = dict(preset_cfg.get("clinical_effect", {}) or {})
    cfg.min_cluster_total_n = clinical_effect.get(
        "min_cluster_total_n",
        cfg.min_cluster_total_n,
    )
    cfg.min_arm_n_per_cluster = clinical_effect.get(
        "min_arm_n_per_cluster",
        cfg.min_arm_n_per_cluster,
    )



# =============================================================================
# Fitness metric alias helpers
# =============================================================================

FITNESS_METRIC_ALIASES: Dict[str, str] = {
    # User-facing readable names -> internal metric names
    "bootstrap_ari_norm": "stability_norm",
    "bootstrap_ari_raw": "stability_raw",
    "silhouette_norm": "cluster_norm",
    "silhouette_raw": "cluster_raw",
    "sclinical_norm": "sclin_norm",
    "sclinical_raw": "sclin_raw",
    "label_alignment_norm": "label_alignment_norm",
    "label_alignment_raw": "label_alignment_raw",
    "label_ari_raw": "label_ari_raw",
    "label_nmi_raw": "label_nmi_raw",
}


def canonical_metric_name(metric_name: str) -> str:
    """
    Convert a user-facing metric alias into the internal metric name.

    Examples
    --------
    bootstrap_ari_norm -> stability_norm
    silhouette_norm    -> cluster_norm
    sclinical_norm     -> sclin_norm
    """
    return FITNESS_METRIC_ALIASES.get(metric_name, metric_name)


def canonicalize_fitness_components(
    components: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Convert user-facing fitness component names into internal metric names.

    This allows notebook configs to use clearer names such as:
        bootstrap_ari_norm
        silhouette_norm
        sclinical_norm

    while the evaluator continues using internal names:
        stability_norm
        cluster_norm
        sclin_norm
    """
    canonical: Dict[str, Dict[str, Any]] = {}

    for metric_name, spec in components.items():
        internal_name = canonical_metric_name(metric_name)

        if internal_name in canonical:
            raise ValueError(
                f"Duplicate fitness metric after alias conversion: {metric_name!r} "
                f"maps to {internal_name!r}, which is already present."
            )

        canonical[internal_name] = dict(spec)

        # Keep the original user-facing name for traceability.
        if internal_name != metric_name:
            canonical[internal_name]["alias"] = metric_name

    return canonical


def display_metric_name_from_component(
    internal_metric_name: str,
    component_spec: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Return the user-facing metric name for a component.

    If the component came from an alias, the alias is stored in component_spec["alias"].
    Otherwise the internal metric name is returned.
    """
    if component_spec is not None and component_spec.get("alias"):
        return str(component_spec["alias"])
    return internal_metric_name


def get_active_metric_display_config(
    cfg: ClinicalResponseGAFSConfig,
) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    """
    Resolve which metrics should appear in generation logs and GA history plots.

    Returns
    -------
    Tuple[List[Tuple[str, str]], Dict[str, str]]
        metric_pairs:
            List of (user_facing_name, internal_metric_name).

        display_names:
            Mapping from user-facing names to short display labels.

    Behavior
    --------
    1. If cfg.logging_config["metrics_to_show"] is provided, use that list.
    2. Otherwise, use the active fitness_components for the current preset.
    3. Aliases are supported, e.g. bootstrap_ari_norm -> stability_norm.
    """
    logging_cfg = dict(cfg.logging_config or {})
    display_names = dict(logging_cfg.get("metric_display_names", {}) or {})

    metrics_to_show = logging_cfg.get("metrics_to_show", None)

    if metrics_to_show is not None:
        metric_pairs = [
            (str(metric_name), canonical_metric_name(str(metric_name)))
            for metric_name in metrics_to_show
        ]
        return metric_pairs, display_names

    components = default_fitness_components(cfg)

    metric_pairs: List[Tuple[str, str]] = []
    for internal_name, spec in components.items():
        user_name = display_metric_name_from_component(internal_name, spec)
        metric_pairs.append((user_name, internal_name))

    return metric_pairs, display_names


def default_metric_display_label(metric_name: str) -> str:
    """
    Provide concise labels for common metric names.
    """
    labels = {
        "bootstrap_ari_norm": "ARI",
        "bootstrap_ari_raw": "ARI_raw",
        "silhouette_norm": "sil",
        "silhouette_raw": "sil_raw",
        "sclinical_norm": "Sclin",
        "sclinical_raw": "Sclin_raw",
        "feature_penalty_norm": "feat_pen",
        "label_alignment_norm": "label_align",
        "label_alignment_raw": "label_align_raw",
        "label_ari_raw": "label_ARI",
        "label_nmi_raw": "label_NMI",
        "stability_norm": "ARI",
        "stability_raw": "ARI_raw",
        "cluster_norm": "sil",
        "cluster_raw": "sil_raw",
        "sclin_norm": "Sclin",
        "sclin_raw": "Sclin_raw",
        "best_fitness": "Fitness",
    }
    return labels.get(metric_name, metric_name)



def get_fitness_preset_components(
    preset_name: str,
    *,
    cfg: ClinicalResponseGAFSConfig,
) -> Dict[str, Dict[str, Any]]:
    """
    Return default component weights for a named Stage 2 fitness function.

    The preset name chooses the standalone fitness function. This component
    dictionary supplies that function's default metric weights.

    Parameters
    ----------
    preset_name:
        Name of the preset/function.

    cfg:
        Clinical-response GA configuration. Used for legacy/default weights.

    Returns
    -------
    dict
        Fitness component configuration.

    Raises
    ------
    ValueError
        If the preset name is unknown.
    """
    name = str(preset_name).lower()

    presets: Dict[str, Dict[str, Dict[str, Any]]] = {
        "balanced_clinical": {
            "stability_norm": {
                "weight": cfg.ws,
                "direction": "maximize",
                "description": "Bootstrap clustering stability",
            },
            "sclin_norm": {
                "weight": cfg.wc,
                "direction": "maximize",
                "description": "Clinical treatment-response meaningfulness",
            },
            "cluster_norm": {
                "weight": cfg.wq,
                "direction": "maximize",
                "description": "Cluster quality from silhouette",
            },
            "feature_penalty_norm": {
                "weight": cfg.wf,
                "direction": "minimize",
                "description": "Feature-count penalty",
            },
        },
        "unsupervised_clustering": {
            "stability_norm": {
                "weight": 0.60,
                "direction": "maximize",
                "description": "Bootstrap clustering stability",
            },
            "cluster_norm": {
                "weight": 0.30,
                "direction": "maximize",
                "description": "Cluster quality from silhouette",
            },
            "feature_penalty_norm": {
                "weight": 0.10,
                "direction": "minimize",
                "description": "Feature-count penalty",
            },
        },
        "label_guided_clustering": {
            "stability_norm": {
                "weight": 0.25,
                "direction": "maximize",
                "description": "Bootstrap clustering stability",
            },
            "cluster_norm": {
                "weight": 0.20,
                "direction": "maximize",
                "description": "Cluster quality from silhouette",
            },
            "label_alignment_norm": {
                "weight": 0.45,
                "direction": "maximize",
                "description": "Agreement between cluster labels and provided label column",
            },
            "feature_penalty_norm": {
                "weight": 0.10,
                "direction": "minimize",
                "description": "Feature-count penalty",
            },
        },
        "stability_only": {
            "stability_norm": {
                "weight": 1.00,
                "direction": "maximize",
                "description": "Bootstrap clustering stability",
            },
            "feature_penalty_norm": {
                "weight": cfg.wf,
                "direction": "minimize",
                "description": "Feature-count penalty",
            },
        },
        "cluster_quality_only": {
            "cluster_norm": {
                "weight": 1.00,
                "direction": "maximize",
                "description": "Cluster quality from silhouette",
            },
            "feature_penalty_norm": {
                "weight": cfg.wf,
                "direction": "minimize",
                "description": "Feature-count penalty",
            },
        },
    }

    if name not in presets:
        raise ValueError(
            f"Unknown fitness_preset={preset_name!r}. "
            f"Available presets/functions: {sorted(presets.keys())}"
        )

    return presets[name]

def default_fitness_components(cfg: ClinicalResponseGAFSConfig) -> Dict[str, Dict[str, Any]]:
    """
    Build the default fitness component dictionary.

    Resolution order
    ----------------
    1. If cfg.fitness_components is provided, it is used directly elsewhere.
    2. Otherwise, cfg.fitness_preset is used.
    3. If cfg.fitness_preset is None, "balanced_clinical" is used.

    The default "balanced_clinical" preset preserves the original teammate-style
    score:

        fitness =
            ws * stability_norm
          + wc * sclin_norm
          + wq * cluster_norm
          - wf * feature_penalty_norm

    Parameters
    ----------
    cfg:
        Clinical-response GA configuration.

    Returns
    -------
    dict
        Fitness component configuration.
    """
    preset_cfg = get_active_fitness_preset_config(cfg)

    if "fitness_components" in preset_cfg and preset_cfg["fitness_components"] is not None:
        return canonicalize_fitness_components(preset_cfg["fitness_components"])

    # Backward compatibility: explicit top-level fitness_components still works.
    if cfg.fitness_components is not None:
        return canonicalize_fitness_components(cfg.fitness_components)

    preset_name = cfg.fitness_preset or "balanced_clinical"
    return canonicalize_fitness_components(get_fitness_preset_components(preset_name, cfg=cfg))

def _metric_value(metrics: Mapping[str, float], metric_name: str) -> float:
    """
    Safely read a metric value, treating missing/NaN values as 0.0.
    """
    value = metrics.get(metric_name, 0.0)
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _component_weight(
    components: Mapping[str, Mapping[str, Any]],
    metric_name: str,
    *,
    default: float = 0.0,
) -> float:
    """
    Read a component weight. Missing components default to 0.0 unless overridden.
    """
    if metric_name not in components:
        return float(default)
    return float(components[metric_name].get("weight", default))


def balanced_clinical_fitness(
    metrics: Mapping[str, float],
    components: Mapping[str, Mapping[str, Any]],
) -> Tuple[float, Dict[str, float]]:
    """
    Standalone balanced clinical-response fitness function.

    Calculation
    -----------
    fitness =
        stability_weight * stability_norm
      + sclinical_weight * sclin_norm
      + cluster_weight * cluster_norm
      - feature_penalty_weight * feature_penalty_norm

    The function is standalone so the calculation can be changed later without
    changing other fitness functions.
    """
    stability = _metric_value(metrics, "stability_norm")
    sclinical = _metric_value(metrics, "sclin_norm")
    cluster_quality = _metric_value(metrics, "cluster_norm")
    feature_penalty = _metric_value(metrics, "feature_penalty_norm")

    w_stability = _component_weight(components, "stability_norm")
    w_sclinical = _component_weight(components, "sclin_norm")
    w_cluster = _component_weight(components, "cluster_norm")
    w_penalty = _component_weight(components, "feature_penalty_norm")

    contributions = {
        "stability_norm": w_stability * stability,
        "sclin_norm": w_sclinical * sclinical,
        "cluster_norm": w_cluster * cluster_quality,
        "feature_penalty_norm": -w_penalty * feature_penalty,
    }

    fitness = sum(contributions.values())
    return float(fitness), {k: float(v) for k, v in contributions.items()}



def unsupervised_clustering_fitness(
    metrics: Mapping[str, float],
    components: Mapping[str, Mapping[str, Any]],
) -> Tuple[float, Dict[str, float]]:
    """
    Standalone unsupervised clustering fitness function.

    Calculation
    -----------
    fitness =
        stability_weight * stability_norm
      + cluster_quality_weight * cluster_norm
      - feature_penalty_weight * feature_penalty_norm

    This function intentionally ignores Sclinical and does not require clinical
    treatment/outcome columns.
    """
    stability = _metric_value(metrics, "stability_norm")
    cluster_quality = _metric_value(metrics, "cluster_norm")
    feature_penalty = _metric_value(metrics, "feature_penalty_norm")

    w_stability = _component_weight(components, "stability_norm")
    w_cluster = _component_weight(components, "cluster_norm")
    w_penalty = _component_weight(components, "feature_penalty_norm")

    contributions = {
        "stability_norm": w_stability * stability,
        "cluster_norm": w_cluster * cluster_quality,
        "feature_penalty_norm": -w_penalty * feature_penalty,
    }

    fitness = sum(contributions.values())
    return float(fitness), {k: float(v) for k, v in contributions.items()}



def label_guided_clustering_fitness(
    metrics: Mapping[str, float],
    components: Mapping[str, Mapping[str, Any]],
) -> Tuple[float, Dict[str, float]]:
    """
    Standalone label-guided clustering fitness function.

    Calculation
    -----------
    fitness =
        stability_weight * stability_norm
      + cluster_quality_weight * cluster_norm
      + label_alignment_weight * label_alignment_norm
      - feature_penalty_weight * feature_penalty_norm

    This function uses a label column to guide feature selection for clustering.
    The clustering model still only sees the selected clustering features; the
    label is used after clustering to score cluster-label agreement.
    """
    stability = _metric_value(metrics, "stability_norm")
    cluster_quality = _metric_value(metrics, "cluster_norm")
    label_alignment = _metric_value(metrics, "label_alignment_norm")
    feature_penalty = _metric_value(metrics, "feature_penalty_norm")

    w_stability = _component_weight(components, "stability_norm")
    w_cluster = _component_weight(components, "cluster_norm")
    w_label = _component_weight(components, "label_alignment_norm")
    w_penalty = _component_weight(components, "feature_penalty_norm")

    contributions = {
        "stability_norm": w_stability * stability,
        "cluster_norm": w_cluster * cluster_quality,
        "label_alignment_norm": w_label * label_alignment,
        "feature_penalty_norm": -w_penalty * feature_penalty,
    }

    fitness = sum(contributions.values())
    return float(fitness), {k: float(v) for k, v in contributions.items()}


def stability_only_fitness(
    metrics: Mapping[str, float],
    components: Mapping[str, Mapping[str, Any]],
) -> Tuple[float, Dict[str, float]]:
    """
    Standalone stability-only fitness function.

    Calculation
    -----------
    fitness =
        stability_weight * stability_norm
      - feature_penalty_weight * feature_penalty_norm

    This function intentionally ignores Sclinical and cluster quality.
    """
    stability = _metric_value(metrics, "stability_norm")
    feature_penalty = _metric_value(metrics, "feature_penalty_norm")

    w_stability = _component_weight(components, "stability_norm")
    w_penalty = _component_weight(components, "feature_penalty_norm")

    contributions = {
        "stability_norm": w_stability * stability,
        "feature_penalty_norm": -w_penalty * feature_penalty,
    }

    fitness = sum(contributions.values())
    return float(fitness), {k: float(v) for k, v in contributions.items()}


def cluster_quality_only_fitness(
    metrics: Mapping[str, float],
    components: Mapping[str, Mapping[str, Any]],
) -> Tuple[float, Dict[str, float]]:
    """
    Standalone cluster-quality-only fitness function.

    Calculation
    -----------
    fitness =
        cluster_quality_weight * cluster_norm
      - feature_penalty_weight * feature_penalty_norm

    This function intentionally ignores Sclinical and bootstrap stability.
    """
    cluster_quality = _metric_value(metrics, "cluster_norm")
    feature_penalty = _metric_value(metrics, "feature_penalty_norm")

    w_cluster = _component_weight(components, "cluster_norm")
    w_penalty = _component_weight(components, "feature_penalty_norm")

    contributions = {
        "cluster_norm": w_cluster * cluster_quality,
        "feature_penalty_norm": -w_penalty * feature_penalty,
    }

    fitness = sum(contributions.values())
    return float(fitness), {k: float(v) for k, v in contributions.items()}


def get_named_fitness_function(
    preset_name: str,
) -> Callable[[Mapping[str, float], Mapping[str, Mapping[str, Any]]], Tuple[float, Dict[str, float]]]:
    """
    Return the standalone fitness function associated with a preset name.

    Parameters
    ----------
    preset_name:
        Name of the built-in fitness preset/function.

    Returns
    -------
    Callable
        Function with signature:
            fitness_function(metrics, components) -> (fitness, contributions)
    """
    registry: Dict[
        str,
        Callable[[Mapping[str, float], Mapping[str, Mapping[str, Any]]], Tuple[float, Dict[str, float]]],
    ] = {
        "balanced_clinical": balanced_clinical_fitness,
        "unsupervised_clustering": unsupervised_clustering_fitness,
        "label_guided_clustering": label_guided_clustering_fitness,
        "stability_only": stability_only_fitness,
        "cluster_quality_only": cluster_quality_only_fitness,
    }

    name = str(preset_name).lower()

    if name not in registry:
        raise ValueError(
            f"Unknown fitness_preset={preset_name!r}. "
            f"Available presets/functions: {sorted(registry.keys())}"
        )

    return registry[name]

def compute_dynamic_fitness(
    metrics: Mapping[str, float],
    *,
    cfg: ClinicalResponseGAFSConfig,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute final fitness using the configured named fitness function.

    Resolution order
    ----------------
    1. If cfg.custom_fitness_function is provided, use it.
    2. Otherwise, use cfg.fitness_preset to select one standalone built-in
       fitness function.
    3. Use cfg.fitness_components as that function's weights if provided.
       Otherwise, use the preset's default component weights.

    Parameters
    ----------
    metrics:
        Dictionary of computed metric values for one feature subset.

    cfg:
        Clinical-response GA configuration.

    Returns
    -------
    Tuple[float, dict]
        fitness:
            Final fitness value used by the GA.

        fitness_details:
            Metadata explaining how the fitness was computed.
    """
    # ------------------------------------------------------------
    # User-provided custom function
    # ------------------------------------------------------------
    if cfg.custom_fitness_function is not None:
        fitness = float(cfg.custom_fitness_function(dict(metrics)))
        return fitness, {
            "fitness_function_name": "custom",
            "fitness_preset": None,
            "fitness_components": None,
            "fitness_contributions": {"custom_fitness_function": fitness},
        }

    # ------------------------------------------------------------
    # Built-in standalone named fitness functions
    # ------------------------------------------------------------
    preset_name = cfg.fitness_preset or "balanced_clinical"

    components = default_fitness_components(cfg)

    fitness_function = get_named_fitness_function(preset_name)
    fitness, contributions = fitness_function(metrics, components)

    return fitness, {
        "fitness_function_name": preset_name,
        "fitness_preset": preset_name,
        "fitness_components": components,
        "fitness_contributions": contributions,
    }


def active_fitness_components(cfg: ClinicalResponseGAFSConfig) -> Dict[str, Dict[str, Any]]:
    """
    Return the active fitness components after applying preset config/defaults.
    """
    return default_fitness_components(cfg)



def active_fitness_uses_label_alignment(cfg: ClinicalResponseGAFSConfig) -> bool:
    """
    Return whether active fitness components use label alignment.
    """
    components = active_fitness_components(cfg)
    return bool(components.get("label_alignment_norm", {}).get("weight", 0.0) != 0)


def resolve_label_scoring_config(
    cfg: ClinicalResponseGAFSConfig,
) -> Dict[str, Any]:
    """
    Resolve label-scoring configuration for label-guided presets.

    Preferred location:
        cfg.fitness_preset_config[cfg.fitness_preset]["scoring_columns"]["label_col"]

    Optional settings:
        cfg.fitness_preset_config[cfg.fitness_preset]["label_alignment"]["metric"]
    """
    preset_name = cfg.fitness_preset or "balanced_clinical"
    preset_cfg = get_active_fitness_preset_config(cfg)
    scoring_columns = dict(preset_cfg.get("scoring_columns", {}) or {})
    label_cfg = dict(preset_cfg.get("label_alignment", {}) or {})

    label_col = scoring_columns.get("label_col")

    if not label_col:
        raise ValueError(
            f"fitness_preset={preset_name!r} uses label_alignment_norm but no label_col "
            f"was provided. Put it under "
            f"fitness_preset_config[{preset_name!r}]['scoring_columns']['label_col']."
        )

    return {
        "label_col": label_col,
        "metric": label_cfg.get("metric", "ari_nmi"),
    }


def active_fitness_uses_sclinical(cfg: ClinicalResponseGAFSConfig) -> bool:
    """
    Return whether active fitness components use Sclinical.
    """
    components = active_fitness_components(cfg)
    return bool(components.get("sclin_norm", {}).get("weight", 0.0) != 0)


def active_fitness_uses_cluster_quality(cfg: ClinicalResponseGAFSConfig) -> bool:
    """
    Return whether active fitness components use cluster quality.
    """
    components = active_fitness_components(cfg)
    return bool(components.get("cluster_norm", {}).get("weight", 0.0) != 0)


def active_fitness_uses_stability(cfg: ClinicalResponseGAFSConfig) -> bool:
    """
    Return whether active fitness components use bootstrap stability.
    """
    components = active_fitness_components(cfg)
    return bool(components.get("stability_norm", {}).get("weight", 0.0) != 0)




# =============================================================================
# Longitudinal clinical-score-guided clustering helpers
# =============================================================================

def _compute_clinical_score_eta_squared(
    clusters: Sequence[Any],
    scores: Sequence[Any],
    *,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Compute ANOVA-style eta-squared for one clinical score at one timepoint.

    This measures how much of the clinical-score variance is explained by
    discovered cluster membership.

    Formula
    -------
    eta_squared = between_cluster_sum_of_squares / total_sum_of_squares

    Interpretation
    --------------
    0.0:
        Cluster membership explains none of the clinical score variation.

    1.0:
        Cluster membership perfectly explains the clinical score variation.

    Notes
    -----
    The clinical score is not used as a clustering feature. It is used only
    after clustering to score whether discovered clusters separate clinical
    severity/functioning.
    """
    cluster_s = pd.Series(clusters, name="cluster").reset_index(drop=True)
    score_s = pd.to_numeric(pd.Series(scores, name="clinical_score"), errors="coerce").reset_index(drop=True)

    if len(cluster_s) != len(score_s):
        raise ValueError(
            f"clusters and scores must have the same length. "
            f"Got len(clusters)={len(cluster_s)} and len(scores)={len(score_s)}."
        )

    d = pd.DataFrame({
        "cluster": cluster_s,
        "clinical_score": score_s,
    }).dropna(subset=["cluster", "clinical_score"])

    n = int(len(d))
    if n < 3:
        return {
            "eta_squared": np.nan,
            "n": n,
            "n_clusters": 0,
            "overall_mean": np.nan,
            "total_ss": np.nan,
            "between_ss": np.nan,
            "cluster_summary": pd.DataFrame(),
        }

    cluster_counts = d["cluster"].value_counts(dropna=False)
    valid_clusters = cluster_counts.index.tolist()
    n_clusters = int(len(valid_clusters))

    if n_clusters < 2:
        return {
            "eta_squared": 0.0,
            "n": n,
            "n_clusters": n_clusters,
            "overall_mean": float(d["clinical_score"].mean()),
            "total_ss": 0.0,
            "between_ss": 0.0,
            "cluster_summary": (
                d.groupby("cluster", dropna=False)["clinical_score"]
                .agg(n="size", mean="mean", sd="std", median="median")
                .reset_index()
            ),
        }

    overall_mean = float(d["clinical_score"].mean())

    total_ss = float(((d["clinical_score"] - overall_mean) ** 2).sum())
    if total_ss <= eps:
        return {
            "eta_squared": 0.0,
            "n": n,
            "n_clusters": n_clusters,
            "overall_mean": overall_mean,
            "total_ss": total_ss,
            "between_ss": 0.0,
            "cluster_summary": (
                d.groupby("cluster", dropna=False)["clinical_score"]
                .agg(n="size", mean="mean", sd="std", median="median")
                .reset_index()
            ),
        }

    cluster_summary = (
        d.groupby("cluster", dropna=False)["clinical_score"]
        .agg(n="size", mean="mean", sd="std", median="median")
        .reset_index()
    )

    between_ss = float(
        sum(
            row["n"] * (row["mean"] - overall_mean) ** 2
            for _, row in cluster_summary.iterrows()
        )
    )

    eta_squared = float(np.clip(between_ss / max(total_ss, eps), 0.0, 1.0))

    return {
        "eta_squared": eta_squared,
        "n": n,
        "n_clusters": n_clusters,
        "overall_mean": overall_mean,
        "total_ss": total_ss,
        "between_ss": between_ss,
        "cluster_summary": cluster_summary,
    }


def _aggregate_clinical_score_values(
    values: Sequence[float],
    *,
    aggregation: str = "mean",
) -> float:
    """
    Aggregate per-timepoint clinical score separation values.

    Supported aggregation:
        mean:
            Average across timepoints.

        min:
            Strict mode. Requires the metric to be good at every timepoint.

        max:
            Exploratory mode. Rewards strong separation at any timepoint.
    """
    vals = pd.Series(values, dtype=float).dropna()

    if len(vals) == 0:
        return np.nan

    aggregation_key = str(aggregation or "mean").lower()

    if aggregation_key == "mean":
        return float(vals.mean())
    if aggregation_key == "min":
        return float(vals.min())
    if aggregation_key == "max":
        return float(vals.max())

    raise ValueError(
        f"Unknown clinical score aggregation={aggregation!r}. "
        "Use 'mean', 'min', or 'max'."
    )


def compute_longitudinal_clinical_score_separation(
    *,
    membership_df: pd.DataFrame,
    timepoint_dfs: Mapping[str, pd.DataFrame],
    clinical_score_cols: Mapping[str, str],
    timepoints: Optional[Sequence[str]] = None,
    aggregation: str = "mean",
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Compute longitudinal clinical score separation using eta-squared.

    For each timepoint:
        eta_squared(cluster_timepoint, clinical_score_timepoint)

    Then aggregate eta-squared across timepoints.

    Example
    -------
    Baseline:
        eta_squared(cluster_baseline, SRS_total_raw_bl)

    Week 6:
        eta_squared(cluster_week6, SRS_total_raw_w6)

    Month 6:
        eta_squared(cluster_month6, SRS_total_raw_m6)

    Returns
    -------
    dict with:
        longitudinal_clinical_score_separation_raw
        longitudinal_clinical_score_separation_norm
        clinical_score_separation_df
        clinical_score_cluster_summary_df
    """
    if membership_df is None or not isinstance(membership_df, pd.DataFrame) or membership_df.empty:
        raise ValueError("membership_df must be a non-empty pandas DataFrame.")

    timepoint_dfs = {str(k): v for k, v in dict(timepoint_dfs).items()}
    clinical_score_cols = {str(k): str(v) for k, v in dict(clinical_score_cols).items()}

    if timepoints is None:
        timepoints = list(clinical_score_cols.keys())

    timepoints = [str(tp) for tp in timepoints]

    rows: List[Dict[str, Any]] = []
    cluster_summary_parts: List[pd.DataFrame] = []

    for tp in timepoints:
        if tp not in timepoint_dfs:
            raise KeyError(f"timepoint_dfs is missing timepoint {tp!r}.")

        if tp not in clinical_score_cols:
            raise KeyError(f"clinical_score_cols is missing timepoint {tp!r}.")

        score_col = clinical_score_cols[tp]
        df_tp = timepoint_dfs[tp]

        if score_col not in df_tp.columns:
            raise KeyError(
                f"Clinical score column {score_col!r} was not found in dataframe "
                f"for timepoint {tp!r}."
            )

        cluster_col = f"cluster_{tp}"
        if cluster_col not in membership_df.columns:
            raise KeyError(f"membership_df is missing cluster column {cluster_col!r}.")

        clusters = membership_df[cluster_col].reset_index(drop=True)
        scores = df_tp[score_col].reset_index(drop=True)

        if len(clusters) != len(scores):
            raise ValueError(
                f"Length mismatch for timepoint {tp!r}: "
                f"len(clusters)={len(clusters)}, len(scores)={len(scores)}. "
                "Rows must be aligned across membership_df and timepoint dataframe."
            )

        eta_result = _compute_clinical_score_eta_squared(
            clusters,
            scores,
            eps=eps,
        )

        rows.append({
            "timepoint": tp,
            "clinical_score_col": score_col,
            "n": int(eta_result["n"]),
            "n_clusters": int(eta_result["n_clusters"]),
            "overall_mean": float(eta_result["overall_mean"]) if not pd.isna(eta_result["overall_mean"]) else np.nan,
            "eta_squared": float(eta_result["eta_squared"]) if not pd.isna(eta_result["eta_squared"]) else np.nan,
            "between_ss": float(eta_result["between_ss"]) if not pd.isna(eta_result["between_ss"]) else np.nan,
            "total_ss": float(eta_result["total_ss"]) if not pd.isna(eta_result["total_ss"]) else np.nan,
        })

        cluster_summary = eta_result["cluster_summary"]
        if isinstance(cluster_summary, pd.DataFrame) and not cluster_summary.empty:
            cluster_summary = cluster_summary.copy()
            cluster_summary.insert(0, "timepoint", tp)
            cluster_summary.insert(1, "clinical_score_col", score_col)
            cluster_summary_parts.append(cluster_summary)

    clinical_score_separation_df = pd.DataFrame(rows)

    if cluster_summary_parts:
        clinical_score_cluster_summary_df = pd.concat(
            cluster_summary_parts,
            ignore_index=True,
        )
    else:
        clinical_score_cluster_summary_df = pd.DataFrame()

    longitudinal_score = _aggregate_clinical_score_values(
        clinical_score_separation_df["eta_squared"].tolist(),
        aggregation=aggregation,
    )

    # eta-squared is already 0..1, so raw and normalized are the same.
    longitudinal_score_norm = (
        float(np.clip(longitudinal_score, 0.0, 1.0))
        if not pd.isna(longitudinal_score)
        else 0.0
    )

    return {
        "longitudinal_clinical_score_separation_raw": longitudinal_score_norm,
        "longitudinal_clinical_score_separation_norm": longitudinal_score_norm,
        "clinical_score_separation_df": clinical_score_separation_df,
        "clinical_score_cluster_summary_df": clinical_score_cluster_summary_df,
        "clinical_score_aggregation": str(aggregation or "mean").lower(),
    }


def longitudinal_clinical_score_guided_clustering_fitness(
    metrics: Mapping[str, float],
    components: Mapping[str, Mapping[str, Any]],
) -> Tuple[float, Dict[str, float]]:
    """
    Standalone longitudinal clinical-score-guided clustering fitness function.

    Calculation
    -----------
    fitness =
        clinical_score_weight * longitudinal_clinical_score_separation_norm
      + optional_label_weight * longitudinal_label_alignment_norm
      + cross_time_weight * cross_time_ari_norm
      + silhouette_weight * longitudinal_silhouette_norm
      - feature_penalty_weight * feature_penalty_norm

    Notes
    -----
    Clinical scores are not used to create clusters. They are used only after
    clustering to score whether discovered clusters separate clinical severity
    or functioning.

    The optional label component is included only when its weight is non-zero.
    Missing metrics are treated as 0.0 by _metric_value.
    """
    clinical_score_sep = _metric_value(
        metrics,
        "longitudinal_clinical_score_separation_norm",
    )
    label_alignment = _metric_value(
        metrics,
        "longitudinal_label_alignment_norm",
    )
    cross_time_ari = _metric_value(
        metrics,
        "cross_time_ari_norm",
    )
    longitudinal_silhouette = _metric_value(
        metrics,
        "longitudinal_silhouette_norm",
    )
    feature_penalty = _metric_value(
        metrics,
        "feature_penalty_norm",
    )

    w_clinical_score = _component_weight(
        components,
        "longitudinal_clinical_score_separation_norm",
    )
    w_label = _component_weight(
        components,
        "longitudinal_label_alignment_norm",
    )
    w_cross_time = _component_weight(
        components,
        "cross_time_ari_norm",
    )
    w_silhouette = _component_weight(
        components,
        "longitudinal_silhouette_norm",
    )
    w_penalty = _component_weight(
        components,
        "feature_penalty_norm",
    )

    contributions = {
        "longitudinal_clinical_score_separation_norm": w_clinical_score * clinical_score_sep,
        "longitudinal_label_alignment_norm": w_label * label_alignment,
        "cross_time_ari_norm": w_cross_time * cross_time_ari,
        "longitudinal_silhouette_norm": w_silhouette * longitudinal_silhouette,
        "feature_penalty_norm": -w_penalty * feature_penalty,
    }

    # Keep the details table clean when labels are not part of this run.
    if w_label == 0.0:
        contributions.pop("longitudinal_label_alignment_norm", None)

    fitness = sum(contributions.values())
    return float(fitness), {k: float(v) for k, v in contributions.items()}


def active_fitness_uses_longitudinal_clinical_score(cfg: ClinicalResponseGAFSConfig) -> bool:
    """
    Return whether the active fitness components use longitudinal clinical-score separation.
    """
    components = active_fitness_components(cfg)
    return bool(
        components.get(
            "longitudinal_clinical_score_separation_norm",
            {},
        ).get("weight", 0.0) != 0
    )


def resolve_longitudinal_clinical_score_config(
    cfg: ClinicalResponseGAFSConfig,
) -> Dict[str, Any]:
    """
    Resolve clinical-score scoring config for longitudinal clinical-score-guided presets.

    Expected config location:
        cfg.fitness_preset_config[cfg.fitness_preset]["scoring_columns"]
            ["timepoint_clinical_score_cols"]

    Example:
        "scoring_columns": {
            "timepoint_clinical_score_cols": {
                "baseline": "SRS_total_raw_bl",
                "week6": "SRS_total_raw_w6",
                "month6": "SRS_total_raw_m6",
            }
        }

    Optional:
        cfg.fitness_preset_config[cfg.fitness_preset]["clinical_score_scoring"]
            ["aggregation"]

    Default aggregation:
        "mean"
    """
    preset_name = cfg.fitness_preset or "balanced_clinical"
    preset_cfg = get_active_fitness_preset_config(cfg)

    scoring_columns = dict(preset_cfg.get("scoring_columns", {}) or {})
    clinical_score_cols = scoring_columns.get("timepoint_clinical_score_cols")

    if clinical_score_cols is None:
        raise ValueError(
            f"fitness_preset={preset_name!r} uses "
            "longitudinal_clinical_score_separation_norm, but no "
            "timepoint_clinical_score_cols were provided. Put them under "
            f"fitness_preset_config[{preset_name!r}]['scoring_columns']"
            "['timepoint_clinical_score_cols']."
        )

    clinical_score_cols = dict(clinical_score_cols)

    if len(clinical_score_cols) == 0:
        raise ValueError("timepoint_clinical_score_cols cannot be empty.")

    clinical_score_cfg = dict(preset_cfg.get("clinical_score_scoring", {}) or {})

    return {
        "timepoint_clinical_score_cols": {
            str(k): str(v)
            for k, v in clinical_score_cols.items()
        },
        "aggregation": clinical_score_cfg.get("aggregation", "mean"),
    }


# =============================================================================
# Feature subset evaluation
# =============================================================================

def evaluate_feature_subset_clinical_response(
    *,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    mask: Sequence[Any],
    cfg: ClinicalResponseGAFSConfig,
    treatment_col: str = "treatment_arm",
    age_col: Optional[str] = "age",
    treated_label: str = "Treatment",
    control_label: str = "Placebo",
) -> Dict[str, Any]:
    """
    Evaluate one binary feature mask using the clinical-response objective.

    Steps
    -----
    1. Repair the mask to respect min/max feature constraints.
    2. Select feature columns.
    3. Cluster samples with KMeans.
    4. Compute bootstrap stability.
    5. Compute silhouette cluster quality.
    6. Compute cluster-specific treatment effects.
    7. Compute Sclinical.
    8. Combine normalized components into final fitness.

    Parameters
    ----------
    df:
        Single dataframe.

    feature_cols:
        Candidate feature columns.

    mask:
        Binary feature mask.

    cfg:
        Clinical-response GA configuration.

    treatment_col:
        Treatment assignment column.

    age_col:
        Optional age column.

    treated_label:
        Standardized treatment label.

    control_label:
        Standardized control/placebo label.

    Returns
    -------
    dict
        Full evaluation object including fitness, selected features, metrics,
        cluster labels, Sclinical details, and treatment effects dataframe.
    """
    rng = np.random.default_rng(cfg.random_seed)
    repaired_mask = _repair_mask(mask, cfg, rng)
    selected_cols = _selected_columns(feature_cols, repaired_mask)

    if len(selected_cols) < cfg.min_features:
        return {
            "fitness": -1e9,
            "selected_cols": selected_cols,
            "n_features": len(selected_cols),
            "error": "Too few selected features.",
        }

    uses_sclinical = active_fitness_uses_sclinical(cfg)
    uses_label_alignment = active_fitness_uses_label_alignment(cfg)
    uses_cluster_quality = active_fitness_uses_cluster_quality(cfg)
    uses_stability = active_fitness_uses_stability(cfg)

    label_config: Optional[Dict[str, Any]] = None
    if uses_label_alignment:
        label_config = resolve_label_scoring_config(cfg)

    effective_age_col = age_col if age_col is not None and age_col in df.columns else None

    if uses_sclinical:
        validate_clinical_response_inputs(
            df=df,
            feature_cols=selected_cols,
            treatment_col=treatment_col,
            baseline_col=cfg.baseline_col,
            followup_col=cfg.followup_col,
            age_col=effective_age_col,
        )

        needed_cols = list(selected_cols) + [treatment_col, cfg.baseline_col, cfg.followup_col]
        if effective_age_col is not None:
            needed_cols.append(effective_age_col)
    else:
        # Non-clinical presets require only feature columns unless a label-guided
        # preset is active.
        missing_features = [col for col in selected_cols if col not in df.columns]
        if missing_features:
            raise ValueError(f"df is missing selected feature columns: {missing_features}")
        needed_cols = list(selected_cols)

    if uses_label_alignment and label_config is not None:
        label_col = label_config["label_col"]
        if label_col not in df.columns:
            raise ValueError(f"df is missing label_col={label_col!r}.")
        needed_cols.append(label_col)

    d = df[needed_cols].copy()

    numeric_cols = list(selected_cols)

    if uses_sclinical:
        d[treatment_col] = standardize_treatment_arm(
            d[treatment_col],
            treated_label=treated_label,
            control_label=control_label,
        )

        numeric_cols = numeric_cols + [cfg.baseline_col, cfg.followup_col]
        if effective_age_col is not None:
            numeric_cols.append(effective_age_col)

    for col in numeric_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    drop_subset = list(selected_cols)
    if uses_sclinical:
        drop_subset += [treatment_col, cfg.baseline_col, cfg.followup_col]
    if uses_label_alignment and label_config is not None:
        drop_subset.append(label_config["label_col"])

    d = d.dropna(subset=drop_subset).copy()

    min_rows_needed = max(cfg.k * cfg.min_cluster_total_n, cfg.k + 2)
    if not uses_sclinical:
        min_rows_needed = max(cfg.k + 2, 5)

    if len(d) < min_rows_needed:
        return {
            "fitness": -1e9,
            "selected_cols": selected_cols,
            "n_features": len(selected_cols),
            "error": "Too few usable rows after dropping missing values.",
        }

    if uses_sclinical and d[treatment_col].nunique() < 2:
        return {
            "fitness": -1e9,
            "selected_cols": selected_cols,
            "n_features": len(selected_cols),
            "error": "Only one treatment arm present.",
        }

    # -------------------------------------------------------------------------
    # Cluster on selected features only.
    # -------------------------------------------------------------------------
    X = d[selected_cols].to_numpy(dtype=float)

    labels, X_scaled, _, _, model_name = _fit_cluster_labels(
        X,
        cfg=cfg,
    )

    # -------------------------------------------------------------------------
    # Metrics.
    # -------------------------------------------------------------------------
    if uses_stability:
        stability_dict = compute_bootstrap_stability(
            X,
            labels,
            cfg=cfg,
        )
    else:
        stability_dict = {"ari_mean": np.nan, "ari_sd": np.nan}

    if uses_cluster_quality:
        quality_dict = compute_cluster_quality(X_scaled, labels)
    else:
        quality_dict = {"silhouette": np.nan}

    if uses_sclinical:
        effects_df = compute_subtype_treatment_effects(
            d,
            labels,
            treatment_col=treatment_col,
            baseline_col=cfg.baseline_col,
            followup_col=cfg.followup_col,
            age_col=effective_age_col,
            cfg=cfg,
            treated_label=treated_label,
            control_label=control_label,
        )

        sclin_dict = compute_sclinical(effects_df, cfg=cfg)
    else:
        effects_df = pd.DataFrame()
        sclin_dict = {
            "sclin_raw": np.nan,
            "sclin_norm": 0.0,
            "spread": np.nan,
            "opposite_sign_bonus": 0.0,
            "precision_mean": np.nan,
            "small_cluster_penalty": np.nan,
            "arm_imbalance_penalty": np.nan,
            "n_valid_effects": 0,
        }

    if uses_label_alignment and label_config is not None:
        label_dict = compute_label_alignment(
            d[label_config["label_col"]],
            labels,
            metric=label_config.get("metric", "ari_nmi"),
        )
    else:
        label_dict = {
            "label_alignment_raw": np.nan,
            "label_alignment_norm": 0.0,
            "label_ari_raw": np.nan,
            "label_nmi_raw": np.nan,
            "metric": None,
        }

    # -------------------------------------------------------------------------
    # Normalize components.
    # -------------------------------------------------------------------------
    stability_raw = stability_dict["ari_mean"]
    cluster_raw = quality_dict["silhouette"]
    sclin_raw = sclin_dict["sclin_raw"]

    stability_norm = (
        0.0 if pd.isna(stability_raw) else float(np.clip(stability_raw, 0.0, 1.0))
    )
    cluster_norm = (
        0.0 if pd.isna(cluster_raw) else float(np.clip((cluster_raw + 1.0) / 2.0, 0.0, 1.0))
    )
    sclin_norm = (
        0.0 if pd.isna(sclin_dict["sclin_norm"]) else float(np.clip(sclin_dict["sclin_norm"], 0.0, 1.0))
    )
    label_alignment_raw = label_dict["label_alignment_raw"]
    label_alignment_norm = (
        0.0
        if pd.isna(label_dict["label_alignment_norm"])
        else float(np.clip(label_dict["label_alignment_norm"], 0.0, 1.0))
    )

    feature_fraction = len(selected_cols) / max(len(feature_cols), cfg.eps)
    feature_penalty = float(feature_fraction ** cfg.feature_fraction_penalty_power)

    # -------------------------------------------------------------------------
    # Dynamic final fitness.
    # -------------------------------------------------------------------------
    metrics_for_fitness: Dict[str, float] = {
        "stability_raw": float(stability_raw) if not pd.isna(stability_raw) else np.nan,
        "cluster_raw": float(cluster_raw) if not pd.isna(cluster_raw) else np.nan,
        "sclin_raw": float(sclin_raw) if not pd.isna(sclin_raw) else np.nan,
        "label_alignment_raw": float(label_alignment_raw) if not pd.isna(label_alignment_raw) else np.nan,
        "label_ari_raw": float(label_dict["label_ari_raw"]) if not pd.isna(label_dict["label_ari_raw"]) else np.nan,
        "label_nmi_raw": float(label_dict["label_nmi_raw"]) if not pd.isna(label_dict["label_nmi_raw"]) else np.nan,
        "feature_penalty_raw": feature_penalty,
        "stability_norm": stability_norm,
        "cluster_norm": cluster_norm,
        "sclin_norm": sclin_norm,
        "label_alignment_norm": label_alignment_norm,
        "feature_penalty_norm": feature_penalty,
        "n_features": float(len(selected_cols)),
        "n_total_features": float(len(feature_cols)),
    }

    fitness, fitness_details = compute_dynamic_fitness(
        metrics_for_fitness,
        cfg=cfg,
    )

    return {
        "fitness": float(fitness),
        "selected_cols": selected_cols,
        "n_features": int(len(selected_cols)),
        "stability_raw": float(stability_raw) if not pd.isna(stability_raw) else np.nan,
        "cluster_raw": float(cluster_raw) if not pd.isna(cluster_raw) else np.nan,
        "sclin_raw": float(sclin_raw) if not pd.isna(sclin_raw) else np.nan,
        "label_alignment_raw": float(label_alignment_raw) if not pd.isna(label_alignment_raw) else np.nan,
        "label_ari_raw": float(label_dict["label_ari_raw"]) if not pd.isna(label_dict["label_ari_raw"]) else np.nan,
        "label_nmi_raw": float(label_dict["label_nmi_raw"]) if not pd.isna(label_dict["label_nmi_raw"]) else np.nan,
        "feature_penalty_raw": feature_penalty,
        "stability_norm": stability_norm,
        "cluster_norm": cluster_norm,
        "sclin_norm": sclin_norm,
        "label_alignment_norm": label_alignment_norm,
        "feature_penalty_norm": feature_penalty,
        "metrics_for_fitness": metrics_for_fitness,
        "fitness_details": fitness_details,
        "cluster_labels": labels,
        "model_name": model_name,
        "effects_df": effects_df,
        "details": {
            "stability_dict": stability_dict,
            "quality_dict": quality_dict,
            "sclin_dict": sclin_dict,
            "label_dict": label_dict,
        },
    }



# =============================================================================
# Dynamic GA logging helpers
# =============================================================================

def _safe_float_or_none(value: Any) -> Optional[float]:
    """
    Convert a value to float if finite; otherwise return None.
    """
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _format_metric_with_contribution(
    label: str,
    raw_value: Any,
    weighted_value: Any,
    *,
    decimals: int = 3,
) -> Optional[str]:
    """
    Format one metric as raw value plus weighted contribution.

    Example
    -------
    ARI=0.674 (weighted=0.101)

    If the raw value is missing/NaN, returns None.
    If the weighted value is missing/NaN, only the raw value is shown.
    """
    raw = _safe_float_or_none(raw_value)
    if raw is None:
        return None

    weighted = _safe_float_or_none(weighted_value)

    if weighted is None:
        return f"{label}={raw:.{decimals}f}"

    return f"{label}={raw:.{decimals}f} (weighted={weighted:.{decimals}f})"


def build_generation_log_message(
    row: Mapping[str, Any],
    *,
    cfg: ClinicalResponseGAFSConfig,
) -> str:
    """
    Build a configurable GA generation log message.

    The metrics shown are controlled by:

        cfg.logging_config["metrics_to_show"]

    If that is not provided, the logger automatically prints the active
    fitness_components for the selected fitness_preset.

    Each displayed metric includes:
        1. raw / normalized metric value
        2. signed weighted contribution to final fitness

    This avoids confusion when a metric is computed but its weight is zero.
    Example:
        feat_pen=0.250 (weighted=-0.000)
    """
    parts = [
        f"Gen {int(row['generation']):03d}",
        f"time/gen={float(row['generation_time_sec']):.2f}s",
        f"total={float(row['total_elapsed_sec']) / 60:.2f}m",
        f"fitness={float(row['best_fitness']):.4f}",
        f"n_feat={int(row['n_features'])}",
    ]

    metric_pairs, display_names = get_active_metric_display_config(cfg)

    metric_parts: List[str] = []
    for user_metric_name, internal_metric_name in metric_pairs:
        raw_value = row.get(user_metric_name, row.get(internal_metric_name))

        # Prefer user-facing weighted column, then internal weighted column.
        weighted_value = row.get(
            f"{user_metric_name}__weighted",
            row.get(f"{internal_metric_name}__weighted"),
        )

        label = display_names.get(
            user_metric_name,
            display_names.get(
                internal_metric_name,
                default_metric_display_label(user_metric_name),
            ),
        )

        formatted = _format_metric_with_contribution(label, raw_value, weighted_value)
        if formatted is not None:
            metric_parts.append(formatted)

    parts.extend(metric_parts)

    return " | ".join(parts)


# =============================================================================
# GA runner
# =============================================================================

class ClinicalResponseFeatureSelectionGA:
    """
    Genetic algorithm runner for clinical-response-guided feature selection.

    Parameters
    ----------
    df:
        Single dataframe containing feature columns and clinical/treatment columns.

    feature_cols:
        Candidate feature columns used for clustering.

    treatment_col:
        Treatment assignment column.

    cfg:
        Clinical-response GA configuration.

    outdir:
        Output directory for history and selected-feature files.

    age_col:
        Optional age column.

    treated_label:
        Standardized label for treated subjects.

    control_label:
        Standardized label for control subjects.
    """

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        cfg: ClinicalResponseGAFSConfig,
        feature_cols: Optional[Sequence[str]] = None,
        treatment_col: str = "treatment_arm",
        outdir: Optional[str] = None,
        save_outputs: bool = False,
        age_col: Optional[str] = "age",
        treated_label: str = "Treatment",
        control_label: str = "Placebo",
    ) -> None:
        self.df = df.copy()
        self.cfg = cfg
        self.feature_cols = resolve_feature_config(cfg, feature_cols=feature_cols)
        self.outdir = outdir
        self.save_outputs = bool(save_outputs)

        # Apply nested preset-specific settings such as Sclinical weights,
        # feature selection constraints, and clinical effect constraints.
        apply_active_fitness_preset_config(self.cfg)

        clinical = resolve_clinical_config(
            cfg,
            treatment_col=treatment_col,
            age_col=age_col,
            treated_label=treated_label,
            control_label=control_label,
            require_clinical=active_fitness_uses_sclinical(self.cfg),
        )

        self.treatment_col = clinical["treatment_col"]
        self.age_col = clinical["age_col"]
        self.treated_label = clinical["treated_label"]
        self.control_label = clinical["control_label"]

        if self.save_outputs:
            if self.outdir is None:
                self.outdir = "clinical_response_ga_output"
            os.makedirs(self.outdir, exist_ok=True)

        if active_fitness_uses_sclinical(self.cfg):
            validate_clinical_response_inputs(
                df=self.df,
                feature_cols=self.feature_cols,
                treatment_col=self.treatment_col,
                baseline_col=self.cfg.baseline_col,
                followup_col=self.cfg.followup_col,
                age_col=self.age_col if self.age_col in self.df.columns else None,
            )
        else:
            missing_features = [col for col in self.feature_cols if col not in self.df.columns]
            if missing_features:
                raise ValueError(f"df is missing feature columns: {missing_features}")

        if active_fitness_uses_label_alignment(self.cfg):
            label_config = resolve_label_scoring_config(self.cfg)
            label_col = label_config["label_col"]
            if label_col not in self.df.columns:
                raise ValueError(f"df is missing label_col={label_col!r}.")

        self._cache: Dict[str, Dict[str, Any]] = {}
        self.history: List[Dict[str, Any]] = []

        # Store the raw full GA population at each generation.
        # This is kept for debugging only.
        self.raw_population_history: List[np.ndarray] = []

        # Store the repaired/evaluated full GA population at each generation.
        # This is used for the separate feature-selection frequency plot.
        self.population_history: List[np.ndarray] = []

        self._start_time: Optional[float] = None
        self._last_gen_time: Optional[float] = None

    def _mask_key(self, mask: Sequence[Any]) -> str:
        """Convert a solution mask into a stable string cache key."""
        arr = (np.asarray(mask, dtype=float) >= 0.5).astype(int)
        return "".join(map(str, arr.tolist()))

    def evaluate(self, mask: Sequence[Any]) -> Dict[str, Any]:
        """
        Evaluate a candidate feature mask.

        This method also handles caching so repeated masks are not recomputed.
        """
        rng = np.random.default_rng(self.cfg.random_seed)
        repaired_mask = _repair_mask(mask, self.cfg, rng)
        key = self._mask_key(repaired_mask)

        if self.cfg.use_cache and key in self._cache:
            return self._cache[key]

        result = evaluate_feature_subset_clinical_response(
            df=self.df,
            feature_cols=self.feature_cols,
            mask=repaired_mask,
            cfg=self.cfg,
            treatment_col=self.treatment_col,
            age_col=self.age_col,
            treated_label=self.treated_label,
            control_label=self.control_label,
        )

        if self.cfg.use_cache:
            self._cache[key] = result

        return result

    def run(self) -> Dict[str, Any]:
        """
        Run the genetic algorithm.

        Returns
        -------
        dict
            {
                "ga_instance": pygad.GA,
                "best_solution": np.ndarray,
                "best_mask": np.ndarray,
                "best_fitness": float,
                "best_eval": dict,
                "selected_cols": list[str],
                "effects_df": pd.DataFrame,
                "history_df": pd.DataFrame,
                "config": dict,
                ...
            }
        """
        try:
            import pygad
        except ImportError as exc:
            raise ImportError(
                "pygad is required to run the GA. Install it with: pip install pygad"
            ) from exc

        n_genes = len(self.feature_cols)

        initial_population = make_sparse_initial_population(
            n_genes,
            sol_per_pop=self.cfg.sol_per_pop,
            min_features=self.cfg.min_features,
            max_features=self.cfg.max_features,
            random_seed=self.cfg.random_seed,
        )

        self._start_time = time.time()
        self._last_gen_time = self._start_time

        def fitness_func(ga_instance: Any, solution: np.ndarray, solution_idx: int) -> float:
            """PyGAD callback: return fitness for one solution."""
            evaluation = self.evaluate(solution)
            return float(evaluation["fitness"])

        def on_generation(ga_instance: Any) -> None:
            """PyGAD callback: log the best solution after each generation."""
            # Store all raw candidate masks from this generation for debugging.
            raw_population = np.asarray(ga_instance.population).copy()
            self.raw_population_history.append(raw_population)

            # Store repaired/evaluated candidate masks for frequency analysis.
            # This matches the masks actually scored by self.evaluate(...), because
            # evaluate() repairs each raw solution before computing fitness.
            repaired_population = np.vstack(
                [
                    _repair_mask(
                        solution,
                        self.cfg,
                        np.random.default_rng(self.cfg.random_seed),
                    )
                    for solution in raw_population
                ]
            )
            self.population_history.append(repaired_population)

            now = time.time()
            gen_time = now - (self._last_gen_time or now)
            total_time = now - (self._start_time or now)
            self._last_gen_time = now

            best_solution, best_fitness, _ = ga_instance.best_solution()
            best_eval = self.evaluate(best_solution)

            row = {
                "generation": int(ga_instance.generations_completed),
                "best_fitness": float(best_fitness),
                "n_features": int(best_eval.get("n_features", 0)),
                "generation_time_sec": float(gen_time),
                "total_elapsed_sec": float(total_time),
                "solution_mask": self._mask_key(best_solution),
                "selected_features": " | ".join(best_eval.get("selected_cols", [])),
                "model_name": best_eval.get("model_name", None),
                "stability_raw": best_eval.get("stability_raw", np.nan),
                "cluster_raw": best_eval.get("cluster_raw", np.nan),
                "sclin_raw": best_eval.get("sclin_raw", np.nan),
                "feature_penalty_raw": best_eval.get("feature_penalty_raw", np.nan),
                "stability_norm": best_eval.get("stability_norm", np.nan),
                "cluster_norm": best_eval.get("cluster_norm", np.nan),
                "sclin_norm": best_eval.get("sclin_norm", np.nan),
                "feature_penalty_norm": best_eval.get("feature_penalty_norm", np.nan),
                "fitness_function_name": best_eval.get("fitness_details", {}).get("fitness_function_name", None),
                "fitness_contributions": best_eval.get("fitness_details", {}).get("fitness_contributions", None),
            }

            # Add user-facing metric aliases to history when configured through
            # fitness_components/logging_config. This makes plots and logs match
            # the metric names users write in the config.
            #
            # For each active metric, store:
            #   metric_name              = raw / normalized metric value
            #   metric_name__weighted    = signed weighted contribution to fitness
            #
            # Example:
            #   bootstrap_ari_norm = 0.674
            #   bootstrap_ari_norm__weighted = 0.101
            metrics_for_fitness = best_eval.get("metrics_for_fitness", {}) or {}
            fitness_contributions = (
                best_eval.get("fitness_details", {}).get("fitness_contributions", {}) or {}
            )
            metric_pairs, _ = get_active_metric_display_config(self.cfg)

            for user_metric_name, internal_metric_name in metric_pairs:
                if internal_metric_name in metrics_for_fitness:
                    row[user_metric_name] = metrics_for_fitness[internal_metric_name]
                elif internal_metric_name in row:
                    row[user_metric_name] = row[internal_metric_name]

                if internal_metric_name in fitness_contributions:
                    row[f"{user_metric_name}__weighted"] = fitness_contributions[internal_metric_name]
            self.history.append(row)

            print(build_generation_log_message(row, cfg=self.cfg))

        ga_instance = pygad.GA(
            num_generations=self.cfg.num_generations,
            sol_per_pop=self.cfg.sol_per_pop,
            num_parents_mating=self.cfg.num_parents_mating,
            keep_parents=self.cfg.keep_parents,
            keep_elitism=self.cfg.keep_elitism,
            num_genes=n_genes,
            gene_space=[0, 1],
            gene_type=int,
            initial_population=initial_population,
            parent_selection_type=self.cfg.parent_selection_type,
            crossover_type=self.cfg.crossover_type,
            mutation_type=self.cfg.mutation_type,
            mutation_percent_genes=self.cfg.mutation_percent_genes,
            random_seed=self.cfg.random_seed,
            fitness_func=fitness_func,
            on_generation=on_generation,
        )

        ga_instance.run()

        elapsed = time.time() - (self._start_time or time.time())

        best_solution, best_fitness, _ = ga_instance.best_solution()
        rng = np.random.default_rng(self.cfg.random_seed)
        best_mask = _repair_mask(best_solution, self.cfg, rng)
        best_eval = self.evaluate(best_mask)

        history_df = pd.DataFrame(self.history)

        feature_selection_frequency_df = compute_feature_selection_frequency(
            self.population_history,
            self.feature_cols,
        )

        result: Dict[str, Any] = {
            "ga_instance": ga_instance,
            "best_solution": np.asarray(best_solution, dtype=int),
            "best_mask": np.asarray(best_mask, dtype=int),
            "best_fitness": float(best_fitness),
            "best_eval": best_eval,
            "selected_cols": best_eval.get("selected_cols", []),
            "effects_df": best_eval.get("effects_df", pd.DataFrame()),
            "history_df": history_df,
            # raw_population_history contains original PyGAD masks.
            # population_history contains repaired/evaluated masks used for frequency.
            "raw_population_history": self.raw_population_history,
            "population_history": self.population_history,
            "feature_selection_frequency_df": feature_selection_frequency_df,
            "config": safe_config_dict(self.cfg),
            "feature_cols": self.feature_cols,
            "treatment_col": self.treatment_col,
            "baseline_col": self.cfg.baseline_col,
            "followup_col": self.cfg.followup_col,
            "age_col": self.age_col,
        }

        if self.save_outputs:
            self._save_outputs(result)

        print("\n" + "=" * 80)
        print(f"GA completed in {elapsed:.2f} sec")
        print(f"GA completed in {elapsed / 60:.2f} min")
        if self.cfg.num_generations > 0:
            print(f"Average time per generation: {elapsed / self.cfg.num_generations:.2f} sec")
        print("=" * 80)

        return result

    def _save_outputs(self, result: Dict[str, Any]) -> None:
        """Save lightweight run outputs to outdir."""
        if self.outdir is None:
            self.outdir = "clinical_response_ga_output"
        os.makedirs(self.outdir, exist_ok=True)

        history_path = os.path.join(self.outdir, "history.csv")
        selected_path = os.path.join(self.outdir, "selected_features.csv")
        effects_path = os.path.join(self.outdir, "best_effects.csv")
        frequency_path = os.path.join(self.outdir, "feature_selection_frequency.csv")

        history_df = result.get("history_df", pd.DataFrame())
        if isinstance(history_df, pd.DataFrame):
            history_df.to_csv(history_path, index=False)

        pd.DataFrame({"selected_feature": result.get("selected_cols", [])}).to_csv(
            selected_path,
            index=False,
        )

        effects_df = result.get("effects_df", pd.DataFrame())
        if isinstance(effects_df, pd.DataFrame):
            effects_df.to_csv(effects_path, index=False)

        frequency_df = result.get("feature_selection_frequency_df", pd.DataFrame())
        if isinstance(frequency_df, pd.DataFrame):
            frequency_df.to_csv(frequency_path, index=False)





# =============================================================================
# Feature-selection frequency helpers
# =============================================================================

def compute_feature_selection_frequency(
    population_history: Sequence[np.ndarray],
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Compute feature-selection frequency across all repaired/evaluated GA candidate solutions.

    This uses the full repaired GA population across all stored generations:

        all generations x all repaired candidate masks in each generation

    Important
    ---------
    The GA may create raw masks that violate min_features or max_features.
    The evaluator repairs those masks before scoring. This frequency table should
    therefore use repaired/evaluated masks, not raw PyGAD masks.

    Example
    -------
    If num_generations=20 and sol_per_pop=100, each feature frequency is
    computed over roughly 2,000 candidate masks.

    Parameters
    ----------
    population_history:
        Sequence of repaired GA population arrays. Each array should have shape:
            (sol_per_pop, n_features)

    feature_cols:
        Feature names corresponding to mask columns.

    Returns
    -------
    pd.DataFrame
        Feature-selection frequency table with columns:
            - feature
            - selection_count
            - total_masks
            - selection_frequency
            - selection_percent
    """
    if population_history is None or len(population_history) == 0:
        raise ValueError(
            "population_history is empty; cannot compute feature-selection frequency."
        )

    feature_cols = list(feature_cols)

    populations: List[np.ndarray] = []

    for pop in population_history:
        arr = np.asarray(pop)

        if arr.ndim != 2:
            raise ValueError(
                "Each population must be a 2D array with shape "
                "(n_candidate_solutions, n_features)."
            )

        populations.append(arr)

    all_masks = np.vstack(populations)

    if all_masks.shape[1] != len(feature_cols):
        raise ValueError(
            f"Population mask width ({all_masks.shape[1]}) does not match "
            f"number of feature columns ({len(feature_cols)})."
        )

    binary_masks = (all_masks > 0).astype(int)

    selection_count = binary_masks.sum(axis=0)
    total_masks = int(binary_masks.shape[0])
    selection_frequency = selection_count / max(total_masks, 1)

    freq_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "selection_count": selection_count.astype(int),
            "total_masks": total_masks,
            "selection_frequency": selection_frequency.astype(float),
            "selection_percent": selection_frequency.astype(float) * 100.0,
        }
    )

    freq_df = (
        freq_df.sort_values(
            ["selection_frequency", "selection_count", "feature"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    return freq_df


def plot_feature_selection_frequency(
    result: Dict[str, Any],
    *,
    mode: str = "global",
    top_n: int = 25,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    font_size: float = 12.0,
    bar_color: str = "darkblue",
    feature_colors: Optional[Mapping[str, str]] = None,
    x_tick_rotation: int = 0,
    x_lim: Optional[Tuple[float, float]] = None,
    annotate_bars: bool = True,
    annotate_decimals: int = 1,
    annotate_font_size: Optional[float] = None,
    annotate_offset: float = 0.5,
    axis_line_color: str = "black",
    axis_line_width: float = 1.4,
    tick_color: str = "black",
    grid_color: str = "black",
    grid_alpha: float = 0.18,
    grid_line_width: float = 0.8,
    show: bool = True,
) -> Tuple[pd.DataFrame, Any, Any]:
    """
    Plot feature-selection frequency across all GA candidate solutions.

    This is intentionally separate from make_clinical_response_plots(...).

    Modes
    -----
    mode="global":
        Plot the top-N most frequently selected features across all repaired /
        evaluated candidate masks.

    mode="selected":
        Plot only the final selected features in result["selected_cols"], using
        their selection frequency across all repaired / evaluated candidate masks.
        Features are sorted from highest to lowest frequency.

    Parameters
    ----------
    result:
        GA result dictionary returned by ClinicalResponseFeatureSelectionGA.run().

    mode:
        Frequency view to plot. Options are "global" and "selected".

    top_n:
        Number of most frequently selected features to show when mode="global".
        Ignored when mode="selected".

    title:
        Optional plot title.

    figsize:
        Optional Matplotlib figure size. If None, height is chosen from top_n.

    font_size:
        Base font size.

    bar_color:
        Default color used for all bars.

    feature_colors:
        Optional mapping from feature name to bar color. This lets you highlight
        selected or known informative features while all other bars use bar_color.

        Example:
            feature_colors={
                "feature_1": "darkred",
                "feature_2": "darkred",
            }

    show:
        If True, display the plot in the notebook.

    Returns
    -------
    Tuple[pd.DataFrame, plt.Figure, plt.Axes]
        plot_df:
            Top-N frequency table used for the plot.

        fig, ax:
            Matplotlib figure and axis.
    """
    import matplotlib.pyplot as plt

    if "feature_selection_frequency_df" not in result:
        raise KeyError(
            "result is missing 'feature_selection_frequency_df'. "
            "Rerun the GA with the updated module so population history is stored."
        )

    freq_df = result["feature_selection_frequency_df"].copy()

    if freq_df.empty:
        raise ValueError("feature_selection_frequency_df is empty; nothing to plot.")

    mode = str(mode).lower()

    if mode not in {"global", "selected"}:
        raise ValueError("mode must be either 'global' or 'selected'.")

    if mode == "global":
        if top_n is None:
            top_n = len(freq_df)

        top_n = int(top_n)

        if top_n <= 0:
            raise ValueError("top_n must be positive.")

        # Global view: show the top-N most frequently selected features overall.
        plot_df = freq_df.head(top_n).copy()

    else:
        # Selected view: show the frequency only for the final selected features.
        selected_cols = list(result.get("selected_cols", []))

        if len(selected_cols) == 0:
            raise ValueError(
                "mode='selected' requires result['selected_cols'], but it is empty or missing."
            )

        freq_features = set(freq_df["feature"])
        missing_selected = [
            feature for feature in selected_cols
            if feature not in freq_features
        ]

        if missing_selected:
            raise ValueError(
                "Some selected features are missing from feature_selection_frequency_df: "
                f"{missing_selected}"
            )

        plot_df = freq_df[freq_df["feature"].isin(selected_cols)].copy()

        # Selected-feature view is also sorted by frequency so the most frequently
        # selected final feature appears first.
        plot_df = (
            plot_df.sort_values(
                ["selection_frequency", "selection_count", "feature"],
                ascending=[False, False, True],
            )
            .reset_index(drop=True)
        )

    # Reverse order for horizontal plotting so the first item appears at the top.
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    if figsize is None:
        height = max(4.0, 0.35 * len(plot_df) + 1.5)
        figsize = (9.0, height)

    fig, ax = plt.subplots(figsize=figsize)

    feature_colors = dict(feature_colors or {})
    bar_colors = [
        feature_colors.get(feature, bar_color)
        for feature in plot_df["feature"]
    ]

    ax.barh(
        plot_df["feature"],
        plot_df["selection_percent"],
        color=bar_colors,
    )

    if title is None:
        if mode == "global":
            title = f"Top {min(top_n, len(freq_df))} Global Feature Selection Frequencies"
        else:
            title = "Final Selected Feature Frequencies"

    _apply_clinical_response_axis_style(
        ax,
        title=title,
        xlabel="Selection frequency across repaired GA candidate masks (%)",
        ylabel="Feature",
        font_size=font_size,
        grid_axis="x",
        x_tick_rotation=x_tick_rotation,
        bold=True,
        axis_line_color=axis_line_color,
        axis_line_width=axis_line_width,
        tick_color=tick_color,
        grid_color=grid_color,
        grid_alpha=grid_alpha,
        grid_line_width=grid_line_width,
    )

    # Add percent labels to the right side of each bar.
    if annotate_bars:
        ann_fs = annotate_font_size if annotate_font_size is not None else max(8, font_size - 2)

        for y_pos, value in enumerate(plot_df["selection_percent"]):
            ax.text(
                value + annotate_offset,
                y_pos,
                f"{value:.{annotate_decimals}f}%",
                va="center",
                fontsize=ann_fs,
                fontweight="bold",
            )

    if x_lim is not None:
        ax.set_xlim(*x_lim)
    else:
        ax.set_xlim(0, max(100.0, float(plot_df["selection_percent"].max()) * 1.10))

    fig.tight_layout()

    if show:
        plt.show()

    return plot_df, fig, ax



# =============================================================================
# Plotting helpers
# =============================================================================

def _apply_clinical_response_axis_style(
    ax: Any,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    font_size: float = 12.0,
    grid_axis: str = "y",
    x_tick_rotation: int = 0,
    bold: bool = True,
    axis_line_color: str = "black",
    axis_line_width: float = 1.4,
    tick_color: str = "black",
    grid_color: str = "black",
    grid_alpha: float = 0.18,
    grid_line_width: float = 0.8,
) -> None:
    """
    Apply a consistent presentation-ready style to clinical-response plots.

    Parameters
    ----------
    ax:
        Matplotlib axis.

    title:
        Axis title.

    xlabel:
        X-axis label.

    ylabel:
        Y-axis label.

    font_size:
        Base font size for title, labels, and ticks.

    grid_axis:
        Axis for light grid lines. Usually "y" for bar plots and "both" for line plots.
    """
    weight = "bold" if bold else "normal"

    ax.set_title(title, fontsize=font_size + 2, fontweight=weight, pad=12)
    ax.set_xlabel(xlabel, fontsize=font_size, fontweight=weight, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=font_size, fontweight=weight, labelpad=8)
    ax.tick_params(
        axis="both",
        labelsize=max(8, font_size - 1),
        color=tick_color,
        labelcolor=tick_color,
        width=axis_line_width,
    )
    ax.tick_params(axis="x", rotation=x_tick_rotation)

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight(weight)
        tick_label.set_color(tick_color)

    ax.grid(
        axis=grid_axis,
        color=grid_color,
        alpha=grid_alpha,
        linewidth=grid_line_width,
    )
    ax.set_axisbelow(True)

    # Cleaner presentation style with black visible axes.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for spine_name in ["bottom", "left"]:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_edgecolor(axis_line_color)
        ax.spines[spine_name].set_color(axis_line_color)
        ax.spines[spine_name].set_linewidth(axis_line_width)

    # Re-apply tick colors after spine styling so external matplotlib styles
    # cannot leave ticks/axes looking gray.
    ax.tick_params(
        axis="both",
        which="both",
        color=tick_color,
        labelcolor=tick_color,
        width=axis_line_width,
    )


def _prepare_clustered_plot_dataframe(
    *,
    df: pd.DataFrame,
    result: Dict[str, Any],
    feature_cols: Sequence[str],
    treatment_col: str,
    baseline_col: str,
    followup_col: str,
    age_col: Optional[str] = "age",
) -> pd.DataFrame:
    """
    Reconstruct the dataframe used by the best feature subset and attach clusters.

    This helper mirrors the row filtering used during feature-subset evaluation.
    It assumes the best cluster labels in result["best_eval"]["cluster_labels"]
    correspond to the rows after dropping missing values in selected features,
    treatment, baseline, and follow-up.

    Parameters
    ----------
    df:
        Original single dataframe.

    result:
        GA result dictionary returned by ClinicalResponseFeatureSelectionGA.run().

    feature_cols:
        Original candidate feature columns. Included for API consistency.

    treatment_col:
        Treatment assignment column.

    baseline_col:
        Baseline outcome column.

    followup_col:
        Follow-up outcome column.

    age_col:
        Optional age column.

    Returns
    -------
    pd.DataFrame
        Plot-ready dataframe containing selected features, treatment, outcomes,
        change score, and cluster assignment.
    """
    # ------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------
    if "selected_cols" not in result:
        raise KeyError("result is missing 'selected_cols'.")

    if "best_eval" not in result or "cluster_labels" not in result["best_eval"]:
        raise KeyError("result['best_eval'] is missing 'cluster_labels'.")

    selected_cols = list(result["selected_cols"])
    labels = np.asarray(result["best_eval"]["cluster_labels"])

    if len(selected_cols) == 0:
        raise ValueError("result['selected_cols'] is empty.")

    required_cols = set(selected_cols + [treatment_col, baseline_col, followup_col])
    missing_cols = sorted(required_cols - set(df.columns))

    if missing_cols:
        raise KeyError(f"df is missing required columns for plotting: {missing_cols}")

    # ------------------------------------------------------------
    # Recreate row-filtered plotting dataframe
    # ------------------------------------------------------------
    needed_cols = list(selected_cols) + [treatment_col, baseline_col, followup_col]

    if age_col is not None and age_col in df.columns:
        needed_cols.append(age_col)

    d = df[needed_cols].copy()

    for col in list(selected_cols) + [baseline_col, followup_col]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    if age_col is not None and age_col in d.columns:
        d[age_col] = pd.to_numeric(d[age_col], errors="coerce")

    d = d.dropna(
        subset=list(selected_cols) + [treatment_col, baseline_col, followup_col]
    ).copy()

    if len(d) != len(labels):
        raise ValueError(
            "The number of plot rows does not match the number of cluster labels. "
            f"Rows={len(d)}, labels={len(labels)}. This can happen if the plotting "
            "dataframe is not the same dataframe used to run the GA."
        )

    d["cluster"] = labels
    d["change"] = d[followup_col] - d[baseline_col]

    return d


def plot_ga_history(
    result: Dict[str, Any],
    *,
    cfg: Optional[ClinicalResponseGAFSConfig] = None,
    title: Optional[str] = None,
    metrics_to_plot: Optional[Sequence[str]] = None,
    plot_value: str = "weighted",
    metric_colors: Optional[Mapping[str, str]] = None,
    figsize: Tuple[float, float] = (9.0, 4.5),
    font_size: float = 12.0,
    legend_loc: str = "best",
    x_tick_rotation: int = 0,
    line_marker: str = "o",
    line_width: float = 2.0,
    y_lim: Optional[Tuple[float, float]] = None,
    annotate_last_value: bool = True,
    annotate_decimals: int = 3,
    annotate_font_size: Optional[float] = None,
    annotate_offset_fraction: float = 0.015,
    axis_line_color: str = "black",
    axis_line_width: float = 1.4,
    tick_color: str = "black",
    grid_color: str = "black",
    grid_alpha: float = 0.18,
    grid_line_width: float = 0.8,
    show: bool = True,
) -> Tuple[pd.DataFrame, Any, Any]:
    """
    Plot GA progress over generations.

    This plot is preset-agnostic and configurable.

    Metric selection
    ----------------
    1. If metrics_to_plot is provided, those metrics are plotted.
    2. Else if cfg.logging_config["metrics_to_show"] exists, those metrics are plotted.
    3. Else if cfg is provided, the active fitness_components are plotted.
    4. Else a default list is used.

    Aliases such as bootstrap_ari_norm and silhouette_norm are supported.

    plot_value controls what is plotted:
        "weighted" : signed weighted contributions to final fitness; default
        "raw"      : raw / normalized metric values

    metric_colors optionally controls line colors. Keys may use either readable
    aliases or internal metric names, for example:
        {
            "best_fitness": "black",
            "bootstrap_ari_norm": "darkblue",
            "silhouette_norm": "darkred",
            "label_alignment_norm": "darkgreen",
            "feature_penalty_norm": "gray",
        }

    Returns
    -------
    Tuple[pd.DataFrame, plt.Figure, plt.Axes]
        plot_df:
            History dataframe columns used for plotting.

        fig, ax:
            Matplotlib figure and axis.
    """
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------
    if "history_df" not in result:
        raise KeyError("result is missing 'history_df'.")

    history_df = result["history_df"].copy()

    if history_df.empty:
        raise ValueError("history_df is empty; nothing to plot.")

    if "generation" not in history_df.columns:
        raise KeyError("history_df is missing required column: 'generation'.")

    plot_value = str(plot_value).lower()
    if plot_value not in {"raw", "weighted"}:
        raise ValueError("plot_value must be either 'raw' or 'weighted'.")

    # ------------------------------------------------------------
    # Choose metrics dynamically
    # ------------------------------------------------------------
    # IMPORTANT:
    # Display names should come from cfg.logging_config whenever cfg is provided,
    # even when metrics_to_plot is passed explicitly. This avoids hard-coded
    # fallback labels such as xTimeARI overriding user labels such as longARI.
    display_names: Dict[str, str] = {}
    if cfg is not None:
        _, display_names = get_active_metric_display_config(cfg)

    if metrics_to_plot is not None:
        metric_pairs = [
            (str(metric_name), canonical_metric_name(str(metric_name)))
            for metric_name in metrics_to_plot
        ]
    elif cfg is not None:
        metric_pairs, display_names = get_active_metric_display_config(cfg)
    else:
        fallback_metrics = [
            "best_fitness",
            "stability_norm",
            "sclin_norm",
            "cluster_norm",
            "feature_penalty_norm",
        ]
        metric_pairs = [
            (metric_name, canonical_metric_name(metric_name))
            for metric_name in fallback_metrics
        ]

    # Always include best_fitness first unless user explicitly excluded it.
    if metrics_to_plot is None:
        if not any(internal == "best_fitness" for _, internal in metric_pairs):
            metric_pairs = [("best_fitness", "best_fitness")] + metric_pairs

    available_pairs: List[Tuple[str, str, str]] = []
    for user_metric_name, internal_metric_name in metric_pairs:
        # best_fitness is already the total weighted objective, so always plot it
        # directly. It does not have a "__weighted" companion column.
        if internal_metric_name == "best_fitness":
            if "best_fitness" in history_df.columns:
                available_pairs.append((user_metric_name, "best_fitness", internal_metric_name))
            continue

        if plot_value == "weighted":
            user_weighted_col = f"{user_metric_name}__weighted"
            internal_weighted_col = f"{internal_metric_name}__weighted"

            if user_weighted_col in history_df.columns:
                available_pairs.append((user_metric_name, user_weighted_col, internal_metric_name))
            elif internal_weighted_col in history_df.columns:
                available_pairs.append((user_metric_name, internal_weighted_col, internal_metric_name))
        else:
            if user_metric_name in history_df.columns:
                available_pairs.append((user_metric_name, user_metric_name, internal_metric_name))
            elif internal_metric_name in history_df.columns:
                available_pairs.append((user_metric_name, internal_metric_name, internal_metric_name))

    if len(available_pairs) == 0:
        raise ValueError(
            "No requested plot metrics were found in history_df. "
            f"Available columns: {list(history_df.columns)}"
        )

    plot_cols = ["generation"] + [history_col for _, history_col, _ in available_pairs]
    plot_df = history_df[plot_cols].copy()

    # ------------------------------------------------------------
    # Resolve optional user-provided line colors
    # ------------------------------------------------------------
    metric_colors = dict(metric_colors or {})

    def _resolve_metric_color(
        user_metric_name: str,
        internal_metric_name: str,
        label: str,
    ) -> Optional[str]:
        """
        Resolve a metric color from several reasonable key styles.

        This supports both user-facing aliases and internal names:
            bootstrap_ari_norm or stability_norm
            silhouette_norm or cluster_norm
        """
        possible_keys = [
            user_metric_name,
            internal_metric_name,
            label,
            default_metric_display_label(user_metric_name),
            default_metric_display_label(internal_metric_name),
        ]

        for key in possible_keys:
            if key in metric_colors:
                return metric_colors[key]

        return None

    # ------------------------------------------------------------
    # Plot available metrics
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    for user_metric_name, history_col, internal_metric_name in available_pairs:
        label = display_names.get(
            user_metric_name,
            default_metric_display_label(user_metric_name),
        )
        color = _resolve_metric_color(user_metric_name, internal_metric_name, label)

        plot_kwargs: Dict[str, Any] = {
            "marker": line_marker,
            "linewidth": line_width,
            "label": label,
        }

        if color is not None:
            plot_kwargs["color"] = color

        ax.plot(
            plot_df["generation"],
            plot_df[history_col],
            **plot_kwargs,
        )

    if title is None:
        if plot_value == "weighted":
            title = "GA Fitness and Weighted Metric Contributions"
        else:
            title = "GA Raw Metric Progress Over Generations"

    if y_lim is not None:
        ax.set_ylim(*y_lim)

    # ------------------------------------------------------------
    # Optional final-value annotations
    # ------------------------------------------------------------
    if annotate_last_value:
        ann_fs = annotate_font_size if annotate_font_size is not None else max(8, font_size - 3)

        y_values_for_offset: List[float] = []
        for _, history_col, _ in available_pairs:
            vals = pd.to_numeric(plot_df[history_col], errors="coerce").dropna()
            if len(vals) > 0:
                y_values_for_offset.extend(vals.astype(float).tolist())

        if y_values_for_offset:
            y_range = max(y_values_for_offset) - min(y_values_for_offset)
            if y_range == 0:
                y_range = max(abs(y_values_for_offset[0]), 1.0)
        else:
            y_range = 1.0

        offset = annotate_offset_fraction * y_range
        last_x = plot_df["generation"].iloc[-1]

        for user_metric_name, history_col, internal_metric_name in available_pairs:
            series = pd.to_numeric(plot_df[history_col], errors="coerce")
            if series.dropna().empty:
                continue

            last_y = float(series.iloc[-1])
            label = display_names.get(
                user_metric_name,
                display_names.get(
                    internal_metric_name,
                    default_metric_display_label(user_metric_name),
                ),
            )

            color = _resolve_metric_color(user_metric_name, internal_metric_name, label)

            ax.text(
                last_x,
                last_y + offset,
                f"{label}: {last_y:.{annotate_decimals}f}",
                fontsize=ann_fs,
                fontweight="bold",
                color=color if color is not None else "black",
                ha="left",
                va="bottom",
            )

    _apply_clinical_response_axis_style(
        ax,
        title=title,
        xlabel="Generation",
        ylabel="Fitness / weighted contribution" if plot_value == "weighted" else "Raw / normalized score",
        font_size=font_size,
        grid_axis="both",
        x_tick_rotation=x_tick_rotation,
        bold=True,
        axis_line_color=axis_line_color,
        axis_line_width=axis_line_width,
        tick_color=tick_color,
        grid_color=grid_color,
        grid_alpha=grid_alpha,
        grid_line_width=grid_line_width,
    )

    legend = ax.legend(
        frameon=False,
        fontsize=max(8, font_size - 1),
        loc=legend_loc,
    )
    if legend is not None:
        for text_obj in legend.get_texts():
            text_obj.set_fontweight("bold")

    fig.tight_layout()

    if show:
        plt.show()

    return plot_df, fig, ax


def plot_treatment_response_by_cluster(
    *,
    df: pd.DataFrame,
    result: Dict[str, Any],
    feature_cols: Sequence[str],
    treatment_col: str = "treatment_arm",
    baseline_col: str = "vi3_v1_soc_ss",
    followup_col: str = "vi3_v7_soc_ss",
    age_col: Optional[str] = "age",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (9.0, 5.0),
    font_size: float = 12.0,
    treatment_colors: Optional[Mapping[Any, str]] = None,
    default_colors: Tuple[str, ...] = ("darkred", "darkblue", "darkgreen", "darkorange", "purple"),
    annotate: bool = True,
    annotate_font_size: Optional[float] = None,
    annotate_offset_fraction: float = 0.03,
    show: bool = True,
) -> Tuple[pd.DataFrame, Any, Any]:
    """
    Plot baseline-to-follow-up change by treatment arm inside each cluster.

    This is the simplest clinical interpretation plot. It answers:

        Within each discovered cluster, did Treatment and Placebo change differently?

    Parameters
    ----------
    df:
        Original single dataframe used for GA.

    result:
        GA result dictionary returned by ClinicalResponseFeatureSelectionGA.run().

    feature_cols:
        Original candidate feature columns.

    treatment_col:
        Treatment assignment column.

    baseline_col:
        Baseline outcome column.

    followup_col:
        Follow-up outcome column.

    age_col:
        Optional age column.

    title:
        Optional custom title.

    figsize:
        Matplotlib figure size.

    font_size:
        Base font size.

    treatment_colors:
        Optional mapping from treatment labels to colors.

    default_colors:
        Colors assigned to treatment arms when treatment_colors is not provided.

    annotate:
        If True, annotate bars with n per cluster/treatment arm.

    annotate_font_size:
        Font size for annotations. If None, uses max(8, font_size - 3).

    annotate_offset_fraction:
        Annotation offset as fraction of y-axis range.

    show:
        If True, display the plot in the notebook.

    Returns
    -------
    Tuple[pd.DataFrame, plt.Figure, plt.Axes]
        plot_df:
            Summary dataframe used for plotting.

        fig, ax:
            Matplotlib figure and axis.
    """
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------
    # Prepare clustered dataframe
    # ------------------------------------------------------------
    d = _prepare_clustered_plot_dataframe(
        df=df,
        result=result,
        feature_cols=feature_cols,
        treatment_col=treatment_col,
        baseline_col=baseline_col,
        followup_col=followup_col,
        age_col=age_col,
    )

    # ------------------------------------------------------------
    # Summarize change by cluster and treatment arm
    # ------------------------------------------------------------
    plot_df = (
        d.groupby(["cluster", treatment_col], dropna=False)["change"]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .rename(columns={"mean": "change_mean", "sem": "change_sem", "count": "n"})
    )

    clusters = sorted(plot_df["cluster"].dropna().unique())
    arms = list(plot_df[treatment_col].dropna().unique())

    if len(clusters) == 0:
        raise ValueError("No clusters available to plot.")

    if len(arms) == 0:
        raise ValueError("No treatment arms available to plot.")

    # ------------------------------------------------------------
    # Color handling
    # ------------------------------------------------------------
    if treatment_colors is None:
        treatment_colors = {
            arm: default_colors[idx % len(default_colors)]
            for idx, arm in enumerate(arms)
        }

    # ------------------------------------------------------------
    # Plot grouped bars
    # ------------------------------------------------------------
    x = np.arange(len(clusters), dtype=float)
    width = 0.8 / max(len(arms), 1)

    fig, ax = plt.subplots(figsize=figsize)

    all_bar_values: List[float] = []

    for arm_idx, arm in enumerate(arms):
        means: List[float] = []
        sems: List[float] = []
        counts: List[int] = []

        for cluster in clusters:
            row = plot_df[
                (plot_df["cluster"] == cluster)
                & (plot_df[treatment_col] == arm)
            ]

            if len(row) == 0:
                means.append(np.nan)
                sems.append(np.nan)
                counts.append(0)
            else:
                means.append(float(row["change_mean"].iloc[0]))
                sem_value = row["change_sem"].iloc[0]
                sems.append(float(sem_value) if not pd.isna(sem_value) else 0.0)
                counts.append(int(row["n"].iloc[0]))

        all_bar_values.extend([v for v in means if np.isfinite(v)])

        offset = (arm_idx - (len(arms) - 1) / 2) * width

        bars = ax.bar(
            x + offset,
            means,
            width=width,
            yerr=sems,
            capsize=4,
            label=str(arm),
            color=treatment_colors.get(arm, default_colors[arm_idx % len(default_colors)]),
            alpha=0.90,
        )

        # ------------------------------------------------------------
        # Annotate bars with sample sizes
        # ------------------------------------------------------------
        if annotate:
            ann_fs = annotate_font_size if annotate_font_size is not None else max(8, font_size - 3)

            finite_values = [v for v in all_bar_values if np.isfinite(v)]
            if finite_values:
                y_range = max(finite_values) - min(finite_values)
                if y_range == 0:
                    y_range = max(abs(finite_values[0]), 1.0)
            else:
                y_range = 1.0

            offset_y = annotate_offset_fraction * y_range

            for bar, mean_val, count_val in zip(bars, means, counts):
                if not np.isfinite(mean_val):
                    continue

                x_text = bar.get_x() + bar.get_width() / 2.0
                y_text = mean_val + offset_y if mean_val >= 0 else mean_val - offset_y
                va = "bottom" if mean_val >= 0 else "top"

                ax.text(
                    x_text,
                    y_text,
                    f"n={count_val}",
                    ha="center",
                    va=va,
                    fontsize=ann_fs,
                    fontweight="bold",
                )

    ax.axhline(0, linestyle="--", linewidth=1.2, color="black")

    if title is None:
        title = "Treatment Response by Cluster"

    _apply_clinical_response_axis_style(
        ax,
        title=title,
        xlabel="Cluster",
        ylabel=f"Change: {followup_col} - {baseline_col}",
        font_size=font_size,
        grid_axis="y",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {int(c)}" for c in clusters])
    ax.legend(title=treatment_col, frameon=False, fontsize=max(8, font_size - 1))

    fig.tight_layout()

    if show:
        plt.show()

    return plot_df, fig, ax


def plot_cluster_treatment_effects(
    result: Dict[str, Any],
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8.5, 5.0),
    font_size: float = 12.0,
    positive_color: str = "darkred",
    negative_color: str = "darkblue",
    zero_color: str = "gray",
    ci_color: str = "black",
    annotate: bool = True,
    annotate_p_values: bool = True,
    annotate_font_size: Optional[float] = None,
    annotate_decimals: int = 2,
    show: bool = True,
) -> Tuple[pd.DataFrame, Any, Any]:
    """
    Plot adjusted treatment effects by cluster with confidence intervals.

    This is the compact one-endpoint version of the teammate-style coefficient
    plot. It visualizes the ANCOVA treatment beta from result["effects_df"].

    Parameters
    ----------
    result:
        GA result dictionary returned by ClinicalResponseFeatureSelectionGA.run().

    title:
        Optional custom title.

    figsize:
        Matplotlib figure size.

    font_size:
        Base font size.

    positive_color:
        Color for positive treatment effects.

    negative_color:
        Color for negative treatment effects.

    zero_color:
        Color for zero treatment effects.

    ci_color:
        Confidence interval color.

    annotate:
        If True, annotate each point with beta value.

    annotate_p_values:
        If True and p_value exists, include p-values in annotations.

    annotate_font_size:
        Font size for annotations. If None, uses max(8, font_size - 3).

    annotate_decimals:
        Number of decimal places for beta annotations.

    show:
        If True, display the plot in the notebook.

    Returns
    -------
    Tuple[pd.DataFrame, plt.Figure, plt.Axes]
        plot_df:
            Effects dataframe used for plotting.

        fig, ax:
            Matplotlib figure and axis.
    """
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------
    if "effects_df" not in result:
        raise KeyError("result is missing 'effects_df'.")

    effects_df = result["effects_df"].copy()

    required_cols = {"cluster", "beta", "ci_low", "ci_high", "n_treated", "n_control"}
    missing_cols = sorted(required_cols - set(effects_df.columns))

    if missing_cols:
        raise KeyError(f"effects_df is missing required columns: {missing_cols}")

    if effects_df.empty:
        raise ValueError("effects_df is empty; nothing to plot.")

    # ------------------------------------------------------------
    # Prepare plotting dataframe
    # ------------------------------------------------------------
    plot_df = effects_df.copy()
    plot_df["beta"] = pd.to_numeric(plot_df["beta"], errors="coerce")
    plot_df["ci_low"] = pd.to_numeric(plot_df["ci_low"], errors="coerce")
    plot_df["ci_high"] = pd.to_numeric(plot_df["ci_high"], errors="coerce")

    if "p_value" in plot_df.columns:
        plot_df["p_value"] = pd.to_numeric(plot_df["p_value"], errors="coerce")

    plot_df = plot_df.sort_values("cluster").reset_index(drop=True)

    if plot_df["beta"].isna().all():
        raise ValueError("All beta values are NaN; cannot plot treatment effects.")

    x = np.arange(len(plot_df))
    beta = plot_df["beta"].to_numpy(dtype=float)
    ci_low = plot_df["ci_low"].to_numpy(dtype=float)
    ci_high = plot_df["ci_high"].to_numpy(dtype=float)

    lower_err = beta - ci_low
    upper_err = ci_high - beta
    yerr = np.vstack([lower_err, upper_err])

    colors = [
        positive_color if value > 0 else negative_color if value < 0 else zero_color
        for value in beta
    ]

    # ------------------------------------------------------------
    # Plot adjusted effects with confidence intervals
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    for idx, value in enumerate(beta):
        ax.errorbar(
            x[idx],
            value,
            yerr=np.array([[lower_err[idx]], [upper_err[idx]]]),
            fmt="o",
            color=colors[idx],
            ecolor=ci_color,
            capsize=5,
            markersize=7,
            linewidth=1.5,
        )

    ax.axhline(0, linestyle="--", linewidth=1.2, color="black")

    cluster_labels = [
        f"Cluster {int(row['cluster'])}\n"
        f"T={int(row['n_treated'])}, C={int(row['n_control'])}"
        for _, row in plot_df.iterrows()
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(cluster_labels)

    # ------------------------------------------------------------
    # Annotate beta values and optional p-values
    # ------------------------------------------------------------
    if annotate:
        ann_fs = annotate_font_size if annotate_font_size is not None else max(8, font_size - 3)

        finite_beta = beta[np.isfinite(beta)]
        y_range = np.nanmax(finite_beta) - np.nanmin(finite_beta) if len(finite_beta) else 1.0
        if not np.isfinite(y_range) or y_range == 0:
            y_range = max(np.nanmax(np.abs(finite_beta)), 1.0) if len(finite_beta) else 1.0

        offset = 0.05 * y_range

        for idx, row in plot_df.iterrows():
            value = row["beta"]

            if pd.isna(value):
                continue

            if value >= 0:
                y_text = value + offset
                va = "bottom"
            else:
                y_text = value - offset
                va = "top"

            label = f"{value:.{annotate_decimals}f}"

            if annotate_p_values and "p_value" in plot_df.columns and not pd.isna(row.get("p_value")):
                label = f"{label}\np={row['p_value']:.3f}"

            ax.text(
                x[idx],
                y_text,
                label,
                ha="center",
                va=va,
                fontsize=ann_fs,
                fontweight="bold",
            )

    if title is None:
        title = "Adjusted Treatment Effect by Cluster"

    _apply_clinical_response_axis_style(
        ax,
        title=title,
        xlabel="Cluster",
        ylabel="Adjusted treatment effect beta",
        font_size=font_size,
        grid_axis="y",
    )

    fig.tight_layout()

    if show:
        plt.show()

    return plot_df, fig, ax


def make_clinical_response_plots(
    *,
    result: Dict[str, Any],
    cfg: Optional[ClinicalResponseGAFSConfig] = None,
    metrics_to_plot: Optional[Sequence[str]] = None,
    plot_value: str = "weighted",
    metric_colors: Optional[Mapping[str, str]] = None,
    figsize: Tuple[float, float] = (9.0, 4.5),
    font_size: float = 12.0,
    legend_loc: str = "best",
    x_tick_rotation: int = 0,
    line_marker: str = "o",
    line_width: float = 2.0,
    y_lim: Optional[Tuple[float, float]] = None,
    annotate_last_value: bool = True,
    annotate_decimals: int = 3,
    annotate_font_size: Optional[float] = None,
    axis_line_color: str = "black",
    axis_line_width: float = 1.4,
    tick_color: str = "black",
    grid_color: str = "black",
    grid_alpha: float = 0.18,
    grid_line_width: float = 0.8,
    show: bool = True,
) -> Dict[str, Tuple[pd.DataFrame, Any, Any]]:
    """
    Create the standard notebook plots for the GA run.

    For now, this intentionally creates only the general GA history plot.

    Why only GA history?
    --------------------
    The GA history plot applies to all fitness presets:
        - balanced_clinical
        - stability_only
        - cluster_quality_only

    The clinical plots below are intentionally not called here because they only
    apply to clinical presets such as balanced_clinical:
        - plot_treatment_response_by_cluster(...)
        - plot_cluster_treatment_effects(...)

    Those functions remain in the module for future/manual use, but this main
    helper stays simple and preset-agnostic.

    Parameters
    ----------
    result:
        GA result dictionary returned by ClinicalResponseFeatureSelectionGA.run().

    cfg:
        Optional GA configuration. When provided, plot metrics follow
        cfg.logging_config or the active fitness_components.

    metrics_to_plot:
        Optional list of history_df metric columns to plot. Overrides cfg.

    metric_colors:
        Optional mapping from metric name to line color. Keys can use readable
        aliases such as "bootstrap_ari_norm" and "silhouette_norm", or internal
        names such as "stability_norm" and "cluster_norm".

    figsize:
        Matplotlib figure size.

    font_size:
        Base font size.

    show:
        If True, display the plot in the notebook.

    Returns
    -------
    dict
        {
            "ga_history": (plot_df, fig, ax)
        }
    """
    outputs: Dict[str, Tuple[pd.DataFrame, Any, Any]] = {}

    outputs["ga_history"] = plot_ga_history(
        result=result,
        cfg=cfg,
        metrics_to_plot=metrics_to_plot,
        plot_value=plot_value,
        metric_colors=metric_colors,
        figsize=figsize,
        font_size=font_size,
        legend_loc=legend_loc,
        x_tick_rotation=x_tick_rotation,
        line_marker=line_marker,
        line_width=line_width,
        y_lim=y_lim,
        annotate_last_value=annotate_last_value,
        annotate_decimals=annotate_decimals,
        annotate_font_size=annotate_font_size,
        axis_line_color=axis_line_color,
        axis_line_width=axis_line_width,
        tick_color=tick_color,
        grid_color=grid_color,
        grid_alpha=grid_alpha,
        grid_line_width=grid_line_width,
        show=show,
    )

    return outputs



# =============================================================================
# Convenience constructors and helpers
# =============================================================================

def make_clinical_response_ga(
    *,
    df: Optional[pd.DataFrame] = None,
    cfg: Optional[ClinicalResponseGAFSConfig] = None,
    feature_cols: Optional[Sequence[str]] = None,
    treatment_col: str = "treatment_arm",
    outdir: Optional[str] = None,
    save_outputs: bool = False,
    age_col: Optional[str] = "age",
    treated_label: str = "Treatment",
    control_label: str = "Placebo",
) -> ClinicalResponseFeatureSelectionGA:
    """
    Convenience constructor for the single-timepoint GA runner.

    Preferred config-driven usage
    -----------------------------
    Put the dataframe under the active preset config:

        cfg.fitness_preset_config[cfg.fitness_preset]["data_config"]["df"] = df

    Then call:

        ga_runner = make_clinical_response_ga(cfg=cfg, save_outputs=False)

    Backward compatibility
    ----------------------
    The older style still works:

        ga_runner = make_clinical_response_ga(df=df, cfg=cfg, outdir="...", save_outputs=True)

    Resolution order
    ----------------
    1. Explicit df argument.
    2. Active preset config data_config["df"] or input_config["df"].
    """
    if cfg is None:
        cfg = ClinicalResponseGAFSConfig()

    resolved_df = resolve_single_timepoint_dataframe(cfg, df=df)

    return ClinicalResponseFeatureSelectionGA(
        df=resolved_df,
        cfg=cfg,
        feature_cols=feature_cols,
        treatment_col=treatment_col,
        outdir=outdir,
        save_outputs=save_outputs,
        age_col=age_col,
        treated_label=treated_label,
        control_label=control_label,
    )

def evaluate_mask_clinical_response(
    *,
    mask: Sequence[Any],
    cfg: ClinicalResponseGAFSConfig,
    df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[Sequence[str]] = None,
    treatment_col: str = "treatment_arm",
    age_col: Optional[str] = "age",
    treated_label: str = "Treatment",
    control_label: str = "Placebo",
) -> Dict[str, Any]:
    """
    Evaluate one feature mask without running the GA.

    This is useful for smoke tests and oracle comparisons.
    """
    apply_active_fitness_preset_config(cfg)
    resolved_df = resolve_single_timepoint_dataframe(cfg, df=df)
    resolved_feature_cols = resolve_feature_config(cfg, feature_cols=feature_cols)

    clinical = resolve_clinical_config(
        cfg,
        treatment_col=treatment_col,
        age_col=age_col,
        treated_label=treated_label,
        control_label=control_label,
        require_clinical=active_fitness_uses_sclinical(cfg),
    )

    return evaluate_feature_subset_clinical_response(
        df=resolved_df,
        feature_cols=resolved_feature_cols,
        mask=mask,
        cfg=cfg,
        treatment_col=clinical["treatment_col"],
        age_col=clinical["age_col"],
        treated_label=clinical["treated_label"],
        control_label=clinical["control_label"],
    )


def evaluate_true_informative_mask(
    *,
    true_informative: Sequence[str],
    cfg: ClinicalResponseGAFSConfig,
    df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[Sequence[str]] = None,
    treatment_col: str = "treatment_arm",
    age_col: Optional[str] = "age",
) -> Dict[str, Any]:
    """
    Evaluate a mask selecting all known true-informative features.

    This is intended for synthetic validation only.

    Parameters
    ----------
    df:
        Single dataframe.

    feature_cols:
        All candidate feature columns.

    true_informative:
        Names of known informative features.

    cfg:
        Clinical-response GA configuration.

    treatment_col:
        Treatment assignment column.

    age_col:
        Optional age column.

    Returns
    -------
    dict
        Evaluation result for the true-informative mask.
    """
    resolved_feature_cols = resolve_feature_config(cfg, feature_cols=feature_cols)

    informative_set = set(true_informative)
    mask = np.array([1 if col in informative_set else 0 for col in resolved_feature_cols], dtype=int)

    return evaluate_mask_clinical_response(
        df=df,
        feature_cols=resolved_feature_cols,
        mask=mask,
        cfg=cfg,
        treatment_col=treatment_col,
        age_col=age_col,
    )


# =============================================================================
# Synthetic clustering dataset generator
# =============================================================================

def make_synthetic_clustering_feature_selection_dataset(
    *,
    n_samples: int = 300,
    n_features: int = 20,
    n_informative: int = 10,
    n_clusters: int = 2,
    cluster_std: float = 1.0,
    cluster_separation: float = 3.0,
    random_state: Optional[int] = None,
    feature_prefix: str = "feature",
    add_collinear_features: bool = False,
    n_collinear: int = 0,
    collinearity_strength: float = 0.98,
    collinear_noise_std: float = 0.02,
) -> Dict[str, Any]:
    """
    Generate synthetic data specifically for clustering feature-selection tests.

    This function is different from a classification-style synthetic dataset.

    Classification-style data can contain features that help a classifier predict
    a label, but those labels may not form clean geometric clusters for KMeans.
    This function intentionally creates true cluster structure in the informative
    features so clustering algorithms such as KMeans can recover the labels.

    Parameters
    ----------
    n_samples:
        Number of rows / subjects.

    n_features:
        Total number of feature columns.

    n_informative:
        Number of features that contain true cluster signal.

    n_clusters:
        Number of true clusters / labels to generate.

    cluster_std:
        Standard deviation of each cluster around its center.
        Smaller values make clusters tighter and easier to recover.

    cluster_separation:
        Distance between cluster centers in informative feature dimensions.
        Larger values make clusters easier to recover.

    random_state:
        Random seed for reproducibility.

    feature_prefix:
        Prefix used to create feature names, e.g. feature_0.

    add_collinear_features:
        If True, replace some non-informative features with noisy copies of
        informative features. This is useful for testing whether the GA handles
        redundant features.

    n_collinear:
        Number of collinear/redundant features to create.

    collinearity_strength:
        Multiplicative strength of the copied informative feature.

    collinear_noise_std:
        Noise added to each collinear feature.

    Returns
    -------
    dict
        X:
            NumPy array of shape (n_samples, n_features).

        y:
            Integer cluster labels of shape (n_samples,).

        feature_names:
            List of feature names.

        true_informative:
            Set of informative feature names.

        true_noise:
            Set of pure-noise feature names.

        true_collinear:
            Set of collinear feature names, if requested.

        cluster_centers:
            Array of cluster centers used for informative dimensions.

        params:
            Dictionary of generation parameters.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if n_features <= 0:
        raise ValueError("n_features must be positive.")

    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")

    if not (1 <= n_informative <= n_features):
        raise ValueError("n_informative must be between 1 and n_features.")

    if n_collinear < 0:
        raise ValueError("n_collinear must be non-negative.")

    if add_collinear_features and n_collinear > (n_features - n_informative):
        raise ValueError(
            "n_collinear cannot exceed the number of non-informative feature slots."
        )

    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Create balanced true cluster labels
    # ------------------------------------------------------------------
    y = np.arange(n_samples) % n_clusters
    rng.shuffle(y)

    # ------------------------------------------------------------------
    # Create informative cluster centers
    # ------------------------------------------------------------------
    # Each cluster gets a center vector in the informative dimensions.
    # We use random directions and then scale them by cluster_separation.
    centers = rng.normal(loc=0.0, scale=1.0, size=(n_clusters, n_informative))

    # Normalize centers so cluster separation has a clear effect.
    center_norms = np.linalg.norm(centers, axis=1, keepdims=True)
    centers = centers / np.maximum(center_norms, 1e-12)
    centers = centers * cluster_separation

    # ------------------------------------------------------------------
    # Generate informative features
    # ------------------------------------------------------------------
    X_informative = np.empty((n_samples, n_informative), dtype=float)

    for cluster_id in range(n_clusters):
        idx = np.where(y == cluster_id)[0]
        X_informative[idx, :] = rng.normal(
            loc=centers[cluster_id],
            scale=cluster_std,
            size=(len(idx), n_informative),
        )

    # ------------------------------------------------------------------
    # Generate non-informative noise features
    # ------------------------------------------------------------------
    n_noise = n_features - n_informative
    X_noise = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_noise))

    X = np.concatenate([X_informative, X_noise], axis=1)

    feature_names = [f"{feature_prefix}_{i}" for i in range(n_features)]

    informative_idx = list(range(n_informative))
    true_informative = {feature_names[i] for i in informative_idx}

    collinear_idx: List[int] = []

    # ------------------------------------------------------------------
    # Optionally add collinear copies of informative features
    # ------------------------------------------------------------------
    if add_collinear_features and n_collinear > 0:
        available_noise_idx = list(range(n_informative, n_features))
        selected_noise_idx = available_noise_idx[:n_collinear]

        source_idx = rng.choice(informative_idx, size=n_collinear, replace=True)

        for target_col, source_col in zip(selected_noise_idx, source_idx):
            X[:, target_col] = (
                collinearity_strength * X[:, source_col]
                + rng.normal(0.0, collinear_noise_std, size=n_samples)
            )
            collinear_idx.append(target_col)

    true_collinear = {feature_names[i] for i in collinear_idx}
    true_noise = set(feature_names[n_informative:]) - true_collinear

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "true_informative": true_informative,
        "true_noise": true_noise,
        "true_collinear": true_collinear,
        "cluster_centers": centers,
        "params": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_informative": n_informative,
            "n_clusters": n_clusters,
            "cluster_std": cluster_std,
            "cluster_separation": cluster_separation,
            "random_state": random_state,
            "feature_prefix": feature_prefix,
            "add_collinear_features": add_collinear_features,
            "n_collinear": n_collinear,
            "collinearity_strength": collinearity_strength,
            "collinear_noise_std": collinear_noise_std,
        },
    }



def make_synthetic_longitudinal_clustering_feature_selection_dataset(
    *,
    n_samples: int = 300,
    n_features: int = 20,
    n_informative: int = 10,
    n_clusters: int = 2,
    cluster_std: float = 1.0,
    week6_cluster_std: Optional[float] = None,
    cluster_separation: float = 3.0,
    switch_probability: float = 0.15,
    week6_center_drift: float = 0.0,
    random_state: Optional[int] = None,
    feature_prefix: str = "feature",
    subject_id_col: str = "subject_id",
    baseline_label_col: str = "label_baseline",
    week6_label_col: str = "label_week6",
    subject_id_prefix: str = "S",
    add_collinear_features: bool = False,
    n_collinear: int = 0,
    collinearity_strength: float = 0.98,
    collinear_noise_std: float = 0.02,
) -> Dict[str, Any]:
    """
    Generate paired baseline and Week 6 synthetic data for longitudinal
    clustering feature-selection tests.

    This helper creates a subject-level paired dataset for testing the
    longitudinal unsupervised GA pathway. It intentionally uses the same feature
    names at baseline and Week 6 so the default longitudinal feature mapping is
    simple.

    Design
    ------
    1. Baseline data are generated with
       make_synthetic_clustering_feature_selection_dataset(...).
    2. Week 6 labels are copied from baseline labels, then a configurable
       fraction of subjects switch to a different cluster.
    3. Week 6 informative features are generated around the same cluster centers
       used at baseline, optionally with small center drift.
    4. Week 6 noise and optional collinear features are regenerated independently.

    Parameters
    ----------
    switch_probability:
        Probability that a subject changes hidden cluster membership between
        baseline and Week 6. Lower values produce higher cross-time ARI.

    week6_center_drift:
        Standard deviation of random drift added to Week 6 cluster centers in
        informative dimensions. Use 0.0 to keep the same centers across time.

    Returns
    -------
    dict
        df_baseline:
            Baseline dataframe with feature columns, subject_id_col, and
            baseline_label_col.

        df_week6:
            Week 6 dataframe with feature columns, subject_id_col, and
            week6_label_col.

        feature_cols:
            Candidate feature columns shared by both timepoints.

        membership_df:
            Subject-level true baseline/Week 6 labels and whether the hidden
            synthetic cluster changed.

        expected_cross_time_ari:
            ARI between true baseline labels and true Week 6 labels. This is a
            useful sanity-check target, not a guarantee that KMeans will recover
            exactly the same value.
    """
    if week6_cluster_std is None:
        week6_cluster_std = cluster_std

    if not (0.0 <= switch_probability <= 1.0):
        raise ValueError("switch_probability must be between 0 and 1.")

    if week6_center_drift < 0:
        raise ValueError("week6_center_drift must be non-negative.")

    baseline_data = make_synthetic_clustering_feature_selection_dataset(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_clusters=n_clusters,
        cluster_std=cluster_std,
        cluster_separation=cluster_separation,
        random_state=random_state,
        feature_prefix=feature_prefix,
        add_collinear_features=add_collinear_features,
        n_collinear=n_collinear,
        collinearity_strength=collinearity_strength,
        collinear_noise_std=collinear_noise_std,
    )

    rng = np.random.default_rng(None if random_state is None else random_state + 10_000)

    feature_cols = list(baseline_data["feature_names"])
    y_baseline = np.asarray(baseline_data["y"], dtype=int)
    centers_baseline = np.asarray(baseline_data["cluster_centers"], dtype=float)

    # ------------------------------------------------------------------
    # Generate Week 6 hidden labels by switching a subset of subjects.
    # ------------------------------------------------------------------
    y_week6 = y_baseline.copy()
    switch_mask = rng.random(n_samples) < switch_probability

    if n_clusters == 2:
        y_week6[switch_mask] = 1 - y_week6[switch_mask]
    else:
        for idx in np.where(switch_mask)[0]:
            current = int(y_week6[idx])
            possible = [cluster for cluster in range(n_clusters) if cluster != current]
            y_week6[idx] = int(rng.choice(possible))

    # ------------------------------------------------------------------
    # Generate Week 6 feature matrix using baseline centers plus optional drift.
    # ------------------------------------------------------------------
    centers_week6 = centers_baseline.copy()
    if week6_center_drift > 0:
        centers_week6 = centers_week6 + rng.normal(
            loc=0.0,
            scale=week6_center_drift,
            size=centers_week6.shape,
        )

    X_week6_informative = np.empty((n_samples, n_informative), dtype=float)
    for cluster_id in range(n_clusters):
        idx = np.where(y_week6 == cluster_id)[0]
        X_week6_informative[idx, :] = rng.normal(
            loc=centers_week6[cluster_id],
            scale=week6_cluster_std,
            size=(len(idx), n_informative),
        )

    n_noise = n_features - n_informative
    X_week6_noise = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_noise))
    X_week6 = np.concatenate([X_week6_informative, X_week6_noise], axis=1)

    # Recreate collinear feature structure at Week 6 using the same target
    # columns identified in the baseline generator. Source columns are sampled
    # reproducibly from informative features so the redundant-feature challenge
    # exists at both timepoints.
    if add_collinear_features and n_collinear > 0:
        informative_idx = list(range(n_informative))
        available_noise_idx = list(range(n_informative, n_features))
        selected_noise_idx = available_noise_idx[:n_collinear]
        source_idx = rng.choice(informative_idx, size=n_collinear, replace=True)

        for target_col, source_col in zip(selected_noise_idx, source_idx):
            X_week6[:, target_col] = (
                collinearity_strength * X_week6[:, source_col]
                + rng.normal(0.0, collinear_noise_std, size=n_samples)
            )

    # ------------------------------------------------------------------
    # Build paired dataframes.
    # ------------------------------------------------------------------
    subject_ids = [f"{subject_id_prefix}{idx:04d}" for idx in range(n_samples)]

    df_baseline = pd.DataFrame(baseline_data["X"], columns=feature_cols)
    df_baseline[subject_id_col] = subject_ids
    df_baseline[baseline_label_col] = y_baseline

    df_week6 = pd.DataFrame(X_week6, columns=feature_cols)
    df_week6[subject_id_col] = subject_ids
    df_week6[week6_label_col] = y_week6

    membership_df = pd.DataFrame(
        {
            subject_id_col: subject_ids,
            baseline_label_col: y_baseline,
            week6_label_col: y_week6,
            "hidden_cluster_changed": y_baseline != y_week6,
        }
    )

    expected_cross_time_ari = float(adjusted_rand_score(y_baseline, y_week6))
    expected_same_cluster_rate = float(np.mean(y_baseline == y_week6))

    return {
        "df_baseline": df_baseline,
        "df_week6": df_week6,
        "feature_cols": feature_cols,
        "week6_feature_cols": feature_cols,
        "subject_id_col": subject_id_col,
        "baseline_label_col": baseline_label_col,
        "week6_label_col": week6_label_col,
        "y_baseline": y_baseline,
        "y_week6": y_week6,
        "membership_df": membership_df,
        "expected_cross_time_ari": expected_cross_time_ari,
        "expected_same_cluster_rate": expected_same_cluster_rate,
        "true_informative": baseline_data.get("true_informative"),
        "true_noise": baseline_data.get("true_noise"),
        "true_collinear": baseline_data.get("true_collinear"),
        "baseline_cluster_centers": centers_baseline,
        "week6_cluster_centers": centers_week6,
        "baseline_data": baseline_data,
        "params": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_informative": n_informative,
            "n_clusters": n_clusters,
            "cluster_std": cluster_std,
            "week6_cluster_std": week6_cluster_std,
            "cluster_separation": cluster_separation,
            "switch_probability": switch_probability,
            "week6_center_drift": week6_center_drift,
            "random_state": random_state,
            "feature_prefix": feature_prefix,
            "subject_id_col": subject_id_col,
            "baseline_label_col": baseline_label_col,
            "week6_label_col": week6_label_col,
            "add_collinear_features": add_collinear_features,
            "n_collinear": n_collinear,
            "collinearity_strength": collinearity_strength,
            "collinear_noise_std": collinear_noise_std,
        },
    }


def summarize_synthetic_longitudinal_membership(
    longitudinal_data: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Summarize true synthetic baseline-to-Week 6 membership changes.

    This helper is for sanity checks before running the longitudinal GA.
    """
    membership_df = pd.DataFrame(longitudinal_data["membership_df"]).copy()
    baseline_label_col = str(longitudinal_data.get("baseline_label_col", "label_baseline"))
    week6_label_col = str(longitudinal_data.get("week6_label_col", "label_week6"))

    table = pd.crosstab(
        membership_df[baseline_label_col],
        membership_df[week6_label_col],
        rownames=["baseline_true_cluster"],
        colnames=["week6_true_cluster"],
    )

    table = table.reset_index()
    table.attrs["expected_cross_time_ari"] = longitudinal_data.get("expected_cross_time_ari")
    table.attrs["expected_same_cluster_rate"] = longitudinal_data.get("expected_same_cluster_rate")
    return table


# =============================================================================
# Longitudinal unsupervised clustering GA extension
# =============================================================================
# This section is intentionally additive. It does not modify or replace the
# single-timepoint ClinicalResponseFeatureSelectionGA pathway.


def _normalize_silhouette(value: Any) -> float:
    """
    Normalize a silhouette value from [-1, 1] into [0, 1].
    Missing or invalid values return 0.0 for fitness safety.
    """
    if value is None or pd.isna(value):
        return 0.0
    return float(np.clip((float(value) + 1.0) / 2.0, 0.0, 1.0))


def _normalize_ari(value: Any) -> float:
    """
    Normalize an ARI-like score for fitness.

    ARI can be negative. For fitness, negative agreement is treated as 0.0,
    matching the existing label-alignment normalization style in this module.
    """
    if value is None or pd.isna(value):
        return 0.0
    return float(np.clip(float(value), 0.0, 1.0))


def _aggregate_two_timepoint_silhouette(
    baseline_silhouette: Any,
    week6_silhouette: Any,
    *,
    method: str = "min",
) -> float:
    """
    Aggregate baseline and Week 6 silhouette into one longitudinal silhouette.

    Supported methods
    -----------------
    "min":
        Conservative guardrail. The feature subset is rewarded only as much as
        its weaker timepoint.

    "mean":
        Average separation across both timepoints.
    """
    method = str(method).lower()

    raw_values = [baseline_silhouette, week6_silhouette]
    if any(v is None or pd.isna(v) for v in raw_values):
        return np.nan

    values = [float(v) for v in raw_values]

    if method == "min":
        return float(np.min(values))

    if method in {"mean", "average"}:
        return float(np.mean(values))

    raise ValueError(
        "Unknown silhouette aggregation method "
        f"{method!r}. Use 'min' or 'mean'."
    )


def resolve_longitudinal_config(
    cfg: ClinicalResponseGAFSConfig,
    *,
    subject_id_col: Optional[str] = None,
    week6_feature_cols: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Resolve longitudinal/timepoint settings for the active preset.

    Preferred config layout
    -----------------------
    cfg.fitness_preset_config[cfg.fitness_preset]["timepoint_config"]:
        subject_id_col:
            Optional subject identifier used to align baseline and Week 6 rows.
            If omitted, rows are aligned by position and both dataframes must
            have the same number of rows.

        week6_feature_cols:
            Optional list or mapping for Week 6 feature names. If omitted, Week
            6 is assumed to use the same feature column names as baseline.

    cfg.fitness_preset_config[cfg.fitness_preset]["longitudinal_scoring"]:
        consistency_metric:
            Currently only "ari" is supported.

        silhouette_aggregation:
            "min" or "mean". Defaults to "min".
    """
    preset_cfg = get_active_fitness_preset_config(cfg)
    timepoint_cfg = dict(preset_cfg.get("timepoint_config", {}) or {})
    scoring_cfg = dict(preset_cfg.get("longitudinal_scoring", {}) or {})

    resolved_subject_id_col = (
        timepoint_cfg.get("subject_id_col")
        if subject_id_col is None
        else subject_id_col
    )

    resolved_week6_feature_cols = (
        timepoint_cfg.get("week6_feature_cols")
        if week6_feature_cols is None
        else week6_feature_cols
    )

    consistency_metric = str(scoring_cfg.get("consistency_metric", "ari")).lower()
    if consistency_metric != "ari":
        raise ValueError(
            f"Unsupported consistency_metric={consistency_metric!r}. "
            "Currently only 'ari' is supported."
        )

    silhouette_aggregation = str(scoring_cfg.get("silhouette_aggregation", "min")).lower()
    if silhouette_aggregation == "average":
        silhouette_aggregation = "mean"
    if silhouette_aggregation not in {"min", "mean"}:
        raise ValueError(
            f"Unsupported silhouette_aggregation={silhouette_aggregation!r}. "
            "Use 'min' or 'mean'."
        )

    return {
        "subject_id_col": resolved_subject_id_col,
        "week6_feature_cols": resolved_week6_feature_cols,
        "consistency_metric": consistency_metric,
        "silhouette_aggregation": silhouette_aggregation,
    }


def _resolve_week6_feature_map(
    baseline_feature_cols: Sequence[str],
    week6_feature_cols: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
) -> Dict[str, str]:
    """
    Resolve a baseline-feature -> Week 6-feature mapping.

    If week6_feature_cols is None, identical names are assumed.
    If it is a mapping, keys are baseline feature names and values are Week 6
    feature names.
    If it is a sequence, it must be the same length and order as baseline_feature_cols.
    """
    baseline_feature_cols = list(baseline_feature_cols)

    if week6_feature_cols is None:
        return {col: col for col in baseline_feature_cols}

    if isinstance(week6_feature_cols, Mapping):
        missing = [col for col in baseline_feature_cols if col not in week6_feature_cols]
        if missing:
            raise ValueError(
                "week6_feature_cols mapping is missing baseline feature keys: "
                f"{missing[:10]}"
            )
        return {col: str(week6_feature_cols[col]) for col in baseline_feature_cols}

    week6_list = list(week6_feature_cols)
    if len(week6_list) != len(baseline_feature_cols):
        raise ValueError(
            "week6_feature_cols must have the same length as baseline feature_cols. "
            f"Got {len(week6_list)} Week 6 columns and "
            f"{len(baseline_feature_cols)} baseline columns."
        )

    return {base: week6 for base, week6 in zip(baseline_feature_cols, week6_list)}


def _prepare_longitudinal_feature_matrices(
    *,
    df_baseline: pd.DataFrame,
    df_week6: pd.DataFrame,
    selected_cols: Sequence[str],
    week6_feature_map: Mapping[str, str],
    subject_id_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    """
    Build paired baseline and Week 6 feature matrices for selected features.

    If subject_id_col is provided, dataframes are aligned by subject ID after
    dropping missing selected-feature values within each timepoint.

    If subject_id_col is omitted, rows are aligned by position, and both
    dataframes must have the same number of rows. Rows with missing values at
    either timepoint are removed jointly.
    """
    selected_cols = list(selected_cols)
    week6_selected_cols = [week6_feature_map[col] for col in selected_cols]

    missing_baseline = [col for col in selected_cols if col not in df_baseline.columns]
    missing_week6 = [col for col in week6_selected_cols if col not in df_week6.columns]

    if missing_baseline:
        raise ValueError(f"df_baseline is missing selected feature columns: {missing_baseline}")
    if missing_week6:
        raise ValueError(f"df_week6 is missing selected feature columns: {missing_week6}")

    if subject_id_col is not None:
        if subject_id_col not in df_baseline.columns:
            raise ValueError(f"df_baseline is missing subject_id_col={subject_id_col!r}.")
        if subject_id_col not in df_week6.columns:
            raise ValueError(f"df_week6 is missing subject_id_col={subject_id_col!r}.")

        baseline_rename = {col: f"{col}__baseline" for col in selected_cols}
        week6_rename = {
            week6_feature_map[col]: f"{col}__week6"
            for col in selected_cols
        }

        b = df_baseline[[subject_id_col] + selected_cols].copy()
        w = df_week6[[subject_id_col] + week6_selected_cols].copy()

        for col in selected_cols:
            b[col] = pd.to_numeric(b[col], errors="coerce")
        for col in week6_selected_cols:
            w[col] = pd.to_numeric(w[col], errors="coerce")

        b = b.dropna(subset=selected_cols).rename(columns=baseline_rename)
        w = w.dropna(subset=week6_selected_cols).rename(columns=week6_rename)

        paired = b.merge(w, on=subject_id_col, how="inner")

        baseline_matrix_cols = [f"{col}__baseline" for col in selected_cols]
        week6_matrix_cols = [f"{col}__week6" for col in selected_cols]

        X_baseline = paired[baseline_matrix_cols].to_numpy(dtype=float)
        X_week6 = paired[week6_matrix_cols].to_numpy(dtype=float)
        subject_ids = paired[subject_id_col].copy().reset_index(drop=True)

        return X_baseline, X_week6, subject_ids

    if len(df_baseline) != len(df_week6):
        raise ValueError(
            "When subject_id_col is not provided, df_baseline and df_week6 must "
            f"have the same number of rows. Got {len(df_baseline)} and {len(df_week6)}."
        )

    b = df_baseline[selected_cols].copy()
    w = df_week6[week6_selected_cols].copy()
    w.columns = selected_cols

    for col in selected_cols:
        b[col] = pd.to_numeric(b[col], errors="coerce")
        w[col] = pd.to_numeric(w[col], errors="coerce")

    valid_mask = ~(b.isna().any(axis=1) | w.isna().any(axis=1))

    X_baseline = b.loc[valid_mask, selected_cols].to_numpy(dtype=float)
    X_week6 = w.loc[valid_mask, selected_cols].to_numpy(dtype=float)
    subject_ids = pd.Series(np.arange(len(df_baseline)), name="row_position").loc[valid_mask].reset_index(drop=True)

    return X_baseline, X_week6, subject_ids


def longitudinal_unsupervised_clustering_fitness(
    metrics: Mapping[str, float],
    components: Mapping[str, Mapping[str, Any]],
) -> Tuple[float, Dict[str, float]]:
    """
    Standalone longitudinal unsupervised clustering fitness function.

    Calculation
    -----------
    fitness =
        cross_time_weight * cross_time_ari_norm
      + silhouette_weight * longitudinal_silhouette_norm
      - feature_penalty_weight * feature_penalty_norm

    This intentionally replaces bootstrap ARI with cross-time membership
    consistency ARI and does not use labels or clinical response columns.
    """
    cross_time_ari = _metric_value(metrics, "cross_time_ari_norm")
    longitudinal_silhouette = _metric_value(metrics, "longitudinal_silhouette_norm")
    feature_penalty = _metric_value(metrics, "feature_penalty_norm")

    w_cross_time = _component_weight(components, "cross_time_ari_norm")
    w_silhouette = _component_weight(components, "longitudinal_silhouette_norm")
    w_penalty = _component_weight(components, "feature_penalty_norm")

    contributions = {
        "cross_time_ari_norm": w_cross_time * cross_time_ari,
        "longitudinal_silhouette_norm": w_silhouette * longitudinal_silhouette,
        "feature_penalty_norm": -w_penalty * feature_penalty,
    }

    fitness = sum(contributions.values())
    return float(fitness), {k: float(v) for k, v in contributions.items()}


# Extend aliases and display labels without changing existing single-timepoint names.
FITNESS_METRIC_ALIASES.update(
    {
        "membership_consistency_ari_norm": "cross_time_ari_norm",
        "membership_consistency_ari_raw": "cross_time_ari_raw",
        "cross_time_membership_ari_norm": "cross_time_ari_norm",
        "cross_time_membership_ari_raw": "cross_time_ari_raw",
        "longitudinal_silhouette_summary_norm": "longitudinal_silhouette_norm",
        "longitudinal_silhouette_summary_raw": "longitudinal_silhouette_raw",
        "baseline_silhouette_norm": "baseline_silhouette_norm",
        "baseline_silhouette_raw": "baseline_silhouette_raw",
        "week6_silhouette_norm": "week6_silhouette_norm",
        "week6_silhouette_raw": "week6_silhouette_raw",
    }
)


# Preserve the original implementations so the extension can delegate to them.
_ORIGINAL_DEFAULT_METRIC_DISPLAY_LABEL = default_metric_display_label
_ORIGINAL_GET_FITNESS_PRESET_COMPONENTS = get_fitness_preset_components
_ORIGINAL_GET_NAMED_FITNESS_FUNCTION = get_named_fitness_function


def default_metric_display_label(metric_name: str) -> str:  # type: ignore[no-redef]
    """
    Provide concise labels for common metric names, including longitudinal metrics.
    """
    labels = {
        "cross_time_ari_norm": "longARI",
        "cross_time_ari_raw": "longARI_raw",
        "membership_consistency_ari_norm": "longARI",
        "membership_consistency_ari_raw": "longARI_raw",
        "cross_time_membership_ari_norm": "longARI",
        "cross_time_membership_ari_raw": "longARI_raw",
        "longitudinal_silhouette_norm": "longSil",
        "longitudinal_silhouette_raw": "longSil_raw",
        "longitudinal_silhouette_summary_norm": "longSil",
        "longitudinal_silhouette_summary_raw": "longSil_raw",
        "baseline_silhouette_norm": "BL_sil",
        "baseline_silhouette_raw": "BL_sil_raw",
        "week6_silhouette_norm": "W6_sil",
        "week6_silhouette_raw": "W6_sil_raw",
    }
    return labels.get(metric_name, _ORIGINAL_DEFAULT_METRIC_DISPLAY_LABEL(metric_name))


def get_fitness_preset_components(  # type: ignore[no-redef]
    preset_name: str,
    *,
    cfg: ClinicalResponseGAFSConfig,
) -> Dict[str, Dict[str, Any]]:
    """
    Return default component weights for a named fitness function.

    Adds longitudinal_unsupervised_clustering while preserving all existing presets.
    """
    name = str(preset_name).lower()

    if name == "longitudinal_unsupervised_clustering":
        return {
            "cross_time_ari_norm": {
                "weight": 0.70,
                "direction": "maximize",
                "description": "Cross-time membership consistency ARI",
            },
            "longitudinal_silhouette_norm": {
                "weight": 0.20,
                "direction": "maximize",
                "description": "Two-timepoint silhouette summary",
            },
            "feature_penalty_norm": {
                "weight": 0.10,
                "direction": "minimize",
                "description": "Feature-count penalty",
            },
        }

    return _ORIGINAL_GET_FITNESS_PRESET_COMPONENTS(preset_name, cfg=cfg)


def get_named_fitness_function(  # type: ignore[no-redef]
    preset_name: str,
) -> Callable[[Mapping[str, float], Mapping[str, Mapping[str, Any]]], Tuple[float, Dict[str, float]]]:
    """
    Return the standalone fitness function associated with a preset name.

    Adds longitudinal_unsupervised_clustering while preserving all existing presets.
    """
    name = str(preset_name).lower()

    if name == "longitudinal_unsupervised_clustering":
        return longitudinal_unsupervised_clustering_fitness

    return _ORIGINAL_GET_NAMED_FITNESS_FUNCTION(preset_name)


def active_fitness_uses_cross_time_ari(cfg: ClinicalResponseGAFSConfig) -> bool:
    """Return whether active fitness components use cross-time ARI."""
    components = active_fitness_components(cfg)
    return bool(components.get("cross_time_ari_norm", {}).get("weight", 0.0) != 0)


def active_fitness_uses_longitudinal_silhouette(cfg: ClinicalResponseGAFSConfig) -> bool:
    """Return whether active fitness components use longitudinal silhouette."""
    components = active_fitness_components(cfg)
    return bool(components.get("longitudinal_silhouette_norm", {}).get("weight", 0.0) != 0)


def evaluate_feature_subset_longitudinal_clustering(
    *,
    df_baseline: pd.DataFrame,
    df_week6: pd.DataFrame,
    feature_cols: Sequence[str],
    mask: Sequence[Any],
    cfg: ClinicalResponseGAFSConfig,
    week6_feature_cols: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
    subject_id_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate one binary feature mask using the longitudinal unsupervised objective.

    Steps
    -----
    1. Repair the mask to respect min/max feature constraints.
    2. Select baseline feature columns.
    3. Align baseline and Week 6 rows by subject ID or row position.
    4. Fit scaler and clustering model on baseline selected features only.
    5. Assign both baseline and Week 6 cluster membership using that baseline model.
    6. Compute cross-time membership consistency ARI.
    7. Compute baseline and Week 6 silhouette, then aggregate by min or mean.
    8. Combine metrics through longitudinal_unsupervised_clustering_fitness.
    """
    rng = np.random.default_rng(cfg.random_seed)
    repaired_mask = _repair_mask(mask, cfg, rng)
    selected_cols = _selected_columns(feature_cols, repaired_mask)

    if len(selected_cols) < cfg.min_features:
        return {
            "fitness": -1e9,
            "selected_cols": selected_cols,
            "n_features": len(selected_cols),
            "error": "Too few selected features.",
        }

    longitudinal_cfg = resolve_longitudinal_config(
        cfg,
        subject_id_col=subject_id_col,
        week6_feature_cols=week6_feature_cols,
    )

    week6_feature_map = _resolve_week6_feature_map(
        feature_cols,
        longitudinal_cfg["week6_feature_cols"],
    )

    try:
        X_baseline, X_week6, subject_ids = _prepare_longitudinal_feature_matrices(
            df_baseline=df_baseline,
            df_week6=df_week6,
            selected_cols=selected_cols,
            week6_feature_map=week6_feature_map,
            subject_id_col=longitudinal_cfg["subject_id_col"],
        )
    except Exception as exc:
        return {
            "fitness": -1e9,
            "selected_cols": selected_cols,
            "n_features": len(selected_cols),
            "error": str(exc),
        }

    min_rows_needed = max(cfg.k + 2, 5)
    if X_baseline.shape[0] < min_rows_needed or X_week6.shape[0] < min_rows_needed:
        return {
            "fitness": -1e9,
            "selected_cols": selected_cols,
            "n_features": len(selected_cols),
            "n_paired_subjects": int(X_baseline.shape[0]),
            "error": "Too few paired usable rows after dropping missing values.",
        }

    # ------------------------------------------------------------------
    # Fit on baseline only, then predict both baseline and Week 6.
    # ------------------------------------------------------------------
    try:
        labels_baseline, X_baseline_scaled, model, scaler, model_name = _fit_cluster_labels(
            X_baseline,
            cfg=cfg,
        )

        X_week6_scaled = scaler.transform(X_week6)

        if hasattr(model, "predict"):
            labels_week6 = np.asarray(model.predict(X_week6_scaled))
        else:
            raise ValueError(
                f"Clustering model {model_name!r} must implement predict() for "
                "longitudinal Week 6 assignment."
            )
    except Exception as exc:
        return {
            "fitness": -1e9,
            "selected_cols": selected_cols,
            "n_features": len(selected_cols),
            "n_paired_subjects": int(X_baseline.shape[0]),
            "error": str(exc),
        }

    # ------------------------------------------------------------------
    # Metrics.
    # ------------------------------------------------------------------
    try:
        cross_time_ari_raw = float(adjusted_rand_score(labels_baseline, labels_week6))
    except Exception:
        cross_time_ari_raw = np.nan

    baseline_quality = compute_cluster_quality(X_baseline_scaled, labels_baseline)
    week6_quality = compute_cluster_quality(X_week6_scaled, labels_week6)

    baseline_silhouette_raw = baseline_quality.get("silhouette", np.nan)
    week6_silhouette_raw = week6_quality.get("silhouette", np.nan)

    longitudinal_silhouette_raw = _aggregate_two_timepoint_silhouette(
        baseline_silhouette_raw,
        week6_silhouette_raw,
        method=longitudinal_cfg["silhouette_aggregation"],
    )

    cross_time_ari_norm = _normalize_ari(cross_time_ari_raw)
    baseline_silhouette_norm = _normalize_silhouette(baseline_silhouette_raw)
    week6_silhouette_norm = _normalize_silhouette(week6_silhouette_raw)
    longitudinal_silhouette_norm = _normalize_silhouette(longitudinal_silhouette_raw)

    feature_fraction = len(selected_cols) / max(len(feature_cols), cfg.eps)
    feature_penalty = float(feature_fraction ** cfg.feature_fraction_penalty_power)

    same_cluster_rate = float(np.mean(labels_baseline == labels_week6))

    metrics_for_fitness: Dict[str, float] = {
        "cross_time_ari_raw": float(cross_time_ari_raw) if not pd.isna(cross_time_ari_raw) else np.nan,
        "cross_time_ari_norm": cross_time_ari_norm,
        "baseline_silhouette_raw": float(baseline_silhouette_raw) if not pd.isna(baseline_silhouette_raw) else np.nan,
        "baseline_silhouette_norm": baseline_silhouette_norm,
        "week6_silhouette_raw": float(week6_silhouette_raw) if not pd.isna(week6_silhouette_raw) else np.nan,
        "week6_silhouette_norm": week6_silhouette_norm,
        "longitudinal_silhouette_raw": float(longitudinal_silhouette_raw) if not pd.isna(longitudinal_silhouette_raw) else np.nan,
        "longitudinal_silhouette_norm": longitudinal_silhouette_norm,
        "same_cluster_rate": same_cluster_rate,
        "feature_penalty_raw": feature_penalty,
        "feature_penalty_norm": feature_penalty,
        "n_features": float(len(selected_cols)),
        "n_total_features": float(len(feature_cols)),
        "n_paired_subjects": float(len(subject_ids)),
    }

    fitness, fitness_details = compute_dynamic_fitness(
        metrics_for_fitness,
        cfg=cfg,
    )

    membership_df = pd.DataFrame(
        {
            "subject_id": subject_ids.to_numpy(),
            "cluster_baseline": labels_baseline,
            "cluster_week6": labels_week6,
            "same_cluster": labels_baseline == labels_week6,
        }
    )

    return {
        "fitness": float(fitness),
        "selected_cols": selected_cols,
        "week6_selected_cols": [week6_feature_map[col] for col in selected_cols],
        "n_features": int(len(selected_cols)),
        "n_paired_subjects": int(len(subject_ids)),
        "cross_time_ari_raw": metrics_for_fitness["cross_time_ari_raw"],
        "cross_time_ari_norm": cross_time_ari_norm,
        "baseline_silhouette_raw": metrics_for_fitness["baseline_silhouette_raw"],
        "baseline_silhouette_norm": baseline_silhouette_norm,
        "week6_silhouette_raw": metrics_for_fitness["week6_silhouette_raw"],
        "week6_silhouette_norm": week6_silhouette_norm,
        "longitudinal_silhouette_raw": metrics_for_fitness["longitudinal_silhouette_raw"],
        "longitudinal_silhouette_norm": longitudinal_silhouette_norm,
        "same_cluster_rate": same_cluster_rate,
        "feature_penalty_raw": feature_penalty,
        "feature_penalty_norm": feature_penalty,
        "metrics_for_fitness": metrics_for_fitness,
        "fitness_details": fitness_details,
        "cluster_labels_baseline": labels_baseline,
        "cluster_labels_week6": labels_week6,
        "cluster_labels": labels_baseline,
        "membership_df": membership_df,
        "model_name": model_name,
        "details": {
            "baseline_quality_dict": baseline_quality,
            "week6_quality_dict": week6_quality,
            "longitudinal_config": longitudinal_cfg,
        },
    }


class LongitudinalFeatureSelectionGA:
    """
    Genetic algorithm runner for longitudinal unsupervised feature selection.

    This runner is separate from ClinicalResponseFeatureSelectionGA so the
    existing single-timepoint workflows remain untouched.

    Parameters
    ----------
    df_baseline:
        Baseline dataframe containing candidate clustering features.

    df_week6:
        Week 6 dataframe containing corresponding candidate features.

    cfg:
        GA configuration. Use fitness_preset="longitudinal_unsupervised_clustering".

    feature_cols:
        Baseline candidate feature columns used by the GA.

    week6_feature_cols:
        Optional Week 6 feature names. If omitted, same names as baseline are assumed.
        Can be a list aligned to feature_cols or a mapping baseline_col -> week6_col.

    subject_id_col:
        Optional subject ID used to align baseline and Week 6 rows. If omitted,
        rows are aligned by position.
    """

    def __init__(
        self,
        *,
        df_baseline: pd.DataFrame,
        df_week6: pd.DataFrame,
        cfg: ClinicalResponseGAFSConfig,
        feature_cols: Optional[Sequence[str]] = None,
        week6_feature_cols: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
        subject_id_col: Optional[str] = None,
        outdir: str = "longitudinal_clustering_ga_output",
    ) -> None:
        self.df_baseline = df_baseline.copy()
        self.df_week6 = df_week6.copy()
        self.cfg = cfg
        self.feature_cols = resolve_feature_config(cfg, feature_cols=feature_cols)
        self.outdir = outdir

        apply_active_fitness_preset_config(self.cfg)

        longitudinal_cfg = resolve_longitudinal_config(
            self.cfg,
            subject_id_col=subject_id_col,
            week6_feature_cols=week6_feature_cols,
        )
        self.subject_id_col = longitudinal_cfg["subject_id_col"]
        self.week6_feature_cols = longitudinal_cfg["week6_feature_cols"]
        self.week6_feature_map = _resolve_week6_feature_map(
            self.feature_cols,
            self.week6_feature_cols,
        )

        missing_baseline_features = [
            col for col in self.feature_cols
            if col not in self.df_baseline.columns
        ]
        missing_week6_features = [
            week6_col for week6_col in self.week6_feature_map.values()
            if week6_col not in self.df_week6.columns
        ]

        if missing_baseline_features:
            raise ValueError(f"df_baseline is missing feature columns: {missing_baseline_features}")
        if missing_week6_features:
            raise ValueError(f"df_week6 is missing feature columns: {missing_week6_features}")

        if self.subject_id_col is not None:
            if self.subject_id_col not in self.df_baseline.columns:
                raise ValueError(f"df_baseline is missing subject_id_col={self.subject_id_col!r}.")
            if self.subject_id_col not in self.df_week6.columns:
                raise ValueError(f"df_week6 is missing subject_id_col={self.subject_id_col!r}.")

        os.makedirs(outdir, exist_ok=True)

        self._cache: Dict[str, Dict[str, Any]] = {}
        self.history: List[Dict[str, Any]] = []
        self.raw_population_history: List[np.ndarray] = []
        self.population_history: List[np.ndarray] = []
        self._start_time: Optional[float] = None
        self._last_gen_time: Optional[float] = None

    def _mask_key(self, mask: Sequence[Any]) -> str:
        """Convert a solution mask into a stable string cache key."""
        arr = (np.asarray(mask, dtype=float) >= 0.5).astype(int)
        return "".join(map(str, arr.tolist()))

    def evaluate(self, mask: Sequence[Any]) -> Dict[str, Any]:
        """Evaluate a candidate feature mask with caching."""
        rng = np.random.default_rng(self.cfg.random_seed)
        repaired_mask = _repair_mask(mask, self.cfg, rng)
        key = self._mask_key(repaired_mask)

        if self.cfg.use_cache and key in self._cache:
            return self._cache[key]

        result = evaluate_feature_subset_longitudinal_clustering(
            df_baseline=self.df_baseline,
            df_week6=self.df_week6,
            feature_cols=self.feature_cols,
            mask=repaired_mask,
            cfg=self.cfg,
            week6_feature_cols=self.week6_feature_cols,
            subject_id_col=self.subject_id_col,
        )

        if self.cfg.use_cache:
            self._cache[key] = result

        return result

    def run(self) -> Dict[str, Any]:
        """Run the longitudinal genetic algorithm."""
        try:
            import pygad
        except ImportError as exc:
            raise ImportError(
                "pygad is required to run the GA. Install it with: pip install pygad"
            ) from exc

        n_genes = len(self.feature_cols)

        initial_population = make_sparse_initial_population(
            n_genes,
            sol_per_pop=self.cfg.sol_per_pop,
            min_features=self.cfg.min_features,
            max_features=self.cfg.max_features,
            random_seed=self.cfg.random_seed,
        )

        self._start_time = time.time()
        self._last_gen_time = self._start_time

        def fitness_func(ga_instance: Any, solution: np.ndarray, solution_idx: int) -> float:
            evaluation = self.evaluate(solution)
            return float(evaluation["fitness"])

        def on_generation(ga_instance: Any) -> None:
            raw_population = np.asarray(ga_instance.population).copy()
            self.raw_population_history.append(raw_population)

            repaired_population = np.vstack(
                [
                    _repair_mask(
                        solution,
                        self.cfg,
                        np.random.default_rng(self.cfg.random_seed),
                    )
                    for solution in raw_population
                ]
            )
            self.population_history.append(repaired_population)

            now = time.time()
            gen_time = now - (self._last_gen_time or now)
            total_time = now - (self._start_time or now)
            self._last_gen_time = now

            best_solution, best_fitness, _ = ga_instance.best_solution()
            best_eval = self.evaluate(best_solution)

            row = {
                "generation": int(ga_instance.generations_completed),
                "best_fitness": float(best_fitness),
                "n_features": int(best_eval.get("n_features", 0)),
                "n_paired_subjects": int(best_eval.get("n_paired_subjects", 0)),
                "generation_time_sec": float(gen_time),
                "total_elapsed_sec": float(total_time),
                "solution_mask": self._mask_key(best_solution),
                "selected_features": " | ".join(best_eval.get("selected_cols", [])),
                "model_name": best_eval.get("model_name", None),
                "cross_time_ari_raw": best_eval.get("cross_time_ari_raw", np.nan),
                "cross_time_ari_norm": best_eval.get("cross_time_ari_norm", np.nan),
                "baseline_silhouette_raw": best_eval.get("baseline_silhouette_raw", np.nan),
                "baseline_silhouette_norm": best_eval.get("baseline_silhouette_norm", np.nan),
                "week6_silhouette_raw": best_eval.get("week6_silhouette_raw", np.nan),
                "week6_silhouette_norm": best_eval.get("week6_silhouette_norm", np.nan),
                "longitudinal_silhouette_raw": best_eval.get("longitudinal_silhouette_raw", np.nan),
                "longitudinal_silhouette_norm": best_eval.get("longitudinal_silhouette_norm", np.nan),
                "same_cluster_rate": best_eval.get("same_cluster_rate", np.nan),
                "feature_penalty_raw": best_eval.get("feature_penalty_raw", np.nan),
                "feature_penalty_norm": best_eval.get("feature_penalty_norm", np.nan),
                "fitness_function_name": best_eval.get("fitness_details", {}).get("fitness_function_name", None),
                "fitness_contributions": best_eval.get("fitness_details", {}).get("fitness_contributions", None),
            }

            metrics_for_fitness = best_eval.get("metrics_for_fitness", {}) or {}
            fitness_contributions = (
                best_eval.get("fitness_details", {}).get("fitness_contributions", {}) or {}
            )
            metric_pairs, _ = get_active_metric_display_config(self.cfg)

            for user_metric_name, internal_metric_name in metric_pairs:
                if internal_metric_name in metrics_for_fitness:
                    row[user_metric_name] = metrics_for_fitness[internal_metric_name]
                elif internal_metric_name in row:
                    row[user_metric_name] = row[internal_metric_name]

                if internal_metric_name in fitness_contributions:
                    row[f"{user_metric_name}__weighted"] = fitness_contributions[internal_metric_name]

            self.history.append(row)
            print(build_generation_log_message(row, cfg=self.cfg))

        ga_instance = pygad.GA(
            num_generations=self.cfg.num_generations,
            sol_per_pop=self.cfg.sol_per_pop,
            num_parents_mating=self.cfg.num_parents_mating,
            keep_parents=self.cfg.keep_parents,
            keep_elitism=self.cfg.keep_elitism,
            num_genes=n_genes,
            gene_space=[0, 1],
            gene_type=int,
            initial_population=initial_population,
            parent_selection_type=self.cfg.parent_selection_type,
            crossover_type=self.cfg.crossover_type,
            mutation_type=self.cfg.mutation_type,
            mutation_percent_genes=self.cfg.mutation_percent_genes,
            random_seed=self.cfg.random_seed,
            fitness_func=fitness_func,
            on_generation=on_generation,
        )

        ga_instance.run()

        elapsed = time.time() - (self._start_time or time.time())

        best_solution, best_fitness, _ = ga_instance.best_solution()
        rng = np.random.default_rng(self.cfg.random_seed)
        best_mask = _repair_mask(best_solution, self.cfg, rng)
        best_eval = self.evaluate(best_mask)

        history_df = pd.DataFrame(self.history)
        feature_selection_frequency_df = compute_feature_selection_frequency(
            self.population_history,
            self.feature_cols,
        )

        result: Dict[str, Any] = {
            "ga_instance": ga_instance,
            "best_solution": np.asarray(best_solution, dtype=int),
            "best_mask": np.asarray(best_mask, dtype=int),
            "best_fitness": float(best_fitness),
            "best_eval": best_eval,
            "selected_cols": best_eval.get("selected_cols", []),
            "week6_selected_cols": best_eval.get("week6_selected_cols", []),
            "membership_df": best_eval.get("membership_df", pd.DataFrame()),
            "history_df": history_df,
            "raw_population_history": self.raw_population_history,
            "population_history": self.population_history,
            "feature_selection_frequency_df": feature_selection_frequency_df,
            "config": safe_config_dict(self.cfg),
            "feature_cols": self.feature_cols,
            "week6_feature_map": self.week6_feature_map,
            "subject_id_col": self.subject_id_col,
        }

        self._save_outputs(result)

        print("\n" + "=" * 80)
        print(f"Longitudinal GA completed in {elapsed:.2f} sec")
        print(f"Longitudinal GA completed in {elapsed / 60:.2f} min")
        if self.cfg.num_generations > 0:
            print(f"Average time per generation: {elapsed / self.cfg.num_generations:.2f} sec")
        print("=" * 80)

        return result

    def _save_outputs(self, result: Dict[str, Any]) -> None:
        """Save lightweight longitudinal run outputs to outdir."""
        history_path = os.path.join(self.outdir, "history.csv")
        selected_path = os.path.join(self.outdir, "selected_features.csv")
        frequency_path = os.path.join(self.outdir, "feature_selection_frequency.csv")
        membership_path = os.path.join(self.outdir, "best_membership_consistency.csv")

        history_df = result.get("history_df", pd.DataFrame())
        if isinstance(history_df, pd.DataFrame):
            history_df.to_csv(history_path, index=False)

        pd.DataFrame(
            {
                "selected_feature_baseline": result.get("selected_cols", []),
                "selected_feature_week6": result.get("week6_selected_cols", []),
            }
        ).to_csv(selected_path, index=False)

        frequency_df = result.get("feature_selection_frequency_df", pd.DataFrame())
        if isinstance(frequency_df, pd.DataFrame):
            frequency_df.to_csv(frequency_path, index=False)

        membership_df = result.get("membership_df", pd.DataFrame())
        if isinstance(membership_df, pd.DataFrame):
            membership_df.to_csv(membership_path, index=False)


def make_longitudinal_clustering_ga(
    *,
    df_baseline: pd.DataFrame,
    df_week6: pd.DataFrame,
    cfg: Optional[ClinicalResponseGAFSConfig] = None,
    feature_cols: Optional[Sequence[str]] = None,
    week6_feature_cols: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
    subject_id_col: Optional[str] = None,
    outdir: str = "longitudinal_clustering_ga_output",
) -> LongitudinalFeatureSelectionGA:
    """
    Convenience constructor for LongitudinalFeatureSelectionGA.

    This is the longitudinal counterpart to make_clinical_response_ga(...).
    It keeps the single-timepoint API untouched.
    """
    if cfg is None:
        cfg = ClinicalResponseGAFSConfig(
            fitness_preset="longitudinal_unsupervised_clustering",
        )

    return LongitudinalFeatureSelectionGA(
        df_baseline=df_baseline,
        df_week6=df_week6,
        cfg=cfg,
        feature_cols=feature_cols,
        week6_feature_cols=week6_feature_cols,
        subject_id_col=subject_id_col,
        outdir=outdir,
    )


def evaluate_mask_longitudinal_clustering(
    *,
    df_baseline: pd.DataFrame,
    df_week6: pd.DataFrame,
    mask: Sequence[Any],
    cfg: ClinicalResponseGAFSConfig,
    feature_cols: Optional[Sequence[str]] = None,
    week6_feature_cols: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
    subject_id_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate one longitudinal feature mask without running the GA.

    This is useful for smoke tests and for comparing candidate feature subsets.
    """
    apply_active_fitness_preset_config(cfg)
    resolved_feature_cols = resolve_feature_config(cfg, feature_cols=feature_cols)
    longitudinal_cfg = resolve_longitudinal_config(
        cfg,
        subject_id_col=subject_id_col,
        week6_feature_cols=week6_feature_cols,
    )

    return evaluate_feature_subset_longitudinal_clustering(
        df_baseline=df_baseline,
        df_week6=df_week6,
        feature_cols=resolved_feature_cols,
        mask=mask,
        cfg=cfg,
        week6_feature_cols=longitudinal_cfg["week6_feature_cols"],
        subject_id_col=longitudinal_cfg["subject_id_col"],
    )


# =============================================================================
# Generalized multi-timepoint longitudinal clustering GA extension
# =============================================================================
# This section supersedes the earlier two-timepoint convenience interface while
# keeping backward compatibility with df_baseline/df_week6 calls. It supports any
# number of timepoints through a dictionary:
#
#     timepoint_dfs={"baseline": df_bl, "week6": df_w6, "week12": df_w12}
#
# The reference timepoint defines the fitted scaler and clustering model. All
# other timepoints are projected into that reference-defined clustering space.


def _aggregate_numeric_values(values: Sequence[Any], *, method: str = "mean") -> float:
    """
    Aggregate numeric values with either mean or min.

    Missing/NaN values are ignored. If no valid values remain, np.nan is returned.
    """
    method = str(method).lower()
    if method == "average":
        method = "mean"

    valid = [float(v) for v in values if v is not None and not pd.isna(v)]
    if len(valid) == 0:
        return np.nan

    if method == "mean":
        return float(np.mean(valid))
    if method == "min":
        return float(np.min(valid))

    raise ValueError(f"Unknown aggregation method={method!r}. Use 'mean' or 'min'.")


def _clean_timepoint_feature_list_from_df(
    df: pd.DataFrame,
    *,
    subject_id_col: Optional[str] = None,
) -> List[str]:
    """
    Infer candidate feature columns from a timepoint dataframe.

    This is used only as a fallback when timepoint_feature_cols is not provided.
    The subject ID column, if present, is excluded because it is not a clustering
    feature.
    """
    cols = list(df.columns)
    if subject_id_col is not None and subject_id_col in cols:
        cols = [c for c in cols if c != subject_id_col]
    return [str(c) for c in cols]


def _summarize_feature_mapping(
    *,
    strategy: str,
    reference_timepoint: str,
    reference_feature_cols_original: Sequence[str],
    resolved_reference_feature_cols: Sequence[str],
    raw_timepoint_feature_cols: Mapping[str, Sequence[str]],
    timepoint_order: Sequence[str],
) -> Dict[str, Any]:
    """
    Build a human-readable summary of how features were mapped across timepoints.

    The summary is returned in result['feature_mapping_summary'] so missing or
    dropped features are visible instead of being silently ignored.
    """
    reference_feature_cols_original = [str(c) for c in reference_feature_cols_original]
    resolved_reference_feature_cols = [str(c) for c in resolved_reference_feature_cols]
    resolved_set = set(resolved_reference_feature_cols)

    summary: Dict[str, Any] = {
        "feature_mapping_strategy": strategy,
        "reference_timepoint": reference_timepoint,
        "n_reference_features_original": int(len(reference_feature_cols_original)),
        "n_features_used_by_ga": int(len(resolved_reference_feature_cols)),
        "n_dropped_reference_features": int(len([c for c in reference_feature_cols_original if c not in resolved_set])),
        "dropped_reference_features": [c for c in reference_feature_cols_original if c not in resolved_set],
        "timepoints": {},
    }

    for tp in timepoint_order:
        available = [str(c) for c in raw_timepoint_feature_cols.get(tp, [])]
        available_set = set(available)
        missing_from_tp = [c for c in reference_feature_cols_original if c not in available_set]
        extra_vs_reference = [c for c in available if c not in set(reference_feature_cols_original)]
        summary["timepoints"][tp] = {
            "n_available_features": int(len(available)),
            "n_missing_reference_features": int(len(missing_from_tp)),
            "missing_reference_features": missing_from_tp,
            "n_extra_features_vs_reference": int(len(extra_vs_reference)),
            "extra_features_vs_reference": extra_vs_reference,
        }

    return summary


def resolve_multitimepoint_longitudinal_config(
    cfg: ClinicalResponseGAFSConfig,
    *,
    timepoint_dfs: Mapping[str, pd.DataFrame],
    feature_cols: Optional[Sequence[str]] = None,
    subject_id_col: Optional[str] = None,
    reference_timepoint: Optional[str] = None,
    timepoint_feature_cols: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    """
    Resolve multi-timepoint longitudinal settings.

    Preferred config layout
    -----------------------
    cfg.fitness_preset_config[cfg.fitness_preset]["timepoint_config"] = {
        "timepoint_dfs": {
            "baseline": df_baseline,
            "week6": df_week6,
        },
        "subject_id_col": "subject_id" or None,
        "reference_timepoint": "baseline",

        # Default is intersection_by_name.
        # This avoids relying on feature-column order across timepoints.
        "feature_mapping_strategy": "intersection_by_name",

        "timepoint_feature_cols": {
            "baseline": feature_cols_bl,
            "week6": feature_cols_w6,
            "week12": feature_cols_w12,
        },
        "timepoint_order": ["baseline", "week6", "week12"],
    }

    Feature mapping strategies
    --------------------------
    intersection_by_name (default):
        Use only features that appear by name in every timepoint. The GA search
        space is the reference-timepoint feature list after dropping features
        missing from any other timepoint. This is safest when pruning may produce
        different feature sets at different visits.

    name:
        Strict name matching. Use the full reference feature list and raise an
        error if any reference feature is missing from any other timepoint.

    position:
        Legacy behavior. Apply the GA mask by feature-list position. This is only
        safe when every timepoint feature list is in exactly corresponding order.

    If subject_id_col is None or not present in every dataframe, rows are assumed
    to already be patient-aligned and are matched by row position.
    """
    if timepoint_dfs is None or len(timepoint_dfs) < 2:
        raise ValueError("timepoint_dfs must contain at least two timepoints.")

    # Preserve dict insertion order unless explicit order is provided.
    preset_cfg = get_active_fitness_preset_config(cfg)
    timepoint_cfg = dict(preset_cfg.get("timepoint_config", {}) or {})
    scoring_cfg = dict(preset_cfg.get("longitudinal_scoring", {}) or {})

    timepoint_names = [str(k) for k in timepoint_dfs.keys()]
    timepoint_dfs_resolved = {str(k): v for k, v in timepoint_dfs.items()}

    cfg_order = timepoint_cfg.get("timepoint_order", None)
    if cfg_order is not None:
        timepoint_order = [str(x) for x in cfg_order]
        missing_order = [tp for tp in timepoint_order if tp not in timepoint_dfs_resolved]
        extra_order = [tp for tp in timepoint_dfs_resolved if tp not in timepoint_order]
        if missing_order:
            raise ValueError(f"timepoint_order contains unknown timepoints: {missing_order}")
        # Append any unmentioned timepoints at the end to be forgiving.
        timepoint_order = timepoint_order + extra_order
    else:
        timepoint_order = timepoint_names

    resolved_reference = (
        reference_timepoint
        or timepoint_cfg.get("reference_timepoint")
        or ("baseline" if "baseline" in timepoint_dfs_resolved else timepoint_order[0])
    )
    resolved_reference = str(resolved_reference)
    if resolved_reference not in timepoint_dfs_resolved:
        raise ValueError(
            f"reference_timepoint={resolved_reference!r} not found in timepoint_dfs. "
            f"Available timepoints: {list(timepoint_dfs_resolved.keys())}"
        )

    resolved_subject_id_col = (
        subject_id_col
        if subject_id_col is not None
        else timepoint_cfg.get("subject_id_col", None)
    )

    # If the requested subject_id_col is missing in any dataframe, fall back to
    # row-order alignment. This matches the requested behavior: if no subject ID
    # is present, assume rows are already aligned by patient.
    if resolved_subject_id_col is not None:
        has_id_everywhere = all(
            resolved_subject_id_col in timepoint_dfs_resolved[tp].columns
            for tp in timepoint_order
        )
        if not has_id_everywhere:
            resolved_subject_id_col = None
            alignment_method = "row_order"
        else:
            alignment_method = "subject_id"
    else:
        alignment_method = "row_order"

    # Feature lists: explicit argument wins, then config, then dataframe columns.
    cfg_tp_feature_cols = timepoint_cfg.get("timepoint_feature_cols", None)
    resolved_tp_feature_cols = timepoint_feature_cols or cfg_tp_feature_cols

    if resolved_tp_feature_cols is None:
        raw_feature_map = {
            tp: _clean_timepoint_feature_list_from_df(
                timepoint_dfs_resolved[tp],
                subject_id_col=resolved_subject_id_col,
            )
            for tp in timepoint_order
        }
    else:
        raw_feature_map = {str(k): [str(c) for c in list(v)] for k, v in dict(resolved_tp_feature_cols).items()}
        missing_tp = [tp for tp in timepoint_order if tp not in raw_feature_map]
        if missing_tp:
            raise ValueError(
                "timepoint_feature_cols is missing feature lists for timepoints: "
                f"{missing_tp}. Provide a feature list for every timepoint."
            )

    if feature_cols is None:
        try:
            feature_cols = resolve_feature_config(cfg, feature_cols=None)
        except Exception:
            feature_cols = raw_feature_map.get(resolved_reference)

    if feature_cols is None:
        raise ValueError(
            "Could not resolve reference feature columns. Provide cfg.feature_config, "
            "feature_cols=..., or timepoint_feature_cols[reference_timepoint]."
        )

    reference_feature_cols_original = [str(c) for c in list(feature_cols)]
    if resolved_reference not in raw_feature_map:
        raw_feature_map[resolved_reference] = list(reference_feature_cols_original)

    strategy = str(timepoint_cfg.get("feature_mapping_strategy", "intersection_by_name")).lower()
    if strategy in {"intersection", "shared", "shared_by_name", "intersect_by_name"}:
        strategy = "intersection_by_name"
    if strategy in {"strict_name", "name_strict"}:
        strategy = "name"
    if strategy not in {"intersection_by_name", "name", "position"}:
        raise ValueError(
            f"Unsupported feature_mapping_strategy={strategy!r}. "
            "Use 'intersection_by_name', 'name', or 'position'."
        )

    # Validate duplicate feature names within each timepoint. Name-based mapping
    # requires unique columns so a selected feature maps to exactly one column.
    duplicate_by_tp = {
        tp: sorted({c for c in raw_feature_map[tp] if raw_feature_map[tp].count(c) > 1})
        for tp in timepoint_order
    }
    duplicate_by_tp = {tp: vals for tp, vals in duplicate_by_tp.items() if vals}
    if duplicate_by_tp:
        raise ValueError(f"Duplicate feature names found in timepoint_feature_cols: {duplicate_by_tp}")

    if strategy == "position":
        # Legacy position-based mapping. This preserves the old behavior, but it
        # is no longer the default because it can silently mismatch features when
        # column order differs across timepoints.
        ref_len = len(reference_feature_cols_original)
        length_mismatch = {
            tp: len(raw_feature_map[tp])
            for tp in timepoint_order
            if len(raw_feature_map[tp]) != ref_len
        }
        if length_mismatch:
            raise ValueError(
                "feature_mapping_strategy='position' requires every timepoint feature "
                f"list to have the same length as the reference feature list ({ref_len}). "
                f"Mismatches: {length_mismatch}"
            )
        reference_feature_cols = list(reference_feature_cols_original)
        feature_map = {tp: list(raw_feature_map[tp]) for tp in timepoint_order}

    else:
        # Name-based mapping. The feature lists for all timepoints are reordered
        # to match the reference feature names. This makes feature-column order
        # irrelevant and prevents silent baseline/week6 mismatches.
        available_sets = {tp: set(raw_feature_map[tp]) for tp in timepoint_order}

        if strategy == "name":
            missing_by_tp = {
                tp: [c for c in reference_feature_cols_original if c not in available_sets[tp]]
                for tp in timepoint_order
            }
            missing_by_tp = {tp: vals for tp, vals in missing_by_tp.items() if vals}
            if missing_by_tp:
                raise ValueError(
                    "feature_mapping_strategy='name' requires every reference feature "
                    f"to exist in every timepoint. Missing features: {missing_by_tp}"
                )
            reference_feature_cols = list(reference_feature_cols_original)

        else:  # intersection_by_name
            reference_feature_cols = [
                c for c in reference_feature_cols_original
                if all(c in available_sets[tp] for tp in timepoint_order)
            ]
            if len(reference_feature_cols) == 0:
                raise ValueError(
                    "No shared features found across timepoints under "
                    "feature_mapping_strategy='intersection_by_name'."
                )

        # After resolution, every timepoint uses the same canonical feature names
        # in the same reference order. A selected baseline feature maps to the
        # same named feature at each other timepoint.
        feature_map = {tp: list(reference_feature_cols) for tp in timepoint_order}

        # Validate that resolved columns really exist in the actual dataframes.
        missing_df_cols = {
            tp: [c for c in feature_map[tp] if c not in timepoint_dfs_resolved[tp].columns]
            for tp in timepoint_order
        }
        missing_df_cols = {tp: vals for tp, vals in missing_df_cols.items() if vals}
        if missing_df_cols:
            raise ValueError(
                "Resolved timepoint feature columns are missing from the dataframes: "
                f"{missing_df_cols}"
            )

    feature_mapping_summary = _summarize_feature_mapping(
        strategy=strategy,
        reference_timepoint=resolved_reference,
        reference_feature_cols_original=reference_feature_cols_original,
        resolved_reference_feature_cols=reference_feature_cols,
        raw_timepoint_feature_cols=raw_feature_map,
        timepoint_order=timepoint_order,
    )

    consistency_metric = str(scoring_cfg.get("consistency_metric", "ari")).lower()
    if consistency_metric != "ari":
        raise ValueError(
            f"Unsupported consistency_metric={consistency_metric!r}. "
            "Currently only 'ari' is supported."
        )

    consistency_pairs = str(scoring_cfg.get("consistency_pairs", "reference_to_all")).lower()
    if consistency_pairs not in {"reference_to_all", "adjacent", "all_pairs"}:
        raise ValueError(
            "Unsupported consistency_pairs="
            f"{consistency_pairs!r}. Use 'reference_to_all', 'adjacent', or 'all_pairs'."
        )

    consistency_aggregation = str(scoring_cfg.get("consistency_aggregation", "mean")).lower()
    if consistency_aggregation == "average":
        consistency_aggregation = "mean"
    if consistency_aggregation not in {"mean", "min"}:
        raise ValueError("consistency_aggregation must be 'mean' or 'min'.")

    silhouette_aggregation = str(scoring_cfg.get("silhouette_aggregation", "min")).lower()
    if silhouette_aggregation == "average":
        silhouette_aggregation = "mean"
    if silhouette_aggregation not in {"mean", "min"}:
        raise ValueError("silhouette_aggregation must be 'mean' or 'min'.")

    return {
        "timepoint_dfs": timepoint_dfs_resolved,
        "timepoint_order": timepoint_order,
        "reference_timepoint": resolved_reference,
        "reference_feature_cols": reference_feature_cols,
        "reference_feature_cols_original": reference_feature_cols_original,
        "timepoint_feature_cols": feature_map,
        "timepoint_feature_cols_original": raw_feature_map,
        "feature_mapping_strategy": strategy,
        "feature_mapping_summary": feature_mapping_summary,
        "subject_id_col": resolved_subject_id_col,
        "alignment_method": alignment_method,
        "consistency_metric": consistency_metric,
        "consistency_pairs": consistency_pairs,
        "consistency_aggregation": consistency_aggregation,
        "silhouette_aggregation": silhouette_aggregation,
    }


def _selected_timepoint_columns(
    *,
    reference_feature_cols: Sequence[str],
    timepoint_feature_cols: Mapping[str, Sequence[str]],
    selected_reference_cols: Sequence[str],
) -> Dict[str, List[str]]:
    """Map selected reference feature names to selected columns for each timepoint."""
    reference_feature_cols = list(reference_feature_cols)
    selected_reference_cols = list(selected_reference_cols)
    index_by_ref_col = {col: idx for idx, col in enumerate(reference_feature_cols)}

    missing_selected = [col for col in selected_reference_cols if col not in index_by_ref_col]
    if missing_selected:
        raise ValueError(f"Selected columns not found in reference feature columns: {missing_selected}")

    selected_indices = [index_by_ref_col[col] for col in selected_reference_cols]
    out: Dict[str, List[str]] = {}

    for tp, cols in timepoint_feature_cols.items():
        cols = list(cols)
        out[tp] = [cols[idx] for idx in selected_indices]

    return out


def _prepare_multitimepoint_feature_matrices(
    *,
    timepoint_dfs: Mapping[str, pd.DataFrame],
    timepoint_order: Sequence[str],
    selected_timepoint_cols: Mapping[str, Sequence[str]],
    subject_id_col: Optional[str],
) -> Tuple[Dict[str, np.ndarray], pd.Series, str]:
    """
    Build aligned feature matrices for all timepoints.

    If subject_id_col is provided and exists in every dataframe, rows are aligned
    by subject ID. Otherwise, rows are assumed to be aligned by position.
    """
    timepoint_order = list(timepoint_order)
    matrices: Dict[str, np.ndarray] = {}

    if subject_id_col is not None:
        prepared: Optional[pd.DataFrame] = None

        for tp in timepoint_order:
            df = timepoint_dfs[tp]
            cols = list(selected_timepoint_cols[tp])
            missing = [col for col in cols if col not in df.columns]
            if missing:
                raise ValueError(f"Dataframe for timepoint {tp!r} is missing columns: {missing}")
            if subject_id_col not in df.columns:
                raise ValueError(f"Dataframe for timepoint {tp!r} is missing subject_id_col={subject_id_col!r}.")

            sub = df[[subject_id_col] + cols].copy()
            for col in cols:
                sub[col] = pd.to_numeric(sub[col], errors="coerce")
            sub = sub.dropna(subset=cols).copy()
            rename = {col: f"{tp}__f{idx}" for idx, col in enumerate(cols)}
            sub = sub.rename(columns=rename)

            if prepared is None:
                prepared = sub
            else:
                prepared = prepared.merge(sub, on=subject_id_col, how="inner")

        if prepared is None:
            raise ValueError("No timepoint dataframes were provided.")

        subject_ids = prepared[subject_id_col].copy().reset_index(drop=True)
        for tp in timepoint_order:
            matrix_cols = [f"{tp}__f{idx}" for idx in range(len(selected_timepoint_cols[tp]))]
            matrices[tp] = prepared[matrix_cols].to_numpy(dtype=float)

        return matrices, subject_ids, "subject_id"

    # Row-order alignment.
    lengths = {tp: len(timepoint_dfs[tp]) for tp in timepoint_order}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            "When subject_id_col is not provided or unavailable, all timepoint "
            f"dataframes must have the same number of rows. Lengths: {lengths}"
        )

    numeric_dfs: Dict[str, pd.DataFrame] = {}
    valid_mask: Optional[pd.Series] = None

    for tp in timepoint_order:
        df = timepoint_dfs[tp]
        cols = list(selected_timepoint_cols[tp])
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f"Dataframe for timepoint {tp!r} is missing columns: {missing}")

        sub = df[cols].copy()
        # Rename to f0, f1, ... so different timepoint feature names still share
        # the same selected feature positions.
        sub.columns = [f"f{idx}" for idx in range(len(cols))]
        for col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")

        numeric_dfs[tp] = sub
        tp_valid = ~sub.isna().any(axis=1)
        valid_mask = tp_valid if valid_mask is None else (valid_mask & tp_valid)

    assert valid_mask is not None
    subject_ids = pd.Series(np.arange(next(iter(lengths.values()))), name="row_position").loc[valid_mask].reset_index(drop=True)

    for tp in timepoint_order:
        matrices[tp] = numeric_dfs[tp].loc[valid_mask].to_numpy(dtype=float)

    return matrices, subject_ids, "row_order"


def _build_consistency_pairs(
    *,
    timepoint_order: Sequence[str],
    reference_timepoint: str,
    consistency_pairs: str,
) -> List[Tuple[str, str]]:
    """Build timepoint pairs for cross-time ARI calculations."""
    order = list(timepoint_order)

    if consistency_pairs == "reference_to_all":
        return [(reference_timepoint, tp) for tp in order if tp != reference_timepoint]

    if consistency_pairs == "adjacent":
        return list(zip(order[:-1], order[1:]))

    if consistency_pairs == "all_pairs":
        pairs: List[Tuple[str, str]] = []
        for i, left in enumerate(order):
            for right in order[i + 1:]:
                pairs.append((left, right))
        return pairs

    raise ValueError(f"Unknown consistency_pairs={consistency_pairs!r}.")


def evaluate_feature_subset_longitudinal_clustering_multitimepoint(
    *,
    timepoint_dfs: Mapping[str, pd.DataFrame],
    feature_cols: Sequence[str],
    mask: Sequence[Any],
    cfg: ClinicalResponseGAFSConfig,
    subject_id_col: Optional[str] = None,
    reference_timepoint: Optional[str] = None,
    timepoint_feature_cols: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    """
    Evaluate one binary feature mask using a multi-timepoint longitudinal objective.

    The clustering model is fit on the reference timepoint only. The same fitted
    scaler and clustering model are then used to assign cluster membership at all
    timepoints. Cross-time ARI is summarized across configured timepoint pairs;
    silhouette is summarized across all timepoints.
    """
    rng = np.random.default_rng(cfg.random_seed)
    repaired_mask = _repair_mask(mask, cfg, rng)
    selected_cols = _selected_columns(feature_cols, repaired_mask)

    if len(selected_cols) < cfg.min_features:
        return {
            "fitness": -1e9,
            "selected_cols": selected_cols,
            "n_features": len(selected_cols),
            "error": "Too few selected features.",
        }

    try:
        longitudinal_cfg = resolve_multitimepoint_longitudinal_config(
            cfg,
            timepoint_dfs=timepoint_dfs,
            feature_cols=feature_cols,
            subject_id_col=subject_id_col,
            reference_timepoint=reference_timepoint,
            timepoint_feature_cols=timepoint_feature_cols,
        )

        selected_tp_cols = _selected_timepoint_columns(
            reference_feature_cols=longitudinal_cfg["reference_feature_cols"],
            timepoint_feature_cols=longitudinal_cfg["timepoint_feature_cols"],
            selected_reference_cols=selected_cols,
        )

        matrices, subject_ids, alignment_method = _prepare_multitimepoint_feature_matrices(
            timepoint_dfs=longitudinal_cfg["timepoint_dfs"],
            timepoint_order=longitudinal_cfg["timepoint_order"],
            selected_timepoint_cols=selected_tp_cols,
            subject_id_col=longitudinal_cfg["subject_id_col"],
        )
    except Exception as exc:
        return {
            "fitness": -1e9,
            "selected_cols": selected_cols,
            "n_features": len(selected_cols),
            "error": str(exc),
        }

    reference_tp = longitudinal_cfg["reference_timepoint"]
    timepoint_order = longitudinal_cfg["timepoint_order"]

    n_subjects = matrices[reference_tp].shape[0]
    min_rows_needed = max(cfg.k + 2, 5)
    if n_subjects < min_rows_needed:
        return {
            "fitness": -1e9,
            "selected_cols": selected_cols,
            "n_features": len(selected_cols),
            "n_paired_subjects": int(n_subjects),
            "error": "Too few aligned usable rows after dropping missing values.",
        }

    try:
        ref_labels, X_ref_scaled, model, scaler, model_name = _fit_cluster_labels(
            matrices[reference_tp],
            cfg=cfg,
        )

        labels_by_tp: Dict[str, np.ndarray] = {reference_tp: np.asarray(ref_labels)}
        scaled_by_tp: Dict[str, np.ndarray] = {reference_tp: X_ref_scaled}

        for tp in timepoint_order:
            if tp == reference_tp:
                continue
            X_scaled = scaler.transform(matrices[tp])
            scaled_by_tp[tp] = X_scaled
            if not hasattr(model, "predict"):
                raise ValueError(
                    f"Clustering model {model_name!r} must implement predict() "
                    "for non-reference timepoint assignment."
                )
            labels_by_tp[tp] = np.asarray(model.predict(X_scaled))
    except Exception as exc:
        return {
            "fitness": -1e9,
            "selected_cols": selected_cols,
            "n_features": len(selected_cols),
            "n_paired_subjects": int(n_subjects),
            "error": str(exc),
        }

    # Silhouette per timepoint.
    silhouette_raw_by_tp: Dict[str, float] = {}
    silhouette_norm_by_tp: Dict[str, float] = {}
    for tp in timepoint_order:
        q = compute_cluster_quality(scaled_by_tp[tp], labels_by_tp[tp])
        sil_raw = q.get("silhouette", np.nan)
        silhouette_raw_by_tp[tp] = float(sil_raw) if not pd.isna(sil_raw) else np.nan
        silhouette_norm_by_tp[tp] = _normalize_silhouette(sil_raw)

    longitudinal_silhouette_raw = _aggregate_numeric_values(
        [silhouette_raw_by_tp[tp] for tp in timepoint_order],
        method=longitudinal_cfg["silhouette_aggregation"],
    )
    longitudinal_silhouette_norm = _normalize_silhouette(longitudinal_silhouette_raw)

    # Cross-time ARI by configured pairs.
    pairs = _build_consistency_pairs(
        timepoint_order=timepoint_order,
        reference_timepoint=reference_tp,
        consistency_pairs=longitudinal_cfg["consistency_pairs"],
    )
    pair_rows: List[Dict[str, Any]] = []
    for left, right in pairs:
        try:
            ari_raw = float(adjusted_rand_score(labels_by_tp[left], labels_by_tp[right]))
        except Exception:
            ari_raw = np.nan
        pair_rows.append(
            {
                "timepoint_left": left,
                "timepoint_right": right,
                "ari_raw": ari_raw,
                "ari_norm": _normalize_ari(ari_raw),
            }
        )

    cross_time_ari_df = pd.DataFrame(pair_rows)
    cross_time_ari_raw = _aggregate_numeric_values(
        cross_time_ari_df["ari_raw"].tolist() if not cross_time_ari_df.empty else [],
        method=longitudinal_cfg["consistency_aggregation"],
    )
    cross_time_ari_norm = _normalize_ari(cross_time_ari_raw)

    feature_fraction = len(selected_cols) / max(len(feature_cols), cfg.eps)
    feature_penalty = float(feature_fraction ** cfg.feature_fraction_penalty_power)

    membership_data: Dict[str, Any] = {
        "subject_id": subject_ids.to_numpy(),
    }
    for tp in timepoint_order:
        membership_data[f"cluster_{tp}"] = labels_by_tp[tp]

    # Backward-compatible same_cluster_rate for two timepoints or reference_to_all.
    same_cluster_rates: Dict[str, float] = {}
    for left, right in pairs:
        same_cluster_rates[f"{left}__{right}"] = float(np.mean(labels_by_tp[left] == labels_by_tp[right]))
    same_cluster_rate = (
        float(np.mean(list(same_cluster_rates.values())))
        if len(same_cluster_rates) > 0
        else np.nan
    )
    for name, value in same_cluster_rates.items():
        membership_data[f"same_cluster_{name}"] = labels_by_tp[name.split("__")[0]] == labels_by_tp[name.split("__")[1]]

    membership_df = pd.DataFrame(membership_data)

    metrics_for_fitness: Dict[str, float] = {
        "cross_time_ari_raw": float(cross_time_ari_raw) if not pd.isna(cross_time_ari_raw) else np.nan,
        "cross_time_ari_norm": cross_time_ari_norm,
        "longitudinal_silhouette_raw": float(longitudinal_silhouette_raw) if not pd.isna(longitudinal_silhouette_raw) else np.nan,
        "longitudinal_silhouette_norm": longitudinal_silhouette_norm,
        "same_cluster_rate": same_cluster_rate,
        "feature_penalty_raw": feature_penalty,
        "feature_penalty_norm": feature_penalty,
        "n_features": float(len(selected_cols)),
        "n_total_features": float(len(feature_cols)),
        "n_paired_subjects": float(n_subjects),
    }

    # Include generic per-timepoint silhouette metrics in history/debug tables.
    for tp in timepoint_order:
        metrics_for_fitness[f"silhouette_{tp}_raw"] = silhouette_raw_by_tp[tp]
        metrics_for_fitness[f"silhouette_{tp}_norm"] = silhouette_norm_by_tp[tp]

    # Preserve previous two-timepoint metric names when baseline/week6 exist.
    if "baseline" in silhouette_raw_by_tp:
        metrics_for_fitness["baseline_silhouette_raw"] = silhouette_raw_by_tp["baseline"]
        metrics_for_fitness["baseline_silhouette_norm"] = silhouette_norm_by_tp["baseline"]
    if "week6" in silhouette_raw_by_tp:
        metrics_for_fitness["week6_silhouette_raw"] = silhouette_raw_by_tp["week6"]
        metrics_for_fitness["week6_silhouette_norm"] = silhouette_norm_by_tp["week6"]

    fitness, fitness_details = compute_dynamic_fitness(metrics_for_fitness, cfg=cfg)

    result: Dict[str, Any] = {
        "fitness": float(fitness),
        "selected_cols": selected_cols,
        "timepoint_selected_cols": selected_tp_cols,
        "feature_mapping_strategy": longitudinal_cfg.get("feature_mapping_strategy"),
        "feature_mapping_summary": longitudinal_cfg.get("feature_mapping_summary", {}),
        "resolved_reference_feature_cols": longitudinal_cfg.get("reference_feature_cols", []),
        "reference_feature_cols_original": longitudinal_cfg.get("reference_feature_cols_original", []),
        "n_features": int(len(selected_cols)),
        "n_paired_subjects": int(n_subjects),
        "cross_time_ari_raw": metrics_for_fitness["cross_time_ari_raw"],
        "cross_time_ari_norm": cross_time_ari_norm,
        "longitudinal_silhouette_raw": metrics_for_fitness["longitudinal_silhouette_raw"],
        "longitudinal_silhouette_norm": longitudinal_silhouette_norm,
        "same_cluster_rate": same_cluster_rate,
        "feature_penalty_raw": feature_penalty,
        "feature_penalty_norm": feature_penalty,
        "metrics_for_fitness": metrics_for_fitness,
        "fitness_details": fitness_details,
        "cluster_labels_by_timepoint": labels_by_tp,
        "cluster_labels": labels_by_tp[reference_tp],
        "membership_df": membership_df,
        "cross_time_ari_df": cross_time_ari_df,
        "timepoint_silhouettes_raw": silhouette_raw_by_tp,
        "timepoint_silhouettes_norm": silhouette_norm_by_tp,
        "model_name": model_name,
        "reference_timepoint": reference_tp,
        "timepoint_order": timepoint_order,
        "alignment_method": alignment_method,
        "details": {
            "longitudinal_config": longitudinal_cfg,
            "same_cluster_rates": same_cluster_rates,
        },
    }

    # Backward-compatible convenience keys for baseline/week6 examples.
    if "week6" in selected_tp_cols:
        result["week6_selected_cols"] = selected_tp_cols["week6"]
    if "baseline" in labels_by_tp:
        result["cluster_labels_baseline"] = labels_by_tp["baseline"]
        result["baseline_silhouette_raw"] = metrics_for_fitness.get("baseline_silhouette_raw", np.nan)
        result["baseline_silhouette_norm"] = metrics_for_fitness.get("baseline_silhouette_norm", np.nan)
    if "week6" in labels_by_tp:
        result["cluster_labels_week6"] = labels_by_tp["week6"]
        result["week6_silhouette_raw"] = metrics_for_fitness.get("week6_silhouette_raw", np.nan)
        result["week6_silhouette_norm"] = metrics_for_fitness.get("week6_silhouette_norm", np.nan)

    return result


class MultiTimepointLongitudinalFeatureSelectionGA:
    """
    Genetic algorithm runner for multi-timepoint longitudinal unsupervised feature selection.

    timepoint_dfs is a dictionary such as:
        {"baseline": df_baseline, "week6": df_week6, "week12": df_week12}

    The reference_timepoint, subject_id_col, and feature mapping are usually set
    under cfg.fitness_preset_config["longitudinal_unsupervised_clustering"]["timepoint_config"].
    """

    def __init__(
        self,
        *,
        timepoint_dfs: Optional[Mapping[str, pd.DataFrame]] = None,
        cfg: ClinicalResponseGAFSConfig,
        feature_cols: Optional[Sequence[str]] = None,
        subject_id_col: Optional[str] = None,
        reference_timepoint: Optional[str] = None,
        timepoint_feature_cols: Optional[Mapping[str, Sequence[str]]] = None,
        outdir: Optional[str] = None,
        save_outputs: bool = False,
    ) -> None:
        if cfg.fitness_preset is None:
            cfg.fitness_preset = "longitudinal_unsupervised_clustering"

        self.cfg = cfg
        self.save_outputs = bool(save_outputs)
        self.outdir = outdir

        apply_active_fitness_preset_config(self.cfg)

        preset_cfg = get_active_fitness_preset_config(self.cfg)
        timepoint_cfg = dict(preset_cfg.get("timepoint_config", {}) or {})

        # Preferred v4 usage keeps timepoint_dfs inside the active preset config.
        if timepoint_dfs is None:
            timepoint_dfs = timepoint_cfg.get("timepoint_dfs", None)

        if timepoint_dfs is None or len(timepoint_dfs) < 2:
            raise ValueError(
                "Longitudinal GA requires at least two timepoint dataframes. Provide "
                "timepoint_dfs=... or put timepoint_config['timepoint_dfs'] inside "
                "cfg.fitness_preset_config[cfg.fitness_preset]."
            )

        self.timepoint_dfs = {str(k): v.copy() for k, v in timepoint_dfs.items()}

        # The feature columns are inferred from the reference timepoint feature list
        # when feature_cols is not passed explicitly.
        self.feature_cols = resolve_feature_config(cfg, feature_cols=feature_cols)

        self.longitudinal_cfg = resolve_multitimepoint_longitudinal_config(
            self.cfg,
            timepoint_dfs=self.timepoint_dfs,
            feature_cols=self.feature_cols,
            subject_id_col=subject_id_col,
            reference_timepoint=reference_timepoint,
            timepoint_feature_cols=timepoint_feature_cols,
        )

        # The resolved longitudinal config may shrink/reorder the GA feature
        # space, especially with feature_mapping_strategy="intersection_by_name".
        # From this point forward, the GA chromosome uses only the resolved
        # reference-timepoint feature list.
        self.feature_cols = list(self.longitudinal_cfg["reference_feature_cols"])

        if self.save_outputs:
            if self.outdir is None:
                self.outdir = "longitudinal_clustering_ga_output"
            os.makedirs(self.outdir, exist_ok=True)

        self._cache: Dict[str, Dict[str, Any]] = {}
        self.history: List[Dict[str, Any]] = []
        self.raw_population_history: List[np.ndarray] = []
        self.population_history: List[np.ndarray] = []
        self._start_time: Optional[float] = None
        self._last_gen_time: Optional[float] = None

    def _mask_key(self, mask: Sequence[Any]) -> str:
        arr = (np.asarray(mask, dtype=float) >= 0.5).astype(int)
        return "".join(map(str, arr.tolist()))

    def evaluate(self, mask: Sequence[Any]) -> Dict[str, Any]:
        rng = np.random.default_rng(self.cfg.random_seed)
        repaired_mask = _repair_mask(mask, self.cfg, rng)
        key = self._mask_key(repaired_mask)

        if self.cfg.use_cache and key in self._cache:
            return self._cache[key]

        result = evaluate_feature_subset_longitudinal_clustering_multitimepoint(
            timepoint_dfs=self.timepoint_dfs,
            feature_cols=self.feature_cols,
            mask=repaired_mask,
            cfg=self.cfg,
            subject_id_col=self.longitudinal_cfg["subject_id_col"],
            reference_timepoint=self.longitudinal_cfg["reference_timepoint"],
            timepoint_feature_cols=self.longitudinal_cfg["timepoint_feature_cols"],
        )

        if self.cfg.use_cache:
            self._cache[key] = result

        return result

    def run(self) -> Dict[str, Any]:
        try:
            import pygad
        except ImportError as exc:
            raise ImportError("pygad is required to run the GA. Install it with: pip install pygad") from exc

        n_genes = len(self.feature_cols)
        initial_population = make_sparse_initial_population(
            n_genes,
            sol_per_pop=self.cfg.sol_per_pop,
            min_features=self.cfg.min_features,
            max_features=self.cfg.max_features,
            random_seed=self.cfg.random_seed,
        )

        self._start_time = time.time()
        self._last_gen_time = self._start_time

        def fitness_func(ga_instance: Any, solution: np.ndarray, solution_idx: int) -> float:
            evaluation = self.evaluate(solution)
            return float(evaluation["fitness"])

        def on_generation(ga_instance: Any) -> None:
            raw_population = np.asarray(ga_instance.population).copy()
            self.raw_population_history.append(raw_population)
            repaired_population = np.vstack(
                [
                    _repair_mask(solution, self.cfg, np.random.default_rng(self.cfg.random_seed))
                    for solution in raw_population
                ]
            )
            self.population_history.append(repaired_population)

            now = time.time()
            gen_time = now - (self._last_gen_time or now)
            total_time = now - (self._start_time or now)
            self._last_gen_time = now

            best_solution, best_fitness, _ = ga_instance.best_solution()
            best_eval = self.evaluate(best_solution)

            row = {
                "generation": int(ga_instance.generations_completed),
                "best_fitness": float(best_fitness),
                "n_features": int(best_eval.get("n_features", 0)),
                "n_paired_subjects": int(best_eval.get("n_paired_subjects", 0)),
                "generation_time_sec": float(gen_time),
                "total_elapsed_sec": float(total_time),
                "solution_mask": self._mask_key(best_solution),
                "selected_features": " | ".join(best_eval.get("selected_cols", [])),
                "model_name": best_eval.get("model_name", None),
                "cross_time_ari_raw": best_eval.get("cross_time_ari_raw", np.nan),
                "cross_time_ari_norm": best_eval.get("cross_time_ari_norm", np.nan),
                "longitudinal_silhouette_raw": best_eval.get("longitudinal_silhouette_raw", np.nan),
                "longitudinal_silhouette_norm": best_eval.get("longitudinal_silhouette_norm", np.nan),
                "same_cluster_rate": best_eval.get("same_cluster_rate", np.nan),
                "feature_penalty_raw": best_eval.get("feature_penalty_raw", np.nan),
                "feature_penalty_norm": best_eval.get("feature_penalty_norm", np.nan),
                "fitness_function_name": best_eval.get("fitness_details", {}).get("fitness_function_name", None),
                "fitness_contributions": best_eval.get("fitness_details", {}).get("fitness_contributions", None),
            }

            # Add any extra metrics requested by logging_config from metrics_for_fitness.
            metrics_for_fitness = best_eval.get("metrics_for_fitness", {}) or {}
            for metric_name, metric_value in metrics_for_fitness.items():
                if metric_name not in row:
                    row[metric_name] = metric_value

            fitness_contributions = best_eval.get("fitness_details", {}).get("fitness_contributions", {}) or {}
            metric_pairs, _ = get_active_metric_display_config(self.cfg)
            for user_metric_name, internal_metric_name in metric_pairs:
                if internal_metric_name in metrics_for_fitness:
                    row[user_metric_name] = metrics_for_fitness[internal_metric_name]
                elif internal_metric_name in row:
                    row[user_metric_name] = row[internal_metric_name]

                if internal_metric_name in fitness_contributions:
                    row[f"{user_metric_name}__weighted"] = fitness_contributions[internal_metric_name]

            self.history.append(row)
            print(build_generation_log_message(row, cfg=self.cfg))

        ga_instance = pygad.GA(
            num_generations=self.cfg.num_generations,
            sol_per_pop=self.cfg.sol_per_pop,
            num_parents_mating=self.cfg.num_parents_mating,
            keep_parents=self.cfg.keep_parents,
            keep_elitism=self.cfg.keep_elitism,
            num_genes=n_genes,
            gene_space=[0, 1],
            gene_type=int,
            initial_population=initial_population,
            parent_selection_type=self.cfg.parent_selection_type,
            crossover_type=self.cfg.crossover_type,
            mutation_type=self.cfg.mutation_type,
            mutation_percent_genes=self.cfg.mutation_percent_genes,
            random_seed=self.cfg.random_seed,
            fitness_func=fitness_func,
            on_generation=on_generation,
        )

        ga_instance.run()
        elapsed = time.time() - (self._start_time or time.time())

        best_solution, best_fitness, _ = ga_instance.best_solution()
        rng = np.random.default_rng(self.cfg.random_seed)
        best_mask = _repair_mask(best_solution, self.cfg, rng)
        best_eval = self.evaluate(best_mask)

        history_df = pd.DataFrame(self.history)
        feature_selection_frequency_df = compute_feature_selection_frequency(self.population_history, self.feature_cols)

        result: Dict[str, Any] = {
            "ga_instance": ga_instance,
            "best_solution": np.asarray(best_solution, dtype=int),
            "best_mask": np.asarray(best_mask, dtype=int),
            "best_fitness": float(best_fitness),
            "best_eval": best_eval,
            "selected_cols": best_eval.get("selected_cols", []),
            "timepoint_selected_cols": best_eval.get("timepoint_selected_cols", {}),
            "membership_df": best_eval.get("membership_df", pd.DataFrame()),
            "cross_time_ari_df": best_eval.get("cross_time_ari_df", pd.DataFrame()),
            "timepoint_silhouettes_raw": best_eval.get("timepoint_silhouettes_raw", {}),
            "timepoint_silhouettes_norm": best_eval.get("timepoint_silhouettes_norm", {}),
            "history_df": history_df,
            "raw_population_history": self.raw_population_history,
            "population_history": self.population_history,
            "feature_selection_frequency_df": feature_selection_frequency_df,
            "config": safe_config_dict(self.cfg),
            "feature_cols": self.feature_cols,
            "feature_mapping_strategy": best_eval.get("feature_mapping_strategy", self.longitudinal_cfg.get("feature_mapping_strategy")),
            "feature_mapping_summary": best_eval.get("feature_mapping_summary", self.longitudinal_cfg.get("feature_mapping_summary", {})),
            "resolved_reference_feature_cols": best_eval.get("resolved_reference_feature_cols", self.longitudinal_cfg.get("reference_feature_cols", [])),
            "reference_feature_cols_original": best_eval.get("reference_feature_cols_original", self.longitudinal_cfg.get("reference_feature_cols_original", [])),
            "timepoint_order": best_eval.get("timepoint_order", self.longitudinal_cfg["timepoint_order"]),
            "reference_timepoint": best_eval.get("reference_timepoint", self.longitudinal_cfg["reference_timepoint"]),
            "subject_id_col": self.longitudinal_cfg["subject_id_col"],
            "alignment_method": best_eval.get("alignment_method", self.longitudinal_cfg["alignment_method"]),
        }

        # Backward-compatible keys for baseline/week6 examples.
        if "week6" in result["timepoint_selected_cols"]:
            result["week6_selected_cols"] = result["timepoint_selected_cols"]["week6"]

        self._save_outputs(result)

        print("\n" + "=" * 80)
        print(f"Longitudinal GA completed in {elapsed:.2f} sec")
        print(f"Longitudinal GA completed in {elapsed / 60:.2f} min")
        if self.cfg.num_generations > 0:
            print(f"Average time per generation: {elapsed / self.cfg.num_generations:.2f} sec")
        print("=" * 80)

        return result

    def _save_outputs(self, result: Dict[str, Any]) -> None:
        if not self.save_outputs:
            return
        if self.outdir is None:
            raise ValueError("outdir must be provided when save_outputs=True.")
        os.makedirs(self.outdir, exist_ok=True)
        history_df = result.get("history_df", pd.DataFrame())
        if isinstance(history_df, pd.DataFrame):
            history_df.to_csv(os.path.join(self.outdir, "history.csv"), index=False)

        selected_cols = result.get("timepoint_selected_cols", {})
        if isinstance(selected_cols, Mapping) and selected_cols:
            pd.DataFrame(dict(selected_cols)).to_csv(os.path.join(self.outdir, "selected_features_by_timepoint.csv"), index=False)
        else:
            pd.DataFrame({"selected_feature": result.get("selected_cols", [])}).to_csv(
                os.path.join(self.outdir, "selected_features.csv"), index=False
            )

        frequency_df = result.get("feature_selection_frequency_df", pd.DataFrame())
        if isinstance(frequency_df, pd.DataFrame):
            frequency_df.to_csv(os.path.join(self.outdir, "feature_selection_frequency.csv"), index=False)

        membership_df = result.get("membership_df", pd.DataFrame())
        if isinstance(membership_df, pd.DataFrame):
            membership_df.to_csv(os.path.join(self.outdir, "best_membership_consistency.csv"), index=False)

        ari_df = result.get("cross_time_ari_df", pd.DataFrame())
        if isinstance(ari_df, pd.DataFrame):
            ari_df.to_csv(os.path.join(self.outdir, "cross_time_ari.csv"), index=False)


def make_longitudinal_clustering_ga(  # type: ignore[no-redef]
    *,
    cfg: Optional[ClinicalResponseGAFSConfig] = None,
    outdir: Optional[str] = None,
    save_outputs: bool = False,
    # Optional overrides / backward-compatible arguments:
    timepoint_dfs: Optional[Mapping[str, pd.DataFrame]] = None,
    feature_cols: Optional[Sequence[str]] = None,
    subject_id_col: Optional[str] = None,
    reference_timepoint: Optional[str] = None,
    timepoint_feature_cols: Optional[Mapping[str, Sequence[str]]] = None,
    df_baseline: Optional[pd.DataFrame] = None,
    df_week6: Optional[pd.DataFrame] = None,
    week6_feature_cols: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
) -> MultiTimepointLongitudinalFeatureSelectionGA:
    """
    Convenience constructor for multi-timepoint longitudinal clustering GA.

    Preferred v4 usage keeps timepoint dataframes and feature mappings in the
    active preset config, so the call can be minimal:

        make_longitudinal_clustering_ga(cfg=cfg, save_outputs=False)

    Backward-compatible usage is still accepted through timepoint_dfs={...} or
    df_baseline/df_week6 arguments. Explicit function arguments override config
    values when provided.
    """
    if cfg is None:
        cfg = ClinicalResponseGAFSConfig(fitness_preset="longitudinal_unsupervised_clustering")

    if timepoint_dfs is None and (df_baseline is not None or df_week6 is not None):
        if df_baseline is None or df_week6 is None:
            raise ValueError("Provide both df_baseline and df_week6, or neither.")
        timepoint_dfs = {"baseline": df_baseline, "week6": df_week6}
        if timepoint_feature_cols is None and week6_feature_cols is not None:
            resolved_feature_cols = resolve_feature_config(cfg, feature_cols=feature_cols)
            if isinstance(week6_feature_cols, Mapping):
                # Convert mapping baseline_col -> week6_col into a week6 ordered list.
                week6_list = [str(week6_feature_cols[col]) for col in resolved_feature_cols]
            else:
                week6_list = list(week6_feature_cols)
            timepoint_feature_cols = {
                "baseline": list(resolved_feature_cols),
                "week6": week6_list,
            }

    return MultiTimepointLongitudinalFeatureSelectionGA(
        timepoint_dfs=timepoint_dfs,
        cfg=cfg,
        feature_cols=feature_cols,
        subject_id_col=subject_id_col,
        reference_timepoint=reference_timepoint,
        timepoint_feature_cols=timepoint_feature_cols,
        outdir=outdir,
        save_outputs=save_outputs,
    )


def evaluate_mask_longitudinal_clustering(  # type: ignore[no-redef]
    *,
    mask: Sequence[Any],
    cfg: ClinicalResponseGAFSConfig,
    timepoint_dfs: Optional[Mapping[str, pd.DataFrame]] = None,
    feature_cols: Optional[Sequence[str]] = None,
    subject_id_col: Optional[str] = None,
    reference_timepoint: Optional[str] = None,
    timepoint_feature_cols: Optional[Mapping[str, Sequence[str]]] = None,
    # Backward-compatible two-timepoint arguments:
    df_baseline: Optional[pd.DataFrame] = None,
    df_week6: Optional[pd.DataFrame] = None,
    week6_feature_cols: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Evaluate one longitudinal feature mask without running the GA.

    Preferred usage passes timepoint_dfs. Backward-compatible df_baseline/df_week6
    arguments are still accepted.
    """
    apply_active_fitness_preset_config(cfg)
    resolved_feature_cols = resolve_feature_config(cfg, feature_cols=feature_cols)

    if timepoint_dfs is None and (df_baseline is not None or df_week6 is not None):
        if df_baseline is None or df_week6 is None:
            raise ValueError("Provide both df_baseline and df_week6, or neither.")
        timepoint_dfs = {"baseline": df_baseline, "week6": df_week6}
        if timepoint_feature_cols is None and week6_feature_cols is not None:
            if isinstance(week6_feature_cols, Mapping):
                week6_list = [str(week6_feature_cols[col]) for col in resolved_feature_cols]
            else:
                week6_list = list(week6_feature_cols)
            timepoint_feature_cols = {
                "baseline": list(resolved_feature_cols),
                "week6": week6_list,
            }

    if timepoint_dfs is None:
        preset_cfg = get_active_fitness_preset_config(cfg)
        timepoint_cfg = dict(preset_cfg.get("timepoint_config", {}) or {})
        timepoint_dfs = timepoint_cfg.get("timepoint_dfs", None)

    if timepoint_dfs is None:
        raise ValueError(
            "Provide timepoint_dfs=..., df_baseline/df_week6, or put "
            "timepoint_config['timepoint_dfs'] inside the active fitness preset config."
        )

    return evaluate_feature_subset_longitudinal_clustering_multitimepoint(
        timepoint_dfs=timepoint_dfs,
        feature_cols=resolved_feature_cols,
        mask=mask,
        cfg=cfg,
        subject_id_col=subject_id_col,
        reference_timepoint=reference_timepoint,
        timepoint_feature_cols=timepoint_feature_cols,
    )


# =============================================================================
# Longitudinal label-guided clustering extension
# =============================================================================
# This section adds a second longitudinal preset:
#
#     longitudinal_label_guided_clustering
#
# The clustering model still uses selected feature columns only. Labels are used
# only after clustering to score how well each timepoint's cluster assignment
# aligns with the corresponding timepoint label column.


def longitudinal_label_guided_clustering_fitness(
    metrics: Mapping[str, float],
    components: Mapping[str, Mapping[str, Any]],
) -> Tuple[float, Dict[str, float]]:
    """
    Standalone longitudinal label-guided clustering fitness function.

    Calculation
    -----------
    fitness =
        cross_time_weight * cross_time_ari_norm
      + silhouette_weight * longitudinal_silhouette_norm
      + label_weight * longitudinal_label_alignment_norm
      - feature_penalty_weight * feature_penalty_norm

    Labels are not used to create clusters. They are used only after clustering
    to score cluster-label agreement at each configured timepoint.
    """
    cross_time_ari = _metric_value(metrics, "cross_time_ari_norm")
    longitudinal_silhouette = _metric_value(metrics, "longitudinal_silhouette_norm")
    longitudinal_label_alignment = _metric_value(metrics, "longitudinal_label_alignment_norm")
    feature_penalty = _metric_value(metrics, "feature_penalty_norm")

    w_cross_time = _component_weight(components, "cross_time_ari_norm")
    w_silhouette = _component_weight(components, "longitudinal_silhouette_norm")
    w_label = _component_weight(components, "longitudinal_label_alignment_norm")
    w_penalty = _component_weight(components, "feature_penalty_norm")

    contributions = {
        "cross_time_ari_norm": w_cross_time * cross_time_ari,
        "longitudinal_silhouette_norm": w_silhouette * longitudinal_silhouette,
        "longitudinal_label_alignment_norm": w_label * longitudinal_label_alignment,
        "feature_penalty_norm": -w_penalty * feature_penalty,
    }

    fitness = sum(contributions.values())
    return float(fitness), {k: float(v) for k, v in contributions.items()}


# Preserve current v4 implementations before extending the registry again.
_V4_DEFAULT_METRIC_DISPLAY_LABEL = default_metric_display_label
_V4_GET_FITNESS_PRESET_COMPONENTS = get_fitness_preset_components
_V4_GET_NAMED_FITNESS_FUNCTION = get_named_fitness_function
_V4_EVALUATE_FEATURE_SUBSET_LONGITUDINAL_MULTITIMEPOINT = evaluate_feature_subset_longitudinal_clustering_multitimepoint


FITNESS_METRIC_ALIASES.update(
    {
        "longitudinal_label_alignment_norm": "longitudinal_label_alignment_norm",
        "longitudinal_label_alignment_raw": "longitudinal_label_alignment_raw",
        "label_alignment_longitudinal_norm": "longitudinal_label_alignment_norm",
        "label_alignment_longitudinal_raw": "longitudinal_label_alignment_raw",
    }
)


def default_metric_display_label(metric_name: str) -> str:  # type: ignore[no-redef]
    """Provide concise labels for common metric names, including longitudinal label metrics."""
    labels = {
        "longitudinal_label_alignment_norm": "longLabel",
        "longitudinal_label_alignment_raw": "longLabel_raw",
        "label_alignment_longitudinal_norm": "longLabel",
        "label_alignment_longitudinal_raw": "longLabel_raw",
    }
    return labels.get(metric_name, _V4_DEFAULT_METRIC_DISPLAY_LABEL(metric_name))


def get_fitness_preset_components(  # type: ignore[no-redef]
    preset_name: str,
    *,
    cfg: ClinicalResponseGAFSConfig,
) -> Dict[str, Dict[str, Any]]:
    """
    Return default component weights for a named fitness function.

    Adds longitudinal_label_guided_clustering while preserving existing presets.
    """
    name = str(preset_name).lower()

    if name == "longitudinal_label_guided_clustering":
        return {
            "cross_time_ari_norm": {
                "weight": 0.45,
                "direction": "maximize",
                "description": "Cross-time membership consistency ARI",
            },
            "longitudinal_silhouette_norm": {
                "weight": 0.15,
                "direction": "maximize",
                "description": "Longitudinal silhouette summary across timepoints",
            },
            "longitudinal_label_alignment_norm": {
                "weight": 0.35,
                "direction": "maximize",
                "description": "Aggregated cluster-label agreement across timepoints",
            },
            "feature_penalty_norm": {
                "weight": 0.10,
                "direction": "minimize",
                "description": "Feature-count penalty",
            },
        }

    return _V4_GET_FITNESS_PRESET_COMPONENTS(preset_name, cfg=cfg)


def get_named_fitness_function(  # type: ignore[no-redef]
    preset_name: str,
) -> Callable[[Mapping[str, float], Mapping[str, Mapping[str, Any]]], Tuple[float, Dict[str, float]]]:
    """
    Return the standalone fitness function associated with a preset name.

    Adds longitudinal_label_guided_clustering while preserving existing presets.
    """
    name = str(preset_name).lower()

    if name == "longitudinal_label_guided_clustering":
        return longitudinal_label_guided_clustering_fitness

    return _V4_GET_NAMED_FITNESS_FUNCTION(preset_name)


def _resolve_longitudinal_label_scoring_config(cfg: ClinicalResponseGAFSConfig) -> Dict[str, Any]:
    """
    Resolve label-scoring configuration for longitudinal label-guided presets.

    Preferred layout
    ----------------
    cfg.fitness_preset_config[preset]["scoring_columns"] = {
        "timepoint_label_cols": {
            "baseline": "label_baseline",
            "week6": "label_week6",
        }
    }

    cfg.fitness_preset_config[preset]["label_alignment"] = {
        "metric": "ari_nmi",      # "ari", "nmi", or "ari_nmi"
        "aggregation": "min",     # "min" or "mean"
    }
    """
    preset_name = cfg.fitness_preset or "longitudinal_label_guided_clustering"
    preset_cfg = get_active_fitness_preset_config(cfg)
    scoring_columns = dict(preset_cfg.get("scoring_columns", {}) or {})
    label_cfg = dict(preset_cfg.get("label_alignment", {}) or {})

    tp_label_cols = scoring_columns.get("timepoint_label_cols", None)

    # Convenience fallback: one label_col name shared by every timepoint.
    if tp_label_cols is None and scoring_columns.get("label_col"):
        timepoint_cfg = dict(preset_cfg.get("timepoint_config", {}) or {})
        timepoint_order = timepoint_cfg.get("timepoint_order")
        timepoint_dfs = timepoint_cfg.get("timepoint_dfs", {}) or {}
        if timepoint_order is None:
            timepoint_order = list(dict(timepoint_dfs).keys())
        tp_label_cols = {str(tp): scoring_columns["label_col"] for tp in timepoint_order}

    if not tp_label_cols:
        raise ValueError(
            f"fitness_preset={preset_name!r} requires scoring_columns['timepoint_label_cols']. "
            "Provide a mapping from timepoint name to label column name."
        )

    return {
        "timepoint_label_cols": {str(k): str(v) for k, v in dict(tp_label_cols).items()},
        "metric": str(label_cfg.get("metric", "ari_nmi")).lower(),
        "aggregation": str(label_cfg.get("aggregation", "min")).lower(),
    }


def _label_values_aligned_to_membership(
    *,
    df: pd.DataFrame,
    label_col: str,
    membership_df: pd.DataFrame,
    cluster_labels: Sequence[Any],
    subject_id_col: Optional[str],
    alignment_method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return label values and cluster labels aligned to membership_df rows.

    If subject IDs are available, alignment uses subject_id. Otherwise,
    membership_df['subject_id'] is interpreted as row position.
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col!r} is missing from dataframe.")

    labels_arr = np.asarray(cluster_labels)
    if len(labels_arr) != len(membership_df):
        raise ValueError(
            "cluster label vector length does not match membership_df length: "
            f"{len(labels_arr)} vs {len(membership_df)}."
        )

    tmp = pd.DataFrame({
        "_cluster": labels_arr,
        "_membership_row": np.arange(len(membership_df)),
    })

    if alignment_method == "subject_id" and subject_id_col is not None and subject_id_col in df.columns:
        tmp["_subject_id"] = membership_df["subject_id"].to_numpy()
        labels_df = df[[subject_id_col, label_col]].copy()
        labels_df = labels_df.rename(columns={subject_id_col: "_subject_id", label_col: "_label"})
        merged = tmp.merge(labels_df, on="_subject_id", how="left")
    else:
        positions = pd.to_numeric(membership_df["subject_id"], errors="coerce")
        if positions.isna().any():
            raise ValueError(
                "Row-order label alignment requires membership_df['subject_id'] "
                "to contain integer row positions."
            )
        positions_int = positions.astype(int).to_numpy()
        if np.any(positions_int < 0) or np.any(positions_int >= len(df)):
            raise ValueError("Row-order label alignment found row positions outside dataframe bounds.")
        merged = tmp.copy()
        merged["_label"] = df.iloc[positions_int][label_col].to_numpy()

    merged = merged.dropna(subset=["_label", "_cluster"]).copy()
    return merged["_label"].to_numpy(), merged["_cluster"].to_numpy()


def _compute_longitudinal_label_alignment_metrics(
    *,
    result: Mapping[str, Any],
    cfg: ClinicalResponseGAFSConfig,
) -> Dict[str, Any]:
    """
    Compute label alignment at each configured timepoint and aggregate it.
    """
    label_cfg = _resolve_longitudinal_label_scoring_config(cfg)
    timepoint_label_cols = label_cfg["timepoint_label_cols"]
    metric = label_cfg["metric"]
    aggregation = label_cfg["aggregation"]

    details = dict(result.get("details", {}) or {})
    longitudinal_cfg = dict(details.get("longitudinal_config", {}) or {})
    timepoint_dfs = dict(longitudinal_cfg.get("timepoint_dfs", {}) or {})
    timepoint_order = list(result.get("timepoint_order", longitudinal_cfg.get("timepoint_order", [])))
    labels_by_tp = dict(result.get("cluster_labels_by_timepoint", {}) or {})
    membership_df = result.get("membership_df", pd.DataFrame())
    alignment_method = str(result.get("alignment_method", longitudinal_cfg.get("alignment_method", "row_order")))
    subject_id_col = longitudinal_cfg.get("subject_id_col", None)

    if not isinstance(membership_df, pd.DataFrame) or membership_df.empty:
        return {
            "label_alignment_df": pd.DataFrame(),
            "longitudinal_label_alignment_raw": np.nan,
            "longitudinal_label_alignment_norm": 0.0,
        }

    rows: List[Dict[str, Any]] = []
    for tp in timepoint_order:
        if tp not in timepoint_label_cols:
            continue
        if tp not in timepoint_dfs:
            continue
        if tp not in labels_by_tp:
            continue

        label_col = timepoint_label_cols[tp]
        try:
            y_true, y_cluster = _label_values_aligned_to_membership(
                df=timepoint_dfs[tp],
                label_col=label_col,
                membership_df=membership_df,
                cluster_labels=labels_by_tp[tp],
                subject_id_col=subject_id_col,
                alignment_method=alignment_method,
            )
            if len(y_true) == 0:
                raise ValueError("No non-missing labels after alignment.")
            label_dict = compute_label_alignment(y_true, y_cluster, metric=metric)
            rows.append(
                {
                    "timepoint": tp,
                    "label_col": label_col,
                    "n_labeled": int(len(y_true)),
                    "label_alignment_raw": float(label_dict["label_alignment_raw"]),
                    "label_alignment_norm": float(label_dict["label_alignment_norm"]),
                    "label_ari_raw": float(label_dict["label_ari_raw"]),
                    "label_nmi_raw": float(label_dict["label_nmi_raw"]),
                    "metric": label_dict.get("metric", metric),
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "timepoint": tp,
                    "label_col": label_col,
                    "n_labeled": 0,
                    "label_alignment_raw": np.nan,
                    "label_alignment_norm": 0.0,
                    "label_ari_raw": np.nan,
                    "label_nmi_raw": np.nan,
                    "metric": metric,
                    "error": str(exc),
                }
            )

    label_alignment_df = pd.DataFrame(rows)
    if label_alignment_df.empty:
        raw = np.nan
        norm = 0.0
    else:
        raw = _aggregate_numeric_values(label_alignment_df["label_alignment_raw"].tolist(), method=aggregation)
        norm = _aggregate_numeric_values(label_alignment_df["label_alignment_norm"].tolist(), method=aggregation)
        norm = 0.0 if pd.isna(norm) else float(np.clip(norm, 0.0, 1.0))

    return {
        "label_alignment_df": label_alignment_df,
        "longitudinal_label_alignment_raw": float(raw) if not pd.isna(raw) else np.nan,
        "longitudinal_label_alignment_norm": norm,
        "label_alignment_aggregation": aggregation,
        "label_alignment_metric": metric,
    }


def evaluate_feature_subset_longitudinal_clustering_multitimepoint(  # type: ignore[no-redef]
    *,
    timepoint_dfs: Mapping[str, pd.DataFrame],
    feature_cols: Sequence[str],
    mask: Sequence[Any],
    cfg: ClinicalResponseGAFSConfig,
    subject_id_col: Optional[str] = None,
    reference_timepoint: Optional[str] = None,
    timepoint_feature_cols: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    """
    Evaluate one binary feature mask for multi-timepoint longitudinal clustering.

    For longitudinal_label_guided_clustering, this wraps the unsupervised
    longitudinal evaluator, then adds timepoint-level label alignment metrics
    and recomputes the final fitness using the label-guided preset.
    """
    result = _V4_EVALUATE_FEATURE_SUBSET_LONGITUDINAL_MULTITIMEPOINT(
        timepoint_dfs=timepoint_dfs,
        feature_cols=feature_cols,
        mask=mask,
        cfg=cfg,
        subject_id_col=subject_id_col,
        reference_timepoint=reference_timepoint,
        timepoint_feature_cols=timepoint_feature_cols,
    )

    # If the base evaluator failed, return the failure as-is.
    if result.get("fitness", None) == -1e9 or result.get("error"):
        return result

    preset_name = str(cfg.fitness_preset or "").lower()
    components = active_fitness_components(cfg)
    needs_longitudinal_label = (
        preset_name == "longitudinal_label_guided_clustering"
        or bool(components.get("longitudinal_label_alignment_norm", {}).get("weight", 0.0) != 0)
    )

    if not needs_longitudinal_label:
        return result

    label_metrics = _compute_longitudinal_label_alignment_metrics(result=result, cfg=cfg)

    metrics_for_fitness = dict(result.get("metrics_for_fitness", {}) or {})
    metrics_for_fitness.update(
        {
            "longitudinal_label_alignment_raw": label_metrics["longitudinal_label_alignment_raw"],
            "longitudinal_label_alignment_norm": label_metrics["longitudinal_label_alignment_norm"],
        }
    )

    label_alignment_df = label_metrics.get("label_alignment_df", pd.DataFrame())
    if isinstance(label_alignment_df, pd.DataFrame) and not label_alignment_df.empty:
        for _, row in label_alignment_df.iterrows():
            tp = str(row["timepoint"])
            metrics_for_fitness[f"label_alignment_{tp}_raw"] = row.get("label_alignment_raw", np.nan)
            metrics_for_fitness[f"label_alignment_{tp}_norm"] = row.get("label_alignment_norm", np.nan)
            metrics_for_fitness[f"label_ari_{tp}_raw"] = row.get("label_ari_raw", np.nan)
            metrics_for_fitness[f"label_nmi_{tp}_raw"] = row.get("label_nmi_raw", np.nan)

    fitness, fitness_details = compute_dynamic_fitness(metrics_for_fitness, cfg=cfg)

    result.update(
        {
            "fitness": float(fitness),
            "longitudinal_label_alignment_raw": metrics_for_fitness["longitudinal_label_alignment_raw"],
            "longitudinal_label_alignment_norm": metrics_for_fitness["longitudinal_label_alignment_norm"],
            "label_alignment_df": label_alignment_df,
            "metrics_for_fitness": metrics_for_fitness,
            "fitness_details": fitness_details,
        }
    )
    result.setdefault("details", {})
    result["details"]["label_alignment"] = {
        "metric": label_metrics.get("label_alignment_metric"),
        "aggregation": label_metrics.get("label_alignment_aggregation"),
    }

    return result




# =============================================================================
# Longitudinal clinical-score-guided clustering extension
# =============================================================================
# This section adds a third longitudinal preset:
#
#     longitudinal_clinical_score_guided_clustering
#
# Clinical scores are NOT used to create clusters.
# They are used only after clustering to score whether discovered clusters
# separate clinical severity/functioning across timepoints.
#
# Recommended first clinical score metric:
#     longitudinal_clinical_score_separation_norm
#
# This uses ANOVA-style eta-squared:
#     between-cluster clinical score variance / total clinical score variance
#
# This works naturally for KMeans k=2, k=3, or k=4.


# Preserve current v5 implementations before extending the registry again.
# At this point, evaluate_feature_subset_longitudinal_clustering_multitimepoint
# is already the label-guided wrapper. We preserve it so the clinical wrapper
# can build on top of both the unsupervised and optional label-guided logic.
_V5_DEFAULT_METRIC_DISPLAY_LABEL = default_metric_display_label
_V5_GET_FITNESS_PRESET_COMPONENTS = get_fitness_preset_components
_V5_GET_NAMED_FITNESS_FUNCTION = get_named_fitness_function
_V5_EVALUATE_FEATURE_SUBSET_LONGITUDINAL_MULTITIMEPOINT = (
    evaluate_feature_subset_longitudinal_clustering_multitimepoint
)


FITNESS_METRIC_ALIASES.update(
    {
        "longitudinal_clinical_score_separation_norm": "longitudinal_clinical_score_separation_norm",
        "longitudinal_clinical_score_separation_raw": "longitudinal_clinical_score_separation_raw",
        "clinical_score_separation_longitudinal_norm": "longitudinal_clinical_score_separation_norm",
        "clinical_score_separation_longitudinal_raw": "longitudinal_clinical_score_separation_raw",
    }
)


def default_metric_display_label(metric_name: str) -> str:  # type: ignore[no-redef]
    """
    Provide concise labels for common metric names, including longitudinal
    clinical-score metrics.
    """
    labels = {
        "longitudinal_clinical_score_separation_norm": "clinSep",
        "longitudinal_clinical_score_separation_raw": "clinSep_raw",
        "clinical_score_separation_longitudinal_norm": "clinSep",
        "clinical_score_separation_longitudinal_raw": "clinSep_raw",
    }
    return labels.get(metric_name, _V5_DEFAULT_METRIC_DISPLAY_LABEL(metric_name))


def get_fitness_preset_components(  # type: ignore[no-redef]
    preset_name: str,
    *,
    cfg: ClinicalResponseGAFSConfig,
) -> Dict[str, Dict[str, Any]]:
    """
    Return default component weights for a named fitness function.

    Adds longitudinal_clinical_score_guided_clustering while preserving all
    existing presets.
    """
    name = str(preset_name).lower()

    if name == "longitudinal_clinical_score_guided_clustering":
        return {
            "longitudinal_clinical_score_separation_norm": {
                "weight": 0.60,
                "direction": "maximize",
                "description": "ANOVA/eta-squared clinical score separation across timepoints",
            },
            "cross_time_ari_norm": {
                "weight": 0.20,
                "direction": "maximize",
                "description": "Cross-time membership consistency",
            },
            "longitudinal_silhouette_norm": {
                "weight": 0.15,
                "direction": "maximize",
                "description": "Cluster separation across timepoints",
            },
            "feature_penalty_norm": {
                "weight": 0.05,
                "direction": "minimize",
                "description": "Feature-count penalty",
            },
        }

    return _V5_GET_FITNESS_PRESET_COMPONENTS(preset_name, cfg=cfg)


def get_named_fitness_function(  # type: ignore[no-redef]
    preset_name: str,
) -> Callable[[Mapping[str, float], Mapping[str, Mapping[str, Any]]], Tuple[float, Dict[str, float]]]:
    """
    Return the standalone fitness function associated with a preset name.

    Adds longitudinal_clinical_score_guided_clustering while preserving all
    existing presets.
    """
    name = str(preset_name).lower()

    if name == "longitudinal_clinical_score_guided_clustering":
        return longitudinal_clinical_score_guided_clustering_fitness

    return _V5_GET_NAMED_FITNESS_FUNCTION(preset_name)


def evaluate_feature_subset_longitudinal_clustering_multitimepoint(  # type: ignore[no-redef]
    *,
    timepoint_dfs: Mapping[str, pd.DataFrame],
    feature_cols: Sequence[str],
    mask: Sequence[Any],
    cfg: ClinicalResponseGAFSConfig,
    subject_id_col: Optional[str] = None,
    reference_timepoint: Optional[str] = None,
    timepoint_feature_cols: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    """
    Evaluate one binary feature mask for multi-timepoint longitudinal clustering.

    This wrapper adds optional longitudinal clinical-score separation scoring
    on top of the existing longitudinal evaluator.

    It supports:
      - longitudinal_unsupervised_clustering
      - longitudinal_label_guided_clustering
      - longitudinal_clinical_score_guided_clustering

    Clinical scores are not used to create clusters. They are used only after
    clustering to score clinical-score separation across timepoints.
    """
    result = _V5_EVALUATE_FEATURE_SUBSET_LONGITUDINAL_MULTITIMEPOINT(
        timepoint_dfs=timepoint_dfs,
        feature_cols=feature_cols,
        mask=mask,
        cfg=cfg,
        subject_id_col=subject_id_col,
        reference_timepoint=reference_timepoint,
        timepoint_feature_cols=timepoint_feature_cols,
    )

    # If the base evaluator failed, return the failure as-is.
    if result.get("fitness", None) == -1e9 or result.get("error"):
        return result

    preset_name = str(cfg.fitness_preset or "").lower()
    components = active_fitness_components(cfg)

    needs_longitudinal_clinical_score = (
        preset_name == "longitudinal_clinical_score_guided_clustering"
        or bool(
            components.get(
                "longitudinal_clinical_score_separation_norm",
                {},
            ).get("weight", 0.0) != 0
        )
    )

    if not needs_longitudinal_clinical_score:
        return result

    clinical_score_cfg = resolve_longitudinal_clinical_score_config(cfg)

    details = dict(result.get("details", {}) or {})
    longitudinal_cfg = dict(details.get("longitudinal_config", {}) or {})

    timepoint_order = list(
        result.get(
            "timepoint_order",
            longitudinal_cfg.get("timepoint_order", list(timepoint_dfs.keys())),
        )
    )

    clinical_score_result = compute_longitudinal_clinical_score_separation(
        membership_df=result["membership_df"],
        timepoint_dfs=timepoint_dfs,
        clinical_score_cols=clinical_score_cfg["timepoint_clinical_score_cols"],
        timepoints=timepoint_order,
        aggregation=clinical_score_cfg.get("aggregation", "mean"),
        eps=cfg.eps,
    )

    metrics_for_fitness = dict(result.get("metrics_for_fitness", {}) or {})

    metrics_for_fitness.update(
        {
            "longitudinal_clinical_score_separation_raw": clinical_score_result[
                "longitudinal_clinical_score_separation_raw"
            ],
            "longitudinal_clinical_score_separation_norm": clinical_score_result[
                "longitudinal_clinical_score_separation_norm"
            ],
        }
    )

    clinical_score_separation_df = clinical_score_result.get(
        "clinical_score_separation_df",
        pd.DataFrame(),
    )

    if isinstance(clinical_score_separation_df, pd.DataFrame) and not clinical_score_separation_df.empty:
        for _, row in clinical_score_separation_df.iterrows():
            tp = str(row["timepoint"])
            metrics_for_fitness[f"clinical_score_separation_{tp}_raw"] = row.get("eta_squared", np.nan)
            metrics_for_fitness[f"clinical_score_separation_{tp}_norm"] = row.get("eta_squared", np.nan)

    fitness, fitness_details = compute_dynamic_fitness(metrics_for_fitness, cfg=cfg)

    result.update(
        {
            "fitness": float(fitness),
            "longitudinal_clinical_score_separation_raw": metrics_for_fitness[
                "longitudinal_clinical_score_separation_raw"
            ],
            "longitudinal_clinical_score_separation_norm": metrics_for_fitness[
                "longitudinal_clinical_score_separation_norm"
            ],
            "clinical_score_separation_df": clinical_score_result[
                "clinical_score_separation_df"
            ],
            "clinical_score_cluster_summary_df": clinical_score_result[
                "clinical_score_cluster_summary_df"
            ],
            "metrics_for_fitness": metrics_for_fitness,
            "fitness_details": fitness_details,
        }
    )

    result.setdefault("details", {})
    result["details"]["clinical_score_scoring"] = {
        "aggregation": clinical_score_result.get("clinical_score_aggregation"),
        "timepoint_clinical_score_cols": clinical_score_cfg.get(
            "timepoint_clinical_score_cols",
            {},
        ),
    }

    return result




# =============================================================================
# Inspect candidate behavioral / demographic columns
# =============================================================================
def build_subtype_analysis_df(
    *,
    result,
    labels,
    behavioral_metadata_df,
    patient_mapping,
    subject_col="subjectkey",
    behavioral_cols=None,
    diagnosis_col="diagnosis",
    target_label=1,
    cluster_timepoints=("baseline", "week6", "month6"),
    nan_strategy="mean",
):
    """
    Build row-aligned dataframes for post-clustering subtype characterization.

    This function is used after GA/clustering.

    It combines:
        - subject IDs
        - diagnosis labels
        - cluster assignments from result["membership_df"]
        - selected behavioral / demographic / IQ / clinical columns

    Returns
    -------
    subtype_analysis_df : pd.DataFrame
        All matched subjects.

    target_subtype_analysis_df : pd.DataFrame
        Subjects with diagnosis_col == target_label.
        For this project, this is usually Diagnosis 1 / ASD only.

    Missing-value handling
    ----------------------
    nan_strategy="mean" by default:
        numeric behavioral columns are mean-imputed.

    nan_strategy="median":
        numeric behavioral columns are median-imputed.

    nan_strategy=None:
        missing values are preserved.

    Categorical columns are not mean- or median-imputed.
    """

    membership_df = result["membership_df"].copy()
    labels = pd.Series(labels, name=diagnosis_col).reset_index(drop=True)

    if len(labels) != len(membership_df):
        raise ValueError(
            f"labels has length {len(labels)}, but membership_df has "
            f"{len(membership_df)} rows. Labels must be row-aligned."
        )

    if nan_strategy not in ["mean", "median", None]:
        raise ValueError("nan_strategy must be one of: 'mean', 'median', or None.")

    if subject_col not in behavioral_metadata_df.columns:
        raise KeyError(f"{subject_col!r} was not found in behavioral_metadata_df.")

    if behavioral_metadata_df[subject_col].duplicated().any():
        dupes = behavioral_metadata_df.loc[
            behavioral_metadata_df[subject_col].duplicated(),
            subject_col,
        ].tolist()

        raise ValueError(
            f"Duplicate subject IDs found in behavioral_metadata_df. "
            f"Examples: {dupes[:10]}"
        )

    # ------------------------------------------------------------
    # Extract patient IDs in the exact same row order as membership_df.
    # ------------------------------------------------------------
    patient_ids = []

    for row_idx in range(len(membership_df)):
        patient_value = patient_mapping[row_idx]

        if isinstance(patient_value, tuple):
            patient_id = patient_value[1]
        else:
            patient_id = patient_value

        patient_ids.append(patient_id)

    behavioral_lookup = behavioral_metadata_df.set_index(subject_col)

    missing_subjects = [
        patient_id for patient_id in patient_ids
        if patient_id not in behavioral_lookup.index
    ]

    if missing_subjects:
        raise ValueError(
            f"{len(missing_subjects)} patients were not found in behavioral_metadata_df. "
            f"Examples: {missing_subjects[:10]}"
        )

    # ------------------------------------------------------------
    # Decide which behavioral columns to pull.
    # ------------------------------------------------------------
    if behavioral_cols is None:
        behavioral_cols = [
            col for col in behavioral_metadata_df.columns
            if col != subject_col
        ]
    else:
        behavioral_cols = list(behavioral_cols)

    missing_cols = [
        col for col in behavioral_cols
        if col not in behavioral_metadata_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"These behavioral_cols were not found in behavioral_metadata_df: "
            f"{missing_cols}"
        )

    # Pull behavioral columns in matched row order.
    behavioral_df = behavioral_lookup.loc[patient_ids, behavioral_cols].copy()


    # ------------------------------------------------------------
    # Mean/median impute numeric behavioral columns only.
    # ------------------------------------------------------------
    imputation_report = {}

    if nan_strategy is not None:
        for col in behavioral_cols:
            if pd.api.types.is_numeric_dtype(behavioral_df[col]):
                n_missing_before = int(behavioral_df[col].isna().sum())

                if n_missing_before > 0:
                    if nan_strategy == "mean":
                        fill_value = behavioral_df[col].mean()
                    elif nan_strategy == "median":
                        fill_value = behavioral_df[col].median()

                    behavioral_df[col] = behavioral_df[col].fillna(fill_value)

                    # If the observed values are all whole-number values,
                    # round the imputed value back to the nearest whole number.
                    observed_values = behavioral_metadata_df[col].dropna()

                    if len(observed_values) > 0 and np.all(
                        np.isclose(observed_values, np.round(observed_values))
                    ):
                        behavioral_df[col] = behavioral_df[col].round(0)

                    imputation_report[col] = {
                        "n_missing_before": n_missing_before,
                        "fill_value": fill_value,
                        "strategy": nan_strategy,
                        "rounded_to_integer_like_scale": bool(
                            len(observed_values) > 0 and np.all(
                                np.isclose(observed_values, np.round(observed_values))
                            )
                        ),
                    }


    subtype_analysis_df = behavioral_df.reset_index()
    subtype_analysis_df = subtype_analysis_df.rename(columns={subject_col: "subjectkey"})

    # Add diagnosis.
    subtype_analysis_df[diagnosis_col] = labels.to_numpy()

    # Add cluster assignments.
    for tp in cluster_timepoints:
        cluster_col = f"cluster_{tp}"

        if cluster_col not in membership_df.columns:
            raise KeyError(f"membership_df is missing {cluster_col!r}.")

        subtype_analysis_df[cluster_col] = membership_df[cluster_col].to_numpy()

    # Optional useful subtype alias.
    if "cluster_baseline" in subtype_analysis_df.columns:
        subtype_analysis_df["baseline_subtype"] = subtype_analysis_df["cluster_baseline"]

    # Store imputation details in attrs so it does not clutter the dataframe.
    subtype_analysis_df.attrs["imputation_report"] = imputation_report
    subtype_analysis_df.attrs["nan_strategy"] = nan_strategy

    # ASD / target-label only dataframe.
    target_subtype_analysis_df = subtype_analysis_df[
        subtype_analysis_df[diagnosis_col] == target_label
    ].copy()

    target_subtype_analysis_df.attrs["imputation_report"] = imputation_report
    target_subtype_analysis_df.attrs["nan_strategy"] = nan_strategy
    target_subtype_analysis_df.attrs["target_label"] = target_label

    return subtype_analysis_df, target_subtype_analysis_df






# =============================================================================
# Longitudinal membership transition visualization
# =============================================================================

def plot_longitudinal_membership_transition(
    result: Dict[str, Any],
    *,
    reference_timepoint: Optional[str] = None,
    comparison_timepoint: Optional[str] = None,
    normalize: Optional[str] = "index",
    annotate: bool = True,
    annotation_format: str = "status_count_percent",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6.8, 5.6),
    font_size: float = 12.0,
    cmap: str = "Blues",
    show_colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    axis_line_color: str = "black",
    axis_line_width: float = 1.0,
    tick_color: str = "black",
    x_tick_rotation: int = 0,
    grid_axis: str = "none",
    grid_color: str = "white",
    grid_alpha: float = 0.6,
    grid_line_width: float = 1.0,
    cell_border: bool = True,
    cell_border_color: str = "white",
    cell_border_width: float = 1.0,
    show_row_separators: bool = True,
    row_separator_color: str = "white",
    row_separator_linewidth: float = 0.8,
    show_column_separators: bool = False,
    column_separator_color: str = "white",
    column_separator_linewidth: float = 0.8,
    hide_spines: bool = True,
    annotation_text_color: str = "auto",
    show: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Any, Any]:
    """
    Plot a longitudinal cluster-membership transition heatmap.

    This plot is intended for longitudinal clustering results produced by
    make_longitudinal_clustering_ga(...). It visualizes how subjects move from
    the reference-timepoint cluster assignment to a comparison-timepoint cluster
    assignment.

    Recommended default
    -------------------
    normalize="index" and annotation_format="status_count_percent".

    With that default, the heatmap color represents row percentages:

        Of subjects who started in baseline cluster X,
        what percent ended in Week 6 cluster Y?

    Each cell annotation shows both the count and row percentage, for example:

        Stayed
        n=123
        89.1%

    Interpretation
    --------------
    Rows:
        Clusters at the reference timepoint, e.g. baseline clusters.

    Columns:
        Clusters at the comparison timepoint, e.g. Week 6 clusters.

    Diagonal cells:
        Subjects who stayed in the same numeric cluster label.

    Off-diagonal cells:
        Subjects who switched cluster labels across timepoints.

    Separator styling
    -----------------
    For heatmaps, regular axis gridlines and full cell borders can look visually
    busy or can make columns appear split. The cleanest style is usually:

        - ax.grid(False)
        - horizontal row separators only
        - no vertical separators
        - hidden spines

    That is why this function now supports show_row_separators and
    show_column_separators as the preferred heatmap-separator controls.

    Parameters
    ----------
    result:
        GA result dictionary returned by LongitudinalFeatureSelectionGA.run().
        Must contain result["membership_df"].

    reference_timepoint:
        Name of the reference timepoint. If omitted, uses
        result["reference_timepoint"] when available.

    comparison_timepoint:
        Name of the comparison timepoint. If omitted, the function tries to use
        the first cross-time ARI pair if available, otherwise the first
        non-reference cluster column in membership_df.

    normalize:
        Controls the heatmap color values:
            None, "count", or "counts": raw counts
            "index", "row", or "rows": row-normalized percentages
            "columns", "column", or "col": column-normalized percentages
            "all": overall percentages

    annotation_format:
        Controls text inside cells:
            "status_count_percent": Stayed/Switched + n=count + percent
            "count_percent": n=count + percent
            "percent": percent only
            "count": count only

        A custom Python format string such as "{:.1f}%" is also accepted.

    grid_axis:
        Legacy axis-grid control: "none", "x", "y", or "both".
        For clean heatmaps, separator controls are preferred over axis grids.

    cell_border:
        Legacy option kept for backward compatibility. It is ignored when
        show_row_separators or show_column_separators are used.

    show_row_separators:
        If True, draw horizontal separator lines between cluster rows.

    show_column_separators:
        If True, draw vertical separator lines between cluster columns.
        This defaults to False because vertical lines can make columns look split.

    hide_spines:
        If True, hide outer axis spines for a cleaner heatmap.

    annotation_text_color:
        "auto" chooses white text on dark cells and black text on light cells.
        Any fixed Matplotlib color string is also accepted.

    Returns
    -------
    percent_or_value_df, count_df, fig, ax
        percent_or_value_df:
            The count or percentage table used for heatmap color values.

        count_df:
            Raw transition counts.

        fig, ax:
            Matplotlib figure and axis.
    """
    import matplotlib.pyplot as plt

    if "membership_df" not in result:
        raise KeyError("result is missing 'membership_df'. This plot requires a longitudinal result.")

    membership_df = result["membership_df"].copy()
    if membership_df.empty:
        raise ValueError("result['membership_df'] is empty; nothing to plot.")

    # ------------------------------------------------------------
    # Resolve reference and comparison timepoints
    # ------------------------------------------------------------
    if reference_timepoint is None:
        reference_timepoint = result.get("reference_timepoint")

    cluster_cols = [col for col in membership_df.columns if str(col).startswith("cluster_")]
    if not cluster_cols:
        raise KeyError("membership_df does not contain any columns starting with 'cluster_'.")

    if reference_timepoint is None:
        reference_timepoint = str(cluster_cols[0]).replace("cluster_", "", 1)

    if comparison_timepoint is None:
        cross_time_ari_df = result.get("cross_time_ari_df")
        if cross_time_ari_df is None:
            cross_time_ari_df = result.get("best_eval", {}).get("cross_time_ari_df")

        if isinstance(cross_time_ari_df, pd.DataFrame) and not cross_time_ari_df.empty:
            match = cross_time_ari_df[
                cross_time_ari_df.get("timepoint_left") == reference_timepoint
            ] if "timepoint_left" in cross_time_ari_df.columns else pd.DataFrame()

            if not match.empty and "timepoint_right" in match.columns:
                comparison_timepoint = str(match.iloc[0]["timepoint_right"])
            elif "timepoint_right" in cross_time_ari_df.columns:
                comparison_timepoint = str(cross_time_ari_df.iloc[0]["timepoint_right"])

    if comparison_timepoint is None:
        non_reference_cols = [
            col for col in cluster_cols
            if str(col) != f"cluster_{reference_timepoint}"
        ]
        if not non_reference_cols:
            raise ValueError("Could not infer comparison_timepoint. Pass comparison_timepoint explicitly.")
        comparison_timepoint = str(non_reference_cols[0]).replace("cluster_", "", 1)

    reference_col = f"cluster_{reference_timepoint}"
    comparison_col = f"cluster_{comparison_timepoint}"

    if reference_col not in membership_df.columns:
        raise KeyError(f"membership_df is missing required column: {reference_col!r}.")
    if comparison_col not in membership_df.columns:
        raise KeyError(f"membership_df is missing required column: {comparison_col!r}.")

    # ------------------------------------------------------------
    # Build count and display tables
    # ------------------------------------------------------------
    count_df = pd.crosstab(
        membership_df[reference_col],
        membership_df[comparison_col],
        rownames=[f"{reference_timepoint}_cluster"],
        colnames=[f"{comparison_timepoint}_cluster"],
        dropna=False,
    ).sort_index(axis=0).sort_index(axis=1)

    normalize_key = None if normalize is None else str(normalize).lower()
    if normalize_key in {None, "none", "count", "counts", "raw"}:
        transition_df = count_df.astype(float)
        value_mode = "count"
        if colorbar_label is None:
            colorbar_label = "Subjects"
    elif normalize_key in {"index", "row", "rows"}:
        transition_df = count_df.div(count_df.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
        value_mode = "row_percent"
        if colorbar_label is None:
            colorbar_label = f"Within-{reference_timepoint}-cluster %"
    elif normalize_key in {"column", "columns", "col"}:
        transition_df = count_df.div(count_df.sum(axis=0).replace(0, np.nan), axis=1) * 100.0
        value_mode = "column_percent"
        if colorbar_label is None:
            colorbar_label = "Column %"
    elif normalize_key == "all":
        total_n = float(count_df.values.sum())
        transition_df = count_df.astype(float) / total_n * 100.0 if total_n > 0 else count_df.astype(float)
        value_mode = "overall_percent"
        if colorbar_label is None:
            colorbar_label = "Overall %"
    else:
        raise ValueError("normalize must be one of None, 'count', 'index', 'row', 'columns', 'column', or 'all'.")

    values_for_plot = transition_df.values

    # ------------------------------------------------------------
    # Plot heatmap
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(values_for_plot, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_xlabel(f"Ended in {comparison_timepoint} cluster", fontsize=font_size)
    ax.set_ylabel(f"Started in {reference_timepoint} cluster", fontsize=font_size)

    if title is None:
        title = f"Cluster membership transition: {reference_timepoint} to {comparison_timepoint}"
    ax.set_title(title, fontsize=font_size + 1)

    ax.set_xticks(np.arange(transition_df.shape[1]))
    ax.set_yticks(np.arange(transition_df.shape[0]))
    ax.set_xticklabels([str(x) for x in transition_df.columns], rotation=x_tick_rotation, fontsize=font_size)
    ax.set_yticklabels([str(y) for y in transition_df.index], fontsize=font_size)

    ax.tick_params(axis="both", colors=tick_color, labelsize=font_size)

    # ------------------------------------------------------------
    # Use a clean heatmap styling approach.
    # ------------------------------------------------------------
    # Always disable regular axis grids first. For heatmaps, generic x/y grids
    # often cut through the middle of cells and create visual clutter.
    ax.grid(False)

    # ------------------------------------------------------------
    # Optional separator lines
    # ------------------------------------------------------------
    # Preferred behavior is horizontal row separators only. Vertical separators
    # are disabled by default because they can make each column appear split.
    n_rows, n_cols = transition_df.shape

    if show_row_separators:
        for y in np.arange(0.5, n_rows, 1.0):
            ax.axhline(
                y,
                color=row_separator_color,
                linewidth=row_separator_linewidth,
            )

    if show_column_separators:
        for x in np.arange(0.5, n_cols, 1.0):
            ax.axvline(
                x,
                color=column_separator_color,
                linewidth=column_separator_linewidth,
            )

    # ------------------------------------------------------------
    # Legacy grid support (optional)
    # ------------------------------------------------------------
    # Keep grid_axis for backward compatibility, but only draw it if requested.
    # This is applied after separators so the user can still force it on.
    grid_axis_key = str(grid_axis or "none").lower()
    if grid_axis_key not in {"none", "x", "y", "both"}:
        raise ValueError("grid_axis must be one of 'none', 'x', 'y', or 'both'.")

    if grid_axis_key != "none":
        axis_arg = "both" if grid_axis_key == "both" else grid_axis_key
        ax.grid(
            which="major",
            axis=axis_arg,
            color=grid_color,
            alpha=grid_alpha,
            linewidth=grid_line_width,
        )

    # ------------------------------------------------------------
    # Backward-compatible cell border behavior
    # ------------------------------------------------------------
    # Only apply this legacy path when the newer separator controls are off.
    if cell_border and (not show_row_separators) and (not show_column_separators):
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(
            which="minor",
            axis="both",
            color=cell_border_color,
            alpha=grid_alpha,
            linewidth=cell_border_width,
        )
        ax.tick_params(which="minor", bottom=False, left=False)

    # ------------------------------------------------------------
    # Spines
    # ------------------------------------------------------------
    if hide_spines:
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(axis_line_color)
            spine.set_linewidth(axis_line_width)

    def _percent_text(row_idx: int, col_idx: int) -> str:
        if value_mode == "count":
            row_total = float(count_df.iloc[row_idx, :].sum())
            pct = (float(count_df.iloc[row_idx, col_idx]) / row_total * 100.0) if row_total > 0 else np.nan
            return "" if pd.isna(pct) else f"{pct:.1f}%"
        value = transition_df.iloc[row_idx, col_idx]
        return "" if pd.isna(value) else f"{float(value):.1f}%"

    if annotate:
        finite_values = values_for_plot[np.isfinite(values_for_plot)]
        threshold = float(np.nanmax(finite_values)) / 2.0 if finite_values.size else 0.0

        for row_idx in range(transition_df.shape[0]):
            for col_idx in range(transition_df.shape[1]):
                plot_value = transition_df.iloc[row_idx, col_idx]
                count_value = int(count_df.iloc[row_idx, col_idx])
                pct_text = _percent_text(row_idx, col_idx)
                status = "Stayed" if str(transition_df.index[row_idx]) == str(transition_df.columns[col_idx]) else "Switched"

                fmt = str(annotation_format or "status_count_percent").lower()
                if fmt == "status_count_percent":
                    text = f"{status}\nn={count_value}\n{pct_text}"
                elif fmt == "count_percent":
                    text = f"n={count_value}\n{pct_text}"
                elif fmt == "percent":
                    text = pct_text
                elif fmt == "count":
                    text = f"n={count_value}"
                else:
                    text = annotation_format.format(float(plot_value)) if not pd.isna(plot_value) else ""

                if str(annotation_text_color).lower() == "auto":
                    text_color = "white" if np.isfinite(plot_value) and float(plot_value) > threshold else "black"
                else:
                    text_color = annotation_text_color
                ax.text(
                    col_idx,
                    row_idx,
                    text,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=font_size,
                )

    if show_colorbar:
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label(colorbar_label, fontsize=font_size)
        cbar.ax.tick_params(labelsize=font_size - 1, colors=tick_color)

    fig.tight_layout()

    if show:
        plt.show()

    return transition_df, count_df, fig, ax



def plot_longitudinal_cluster_projection(
    result: Dict[str, Any],
    cfg: ClinicalResponseGAFSConfig,
    *,
    timepoints: Optional[Sequence[str]] = None,
    method: str = "pca",
    n_components: int = 2,
    standardize: bool = True,
    color_by: str = "cluster",
    cluster_colors: Optional[Dict[Any, str]] = None,
    cluster_palette: Optional[Sequence[str]] = None,

    # Optional external labels, e.g. final diagnosis.
    labels: Optional[Sequence[Any]] = None,
    label_name: str = "Label",
    label_colors: Optional[Dict[Any, str]] = None,
    label_palette: Optional[Sequence[str]] = None,

    # Optional label overlay.
    # None:
    #   no label overlay
    # "outline":
    #   fill color = cluster, point outline color = label
    # "marker":
    #   fill color = cluster, extra marker layer = label
    label_overlay_mode: Optional[str] = None,
    label_overlay_marker: Optional[str] = None,
    label_overlay_colors: Optional[Dict[Any, str]] = None,
    label_overlay_size: Optional[float] = None,
    label_overlay_linewidth: float = 1.4,
    label_overlay_alpha: float = 1.0,

    # Optional diagnosis/label marker-shape mode.
    # Used when label_overlay_mode="shape":
    #   fill color = cluster/subtype
    #   marker shape = external label, e.g. ASD versus TD
    label_marker_map: Optional[Dict[Any, str]] = None,
    label_display_map: Optional[Dict[Any, str]] = None,

    # Optional label-specific point outlines for shape mode.
    # Example for a dark TD outline and a light ASD outline:
    #   label_edgecolor_map={0: "black", 1: "white"}
    #   label_linewidth_map={0: 1.2, 1: 0.4}
    label_edgecolor_map: Optional[Dict[Any, str]] = None,
    label_linewidth_map: Optional[Dict[Any, float]] = None,

    cluster_display_map: Optional[Dict[Any, str]] = None,
    timepoint_label_map: Optional[Dict[str, str]] = None,
    show_group_counts_in_legend: bool = True,

    # NEW: switcher overlay.
    # Switcher = subject changed cluster at least once across displayed timepoints.
    highlight_switchers: bool = False,
    switcher_marker: str = "o",
    switcher_edgecolor: str = "black",
    switcher_linewidth: float = 1.8,
    switcher_size_multiplier: float = 1.9,
    switcher_alpha: float = 1.0,
    switcher_label: str = "Switched at least once",

    point_size: float = 35.0,
    alpha: float = 0.75,
    edgecolor: Optional[str] = "white",
    linewidth: float = 0.4,
    figsize: Optional[Tuple[float, float]] = None,
    font_size: float = 12.0,
    title: Optional[str] = None,
    show_legend: bool = True,
    legend_loc: str = "best",
    grid_axis: str = "both",
    grid_color: str = "gray",
    grid_alpha: float = 0.15,
    grid_line_width: float = 0.8,
    axis_line_color: str = "black",
    axis_line_width: float = 1.0,
    tick_color: str = "black",
    show: bool = True,
) -> Dict[str, Any]:
    """
    Plot selected-feature clusters in a shared 2D PCA projection space.

    Main visual encoding
    --------------------
    Fill color:
        Cluster assignment at each timepoint.

    Optional switcher overlay:
        If highlight_switchers=True, subjects who switch cluster at least once
        across the displayed timepoints are highlighted with a larger hollow marker.

        Stable subject:
            cluster_baseline == cluster_week6 == cluster_month6

        Switcher:
            subject changed cluster at least once across displayed timepoints.

    Optional label overlay:
        If label_overlay_mode="outline", point outline color represents the
        external label, e.g. final diagnosis.

        If label_overlay_mode="marker", an extra marker layer represents the
        external label.

        If label_overlay_mode="shape", the point marker shape represents the
        external label while the fill color continues to represent cluster.
        This supports figures such as:
            ASD Subtype A = colored circle
            ASD Subtype B = colored circle
            TD Subtype A = colored triangle
            TD Subtype B = colored triangle

        In shape mode, each timepoint receives its own legend so the displayed
        group counts can reflect that timepoint. Optional label-specific edge
        colors and line widths can further distinguish diagnosis groups.

    Notes
    -----
    Labels and switcher status are not used to compute PCA or clusters.
    They are only added as plotting metadata.
    """

    from matplotlib.lines import Line2D
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if str(method).lower() != "pca":
        raise ValueError("Only method='pca' is currently supported.")

    selected_cols = list(result.get("selected_cols", []))
    if len(selected_cols) == 0:
        raise ValueError("result['selected_cols'] is empty or missing.")

    preset_cfg = get_active_fitness_preset_config(cfg)
    timepoint_cfg = dict(preset_cfg.get("timepoint_config", {}) or {})
    timepoint_dfs = dict(timepoint_cfg.get("timepoint_dfs", {}) or {})

    if timepoints is None:
        timepoints = (
            result.get("timepoint_order")
            or timepoint_cfg.get("timepoint_order")
            or list(timepoint_dfs.keys())
        )

    timepoints = [str(tp) for tp in timepoints]

    if timepoint_label_map is None:
        timepoint_label_map = {
            tp: tp
            for tp in timepoints
        }
    else:
        timepoint_label_map = dict(timepoint_label_map)

    if len(timepoints) == 0:
        raise ValueError("No timepoints were provided or found.")

    missing_timepoints = [tp for tp in timepoints if tp not in timepoint_dfs]
    if missing_timepoints:
        raise KeyError(f"timepoint_dfs is missing timepoints: {missing_timepoints}")

    membership_df = result.get("membership_df")
    if membership_df is None or not isinstance(membership_df, pd.DataFrame):
        raise ValueError("result['membership_df'] must be a pandas DataFrame.")

    for tp in timepoints:
        cluster_col = f"cluster_{tp}"
        if cluster_col not in membership_df.columns:
            raise KeyError(f"membership_df is missing cluster column {cluster_col!r}.")

    requested_cluster_cols = [f"cluster_{tp}" for tp in timepoints]

    # ------------------------------------------------------------------
    # Compute subject-level transition status.
    # ------------------------------------------------------------------
    # Stable = same cluster across all displayed timepoints.
    # Switched = changed cluster at least once.
    # ------------------------------------------------------------------

    cluster_matrix = membership_df[requested_cluster_cols].to_numpy()
    first_cluster = cluster_matrix[:, [0]]
    stable_trajectory = np.all(cluster_matrix == first_cluster, axis=1)
    switched_trajectory = ~stable_trajectory

    transition_status_df = pd.DataFrame({
        "row_index": np.arange(len(membership_df)),
        "stable_trajectory": stable_trajectory,
        "switched_trajectory": switched_trajectory,
        "transition_status": np.where(
            switched_trajectory,
            "Switched at least once",
            "Stable across all timepoints",
        ),
    })

    switcher_summary = {
        "n_total": int(len(transition_status_df)),
        "n_stable": int(transition_status_df["stable_trajectory"].sum()),
        "n_switched": int(transition_status_df["switched_trajectory"].sum()),
    }

    if switcher_summary["n_total"] > 0:
        switcher_summary["percent_stable"] = (
            100 * switcher_summary["n_stable"] / switcher_summary["n_total"]
        )
        switcher_summary["percent_switched"] = (
            100 * switcher_summary["n_switched"] / switcher_summary["n_total"]
        )
    else:
        switcher_summary["percent_stable"] = np.nan
        switcher_summary["percent_switched"] = np.nan

    # ------------------------------------------------------------------
    # Build stacked selected-feature matrix across timepoints.
    # ------------------------------------------------------------------

    stacked_parts = []
    metadata_parts = []

    for tp in timepoints:
        df_tp = timepoint_dfs[tp]

        if len(df_tp) != len(membership_df):
            raise ValueError(
                f"Timepoint {tp!r} has {len(df_tp)} rows, but membership_df has "
                f"{len(membership_df)} rows. They must be row-aligned."
            )

        missing_features = [col for col in selected_cols if col not in df_tp.columns]
        if missing_features:
            raise KeyError(
                f"Timepoint {tp!r} is missing selected feature columns: {missing_features}"
            )

        X_tp = df_tp[selected_cols].apply(pd.to_numeric, errors="coerce")
        cluster_col = f"cluster_{tp}"

        meta_tp = pd.DataFrame({
            "timepoint": tp,
            "row_index": np.arange(len(df_tp)),
            "cluster": membership_df[cluster_col].to_numpy(),
            "stable_trajectory": transition_status_df["stable_trajectory"].to_numpy(),
            "switched_trajectory": transition_status_df["switched_trajectory"].to_numpy(),
            "transition_status": transition_status_df["transition_status"].to_numpy(),
        })

        if labels is not None:
            if len(labels) != len(df_tp):
                raise ValueError(
                    f"labels has length {len(labels)}, but timepoint {tp!r} has "
                    f"{len(df_tp)} rows. Labels must be row-aligned."
                )
            meta_tp[label_name] = np.asarray(labels)

        stacked_parts.append(X_tp)
        metadata_parts.append(meta_tp)

    X_stacked = pd.concat(stacked_parts, axis=0, ignore_index=True)
    projection_df = pd.concat(metadata_parts, axis=0, ignore_index=True)

    valid_mask = ~X_stacked.isna().any(axis=1)
    X_valid = X_stacked.loc[valid_mask].to_numpy(dtype=float)
    projection_df = projection_df.loc[valid_mask].reset_index(drop=True)

    if X_valid.shape[0] < 3:
        raise ValueError("Too few complete rows available for PCA projection.")

    if X_valid.shape[1] < 2:
        raise ValueError("At least two selected features are required for a 2D PCA plot.")

    if standardize:
        scaler = StandardScaler()
        X_for_pca = scaler.fit_transform(X_valid)
    else:
        scaler = None
        X_for_pca = X_valid

    pca = PCA(n_components=n_components, random_state=getattr(cfg, "random_seed", None))
    X_proj = pca.fit_transform(X_for_pca)

    projection_df["PC1"] = X_proj[:, 0]
    projection_df["PC2"] = X_proj[:, 1]

    explained_variance_ratio = pca.explained_variance_ratio_

    # ------------------------------------------------------------------
    # Resolve cluster colors.
    # ------------------------------------------------------------------

    all_clusters = sorted(projection_df["cluster"].dropna().unique())

    default_cluster_palette = [
        "#1587F8",
        "#FFAE17",
        "#049B4F",
        "#C04AE2",
        "#F14949",
        "#7A5CFF",
        "#00A6A6",
    ]

    if cluster_palette is None:
        cluster_palette = default_cluster_palette

    cluster_to_color = dict(cluster_colors or {})
    for idx, cluster in enumerate(all_clusters):
        if cluster not in cluster_to_color:
            cluster_to_color[cluster] = cluster_palette[idx % len(cluster_palette)]

    cluster_to_display = {
        cluster: str(cluster)
        for cluster in all_clusters
    }

    if cluster_display_map is not None:
        cluster_to_display.update(dict(cluster_display_map))

    # ------------------------------------------------------------------
    # Resolve label colors, marker shapes, and display names.
    # ------------------------------------------------------------------

    label_to_color = {}
    label_to_marker = {}
    label_to_display = {}
    label_to_edgecolor = {}
    label_to_linewidth = {}
    all_label_values = []

    if labels is not None:
        observed_label_values = list(
            pd.Series(projection_df[label_name])
            .dropna()
            .unique()
        )

        if label_display_map is not None:
            requested_label_order = list(label_display_map.keys())
            all_label_values = [
                label_value
                for label_value in requested_label_order
                if label_value in observed_label_values
            ]
            all_label_values += [
                label_value
                for label_value in observed_label_values
                if label_value not in all_label_values
            ]
        else:
            all_label_values = sorted(observed_label_values)

        default_label_palette = [
            "black",
            "red",
            "dimgray",
            "purple",
            "brown",
        ]

        default_marker_palette = [
            "o",
            "^",
            "s",
            "D",
            "P",
            "X",
        ]

        if label_palette is None:
            label_palette = default_label_palette

        label_to_color = dict(label_colors or {})
        label_to_color.update(dict(label_overlay_colors or {}))

        label_to_marker = dict(label_marker_map or {})
        label_to_edgecolor = dict(label_edgecolor_map or {})
        label_to_linewidth = dict(label_linewidth_map or {})

        label_to_display = {
            label_value: str(label_value)
            for label_value in all_label_values
        }

        if label_display_map is not None:
            label_to_display.update(dict(label_display_map))

        for idx, label_value in enumerate(all_label_values):
            if label_value not in label_to_color:
                label_to_color[label_value] = label_palette[idx % len(label_palette)]

            if label_value not in label_to_marker:
                label_to_marker[label_value] = default_marker_palette[
                    idx % len(default_marker_palette)
                ]

            if label_value not in label_to_edgecolor:
                label_to_edgecolor[label_value] = (
                    edgecolor
                    if edgecolor is not None
                    else "none"
                )

            if label_value not in label_to_linewidth:
                label_to_linewidth[label_value] = (
                    float(linewidth)
                    if edgecolor is not None
                    else 0.0
                )

    overlay_mode = None if label_overlay_mode is None else str(label_overlay_mode).lower()

    if overlay_mode not in {None, "outline", "marker", "shape"}:
        raise ValueError(
            "label_overlay_mode must be None, 'outline', 'marker', or 'shape'."
        )

    if overlay_mode == "shape" and labels is None:
        raise ValueError(
            "labels must be provided when label_overlay_mode='shape'."
        )

    # ------------------------------------------------------------------
    # Plot.
    # ------------------------------------------------------------------

    n_panels = len(timepoints)

    if figsize is None:
        figsize = (5.5 * n_panels, 5.0)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )

    if n_panels == 1:
        axes = [axes]

    for ax, tp in zip(axes, timepoints):
        df_tp = projection_df[projection_df["timepoint"] == tp].copy()

        # --------------------------------------------------------------
        # Jarrod-style shape mode:
        #   color = cluster/subtype
        #   marker shape = diagnosis/external label
        # --------------------------------------------------------------
        if overlay_mode == "shape":
            panel_handles = []

            for label_value in all_label_values:
                for cluster in all_clusters:
                    df_group = df_tp[
                        (df_tp[label_name] == label_value)
                        & (df_tp["cluster"] == cluster)
                    ].copy()

                    if df_group.empty:
                        continue

                    ax.scatter(
                        df_group["PC1"],
                        df_group["PC2"],
                        s=point_size,
                        marker=label_to_marker[label_value],
                        alpha=alpha,
                        color=cluster_to_color.get(cluster),
                        edgecolors=label_to_edgecolor[label_value],
                        linewidths=label_to_linewidth[label_value],
                        zorder=2,
                    )

                    group_label = (
                        f"{label_to_display[label_value]} "
                        f"{cluster_to_display[cluster]}"
                    )

                    if show_group_counts_in_legend:
                        group_label += f" (n={len(df_group)})"

                    panel_handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker=label_to_marker[label_value],
                            linestyle="",
                            markersize=7,
                            markerfacecolor=cluster_to_color.get(cluster),
                            markeredgecolor=(
                                label_to_edgecolor[label_value]
                            ),
                            markeredgewidth=(
                                label_to_linewidth[label_value]
                            ),
                            label=group_label,
                        )
                    )

        else:
            for cluster in all_clusters:
                df_c = df_tp[df_tp["cluster"] == cluster]
                if df_c.empty:
                    continue

                # Fill color = cluster.
                # Optional outline color = label.
                if labels is not None and overlay_mode == "outline":
                    point_edgecolors = (
                        df_c[label_name]
                        .map(label_to_color)
                        .fillna(
                            edgecolor
                            if edgecolor is not None
                            else "black"
                        )
                        .tolist()
                    )
                    point_linewidths = label_overlay_linewidth
                else:
                    point_edgecolors = (
                        edgecolor
                        if edgecolor is not None
                        else "none"
                    )
                    point_linewidths = (
                        linewidth
                        if edgecolor is not None
                        else 0.0
                    )

                ax.scatter(
                    df_c["PC1"],
                    df_c["PC2"],
                    s=point_size,
                    alpha=alpha,
                    color=cluster_to_color.get(cluster),
                    edgecolors=point_edgecolors,
                    linewidths=point_linewidths,
                    label=str(
                        cluster_to_display.get(
                            cluster,
                            cluster,
                        )
                    ),
                    zorder=2,
                )

        # Overlay switchers as larger hollow markers.
        if highlight_switchers:
            df_switched = df_tp[df_tp["switched_trajectory"]].copy()

            if not df_switched.empty:
                ax.scatter(
                    df_switched["PC1"],
                    df_switched["PC2"],
                    s=point_size * switcher_size_multiplier,
                    marker=switcher_marker,
                    facecolors="none",
                    edgecolors=switcher_edgecolor,
                    linewidths=switcher_linewidth,
                    alpha=switcher_alpha,
                    zorder=6,
                    label=switcher_label,
                )

                if overlay_mode == "shape":
                    panel_handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker=switcher_marker,
                            linestyle="",
                            markersize=8,
                            markerfacecolor="none",
                            markeredgecolor=switcher_edgecolor,
                            markeredgewidth=switcher_linewidth,
                            label=switcher_label,
                        )
                    )

        # Optional old-style marker overlay.
        if (
            labels is not None
            and overlay_mode == "marker"
            and label_overlay_marker is not None
        ):
            overlay_size = (
                label_overlay_size
                if label_overlay_size is not None
                else point_size
            )

            for label_value, df_l in df_tp.groupby(
                label_name,
                dropna=True,
            ):
                ax.scatter(
                    df_l["PC1"],
                    df_l["PC2"],
                    s=overlay_size,
                    marker=label_overlay_marker,
                    color=label_to_color.get(label_value, "black"),
                    linewidths=label_overlay_linewidth,
                    alpha=label_overlay_alpha,
                    label=f"{label_name} {label_value}",
                    zorder=5,
                )

        ax.set_title(
            str(timepoint_label_map.get(tp, tp)),
            fontsize=font_size + 1,
        )

        ax.set_xlabel(
            f"PC1 ({explained_variance_ratio[0] * 100:.1f}%)",
            fontsize=font_size,
        )

        ax.set_ylabel(
            f"PC2 ({explained_variance_ratio[1] * 100:.1f}%)",
            fontsize=font_size,
        )

        if str(grid_axis).lower() != "none":
            ax.grid(
                True,
                axis=grid_axis,
                color=grid_color,
                alpha=grid_alpha,
                linewidth=grid_line_width,
            )
        else:
            ax.grid(False)

        for spine in ax.spines.values():
            spine.set_color(axis_line_color)
            spine.set_linewidth(axis_line_width)

        ax.tick_params(
            axis="both",
            colors=tick_color,
            labelsize=font_size - 1,
        )

        # Shape mode uses one legend per panel because group counts can differ
        # across Baseline, Week 6, and Month 6.
        if show_legend and overlay_mode == "shape":
            ax.legend(
                handles=panel_handles,
                loc=legend_loc,
                fontsize=font_size - 1,
                frameon=True,
            )

    if title is not None:
        fig.suptitle(title, fontsize=font_size + 2)

    # ------------------------------------------------------------------
    # Legends for the original overlay modes.
    # Shape mode already created one count-aware legend per panel.
    # ------------------------------------------------------------------

    if show_legend and overlay_mode != "shape":
        cluster_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=7,
                markerfacecolor=cluster_to_color[cluster],
                markeredgecolor="none",
                label=str(
                    cluster_to_display.get(
                        cluster,
                        cluster,
                    )
                ),
            )
            for cluster in all_clusters
        ]

        label_handles = []

        if labels is not None and overlay_mode == "outline":
            for label_value in all_label_values:
                label_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="",
                        markersize=7,
                        markerfacecolor="white",
                        markeredgecolor=label_to_color[label_value],
                        markeredgewidth=label_overlay_linewidth,
                        label=(
                            f"{label_name} "
                            f"{label_to_display.get(label_value, label_value)}"
                        ),
                    )
                )

        elif (
            labels is not None
            and overlay_mode == "marker"
            and label_overlay_marker is not None
        ):
            for label_value in all_label_values:
                label_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=label_overlay_marker,
                        linestyle="",
                        markersize=7,
                        color=label_to_color[label_value],
                        label=(
                            f"{label_name} "
                            f"{label_to_display.get(label_value, label_value)}"
                        ),
                    )
                )

        switcher_handles = []

        if highlight_switchers:
            switcher_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=switcher_marker,
                    linestyle="",
                    markersize=8,
                    markerfacecolor="none",
                    markeredgecolor=switcher_edgecolor,
                    markeredgewidth=switcher_linewidth,
                    label=switcher_label,
                )
            )

        handles = cluster_handles + label_handles + switcher_handles

        axes[-1].legend(
            handles=handles,
            loc=legend_loc,
            fontsize=font_size - 1,
            frameon=True,
        )

    fig.tight_layout()

    if title is not None:
        fig.subplots_adjust(top=0.86)

    if show:
        plt.show()

    return {
        "projection_df": projection_df,
        "transition_status_df": transition_status_df,
        "switcher_summary": switcher_summary,
        "explained_variance_ratio": explained_variance_ratio,
        "fig": fig,
        "axes": axes,
        "pca": pca,
        "scaler": scaler,
        "cluster_colors": cluster_to_color,
        "cluster_display_map": cluster_to_display,
        "label_colors": label_to_color,
        "label_marker_map": label_to_marker,
        "label_display_map": label_to_display,
        "label_edgecolor_map": label_to_edgecolor,
        "label_linewidth_map": label_to_linewidth,
    }



def summarize_longitudinal_cluster_sizes(
    result: Dict[str, Any],
    *,
    timepoints: Optional[Sequence[str]] = None,
    normalize: bool = False,
) -> pd.DataFrame:
    """
    Summarize cluster sizes at each longitudinal timepoint.

    This helper is useful when testing k=3 or more clusters because small
    clusters become more likely as k increases.

    Parameters
    ----------
    result:
        Longitudinal GA result dictionary containing membership_df.

    timepoints:
        Optional subset/order of timepoints. If omitted, uses
        result["timepoint_order"] when available, otherwise all cluster_* columns
        in membership_df.

    normalize:
        If True, return percentages within each timepoint. If False, return
        subject counts.

    Returns
    -------
    size_df:
        DataFrame with columns: timepoint, cluster, n, percent.
    """
    if "membership_df" not in result:
        raise KeyError("result is missing 'membership_df'.")

    membership_df = result["membership_df"]
    if not isinstance(membership_df, pd.DataFrame) or membership_df.empty:
        raise ValueError("result['membership_df'] must be a non-empty DataFrame.")

    cluster_cols = [col for col in membership_df.columns if str(col).startswith("cluster_")]
    if not cluster_cols:
        raise KeyError("membership_df does not contain any cluster_* columns.")

    if timepoints is None:
        timepoints = result.get("timepoint_order")
        if timepoints is None:
            timepoints = [str(col).replace("cluster_", "", 1) for col in cluster_cols]
    timepoints = [str(tp) for tp in timepoints]

    rows = []
    for tp in timepoints:
        col = f"cluster_{tp}"
        if col not in membership_df.columns:
            raise KeyError(f"membership_df is missing cluster column {col!r}.")

        counts = membership_df[col].value_counts(dropna=False).sort_index()
        total = float(counts.sum())
        for cluster, n in counts.items():
            pct = (float(n) / total * 100.0) if total > 0 else np.nan
            rows.append({
                "timepoint": tp,
                "cluster": cluster,
                "n": int(n),
                "percent": pct,
                "value": pct if normalize else int(n),
            })

    return pd.DataFrame(rows)


def plot_longitudinal_cluster_sizes(
    result: Dict[str, Any],
    *,
    timepoints: Optional[Sequence[str]] = None,
    normalize: bool = False,
    cluster_colors: Optional[Dict[Any, str]] = None,
    cluster_palette: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    font_size: float = 12.0,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_legend: bool = True,
    legend_loc: str = "best",
    bar_width: float = 0.8,
    axis_line_color: str = "black",
    axis_line_width: float = 1.0,
    tick_color: str = "black",
    grid_axis: str = "y",
    grid_color: str = "gray",
    grid_alpha: float = 0.18,
    grid_line_width: float = 0.8,
    annotate_bars: bool = True,
    annotate_decimals: int = 1,
    show: bool = True,
) -> Tuple[pd.DataFrame, Any, Any]:
    """
    Plot cluster sizes at each longitudinal timepoint.

    This is especially useful when experimenting with k=3 or more clusters.
    It helps identify whether a higher-k solution produces very small clusters.

    Parameters
    ----------
    result:
        Longitudinal GA result dictionary containing membership_df.

    timepoints:
        Optional subset/order of timepoints. If omitted, uses result timepoint order.

    normalize:
        If False, bar heights are counts. If True, bar heights are percentages
        within each timepoint.

    cluster_colors:
        Optional mapping from cluster label to color.

    cluster_palette:
        Optional ordered list of colors used when cluster_colors is missing or
        incomplete.

    Returns
    -------
    size_df, fig, ax
        size_df contains count and percent for each cluster at each timepoint.
    """
    import matplotlib.pyplot as plt

    size_df = summarize_longitudinal_cluster_sizes(
        result,
        timepoints=timepoints,
        normalize=normalize,
    )

    if size_df.empty:
        raise ValueError("No cluster-size data available to plot.")

    timepoints = list(size_df["timepoint"].drop_duplicates())
    clusters = sorted(size_df["cluster"].drop_duplicates())

    if figsize is None:
        figsize = (max(6.5, 1.5 * len(timepoints) + 2.5), 5.0)

    fig, ax = plt.subplots(figsize=figsize)

    cluster_colors = dict(cluster_colors or {})
    if cluster_palette is not None:
        color_cycle = list(cluster_palette)
    else:
        color_cycle = list(plt.rcParams["axes.prop_cycle"].by_key().get("color", []))
    if len(color_cycle) == 0:
        cmap = plt.get_cmap("tab20")
        color_cycle = [cmap(i % cmap.N) for i in range(max(len(clusters), 1))]
    if len(color_cycle) < len(clusters):
        cmap = plt.get_cmap("tab20")
        color_cycle = color_cycle + [cmap(i % cmap.N) for i in range(len(clusters) - len(color_cycle))]

    cluster_to_color = {}
    for i, cluster in enumerate(clusters):
        if cluster in cluster_colors:
            cluster_to_color[cluster] = cluster_colors[cluster]
        elif str(cluster) in cluster_colors:
            cluster_to_color[cluster] = cluster_colors[str(cluster)]
        else:
            cluster_to_color[cluster] = color_cycle[i % len(color_cycle)]

    x = np.arange(len(timepoints))
    n_clusters = len(clusters)
    width = bar_width / max(n_clusters, 1)

    for i, cluster in enumerate(clusters):
        vals = []
        labels = []
        for tp in timepoints:
            row = size_df[(size_df["timepoint"] == tp) & (size_df["cluster"] == cluster)]
            if row.empty:
                vals.append(0.0)
                labels.append("")
            else:
                vals.append(float(row.iloc[0]["value"]))
                if normalize:
                    labels.append(f"{float(row.iloc[0]['percent']):.{annotate_decimals}f}%")
                else:
                    labels.append(str(int(row.iloc[0]["n"])))

        offsets = x - bar_width / 2.0 + width / 2.0 + i * width
        bars = ax.bar(
            offsets,
            vals,
            width=width,
            label=f"cluster {cluster}",
            color=cluster_to_color.get(cluster),
        )

        if annotate_bars:
            for bar, label in zip(bars, labels):
                if label == "":
                    continue
                height = bar.get_height()
                ax.annotate(
                    label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=font_size - 1,
                    color=tick_color,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(timepoints, fontsize=font_size)

    if ylabel is None:
        ylabel = "Percent of subjects" if normalize else "Number of subjects"
    ax.set_ylabel(ylabel, fontsize=font_size)

    if title is None:
        title = "Cluster sizes by timepoint"
    ax.set_title(title, fontsize=font_size + 1)

    ax.tick_params(axis="both", colors=tick_color, labelsize=font_size)
    for spine in ax.spines.values():
        spine.set_color(axis_line_color)
        spine.set_linewidth(axis_line_width)

    grid_axis_key = str(grid_axis or "none").lower()
    if grid_axis_key not in {"none", "x", "y", "both"}:
        raise ValueError("grid_axis must be one of 'none', 'x', 'y', or 'both'.")
    if grid_axis_key != "none":
        axis_arg = "both" if grid_axis_key == "both" else grid_axis_key
        ax.grid(
            True,
            axis=axis_arg,
            color=grid_color,
            alpha=grid_alpha,
            linewidth=grid_line_width,
        )
    else:
        ax.grid(False)

    if show_legend:
        ax.legend(loc=legend_loc, fontsize=font_size - 1, frameon=True)

    fig.tight_layout()

    if show:
        plt.show()

    return size_df, fig, ax



def plot_longitudinal_cluster_label_composition(
    *,
    result,
    labels,
    timepoints=("baseline", "week6", "month6"),
    label_name="Diagnosis",
    positive_label=1,
    positive_label_display="ASD",
    timepoint_label_map=None,
    label_colors=None,
    label_value_display_map=None,
    cluster_colors=None,
    cluster_label_map=None,
    cluster_group_name="cluster",

    # Subtype/cluster label above each bar
    show_cluster_label_above_bar=True,
    cluster_label_font_size=10,
    cluster_label_font_weight="bold",
    cluster_label_offset_points=54,

    # Statistics above each bar
    stats_offset_points=5,
    annotation_font_size=9,

    # Legend controls
    show_cluster_outline_legend=True,

    figsize=(16, 5),
    title="Cluster diagnosis composition plot",
    show_positive_label_within_cluster=True,
    show_positive_label_cohort_share=True,
    ylim_multiplier=1.35,
    show=True,
):
    """
    Plot external-label composition within each cluster/subtype.

    Visual encoding
    ---------------
    Bar height:
        Total number of subjects in the cluster/subtype.

    Bar fill:
        External label composition, such as responder vs non-responder.

    Bar outline:
        Cluster/subtype identity.

    Colored text above bar:
        Displayed cluster/subtype name using the same color as the outline.

    Black text above bar:
        Total n, positive-label percentage within the subtype, and percentage
        of the positive-label cohort assigned to the subtype.

    Parameters
    ----------
    label_value_display_map:
        Optional readable labels for external label values.

        Example:
            {
                0: "Non-responder",
                1: "Responder",
            }

    cluster_label_map:
        Optional readable names for cluster values.

        Example:
            {
                0: "Subtype 1",
                1: "Subtype 2",
            }

    cluster_group_name:
        Wording used in annotations and legend titles, such as
        "cluster" or "subtype".

    show_cluster_outline_legend:
        If False, the outline legend is hidden. This is useful when the
        colored subtype name is displayed directly above each bar.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # ------------------------------------------------------------------
    # Validate and resolve inputs
    # ------------------------------------------------------------------

    if "membership_df" not in result:
        raise KeyError("result is missing 'membership_df'.")

    membership_df = result["membership_df"].copy()

    labels = pd.Series(labels, name="label").reset_index(drop=True)

    if len(labels) != len(membership_df):
        raise ValueError(
            f"labels has length {len(labels)}, but membership_df has "
            f"{len(membership_df)} rows. Labels must be row-aligned."
        )

    if timepoint_label_map is None:
        timepoint_label_map = {
            tp: tp
            for tp in timepoints
        }

    if label_colors is None:
        label_colors = {
            0: "#C9C9C9",
            1: "#E85C5C",
        }

    if label_value_display_map is None:
        label_value_display_map = {}

    if cluster_colors is None:
        cluster_colors = {
            0: "#1587F8",
            1: "#FFAE17",
            2: "#049B4F",
            3: "#C04AE2",
        }

    if cluster_label_map is None:
        cluster_label_map = {}

    cluster_group_name = str(cluster_group_name)

    def _display_label_value(label_value):
        return str(
            label_value_display_map.get(
                label_value,
                label_value_display_map.get(
                    str(label_value),
                    label_value,
                ),
            )
        )

    def _display_cluster_label(cluster):
        default_label = (
            f"{cluster_group_name.title()} {cluster}"
        )

        return str(
            cluster_label_map.get(
                cluster,
                cluster_label_map.get(
                    str(cluster),
                    default_label,
                ),
            )
        )

    # ------------------------------------------------------------------
    # Build composition dataframe
    # ------------------------------------------------------------------

    rows = []

    for tp in timepoints:
        cluster_col = f"cluster_{tp}"

        if cluster_col not in membership_df.columns:
            raise KeyError(
                f"membership_df is missing {cluster_col!r}."
            )

        tmp = pd.DataFrame({
            "timepoint": tp,
            "timepoint_label": timepoint_label_map.get(tp, tp),
            "cluster": membership_df[cluster_col].to_numpy(),
            "label": labels.to_numpy(),
        })

        tmp = tmp.dropna(
            subset=["cluster", "label"]
        )

        total_positive_n = int(
            (tmp["label"] == positive_label).sum()
        )

        for cluster in sorted(tmp["cluster"].unique()):
            d_cluster = tmp[
                tmp["cluster"] == cluster
            ].copy()

            total_cluster_n = len(d_cluster)

            for label_value in sorted(tmp["label"].unique()):
                count = int(
                    (d_cluster["label"] == label_value).sum()
                )

                rows.append({
                    "timepoint": tp,
                    "timepoint_label": timepoint_label_map.get(tp, tp),
                    "cluster": cluster,
                    "cluster_display": _display_cluster_label(cluster),
                    "label": label_value,
                    "label_display": _display_label_value(label_value),
                    "count": count,
                    "cluster_total_n": total_cluster_n,
                    "timepoint_positive_total_n": total_positive_n,
                })

    comp_df = pd.DataFrame(rows)

    if comp_df.empty:
        raise ValueError("No composition rows were created.")

    timepoints_order = list(timepoints)

    clusters = sorted(
        comp_df["cluster"]
        .dropna()
        .unique()
    )

    label_values = sorted(
        comp_df["label"]
        .dropna()
        .unique()
    )

    # ------------------------------------------------------------------
    # Plot setup
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=figsize
    )

    x = np.arange(
        len(timepoints_order)
    )

    n_clusters = len(clusters)

    bar_width = (
        0.22
        if n_clusters > 2
        else 0.24
    )

    cluster_offsets = (
        np.arange(n_clusters)
        - (n_clusters - 1) / 2
    ) * (bar_width * 1.15)

    max_bar_height = int(
        comp_df.groupby(
            ["timepoint", "cluster"]
        )["cluster_total_n"]
        .first()
        .max()
    )

    # ------------------------------------------------------------------
    # Draw bars and annotations
    # ------------------------------------------------------------------

    for cluster_idx, cluster in enumerate(clusters):
        offset = cluster_offsets[cluster_idx]

        for tp_idx, tp in enumerate(timepoints_order):
            d = comp_df[
                (comp_df["timepoint"] == tp)
                & (comp_df["cluster"] == cluster)
            ].copy()

            if d.empty:
                continue

            cluster_total_n = int(
                d["cluster_total_n"].iloc[0]
            )

            total_positive_n = int(
                d["timepoint_positive_total_n"].iloc[0]
            )

            positive_count = int(
                d.loc[
                    d["label"] == positive_label,
                    "count",
                ].sum()
            )

            if cluster_total_n > 0:
                positive_within_cluster_pct = (
                    100
                    * positive_count
                    / cluster_total_n
                )
            else:
                positive_within_cluster_pct = np.nan

            if total_positive_n > 0:
                positive_cohort_pct = (
                    100
                    * positive_count
                    / total_positive_n
                )
            else:
                positive_cohort_pct = np.nan

            bar_x = x[tp_idx] + offset
            bottom = 0

            for label_value in label_values:
                count_series = d.loc[
                    d["label"] == label_value,
                    "count",
                ]

                count = (
                    0
                    if count_series.empty
                    else int(count_series.iloc[0])
                )

                ax.bar(
                    bar_x,
                    count,
                    width=bar_width,
                    bottom=bottom,
                    color=label_colors.get(
                        label_value,
                        "gray",
                    ),
                    edgecolor=cluster_colors.get(
                        cluster,
                        "black",
                    ),
                    linewidth=2.2,
                )

                if count > 0:
                    ax.text(
                        bar_x,
                        bottom + count / 2,
                        f"{count}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color="black",
                    )

                bottom += count

            # ----------------------------------------------------------
            # Black statistical annotation
            # ----------------------------------------------------------

            stats_lines = [
                f"n={cluster_total_n}"
            ]

            if show_positive_label_within_cluster:
                stats_lines.append(
                    f"{positive_within_cluster_pct:.0f}% "
                    f"{positive_label_display} within "
                    f"{cluster_group_name}"
                )

            if show_positive_label_cohort_share:
                stats_lines.append(
                    f"{positive_cohort_pct:.0f}% of "
                    f"{positive_label_display} cohort"
                )

            ax.annotate(
                "\n".join(stats_lines),
                xy=(bar_x, cluster_total_n),
                xytext=(0, stats_offset_points),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=annotation_font_size,
                fontweight="bold",
                color="black",
                annotation_clip=False,
            )

            # ----------------------------------------------------------
            # Colored subtype/cluster name above statistics
            # ----------------------------------------------------------

            if show_cluster_label_above_bar:
                ax.annotate(
                    _display_cluster_label(cluster),
                    xy=(bar_x, cluster_total_n),
                    xytext=(0, cluster_label_offset_points),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=cluster_label_font_size,
                    fontweight=cluster_label_font_weight,
                    color=cluster_colors.get(
                        cluster,
                        "black",
                    ),
                    annotation_clip=False,
                )

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            timepoint_label_map.get(tp, tp)
            for tp in timepoints_order
        ],
        fontsize=11,
    )

    ax.set_ylabel(
        "Number of subjects",
        fontsize=12,
    )

    ax.set_title(
        title,
        fontsize=14,
        pad=16,
    )

    ax.grid(
        axis="y",
        color="gray",
        alpha=0.18,
        linewidth=0.8,
    )

    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    # ------------------------------------------------------------------
    # Fill legend
    # ------------------------------------------------------------------

    label_handles = [
        Patch(
            facecolor=label_colors.get(
                label_value,
                "gray",
            ),
            edgecolor="black",
            label=_display_label_value(
                label_value
            ),
        )
        for label_value in label_values
    ]

    legend_1 = ax.legend(
        handles=label_handles,
        title=f"Fill: {label_name}",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
    )

    ax.add_artist(legend_1)

    # ------------------------------------------------------------------
    # Optional outline legend
    # ------------------------------------------------------------------

    if show_cluster_outline_legend:
        cluster_handles = [
            Patch(
                facecolor="white",
                edgecolor=cluster_colors.get(
                    cluster,
                    "black",
                ),
                linewidth=2.2,
                label=_display_cluster_label(
                    cluster
                ),
            )
            for cluster in clusters
        ]

        ax.legend(
            handles=cluster_handles,
            title=f"Outline: {cluster_group_name}",
            loc="upper left",
            bbox_to_anchor=(1.02, 0.62),
            frameon=True,
        )

    # Extra vertical room for subtype names and statistics.
    ax.set_ylim(
        0,
        max(
            max_bar_height * ylim_multiplier,
            max_bar_height + 10,
        ),
    )

    plt.tight_layout()

    if show:
        plt.show()

    return comp_df, fig, ax




def plot_longitudinal_cluster_alluvial(
    result,
    *,
    labels=None,
    label_filter=None,
    label_filter_name=None,
    timepoints=None,
    reference_timepoint="baseline",
    cluster_colors=None,
    cluster_palette=None,
    cluster_label_map=None,
    timepoint_label_map=None,
    color_by="reference_cluster",

    # ------------------------------------------------------------------
    # Node / block styling
    # ------------------------------------------------------------------
    node_width=0.30,
    node_gap=12.0,
    node_color_by="fixed",          # "cluster" or "fixed"
    node_facecolor="white",         # used when node_color_by="fixed"
    node_alpha=1.0,                 # used for nodes
    node_edgecolor="black",
    node_linewidth=0.9,
    annotate_nodes=True,
    node_label_format="cluster_n",  # "cluster", "cluster_n", or "n"
    node_label_font_size=None,
    node_text_color="black",
    node_label_use_cluster_color=True,
    node_count_text_color="black",
    node_label_fontweight="bold",

    # Small-node label handling.
    # "auto"    = inside large nodes; above small nodes
    # "inside"  = always inside
    # "above"   = always above
    # "outside" = backward-compatible alias for "above"
    small_node_label_mode="auto",
    small_node_label_min_height=18.0,
    small_node_label_outside_offset=0.04,
    small_node_label_line=True,

    # ------------------------------------------------------------------
    # Flow / ribbon styling
    # ------------------------------------------------------------------
    flow_alpha=0.42,
    flow_edgecolor="none",
    flow_linewidth=0.0,
    min_flow_count=1,

    # ------------------------------------------------------------------
    # Stability / stayed-switched summary annotation
    # ------------------------------------------------------------------
    summary_mode="trajectory",
    # Options:
    #   "trajectory"        = stable across all displayed timepoints vs ever switched
    #   "baseline_pairwise" = each follow-up vs reference_timepoint
    #   "adjacent_pairwise" = each timepoint vs immediately previous timepoint
    #   None or "none"      = no summary annotation

    summary_font_size=None,
    summary_text_color="black",
    summary_box=True,
    summary_box_facecolor="white",
    summary_box_alpha=0.88,
    summary_box_edgecolor="none",
    summary_y_multiplier=1.065,

    # ------------------------------------------------------------------
    # Figure / axis styling
    # ------------------------------------------------------------------
    annotate_timepoints=True,
    title=None,
    subtitle=None,
    figsize=None,
    font_size=12.0,
    horizontal_padding=0.06,
    axis_line_color="black",
    show_frame=False,
    show_y_axis=True,
    ylabel="Number of subjects",
    show_grid=True,
    grid_color="gray",
    grid_alpha=0.18,
    grid_linewidth=0.8,
    show_legend=True,
    legend_title=None,
    legend_outside=False,
    legend_loc="upper right",
    legend_bbox_to_anchor=None,
    legend_frameon=False,
    show=True,
):
    """
    Plot an aggregated alluvial / Sankey-style longitudinal cluster transition diagram.

    This is intended for longitudinal clustering results produced by
    make_longitudinal_clustering_ga(...).

    Core visual structure
    ---------------------
    - Each vertical column is a timepoint.
    - Each stacked block is a cluster at that timepoint.
    - Each ribbon is an aggregated flow of subjects between adjacent timepoints.
    - Ribbon width is proportional to the number of subjects in that transition.

    Why this plot is useful
    -----------------------
    Pairwise heatmaps are clear for 2 timepoints, but become harder to interpret
    with 3+ timepoints. This alluvial plot shows the full longitudinal cluster
    trajectory in one figure.

    Recommended use
    ---------------
    For baseline-defined longitudinal clustering, use:

        color_by="reference_cluster"
        node_color_by="fixed"
        node_facecolor="white"
        summary_mode="trajectory"

    This tracks what happens over time to subjects who started in each baseline
    cluster and summarizes full-trajectory stability.

    Parameters
    ----------
    result:
        GA result dictionary returned by ga_runner.run().
        Must contain result["membership_df"] with columns such as:
            cluster_baseline
            cluster_week6
            cluster_month6

    labels:
        Optional row-aligned class labels, such as diagnosis labels where
        0 = TD and 1 = ASD. Required when label_filter is not None.

    label_filter:
        Optional single label value used to restrict the alluvial plot to one
        class before nodes, ribbons, and transition summaries are calculated.
        For example, label_filter=0 can select TD subjects.

    label_filter_name:
        Optional readable name for the selected class, such as "TD" or
        "Typically Developing". When title is None, this name is included in
        the automatically generated title. It does not affect filtering.

    timepoints:
        Ordered list of timepoints to show.
        If None, uses result["timepoint_order"] when available, otherwise infers
        from membership_df cluster_* columns.

    reference_timepoint:
        Timepoint used to define reference cluster colors when
        color_by="reference_cluster".

    cluster_colors:
        Optional mapping from cluster label to color.

        Example:
            {
                0: "#1587F8",
                1: "#FFAE17",
                2: "#049B4F",
            }

    cluster_label_map:
        Optional mapping from raw cluster labels to display labels.

        Example:
            {
                0: "Cluster 0",
                1: "Cluster 1",
                2: "Cluster 2",
            }

    timepoint_label_map:
        Optional mapping from timepoint names to display labels.

        Example:
            {
                "baseline": "Baseline",
                "week6": "Week 6",
                "month6": "Month 6",
            }

    color_by:
        Controls ribbon color.

        "reference_cluster":
            Color each subject/flow by the subject's cluster at reference_timepoint.
            Recommended for baseline-defined longitudinal clustering.

        "source_cluster":
            Color each adjacent transition by the source cluster at that transition.

        "target_cluster":
            Color each adjacent transition by the target cluster at that transition.

    node_color_by:
        Controls node block color.

        "fixed":
            Use node_facecolor for all node blocks.

        "cluster":
            Tint each node by its cluster color.

    node_label_use_cluster_color:
        If True, subtype names use the corresponding cluster color from
        cluster_colors. Counts remain controlled by node_count_text_color.

    node_count_text_color:
        Text color used for the node count line, such as "n=98".

    node_label_fontweight:
        Font weight used for subtype names.

    small_node_label_mode:
        Controls label placement for node blocks.

        "auto":
            Keep labels inside sufficiently tall nodes and move labels centered
            above nodes whose height is below small_node_label_min_height.

        "inside":
            Always draw node labels inside their blocks.

        "above":
            Always draw node labels centered above their blocks.

        "outside":
            Backward-compatible alias for "above".

    small_node_label_min_height:
        Minimum node height required to keep the label inside when
        small_node_label_mode="auto".

    small_node_label_outside_offset:
        Vertical offset above a small node, expressed as a fraction of the
        maximum node-stack height. For example, 0.04 places the label roughly
        4% of the plot's node height above the block.

    small_node_label_line:
        If True, draw a short vertical connector line from an above-node label
        to the top center of its node.

    summary_mode:
        Controls the top stayed/switched annotation.

        "trajectory":
            Full longitudinal stability across all displayed timepoints.
            A subject is stable if all cluster labels across the displayed
            timepoints are identical.

            Example:
                Stable across all timepoints: 70.2% (207/295)
                Ever switched at least once: 29.8% (88/295)

        "baseline_pairwise":
            For each follow-up timepoint, compare cluster assignment against
            reference_timepoint.

        "adjacent_pairwise":
            For each timepoint after the first, compare cluster assignment against
            the immediately previous timepoint.

        None or "none":
            Do not show a stayed/switched summary annotation.

    horizontal_padding:
        Compact horizontal padding beyond the outer node edges. Smaller values
        maximize the space occupied by the blocks and ribbons.

    legend_outside:
        If True, place the legend outside the plotting area on the right.
        This prevents overlap with small upper-right nodes.

    legend_loc:
        Matplotlib legend location. Examples:
            "upper right"
            "lower right"
            "upper left"
            "lower left"
            "center right"
            "center left"
            "best"

    legend_bbox_to_anchor:
        Optional Matplotlib bbox_to_anchor.
        Use None to place the legend inside the plot using legend_loc.
        Use a tuple to place outside or custom-position the legend.

        Examples:
            legend_loc="upper right", legend_bbox_to_anchor=None
            legend_loc="center left", legend_bbox_to_anchor=(1.02, 0.5)

    Returns
    -------
    outputs:
        Dictionary containing:
            "fig"
            "ax"
            "node_df"
            "flow_df"
            "summary_df"
            "membership_df"
            "cluster_colors"
    """


    # ------------------------------------------------------------------
    # Validate and resolve input data
    # ------------------------------------------------------------------
    if "membership_df" not in result:
        raise KeyError("result is missing 'membership_df'.")

    membership_df = result["membership_df"].copy().reset_index(drop=True)

    if not isinstance(membership_df, pd.DataFrame) or membership_df.empty:
        raise ValueError("result['membership_df'] must be a non-empty DataFrame.")

    # ------------------------------------------------------------------
    # Optional class-label filtering
    # ------------------------------------------------------------------
    if labels is not None:
        labels_series = pd.Series(
            labels,
            name="_filter_label",
        ).reset_index(drop=True)

        if len(labels_series) != len(membership_df):
            raise ValueError(
                f"labels has length {len(labels_series)}, but membership_df has "
                f"{len(membership_df)} rows. Labels must be row-aligned."
            )

        membership_df["_filter_label"] = labels_series.to_numpy()

    if label_filter is not None:
        if labels is None:
            raise ValueError(
                "labels must be provided when label_filter is not None."
            )

        membership_df = membership_df.loc[
            membership_df["_filter_label"] == label_filter
        ].copy()

        if membership_df.empty:
            raise ValueError(
                f"No subjects matched label_filter={label_filter!r}."
            )

        if label_filter_name is None:
            label_filter_name = str(label_filter)

    cluster_cols = [
        col for col in membership_df.columns
        if str(col).startswith("cluster_")
    ]

    if len(cluster_cols) == 0:
        raise KeyError("membership_df does not contain any columns starting with 'cluster_'.")

    if timepoints is None:
        timepoints = result.get("timepoint_order", None)

        if timepoints is None:
            timepoints = [
                str(col).replace("cluster_", "", 1)
                for col in cluster_cols
            ]

    timepoints = [str(tp) for tp in timepoints]

    if len(timepoints) < 2:
        raise ValueError("At least two timepoints are required for an alluvial plot.")

    missing_cluster_cols = [
        f"cluster_{tp}" for tp in timepoints
        if f"cluster_{tp}" not in membership_df.columns
    ]

    if missing_cluster_cols:
        raise KeyError(
            "membership_df is missing required cluster columns: "
            f"{missing_cluster_cols}"
        )

    reference_col = f"cluster_{reference_timepoint}"
    if reference_col not in membership_df.columns:
        raise KeyError(
            f"membership_df is missing reference cluster column: {reference_col!r}."
        )

    requested_cluster_cols = [f"cluster_{tp}" for tp in timepoints]

    selected_columns = list(requested_cluster_cols)
    if "_filter_label" in membership_df.columns:
        selected_columns.append("_filter_label")

    d = membership_df[selected_columns].copy()
    d = d.dropna(subset=requested_cluster_cols).reset_index(drop=True)

    if d.empty:
        raise ValueError("No complete cluster-assignment rows are available to plot.")

    # ------------------------------------------------------------------
    # Resolve display labels
    # ------------------------------------------------------------------
    cluster_label_map = dict(cluster_label_map or {})
    timepoint_label_map = dict(timepoint_label_map or {})

    def _display_cluster_label(cluster):
        return str(cluster_label_map.get(cluster, cluster_label_map.get(str(cluster), cluster)))

    def _display_timepoint_label(tp):
        return str(timepoint_label_map.get(tp, tp))

    small_node_label_mode_key = str(small_node_label_mode).lower()

    if small_node_label_mode_key not in {
        "auto",
        "inside",
        "above",
        "outside",
    }:
        raise ValueError(
            "small_node_label_mode must be 'auto', 'inside', 'above', "
            "or 'outside'."
        )

    # Preserve compatibility with the previous name while using the clearer
    # above-node placement behavior.
    if small_node_label_mode_key == "outside":
        small_node_label_mode_key = "above"

    if float(small_node_label_min_height) < 0:
        raise ValueError("small_node_label_min_height must be non-negative.")

    if float(small_node_label_outside_offset) < 0:
        raise ValueError("small_node_label_outside_offset must be non-negative.")

    if float(horizontal_padding) < 0:
        raise ValueError("horizontal_padding must be non-negative.")

    all_clusters = sorted(
        pd.unique(
            pd.concat(
                [d[f"cluster_{tp}"] for tp in timepoints],
                ignore_index=True,
            )
        )
    )

    # ------------------------------------------------------------------
    # Resolve colors
    # ------------------------------------------------------------------
    cluster_colors = dict(cluster_colors or {})

    if cluster_palette is None:
        cluster_palette = [
            "#1587F8",  # blue
            "#FFAE17",  # orange
            "#049B4F",  # green
            "#B14DFF",  # purple
            "#F14949",  # red
            "#00A6A6",  # teal
        ]

    cluster_palette = list(cluster_palette)

    if len(cluster_palette) < len(all_clusters):
        cmap = plt.get_cmap("tab20")
        extra_needed = len(all_clusters) - len(cluster_palette)
        cluster_palette = cluster_palette + [
            cmap(i % cmap.N) for i in range(extra_needed)
        ]

    resolved_cluster_colors = {}
    for i, cluster in enumerate(all_clusters):
        if cluster in cluster_colors:
            resolved_cluster_colors[cluster] = cluster_colors[cluster]
        elif str(cluster) in cluster_colors:
            resolved_cluster_colors[cluster] = cluster_colors[str(cluster)]
        else:
            resolved_cluster_colors[cluster] = cluster_palette[i % len(cluster_palette)]

    # ------------------------------------------------------------------
    # Compute node/block positions
    # ------------------------------------------------------------------
    node_rows = []
    max_total_height = 0.0

    for tp_idx, tp in enumerate(timepoints):
        col = f"cluster_{tp}"
        counts = d[col].value_counts(dropna=False).sort_index()

        total_height = float(counts.sum()) + node_gap * max(len(counts) - 1, 0)
        max_total_height = max(max_total_height, total_height)

        y_cursor = 0.0

        for cluster in sorted(counts.index):
            n = int(counts.loc[cluster])
            y0 = y_cursor
            y1 = y_cursor + n

            node_rows.append(
                {
                    "timepoint": tp,
                    "timepoint_index": tp_idx,
                    "cluster": cluster,
                    "n": n,
                    "x_center": float(tp_idx),
                    "x_left": float(tp_idx) - node_width / 2.0,
                    "x_right": float(tp_idx) + node_width / 2.0,
                    "y0": float(y0),
                    "y1": float(y1),
                    "y_mid": float((y0 + y1) / 2.0),
                }
            )

            y_cursor = y1 + node_gap

    node_df = pd.DataFrame(node_rows)

    # Center each timepoint stack vertically.
    centered_rows = []

    for tp in timepoints:
        sub = node_df[node_df["timepoint"] == tp].copy()
        stack_height = float(sub["y1"].max() - sub["y0"].min())
        y_shift = (max_total_height - stack_height) / 2.0

        for col in ["y0", "y1", "y_mid"]:
            sub[col] = sub[col] + y_shift

        centered_rows.append(sub)

    node_df = pd.concat(centered_rows, ignore_index=True)

    node_lookup = {
        (row["timepoint"], row["cluster"]): row
        for _, row in node_df.iterrows()
    }

    # ------------------------------------------------------------------
    # Compute adjacent transition flows
    # ------------------------------------------------------------------
    flow_rows = []

    for i in range(len(timepoints) - 1):
        source_tp = timepoints[i]
        target_tp = timepoints[i + 1]

        source_col = f"cluster_{source_tp}"
        target_col = f"cluster_{target_tp}"

        tmp = pd.DataFrame(
            {
                "_reference_cluster": d[reference_col].to_numpy(),
                "_source_cluster": d[source_col].to_numpy(),
                "_target_cluster": d[target_col].to_numpy(),
            }
        )

        grouped = (
            tmp.groupby(
                ["_reference_cluster", "_source_cluster", "_target_cluster"],
                dropna=False,
            )
            .size()
            .reset_index(name="n")
            .rename(
                columns={
                    "_reference_cluster": "reference_cluster",
                    "_source_cluster": "source_cluster",
                    "_target_cluster": "target_cluster",
                }
            )
        )

        grouped["source_timepoint"] = source_tp
        grouped["target_timepoint"] = target_tp
        grouped["transition_index"] = i

        flow_rows.append(grouped)

    flow_df = pd.concat(flow_rows, ignore_index=True)

    flow_df = flow_df[flow_df["n"] >= int(min_flow_count)].copy()

    if flow_df.empty:
        raise ValueError(
            "No flows remain after applying min_flow_count. "
            "Lower min_flow_count or check membership_df."
        )

    flow_df = (
        flow_df.sort_values(
            [
                "transition_index",
                "source_cluster",
                "target_cluster",
                "reference_cluster",
            ]
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Allocate vertical flow subsegments inside source and target nodes
    # ------------------------------------------------------------------
    source_offsets = {}
    target_offsets = {}
    allocated_rows = []

    for _, row in flow_df.iterrows():
        source_key = (
            row["transition_index"],
            row["source_timepoint"],
            row["source_cluster"],
        )
        target_key = (
            row["transition_index"],
            row["target_timepoint"],
            row["target_cluster"],
        )

        source_node = node_lookup[(row["source_timepoint"], row["source_cluster"])]
        target_node = node_lookup[(row["target_timepoint"], row["target_cluster"])]

        if source_key not in source_offsets:
            source_offsets[source_key] = float(source_node["y0"])

        if target_key not in target_offsets:
            target_offsets[target_key] = float(target_node["y0"])

        n = float(row["n"])

        source_y0 = source_offsets[source_key]
        source_y1 = source_y0 + n
        source_offsets[source_key] = source_y1

        target_y0 = target_offsets[target_key]
        target_y1 = target_y0 + n
        target_offsets[target_key] = target_y1

        allocated = dict(row)
        allocated.update(
            {
                "source_x": float(source_node["x_right"]),
                "target_x": float(target_node["x_left"]),
                "source_y0": source_y0,
                "source_y1": source_y1,
                "target_y0": target_y0,
                "target_y1": target_y1,
            }
        )
        allocated_rows.append(allocated)

    flow_df = pd.DataFrame(allocated_rows)

    # ------------------------------------------------------------------
    # Compute stability / stayed-switched summary
    # ------------------------------------------------------------------
    summary_mode_key = None if summary_mode is None else str(summary_mode).lower()

    if summary_mode_key in {"none", "false", "off"}:
        summary_mode_key = None

    valid_summary_modes = {None, "trajectory", "baseline_pairwise", "adjacent_pairwise"}
    if summary_mode_key not in valid_summary_modes:
        raise ValueError(
            "summary_mode must be one of: "
            "'trajectory', 'baseline_pairwise', 'adjacent_pairwise', or None."
        )

    summary_rows = []

    if summary_mode_key == "trajectory":
        cluster_matrix = d[requested_cluster_cols].to_numpy()

        first_labels = cluster_matrix[:, [0]]
        stable_mask = np.all(cluster_matrix == first_labels, axis=1)

        total_n = int(len(d))
        n_stable = int(stable_mask.sum())
        n_switched = int(total_n - n_stable)

        pct_stable = float(n_stable / total_n * 100.0) if total_n > 0 else np.nan
        pct_switched = float(n_switched / total_n * 100.0) if total_n > 0 else np.nan

        summary_rows.append(
            {
                "summary_mode": "trajectory",
                "comparison": "all_timepoints",
                "n_total": total_n,
                "n_stable": n_stable,
                "n_switched": n_switched,
                "percent_stable": pct_stable,
                "percent_switched": pct_switched,
            }
        )

    elif summary_mode_key == "baseline_pairwise":
        for i, tp in enumerate(timepoints):
            current_col = f"cluster_{tp}"

            if i == 0:
                compare_tp = tp
                compare_col = current_col
            else:
                compare_tp = reference_timepoint
                compare_col = reference_col

            total_n = int(len(d))
            n_same = int((d[current_col].to_numpy() == d[compare_col].to_numpy()).sum())
            n_switched = int(total_n - n_same)

            pct_same = float(n_same / total_n * 100.0) if total_n > 0 else np.nan
            pct_switched = float(n_switched / total_n * 100.0) if total_n > 0 else np.nan

            summary_rows.append(
                {
                    "summary_mode": "baseline_pairwise",
                    "timepoint": tp,
                    "compare_timepoint": compare_tp,
                    "n_total": total_n,
                    "n_same": n_same,
                    "n_switched": n_switched,
                    "percent_same": pct_same,
                    "percent_switched": pct_switched,
                }
            )

    elif summary_mode_key == "adjacent_pairwise":
        for i, tp in enumerate(timepoints):
            current_col = f"cluster_{tp}"

            if i == 0:
                compare_tp = tp
                compare_col = current_col
            else:
                compare_tp = timepoints[i - 1]
                compare_col = f"cluster_{compare_tp}"

            total_n = int(len(d))
            n_same = int((d[current_col].to_numpy() == d[compare_col].to_numpy()).sum())
            n_switched = int(total_n - n_same)

            pct_same = float(n_same / total_n * 100.0) if total_n > 0 else np.nan
            pct_switched = float(n_switched / total_n * 100.0) if total_n > 0 else np.nan

            summary_rows.append(
                {
                    "summary_mode": "adjacent_pairwise",
                    "timepoint": tp,
                    "compare_timepoint": compare_tp,
                    "n_total": total_n,
                    "n_same": n_same,
                    "n_switched": n_switched,
                    "percent_same": pct_same,
                    "percent_switched": pct_switched,
                }
            )

    summary_df = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------------
    # Ribbon helper
    # ------------------------------------------------------------------
    def _draw_ribbon(ax, x0, x1, y0_low, y0_high, y1_low, y1_high, color):
        dx = x1 - x0
        c0 = x0 + 0.45 * dx
        c1 = x1 - 0.45 * dx

        verts = [
            (x0, y0_high),
            (c0, y0_high),
            (c1, y1_high),
            (x1, y1_high),
            (x1, y1_low),
            (c1, y1_low),
            (c0, y0_low),
            (x0, y0_low),
            (x0, y0_high),
        ]

        codes = [
            Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CLOSEPOLY,
        ]

        patch = PathPatch(
            Path(verts, codes),
            facecolor=color,
            edgecolor=flow_edgecolor,
            linewidth=flow_linewidth,
            alpha=flow_alpha,
            zorder=1,
        )
        ax.add_patch(patch)

    # ------------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------------
    if figsize is None:
        figsize = (max(8.5, 2.8 * len(timepoints)), 6.2)

    fig, ax = plt.subplots(figsize=figsize)

    # ------------------------------------------------------------------
    # Draw flows first
    # ------------------------------------------------------------------
    color_by = str(color_by).lower()

    if color_by not in {"reference_cluster", "source_cluster", "target_cluster"}:
        raise ValueError(
            "color_by must be one of: "
            "'reference_cluster', 'source_cluster', or 'target_cluster'."
        )

    for _, row in flow_df.iterrows():
        if color_by == "reference_cluster":
            color_cluster = row["reference_cluster"]
        elif color_by == "source_cluster":
            color_cluster = row["source_cluster"]
        else:
            color_cluster = row["target_cluster"]

        color = resolved_cluster_colors.get(color_cluster, "#999999")

        _draw_ribbon(
            ax,
            row["source_x"],
            row["target_x"],
            row["source_y0"],
            row["source_y1"],
            row["target_y0"],
            row["target_y1"],
            color,
        )

    # ------------------------------------------------------------------
    # Draw nodes on top
    # ------------------------------------------------------------------
    node_color_by_key = str(node_color_by).lower()
    if node_color_by_key not in {"cluster", "fixed"}:
        raise ValueError("node_color_by must be either 'cluster' or 'fixed'.")

    label_fs = node_label_font_size
    if label_fs is None:
        label_fs = max(8.0, font_size - 1.0)

    # Track above-node labels so the final y limits and summary placement
    # reserve enough vertical space for them.
    above_label_rows = []

    node_df = node_df.copy()
    node_df["label_outside"] = False
    node_df["label_side"] = "inside"
    node_df["label_y"] = np.nan

    # Convert the user-facing fractional offset into data-axis units.
    node_stack_height = max(
        float(node_df["y1"].max()),
        1.0,
    )
    above_label_offset = (
        float(small_node_label_outside_offset)
        * node_stack_height
    )

    for row_index, row in node_df.iterrows():
        if node_color_by_key == "cluster":
            facecolor = resolved_cluster_colors.get(
                row["cluster"],
                node_facecolor,
            )
        else:
            facecolor = node_facecolor

        rect = Rectangle(
            (row["x_left"], row["y0"]),
            node_width,
            row["y1"] - row["y0"],
            facecolor=facecolor,
            edgecolor=node_edgecolor,
            linewidth=node_linewidth,
            alpha=node_alpha,
            zorder=3,
        )
        ax.add_patch(rect)

        if not annotate_nodes:
            continue

        cluster_text = _display_cluster_label(row["cluster"])
        count_text = f"n={int(row['n'])}"
        fmt = str(node_label_format).lower()

        if fmt not in {"cluster", "n", "cluster_n"}:
            raise ValueError(
                "node_label_format must be one of: "
                "'cluster', 'n', or 'cluster_n'."
            )

        cluster_text_color = (
            resolved_cluster_colors.get(row["cluster"], node_text_color)
            if node_label_use_cluster_color
            else node_text_color
        )

        node_height = float(row["y1"] - row["y0"])

        label_above = (
            small_node_label_mode_key == "above"
            or (
                small_node_label_mode_key == "auto"
                and node_height < float(small_node_label_min_height)
            )
        )

        # Draw subtype name and count separately so that the subtype name can
        # use the same color as the legend while the count remains black.
        line_gap = max(1.2, 0.055 * node_stack_height)

        if not label_above:
            if fmt == "cluster":
                ax.text(
                    row["x_center"],
                    row["y_mid"],
                    cluster_text,
                    ha="center",
                    va="center",
                    fontsize=label_fs,
                    fontweight=node_label_fontweight,
                    color=cluster_text_color,
                    zorder=4,
                )
            elif fmt == "n":
                ax.text(
                    row["x_center"],
                    row["y_mid"],
                    count_text,
                    ha="center",
                    va="center",
                    fontsize=label_fs,
                    color=node_count_text_color,
                    zorder=4,
                )
            else:
                ax.text(
                    row["x_center"],
                    row["y_mid"] + 0.5 * line_gap,
                    cluster_text,
                    ha="center",
                    va="center",
                    fontsize=label_fs,
                    fontweight=node_label_fontweight,
                    color=cluster_text_color,
                    zorder=4,
                )
                ax.text(
                    row["x_center"],
                    row["y_mid"] - 0.5 * line_gap,
                    count_text,
                    ha="center",
                    va="center",
                    fontsize=label_fs,
                    color=node_count_text_color,
                    zorder=4,
                )
            continue

        anchor_x = float(row["x_center"])
        anchor_y = float(row["y1"])
        label_y = anchor_y + above_label_offset

        if small_node_label_line:
            ax.plot(
                [anchor_x, anchor_x],
                [anchor_y, label_y - 0.20 * line_gap],
                color=node_edgecolor,
                linewidth=max(0.6, 0.8 * float(node_linewidth)),
                clip_on=False,
                zorder=4,
            )

        if fmt == "cluster":
            ax.text(
                anchor_x,
                label_y,
                cluster_text,
                ha="center",
                va="bottom",
                fontsize=label_fs,
                fontweight=node_label_fontweight,
                color=cluster_text_color,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.90,
                    "pad": 0.15,
                },
                clip_on=False,
                zorder=5,
            )
        elif fmt == "n":
            ax.text(
                anchor_x,
                label_y,
                count_text,
                ha="center",
                va="bottom",
                fontsize=label_fs,
                color=node_count_text_color,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.90,
                    "pad": 0.15,
                },
                clip_on=False,
                zorder=5,
            )
        else:
            ax.text(
                anchor_x,
                label_y + 0.35 * line_gap,
                cluster_text,
                ha="center",
                va="bottom",
                fontsize=label_fs,
                fontweight=node_label_fontweight,
                color=cluster_text_color,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.90,
                    "pad": 0.15,
                },
                clip_on=False,
                zorder=5,
            )
            ax.text(
                anchor_x,
                label_y - 0.35 * line_gap,
                count_text,
                ha="center",
                va="top",
                fontsize=label_fs,
                color=node_count_text_color,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.90,
                    "pad": 0.15,
                },
                clip_on=False,
                zorder=5,
            )

        node_df.at[row_index, "label_outside"] = True
        node_df.at[row_index, "label_side"] = "above"
        node_df.at[row_index, "label_y"] = label_y

        above_label_rows.append({
            "timepoint_index": int(row["timepoint_index"]),
            "label_y": label_y,
        })

    # ------------------------------------------------------------------
    # Axes limits first, so summary annotations can use stable y position
    # ------------------------------------------------------------------
    y_max = float(node_df["y1"].max())

    highest_above_label_y = y_max
    if above_label_rows:
        highest_above_label_y = max(
            item["label_y"]
            for item in above_label_rows
        )

    # Keep the transition summary above any small-node labels so the two
    # annotation layers cannot overlap.
    annotation_clearance = 0.045 * max(y_max, 1.0)
    summary_annotation_y = max(
        y_max * float(summary_y_multiplier),
        highest_above_label_y + annotation_clearance,
    )

    if summary_mode_key == "trajectory":
        y_top = max(
            y_max * 1.20,
            summary_annotation_y + 0.085 * max(y_max, 1.0),
        )
    elif summary_mode_key in {"baseline_pairwise", "adjacent_pairwise"}:
        y_top = max(
            y_max * 1.18,
            summary_annotation_y + 0.080 * max(y_max, 1.0),
        )
    else:
        y_top = max(
            y_max * 1.08,
            highest_above_label_y + 0.060 * max(y_max, 1.0),
        )

    ax.set_ylim(0, y_top)

    x_min = float(node_df["x_left"].min()) - float(horizontal_padding)
    x_max = float(node_df["x_right"].max()) + float(horizontal_padding)

    ax.set_xlim(
        x_min,
        x_max,
    )

    # ------------------------------------------------------------------
    # Summary annotations
    # ------------------------------------------------------------------
    if summary_mode_key is not None and not summary_df.empty:
        summary_fs = summary_font_size
        if summary_fs is None:
            summary_fs = max(8.0, font_size - 1.0)

        bbox = None
        if summary_box:
            bbox = {
                "boxstyle": "round,pad=0.30",
                "facecolor": summary_box_facecolor,
                "alpha": summary_box_alpha,
                "edgecolor": summary_box_edgecolor,
            }

        annotation_y = summary_annotation_y

        if summary_mode_key == "trajectory":
            row = summary_df.iloc[0]

            label = (
                f"Stable across all timepoints: "
                f"{row['percent_stable']:.1f}% ({int(row['n_stable'])}/{int(row['n_total'])})\n"
                f"Ever switched at least once: "
                f"{row['percent_switched']:.1f}% ({int(row['n_switched'])}/{int(row['n_total'])})"
            )

            ax.text(
                (len(timepoints) - 1) / 2.0,
                annotation_y,
                label,
                ha="center",
                va="bottom",
                fontsize=summary_fs,
                color=summary_text_color,
                bbox=bbox,
                zorder=5,
            )

        elif summary_mode_key in {"baseline_pairwise", "adjacent_pairwise"}:
            for i, tp in enumerate(timepoints[1:], start=1):
                row = summary_df[summary_df["timepoint"] == tp].iloc[0]

                label = (
                    f"vs {_display_timepoint_label(row['compare_timepoint'])}\n"
                    f"Stayed {row['percent_same']:.1f}%\n"
                    f"Switched {row['percent_switched']:.1f}%"
                )

                ax.text(
                    i,
                    annotation_y,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=summary_fs,
                    color=summary_text_color,
                    bbox=bbox,
                    zorder=5,
                )

    # ------------------------------------------------------------------
    # Axes, labels, title
    # ------------------------------------------------------------------
    x_positions = np.arange(len(timepoints))

    if annotate_timepoints:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [_display_timepoint_label(tp) for tp in timepoints],
            fontsize=font_size,
        )
    else:
        ax.set_xticks([])

    if show_y_axis:
        ax.set_ylabel(ylabel, fontsize=font_size, fontweight="bold")
        ax.tick_params(axis="y", labelsize=font_size - 1)
    else:
        ax.set_yticks([])

    ax.tick_params(axis="x", labelsize=font_size)

    if title is None:
        if label_filter_name is not None:
            title = (
                f"{label_filter_name}: "
                "Longitudinal cluster transitions"
            )
        else:
            title = "Longitudinal cluster transitions"

    # The plot title is always centered for a consistent presentation.
    if subtitle is not None:
        ax.set_title(
            f"{title}\n{subtitle}",
            fontsize=font_size + 2,
            fontweight="bold",
            loc="center",
            pad=12,
        )
    else:
        ax.set_title(
            title,
            fontsize=font_size + 2,
            fontweight="bold",
            loc="center",
            pad=12,
        )

    if show_grid and show_y_axis:
        ax.grid(
            True,
            axis="y",
            color=grid_color,
            alpha=grid_alpha,
            linewidth=grid_linewidth,
            zorder=0,
        )
    else:
        ax.grid(False)

    if show_frame:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(axis_line_color)
            spine.set_linewidth(1.0)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    if show_legend:
        if legend_title is None:
            if color_by == "reference_cluster":
                legend_title = (
                    f"{_display_timepoint_label(reference_timepoint)} cluster"
                )
            elif color_by == "source_cluster":
                legend_title = "Source cluster"
            else:
                legend_title = "Target cluster"

        handles = [
            Patch(
                facecolor=resolved_cluster_colors[cluster],
                edgecolor="none",
                alpha=flow_alpha,
                label=_display_cluster_label(cluster),
            )
            for cluster in all_clusters
        ]

        resolved_legend_loc = legend_loc
        resolved_legend_bbox = legend_bbox_to_anchor

        if legend_outside:
            resolved_legend_loc = "center left"

            if resolved_legend_bbox is None:
                resolved_legend_bbox = (1.01, 0.5)

        legend_kwargs = {
            "handles": handles,
            "title": legend_title,
            "loc": resolved_legend_loc,
            "fontsize": font_size - 1,
            "title_fontsize": font_size,
            "frameon": legend_frameon,
        }

        if resolved_legend_bbox is not None:
            legend_kwargs["bbox_to_anchor"] = resolved_legend_bbox

        ax.legend(**legend_kwargs)

    if legend_outside and show_legend:
        fig.tight_layout(rect=[0, 0, 0.86, 1])
    else:
        fig.tight_layout()

    if show:
        plt.show()

    return {
        "fig": fig,
        "ax": ax,
        "node_df": node_df,
        "flow_df": flow_df,
        "summary_df": summary_df,
        "membership_df": d,
        "cluster_colors": resolved_cluster_colors,
        "label_filter": label_filter,
        "label_filter_name": label_filter_name,
        "small_node_label_mode": small_node_label_mode_key,
        "legend_outside": bool(legend_outside),
    }



def plot_within_label_cluster_characterization(
    *,
    result,
    cfg,
    labels,
    target_label=1,
    variables,
    variable_order=None,
    variable_order_col="feature",
    timepoints=("baseline", "week6", "month6"),
    timepoint_label_map=None,

    # Single-timepoint support
    single_timepoint_df=None,

    # Plotting mode
    plot_mode="within_label",  # "within_label" or "td_plus_target"
    reference_label=0,
    reference_group_name="TD",
    target_group_prefix="ASD",

    # Optional custom cluster/subtype names
    # Example:
    # {
    #     0: "ASD Subtype 1",
    #     1: "ASD Subtype 2",
    # }
    cluster_label_map=None,

    group_order=None,

    # Pairwise title statistics
    pairwise_comparisons_order=None,
    pairwise_stats_per_title_row=2,
    pairwise_stats_line_sep="\n",

    # Optional FDR-corrected significance brackets
    show_significance_brackets=False,

    # Styling
    cluster_colors=None,
    title="Within-label cluster characterization",
    ylabel=None,
    figsize=None,
    jitter=0.06,
    point_alpha=0.55,
    point_size=18,
    box_width=0.55,
    show_stats_in_title=True,
    suptitle_y=1.01,
    show=True,
):
    """
    Plot selected feature values by cluster/subtype across one or more timepoints.

    Modes
    -----
    plot_mode="within_label":
        Keeps only target_label subjects, such as ASD subjects.

        Default group names:
            C0, C1, ...

        Custom names can be supplied with cluster_label_map:
            {
                0: "ASD Subtype 1",
                1: "ASD Subtype 2",
            }

    plot_mode="td_plus_target":
        Keeps reference_label and target_label subjects.

        The reference label becomes one group, such as TD.
        Target-label subjects are divided according to cluster/subtype.

        Default target group names:
            ASD 0, ASD 1, ...

        Custom target names can be supplied with cluster_label_map:
            {
                0: "ASD Subtype 1",
                1: "ASD Subtype 2",
            }

    Single-timepoint support
    ------------------------
    For a single-timepoint dataset, pass:

        timepoints=["baseline"]
        single_timepoint_df=<your dataframe>

    For a single timepoint, variables can be written in either form:

        {
            "feature_1": {
                "baseline": "feature_1",
            }
        }

    or:

        {
            "feature_1": "feature_1",
        }

    Statistics
    ----------
    Omnibus:
        2 groups: Mann-Whitney U
        3+ groups: Kruskal-Wallis

    Pairwise:
        Mann-Whitney U for each requested group pair.
        Pairwise FDR-adjusted values are shown in panel titles.

    Significance brackets
    ---------------------
    When show_significance_brackets=True, significant pairwise comparisons
    are also drawn as horizontal brackets above the groups. Asterisks are
    based on the same FDR-corrected p-values shown in the panel titles.

    Returns
    -------
    plot_df, stats_df, pairwise_stats_df, fig, axes
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    from itertools import combinations

    # ------------------------------------------------------------------
    # Normalize inputs
    # ------------------------------------------------------------------

    if isinstance(timepoints, str):
        timepoints = [timepoints]
    else:
        timepoints = list(timepoints)

    if len(timepoints) == 0:
        raise ValueError("At least one timepoint must be provided.")

    if plot_mode not in ["within_label", "td_plus_target"]:
        raise ValueError(
            "plot_mode must be either 'within_label' or 'td_plus_target'."
        )

    membership_df = result["membership_df"].copy().reset_index(drop=True)

    labels = pd.Series(
        labels,
        name="label",
    ).reset_index(drop=True)

    if len(labels) != len(membership_df):
        raise ValueError(
            f"labels has length {len(labels)}, but membership_df has "
            f"{len(membership_df)} rows. Labels must be row-aligned."
        )

    # ------------------------------------------------------------------
    # Obtain the timepoint dataframes
    # ------------------------------------------------------------------

    preset_cfg = get_active_fitness_preset_config(cfg)

    timepoint_cfg = dict(
        preset_cfg.get("timepoint_config", {}) or {}
    )

    timepoint_dfs = dict(
        timepoint_cfg.get("timepoint_dfs", {}) or {}
    )

    # Explicit single-timepoint dataframe takes priority.
    if single_timepoint_df is not None:
        if len(timepoints) != 1:
            raise ValueError(
                "single_timepoint_df can only be used when exactly one "
                "timepoint is supplied."
            )

        single_tp = timepoints[0]

        timepoint_dfs = {
            single_tp: single_timepoint_df.copy(),
        }

    # Automatic single-timepoint fallback from data_config["df"].
    elif len(timepoints) == 1 and timepoints[0] not in timepoint_dfs:
        data_cfg = dict(
            preset_cfg.get("data_config", {}) or {}
        )

        config_df = data_cfg.get("df")

        if isinstance(config_df, pd.DataFrame):
            timepoint_dfs = {
                timepoints[0]: config_df.copy(),
            }

    # Confirm that all requested timepoints are available.
    missing_timepoints = [
        tp
        for tp in timepoints
        if tp not in timepoint_dfs
    ]

    if missing_timepoints:
        if len(timepoints) == 1:
            raise KeyError(
                f"No dataframe was found for timepoint "
                f"{missing_timepoints[0]!r}. For a single-timepoint analysis, "
                f"pass single_timepoint_df=<your dataframe>."
            )

        raise KeyError(
            f"timepoint_dfs is missing the following timepoints: "
            f"{missing_timepoints}"
        )

    if timepoint_label_map is None:
        timepoint_label_map = {
            tp: tp
            for tp in timepoints
        }

    # ------------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------------

    default_colors = {
        # Numeric clusters
        0: "#1587F8",
        1: "#FFAE17",
        2: "#049B4F",
        3: "#C04AE2",

        # Original within-label names
        "C0": "#1587F8",
        "C1": "#FFAE17",
        "C2": "#049B4F",
        "C3": "#C04AE2",

        # Original TD + ASD names
        "TD": "#7F7F7F",
        "ASD 0": "#1587F8",
        "ASD 1": "#FFAE17",
        "ASD 2": "#049B4F",
        "ASD 3": "#C04AE2",

        # New subtype names
        "ASD Subtype 1": "#1587F8",
        "ASD Subtype 2": "#FFAE17",
        "ASD Subtype 3": "#049B4F",
        "ASD Subtype 4": "#C04AE2",
    }

    if cluster_colors is None:
        cluster_colors = default_colors.copy()
    else:
        cluster_colors = {
            **default_colors,
            **dict(cluster_colors),
        }

    # ------------------------------------------------------------------
    # Helper: map cluster numbers to display names
    # ------------------------------------------------------------------

    def map_cluster_names(cluster_series, mode):
        cluster_values = pd.to_numeric(
            cluster_series,
            errors="coerce",
        )

        if cluster_values.isna().any():
            raise ValueError(
                "Cluster assignments contain missing or non-numeric values."
            )

        cluster_values = cluster_values.astype(int)

        if cluster_label_map is not None:
            mapped_values = cluster_values.map(cluster_label_map)

            missing_cluster_ids = sorted(
                cluster_values[
                    mapped_values.isna()
                ].unique().tolist()
            )

            if missing_cluster_ids:
                raise ValueError(
                    "cluster_label_map does not contain names for cluster IDs: "
                    f"{missing_cluster_ids}"
                )

            return mapped_values.astype(object)

        if mode == "within_label":
            return (
                "C"
                + cluster_values.astype(str)
            ).astype(object)

        return (
            target_group_prefix
            + " "
            + cluster_values.astype(str)
        ).astype(object)

    # ------------------------------------------------------------------
    # Build long plotting dataframe
    # ------------------------------------------------------------------

    rows = []

    for tp in timepoints:
        df_tp = timepoint_dfs[tp].copy().reset_index(drop=True)

        if len(df_tp) != len(membership_df):
            raise ValueError(
                f"The dataframe for timepoint {tp!r} has {len(df_tp)} rows, "
                f"but membership_df has {len(membership_df)} rows. "
                f"They must be row-aligned."
            )

        cluster_col = f"cluster_{tp}"

        if cluster_col not in membership_df.columns:
            raise KeyError(
                f"membership_df is missing {cluster_col!r}."
            )

        for variable_name, tp_col_map in variables.items():

            # Longitudinal form:
            # {
            #     "feature": {
            #         "baseline": "feature",
            #         "week6": "feature",
            #     }
            # }
            if isinstance(tp_col_map, dict):
                if tp not in tp_col_map:
                    continue

                value_col = tp_col_map[tp]

            # Simplified single-timepoint form:
            # {
            #     "feature": "feature"
            # }
            elif len(timepoints) == 1:
                value_col = tp_col_map

            else:
                raise TypeError(
                    f"Variable {variable_name!r} must map to a dictionary "
                    f"of timepoint-specific columns when multiple timepoints "
                    f"are requested."
                )

            if value_col not in df_tp.columns:
                raise KeyError(
                    f"Column {value_col!r} for variable "
                    f"{variable_name!r} was not found in the dataframe "
                    f"for timepoint {tp!r}."
                )

            tmp = pd.DataFrame({
                "timepoint": tp,
                "timepoint_label": timepoint_label_map.get(tp, tp),
                "variable": variable_name,
                "cluster": membership_df[cluster_col].to_numpy(),
                "label": labels.to_numpy(),
                "value": pd.to_numeric(
                    df_tp[value_col],
                    errors="coerce",
                ).to_numpy(),
            })

            # ----------------------------------------------------------
            # Create plotting groups
            # ----------------------------------------------------------

            if plot_mode == "within_label":
                tmp = tmp[
                    tmp["label"] == target_label
                ].copy()

                tmp["plot_group"] = map_cluster_names(
                    tmp["cluster"],
                    mode="within_label",
                )

            elif plot_mode == "td_plus_target":
                tmp = tmp[
                    tmp["label"].isin(
                        [reference_label, target_label]
                    )
                ].copy()

                tmp["plot_group"] = pd.Series(
                    index=tmp.index,
                    dtype="object",
                )

                reference_mask = (
                    tmp["label"] == reference_label
                )

                target_mask = (
                    tmp["label"] == target_label
                )

                tmp.loc[
                    reference_mask,
                    "plot_group",
                ] = reference_group_name

                target_group_names = map_cluster_names(
                    tmp.loc[target_mask, "cluster"],
                    mode="td_plus_target",
                )

                tmp.loc[
                    target_mask,
                    "plot_group",
                ] = target_group_names.to_numpy()

            rows.append(tmp)

    if len(rows) == 0:
        raise ValueError(
            "No plotting rows were created. Check variables and timepoints."
        )

    plot_df = pd.concat(
        rows,
        ignore_index=True,
    )

    plot_df = plot_df.dropna(
        subset=["plot_group", "value"]
    )

    if plot_df.empty:
        raise ValueError(
            "No valid plotting rows remain after removing missing values."
        )

    # ------------------------------------------------------------------
    # Variable display order
    # ------------------------------------------------------------------

    if variable_order is None:
        variables_order = list(variables.keys())

    elif isinstance(variable_order, pd.DataFrame):
        if variable_order_col not in variable_order.columns:
            raise KeyError(
                f"variable_order_col={variable_order_col!r} was not found "
                f"in variable_order DataFrame columns: "
                f"{list(variable_order.columns)}"
            )

        variables_order = (
            variable_order[variable_order_col]
            .dropna()
            .astype(str)
            .tolist()
        )

    else:
        variables_order = list(variable_order)

    # Remove duplicate variable names while preserving order.
    variables_order = list(
        dict.fromkeys(variables_order)
    )

    variables_order = [
        variable_name
        for variable_name in variables_order
        if variable_name in variables
    ]

    remaining_variables = [
        variable_name
        for variable_name in variables.keys()
        if variable_name not in variables_order
    ]

    variables_order = (
        variables_order
        + remaining_variables
    )

    if len(variables_order) == 0:
        raise ValueError(
            "No variables are available to plot after applying variable_order."
        )

    timepoints_order = list(timepoints)

    # ------------------------------------------------------------------
    # Group display order
    # ------------------------------------------------------------------

    available_groups = list(
        plot_df["plot_group"]
        .dropna()
        .unique()
    )

    if group_order is None:

        if cluster_label_map is not None:
            mapped_cluster_order = [
                group_name
                for group_name in cluster_label_map.values()
                if group_name in available_groups
            ]

            if plot_mode == "td_plus_target":
                group_order = mapped_cluster_order.copy()

                if reference_group_name in available_groups:
                    group_order.append(reference_group_name)

            else:
                group_order = mapped_cluster_order.copy()

        elif plot_mode == "td_plus_target":
            target_groups = sorted([
                group
                for group in available_groups
                if str(group).startswith(target_group_prefix)
            ])

            group_order = target_groups.copy()

            if reference_group_name in available_groups:
                group_order.append(reference_group_name)

        else:
            group_order = sorted(available_groups)

    else:
        group_order = list(group_order)

    group_order = [
        group
        for group in group_order
        if group in available_groups
    ]

    if len(group_order) < 2:
        raise ValueError(
            "At least two plotting groups are required."
        )

    # Add colors for any custom group names that were not supplied.
    fallback_palette = [
        "#1587F8",
        "#FFAE17",
        "#049B4F",
        "#C04AE2",
        "#7F7F7F",
    ]

    for group_index, group_name in enumerate(group_order):
        if group_name not in cluster_colors:
            cluster_colors[group_name] = fallback_palette[
                group_index % len(fallback_palette)
            ]

    # ------------------------------------------------------------------
    # Pairwise p-value helpers
    # ------------------------------------------------------------------

    def format_p_value(p_value):
        if pd.isna(p_value):
            return "NA"

        return f"{p_value:.3g}"

    def fdr_p_value_to_stars(fdr_p_value):
        """Convert an FDR-corrected p-value to significance stars."""

        if pd.isna(fdr_p_value) or fdr_p_value >= 0.05:
            return None

        if fdr_p_value < 0.0001:
            return "****"

        if fdr_p_value < 0.001:
            return "***"

        if fdr_p_value < 0.01:
            return "**"

        return "*"

    def find_pairwise_row(
        pairwise_rows_for_panel,
        group_1,
        group_2,
    ):
        """Return the stored pairwise row for a group pair, in either order."""

        for row in pairwise_rows_for_panel:
            if (
                row["group_1"] == group_1
                and row["group_2"] == group_2
            ) or (
                row["group_1"] == group_2
                and row["group_2"] == group_1
            ):
                return row

        return None

    def short_group_label(group_name):
        group_text = str(group_name)

        subtype_prefix = "ASD Subtype "

        if group_text.startswith(subtype_prefix):
            subtype_number = group_text.replace(
                subtype_prefix,
                "",
            )

            return f"ASD-S{subtype_number}"

        label_map = {
            "ASD 0": "ASD0",
            "ASD 1": "ASD1",
            "ASD 2": "ASD2",
            "ASD 3": "ASD3",
            "TD": "TD",
            "C0": "C0",
            "C1": "C1",
            "C2": "C2",
            "C3": "C3",
        }

        return label_map.get(
            group_text,
            group_text.replace(" ", ""),
        )

    if pairwise_comparisons_order is None:
        pairwise_comparisons_order = list(
            combinations(group_order, 2)
        )
    else:
        pairwise_comparisons_order = list(
            pairwise_comparisons_order
        )

    # Keep only comparisons involving available groups.
    pairwise_comparisons_order = [
        (group_1, group_2)
        for group_1, group_2 in pairwise_comparisons_order
        if group_1 in group_order
        and group_2 in group_order
    ]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    stats_rows = []
    pairwise_rows = []

    for variable_name in variables_order:
        for tp in timepoints_order:
            d = plot_df[
                (plot_df["variable"] == variable_name)
                & (plot_df["timepoint"] == tp)
            ].copy()

            if d.empty:
                continue

            groups = [
                d.loc[
                    d["plot_group"] == group,
                    "value",
                ]
                .dropna()
                .to_numpy()
                for group in group_order
            ]

            group_ns = [
                len(group_values)
                for group_values in groups
            ]

            # Omnibus/original-style statistical test
            if (
                len(group_order) == 2
                and all(n > 0 for n in group_ns)
            ):
                stat, p_value = stats.mannwhitneyu(
                    groups[0],
                    groups[1],
                    alternative="two-sided",
                )

                test_name = "Mann-Whitney U"

            elif (
                len(group_order) > 2
                and all(n > 0 for n in group_ns)
            ):
                stat, p_value = stats.kruskal(
                    *groups
                )

                test_name = "Kruskal-Wallis"

            else:
                stat = np.nan
                p_value = np.nan
                test_name = "Unavailable"

            row = {
                "variable": variable_name,
                "timepoint": tp,
                "test": test_name,
                "n_groups": len(group_order),
                "statistic": stat,
                "p_value": p_value,
            }

            for group, values in zip(
                group_order,
                groups,
            ):
                safe_group = (
                    str(group)
                    .replace(" ", "_")
                )

                row[f"{safe_group}_n"] = len(values)

                row[f"{safe_group}_mean"] = (
                    np.mean(values)
                    if len(values) > 0
                    else np.nan
                )

                row[f"{safe_group}_median"] = (
                    np.median(values)
                    if len(values) > 0
                    else np.nan
                )

            stats_rows.append(row)

            # Pairwise tests
            for group_1, group_2 in pairwise_comparisons_order:
                values_1 = (
                    d.loc[
                        d["plot_group"] == group_1,
                        "value",
                    ]
                    .dropna()
                    .to_numpy()
                )

                values_2 = (
                    d.loc[
                        d["plot_group"] == group_2,
                        "value",
                    ]
                    .dropna()
                    .to_numpy()
                )

                if (
                    len(values_1) > 0
                    and len(values_2) > 0
                ):
                    pairwise_result = stats.mannwhitneyu(
                        values_1,
                        values_2,
                        alternative="two-sided",
                    )

                    pairwise_stat = (
                        pairwise_result.statistic
                    )

                    pairwise_p = (
                        pairwise_result.pvalue
                    )

                    pairwise_test = "Mann-Whitney U"

                else:
                    pairwise_stat = np.nan
                    pairwise_p = np.nan
                    pairwise_test = (
                        "Mann-Whitney U unavailable"
                    )

                pairwise_rows.append({
                    "variable": variable_name,
                    "timepoint": tp,
                    "group_1": group_1,
                    "group_2": group_2,
                    "comparison": (
                        f"{group_1} vs {group_2}"
                    ),
                    "test": pairwise_test,
                    "statistic": pairwise_stat,
                    "p_value": pairwise_p,
                    "n_group_1": len(values_1),
                    "n_group_2": len(values_2),
                    "median_group_1": (
                        np.nanmedian(values_1)
                        if len(values_1) > 0
                        else np.nan
                    ),
                    "median_group_2": (
                        np.nanmedian(values_2)
                        if len(values_2) > 0
                        else np.nan
                    ),
                    "mean_group_1": (
                        np.nanmean(values_1)
                        if len(values_1) > 0
                        else np.nan
                    ),
                    "mean_group_2": (
                        np.nanmean(values_2)
                        if len(values_2) > 0
                        else np.nan
                    ),
                })

    stats_df = pd.DataFrame(
        stats_rows
    )

    pairwise_stats_df = pd.DataFrame(
        pairwise_rows
    )

    # ------------------------------------------------------------------
    # Benjamini-Hochberg FDR helper
    # ------------------------------------------------------------------

    def add_fdr_column(df, p_col="p_value"):
        df = df.copy()
        df["fdr_p_value"] = np.nan

        if df.empty or p_col not in df.columns:
            return df

        valid_mask = df[p_col].notna()

        if valid_mask.sum() == 0:
            return df

        p_values = (
            df.loc[valid_mask, p_col]
            .to_numpy(dtype=float)
        )

        order = np.argsort(p_values)
        ranked_p = p_values[order]
        m = len(p_values)

        q_values = (
            ranked_p
            * m
            / (np.arange(m) + 1)
        )

        q_values = np.minimum.accumulate(
            q_values[::-1]
        )[::-1]

        q_values = np.clip(
            q_values,
            0,
            1,
        )

        fdr_values = np.empty_like(
            q_values
        )

        fdr_values[order] = q_values

        df.loc[
            valid_mask,
            "fdr_p_value",
        ] = fdr_values

        return df

    stats_df = add_fdr_column(
        stats_df
    )

    pairwise_stats_df = add_fdr_column(
        pairwise_stats_df
    )

    # ------------------------------------------------------------------
    # Statistics lookup tables
    # ------------------------------------------------------------------

    stats_lookup = {}

    if not stats_df.empty:
        for _, row in stats_df.iterrows():
            stats_lookup[
                (
                    row["variable"],
                    row["timepoint"],
                )
            ] = row

    pairwise_stats_lookup = {}

    if not pairwise_stats_df.empty:
        for _, row in pairwise_stats_df.iterrows():
            key = (
                row["variable"],
                row["timepoint"],
            )

            pairwise_stats_lookup.setdefault(
                key,
                [],
            ).append(row)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    n_rows = len(variables_order)
    n_cols = len(timepoints_order)

    if figsize is None:
        figsize = (
            4.5 * n_cols,
            3.2 * n_rows,
        )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
        sharex=False,
    )

    rng = np.random.default_rng(42)

    for row_idx, variable_name in enumerate(variables_order):
        for col_idx, tp in enumerate(timepoints_order):
            ax = axes[row_idx, col_idx]

            d = plot_df[
                (plot_df["variable"] == variable_name)
                & (plot_df["timepoint"] == tp)
            ].copy()

            if d.empty:
                ax.axis("off")
                continue

            data_by_group = [
                d.loc[
                    d["plot_group"] == group,
                    "value",
                ]
                .dropna()
                .to_numpy()
                for group in group_order
            ]

            positions = np.arange(
                len(group_order)
            )

            bp = ax.boxplot(
                data_by_group,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
            )

            for patch, group in zip(
                bp["boxes"],
                group_order,
            ):
                patch.set_facecolor(
                    cluster_colors.get(
                        group,
                        "gray",
                    )
                )

                patch.set_alpha(0.35)

                patch.set_edgecolor(
                    cluster_colors.get(
                        group,
                        "black",
                    )
                )

                patch.set_linewidth(1.6)

            for median in bp["medians"]:
                median.set_color("black")
                median.set_linewidth(1.5)

            # Jittered subject-level points
            for pos, group, values in zip(
                positions,
                group_order,
                data_by_group,
            ):
                if len(values) == 0:
                    continue

                x_jitter = rng.normal(
                    loc=pos,
                    scale=jitter,
                    size=len(values),
                )

                ax.scatter(
                    x_jitter,
                    values,
                    s=point_size,
                    alpha=point_alpha,
                    color=cluster_colors.get(
                        group,
                        "gray",
                    ),
                    edgecolors="white",
                    linewidths=0.3,
                )

            # ----------------------------------------------------------
            # Pairwise results used by both titles and significance bars
            # ----------------------------------------------------------

            pairwise_rows_for_panel = (
                pairwise_stats_lookup.get(
                    (
                        variable_name,
                        tp,
                    ),
                    [],
                )
            )

            significant_brackets = []

            if show_significance_brackets:
                for comparison_index, (
                    group_1,
                    group_2,
                ) in enumerate(pairwise_comparisons_order):
                    matched_row = find_pairwise_row(
                        pairwise_rows_for_panel,
                        group_1,
                        group_2,
                    )

                    if matched_row is None:
                        continue

                    stars = fdr_p_value_to_stars(
                        matched_row["fdr_p_value"]
                    )

                    if stars is None:
                        continue

                    group_1_position = group_order.index(group_1)
                    group_2_position = group_order.index(group_2)

                    significant_brackets.append({
                        "group_1": group_1,
                        "group_2": group_2,
                        "x_1": min(group_1_position, group_2_position),
                        "x_2": max(group_1_position, group_2_position),
                        "stars": stars,
                        "comparison_index": comparison_index,
                    })

                # Draw shorter comparisons first and wider comparisons higher.
                significant_brackets.sort(
                    key=lambda bracket: (
                        bracket["x_2"] - bracket["x_1"],
                        bracket["comparison_index"],
                    )
                )

            # ----------------------------------------------------------
            # Add y-axis headroom for sample-size labels and brackets
            # ----------------------------------------------------------

            y_min_current, y_max_current = ax.get_ylim()

            finite_values = np.concatenate([
                np.asarray(values, dtype=float)
                for values in data_by_group
                if len(values) > 0
            ])

            data_y_min = float(np.nanmin(finite_values))
            data_y_max = float(np.nanmax(finite_values))
            data_y_range = data_y_max - data_y_min

            if (
                not np.isfinite(data_y_range)
                or data_y_range <= 0
            ):
                data_y_range = max(
                    abs(data_y_max) * 0.10,
                    1.0,
                )

            sample_label_offset = 0.03 * data_y_range

            # ----------------------------------------------------------
            # Reserve the lower portion of the plot for data and n labels.
            # The upper portion is reserved for significance brackets.
            # ----------------------------------------------------------

            if significant_brackets:
                data_top_with_label = (
                    data_y_max
                    + sample_label_offset
                    + 0.05 * data_y_range
                )

                # Keep data and sample-size labels within the lower 68%
                # of the plotting area.
                data_top_fraction = 0.68

                required_y_max = (
                    y_min_current
                    + (
                        data_top_with_label
                        - y_min_current
                    )
                    / data_top_fraction
                )

                required_y_max = max(
                    required_y_max,
                    y_max_current,
                )

            else:
                required_y_max = max(
                    y_max_current + 0.14 * data_y_range,
                    data_y_max + 0.14 * data_y_range,
                )

            ax.set_ylim(
                y_min_current,
                required_y_max,
            )

            # ----------------------------------------------------------
            # Sample-size labels
            # ----------------------------------------------------------

            for pos, group, values in zip(
                positions,
                group_order,
                data_by_group,
            ):
                if len(values) == 0:
                    continue

                ax.text(
                    pos,
                    np.nanmax(values) + sample_label_offset,
                    f"n={len(values)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    clip_on=True,
                )

            # ----------------------------------------------------------
            # FDR-corrected significance brackets
            #
            # X coordinates use the group positions.
            # Y coordinates use fractions of the plotting area.
            # This keeps every bracket and asterisk inside the plot box.
            # ----------------------------------------------------------

            if significant_brackets:
                bracket_transform = ax.get_xaxis_transform()

                # ----------------------------------------------------------
                # Sort brackets so narrower comparisons are lower and wider
                # comparisons are higher. This makes spacing look cleaner.
                # ----------------------------------------------------------
                significant_brackets = sorted(
                    significant_brackets,
                    key=lambda b: (
                        b["x_2"] - b["x_1"],
                        b["x_1"],
                    ),
                )

                n_brackets = len(significant_brackets)

                # ----------------------------------------------------------
                # Use evenly spaced rows inside the plot box.
                # Keep everything comfortably below the title area.
                # ----------------------------------------------------------
                if n_brackets == 1:
                    bracket_levels = np.asarray([0.80])
                else:
                    bracket_levels = np.linspace(
                        0.70,
                        0.88,
                        n_brackets,
                    )

                bracket_tick_height = 0.018
                bracket_star_offset = 0.010

                for bracket_y, bracket in zip(
                    bracket_levels,
                    significant_brackets,
                ):
                    bracket_top_y = bracket_y + bracket_tick_height
                    star_y = bracket_top_y + bracket_star_offset

                    # Draw bracket
                    ax.plot(
                        [
                            bracket["x_1"],
                            bracket["x_1"],
                            bracket["x_2"],
                            bracket["x_2"],
                        ],
                        [
                            bracket_y,
                            bracket_top_y,
                            bracket_top_y,
                            bracket_y,
                        ],
                        color="black",
                        linewidth=1.3,
                        transform=bracket_transform,
                        clip_on=True,
                    )

                    # Draw asterisks above bracket
                    ax.text(
                        (bracket["x_1"] + bracket["x_2"]) / 2.0,
                        star_y,
                        bracket["stars"],
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                        color="black",
                        transform=bracket_transform,
                        clip_on=True,
                    )

            # ----------------------------------------------------------
            # X-axis group labels
            # ----------------------------------------------------------

            ax.set_xticks(
                positions
            )

            ax.set_xticklabels(
                [
                    str(group)
                    for group in group_order
                ],
                rotation=0,
                ha="center",
            )


            # ----------------------------------------------------------
            # Panel title and pairwise FDR values
            # ----------------------------------------------------------

            panel_title = timepoint_label_map.get(
                tp,
                tp,
            )

            if show_stats_in_title:
                if len(pairwise_rows_for_panel) > 0:
                    pairwise_labels = []

                    for (
                        group_1,
                        group_2,
                    ) in pairwise_comparisons_order:

                        matched_row = find_pairwise_row(
                            pairwise_rows_for_panel,
                            group_1,
                            group_2,
                        )

                        if matched_row is None:
                            continue

                        pairwise_labels.append(
                            f"{short_group_label(group_1)}-"
                            f"{short_group_label(group_2)}="
                            f"{format_p_value(matched_row['fdr_p_value'])}"
                        )

                    if len(pairwise_labels) > 0:
                        title_rows = []

                        for start_idx in range(
                            0,
                            len(pairwise_labels),
                            pairwise_stats_per_title_row,
                        ):
                            row_labels = pairwise_labels[
                                start_idx:
                                start_idx
                                + pairwise_stats_per_title_row
                            ]

                            if start_idx == 0:
                                title_rows.append(
                                    "FDR: "
                                    + " | ".join(row_labels)
                                )
                            else:
                                title_rows.append(
                                    " | ".join(row_labels)
                                )

                        panel_title += (
                            "\n"
                            + pairwise_stats_line_sep.join(
                                title_rows
                            )
                        )

                else:
                    stat_row = stats_lookup.get(
                        (
                            variable_name,
                            tp,
                        )
                    )

                    if stat_row is not None:
                        if pd.notna(
                            stat_row["p_value"]
                        ):
                            panel_title += (
                                f"\np={stat_row['p_value']:.3g}, "
                                f"FDR={stat_row['fdr_p_value']:.3g}"
                            )
                        else:
                            panel_title += (
                                "\np=NA, FDR=NA"
                            )

            ax.set_title(
                panel_title,
                fontsize=11,
            )

            if col_idx == 0:
                ax.set_ylabel(
                    ylabel or variable_name,
                    fontsize=11,
                )
            else:
                ax.set_ylabel("")

            ax.grid(
                axis="y",
                color="gray",
                alpha=0.18,
                linewidth=0.8,
            )

            ax.set_axisbelow(True)

            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(1.0)

    fig.suptitle(
        title,
        fontsize=14,
        y=suptitle_y,
    )

    plt.tight_layout()

    if show:
        plt.show()

    return (
        plot_df,
        stats_df,
        pairwise_stats_df,
        fig,
        axes,
    )




# def plot_within_label_cluster_characterization(
#     *,
#     result,
#     cfg,
#     labels,
#     target_label=1,
#     variables,
#     variable_order=None,
#     variable_order_col="feature",
#     timepoints=("baseline", "week6", "month6"),
#     timepoint_label_map=None,

#     # Single-timepoint support
#     single_timepoint_df=None,

#     # Plotting mode
#     plot_mode="within_label",  # "within_label" or "td_plus_target"
#     reference_label=0,
#     reference_group_name="TD",
#     target_group_prefix="ASD",

#     # Optional custom cluster/subtype names
#     # Example:
#     # {
#     #     0: "ASD Subtype 1",
#     #     1: "ASD Subtype 2",
#     # }
#     cluster_label_map=None,

#     group_order=None,

#     # Pairwise title statistics
#     pairwise_comparisons_order=None,
#     pairwise_stats_per_title_row=2,
#     pairwise_stats_line_sep="\n",

#     # Styling
#     cluster_colors=None,
#     title="Within-label cluster characterization",
#     ylabel=None,
#     figsize=None,
#     jitter=0.06,
#     point_alpha=0.55,
#     point_size=18,
#     box_width=0.55,
#     show_stats_in_title=True,
#     suptitle_y=1.01,
#     show=True,
# ):
#     """
#     Plot selected feature values by cluster/subtype across one or more timepoints.

#     Modes
#     -----
#     plot_mode="within_label":
#         Keeps only target_label subjects, such as ASD subjects.

#         Default group names:
#             C0, C1, ...

#         Custom names can be supplied with cluster_label_map:
#             {
#                 0: "ASD Subtype 1",
#                 1: "ASD Subtype 2",
#             }

#     plot_mode="td_plus_target":
#         Keeps reference_label and target_label subjects.

#         The reference label becomes one group, such as TD.
#         Target-label subjects are divided according to cluster/subtype.

#         Default target group names:
#             ASD 0, ASD 1, ...

#         Custom target names can be supplied with cluster_label_map:
#             {
#                 0: "ASD Subtype 1",
#                 1: "ASD Subtype 2",
#             }

#     Single-timepoint support
#     ------------------------
#     For a single-timepoint dataset, pass:

#         timepoints=["baseline"]
#         single_timepoint_df=<your dataframe>

#     For a single timepoint, variables can be written in either form:

#         {
#             "feature_1": {
#                 "baseline": "feature_1",
#             }
#         }

#     or:

#         {
#             "feature_1": "feature_1",
#         }

#     Statistics
#     ----------
#     Omnibus:
#         2 groups: Mann-Whitney U
#         3+ groups: Kruskal-Wallis

#     Pairwise:
#         Mann-Whitney U for each requested group pair.
#         Pairwise FDR-adjusted values are shown in panel titles.

#     Returns
#     -------
#     plot_df, stats_df, pairwise_stats_df, fig, axes
#     """

#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     from scipy import stats
#     from itertools import combinations

#     # ------------------------------------------------------------------
#     # Normalize inputs
#     # ------------------------------------------------------------------

#     if isinstance(timepoints, str):
#         timepoints = [timepoints]
#     else:
#         timepoints = list(timepoints)

#     if len(timepoints) == 0:
#         raise ValueError("At least one timepoint must be provided.")

#     if plot_mode not in ["within_label", "td_plus_target"]:
#         raise ValueError(
#             "plot_mode must be either 'within_label' or 'td_plus_target'."
#         )

#     membership_df = result["membership_df"].copy().reset_index(drop=True)

#     labels = pd.Series(
#         labels,
#         name="label",
#     ).reset_index(drop=True)

#     if len(labels) != len(membership_df):
#         raise ValueError(
#             f"labels has length {len(labels)}, but membership_df has "
#             f"{len(membership_df)} rows. Labels must be row-aligned."
#         )

#     # ------------------------------------------------------------------
#     # Obtain the timepoint dataframes
#     # ------------------------------------------------------------------

#     preset_cfg = get_active_fitness_preset_config(cfg)

#     timepoint_cfg = dict(
#         preset_cfg.get("timepoint_config", {}) or {}
#     )

#     timepoint_dfs = dict(
#         timepoint_cfg.get("timepoint_dfs", {}) or {}
#     )

#     # Explicit single-timepoint dataframe takes priority.
#     if single_timepoint_df is not None:
#         if len(timepoints) != 1:
#             raise ValueError(
#                 "single_timepoint_df can only be used when exactly one "
#                 "timepoint is supplied."
#             )

#         single_tp = timepoints[0]

#         timepoint_dfs = {
#             single_tp: single_timepoint_df.copy(),
#         }

#     # Automatic single-timepoint fallback from data_config["df"].
#     elif len(timepoints) == 1 and timepoints[0] not in timepoint_dfs:
#         data_cfg = dict(
#             preset_cfg.get("data_config", {}) or {}
#         )

#         config_df = data_cfg.get("df")

#         if isinstance(config_df, pd.DataFrame):
#             timepoint_dfs = {
#                 timepoints[0]: config_df.copy(),
#             }

#     # Confirm that all requested timepoints are available.
#     missing_timepoints = [
#         tp
#         for tp in timepoints
#         if tp not in timepoint_dfs
#     ]

#     if missing_timepoints:
#         if len(timepoints) == 1:
#             raise KeyError(
#                 f"No dataframe was found for timepoint "
#                 f"{missing_timepoints[0]!r}. For a single-timepoint analysis, "
#                 f"pass single_timepoint_df=<your dataframe>."
#             )

#         raise KeyError(
#             f"timepoint_dfs is missing the following timepoints: "
#             f"{missing_timepoints}"
#         )

#     if timepoint_label_map is None:
#         timepoint_label_map = {
#             tp: tp
#             for tp in timepoints
#         }

#     # ------------------------------------------------------------------
#     # Colors
#     # ------------------------------------------------------------------

#     default_colors = {
#         # Numeric clusters
#         0: "#1587F8",
#         1: "#FFAE17",
#         2: "#049B4F",
#         3: "#C04AE2",

#         # Original within-label names
#         "C0": "#1587F8",
#         "C1": "#FFAE17",
#         "C2": "#049B4F",
#         "C3": "#C04AE2",

#         # Original TD + ASD names
#         "TD": "#7F7F7F",
#         "ASD 0": "#1587F8",
#         "ASD 1": "#FFAE17",
#         "ASD 2": "#049B4F",
#         "ASD 3": "#C04AE2",

#         # New subtype names
#         "ASD Subtype 1": "#1587F8",
#         "ASD Subtype 2": "#FFAE17",
#         "ASD Subtype 3": "#049B4F",
#         "ASD Subtype 4": "#C04AE2",
#     }

#     if cluster_colors is None:
#         cluster_colors = default_colors.copy()
#     else:
#         cluster_colors = {
#             **default_colors,
#             **dict(cluster_colors),
#         }

#     # ------------------------------------------------------------------
#     # Helper: map cluster numbers to display names
#     # ------------------------------------------------------------------

#     def map_cluster_names(cluster_series, mode):
#         cluster_values = pd.to_numeric(
#             cluster_series,
#             errors="coerce",
#         )

#         if cluster_values.isna().any():
#             raise ValueError(
#                 "Cluster assignments contain missing or non-numeric values."
#             )

#         cluster_values = cluster_values.astype(int)

#         if cluster_label_map is not None:
#             mapped_values = cluster_values.map(cluster_label_map)

#             missing_cluster_ids = sorted(
#                 cluster_values[
#                     mapped_values.isna()
#                 ].unique().tolist()
#             )

#             if missing_cluster_ids:
#                 raise ValueError(
#                     "cluster_label_map does not contain names for cluster IDs: "
#                     f"{missing_cluster_ids}"
#                 )

#             return mapped_values.astype(object)

#         if mode == "within_label":
#             return (
#                 "C"
#                 + cluster_values.astype(str)
#             ).astype(object)

#         return (
#             target_group_prefix
#             + " "
#             + cluster_values.astype(str)
#         ).astype(object)

#     # ------------------------------------------------------------------
#     # Build long plotting dataframe
#     # ------------------------------------------------------------------

#     rows = []

#     for tp in timepoints:
#         df_tp = timepoint_dfs[tp].copy().reset_index(drop=True)

#         if len(df_tp) != len(membership_df):
#             raise ValueError(
#                 f"The dataframe for timepoint {tp!r} has {len(df_tp)} rows, "
#                 f"but membership_df has {len(membership_df)} rows. "
#                 f"They must be row-aligned."
#             )

#         cluster_col = f"cluster_{tp}"

#         if cluster_col not in membership_df.columns:
#             raise KeyError(
#                 f"membership_df is missing {cluster_col!r}."
#             )

#         for variable_name, tp_col_map in variables.items():

#             # Longitudinal form:
#             # {
#             #     "feature": {
#             #         "baseline": "feature",
#             #         "week6": "feature",
#             #     }
#             # }
#             if isinstance(tp_col_map, dict):
#                 if tp not in tp_col_map:
#                     continue

#                 value_col = tp_col_map[tp]

#             # Simplified single-timepoint form:
#             # {
#             #     "feature": "feature"
#             # }
#             elif len(timepoints) == 1:
#                 value_col = tp_col_map

#             else:
#                 raise TypeError(
#                     f"Variable {variable_name!r} must map to a dictionary "
#                     f"of timepoint-specific columns when multiple timepoints "
#                     f"are requested."
#                 )

#             if value_col not in df_tp.columns:
#                 raise KeyError(
#                     f"Column {value_col!r} for variable "
#                     f"{variable_name!r} was not found in the dataframe "
#                     f"for timepoint {tp!r}."
#                 )

#             tmp = pd.DataFrame({
#                 "timepoint": tp,
#                 "timepoint_label": timepoint_label_map.get(tp, tp),
#                 "variable": variable_name,
#                 "cluster": membership_df[cluster_col].to_numpy(),
#                 "label": labels.to_numpy(),
#                 "value": pd.to_numeric(
#                     df_tp[value_col],
#                     errors="coerce",
#                 ).to_numpy(),
#             })

#             # ----------------------------------------------------------
#             # Create plotting groups
#             # ----------------------------------------------------------

#             if plot_mode == "within_label":
#                 tmp = tmp[
#                     tmp["label"] == target_label
#                 ].copy()

#                 tmp["plot_group"] = map_cluster_names(
#                     tmp["cluster"],
#                     mode="within_label",
#                 )

#             elif plot_mode == "td_plus_target":
#                 tmp = tmp[
#                     tmp["label"].isin(
#                         [reference_label, target_label]
#                     )
#                 ].copy()

#                 tmp["plot_group"] = pd.Series(
#                     index=tmp.index,
#                     dtype="object",
#                 )

#                 reference_mask = (
#                     tmp["label"] == reference_label
#                 )

#                 target_mask = (
#                     tmp["label"] == target_label
#                 )

#                 tmp.loc[
#                     reference_mask,
#                     "plot_group",
#                 ] = reference_group_name

#                 target_group_names = map_cluster_names(
#                     tmp.loc[target_mask, "cluster"],
#                     mode="td_plus_target",
#                 )

#                 tmp.loc[
#                     target_mask,
#                     "plot_group",
#                 ] = target_group_names.to_numpy()

#             rows.append(tmp)

#     if len(rows) == 0:
#         raise ValueError(
#             "No plotting rows were created. Check variables and timepoints."
#         )

#     plot_df = pd.concat(
#         rows,
#         ignore_index=True,
#     )

#     plot_df = plot_df.dropna(
#         subset=["plot_group", "value"]
#     )

#     if plot_df.empty:
#         raise ValueError(
#             "No valid plotting rows remain after removing missing values."
#         )

#     # ------------------------------------------------------------------
#     # Variable display order
#     # ------------------------------------------------------------------

#     if variable_order is None:
#         variables_order = list(variables.keys())

#     elif isinstance(variable_order, pd.DataFrame):
#         if variable_order_col not in variable_order.columns:
#             raise KeyError(
#                 f"variable_order_col={variable_order_col!r} was not found "
#                 f"in variable_order DataFrame columns: "
#                 f"{list(variable_order.columns)}"
#             )

#         variables_order = (
#             variable_order[variable_order_col]
#             .dropna()
#             .astype(str)
#             .tolist()
#         )

#     else:
#         variables_order = list(variable_order)

#     # Remove duplicate variable names while preserving order.
#     variables_order = list(
#         dict.fromkeys(variables_order)
#     )

#     variables_order = [
#         variable_name
#         for variable_name in variables_order
#         if variable_name in variables
#     ]

#     remaining_variables = [
#         variable_name
#         for variable_name in variables.keys()
#         if variable_name not in variables_order
#     ]

#     variables_order = (
#         variables_order
#         + remaining_variables
#     )

#     if len(variables_order) == 0:
#         raise ValueError(
#             "No variables are available to plot after applying variable_order."
#         )

#     timepoints_order = list(timepoints)

#     # ------------------------------------------------------------------
#     # Group display order
#     # ------------------------------------------------------------------

#     available_groups = list(
#         plot_df["plot_group"]
#         .dropna()
#         .unique()
#     )

#     if group_order is None:

#         if cluster_label_map is not None:
#             mapped_cluster_order = [
#                 group_name
#                 for group_name in cluster_label_map.values()
#                 if group_name in available_groups
#             ]

#             if plot_mode == "td_plus_target":
#                 group_order = mapped_cluster_order.copy()

#                 if reference_group_name in available_groups:
#                     group_order.append(reference_group_name)

#             else:
#                 group_order = mapped_cluster_order.copy()

#         elif plot_mode == "td_plus_target":
#             target_groups = sorted([
#                 group
#                 for group in available_groups
#                 if str(group).startswith(target_group_prefix)
#             ])

#             group_order = target_groups.copy()

#             if reference_group_name in available_groups:
#                 group_order.append(reference_group_name)

#         else:
#             group_order = sorted(available_groups)

#     else:
#         group_order = list(group_order)

#     group_order = [
#         group
#         for group in group_order
#         if group in available_groups
#     ]

#     if len(group_order) < 2:
#         raise ValueError(
#             "At least two plotting groups are required."
#         )

#     # Add colors for any custom group names that were not supplied.
#     fallback_palette = [
#         "#1587F8",
#         "#FFAE17",
#         "#049B4F",
#         "#C04AE2",
#         "#7F7F7F",
#     ]

#     for group_index, group_name in enumerate(group_order):
#         if group_name not in cluster_colors:
#             cluster_colors[group_name] = fallback_palette[
#                 group_index % len(fallback_palette)
#             ]

#     # ------------------------------------------------------------------
#     # Pairwise p-value helpers
#     # ------------------------------------------------------------------

#     def format_p_value(p_value):
#         if pd.isna(p_value):
#             return "NA"

#         return f"{p_value:.3g}"

#     def short_group_label(group_name):
#         group_text = str(group_name)

#         subtype_prefix = "ASD Subtype "

#         if group_text.startswith(subtype_prefix):
#             subtype_number = group_text.replace(
#                 subtype_prefix,
#                 "",
#             )

#             return f"ASD-S{subtype_number}"

#         label_map = {
#             "ASD 0": "ASD0",
#             "ASD 1": "ASD1",
#             "ASD 2": "ASD2",
#             "ASD 3": "ASD3",
#             "TD": "TD",
#             "C0": "C0",
#             "C1": "C1",
#             "C2": "C2",
#             "C3": "C3",
#         }

#         return label_map.get(
#             group_text,
#             group_text.replace(" ", ""),
#         )

#     if pairwise_comparisons_order is None:
#         pairwise_comparisons_order = list(
#             combinations(group_order, 2)
#         )
#     else:
#         pairwise_comparisons_order = list(
#             pairwise_comparisons_order
#         )

#     # Keep only comparisons involving available groups.
#     pairwise_comparisons_order = [
#         (group_1, group_2)
#         for group_1, group_2 in pairwise_comparisons_order
#         if group_1 in group_order
#         and group_2 in group_order
#     ]

#     # ------------------------------------------------------------------
#     # Statistics
#     # ------------------------------------------------------------------

#     stats_rows = []
#     pairwise_rows = []

#     for variable_name in variables_order:
#         for tp in timepoints_order:
#             d = plot_df[
#                 (plot_df["variable"] == variable_name)
#                 & (plot_df["timepoint"] == tp)
#             ].copy()

#             if d.empty:
#                 continue

#             groups = [
#                 d.loc[
#                     d["plot_group"] == group,
#                     "value",
#                 ]
#                 .dropna()
#                 .to_numpy()
#                 for group in group_order
#             ]

#             group_ns = [
#                 len(group_values)
#                 for group_values in groups
#             ]

#             # Omnibus/original-style statistical test
#             if (
#                 len(group_order) == 2
#                 and all(n > 0 for n in group_ns)
#             ):
#                 stat, p_value = stats.mannwhitneyu(
#                     groups[0],
#                     groups[1],
#                     alternative="two-sided",
#                 )

#                 test_name = "Mann-Whitney U"

#             elif (
#                 len(group_order) > 2
#                 and all(n > 0 for n in group_ns)
#             ):
#                 stat, p_value = stats.kruskal(
#                     *groups
#                 )

#                 test_name = "Kruskal-Wallis"

#             else:
#                 stat = np.nan
#                 p_value = np.nan
#                 test_name = "Unavailable"

#             row = {
#                 "variable": variable_name,
#                 "timepoint": tp,
#                 "test": test_name,
#                 "n_groups": len(group_order),
#                 "statistic": stat,
#                 "p_value": p_value,
#             }

#             for group, values in zip(
#                 group_order,
#                 groups,
#             ):
#                 safe_group = (
#                     str(group)
#                     .replace(" ", "_")
#                 )

#                 row[f"{safe_group}_n"] = len(values)

#                 row[f"{safe_group}_mean"] = (
#                     np.mean(values)
#                     if len(values) > 0
#                     else np.nan
#                 )

#                 row[f"{safe_group}_median"] = (
#                     np.median(values)
#                     if len(values) > 0
#                     else np.nan
#                 )

#             stats_rows.append(row)

#             # Pairwise tests
#             for group_1, group_2 in pairwise_comparisons_order:
#                 values_1 = (
#                     d.loc[
#                         d["plot_group"] == group_1,
#                         "value",
#                     ]
#                     .dropna()
#                     .to_numpy()
#                 )

#                 values_2 = (
#                     d.loc[
#                         d["plot_group"] == group_2,
#                         "value",
#                     ]
#                     .dropna()
#                     .to_numpy()
#                 )

#                 if (
#                     len(values_1) > 0
#                     and len(values_2) > 0
#                 ):
#                     pairwise_result = stats.mannwhitneyu(
#                         values_1,
#                         values_2,
#                         alternative="two-sided",
#                     )

#                     pairwise_stat = (
#                         pairwise_result.statistic
#                     )

#                     pairwise_p = (
#                         pairwise_result.pvalue
#                     )

#                     pairwise_test = "Mann-Whitney U"

#                 else:
#                     pairwise_stat = np.nan
#                     pairwise_p = np.nan
#                     pairwise_test = (
#                         "Mann-Whitney U unavailable"
#                     )

#                 pairwise_rows.append({
#                     "variable": variable_name,
#                     "timepoint": tp,
#                     "group_1": group_1,
#                     "group_2": group_2,
#                     "comparison": (
#                         f"{group_1} vs {group_2}"
#                     ),
#                     "test": pairwise_test,
#                     "statistic": pairwise_stat,
#                     "p_value": pairwise_p,
#                     "n_group_1": len(values_1),
#                     "n_group_2": len(values_2),
#                     "median_group_1": (
#                         np.nanmedian(values_1)
#                         if len(values_1) > 0
#                         else np.nan
#                     ),
#                     "median_group_2": (
#                         np.nanmedian(values_2)
#                         if len(values_2) > 0
#                         else np.nan
#                     ),
#                     "mean_group_1": (
#                         np.nanmean(values_1)
#                         if len(values_1) > 0
#                         else np.nan
#                     ),
#                     "mean_group_2": (
#                         np.nanmean(values_2)
#                         if len(values_2) > 0
#                         else np.nan
#                     ),
#                 })

#     stats_df = pd.DataFrame(
#         stats_rows
#     )

#     pairwise_stats_df = pd.DataFrame(
#         pairwise_rows
#     )

#     # ------------------------------------------------------------------
#     # Benjamini-Hochberg FDR helper
#     # ------------------------------------------------------------------

#     def add_fdr_column(df, p_col="p_value"):
#         df = df.copy()
#         df["fdr_p_value"] = np.nan

#         if df.empty or p_col not in df.columns:
#             return df

#         valid_mask = df[p_col].notna()

#         if valid_mask.sum() == 0:
#             return df

#         p_values = (
#             df.loc[valid_mask, p_col]
#             .to_numpy(dtype=float)
#         )

#         order = np.argsort(p_values)
#         ranked_p = p_values[order]
#         m = len(p_values)

#         q_values = (
#             ranked_p
#             * m
#             / (np.arange(m) + 1)
#         )

#         q_values = np.minimum.accumulate(
#             q_values[::-1]
#         )[::-1]

#         q_values = np.clip(
#             q_values,
#             0,
#             1,
#         )

#         fdr_values = np.empty_like(
#             q_values
#         )

#         fdr_values[order] = q_values

#         df.loc[
#             valid_mask,
#             "fdr_p_value",
#         ] = fdr_values

#         return df

#     stats_df = add_fdr_column(
#         stats_df
#     )

#     pairwise_stats_df = add_fdr_column(
#         pairwise_stats_df
#     )

#     # ------------------------------------------------------------------
#     # Statistics lookup tables
#     # ------------------------------------------------------------------

#     stats_lookup = {}

#     if not stats_df.empty:
#         for _, row in stats_df.iterrows():
#             stats_lookup[
#                 (
#                     row["variable"],
#                     row["timepoint"],
#                 )
#             ] = row

#     pairwise_stats_lookup = {}

#     if not pairwise_stats_df.empty:
#         for _, row in pairwise_stats_df.iterrows():
#             key = (
#                 row["variable"],
#                 row["timepoint"],
#             )

#             pairwise_stats_lookup.setdefault(
#                 key,
#                 [],
#             ).append(row)

#     # ------------------------------------------------------------------
#     # Plot
#     # ------------------------------------------------------------------

#     n_rows = len(variables_order)
#     n_cols = len(timepoints_order)

#     if figsize is None:
#         figsize = (
#             4.5 * n_cols,
#             3.2 * n_rows,
#         )

#     fig, axes = plt.subplots(
#         n_rows,
#         n_cols,
#         figsize=figsize,
#         squeeze=False,
#         sharex=False,
#     )

#     rng = np.random.default_rng(42)

#     for row_idx, variable_name in enumerate(variables_order):
#         for col_idx, tp in enumerate(timepoints_order):
#             ax = axes[row_idx, col_idx]

#             d = plot_df[
#                 (plot_df["variable"] == variable_name)
#                 & (plot_df["timepoint"] == tp)
#             ].copy()

#             if d.empty:
#                 ax.axis("off")
#                 continue

#             data_by_group = [
#                 d.loc[
#                     d["plot_group"] == group,
#                     "value",
#                 ]
#                 .dropna()
#                 .to_numpy()
#                 for group in group_order
#             ]

#             positions = np.arange(
#                 len(group_order)
#             )

#             bp = ax.boxplot(
#                 data_by_group,
#                 positions=positions,
#                 widths=box_width,
#                 patch_artist=True,
#                 showfliers=False,
#             )

#             for patch, group in zip(
#                 bp["boxes"],
#                 group_order,
#             ):
#                 patch.set_facecolor(
#                     cluster_colors.get(
#                         group,
#                         "gray",
#                     )
#                 )

#                 patch.set_alpha(0.35)

#                 patch.set_edgecolor(
#                     cluster_colors.get(
#                         group,
#                         "black",
#                     )
#                 )

#                 patch.set_linewidth(1.6)

#             for median in bp["medians"]:
#                 median.set_color("black")
#                 median.set_linewidth(1.5)

#             # Jittered subject-level points
#             for pos, group, values in zip(
#                 positions,
#                 group_order,
#                 data_by_group,
#             ):
#                 if len(values) == 0:
#                     continue

#                 x_jitter = rng.normal(
#                     loc=pos,
#                     scale=jitter,
#                     size=len(values),
#                 )

#                 ax.scatter(
#                     x_jitter,
#                     values,
#                     s=point_size,
#                     alpha=point_alpha,
#                     color=cluster_colors.get(
#                         group,
#                         "gray",
#                     ),
#                     edgecolors="white",
#                     linewidths=0.3,
#                 )

#             # Add y-axis headroom for sample-size labels.
#             y_min_current, y_max_current = ax.get_ylim()

#             y_range_current = (
#                 y_max_current
#                 - y_min_current
#             )

#             if (
#                 not np.isfinite(y_range_current)
#                 or y_range_current <= 0
#             ):
#                 y_range_current = 1.0

#             ax.set_ylim(
#                 y_min_current,
#                 y_max_current
#                 + 0.14 * y_range_current,
#             )

#             y_min_current, y_max_current = ax.get_ylim()

#             y_range_current = (
#                 y_max_current
#                 - y_min_current
#             )

#             for pos, group, values in zip(
#                 positions,
#                 group_order,
#                 data_by_group,
#             ):
#                 if len(values) == 0:
#                     continue

#                 ax.text(
#                     pos,
#                     np.nanmax(values)
#                     + 0.03 * y_range_current,
#                     f"n={len(values)}",
#                     ha="center",
#                     va="bottom",
#                     fontsize=9,
#                     fontweight="bold",
#                     clip_on=False,
#                 )

#             ax.set_xticks(
#                 positions
#             )

#             ax.set_xticklabels(
#                 [
#                     str(group)
#                     for group in group_order
#                 ],
#                 rotation=0,
#                 ha="center",
#             )

#             # ----------------------------------------------------------
#             # Panel title and pairwise FDR values
#             # ----------------------------------------------------------

#             panel_title = timepoint_label_map.get(
#                 tp,
#                 tp,
#             )

#             if show_stats_in_title:
#                 pairwise_rows_for_panel = (
#                     pairwise_stats_lookup.get(
#                         (
#                             variable_name,
#                             tp,
#                         ),
#                         [],
#                     )
#                 )

#                 if len(pairwise_rows_for_panel) > 0:
#                     pairwise_labels = []

#                     for (
#                         group_1,
#                         group_2,
#                     ) in pairwise_comparisons_order:

#                         matched_row = None

#                         for row in pairwise_rows_for_panel:
#                             if (
#                                 row["group_1"] == group_1
#                                 and row["group_2"] == group_2
#                             ) or (
#                                 row["group_1"] == group_2
#                                 and row["group_2"] == group_1
#                             ):
#                                 matched_row = row
#                                 break

#                         if matched_row is None:
#                             continue

#                         pairwise_labels.append(
#                             f"{short_group_label(group_1)}-"
#                             f"{short_group_label(group_2)}="
#                             f"{format_p_value(matched_row['fdr_p_value'])}"
#                         )

#                     if len(pairwise_labels) > 0:
#                         title_rows = []

#                         for start_idx in range(
#                             0,
#                             len(pairwise_labels),
#                             pairwise_stats_per_title_row,
#                         ):
#                             row_labels = pairwise_labels[
#                                 start_idx:
#                                 start_idx
#                                 + pairwise_stats_per_title_row
#                             ]

#                             if start_idx == 0:
#                                 title_rows.append(
#                                     "FDR: "
#                                     + " | ".join(row_labels)
#                                 )
#                             else:
#                                 title_rows.append(
#                                     " | ".join(row_labels)
#                                 )

#                         panel_title += (
#                             "\n"
#                             + pairwise_stats_line_sep.join(
#                                 title_rows
#                             )
#                         )

#                 else:
#                     stat_row = stats_lookup.get(
#                         (
#                             variable_name,
#                             tp,
#                         )
#                     )

#                     if stat_row is not None:
#                         if pd.notna(
#                             stat_row["p_value"]
#                         ):
#                             panel_title += (
#                                 f"\np={stat_row['p_value']:.3g}, "
#                                 f"FDR={stat_row['fdr_p_value']:.3g}"
#                             )
#                         else:
#                             panel_title += (
#                                 "\np=NA, FDR=NA"
#                             )

#             ax.set_title(
#                 panel_title,
#                 fontsize=11,
#             )

#             if col_idx == 0:
#                 ax.set_ylabel(
#                     ylabel or variable_name,
#                     fontsize=11,
#                 )
#             else:
#                 ax.set_ylabel("")

#             # Variable label inside each subplot row
#             if n_rows > 1:
#                 ax.text(
#                     0.01,
#                     0.98,
#                     variable_name,
#                     transform=ax.transAxes,
#                     ha="left",
#                     va="top",
#                     fontsize=10,
#                     fontweight="bold",
#                     bbox=dict(
#                         facecolor="white",
#                         edgecolor="none",
#                         alpha=0.8,
#                     ),
#                 )

#             ax.grid(
#                 axis="y",
#                 color="gray",
#                 alpha=0.18,
#                 linewidth=0.8,
#             )

#             ax.set_axisbelow(True)

#             for spine in ax.spines.values():
#                 spine.set_color("black")
#                 spine.set_linewidth(1.0)

#     fig.suptitle(
#         title,
#         fontsize=14,
#         y=suptitle_y,
#     )

#     plt.tight_layout()

#     if show:
#         plt.show()

#     return (
#         plot_df,
#         stats_df,
#         pairwise_stats_df,
#         fig,
#         axes,
#     )


# def plot_within_label_cluster_characterization(
#     *,
#     result,
#     cfg,
#     labels,
#     target_label=1,
#     variables,
#     variable_order=None,
#     variable_order_col="feature",
#     timepoints=("baseline", "week6", "month6"),
#     timepoint_label_map=None,

#     # New: plotting mode
#     plot_mode="within_label",  # "within_label" or "td_plus_target"
#     reference_label=0,
#     reference_group_name="TD",
#     target_group_prefix="ASD",
#     group_order=None,

#     # Pairwise title statistics
#     pairwise_comparisons_order=None,
#     pairwise_stats_per_title_row=2,
#     pairwise_stats_line_sep="\n",

#     # Styling
#     cluster_colors=None,
#     title="Within-label cluster characterization",
#     ylabel=None,
#     figsize=None,
#     jitter=0.06,
#     point_alpha=0.55,
#     point_size=18,
#     box_width=0.55,
#     show_stats_in_title=True,
#     suptitle_y=1.01,
#     show=True,
# ):
#     """
#     Plot selected feature values by cluster/subtype across timepoints.

#     Modes
#     -----
#     plot_mode="within_label":
#         Original behavior.
#         Keeps only target_label subjects, e.g. ASD only.
#         Groups are C0, C1, ...

#     plot_mode="td_plus_target":
#         Keeps reference_label and target_label subjects.
#         Reference label becomes one group, e.g. TD.
#         Target label subjects are split by cluster, e.g. ASD 0, ASD 1.

#     Statistics
#     ----------
#     Omnibus:
#         2 groups: Mann-Whitney U
#         3+ groups: Kruskal-Wallis

#     Pairwise:
#         Mann-Whitney U for each requested group pair.
#         Pairwise FDR values are shown in panel titles.
#     """

#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     from scipy import stats

#     membership_df = result["membership_df"].copy()

#     preset_cfg = get_active_fitness_preset_config(cfg)
#     timepoint_cfg = dict(preset_cfg.get("timepoint_config", {}) or {})
#     timepoint_dfs = dict(timepoint_cfg.get("timepoint_dfs", {}) or {})

#     labels = pd.Series(labels, name="label").reset_index(drop=True)

#     if len(labels) != len(membership_df):
#         raise ValueError(
#             f"labels has length {len(labels)}, but membership_df has "
#             f"{len(membership_df)} rows. Labels must be row-aligned."
#         )

#     if plot_mode not in ["within_label", "td_plus_target"]:
#         raise ValueError("plot_mode must be 'within_label' or 'td_plus_target'.")

#     if timepoint_label_map is None:
#         timepoint_label_map = {tp: tp for tp in timepoints}

#     if cluster_colors is None:
#         cluster_colors = {
#             # Old numeric cluster colors
#             0: "#1587F8",
#             1: "#FFAE17",
#             2: "#049B4F",
#             3: "#C04AE2",

#             # Old C-label colors
#             "C0": "#1587F8",
#             "C1": "#FFAE17",
#             "C2": "#049B4F",
#             "C3": "#C04AE2",

#             # TD + ASD labels
#             "TD": "#7F7F7F",
#             "ASD 0": "#1587F8",
#             "ASD 1": "#FFAE17",
#             "ASD 2": "#049B4F",
#             "ASD 3": "#C04AE2",
#         }

#     # ------------------------------------------------------------------
#     # Build long dataframe
#     # ------------------------------------------------------------------

#     rows = []

#     for tp in timepoints:
#         if tp not in timepoint_dfs:
#             raise KeyError(f"timepoint_dfs is missing timepoint {tp!r}.")

#         df_tp = timepoint_dfs[tp]
#         cluster_col = f"cluster_{tp}"

#         if cluster_col not in membership_df.columns:
#             raise KeyError(f"membership_df is missing {cluster_col!r}.")

#         for variable_name, tp_col_map in variables.items():
#             if tp not in tp_col_map:
#                 continue

#             value_col = tp_col_map[tp]

#             if value_col not in df_tp.columns:
#                 raise KeyError(
#                     f"Column {value_col!r} for variable {variable_name!r} "
#                     f"not found in dataframe for timepoint {tp!r}."
#                 )

#             tmp = pd.DataFrame({
#                 "timepoint": tp,
#                 "timepoint_label": timepoint_label_map.get(tp, tp),
#                 "variable": variable_name,
#                 "cluster": membership_df[cluster_col].to_numpy(),
#                 "label": labels.to_numpy(),
#                 "value": pd.to_numeric(df_tp[value_col], errors="coerce").to_numpy(),
#             })

#             # ----------------------------------------------------------
#             # Create plotting groups
#             # ----------------------------------------------------------
#             if plot_mode == "within_label":
#                 tmp = tmp[tmp["label"] == target_label].copy()

#                 # Avoid pandas dtype warning by explicitly creating object dtype.
#                 tmp["plot_group"] = pd.Series(index=tmp.index, dtype="object")
#                 tmp["plot_group"] = "C" + tmp["cluster"].astype(int).astype(str)

#             elif plot_mode == "td_plus_target":
#                 tmp = tmp[tmp["label"].isin([reference_label, target_label])].copy()

#                 # Important: object dtype avoids FutureWarning when assigning strings.
#                 tmp["plot_group"] = pd.Series(index=tmp.index, dtype="object")

#                 reference_mask = tmp["label"] == reference_label
#                 target_mask = tmp["label"] == target_label

#                 tmp.loc[reference_mask, "plot_group"] = reference_group_name

#                 tmp.loc[target_mask, "plot_group"] = (
#                     target_group_prefix
#                     + " "
#                     + tmp.loc[target_mask, "cluster"].astype(int).astype(str)
#                 )

#             rows.append(tmp)

#     if len(rows) == 0:
#         raise ValueError("No plotting rows were created. Check variables and timepoints.")

#     plot_df = pd.concat(rows, ignore_index=True)
#     plot_df = plot_df.dropna(subset=["plot_group", "value"])

#     # ------------------------------------------------------------------
#     # Variable display order
#     # ------------------------------------------------------------------

#     if variable_order is None:
#         variables_order = list(variables.keys())

#     elif isinstance(variable_order, pd.DataFrame):
#         if variable_order_col not in variable_order.columns:
#             raise KeyError(
#                 f"variable_order_col={variable_order_col!r} was not found in "
#                 f"variable_order DataFrame columns: {list(variable_order.columns)}"
#             )

#         variables_order = (
#             variable_order[variable_order_col]
#             .dropna()
#             .astype(str)
#             .tolist()
#         )

#     else:
#         variables_order = list(variable_order)

#     variables_order = [v for v in variables_order if v in variables]
#     remaining_variables = [v for v in variables.keys() if v not in variables_order]
#     variables_order = variables_order + remaining_variables

#     if len(variables_order) == 0:
#         raise ValueError("No variables available to plot after applying variable_order.")

#     timepoints_order = list(timepoints)

#     # ------------------------------------------------------------------
#     # Group display order
#     # ------------------------------------------------------------------

#     available_groups = list(plot_df["plot_group"].dropna().unique())

#     if group_order is None:
#         if plot_mode == "td_plus_target":
#             asd_groups = sorted([
#                 group for group in available_groups
#                 if str(group).startswith(target_group_prefix)
#             ])
#             group_order = asd_groups + [
#                 group for group in [reference_group_name]
#                 if group in available_groups
#             ]
#         else:
#             group_order = sorted(available_groups)

#     group_order = [group for group in group_order if group in available_groups]

#     if len(group_order) < 2:
#         raise ValueError("At least two plotting groups are required.")

#     # ------------------------------------------------------------------
#     # Pairwise p-value helpers
#     # ------------------------------------------------------------------

#     def format_p_value(p_value):
#         if pd.isna(p_value):
#             return "NA"
#         return f"{p_value:.3g}"

#     def short_group_label(group_name):
#         label_map = {
#             "ASD 0": "ASD0",
#             "ASD 1": "ASD1",
#             "ASD 2": "ASD2",
#             "ASD 3": "ASD3",
#             "TD": "TD",
#             "C0": "C0",
#             "C1": "C1",
#             "C2": "C2",
#             "C3": "C3",
#         }
#         return label_map.get(str(group_name), str(group_name).replace(" ", ""))

#     if pairwise_comparisons_order is None:
#         if plot_mode == "td_plus_target":
#             pairwise_comparisons_order = [
#                 ("TD", "ASD 0"),
#                 ("TD", "ASD 1"),
#                 ("ASD 0", "ASD 1"),
#             ]
#         else:
#             pairwise_comparisons_order = [
#                 ("C0", "C1"),
#             ]

#     # ------------------------------------------------------------------
#     # Statistics
#     # ------------------------------------------------------------------

#     stats_rows = []
#     pairwise_rows = []

#     for variable_name in variables_order:
#         for tp in timepoints_order:
#             d = plot_df[
#                 (plot_df["variable"] == variable_name)
#                 & (plot_df["timepoint"] == tp)
#             ].copy()

#             if d.empty:
#                 continue

#             groups = [
#                 d.loc[d["plot_group"] == group, "value"].dropna().to_numpy()
#                 for group in group_order
#             ]

#             group_ns = [len(group_values) for group_values in groups]

#             # Omnibus / original-style test
#             if len(group_order) == 2 and all(n > 0 for n in group_ns):
#                 stat, p_value = stats.mannwhitneyu(
#                     groups[0],
#                     groups[1],
#                     alternative="two-sided",
#                 )
#                 test_name = "Mann-Whitney U"

#             elif len(group_order) > 2 and all(n > 0 for n in group_ns):
#                 stat, p_value = stats.kruskal(*groups)
#                 test_name = "Kruskal-Wallis"

#             else:
#                 stat = np.nan
#                 p_value = np.nan
#                 test_name = "Unavailable"

#             row = {
#                 "variable": variable_name,
#                 "timepoint": tp,
#                 "test": test_name,
#                 "n_groups": len(group_order),
#                 "statistic": stat,
#                 "p_value": p_value,
#             }

#             for group, values in zip(group_order, groups):
#                 safe_group = str(group).replace(" ", "_")
#                 row[f"{safe_group}_n"] = len(values)
#                 row[f"{safe_group}_mean"] = np.mean(values) if len(values) > 0 else np.nan
#                 row[f"{safe_group}_median"] = np.median(values) if len(values) > 0 else np.nan

#             stats_rows.append(row)

#             # Pairwise tests for title display
#             for group_1, group_2 in pairwise_comparisons_order:
#                 values_1 = d.loc[d["plot_group"] == group_1, "value"].dropna().to_numpy()
#                 values_2 = d.loc[d["plot_group"] == group_2, "value"].dropna().to_numpy()

#                 if len(values_1) > 0 and len(values_2) > 0:
#                     pairwise_result = stats.mannwhitneyu(
#                         values_1,
#                         values_2,
#                         alternative="two-sided",
#                     )
#                     pairwise_stat = pairwise_result.statistic
#                     pairwise_p = pairwise_result.pvalue
#                     pairwise_test = "Mann-Whitney U"
#                 else:
#                     pairwise_stat = np.nan
#                     pairwise_p = np.nan
#                     pairwise_test = "Mann-Whitney U unavailable"

#                 pairwise_rows.append({
#                     "variable": variable_name,
#                     "timepoint": tp,
#                     "group_1": group_1,
#                     "group_2": group_2,
#                     "comparison": f"{group_1} vs {group_2}",
#                     "test": pairwise_test,
#                     "statistic": pairwise_stat,
#                     "p_value": pairwise_p,
#                     "n_group_1": len(values_1),
#                     "n_group_2": len(values_2),
#                     "median_group_1": np.nanmedian(values_1) if len(values_1) > 0 else np.nan,
#                     "median_group_2": np.nanmedian(values_2) if len(values_2) > 0 else np.nan,
#                     "mean_group_1": np.nanmean(values_1) if len(values_1) > 0 else np.nan,
#                     "mean_group_2": np.nanmean(values_2) if len(values_2) > 0 else np.nan,
#                 })

#     stats_df = pd.DataFrame(stats_rows)
#     pairwise_stats_df = pd.DataFrame(pairwise_rows)

#     # FDR correction for omnibus stats
#     if not stats_df.empty:
#         valid_mask = stats_df["p_value"].notna()
#         stats_df["fdr_p_value"] = np.nan

#         if valid_mask.sum() > 0:
#             p = stats_df.loc[valid_mask, "p_value"].to_numpy()
#             order = np.argsort(p)
#             ranked_p = p[order]
#             m = len(p)

#             q = ranked_p * m / (np.arange(m) + 1)
#             q = np.minimum.accumulate(q[::-1])[::-1]
#             q = np.clip(q, 0, 1)

#             fdr = np.empty_like(q)
#             fdr[order] = q
#             stats_df.loc[valid_mask, "fdr_p_value"] = fdr

#     # FDR correction for pairwise stats
#     if not pairwise_stats_df.empty:
#         valid_mask = pairwise_stats_df["p_value"].notna()
#         pairwise_stats_df["fdr_p_value"] = np.nan

#         if valid_mask.sum() > 0:
#             p = pairwise_stats_df.loc[valid_mask, "p_value"].to_numpy()
#             order = np.argsort(p)
#             ranked_p = p[order]
#             m = len(p)

#             q = ranked_p * m / (np.arange(m) + 1)
#             q = np.minimum.accumulate(q[::-1])[::-1]
#             q = np.clip(q, 0, 1)

#             fdr = np.empty_like(q)
#             fdr[order] = q
#             pairwise_stats_df.loc[valid_mask, "fdr_p_value"] = fdr

#     stats_lookup = {}
#     if not stats_df.empty:
#         for _, row in stats_df.iterrows():
#             stats_lookup[(row["variable"], row["timepoint"])] = row

#     pairwise_stats_lookup = {}
#     if not pairwise_stats_df.empty:
#         for _, row in pairwise_stats_df.iterrows():
#             key = (row["variable"], row["timepoint"])
#             pairwise_stats_lookup.setdefault(key, []).append(row)

#     # ------------------------------------------------------------------
#     # Plot
#     # ------------------------------------------------------------------

#     n_rows = len(variables_order)
#     n_cols = len(timepoints_order)

#     if figsize is None:
#         figsize = (4.2 * n_cols, 3.0 * n_rows)

#     fig, axes = plt.subplots(
#         n_rows,
#         n_cols,
#         figsize=figsize,
#         squeeze=False,
#         sharex=False,
#     )

#     rng = np.random.default_rng(42)

#     for row_idx, variable_name in enumerate(variables_order):
#         for col_idx, tp in enumerate(timepoints_order):
#             ax = axes[row_idx, col_idx]

#             d = plot_df[
#                 (plot_df["variable"] == variable_name)
#                 & (plot_df["timepoint"] == tp)
#             ].copy()

#             if d.empty:
#                 ax.axis("off")
#                 continue

#             data_by_group = [
#                 d.loc[d["plot_group"] == group, "value"].dropna().to_numpy()
#                 for group in group_order
#             ]

#             positions = np.arange(len(group_order))

#             bp = ax.boxplot(
#                 data_by_group,
#                 positions=positions,
#                 widths=box_width,
#                 patch_artist=True,
#                 showfliers=False,
#             )

#             for patch, group in zip(bp["boxes"], group_order):
#                 patch.set_facecolor(cluster_colors.get(group, "gray"))
#                 patch.set_alpha(0.35)
#                 patch.set_edgecolor(cluster_colors.get(group, "black"))
#                 patch.set_linewidth(1.6)

#             for median in bp["medians"]:
#                 median.set_color("black")
#                 median.set_linewidth(1.5)

#             # Jittered points
#             for pos, group, values in zip(positions, group_order, data_by_group):
#                 if len(values) == 0:
#                     continue

#                 x_jitter = rng.normal(loc=pos, scale=jitter, size=len(values))

#                 ax.scatter(
#                     x_jitter,
#                     values,
#                     s=point_size,
#                     alpha=point_alpha,
#                     color=cluster_colors.get(group, "gray"),
#                     edgecolors="white",
#                     linewidths=0.3,
#                 )

#             # Add y-axis headroom for n labels
#             y_min_current, y_max_current = ax.get_ylim()
#             y_range_current = y_max_current - y_min_current
#             ax.set_ylim(y_min_current, y_max_current + 0.14 * y_range_current)

#             y_min_current, y_max_current = ax.get_ylim()
#             y_range_current = y_max_current - y_min_current

#             for pos, group, values in zip(positions, group_order, data_by_group):
#                 if len(values) == 0:
#                     continue

#                 ax.text(
#                     pos,
#                     np.nanmax(values) + 0.03 * y_range_current,
#                     f"n={len(values)}",
#                     ha="center",
#                     va="bottom",
#                     fontsize=9,
#                     fontweight="bold",
#                     clip_on=False,
#                 )

#             ax.set_xticks(positions)
#             ax.set_xticklabels(
#                 [str(group) for group in group_order],
#                 rotation=0,
#                 ha="center",
#             )

#             # Panel title with pairwise FDR values
#             panel_title = timepoint_label_map.get(tp, tp)

#             if show_stats_in_title:
#                 pairwise_rows_for_panel = pairwise_stats_lookup.get(
#                     (variable_name, tp),
#                     [],
#                 )

#                 if len(pairwise_rows_for_panel) > 0:
#                     pairwise_labels = []

#                     for group_1, group_2 in pairwise_comparisons_order:
#                         matched_row = None

#                         for row in pairwise_rows_for_panel:
#                             if (
#                                 row["group_1"] == group_1
#                                 and row["group_2"] == group_2
#                             ) or (
#                                 row["group_1"] == group_2
#                                 and row["group_2"] == group_1
#                             ):
#                                 matched_row = row
#                                 break

#                         if matched_row is None:
#                             continue

#                         pairwise_labels.append(
#                             f"{short_group_label(group_1)}-{short_group_label(group_2)}="
#                             f"{format_p_value(matched_row['fdr_p_value'])}"
#                         )

#                     if len(pairwise_labels) > 0:
#                         title_rows = []

#                         for start_idx in range(
#                             0,
#                             len(pairwise_labels),
#                             pairwise_stats_per_title_row,
#                         ):
#                             row_labels = pairwise_labels[
#                                 start_idx:start_idx + pairwise_stats_per_title_row
#                             ]

#                             if start_idx == 0:
#                                 title_rows.append("FDR: " + " | ".join(row_labels))
#                             else:
#                                 title_rows.append(" | ".join(row_labels))

#                         panel_title += "\n" + pairwise_stats_line_sep.join(title_rows)

#                 else:
#                     stat_row = stats_lookup.get((variable_name, tp))

#                     if stat_row is not None:
#                         if pd.notna(stat_row["p_value"]):
#                             panel_title += (
#                                 f"\np={stat_row['p_value']:.3g}, "
#                                 f"FDR={stat_row['fdr_p_value']:.3g}"
#                             )
#                         else:
#                             panel_title += "\np=NA, FDR=NA"

#             ax.set_title(panel_title, fontsize=11)

#             if col_idx == 0:
#                 ax.set_ylabel(ylabel or variable_name, fontsize=11)
#             else:
#                 ax.set_ylabel("")

#             # Variable label inside each subplot row
#             if n_rows > 1:
#                 ax.text(
#                     0.01,
#                     0.98,
#                     variable_name,
#                     transform=ax.transAxes,
#                     ha="left",
#                     va="top",
#                     fontsize=10,
#                     fontweight="bold",
#                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
#                 )

#             ax.grid(axis="y", color="gray", alpha=0.18, linewidth=0.8)
#             ax.set_axisbelow(True)

#             for spine in ax.spines.values():
#                 spine.set_color("black")
#                 spine.set_linewidth(1.0)

#     fig.suptitle(title, fontsize=14, y=suptitle_y)
#     plt.tight_layout()

#     if show:
#         plt.show()

#     return plot_df, stats_df, pairwise_stats_df, fig, axes


def plot_cluster_stratified_longitudinal_icc(
    *,
    result: Dict[str, Any],
    cfg: ClinicalResponseGAFSConfig,
    labels: Sequence[Any],
    target_label: Any = 1,
    variables: Mapping[str, Mapping[str, str]],
    variable_order: Optional[
        Union[Sequence[str], pd.DataFrame]
    ] = None,
    variable_order_col: str = "feature",
    timepoints: Sequence[str] = (
        "baseline",
        "week6",
        "month6",
    ),
    cluster_timepoint: str = "baseline",

    # Cluster/subtype display settings
    cluster_colors: Optional[Dict[Any, str]] = None,
    cluster_label_map: Optional[Dict[Any, str]] = None,

    title: str = "ICC-based longitudinal stability by cluster",
    ylabel: str = "ICC",
    icc_type: str = "consistency",
    figsize: Optional[Tuple[float, float]] = None,
    bar_width: float = 0.34,
    annotate_bars: bool = True,
    y_lim: Tuple[float, float] = (-0.05, 1.0),
    font_size: float = 12.0,
    x_tick_rotation: float = 45,
    x_tick_ha: str = "right",
    show_legend: bool = True,
    legend_loc: str = "best",
    legend_bbox_to_anchor: Optional[
        Tuple[float, float]
    ] = None,
    legend_title: Optional[str] = None,
    show: bool = True,
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """
    Plot ICC-based longitudinal stability by baseline-defined cluster
    or subtype.

    Main question
    -------------
    Within target-label subjects and within each cluster defined at
    cluster_timepoint:

        If a subject has a high feature value at Baseline, does that
        subject tend to remain high at Week 6 and Month 6?

    Grouping
    --------
    Subjects are first filtered to target_label, usually Diagnosis = 1.

    They are then grouped by their cluster assignment at
    cluster_timepoint, usually Baseline.

    Cluster display labels
    ----------------------
    cluster_label_map can be used to replace internal cluster numbers
    with presentation-ready subtype names.

    Example:

        cluster_label_map={
            0: "Subtype A",
            1: "Subtype B",
        }

    ICC interpretation
    ------------------
    Higher ICC:
        More stable subject-level feature values or ranking over time.

    Lower ICC:
        Less stable feature values or ranking over time.

    icc_type
    --------
    "consistency":
        Evaluates whether subjects maintain their relative ranking
        across timepoints.

    "absolute":
        Evaluates whether the actual feature values remain similar.
        This is stricter because it penalizes systematic timepoint
        shifts.
    """

    # ==================================================================
    # Helper: compute ICC from a wide subject-by-timepoint dataframe
    # ==================================================================

    def _compute_icc_from_wide(
        wide_df: pd.DataFrame,
        icc_type: str = "consistency",
    ) -> Dict[str, Any]:
        """
        Compute ICC from a wide matrix.

        Rows:
            Subjects

        Columns:
            Timepoints
        """

        x = (
            wide_df
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )

        n_subjects, n_timepoints = x.shape

        if n_subjects < 2 or n_timepoints < 2:
            return {
                "icc": np.nan,
                "n_subjects": n_subjects,
                "n_timepoints": n_timepoints,
                "ms_subject": np.nan,
                "ms_timepoint": np.nan,
                "ms_error": np.nan,
            }

        grand_mean = np.mean(x)

        subject_means = np.mean(
            x,
            axis=1,
            keepdims=True,
        )

        timepoint_means = np.mean(
            x,
            axis=0,
            keepdims=True,
        )

        ss_subject = (
            n_timepoints
            * np.sum(
                (subject_means - grand_mean) ** 2
            )
        )

        ss_timepoint = (
            n_subjects
            * np.sum(
                (timepoint_means - grand_mean) ** 2
            )
        )

        ss_error = np.sum(
            (
                x
                - subject_means
                - timepoint_means
                + grand_mean
            ) ** 2
        )

        df_subject = n_subjects - 1
        df_timepoint = n_timepoints - 1
        df_error = df_subject * df_timepoint

        ms_subject = (
            ss_subject / df_subject
            if df_subject > 0
            else np.nan
        )

        ms_timepoint = (
            ss_timepoint / df_timepoint
            if df_timepoint > 0
            else np.nan
        )

        ms_error = (
            ss_error / df_error
            if df_error > 0
            else np.nan
        )

        icc_key = str(icc_type).lower()

        if icc_key == "consistency":
            # Consistency ICC:
            # asks whether subjects maintain their relative ranking.
            denom = (
                ms_subject
                + (n_timepoints - 1) * ms_error
            )

        elif icc_key == "absolute":
            # Absolute-agreement ICC:
            # also penalizes systematic shifts between timepoints.
            denom = (
                ms_subject
                + (n_timepoints - 1) * ms_error
                + (
                    n_timepoints
                    * (ms_timepoint - ms_error)
                    / n_subjects
                )
            )

        else:
            raise ValueError(
                "icc_type must be 'consistency' or 'absolute'."
            )

        if (
            pd.isna(denom)
            or np.isclose(denom, 0)
        ):
            icc = np.nan
        else:
            icc = (
                ms_subject - ms_error
            ) / denom

        return {
            "icc": (
                float(icc)
                if not pd.isna(icc)
                else np.nan
            ),
            "n_subjects": int(n_subjects),
            "n_timepoints": int(n_timepoints),
            "ms_subject": float(ms_subject),
            "ms_timepoint": float(ms_timepoint),
            "ms_error": float(ms_error),
        }

    # ==================================================================
    # Load membership and longitudinal data
    # ==================================================================

    membership_df = (
        result["membership_df"]
        .copy()
        .reset_index(drop=True)
    )

    preset_cfg = get_active_fitness_preset_config(cfg)

    timepoint_cfg = dict(
        preset_cfg.get(
            "timepoint_config",
            {},
        )
        or {}
    )

    timepoint_dfs = dict(
        timepoint_cfg.get(
            "timepoint_dfs",
            {},
        )
        or {}
    )

    labels = pd.Series(
        labels,
        name="label",
    ).reset_index(drop=True)

    if len(labels) != len(membership_df):
        raise ValueError(
            f"labels has length {len(labels)}, but membership_df has "
            f"{len(membership_df)} rows. Labels must be row-aligned."
        )

    cluster_col = f"cluster_{cluster_timepoint}"

    if cluster_col not in membership_df.columns:
        raise KeyError(
            f"membership_df is missing cluster column "
            f"{cluster_col!r}."
        )

    # ==================================================================
    # Resolve cluster colors and display labels
    # ==================================================================

    if cluster_colors is None:
        cluster_colors = {
            0: "#1587F8",
            1: "#FFAE17",
            2: "#049B4F",
            3: "#C04AE2",
        }
    else:
        cluster_colors = dict(cluster_colors)

    cluster_label_map = dict(
        cluster_label_map or {}
    )

    def _display_cluster_label(cluster):
        """
        Convert an internal cluster ID into its display label.

        Supports mappings using either numeric or string keys.
        """

        if cluster in cluster_label_map:
            return str(
                cluster_label_map[cluster]
            )

        if str(cluster) in cluster_label_map:
            return str(
                cluster_label_map[str(cluster)]
            )

        return f"Cluster {cluster}"

    # ==================================================================
    # Compute ICC for each variable within each baseline-defined cluster
    # ==================================================================

    icc_rows = []

    analysis_mask = labels.eq(target_label)

    baseline_clusters = sorted(
        membership_df.loc[
            analysis_mask,
            cluster_col,
        ]
        .dropna()
        .unique()
    )

    for variable_name, tp_col_map in variables.items():

        missing_tps = [
            tp
            for tp in timepoints
            if tp not in tp_col_map
        ]

        if missing_tps:
            raise KeyError(
                f"Variable {variable_name!r} is missing "
                f"timepoint mappings: {missing_tps}"
            )

        wide = pd.DataFrame(
            index=membership_df.index
        )

        for tp in timepoints:

            if tp not in timepoint_dfs:
                raise KeyError(
                    f"timepoint_dfs is missing "
                    f"timepoint {tp!r}."
                )

            df_tp = timepoint_dfs[tp]
            value_col = tp_col_map[tp]

            if value_col not in df_tp.columns:
                raise KeyError(
                    f"Column {value_col!r} for variable "
                    f"{variable_name!r} was not found in "
                    f"the dataframe for timepoint {tp!r}."
                )

            wide[tp] = pd.to_numeric(
                df_tp[value_col],
                errors="coerce",
            ).to_numpy()

        for cluster in baseline_clusters:

            cluster_mask = membership_df[
                cluster_col
            ].eq(cluster)

            use_mask = (
                analysis_mask
                & cluster_mask
            )

            wide_cluster = wide.loc[
                use_mask,
                list(timepoints),
            ].copy()

            icc_result = _compute_icc_from_wide(
                wide_cluster,
                icc_type=icc_type,
            )

            icc_rows.append({
                "variable": variable_name,
                "cluster_timepoint": cluster_timepoint,
                "cluster": cluster,
                "cluster_label": (
                    _display_cluster_label(cluster)
                ),
                "target_label": target_label,
                "icc_type": icc_type,
                "icc": icc_result["icc"],
                "n_subjects_complete": (
                    icc_result["n_subjects"]
                ),
                "n_timepoints": (
                    icc_result["n_timepoints"]
                ),
                "ms_subject": (
                    icc_result["ms_subject"]
                ),
                "ms_timepoint": (
                    icc_result["ms_timepoint"]
                ),
                "ms_error": (
                    icc_result["ms_error"]
                ),
            })

    icc_df = pd.DataFrame(
        icc_rows
    )

    # ==================================================================
    # Determine feature display order
    # ==================================================================

    if variable_order is None:
        variables_order = list(
            variables.keys()
        )

    elif isinstance(
        variable_order,
        pd.DataFrame,
    ):
        if variable_order_col not in variable_order.columns:
            raise KeyError(
                f"variable_order_col="
                f"{variable_order_col!r} was not found in "
                f"variable_order DataFrame columns: "
                f"{list(variable_order.columns)}"
            )

        variables_order = (
            variable_order[
                variable_order_col
            ]
            .dropna()
            .astype(str)
            .tolist()
        )

    else:
        variables_order = list(
            variable_order
        )

    # Keep variables present in the supplied dictionary.
    variables_order = [
        variable_name
        for variable_name in variables_order
        if variable_name in variables
    ]

    # Add variables omitted from variable_order at the end.
    remaining_variables = [
        variable_name
        for variable_name in variables.keys()
        if variable_name not in variables_order
    ]

    variables_order = (
        variables_order
        + remaining_variables
    )

    if len(variables_order) == 0:
        raise ValueError(
            "No variables are available to plot after "
            "applying variable_order."
        )

    clusters = sorted(
        icc_df["cluster"]
        .dropna()
        .unique()
    )

    # ==================================================================
    # Create figure
    # ==================================================================

    if figsize is None:
        figsize = (
            max(
                8.0,
                1.25 * len(variables_order),
            ),
            5.2,
        )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    x = np.arange(
        len(variables_order)
    )

    # ==================================================================
    # Plot cluster/subtype bars
    # ==================================================================

    for cluster_idx, cluster in enumerate(clusters):

        offset = (
            cluster_idx
            - (len(clusters) - 1) / 2
        ) * bar_width

        heights = []
        ns = []

        for variable_name in variables_order:

            row = icc_df[
                (
                    icc_df["variable"]
                    == variable_name
                )
                & (
                    icc_df["cluster"]
                    == cluster
                )
            ]

            if row.empty:
                heights.append(np.nan)
                ns.append(0)

            else:
                heights.append(
                    row["icc"].iloc[0]
                )

                ns.append(
                    int(
                        row[
                            "n_subjects_complete"
                        ].iloc[0]
                    )
                )

        bars = ax.bar(
            x + offset,
            heights,
            width=bar_width,
            color=cluster_colors.get(
                cluster,
                cluster_colors.get(
                    str(cluster),
                    "gray",
                ),
            ),
            alpha=0.75,
            edgecolor="black",
            linewidth=0.8,

            # Updated display label:
            label=_display_cluster_label(cluster),
        )

        if annotate_bars:
            for bar, height, n_val in zip(
                bars,
                heights,
                ns,
            ):
                if pd.isna(height):
                    continue

                ax.text(
                    (
                        bar.get_x()
                        + bar.get_width() / 2
                    ),
                    height + 0.025,
                    f"{height:.2f}\nn={n_val}",
                    ha="center",
                    va="bottom",
                    fontsize=font_size - 3,
                    fontweight="bold",
                )

    # ==================================================================
    # Axes and labels
    # ==================================================================

    ax.set_xticks(x)

    ax.set_xticklabels(
        variables_order,
        rotation=x_tick_rotation,
        ha=x_tick_ha,
        fontsize=font_size - 1,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=font_size,
    )

    ax.set_title(
        title,
        fontsize=font_size + 2,
        pad=14,
    )

    if y_lim is not None:
        ax.set_ylim(
            *y_lim
        )

    ax.grid(
        axis="y",
        color="gray",
        alpha=0.18,
        linewidth=0.8,
    )

    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    ax.tick_params(
        axis="both",
        colors="black",
        labelsize=font_size - 1,
    )

    # ==================================================================
    # Legend
    # ==================================================================

    if show_legend:

        if legend_title is None:
            legend_title = (
                f"{cluster_timepoint.capitalize()} cluster"
            )

        legend_kwargs = {
            "title": legend_title,
            "loc": legend_loc,
            "frameon": True,
            "fontsize": font_size - 1,
            "title_fontsize": font_size - 1,
        }

        if legend_bbox_to_anchor is not None:
            legend_kwargs[
                "bbox_to_anchor"
            ] = legend_bbox_to_anchor

        ax.legend(
            **legend_kwargs
        )

    plt.tight_layout()

    if show:
        plt.show()

    return (
        icc_df,
        fig,
        ax,
    )


# def plot_cluster_stratified_longitudinal_icc(
#     *,
#     result: Dict[str, Any],
#     cfg: ClinicalResponseGAFSConfig,
#     labels: Sequence[Any],
#     target_label: Any = 1,
#     variables: Mapping[str, Mapping[str, str]],
#     variable_order: Optional[Union[Sequence[str], pd.DataFrame]] = None,
#     variable_order_col: str = "feature",
#     timepoints: Sequence[str] = ("baseline", "week6", "month6"),
#     cluster_timepoint: str = "baseline",
#     cluster_colors: Optional[Dict[Any, str]] = None,
#     title: str = "ICC-based longitudinal stability by cluster",
#     ylabel: str = "ICC",
#     icc_type: str = "consistency",
#     figsize: Optional[Tuple[float, float]] = None,
#     bar_width: float = 0.34,
#     annotate_bars: bool = True,
#     y_lim: Tuple[float, float] = (-0.05, 1.0),
#     font_size: float = 12.0,
#     x_tick_rotation: float = 45,
#     x_tick_ha: str = "right",
#     show_legend: bool = True,
#     legend_loc: str = "best",
#     legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
#     legend_title: Optional[str] = None,
#     show: bool = True,
# ) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
#     """
#     Plot ICC-based longitudinal stability by baseline-defined cluster.

#     Main question
#     -------------
#     Within Diagnosis = target_label subjects, and within each baseline-defined cluster:
#         if a subject has a high value at Baseline,
#         does that subject tend to remain high at Week 6 and Month 6?

#     Grouping
#     --------
#     Subjects are filtered to target_label, usually Diagnosis = 1.
#     Then subjects are grouped by cluster assignment at cluster_timepoint,
#     usually baseline.

#     ICC interpretation
#     ------------------
#     Higher ICC:
#         more stable subject-level ranking over time.

#     Lower ICC:
#         less stable feature values over time.

#     icc_type
#     --------
#     "consistency":
#         Best for asking whether high subjects stay relatively high.

#     "absolute":
#         Stricter; penalizes systematic timepoint shifts.
#     """

#     def _compute_icc_from_wide(
#         wide_df: pd.DataFrame,
#         icc_type: str = "consistency",
#     ) -> Dict[str, Any]:
#         """
#         Compute ICC from a wide matrix:
#             rows = subjects
#             columns = timepoints
#         """
#         x = (
#             wide_df
#             .apply(pd.to_numeric, errors="coerce")
#             .dropna()
#             .to_numpy(dtype=float)
#         )

#         n_subjects, n_timepoints = x.shape

#         if n_subjects < 2 or n_timepoints < 2:
#             return {
#                 "icc": np.nan,
#                 "n_subjects": n_subjects,
#                 "n_timepoints": n_timepoints,
#                 "ms_subject": np.nan,
#                 "ms_timepoint": np.nan,
#                 "ms_error": np.nan,
#             }

#         grand_mean = np.mean(x)
#         subject_means = np.mean(x, axis=1, keepdims=True)
#         timepoint_means = np.mean(x, axis=0, keepdims=True)

#         ss_subject = n_timepoints * np.sum((subject_means - grand_mean) ** 2)
#         ss_timepoint = n_subjects * np.sum((timepoint_means - grand_mean) ** 2)
#         ss_error = np.sum((x - subject_means - timepoint_means + grand_mean) ** 2)

#         df_subject = n_subjects - 1
#         df_timepoint = n_timepoints - 1
#         df_error = df_subject * df_timepoint

#         ms_subject = ss_subject / df_subject if df_subject > 0 else np.nan
#         ms_timepoint = ss_timepoint / df_timepoint if df_timepoint > 0 else np.nan
#         ms_error = ss_error / df_error if df_error > 0 else np.nan

#         icc_key = str(icc_type).lower()

#         if icc_key == "consistency":
#             # ICC consistency:
#             # asks whether subjects maintain their relative ranking over time.
#             denom = ms_subject + (n_timepoints - 1) * ms_error

#         elif icc_key == "absolute":
#             # ICC absolute agreement:
#             # stricter because it also penalizes systematic timepoint shifts.
#             denom = (
#                 ms_subject
#                 + (n_timepoints - 1) * ms_error
#                 + (n_timepoints * (ms_timepoint - ms_error) / n_subjects)
#             )

#         else:
#             raise ValueError("icc_type must be 'consistency' or 'absolute'.")

#         if pd.isna(denom) or np.isclose(denom, 0):
#             icc = np.nan
#         else:
#             icc = (ms_subject - ms_error) / denom

#         return {
#             "icc": float(icc) if not pd.isna(icc) else np.nan,
#             "n_subjects": int(n_subjects),
#             "n_timepoints": int(n_timepoints),
#             "ms_subject": float(ms_subject),
#             "ms_timepoint": float(ms_timepoint),
#             "ms_error": float(ms_error),
#         }

#     membership_df = result["membership_df"].copy()

#     preset_cfg = get_active_fitness_preset_config(cfg)
#     timepoint_cfg = dict(preset_cfg.get("timepoint_config", {}) or {})
#     timepoint_dfs = dict(timepoint_cfg.get("timepoint_dfs", {}) or {})

#     labels = pd.Series(labels, name="label").reset_index(drop=True)

#     if len(labels) != len(membership_df):
#         raise ValueError(
#             f"labels has length {len(labels)}, but membership_df has "
#             f"{len(membership_df)} rows. Labels must be row-aligned."
#         )

#     cluster_col = f"cluster_{cluster_timepoint}"

#     if cluster_col not in membership_df.columns:
#         raise KeyError(f"membership_df is missing cluster column {cluster_col!r}.")

#     if cluster_colors is None:
#         cluster_colors = {
#             0: "#1587F8",
#             1: "#FFAE17",
#             2: "#049B4F",
#             3: "#C04AE2",
#         }

#     # ------------------------------------------------------------------
#     # Compute ICC for each variable within each baseline-defined cluster.
#     # ------------------------------------------------------------------

#     icc_rows = []

#     analysis_mask = labels.eq(target_label)

#     baseline_clusters = sorted(
#         membership_df.loc[analysis_mask, cluster_col].dropna().unique()
#     )

#     for variable_name, tp_col_map in variables.items():
#         missing_tps = [tp for tp in timepoints if tp not in tp_col_map]

#         if missing_tps:
#             raise KeyError(
#                 f"Variable {variable_name!r} is missing timepoint mappings: {missing_tps}"
#             )

#         wide = pd.DataFrame(index=membership_df.index)

#         for tp in timepoints:
#             if tp not in timepoint_dfs:
#                 raise KeyError(f"timepoint_dfs is missing timepoint {tp!r}.")

#             df_tp = timepoint_dfs[tp]
#             value_col = tp_col_map[tp]

#             if value_col not in df_tp.columns:
#                 raise KeyError(
#                     f"Column {value_col!r} for variable {variable_name!r} "
#                     f"not found in dataframe for timepoint {tp!r}."
#                 )

#             wide[tp] = pd.to_numeric(df_tp[value_col], errors="coerce").to_numpy()

#         for cluster in baseline_clusters:
#             cluster_mask = membership_df[cluster_col].eq(cluster)
#             use_mask = analysis_mask & cluster_mask

#             wide_cluster = wide.loc[use_mask, list(timepoints)].copy()

#             icc_result = _compute_icc_from_wide(
#                 wide_cluster,
#                 icc_type=icc_type,
#             )

#             icc_rows.append({
#                 "variable": variable_name,
#                 "cluster_timepoint": cluster_timepoint,
#                 "cluster": cluster,
#                 "target_label": target_label,
#                 "icc_type": icc_type,
#                 "icc": icc_result["icc"],
#                 "n_subjects_complete": icc_result["n_subjects"],
#                 "n_timepoints": icc_result["n_timepoints"],
#                 "ms_subject": icc_result["ms_subject"],
#                 "ms_timepoint": icc_result["ms_timepoint"],
#                 "ms_error": icc_result["ms_error"],
#             })

#     icc_df = pd.DataFrame(icc_rows)

#     # ------------------------------------------------------------------
#     # Plot ICC summary.
#     # ------------------------------------------------------------------

#     # ------------------------------------------------------------------
#     # Determine feature / variable display order.
#     # ------------------------------------------------------------------
#     # variable_order can be:
#     #   1. None:
#     #        use the order in variables
#     #   2. list-like:
#     #        use that list directly
#     #   3. DataFrame:
#     #        use variable_order[variable_order_col] in row order
#     #
#     # This is useful when matching the ICC plot order to the
#     # feature-selection-frequency plot order.

#     if variable_order is None:
#         variables_order = list(variables.keys())

#     elif isinstance(variable_order, pd.DataFrame):
#         if variable_order_col not in variable_order.columns:
#             raise KeyError(
#                 f"variable_order_col={variable_order_col!r} was not found in "
#                 f"variable_order DataFrame columns: {list(variable_order.columns)}"
#             )

#         variables_order = (
#             variable_order[variable_order_col]
#             .dropna()
#             .astype(str)
#             .tolist()
#         )

#     else:
#         variables_order = list(variable_order)

#     # Keep only variables that are actually present in the variables dictionary.
#     variables_order = [v for v in variables_order if v in variables]

#     # Add any variables not included in variable_order at the end.
#     remaining_variables = [v for v in variables.keys() if v not in variables_order]
#     variables_order = variables_order + remaining_variables

#     if len(variables_order) == 0:
#         raise ValueError("No variables available to plot after applying variable_order.")

#     clusters = sorted(icc_df["cluster"].dropna().unique())

#     if figsize is None:
#         figsize = (max(8.0, 1.25 * len(variables_order)), 5.2)

#     fig, ax = plt.subplots(figsize=figsize)

#     x = np.arange(len(variables_order))

#     for cluster_idx, cluster in enumerate(clusters):
#         offset = (cluster_idx - (len(clusters) - 1) / 2) * bar_width

#         heights = []
#         ns = []

#         for variable_name in variables_order:
#             row = icc_df[
#                 (icc_df["variable"] == variable_name)
#                 & (icc_df["cluster"] == cluster)
#             ]

#             if row.empty:
#                 heights.append(np.nan)
#                 ns.append(0)
#             else:
#                 heights.append(row["icc"].iloc[0])
#                 ns.append(int(row["n_subjects_complete"].iloc[0]))

#         bars = ax.bar(
#             x + offset,
#             heights,
#             width=bar_width,
#             color=cluster_colors.get(cluster, "gray"),
#             alpha=0.75,
#             edgecolor="black",
#             linewidth=0.8,
#             label=f"Cluster {cluster}",
#         )

#         if annotate_bars:
#             for bar, height, n_val in zip(bars, heights, ns):
#                 if pd.isna(height):
#                     continue

#                 ax.text(
#                     bar.get_x() + bar.get_width() / 2,
#                     height + 0.025,
#                     f"{height:.2f}\nn={n_val}",
#                     ha="center",
#                     va="bottom",
#                     fontsize=font_size - 3,
#                     fontweight="bold",
#                 )

#     ax.set_xticks(x)
#     ax.set_xticklabels(
#         variables_order,
#         rotation=x_tick_rotation,
#         ha=x_tick_ha,
#         fontsize=font_size - 1,
#     )

#     ax.set_ylabel(ylabel, fontsize=font_size)
#     ax.set_title(title, fontsize=font_size + 2, pad=14)

#     if y_lim is not None:
#         ax.set_ylim(*y_lim)

#     ax.grid(axis="y", color="gray", alpha=0.18, linewidth=0.8)
#     ax.set_axisbelow(True)

#     for spine in ax.spines.values():
#         spine.set_color("black")
#         spine.set_linewidth(1.0)

#     ax.tick_params(axis="both", colors="black", labelsize=font_size - 1)

#     if show_legend:
#         if legend_title is None:
#             legend_title = f"{cluster_timepoint.capitalize()} cluster"

#         legend_kwargs = {
#             "title": legend_title,
#             "loc": legend_loc,
#             "frameon": True,
#             "fontsize": font_size - 1,
#             "title_fontsize": font_size - 1,
#         }

#         if legend_bbox_to_anchor is not None:
#             legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor

#         ax.legend(**legend_kwargs)

#     plt.tight_layout()

#     if show:
#         plt.show()

#     return icc_df, fig, ax




def summarize_asd_subtype_characteristics(
    *,
    df,
    subtype_col="baseline_subtype",
    continuous_vars=None,
    categorical_vars=None,
    fdr_correction=True,
):
    """
    Summarize post-clustering ASD subtype differences for continuous and
    categorical variables.

    This function is for subtype characterization after the GA/clustering
    result has already been created.

    What this function does
    -----------------------
    It compares ASD subtype groups on each variable separately.

    For example, if subtype_col="baseline_subtype", then the subtype groups
    are defined once using baseline cluster assignment:

        subtype 0 = ASD subjects assigned to baseline cluster 0
        subtype 1 = ASD subjects assigned to baseline cluster 1

    Then, for each variable, the function compares the values between those
    subtype groups.

    Example for Vineland:
        Vineland_composite_standard_score_bl:
            compare baseline Vineland values between subtype 0 and subtype 1

        Vineland_composite_standard_score_w6:
            compare Week 6 Vineland values between the same subtype 0 and
            subtype 1 groups

        Vineland_composite_standard_score_m6:
            compare Month 6 Vineland values between the same subtype 0 and
            subtype 1 groups

    Important
    ---------
    This function does NOT calculate change over time.

    It does NOT calculate:
        Vineland_m6 - Vineland_bl
        SRS_m6 - SRS_bl
        age_m6 - age_bl

    Instead, it asks:
        At each variable/timepoint, do the ASD subtypes differ in their
        absolute values?

    If you want to test whether subtypes differ in change over time, first
    create change-score columns, such as:

        Vineland_change_m6_minus_bl =
            Vineland_composite_standard_score_m6
            - Vineland_composite_standard_score_bl

    and then pass those change-score columns as continuous_vars.

    Inputs
    ------
    df:
        ASD-only subtype dataframe, usually asd_subtype_analysis_df.

    subtype_col:
        Column defining ASD subtype.
        Usually "baseline_subtype", which means subtypes are defined by
        baseline cluster assignment and then used for all comparisons.

    continuous_vars:
        List of continuous variables to compare across subtypes.
        Examples:
            age_bl
            SRS_total_raw_bl
            SRS_total_raw_w6
            Vineland_composite_standard_score_bl

    categorical_vars:
        List of categorical variables to compare across subtypes.
        Examples:
            sex_bl
            site
            race
            ethnicity

    Statistical tests
    -----------------
    Continuous variables:
        If there are 2 subtypes, uses Mann-Whitney U.
        If there are 3 or more subtypes, uses Kruskal-Wallis.

    Categorical variables:
        Uses chi-square test on the subtype-by-category contingency table.

    FDR correction
    --------------
    If fdr_correction=True:
        Applies Benjamini-Hochberg FDR correction separately to:
            1. continuous-variable p-values
            2. categorical-variable p-values

    Outputs
    -------
    continuous_summary_df:
        One row per continuous variable.
        Includes subtype-specific n, mean, median, standard deviation,
        test statistic, raw p-value, and optional FDR-adjusted p-value.

    categorical_summary_df:
        One row per categorical variable.
        Includes chi-square test result, raw p-value, and optional
        FDR-adjusted p-value.

    categorical_detail_df:
        Counts and percentages by subtype for each categorical variable.
    """

    if continuous_vars is None:
        continuous_vars = []

    if categorical_vars is None:
        categorical_vars = []

    if subtype_col not in df.columns:
        raise KeyError(f"{subtype_col!r} was not found in df.")

    subtypes = sorted(df[subtype_col].dropna().unique())

    if len(subtypes) < 2:
        raise ValueError("At least two subtypes are required for comparison.")

    # ------------------------------------------------------------------
    # Continuous variables
    # ------------------------------------------------------------------
    continuous_rows = []

    for var in continuous_vars:
        if var not in df.columns:
            raise KeyError(f"Continuous variable {var!r} was not found in df.")

        d = df[[subtype_col, var]].copy()
        d[var] = pd.to_numeric(d[var], errors="coerce")
        d = d.dropna(subset=[subtype_col, var])

        groups = [
            d.loc[d[subtype_col] == subtype, var].dropna().to_numpy()
            for subtype in subtypes
        ]

        group_ns = [len(g) for g in groups]

        if len(subtypes) == 2 and all(n > 0 for n in group_ns):
            stat, p_value = stats.mannwhitneyu(
                groups[0],
                groups[1],
                alternative="two-sided",
            )
            test_name = "Mann-Whitney U"

        elif len(subtypes) > 2 and all(n > 0 for n in group_ns):
            stat, p_value = stats.kruskal(*groups)
            test_name = "Kruskal-Wallis"

        else:
            stat = np.nan
            p_value = np.nan
            test_name = "Insufficient data"

        row = {
            "variable": var,
            "test": test_name,
            "statistic": stat,
            "p_value": p_value,
            "n_total": int(sum(group_ns)),
        }

        for subtype, values in zip(subtypes, groups):
            row[f"subtype_{subtype}_n"] = int(len(values))
            row[f"subtype_{subtype}_mean"] = np.mean(values) if len(values) else np.nan
            row[f"subtype_{subtype}_median"] = np.median(values) if len(values) else np.nan
            row[f"subtype_{subtype}_std"] = np.std(values, ddof=1) if len(values) > 1 else np.nan

        if len(subtypes) == 2 and all(n > 0 for n in group_ns):
            row[f"mean_diff_subtype_{subtypes[1]}_minus_{subtypes[0]}"] = (
                np.mean(groups[1]) - np.mean(groups[0])
            )
            row[f"median_diff_subtype_{subtypes[1]}_minus_{subtypes[0]}"] = (
                np.median(groups[1]) - np.median(groups[0])
            )

        continuous_rows.append(row)

    continuous_summary_df = pd.DataFrame(continuous_rows)

    # ------------------------------------------------------------------
    # Categorical variables
    # ------------------------------------------------------------------
    categorical_rows = []
    categorical_detail_rows = []

    for var in categorical_vars:
        if var not in df.columns:
            raise KeyError(f"Categorical variable {var!r} was not found in df.")

        d = df[[subtype_col, var]].copy()
        d = d.dropna(subset=[subtype_col, var])

        contingency = pd.crosstab(d[subtype_col], d[var])

        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            stat, p_value, dof, expected = stats.chi2_contingency(contingency)
            test_name = "Chi-square"
        else:
            stat = np.nan
            p_value = np.nan
            dof = np.nan
            test_name = "Insufficient categories"

        categorical_rows.append({
            "variable": var,
            "test": test_name,
            "statistic": stat,
            "p_value": p_value,
            "dof": dof,
            "n_total": int(contingency.to_numpy().sum()),
            "n_categories": int(contingency.shape[1]),
        })

        for subtype in contingency.index:
            subtype_total = contingency.loc[subtype].sum()

            for category in contingency.columns:
                count = int(contingency.loc[subtype, category])

                if subtype_total > 0:
                    pct_within_subtype = 100 * count / subtype_total
                else:
                    pct_within_subtype = np.nan

                categorical_detail_rows.append({
                    "variable": var,
                    "subtype": subtype,
                    "category": category,
                    "count": count,
                    "subtype_total": int(subtype_total),
                    "percent_within_subtype": pct_within_subtype,
                })

    categorical_summary_df = pd.DataFrame(categorical_rows)
    categorical_detail_df = pd.DataFrame(categorical_detail_rows)

    # ------------------------------------------------------------------
    # FDR correction
    # ------------------------------------------------------------------
    if fdr_correction:
        if not continuous_summary_df.empty and "p_value" in continuous_summary_df:
            p = continuous_summary_df["p_value"].to_numpy(dtype=float)
            valid = ~np.isnan(p)

            fdr = np.full_like(p, np.nan, dtype=float)

            if valid.sum() > 0:
                valid_p = p[valid]
                order = np.argsort(valid_p)
                ranked_p = valid_p[order]
                m = len(valid_p)

                q = ranked_p * m / (np.arange(m) + 1)
                q = np.minimum.accumulate(q[::-1])[::-1]
                q = np.clip(q, 0, 1)

                valid_fdr = np.empty_like(q)
                valid_fdr[order] = q
                fdr[valid] = valid_fdr

            continuous_summary_df["fdr_p_value"] = fdr

        if not categorical_summary_df.empty and "p_value" in categorical_summary_df:
            p = categorical_summary_df["p_value"].to_numpy(dtype=float)
            valid = ~np.isnan(p)

            fdr = np.full_like(p, np.nan, dtype=float)

            if valid.sum() > 0:
                valid_p = p[valid]
                order = np.argsort(valid_p)
                ranked_p = valid_p[order]
                m = len(valid_p)

                q = ranked_p * m / (np.arange(m) + 1)
                q = np.minimum.accumulate(q[::-1])[::-1]
                q = np.clip(q, 0, 1)

                valid_fdr = np.empty_like(q)
                valid_fdr[order] = q
                fdr[valid] = valid_fdr

            categorical_summary_df["fdr_p_value"] = fdr

    return continuous_summary_df, categorical_summary_df, categorical_detail_df


 
def plot_asd_subtype_characteristics_panel(
    *,
    df,
    subtype_col="baseline_subtype",
    continuous_variables=None,
    categorical_variables=None,
    stacked_categorical_variables=None,
    continuous_summary_df=None,
    categorical_summary_df=None,
    continuous_pairwise_df=None,
    categorical_pairwise_df=None,
    show_pairwise_stats_in_title=False,
    pairwise_p_col="fdr_p_value",
    pairwise_comparisons_order=None,
    pairwise_stats_per_title_row=2,
    pairwise_stats_line_sep="\n",
    show_significance_brackets=False,
    subtype_colors=None,
    category_colors=None,
    title="ASD subtype characteristics",
    ylabel_font_size=11,
    title_font_size=14,
    panel_title_font_size=11,
    tick_font_size=10,
    x_tick_rotation=0,
    x_tick_ha="center",
    figsize=None,
    n_cols=5,
    panel_width=3.6,
    panel_height=4.2,
    jitter=0.06,
    point_alpha=0.55,
    point_size=18,
    box_width=0.55,
    bar_width=0.24,
    stacked_bar_width=0.62,
    annotation_y_pad=0.04,
    title_pad=10,
    stacked_legend_outside=True,
    stacked_legend_bbox=(1.04, 1.0),
    show_stats_in_title=True,
    show_n=True,
    categorical_percent=True,
    suptitle_y=1.04,
    show=True,
):
    """
    Plot subtype characteristics for mixed continuous and categorical variables.

    Supported panel types
    ---------------------
    1. Continuous variables:
       Boxplot plus jittered subject-level points by subtype/group.

    2. Regular categorical variables:
       Grouped bars by category and subtype/group.

    3. Stacked categorical variables:
       One stacked bar per subtype/group.

    Pairwise statistics
    -------------------
    When ``show_pairwise_stats_in_title=True``, pairwise values are printed
    beneath each panel title.

    When ``show_significance_brackets=True``, significant comparisons are also
    shown as horizontal brackets with asterisks. The brackets use the same
    corrected p-value column selected by ``pairwise_p_col``. Brackets are added
    to continuous panels and stacked categorical panels. They are intentionally
    not added to regular grouped categorical panels because their x-axis
    represents categories rather than subtype/group positions.

    Significance symbols
    --------------------
    p < 0.0001 : ****
    p < 0.001  : ***
    p < 0.01   : **
    p < 0.05   : *
    p >= 0.05  : no bracket

    Expected pairwise dataframe columns
    -----------------------------------
    variable, group_1, group_2, p_value, fdr_p_value

    Important
    ---------
    This function does not calculate statistics. It visualizes statistics
    calculated elsewhere.

    Returns
    -------
    fig, axes
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------------
    # Normalize inputs
    # ------------------------------------------------------------------

    if continuous_variables is None:
        continuous_variables = {}

    if categorical_variables is None:
        categorical_variables = {}

    if stacked_categorical_variables is None:
        stacked_categorical_variables = set()
    else:
        stacked_categorical_variables = set(stacked_categorical_variables)

    if pairwise_comparisons_order is None:
        pairwise_comparisons_order = [
            ("TD", "ASD 0"),
            ("TD", "ASD 1"),
            ("ASD 0", "ASD 1"),
        ]
    else:
        pairwise_comparisons_order = list(pairwise_comparisons_order)

    if subtype_col not in df.columns:
        raise KeyError(f"{subtype_col!r} was not found in df.")

    if not isinstance(continuous_variables, dict):
        raise TypeError(
            "continuous_variables must be a dict: "
            "display_name -> column_name."
        )

    if not isinstance(categorical_variables, dict):
        raise TypeError(
            "categorical_variables must be a dict: "
            "display_name -> column_name."
        )

    for _, col in continuous_variables.items():
        if col not in df.columns:
            raise KeyError(
                f"Continuous variable column {col!r} was not found in df."
            )

    for _, col in categorical_variables.items():
        if col not in df.columns:
            raise KeyError(
                f"Categorical variable column {col!r} was not found in df."
            )

    subtypes = sorted(df[subtype_col].dropna().unique())

    if len(subtypes) < 2:
        raise ValueError(
            "At least two subtypes/groups are required for plotting."
        )

    # ------------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------------

    if subtype_colors is None:
        subtype_colors = {
            0: "#1587F8",
            1: "#FFAE17",
            2: "#049B4F",
            3: "#C04AE2",
            "ASD 0": "#1587F8",
            "ASD 1": "#FFAE17",
            "ASD 2": "#049B4F",
            "ASD 3": "#C04AE2",
            "TD": "#7F7F7F",
        }

    if category_colors is None:
        category_colors = {
            0: "#1587F8",
            1: "#FFAE17",
            2: "#049B4F",
            3: "#C04AE2",
            4: "#8E63C7",
            5: "#8C564B",
            1.0: "#FFAE17",
            2.0: "#049B4F",
            3.0: "#C04AE2",
            4.0: "#8E63C7",
            5.0: "#8C564B",
            "0": "#1587F8",
            "1": "#FFAE17",
            "2": "#049B4F",
            "3": "#C04AE2",
            "4": "#8E63C7",
            "5": "#8C564B",
            "1.0": "#FFAE17",
            "2.0": "#049B4F",
            "3.0": "#C04AE2",
            "4.0": "#8E63C7",
            "5.0": "#8C564B",
            "Missing": "#A9A9A9",
        }

    # ------------------------------------------------------------------
    # Summary-statistic lookup tables
    # ------------------------------------------------------------------

    continuous_stats_lookup = {}

    if (
        continuous_summary_df is not None
        and not continuous_summary_df.empty
    ):
        for _, row in continuous_summary_df.iterrows():
            continuous_stats_lookup[row["variable"]] = row

    categorical_stats_lookup = {}

    if (
        categorical_summary_df is not None
        and not categorical_summary_df.empty
    ):
        for _, row in categorical_summary_df.iterrows():
            categorical_stats_lookup[row["variable"]] = row

    # ------------------------------------------------------------------
    # Pairwise-statistic helpers
    # ------------------------------------------------------------------

    def format_p_value(p_value):
        if pd.isna(p_value):
            return "NA"

        return f"{p_value:.3g}"

    def short_group_label(group_name):
        group_text = str(group_name)

        label_map = {
            "ASD 0": "ASD0",
            "ASD 1": "ASD1",
            "ASD 2": "ASD2",
            "ASD 3": "ASD3",
            "TD": "TD",
        }

        return label_map.get(
            group_text,
            group_text.replace(" ", ""),
        )

    def find_pairwise_row(
        pairwise_df,
        *,
        variable_col,
        group_1,
        group_2,
    ):
        if pairwise_df is None or pairwise_df.empty:
            return None

        required_cols = {
            "variable",
            "group_1",
            "group_2",
            pairwise_p_col,
        }

        if not required_cols.issubset(set(pairwise_df.columns)):
            return None

        match = pairwise_df[
            (pairwise_df["variable"] == variable_col)
            & (
                (
                    (pairwise_df["group_1"] == group_1)
                    & (pairwise_df["group_2"] == group_2)
                )
                |
                (
                    (pairwise_df["group_1"] == group_2)
                    & (pairwise_df["group_2"] == group_1)
                )
            )
        ]

        if match.empty:
            return None

        return match.iloc[0]

    def get_pairwise_stats_text(pairwise_df, variable_col):
        pairwise_labels = []

        for group_1, group_2 in pairwise_comparisons_order:
            matched_row = find_pairwise_row(
                pairwise_df,
                variable_col=variable_col,
                group_1=group_1,
                group_2=group_2,
            )

            if matched_row is None:
                continue

            p_value = matched_row[pairwise_p_col]

            pairwise_labels.append(
                f"{short_group_label(group_1)}-"
                f"{short_group_label(group_2)}="
                f"{format_p_value(p_value)}"
            )

        if len(pairwise_labels) == 0:
            return None

        if pairwise_p_col == "fdr_p_value":
            prefix = "FDR"
        elif pairwise_p_col == "p_value":
            prefix = "p"
        else:
            prefix = pairwise_p_col

        if (
            pairwise_stats_per_title_row is None
            or pairwise_stats_per_title_row <= 0
        ):
            return f"{prefix}: " + " | ".join(pairwise_labels)

        title_rows = []

        for start_idx in range(
            0,
            len(pairwise_labels),
            pairwise_stats_per_title_row,
        ):
            row_labels = pairwise_labels[
                start_idx:
                start_idx + pairwise_stats_per_title_row
            ]

            if start_idx == 0:
                title_rows.append(
                    f"{prefix}: " + " | ".join(row_labels)
                )
            else:
                title_rows.append(" | ".join(row_labels))

        return pairwise_stats_line_sep.join(title_rows)

    def p_value_to_stars(p_value):
        if pd.isna(p_value):
            return None

        p_value = float(p_value)

        if p_value < 0.0001:
            return "****"
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"

        return None

    def get_significant_brackets(
        pairwise_df,
        *,
        variable_col,
        group_positions,
    ):
        brackets = []

        if not show_significance_brackets:
            return brackets

        for comparison_index, (
            group_1,
            group_2,
        ) in enumerate(pairwise_comparisons_order):
            if (
                group_1 not in group_positions
                or group_2 not in group_positions
            ):
                continue

            matched_row = find_pairwise_row(
                pairwise_df,
                variable_col=variable_col,
                group_1=group_1,
                group_2=group_2,
            )

            if matched_row is None:
                continue

            stars = p_value_to_stars(
                matched_row[pairwise_p_col]
            )

            if stars is None:
                continue

            x_1 = group_positions[group_1]
            x_2 = group_positions[group_2]

            brackets.append({
                "x_1": min(x_1, x_2),
                "x_2": max(x_1, x_2),
                "stars": stars,
                "comparison_index": comparison_index,
            })

        # Shorter comparisons are lower; wider comparisons are higher.
        brackets.sort(
            key=lambda bracket: (
                bracket["x_2"] - bracket["x_1"],
                bracket["x_1"],
                bracket["comparison_index"],
            )
        )

        return brackets

    def draw_significance_brackets(ax, brackets):
        if len(brackets) == 0:
            return

        bracket_transform = ax.get_xaxis_transform()
        n_brackets = len(brackets)

        if n_brackets == 1:
            bracket_levels = np.asarray([0.82])
        else:
            # Even spacing for every bracket and every asterisk.
            bracket_levels = np.linspace(
                0.70,
                0.88,
                n_brackets,
            )

        bracket_tick_height = 0.018
        bracket_star_offset = 0.010

        for bracket_y, bracket in zip(
            bracket_levels,
            brackets,
        ):
            bracket_top_y = bracket_y + bracket_tick_height
            star_y = bracket_top_y + bracket_star_offset

            ax.plot(
                [
                    bracket["x_1"],
                    bracket["x_1"],
                    bracket["x_2"],
                    bracket["x_2"],
                ],
                [
                    bracket_y,
                    bracket_top_y,
                    bracket_top_y,
                    bracket_y,
                ],
                color="black",
                linewidth=1.3,
                transform=bracket_transform,
                clip_on=True,
            )

            ax.text(
                (bracket["x_1"] + bracket["x_2"]) / 2.0,
                star_y,
                bracket["stars"],
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="black",
                transform=bracket_transform,
                clip_on=True,
            )

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------

    panel_specs = []

    for display_name, col in continuous_variables.items():
        panel_specs.append({
            "display_name": display_name,
            "column": col,
            "type": "continuous",
        })

    for display_name, col in categorical_variables.items():
        if (
            display_name in stacked_categorical_variables
            or col in stacked_categorical_variables
        ):
            categorical_type = "stacked_categorical"
        else:
            categorical_type = "categorical"

        panel_specs.append({
            "display_name": display_name,
            "column": col,
            "type": categorical_type,
        })

    n_panels = len(panel_specs)

    if n_panels == 0:
        raise ValueError("No variables were provided for plotting.")

    if n_cols is None:
        n_cols = n_panels

    n_cols = min(n_cols, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))

    if figsize is None:
        figsize = (
            panel_width * n_cols,
            panel_height * n_rows,
        )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
    )

    axes = axes.flatten()
    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Plot each panel
    # ------------------------------------------------------------------

    for ax, spec in zip(axes, panel_specs):
        display_name = spec["display_name"]
        col = spec["column"]
        var_type = spec["type"]

        # --------------------------------------------------------------
        # Continuous panel: boxplot plus jittered points
        # --------------------------------------------------------------

        if var_type == "continuous":
            d = df[[subtype_col, col]].copy()
            d[col] = pd.to_numeric(
                d[col],
                errors="coerce",
            )
            d = d.dropna(subset=[subtype_col, col])

            data_by_subtype = [
                d.loc[
                    d[subtype_col] == subtype,
                    col,
                ]
                .dropna()
                .to_numpy()
                for subtype in subtypes
            ]

            positions = np.arange(len(subtypes))
            group_positions = dict(zip(subtypes, positions))

            bp = ax.boxplot(
                data_by_subtype,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
            )

            for patch, subtype in zip(
                bp["boxes"],
                subtypes,
            ):
                patch.set_facecolor(
                    subtype_colors.get(subtype, "gray")
                )
                patch.set_alpha(0.35)
                patch.set_edgecolor(
                    subtype_colors.get(subtype, "black")
                )
                patch.set_linewidth(1.6)

            for median in bp["medians"]:
                median.set_color("black")
                median.set_linewidth(1.5)

            for pos, subtype, values in zip(
                positions,
                subtypes,
                data_by_subtype,
            ):
                if len(values) == 0:
                    continue

                x_jitter = rng.normal(
                    loc=pos,
                    scale=jitter,
                    size=len(values),
                )

                ax.scatter(
                    x_jitter,
                    values,
                    s=point_size,
                    alpha=point_alpha,
                    color=subtype_colors.get(
                        subtype,
                        "gray",
                    ),
                    edgecolors="white",
                    linewidths=0.3,
                )

            significant_brackets = get_significant_brackets(
                continuous_pairwise_df,
                variable_col=col,
                group_positions=group_positions,
            )

            finite_values = np.concatenate([
                np.asarray(values, dtype=float)
                for values in data_by_subtype
                if len(values) > 0
            ])

            data_y_min = float(np.nanmin(finite_values))
            data_y_max = float(np.nanmax(finite_values))
            data_y_range = data_y_max - data_y_min

            if (
                not np.isfinite(data_y_range)
                or data_y_range <= 0
            ):
                data_y_range = max(
                    abs(data_y_max) * 0.10,
                    1.0,
                )

            y_min_current, y_max_current = ax.get_ylim()
            sample_label_offset = annotation_y_pad * data_y_range

            if significant_brackets:
                # Keep data and n labels in the lower portion of the axes.
                data_top_with_label = (
                    data_y_max
                    + sample_label_offset
                    + 0.05 * data_y_range
                )
                data_top_fraction = 0.66

                required_y_max = (
                    y_min_current
                    + (
                        data_top_with_label
                        - y_min_current
                    )
                    / data_top_fraction
                )

                required_y_max = max(
                    required_y_max,
                    y_max_current,
                )
            else:
                required_y_max = max(
                    y_max_current + 0.14 * data_y_range,
                    data_y_max + 0.14 * data_y_range,
                )

            ax.set_ylim(
                y_min_current,
                required_y_max,
            )

            if show_n:
                for pos, subtype, values in zip(
                    positions,
                    subtypes,
                    data_by_subtype,
                ):
                    if len(values) == 0:
                        continue

                    ax.text(
                        pos,
                        np.nanmax(values) + sample_label_offset,
                        f"n={len(values)}",
                        ha="center",
                        va="bottom",
                        fontsize=tick_font_size - 1,
                        fontweight="bold",
                        clip_on=True,
                    )

            draw_significance_brackets(
                ax,
                significant_brackets,
            )

            ax.set_xticks(positions)
            ax.set_xticklabels(
                [str(subtype) for subtype in subtypes],
                fontsize=tick_font_size,
                rotation=x_tick_rotation,
                ha=x_tick_ha,
            )

            ax.set_ylabel(
                display_name,
                fontsize=ylabel_font_size,
            )

            panel_title = display_name

            if show_pairwise_stats_in_title:
                pairwise_text = get_pairwise_stats_text(
                    pairwise_df=continuous_pairwise_df,
                    variable_col=col,
                )

                if pairwise_text is not None:
                    panel_title += f"\n{pairwise_text}"

            elif show_stats_in_title:
                stat_row = continuous_stats_lookup.get(col)

                if stat_row is not None:
                    panel_title += (
                        f"\np={stat_row['p_value']:.3g}"
                    )

                    if "fdr_p_value" in stat_row.index:
                        panel_title += (
                            f", FDR={stat_row['fdr_p_value']:.3g}"
                        )

            ax.set_title(
                panel_title,
                fontsize=panel_title_font_size,
                pad=title_pad,
            )

        # --------------------------------------------------------------
        # Regular categorical panel: grouped bars
        # --------------------------------------------------------------

        elif var_type == "categorical":
            d = df[[subtype_col, col]].copy()
            d = d.dropna(subset=[subtype_col, col])

            categories = sorted(d[col].dropna().unique())
            x = np.arange(len(categories))
            n_subtypes = len(subtypes)

            offsets = (
                np.arange(n_subtypes)
                - (n_subtypes - 1) / 2
            ) * (bar_width * 1.1)

            max_height = 0.0

            for subtype_idx, subtype in enumerate(subtypes):
                d_sub = d[
                    d[subtype_col] == subtype
                ].copy()
                subtype_total = len(d_sub)
                heights = []

                for category in categories:
                    count = int(
                        (d_sub[col] == category).sum()
                    )

                    if categorical_percent:
                        if subtype_total > 0:
                            value = 100 * count / subtype_total
                        else:
                            value = np.nan
                    else:
                        value = count

                    heights.append(value)

                bars = ax.bar(
                    x + offsets[subtype_idx],
                    heights,
                    width=bar_width,
                    color=subtype_colors.get(
                        subtype,
                        "gray",
                    ),
                    alpha=0.75,
                    edgecolor="black",
                    linewidth=0.8,
                    label=str(subtype),
                )

                finite_heights = [
                    height
                    for height in heights
                    if pd.notna(height)
                ]
                local_max_height = (
                    max(finite_heights)
                    if finite_heights
                    else 0.0
                )

                for bar, height, category in zip(
                    bars,
                    heights,
                    categories,
                ):
                    if pd.isna(height):
                        continue

                    count = int(
                        (d_sub[col] == category).sum()
                    )

                    if categorical_percent:
                        label_text = (
                            f"{height:.0f}%\nn={count}"
                        )
                    else:
                        label_text = f"{count}"

                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + max(
                            1,
                            0.02 * local_max_height,
                        ),
                        label_text,
                        ha="center",
                        va="bottom",
                        fontsize=tick_font_size - 2,
                        fontweight="bold",
                        clip_on=False,
                    )

                if finite_heights:
                    max_height = max(
                        max_height,
                        local_max_height,
                    )

            ax.set_xticks(x)
            ax.set_xticklabels(
                [str(category) for category in categories],
                fontsize=tick_font_size,
                rotation=x_tick_rotation,
                ha=x_tick_ha,
            )

            if categorical_percent:
                ax.set_ylabel(
                    "Percent within subtype",
                    fontsize=ylabel_font_size,
                )
                ax.set_ylim(
                    0,
                    max(112, max_height * 1.25),
                )
            else:
                ax.set_ylabel(
                    "Count",
                    fontsize=ylabel_font_size,
                )
                ax.set_ylim(
                    0,
                    max(1.0, max_height * 1.25),
                )

            panel_title = display_name

            if show_pairwise_stats_in_title:
                pairwise_text = get_pairwise_stats_text(
                    pairwise_df=categorical_pairwise_df,
                    variable_col=col,
                )

                if pairwise_text is not None:
                    panel_title += f"\n{pairwise_text}"

            elif show_stats_in_title:
                stat_row = categorical_stats_lookup.get(col)

                if stat_row is not None:
                    panel_title += (
                        f"\np={stat_row['p_value']:.3g}"
                    )

                    if "fdr_p_value" in stat_row.index:
                        panel_title += (
                            f", FDR={stat_row['fdr_p_value']:.3g}"
                        )

            ax.set_title(
                panel_title,
                fontsize=panel_title_font_size,
                pad=title_pad,
            )

            ax.legend(
                frameon=True,
                fontsize=tick_font_size - 1,
                loc="best",
            )

        # --------------------------------------------------------------
        # Stacked categorical panel: one stacked bar per group
        # --------------------------------------------------------------

        elif var_type == "stacked_categorical":
            d = df[[subtype_col, col]].copy()
            d = d.dropna(subset=[subtype_col, col])

            categories = sorted(d[col].dropna().unique())
            x = np.arange(len(subtypes))
            group_positions = dict(zip(subtypes, x))

            count_table = pd.crosstab(
                d[subtype_col],
                d[col],
                dropna=False,
            )

            count_table = count_table.reindex(
                index=subtypes,
                fill_value=0,
            )
            count_table = count_table.reindex(
                columns=categories,
                fill_value=0,
            )

            if categorical_percent:
                plot_table = (
                    count_table
                    .div(count_table.sum(axis=1), axis=0)
                    * 100
                )
                plot_table = plot_table.fillna(0)
                ax.set_ylabel(
                    "Percent within subtype",
                    fontsize=ylabel_font_size,
                )
                data_top = 100.0
            else:
                plot_table = count_table.copy()
                ax.set_ylabel(
                    "Count",
                    fontsize=ylabel_font_size,
                )
                data_top = float(
                    plot_table.to_numpy().sum(axis=1).max()
                )

            significant_brackets = get_significant_brackets(
                categorical_pairwise_df,
                variable_col=col,
                group_positions=group_positions,
            )

            if categorical_percent:
                sample_label_y = 103.0
            else:
                sample_label_y = data_top + max(
                    0.03 * data_top,
                    0.5,
                )

            if significant_brackets:
                data_top_with_label = (
                    sample_label_y
                    + max(0.04 * max(data_top, 1.0), 1.0)
                )
                data_top_fraction = 0.66
                required_y_max = (
                    data_top_with_label
                    / data_top_fraction
                )
            else:
                if categorical_percent:
                    required_y_max = 112.0
                else:
                    required_y_max = max(
                        1.0,
                        data_top * 1.18,
                    )

            ax.set_ylim(
                0,
                required_y_max,
            )

            bottom = np.zeros(len(subtypes))

            for category in categories:
                values = plot_table[category].to_numpy()
                counts = count_table[category].to_numpy()

                bars = ax.bar(
                    x,
                    values,
                    bottom=bottom,
                    width=stacked_bar_width,
                    color=category_colors.get(
                        category,
                        None,
                    ),
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.8,
                    label=str(category),
                )

                for bar, value, count, base in zip(
                    bars,
                    values,
                    counts,
                    bottom,
                ):
                    if count == 0:
                        continue

                    if categorical_percent and value < 6:
                        continue

                    if not categorical_percent and value < 1:
                        continue

                    if categorical_percent:
                        label_text = f"{value:.0f}%"
                    else:
                        label_text = f"n={count}"

                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        base + value / 2,
                        label_text,
                        ha="center",
                        va="center",
                        fontsize=tick_font_size - 2,
                        fontweight="bold",
                    )

                bottom = bottom + values

            ax.set_xticks(x)
            ax.set_xticklabels(
                [str(subtype) for subtype in subtypes],
                fontsize=tick_font_size,
                rotation=x_tick_rotation,
                ha=x_tick_ha,
            )

            if show_n:
                for idx, subtype in enumerate(subtypes):
                    n_total = int(
                        count_table.loc[subtype].sum()
                    )

                    ax.text(
                        idx,
                        sample_label_y,
                        f"n={n_total}",
                        ha="center",
                        va="bottom",
                        fontsize=tick_font_size - 1,
                        fontweight="bold",
                        clip_on=True,
                    )

            draw_significance_brackets(
                ax,
                significant_brackets,
            )

            panel_title = display_name

            if show_pairwise_stats_in_title:
                pairwise_text = get_pairwise_stats_text(
                    pairwise_df=categorical_pairwise_df,
                    variable_col=col,
                )

                if pairwise_text is not None:
                    panel_title += f"\n{pairwise_text}"

            elif show_stats_in_title:
                stat_row = categorical_stats_lookup.get(col)

                if stat_row is not None:
                    panel_title += (
                        f"\np={stat_row['p_value']:.3g}"
                    )

                    if "fdr_p_value" in stat_row.index:
                        panel_title += (
                            f", FDR={stat_row['fdr_p_value']:.3g}"
                        )

            ax.set_title(
                panel_title,
                fontsize=panel_title_font_size,
                pad=title_pad,
            )

            if stacked_legend_outside:
                ax.legend(
                    title=display_name,
                    frameon=True,
                    fontsize=tick_font_size - 1,
                    title_fontsize=tick_font_size,
                    loc="upper left",
                    bbox_to_anchor=stacked_legend_bbox,
                    borderaxespad=0.0,
                )
            else:
                ax.legend(
                    title=display_name,
                    frameon=True,
                    fontsize=tick_font_size - 1,
                    title_fontsize=tick_font_size,
                    loc="best",
                )

        ax.grid(
            axis="y",
            color="gray",
            alpha=0.18,
            linewidth=0.8,
        )
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

        ax.tick_params(
            axis="both",
            colors="black",
            labelsize=tick_font_size,
        )

    # ------------------------------------------------------------------
    # Final figure layout
    # ------------------------------------------------------------------

    for unused_ax in axes[n_panels:]:
        unused_ax.axis("off")

    fig.suptitle(
        title,
        fontsize=title_font_size,
        y=suptitle_y,
    )

    plt.tight_layout(
        rect=[0, 0, 0.96, 0.94]
    )

    if show:
        plt.show()

    return fig, axes[:n_panels]



# def plot_asd_subtype_characteristics_panel(
#     *,
#     df,
#     subtype_col="baseline_subtype",
#     continuous_variables=None,
#     categorical_variables=None,
#     stacked_categorical_variables=None,
#     continuous_summary_df=None,
#     categorical_summary_df=None,
#     continuous_pairwise_df=None,
#     categorical_pairwise_df=None,
#     show_pairwise_stats_in_title=False,
#     pairwise_p_col="fdr_p_value",
#     pairwise_comparisons_order=None,
#     pairwise_stats_per_title_row=2,
#     pairwise_stats_line_sep="\n",
#     subtype_colors=None,
#     category_colors=None,
#     title="ASD subtype characteristics",
#     ylabel_font_size=11,
#     title_font_size=14,
#     panel_title_font_size=11,
#     tick_font_size=10,
#     x_tick_rotation=0,
#     x_tick_ha="center",
#     figsize=None,
#     n_cols=5,
#     panel_width=3.6,
#     panel_height=4.2,
#     jitter=0.06,
#     point_alpha=0.55,
#     point_size=18,
#     box_width=0.55,
#     bar_width=0.24,
#     stacked_bar_width=0.62,
#     annotation_y_pad=0.04,
#     title_pad=10,
#     stacked_legend_outside=True,
#     stacked_legend_bbox=(1.04, 1.0),
#     show_stats_in_title=True,
#     show_n=True,
#     categorical_percent=True,
#     suptitle_y=1.04,
#     show=True,
# ):
#     """
#     Plot ASD subtype characteristics for mixed continuous and categorical variables.

#     This function is for post-clustering subtype characterization.

#     It supports three panel types in the same figure:

#         1. Continuous variables
#            Plot:
#                boxplot + jittered points by subtype/group

#         2. Regular categorical variables
#            Plot:
#                grouped bar plot by category and subtype/group

#         3. Stacked categorical variables
#            Plot:
#                one stacked bar per subtype/group, where each segment is a category.

#     Pairwise statistics
#     -------------------
#     If show_pairwise_stats_in_title=True, the function adds pairwise p-values
#     directly to each panel title.

#     Expected pairwise dataframe columns:
#         variable
#         group_1
#         group_2
#         p_value
#         fdr_p_value

#     Example pairwise title:
#         SRS raw
#         FDR: TD-ASD0<0.001 | TD-ASD1<0.001 | ASD0-ASD1=0.004

#     Important
#     ---------
#     This function does not calculate statistics.
#     It only visualizes the dataframe and optionally displays statistics
#     calculated elsewhere.
#     """

#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt

#     if continuous_variables is None:
#         continuous_variables = {}

#     if categorical_variables is None:
#         categorical_variables = {}

#     if stacked_categorical_variables is None:
#         stacked_categorical_variables = set()
#     else:
#         stacked_categorical_variables = set(stacked_categorical_variables)

#     if pairwise_comparisons_order is None:
#         pairwise_comparisons_order = [
#             ("TD", "ASD 0"),
#             ("TD", "ASD 1"),
#             ("ASD 0", "ASD 1"),
#         ]

#     if subtype_col not in df.columns:
#         raise KeyError(f"{subtype_col!r} was not found in df.")

#     if not isinstance(continuous_variables, dict):
#         raise TypeError("continuous_variables must be a dict: display_name -> column_name.")

#     if not isinstance(categorical_variables, dict):
#         raise TypeError("categorical_variables must be a dict: display_name -> column_name.")

#     for display_name, col in continuous_variables.items():
#         if col not in df.columns:
#             raise KeyError(f"Continuous variable column {col!r} was not found in df.")

#     for display_name, col in categorical_variables.items():
#         if col not in df.columns:
#             raise KeyError(f"Categorical variable column {col!r} was not found in df.")

#     subtypes = sorted(df[subtype_col].dropna().unique())

#     if len(subtypes) < 2:
#         raise ValueError("At least two subtypes/groups are required for plotting.")

#     if subtype_colors is None:
#         subtype_colors = {
#             0: "#1587F8",
#             1: "#FFAE17",
#             2: "#049B4F",
#             3: "#C04AE2",
#             "ASD 0": "#1587F8",
#             "ASD 1": "#FFAE17",
#             "ASD 2": "#049B4F",
#             "ASD 3": "#C04AE2",
#             "TD": "#7F7F7F",
#         }

#     if category_colors is None:
#         category_colors = {
#             0: "#1587F8",
#             1: "#FFAE17",
#             2: "#049B4F",
#             3: "#C04AE2",
#             4: "#8E63C7",
#             5: "#8C564B",
#             1.0: "#FFAE17",
#             2.0: "#049B4F",
#             3.0: "#C04AE2",
#             4.0: "#8E63C7",
#             5.0: "#8C564B",
#             "0": "#1587F8",
#             "1": "#FFAE17",
#             "2": "#049B4F",
#             "3": "#C04AE2",
#             "4": "#8E63C7",
#             "5": "#8C564B",
#             "1.0": "#FFAE17",
#             "2.0": "#049B4F",
#             "3.0": "#C04AE2",
#             "4.0": "#8E63C7",
#             "5.0": "#8C564B",
#             "Missing": "#A9A9A9",
#         }

#     # ------------------------------------------------------------------
#     # Stats lookup
#     # ------------------------------------------------------------------
#     continuous_stats_lookup = {}

#     if continuous_summary_df is not None and not continuous_summary_df.empty:
#         for _, row in continuous_summary_df.iterrows():
#             continuous_stats_lookup[row["variable"]] = row

#     categorical_stats_lookup = {}

#     if categorical_summary_df is not None and not categorical_summary_df.empty:
#         for _, row in categorical_summary_df.iterrows():
#             categorical_stats_lookup[row["variable"]] = row

#     # ------------------------------------------------------------------
#     # Pairwise p-value helpers
#     # ------------------------------------------------------------------

#     def format_p_value(p_value):
#         if pd.isna(p_value):
#             return "NA"

#         return f"{p_value:.3g}"

#     def short_group_label(group_name):
#         group_name = str(group_name)

#         label_map = {
#             "ASD 0": "ASD0",
#             "ASD 1": "ASD1",
#             "ASD 2": "ASD2",
#             "ASD 3": "ASD3",
#             "TD": "TD",
#         }

#         return label_map.get(group_name, group_name.replace(" ", ""))

#     def get_pairwise_stats_text(pairwise_df, variable_col):
#         if pairwise_df is None or pairwise_df.empty:
#             return None

#         required_cols = {"variable", "group_1", "group_2", pairwise_p_col}

#         if not required_cols.issubset(set(pairwise_df.columns)):
#             return None

#         d_pairwise = pairwise_df[pairwise_df["variable"] == variable_col].copy()

#         if d_pairwise.empty:
#             return None

#         pairwise_labels = []

#         for group_1, group_2 in pairwise_comparisons_order:
#             match = d_pairwise[
#                 (
#                     (d_pairwise["group_1"] == group_1)
#                     & (d_pairwise["group_2"] == group_2)
#                 )
#                 |
#                 (
#                     (d_pairwise["group_1"] == group_2)
#                     & (d_pairwise["group_2"] == group_1)
#                 )
#             ]

#             if match.empty:
#                 continue

#             p_value = match.iloc[0][pairwise_p_col]

#             pairwise_labels.append(
#                 f"{short_group_label(group_1)}-{short_group_label(group_2)}={format_p_value(p_value)}"
#             )

#         if len(pairwise_labels) == 0:
#             return None

#         if pairwise_p_col == "fdr_p_value":
#             prefix = "FDR"
#         elif pairwise_p_col == "p_value":
#             prefix = "p"
#         else:
#             prefix = pairwise_p_col

#         # Split long pairwise-stat labels across title rows.
#         if pairwise_stats_per_title_row is None or pairwise_stats_per_title_row <= 0:
#             return f"{prefix}: " + " | ".join(pairwise_labels)

#         title_rows = []
#         for start_idx in range(0, len(pairwise_labels), pairwise_stats_per_title_row):
#             row_labels = pairwise_labels[start_idx:start_idx + pairwise_stats_per_title_row]

#             if start_idx == 0:
#                 title_rows.append(f"{prefix}: " + " | ".join(row_labels))
#             else:
#                 title_rows.append(" | ".join(row_labels))

#         return pairwise_stats_line_sep.join(title_rows)

#     # def get_pairwise_stats_text(pairwise_df, variable_col):
#     #     if pairwise_df is None or pairwise_df.empty:
#     #         return None

#     #     required_cols = {"variable", "group_1", "group_2", pairwise_p_col}

#     #     if not required_cols.issubset(set(pairwise_df.columns)):
#     #         return None

#     #     d_pairwise = pairwise_df[pairwise_df["variable"] == variable_col].copy()

#     #     if d_pairwise.empty:
#     #         return None

#     #     pairwise_labels = []

#     #     for group_1, group_2 in pairwise_comparisons_order:
#     #         match = d_pairwise[
#     #             (
#     #                 (d_pairwise["group_1"] == group_1)
#     #                 & (d_pairwise["group_2"] == group_2)
#     #             )
#     #             |
#     #             (
#     #                 (d_pairwise["group_1"] == group_2)
#     #                 & (d_pairwise["group_2"] == group_1)
#     #             )
#     #         ]

#     #         if match.empty:
#     #             continue

#     #         p_value = match.iloc[0][pairwise_p_col]

#     #         pairwise_labels.append(
#     #             f"{short_group_label(group_1)}-{short_group_label(group_2)}={format_p_value(p_value)}"
#     #         )

#     #     if len(pairwise_labels) == 0:
#     #         return None

#     #     if pairwise_p_col == "fdr_p_value":
#     #         prefix = "FDR"
#     #     elif pairwise_p_col == "p_value":
#     #         prefix = "p"
#     #     else:
#     #         prefix = pairwise_p_col

#     #     return f"{prefix}: " + " | ".join(pairwise_labels)

#     # ------------------------------------------------------------------
#     # Figure setup
#     # ------------------------------------------------------------------
#     panel_specs = []

#     for display_name, col in continuous_variables.items():
#         panel_specs.append(
#             {
#                 "display_name": display_name,
#                 "column": col,
#                 "type": "continuous",
#             }
#         )

#     for display_name, col in categorical_variables.items():
#         if display_name in stacked_categorical_variables or col in stacked_categorical_variables:
#             categorical_type = "stacked_categorical"
#         else:
#             categorical_type = "categorical"

#         panel_specs.append(
#             {
#                 "display_name": display_name,
#                 "column": col,
#                 "type": categorical_type,
#             }
#         )

#     n_panels = len(panel_specs)

#     if n_panels == 0:
#         raise ValueError("No variables were provided for plotting.")

#     if n_cols is None:
#         n_cols = n_panels

#     n_cols = min(n_cols, n_panels)
#     n_rows = int(np.ceil(n_panels / n_cols))

#     if figsize is None:
#         figsize = (panel_width * n_cols, panel_height * n_rows)

#     fig, axes = plt.subplots(
#         n_rows,
#         n_cols,
#         figsize=figsize,
#         squeeze=False,
#     )

#     axes = axes.flatten()
#     rng = np.random.default_rng(42)

#     # ------------------------------------------------------------------
#     # Plot each panel
#     # ------------------------------------------------------------------
#     for ax, spec in zip(axes, panel_specs):
#         display_name = spec["display_name"]
#         col = spec["column"]
#         var_type = spec["type"]

#         # ------------------------------------------------------------------
#         # Continuous panel: boxplot + jittered points
#         # ------------------------------------------------------------------
#         if var_type == "continuous":
#             d = df[[subtype_col, col]].copy()
#             d[col] = pd.to_numeric(d[col], errors="coerce")
#             d = d.dropna(subset=[subtype_col, col])

#             data_by_subtype = [
#                 d.loc[d[subtype_col] == subtype, col].dropna().to_numpy()
#                 for subtype in subtypes
#             ]

#             positions = np.arange(len(subtypes))

#             bp = ax.boxplot(
#                 data_by_subtype,
#                 positions=positions,
#                 widths=box_width,
#                 patch_artist=True,
#                 showfliers=False,
#             )

#             for patch, subtype in zip(bp["boxes"], subtypes):
#                 patch.set_facecolor(subtype_colors.get(subtype, "gray"))
#                 patch.set_alpha(0.35)
#                 patch.set_edgecolor(subtype_colors.get(subtype, "black"))
#                 patch.set_linewidth(1.6)

#             for median in bp["medians"]:
#                 median.set_color("black")
#                 median.set_linewidth(1.5)

#             for pos, subtype, values in zip(positions, subtypes, data_by_subtype):
#                 if len(values) == 0:
#                     continue

#                 x_jitter = rng.normal(loc=pos, scale=jitter, size=len(values))

#                 ax.scatter(
#                     x_jitter,
#                     values,
#                     s=point_size,
#                     alpha=point_alpha,
#                     color=subtype_colors.get(subtype, "gray"),
#                     edgecolors="white",
#                     linewidths=0.3,
#                 )

#             # Add y-axis headroom before placing n labels.
#             y_min_current, y_max_current = ax.get_ylim()
#             y_range_current = y_max_current - y_min_current
#             ax.set_ylim(y_min_current, y_max_current + 0.14 * y_range_current)

#             if show_n:
#                 y_min_current, y_max_current = ax.get_ylim()
#                 y_range_current = y_max_current - y_min_current

#                 for pos, subtype, values in zip(positions, subtypes, data_by_subtype):
#                     if len(values) == 0:
#                         continue

#                     ax.text(
#                         pos,
#                         np.nanmax(values) + annotation_y_pad * y_range_current,
#                         f"n={len(values)}",
#                         ha="center",
#                         va="bottom",
#                         fontsize=tick_font_size - 1,
#                         fontweight="bold",
#                         clip_on=False,
#                     )

#             ax.set_xticks(positions)
#             ax.set_xticklabels(
#                 [str(subtype) for subtype in subtypes],
#                 fontsize=tick_font_size,
#                 rotation=x_tick_rotation,
#                 ha=x_tick_ha,
#             )

#             ax.set_ylabel(display_name, fontsize=ylabel_font_size)

#             panel_title = display_name

#             if show_pairwise_stats_in_title:
#                 pairwise_text = get_pairwise_stats_text(
#                     pairwise_df=continuous_pairwise_df,
#                     variable_col=col,
#                 )

#                 if pairwise_text is not None:
#                     panel_title += f"\n{pairwise_text}"

#             elif show_stats_in_title:
#                 stat_row = continuous_stats_lookup.get(col)

#                 if stat_row is not None:
#                     panel_title += f"\np={stat_row['p_value']:.3g}"

#                     if "fdr_p_value" in stat_row.index:
#                         panel_title += f", FDR={stat_row['fdr_p_value']:.3g}"

#             ax.set_title(
#                 panel_title,
#                 fontsize=panel_title_font_size,
#                 pad=title_pad,
#             )

#         # ------------------------------------------------------------------
#         # Regular categorical panel: grouped bars
#         # ------------------------------------------------------------------
#         elif var_type == "categorical":
#             d = df[[subtype_col, col]].copy()
#             d = d.dropna(subset=[subtype_col, col])

#             categories = sorted(d[col].dropna().unique())

#             x = np.arange(len(categories))
#             n_subtypes = len(subtypes)

#             offsets = (
#                 np.arange(n_subtypes) - (n_subtypes - 1) / 2
#             ) * (bar_width * 1.1)

#             max_height = 0

#             for subtype_idx, subtype in enumerate(subtypes):
#                 d_sub = d[d[subtype_col] == subtype].copy()
#                 subtype_total = len(d_sub)

#                 heights = []

#                 for category in categories:
#                     count = int((d_sub[col] == category).sum())

#                     if categorical_percent:
#                         if subtype_total > 0:
#                             value = 100 * count / subtype_total
#                         else:
#                             value = np.nan
#                     else:
#                         value = count

#                     heights.append(value)

#                 bars = ax.bar(
#                     x + offsets[subtype_idx],
#                     heights,
#                     width=bar_width,
#                     color=subtype_colors.get(subtype, "gray"),
#                     alpha=0.75,
#                     edgecolor="black",
#                     linewidth=0.8,
#                     label=str(subtype),
#                 )

#                 for bar, height, category in zip(bars, heights, categories):
#                     if pd.isna(height):
#                         continue

#                     count = int((d_sub[col] == category).sum())

#                     if categorical_percent:
#                         label_text = f"{height:.0f}%\nn={count}"
#                     else:
#                         label_text = f"{count}"

#                     ax.text(
#                         bar.get_x() + bar.get_width() / 2,
#                         height + max(1, 0.02 * max(heights)),
#                         label_text,
#                         ha="center",
#                         va="bottom",
#                         fontsize=tick_font_size - 2,
#                         fontweight="bold",
#                         clip_on=False,
#                     )

#                 if len(heights) > 0:
#                     max_height = max(max_height, np.nanmax(heights))

#             ax.set_xticks(x)
#             ax.set_xticklabels(
#                 [str(category) for category in categories],
#                 fontsize=tick_font_size,
#                 rotation=x_tick_rotation,
#                 ha=x_tick_ha,
#             )

#             if categorical_percent:
#                 ax.set_ylabel("Percent within subtype", fontsize=ylabel_font_size)
#                 ax.set_ylim(0, max(112, max_height * 1.25))
#             else:
#                 ax.set_ylabel("Count", fontsize=ylabel_font_size)
#                 ax.set_ylim(0, max_height * 1.25)

#             panel_title = display_name

#             if show_pairwise_stats_in_title:
#                 pairwise_text = get_pairwise_stats_text(
#                     pairwise_df=categorical_pairwise_df,
#                     variable_col=col,
#                 )

#                 if pairwise_text is not None:
#                     panel_title += f"\n{pairwise_text}"

#             elif show_stats_in_title:
#                 stat_row = categorical_stats_lookup.get(col)

#                 if stat_row is not None:
#                     panel_title += f"\np={stat_row['p_value']:.3g}"

#                     if "fdr_p_value" in stat_row.index:
#                         panel_title += f", FDR={stat_row['fdr_p_value']:.3g}"

#             ax.set_title(
#                 panel_title,
#                 fontsize=panel_title_font_size,
#                 pad=title_pad,
#             )

#             ax.legend(
#                 frameon=True,
#                 fontsize=tick_font_size - 1,
#                 loc="best",
#             )

#         # ------------------------------------------------------------------
#         # Stacked categorical panel: one stacked bar per subtype/group
#         # ------------------------------------------------------------------
#         elif var_type == "stacked_categorical":
#             d = df[[subtype_col, col]].copy()
#             d = d.dropna(subset=[subtype_col, col])

#             categories = sorted(d[col].dropna().unique())
#             x = np.arange(len(subtypes))

#             count_table = pd.crosstab(
#                 d[subtype_col],
#                 d[col],
#                 dropna=False,
#             )

#             count_table = count_table.reindex(index=subtypes, fill_value=0)
#             count_table = count_table.reindex(columns=categories, fill_value=0)

#             if categorical_percent:
#                 plot_table = count_table.div(count_table.sum(axis=1), axis=0) * 100
#                 plot_table = plot_table.fillna(0)
#                 ax.set_ylabel("Percent within subtype", fontsize=ylabel_font_size)

#                 # Headroom for n annotations above the stacked bars.
#                 ax.set_ylim(0, 112)
#             else:
#                 plot_table = count_table.copy()
#                 ax.set_ylabel("Count", fontsize=ylabel_font_size)

#                 max_count = plot_table.to_numpy().sum(axis=1).max()
#                 ax.set_ylim(0, max_count * 1.18)

#             bottom = np.zeros(len(subtypes))

#             for category in categories:
#                 values = plot_table[category].to_numpy()
#                 counts = count_table[category].to_numpy()

#                 bars = ax.bar(
#                     x,
#                     values,
#                     bottom=bottom,
#                     width=stacked_bar_width,
#                     color=category_colors.get(category, None),
#                     alpha=0.85,
#                     edgecolor="white",
#                     linewidth=0.8,
#                     label=str(category),
#                 )

#                 # Only label readable segments.
#                 for bar, value, count, base in zip(bars, values, counts, bottom):
#                     if count == 0:
#                         continue

#                     if categorical_percent and value < 6:
#                         continue

#                     if not categorical_percent and value < 1:
#                         continue

#                     if categorical_percent:
#                         label_text = f"{value:.0f}%"
#                     else:
#                         label_text = f"n={count}"

#                     ax.text(
#                         bar.get_x() + bar.get_width() / 2,
#                         base + value / 2,
#                         label_text,
#                         ha="center",
#                         va="center",
#                         fontsize=tick_font_size - 2,
#                         fontweight="bold",
#                     )

#                 bottom = bottom + values

#             ax.set_xticks(x)
#             ax.set_xticklabels(
#                 [str(subtype) for subtype in subtypes],
#                 fontsize=tick_font_size,
#                 rotation=x_tick_rotation,
#                 ha=x_tick_ha,
#             )

#             if show_n:
#                 for idx, subtype in enumerate(subtypes):
#                     n_total = int(count_table.loc[subtype].sum())

#                     if categorical_percent:
#                         ax.text(
#                             idx,
#                             103,
#                             f"n={n_total}",
#                             ha="center",
#                             va="bottom",
#                             fontsize=tick_font_size - 1,
#                             fontweight="bold",
#                             clip_on=False,
#                         )
#                     else:
#                         ax.text(
#                             idx,
#                             count_table.loc[subtype].sum(),
#                             f"n={n_total}",
#                             ha="center",
#                             va="bottom",
#                             fontsize=tick_font_size - 1,
#                             fontweight="bold",
#                             clip_on=False,
#                         )

#             panel_title = display_name

#             if show_pairwise_stats_in_title:
#                 pairwise_text = get_pairwise_stats_text(
#                     pairwise_df=categorical_pairwise_df,
#                     variable_col=col,
#                 )

#                 if pairwise_text is not None:
#                     panel_title += f"\n{pairwise_text}"

#             elif show_stats_in_title:
#                 stat_row = categorical_stats_lookup.get(col)

#                 if stat_row is not None:
#                     panel_title += f"\np={stat_row['p_value']:.3g}"

#                     if "fdr_p_value" in stat_row.index:
#                         panel_title += f", FDR={stat_row['fdr_p_value']:.3g}"

#             ax.set_title(
#                 panel_title,
#                 fontsize=panel_title_font_size,
#                 pad=title_pad,
#             )

#             if stacked_legend_outside:
#                 ax.legend(
#                     title=display_name,
#                     frameon=True,
#                     fontsize=tick_font_size - 1,
#                     title_fontsize=tick_font_size,
#                     loc="upper left",
#                     bbox_to_anchor=stacked_legend_bbox,
#                     borderaxespad=0.0,
#                 )
#             else:
#                 ax.legend(
#                     title=display_name,
#                     frameon=True,
#                     fontsize=tick_font_size - 1,
#                     title_fontsize=tick_font_size,
#                     loc="best",
#                 )

#         ax.grid(axis="y", color="gray", alpha=0.18, linewidth=0.8)
#         ax.set_axisbelow(True)

#         for spine in ax.spines.values():
#             spine.set_color("black")
#             spine.set_linewidth(1.0)

#         ax.tick_params(axis="both", colors="black", labelsize=tick_font_size)

#     # Hide any unused axes when n_panels does not fill the final row.
#     for unused_ax in axes[n_panels:]:
#         unused_ax.axis("off")

#     fig.suptitle(title, fontsize=title_font_size, y=suptitle_y)

#     # Leave room on the right for outside legends and on top for panel titles.
#     plt.tight_layout(rect=[0, 0, 0.96, 0.94])

#     if show:
#         plt.show()

#     return fig, axes[:n_panels]



def plot_longitudinal_cluster_label_matrix(
    *,
    result,
    labels,
    timepoints=("baseline", "week6", "month6"),
    label_name="Diagnosis",
    positive_label=1,
    positive_label_display="ASD",
    timepoint_label_map=None,
    label_value_display_map=None,
    cluster_label_map=None,
    cmap="Greens",
    figsize=(15, 5),
    title="Cluster diagnosis composition matrix",
    show_positive_label_within_cluster=True,
    show_positive_label_cohort_share=True,
    show_cell_grid=False,
    annotation_text_color="white",
    axis_label_font_size=12,
    tick_label_font_size=11,
    annotation_font_size=9,
    subplot_title_font_size=12,
    title_font_size=15,
    annotation_font_weight="bold",
    show=True,
):
    """
    Plot cluster-by-label composition matrices across timepoints.

    This is not a prediction confusion matrix.
    It is a composition matrix showing how diagnosis / label groups
    are distributed within clusters at each timepoint.

    Matrix interpretation
    ---------------------
    - Rows = clusters
    - Columns = diagnosis / label groups
    - Cell color = number of subjects in that cluster-label combination
    - Cell text = count

    For the positive-label column, for example ASD, each cell can also show:
    - % positive label within cluster
    - % of positive-label cohort

    Example:
    - "90% ASD within cluster" means 90% of subjects in that cluster are ASD.
    - "41% of ASD cohort" means 41% of all ASD subjects at that timepoint
      fall into that cluster.

    Returns
    -------
    comp_df, fig, axes
    """

    membership_df = result["membership_df"].copy()
    labels = pd.Series(labels, name="label").reset_index(drop=True)

    if len(labels) != len(membership_df):
        raise ValueError(
            f"labels has length {len(labels)}, but membership_df has "
            f"{len(membership_df)} rows. Labels must be row-aligned."
        )

    if timepoint_label_map is None:
        timepoint_label_map = {tp: tp for tp in timepoints}

    # ------------------------------------------------------------------
    # Build long-format composition dataframe
    # ------------------------------------------------------------------
    rows = []

    for tp in timepoints:
        cluster_col = f"cluster_{tp}"

        if cluster_col not in membership_df.columns:
            raise KeyError(f"membership_df is missing {cluster_col!r}.")

        tmp = pd.DataFrame({
            "timepoint": tp,
            "timepoint_label": timepoint_label_map.get(tp, tp),
            "cluster": membership_df[cluster_col].to_numpy(),
            "label": labels.to_numpy(),
        })

        tmp = tmp.dropna(subset=["cluster", "label"])

        total_positive_n = int((tmp["label"] == positive_label).sum())

        for cluster in sorted(tmp["cluster"].unique()):
            d_cluster = tmp[tmp["cluster"] == cluster].copy()
            cluster_total_n = len(d_cluster)
            positive_count = int((d_cluster["label"] == positive_label).sum())

            positive_within_cluster_pct = (
                100 * positive_count / cluster_total_n
                if cluster_total_n > 0 else np.nan
            )

            positive_cohort_pct = (
                100 * positive_count / total_positive_n
                if total_positive_n > 0 else np.nan
            )

            for label_value in sorted(tmp["label"].unique()):
                count = int((d_cluster["label"] == label_value).sum())

                rows.append({
                    "timepoint": tp,
                    "timepoint_label": timepoint_label_map.get(tp, tp),
                    "cluster": cluster,
                    "label": label_value,
                    "count": count,
                    "cluster_total_n": cluster_total_n,
                    "timepoint_positive_total_n": total_positive_n,
                    "positive_count": positive_count,
                    "positive_within_cluster_pct": positive_within_cluster_pct,
                    "positive_cohort_pct": positive_cohort_pct,
                })

    comp_df = pd.DataFrame(rows)

    if comp_df.empty:
        raise ValueError("No composition rows were created.")

    timepoints_order = list(timepoints)
    clusters = sorted(comp_df["cluster"].dropna().unique())
    label_values = sorted(comp_df["label"].dropna().unique())

    if label_value_display_map is None:
        label_value_display_map = {
            val: f"{label_name} {val}"
            for val in label_values
        }

    if cluster_label_map is None:
        cluster_label_map = {
            val: f"Cluster {val}"
            for val in clusters
        }

    vmax = int(comp_df["count"].max())

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        len(timepoints_order),
        figsize=figsize,
        squeeze=False,
    )

    axes = axes.ravel()

    for ax_idx, tp in enumerate(timepoints_order):
        ax = axes[ax_idx]

        d_tp = comp_df[comp_df["timepoint"] == tp].copy()

        matrix = np.zeros((len(clusters), len(label_values)), dtype=int)
        annotations = [["" for _ in label_values] for _ in clusters]

        for i, cluster in enumerate(clusters):
            d_cluster = d_tp[d_tp["cluster"] == cluster].copy()

            if d_cluster.empty:
                continue

            positive_within_cluster_pct = float(
                d_cluster["positive_within_cluster_pct"].iloc[0]
            )
            positive_cohort_pct = float(
                d_cluster["positive_cohort_pct"].iloc[0]
            )

            for j, label_value in enumerate(label_values):
                d_cell = d_cluster[d_cluster["label"] == label_value]

                count = int(d_cell["count"].iloc[0]) if not d_cell.empty else 0
                matrix[i, j] = count

                lines = [f"n={count}"]

                if label_value == positive_label:
                    if show_positive_label_within_cluster:
                        lines.append(
                            f"{positive_within_cluster_pct:.0f}% "
                            f"{positive_label_display} within cluster"
                        )

                    if show_positive_label_cohort_share:
                        lines.append(
                            f"{positive_cohort_pct:.0f}% of "
                            f"{positive_label_display} cohort"
                        )

                annotations[i][j] = "\n".join(lines)

        ax.imshow(
            matrix,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            aspect="auto",
        )

        # ------------------------------------------------------------------
        # Cell annotations
        # ------------------------------------------------------------------
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    annotations[i][j],
                    ha="center",
                    va="center",
                    fontsize=annotation_font_size,
                    fontweight=annotation_font_weight,
                    color=annotation_text_color,
                )

        ax.set_title(
            timepoint_label_map.get(tp, tp),
            fontsize=subplot_title_font_size,
            pad=12,
            fontweight="bold",
        )

        ax.set_xticks(np.arange(len(label_values)))
        ax.set_xticklabels(
            [label_value_display_map[val] for val in label_values],
            fontsize=tick_label_font_size,
        )

        ax.set_yticks(np.arange(len(clusters)))
        ax.set_yticklabels(
            [cluster_label_map[val] for val in clusters],
            fontsize=tick_label_font_size,
        )

        ax.set_xlabel(label_name, fontsize=axis_label_font_size)

        if ax_idx == 0:
            ax.set_ylabel("Cluster", fontsize=axis_label_font_size)

        if show_cell_grid:
            ax.set_xticks(np.arange(-0.5, len(label_values), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(clusters), 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
            ax.tick_params(which="minor", bottom=False, left=False)
        else:
            ax.grid(False)

        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

    fig.suptitle(
        title,
        fontsize=title_font_size,
        #fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if show:
        plt.show()

    return comp_df, fig, axes






#### END ####