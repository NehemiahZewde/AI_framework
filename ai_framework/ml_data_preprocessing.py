# ml_data_preprocessing.py
# ML data preprocessing functions for EEG feature matrices.



from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import missingno as msno

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler




def summarize_feature_matrix(
    X_raw: np.ndarray,
    feature_names: List[str],
    percentiles: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Compute descriptive statistics for the feature matrix using pandas.

    Parameters
    ----------
    X_raw : np.ndarray, shape (n_samples, n_features)
        Feature matrix (e.g., output from stack_features_with_groups).
    feature_names : list[str]
        Names of the features, ordered to match the columns of X_raw.
    percentiles : list[float], optional
        List of percentiles to include in the output (values between 0 and 1).
        If None, uses pandas' default: [0.25, 0.5, 0.75].

    Returns
    -------
    summary_df : pd.DataFrame
        DataFrame where rows are summary statistics (count, mean, std, min,
        selected percentiles, max) and columns are feature names.
    """
    if percentiles is None:
        percentiles = [0.25, 0.5, 0.75]

    df = pd.DataFrame(X_raw, columns=feature_names)
    summary_df = df.describe(percentiles=percentiles)

    return summary_df


def plot_feature_stat_distribution(
    summary_df: pd.DataFrame,
    stat: str = "std",
    kind: Literal["hist", "box"] = "hist",
    bins: int = 50,
    figsize=(8, 3),
    font_size=12,
    xlabel: str | None = None,
) -> None:
    if stat not in summary_df.index:
        raise ValueError(
            f"stat='{stat}' not found in summary_df.index. "
            f"Available: {list(summary_df.index)}"
        )

    values = summary_df.loc[stat].values.astype(float)

    # Auto-generate a more informative x-label
    if xlabel is None:
        if stat.endswith("%") and stat[:-1].isdigit():
            p = stat
            extra = " (median)" if p == "50%" else ""
            xlabel = f"Per-feature {p} percentile{extra} (across samples)"
        else:
            stat_word = "standard deviation" if stat == "std" else stat
            xlabel = f"Per-feature {stat_word} (across samples)"

    plt.figure(figsize=figsize)

    if kind == "hist":
        plt.hist(values, bins=bins)
        plt.xlabel(xlabel, fontsize=font_size, fontweight="bold")
        plt.ylabel("Number of EEG features", fontsize=font_size, fontweight="bold")
        plt.title(
            f"Distribution across EEG features: {stat}",
            fontsize=font_size,
            fontweight="bold",
        )

    elif kind == "box":
        plt.boxplot(values, vert=False)
        plt.xlabel(xlabel, fontsize=font_size, fontweight="bold")
        plt.title(
            f"Boxplot across EEG features: {stat}",
            fontsize=font_size,
            fontweight="bold",
        )

    else:
        raise ValueError("kind must be 'hist' or 'box'.")

    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=font_size)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    plt.tight_layout()
    plt.show()


def plot_feature_stat_raincloud(
    summary_df: pd.DataFrame,
    stat: str = "std",
    title: Optional[str] = None,
    show_points: bool = True,
    figsize: tuple[float, float] = (8, 3),
    font_size: int = 12,
    xlabel: Optional[str] = None,
    # optional fallback color
    base_color: str = "#FFB400",
    # per-element colors
    violin_color: Optional[str] = None,
    point_color: Optional[str] = None,
    # jitter point controls
    jitter_width: float = 0.05,
    point_size: float = 10,
    point_alpha: float = 0.15,
    point_edgecolors: str = "none",
    # box controls
    box_linewidth: float = 1.5,
    median_linewidth: float = 2.0,
    # violin controls
    violin_alpha: float = 0.5,
    violin_edgecolor: str = "black",
    violin_linewidth: float = 1.0,
    violin_half: Literal["full", "left", "right"] = "left",
) -> None:
    """
    Raincloud-style plot for one feature statistic across all EEG features.

    Single axis: violin (left) + boxplot (right) + optional jittered points.
    `violin_half` controls which side of the violin is clipped.
    """
    if stat not in summary_df.index:
        raise ValueError(
            f"stat='{stat}' not found in summary_df.index. "
            f"Available: {list(summary_df.index)}"
        )

    values: np.ndarray = summary_df.loc[stat].values.astype(float)
    values = values[np.isfinite(values)]

    # Auto-generate informative X label
    if xlabel is None:
        if stat.endswith("%") and stat[:-1].isdigit():
            p = stat
            extra = " (median)" if p == "50%" else ""
            xlabel = f"Per-feature {p} percentile{extra} (across samples)"
        else:
            stat_word = "standard deviation" if stat == "std" else stat
            xlabel = f"Per-feature {stat_word} (across samples)"

    if title is None:
        title = "Raincloud plot across EEG features"

    if violin_color is None:
        violin_color = base_color
    if point_color is None:
        point_color = base_color

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    pos: float = 0.0

    # SWAPPED: violin first (left), box + jitter second (right)
    viol_offset: float = -0.20
    box_offset: float = 0.20

    violin_center_x: float = pos + viol_offset
    box_center_x: float = pos + box_offset

    # Violin on the LEFT
    viol_parts = ax.violinplot(
        [values],
        positions=[violin_center_x],
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in viol_parts["bodies"]:
        body.set_facecolor(violin_color)
        body.set_edgecolor(violin_edgecolor)
        body.set_alpha(violin_alpha)
        body.set_linewidth(violin_linewidth)

        # Half-violin clipping
        if violin_half != "full":
            path = body.get_paths()[0]
            verts = path.vertices

            if violin_half == "right":
                # CLIP RIGHT SIDE (keep LEFT half)
                verts[:, 0] = np.minimum(verts[:, 0], violin_center_x)
            elif violin_half == "left":
                # CLIP LEFT SIDE (keep RIGHT half)
                verts[:, 0] = np.maximum(verts[:, 0], violin_center_x)

            path.vertices = verts

    # Boxplot on the RIGHT
    box_parts = ax.boxplot(
        [values],
        positions=[box_center_x],
        widths=0.25,
        patch_artist=True,
        showfliers=False,
    )
    for box, median in zip(box_parts["boxes"], box_parts["medians"]):
        box.set_facecolor("none")
        box.set_edgecolor("black")
        box.set_linewidth(box_linewidth)
        median.set_color("black")
        median.set_linewidth(median_linewidth)
        box.set_zorder(3)

    # Optional jittered points ("rain") at the BOX position (right)
    if show_points:
        x_jitter: np.ndarray = box_center_x + np.random.uniform(
            -jitter_width, jitter_width, size=len(values)
        )
        ax.scatter(
            x_jitter,
            values,
            s=point_size,
            alpha=point_alpha,
            color=point_color,
            edgecolors=point_edgecolors,
            zorder=1,
        )

    ax.set_xticks([])
    ax.set_xlim(-0.6, 0.6)

    ax.set_title(title, fontsize=font_size + 2, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=font_size, fontweight="bold")
    ax.set_ylabel("Values", fontsize=font_size, fontweight="bold")

    ax.tick_params(axis="both", labelsize=font_size)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    plt.tight_layout()
    plt.show()



    
def visualize_missingness(
    X_raw: np.ndarray,
    feature_names: List[str],
    kind: Literal["matrix", "bar"] = "matrix",
    max_features: Optional[int] = None,
    **msno_kwargs: Any,
) -> None:
    """
    Visualize missing data patterns in the feature matrix using `missingno`.

    Parameters
    ----------
    X_raw : np.ndarray, shape (n_samples, n_features)
        Feature matrix, potentially containing NaNs for missing values.
    feature_names : list[str]
        Names of the features, ordered to match the columns of X_raw.
    kind : {"matrix", "bar"}, default "matrix"
        Type of plot:
          - "matrix" : sparkline-style overview of missingness per sample/feature
          - "bar"    : bar plot showing count of non-missing per feature
    max_features : int, optional
        If set and the number of features is larger than this, only the first
        `max_features` columns are visualized (to keep plots readable).
    **msno_kwargs :
        Additional keyword arguments passed directly to the underlying
        `missingno.matrix` or `missingno.bar` call.
        Examples: figsize=(16, 6), fontsize=12, color="maroon", sort="ascending".
    """
    df = pd.DataFrame(X_raw, columns=feature_names)

    if max_features is not None and df.shape[1] > max_features:
        df = df.iloc[:, :max_features]

    # Normalize color if provided as a name; otherwise leave as-is
    if "color" in msno_kwargs and isinstance(msno_kwargs["color"], str):
        msno_kwargs = {**msno_kwargs}  # shallow copy so we don't mutate caller's dict
        msno_kwargs["color"] = mcolors.to_rgb(msno_kwargs["color"])

    if kind == "matrix":
        msno.matrix(df, **msno_kwargs)
    elif kind == "bar":
        msno.bar(df, **msno_kwargs)
    else:
        raise ValueError(f"Unknown kind='{kind}'. Use 'matrix' or 'bar'.")

    #plt.tight_layout()
    plt.show()


# Winsorization / outlier capping

# Winsorization / outlier capping
def cap_outliers_percentile(
    X_raw: np.ndarray,
    feature_names: List[str],
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Winsorize features column-wise by capping values at given percentiles.

    Parameters
    ----------
    X_raw : np.ndarray, shape (n_samples, n_features)
        Raw feature matrix.
    feature_names : list[str]
        Names of the features, ordered to match the columns of X_raw.
    lower_q : float, default 0.05
        Lower percentile (between 0 and 1). Values below this will be
        set to the lower_q percentile value for that feature.
    upper_q : float, default 0.95
        Upper percentile (between 0 and 1). Values above this will be
        set to the upper_q percentile value for that feature.

    Returns
    -------
    X_capped : np.ndarray, shape (n_samples, n_features)
        Feature matrix after percentile capping.
    caps_df : pd.DataFrame
        DataFrame with index = feature_names and two columns:
        'lower' and 'upper', containing the percentile cutoffs used
        for each feature.
    """
    # Ensure a stable numeric dtype for quantiles and clipping
    X = np.asarray(X_raw, dtype=np.float32)

    # Compute per-feature percentiles (column-wise)
    # Note: use np.nanquantile if your matrix can contain NaNs.
    lower = np.nanquantile(X, lower_q, axis=0).astype(np.float32, copy=False)
    upper = np.nanquantile(X, upper_q, axis=0).astype(np.float32, copy=False)

    # Store caps in a small DataFrame (handy for inspection/debug)
    # Keep index aligned with feature_names, just like the original version.
    caps_df = pd.DataFrame({"lower": lower, "upper": upper}, index=feature_names)

    # Apply capping (winsorization)
    # np.clip supports per-column bounds when lower/upper are 1D arrays of length n_features
    X_capped = np.clip(X, lower, upper).astype(np.float32, copy=False)

    return X_capped, caps_df

# def cap_outliers_percentile(
#     X_raw: np.ndarray,
#     feature_names: List[str],
#     lower_q: float = 0.05,
#     upper_q: float = 0.95,
# ) -> Tuple[np.ndarray, pd.DataFrame]:
#     """
#     Winsorize features column-wise by capping values at given percentiles.

#     Parameters
#     ----------
#     X_raw : np.ndarray, shape (n_samples, n_features)
#         Raw feature matrix.
#     feature_names : list[str]
#         Names of the features, ordered to match the columns of X_raw.
#     lower_q : float, default 0.05
#         Lower percentile (between 0 and 1). Values below this will be
#         set to the lower_q percentile value for that feature.
#     upper_q : float, default 0.95
#         Upper percentile (between 0 and 1). Values above this will be
#         set to the upper_q percentile value for that feature.

#     Returns
#     -------
#     X_capped : np.ndarray, shape (n_samples, n_features)
#         Feature matrix after percentile capping.
#     caps_df : pd.DataFrame
#         DataFrame with index = feature_names and two columns:
#         'lower' and 'upper', containing the percentile cutoffs used
#         for each feature.
#     """
#     df = pd.DataFrame(X_raw, columns=feature_names).astype(np.float32)

#     # Compute per-feature percentiles
#     lower = df.quantile(lower_q)
#     upper = df.quantile(upper_q)

#     # Store caps in a small DataFrame (handy for inspection/debug)
#     caps_df = pd.DataFrame({"lower": lower, "upper": upper})

#     # Apply capping (winsorization)
#     df_capped = df.clip(lower=lower, upper=upper, axis=1).astype(np.float32)

#     return df_capped.values, caps_df


# Data Standardization
def standardize_features(
    X_raw: np.ndarray,
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Apply column-wise standardization (zero mean, unit variance per feature).

    Parameters
    ----------
    X_raw : np.ndarray, shape (n_samples, n_features)
        Raw feature matrix.

    Returns
    -------
    X_scaled : np.ndarray, shape (n_samples, n_features)
        Standardized feature matrix.
    scaler : StandardScaler
        Fitted scaler (so you can apply the same transform to new data).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return X_scaled, scaler



# Missing Value Imputation
def impute_missing_features(
    X_raw: np.ndarray,
    strategy: str = "median",
) -> Tuple[np.ndarray, SimpleImputer]:
    """
    Impute missing values column-wise using a simple strategy
    (median by default).

    Parameters
    ----------
    X_raw : np.ndarray, shape (n_samples, n_features)
        Feature matrix with possible NaNs.
    strategy : {"mean", "median", "most_frequent", "constant"}, default "median"
        Imputation strategy passed to sklearn.SimpleImputer.

    Returns
    -------
    X_imputed : np.ndarray
        Feature matrix with NaNs filled in.
    imputer : SimpleImputer
        Fitted imputer (so you can apply the same transform to new data).
    """
    imputer = SimpleImputer(strategy=strategy)
    X_imputed = imputer.fit_transform(X_raw)
    return X_imputed, imputer




def data_preprocessing_pipeline(
    bundle: Dict[str, Any],
    lower_q: float = 0.05,
    upper_q: float = 0.95,
    impute_strategy: str = "median",
    preproc_key: str = "preproc",
) -> Dict[str, Any]:
    """
    Preprocess a bundle's feature matrix in a fixed, reproducible order and
    attach the fitted preprocessing artifacts back onto the bundle.

    This function assumes the bundle represents a single, aligned feature space:
    `bundle["X_raw"]` must have columns ordered exactly as `bundle["feature_names"]`.

    Processing steps (in order)
    ---------------------------
    1) Winsorization / percentile capping (column-wise):
       Values below the `lower_q` percentile and above the `upper_q` percentile
       are clipped per feature. The per-feature cap values are stored in `caps_df`.

    2) Missing-value imputation (column-wise):
       NaNs are filled using a fitted `SimpleImputer` with the given strategy
       (median by default).

    3) Standardization (column-wise):
       A fitted `StandardScaler` is applied to produce zero-mean, unit-variance
       features per column.

    Inputs
    ------
    bundle : dict
        Must contain:
          - "X_raw": np.ndarray of shape (n_samples, n_features)
              Raw feature matrix.
          - "feature_names": list[str] of length n_features
              Feature names aligned to the columns of X_raw.

        Any additional keys (labels, groups, metadata, etc.) are preserved.

    lower_q, upper_q : float
        Percentiles for winsorization (0 < lower_q < upper_q < 1). Default 0.05/0.95.

    impute_strategy : str
        Strategy passed to `sklearn.impute.SimpleImputer`, e.g. "median", "mean",
        "most_frequent", "constant".

    preproc_key : str
        Key under which preprocessing artifacts and configuration are stored
        (default: "preproc").

    Bundle updates (added/overwritten keys)
    ---------------------------------------
    - bundle["X_scaled"] : np.ndarray, shape (n_samples, n_features)
        Result after capping -> imputation -> standard scaling.

    - bundle["feature_name_to_idx"] : dict[str, int]
        Mapping from feature name to its column index in the full feature space.
        This is important for later reduced-feature bundles (e.g., 6 features)
        so you can recover their indices in the original space and invert scaling
        for interpretability (e.g., PDP axes).

    - bundle[preproc_key] : dict
        Contains fitted objects and config needed to reproduce transforms:
          - "feature_names": list[str] (the column order used during fitting)
          - "caps_df": pd.DataFrame with columns ["lower","upper"] indexed by feature name
          - "imputer": fitted SimpleImputer
          - "scaler": fitted StandardScaler
          - "lower_q", "upper_q", "impute_strategy": config values
          - "n_features_fit": int, number of features fit on

    Returns
    -------
    bundle : dict
        The same dictionary object, mutated in-place for convenience.

    Notes
    -----
    - The `StandardScaler` inversion (via mean_/scale_) returns values in the
      *pre-scaled* space (i.e., after capping + imputation). Exact recovery of
      the original uncapped/unimputed raw values is not possible unless you use
      `bundle["X_raw"]`.
    - If you later create a reduced-feature bundle (e.g., 6 features) and only
      store its scaled matrix, you can map its feature names back to full indices
      using `bundle["feature_name_to_idx"]` and invert scaling for those columns.
    """
    if "X_raw" not in bundle:
        raise KeyError("bundle must contain key 'X_raw'")
    if "feature_names" not in bundle:
        raise KeyError("bundle must contain key 'feature_names'")

    X_raw = bundle["X_raw"]
    feature_names = list(bundle["feature_names"])

    # mapping (needed later to compute selected_idx_full by name)
    bundle["feature_name_to_idx"] = {name: i for i, name in enumerate(feature_names)}

    # 1) cap
    X_capped, caps_df = cap_outliers_percentile(
        X_raw, feature_names, lower_q=lower_q, upper_q=upper_q
    )

    # 2) impute
    X_imputed, imputer = impute_missing_features(X_capped, strategy=impute_strategy)

    # 3) scale
    X_scaled, scaler = standardize_features(X_imputed)

    bundle["X_scaled"] = X_scaled

    bundle[preproc_key] = {
        "feature_names": feature_names,
        "caps_df": caps_df,
        "imputer": imputer,
        "scaler": scaler,
        "lower_q": lower_q,
        "upper_q": upper_q,
        "impute_strategy": impute_strategy,
        "n_features_fit": int(X_raw.shape[1]),
    }

    return bundle


# def data_preprocessing_pipeline(
#     X_raw: np.ndarray,
#     feature_names: List[str],
#     lower_q: float = 0.05,
#     upper_q: float = 0.95,
#     impute_strategy: str = "median",
# ) -> Tuple[np.ndarray, Dict[str, Any]]:
#     """
#     Run numeric data preprocessing in a fixed order:
#     (1) percentile capping (winsorization) -> (2) missing value imputation -> (3) standard scaling.

#     Returns the transformed feature matrix and an `artifacts` dict containing the learned
#     preprocessing objects/parameters (caps_df, imputer, scaler, and config values) for reuse.
#     """
#     # 1) Percentile capping
#     X_capped, caps_df = cap_outliers_percentile(
#         X_raw, feature_names, lower_q=lower_q, upper_q=upper_q
#     )

#     # 2) Impute missing values
#     X_imputed, imputer = impute_missing_features(X_capped, strategy=impute_strategy)

#     # 3) Standard scaling
#     X_scaled, scaler = standardize_features(X_imputed)

#     artifacts = {
#         "feature_names": feature_names,
#         "caps_df": caps_df,
#         "imputer": imputer,
#         "scaler": scaler,
#         "lower_q": lower_q,
#         "upper_q": upper_q,
#         "impute_strategy": impute_strategy,
#     }

#     return X_scaled, artifacts