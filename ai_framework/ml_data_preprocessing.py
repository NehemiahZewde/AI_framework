# ml_data_preprocessing.py
# ML data preprocessing functions for EEG feature matrices.



from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Type, Mapping, Literal, Callable

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import missingno as msno

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler




import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

import pandas as pd
from typing import Any, Dict, Optional


def merge_subject_df_into_bundle(
    bundle: Dict[str, Any],
    df: pd.DataFrame,
    *,
    uuid_col: str = "UUID",
    bundle_uuid_in_group_id_to_key_index: int = 1,  # group_id_to_key[gid] == (label, UUID)
    keep_df_cols: Optional[list] = None,
    how: str = "left",
    store_key: str = "subject_table",
) -> Dict[str, Any]:
    """
    SUBJECT-level merge:
      - Build a bundle subject index table: group_id, label, uuid (from group_id_to_key)
      - Merge df (keyed by uuid_col) onto it using UUID
      - Store merged table into bundle[store_key] and stats into bundle[f"{store_key}__meta"]
    """
    if "group_id_to_key" not in bundle:
        raise KeyError("bundle must contain 'group_id_to_key'")

    # 1) Clean/standardize UUID in df
    df_in = df.copy()
    if keep_df_cols is not None:
        df_in = df_in[keep_df_cols].copy()

    if uuid_col not in df_in.columns:
        raise KeyError(f"df must contain uuid_col='{uuid_col}'")

    df_in[uuid_col] = df_in[uuid_col].astype(str).str.strip()

    # If duplicates in df by UUID, keep the first (policy; change if desired)
    df_in = df_in.drop_duplicates(subset=[uuid_col], keep="first")

    # UUID set for match stats
    df_uuid_set = set(df_in[uuid_col].tolist())

    # 2) Build bundle subject index: group_id -> (label, uuid)
    rows = []
    for gid, key_tuple in bundle["group_id_to_key"].items():
        label = key_tuple[0]
        uuid = key_tuple[bundle_uuid_in_group_id_to_key_index]
        rows.append({"group_id": int(gid), "label": label, "uuid": str(uuid).strip()})

    bundle_subject = pd.DataFrame(rows).sort_values("group_id").reset_index(drop=True)

    # 3) Merge on UUID (rename only for the merge input)
    merged = bundle_subject.merge(
        df_in.rename(columns={uuid_col: "uuid"}),
        on="uuid",
        how=how,
        validate="1:1",
    )

    # 4) Store + bookkeeping (match = uuid existed in the df uuid set)
    matched_mask = merged["uuid"].isin(df_uuid_set)
    n_matched = int(matched_mask.sum())
    n_unmatched = int((~matched_mask).sum())

    bundle[store_key] = merged
    bundle[f"{store_key}__meta"] = {
        "uuid_col_in_df": uuid_col,
        "how": how,
        "n_groups_in_bundle": int(bundle_subject.shape[0]),
        "n_rows_in_df_after_dedup": int(df_in.shape[0]),
        "n_rows_merged": int(merged.shape[0]),
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "df_columns_merged_in": [c for c in df_in.columns if c != uuid_col],
        "dedup_policy": "drop_duplicates(keep='first') on UUID",
    }

    return bundle

def encode_categorical_and_ordinal(
    df: pd.DataFrame,
    *,
    # Auto-bucketing defaults (your original behavior)
    low_card_max: int = 3,
    ordinal_card_max: int = 10,

    # Optional overrides
    cat_cols: Optional[List[str]] = None,
    ord_cols: Optional[List[str]] = None,
    drop_cols: Optional[List[str]] = None,

    # Optional explicit orderings for ordinals
    ord_categories: Optional[Dict[str, List[Any]]] = None,

    drop_first: bool = True,

    # What to do with non-numeric columns that end up in "passthrough" bucket
    # Default False: drop them to keep output numeric/model-ready
    allow_non_numeric_passthrough: bool = False,

    return_metadata: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Encode a DataFrame using sklearn encoders.

    Behavior:
    - Normalize all missing-like values to np.nan in the input.
    - Numeric columns passthrough (unless explicitly forced into cat/ord lists).
    - Non-numeric columns are auto-bucketed by cardinality:
        * nunique <= low_card_max                    -> OneHotEncoder
        * low_card_max < nunique <= ordinal_card_max -> OrdinalEncoder
        * nunique > ordinal_card_max                 -> passthrough (or drop by default)
    - Overrides:
        * cat_cols forces OneHotEncoder even if high cardinality
        * ord_cols forces OrdinalEncoder
      Forced columns are removed from auto-bucketing and take precedence.
    - Missing output semantics:
        * numeric/ordinal missing -> np.nan
        * categorical missing -> entire one-hot block set to np.nan (not zeros)

    Notes:
    - If allow_non_numeric_passthrough=False (default), any non-numeric passthrough columns
      are dropped to avoid strings in the output.
    """
    cat_cols = list(cat_cols or [])
    ord_cols = list(ord_cols or [])
    drop_cols = list(drop_cols or [])
    ord_categories = dict(ord_categories or {})

    # 0) Normalize missing values to np.nan + drop requested columns
    data = df.copy()
    data = data.where(pd.notna(data), np.nan)

    existing_drop_cols = [c for c in drop_cols if c in data.columns]
    if existing_drop_cols:
        data = data.drop(columns=existing_drop_cols)

    # Validate override columns exist
    missing_cat = [c for c in cat_cols if c not in data.columns]
    missing_ord = [c for c in ord_cols if c not in data.columns]
    if missing_cat or missing_ord:
        raise KeyError(
            f"Columns not found in df. Missing cat_cols={missing_cat}, ord_cols={missing_ord}"
        )

    forced = set(cat_cols + ord_cols)

    # 1) Numeric passthrough (exclude forced columns; forced wins)
    numeric_cols: List[str] = [
        c for c in data.columns if is_numeric_dtype(data[c]) and c not in forced
    ]

    # 2) Auto-bucket ONLY among remaining non-numeric columns not forced
    non_numeric_cols: List[str] = [
        c for c in data.columns if c not in numeric_cols and c not in forced
    ]

    # Auto-bucket by cardinality (for remaining non-numeric columns)
    categorical_auto: List[str] = []
    ordinal_auto: List[str] = []
    passthrough_auto: List[str] = []
    unique_counts: Dict[str, int] = {}

    if non_numeric_cols:
        counts = data[non_numeric_cols].nunique(dropna=False)
        unique_counts = counts.to_dict()

        categorical_auto = [c for c in non_numeric_cols if counts[c] <= low_card_max]
        ordinal_auto = [c for c in non_numeric_cols if low_card_max < counts[c] <= ordinal_card_max]
        passthrough_auto = [c for c in non_numeric_cols if counts[c] > ordinal_card_max]

    # 3) Final buckets = overrides + auto (no overlaps)
    categorical_cols = list(dict.fromkeys(cat_cols + categorical_auto))
    ordinal_cols = list(dict.fromkeys(ord_cols + ordinal_auto))
    non_numeric_passthrough_cols = passthrough_auto  # only from auto (forced handled above)

    # If we don't allow non-numeric passthrough, drop them (keep output numeric)
    dropped_unspecified_non_numeric: List[str] = []
    if non_numeric_passthrough_cols and not allow_non_numeric_passthrough:
        dropped_unspecified_non_numeric = non_numeric_passthrough_cols
        non_numeric_passthrough_cols = []

    # 4) Build explicit category lists to stabilize feature names + keep NaN behavior consistent
    def categories_in_appearance_order(series: pd.Series) -> List[Any]:
        vals = []
        seen = set()
        for v in series.tolist():
            if pd.isna(v):
                continue
            if v not in seen:
                vals.append(v)
                seen.add(v)
        return vals

    cat_categories: Dict[str, List[Any]] = {
        c: categories_in_appearance_order(data[c].astype("object")) for c in categorical_cols
    }

    ord_categories_final: Dict[str, List[Any]] = {}
    for c in ordinal_cols:
        if c in ord_categories and ord_categories[c] is not None:
            ord_categories_final[c] = [v for v in ord_categories[c] if not pd.isna(v)]
        else:
            ord_categories_final[c] = categories_in_appearance_order(data[c].astype("object"))

    # 5) Encoders
    ohe = OneHotEncoder(
        categories=[cat_categories[c] for c in categorical_cols] if categorical_cols else "auto",
        handle_unknown="ignore",
        drop="first" if drop_first else None,
        sparse_output=False,
    )

    # Use np.nan for missing/unknown if supported; otherwise enforce after transform
    try:
        ord_enc = OrdinalEncoder(
            categories=[ord_categories_final[c] for c in ordinal_cols] if ordinal_cols else "auto",
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        )
        _ordinal_has_encoded_missing = True
    except TypeError:
        ord_enc = OrdinalEncoder(
            categories=[ord_categories_final[c] for c in ordinal_cols] if ordinal_cols else "auto",
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
        )
        _ordinal_has_encoded_missing = False

    # 6) ColumnTransformer
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", "passthrough", numeric_cols))
    if non_numeric_passthrough_cols:
        transformers.append(("non_numeric_passthrough", "passthrough", non_numeric_passthrough_cols))
    if categorical_cols:
        transformers.append(("categorical_one_hot", ohe, categorical_cols))
    if ordinal_cols:
        transformers.append(("ordinal", ord_enc, ordinal_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    encoded = preprocessor.fit_transform(data)

    # 7) Missing-output semantics
    # - categorical: missing -> all-NaN block
    if categorical_cols:
        ohe_fnames = preprocessor.named_transformers_["categorical_one_hot"].get_feature_names_out(categorical_cols)
        ohe_fnames = list(ohe_fnames)
        for col in categorical_cols:
            prefix = f"{col}_"
            block_cols = [c for c in ohe_fnames if c.startswith(prefix)]
            if not block_cols:
                continue
            missing_mask = data[col].isna()
            if missing_mask.any():
                encoded.loc[missing_mask, block_cols] = np.nan

    # - ordinal: missing -> np.nan
    if ordinal_cols:
        for col in ordinal_cols:
            missing_mask = data[col].isna()
            if missing_mask.any() and col in encoded.columns:
                encoded.loc[missing_mask, col] = np.nan

    # 8) Order output columns to follow original input order
    out_cols = list(encoded.columns)

    def ohe_block_for(col: str) -> List[str]:
        prefix = f"{col}_"
        return [c for c in out_cols if c.startswith(prefix)]

    desired_order: List[str] = []
    for col in data.columns:
        if col in numeric_cols or col in non_numeric_passthrough_cols or col in ordinal_cols:
            if col in out_cols:
                desired_order.append(col)
        elif col in categorical_cols:
            desired_order.extend(ohe_block_for(col))

    desired_order = [c for c in desired_order if c in out_cols]
    encoded = encoded[desired_order]

    # 9) Metadata
    output_to_source: Dict[str, Dict[str, Any]] = {}
    for c in numeric_cols:
        if c in encoded.columns:
            output_to_source[c] = {"source_col": c, "type": "numeric", "detail": None}
    for c in non_numeric_passthrough_cols:
        if c in encoded.columns:
            output_to_source[c] = {"source_col": c, "type": "passthrough", "detail": None}
    for c in ordinal_cols:
        if c in encoded.columns:
            output_to_source[c] = {"source_col": c, "type": "ordinal", "detail": {"categories": ord_categories_final.get(c, [])}}
    if categorical_cols:
        for c in categorical_cols:
            prefix = f"{c}_"
            for oc in encoded.columns:
                if oc.startswith(prefix):
                    level = oc[len(prefix):]
                    output_to_source[oc] = {"source_col": c, "type": "onehot", "detail": {"level": level}}

    metadata: Dict[str, Any] = {
        "low_card_max": low_card_max,
        "ordinal_card_max": ordinal_card_max,
        "drop_first": drop_first,
        "missing_value": "np.nan",
        "missing_categorical_output": "all-NaN block",
        "numeric_passthrough_cols": numeric_cols,
        "categorical_one_hot_input_cols": categorical_cols,
        "ordinal_encoded_input_cols": ordinal_cols,
        "non_numeric_passthrough_cols": non_numeric_passthrough_cols,
        "dropped_input_cols": existing_drop_cols,
        "dropped_high_card_non_numeric": dropped_unspecified_non_numeric,
        "unique_counts_non_numeric_auto": unique_counts,
        "categorical_cols_forced": cat_cols,
        "ordinal_cols_forced": ord_cols,
        "categorical_cols_auto": categorical_auto,
        "ordinal_cols_auto": ordinal_auto,
        "passthrough_cols_auto": passthrough_auto,
        "categorical_categories": cat_categories,
        "ordinal_categories": ord_categories_final,
        "ordinal_encoder_supports_encoded_missing_value": _ordinal_has_encoded_missing,
        "feature_names_out": list(encoded.columns),
        "output_to_source": output_to_source,
    }

    return (encoded, metadata) if return_metadata else encoded


def append_subject_tabular_to_X_raw(
    bundle: Dict[str, Any],
    *,
    subject_table_key: str = "subject_table",
    group_col: str = "group_id",
    drop_feature_cols: Sequence[str] = ("group_id", "label", "uuid"),
    # If None, use all columns except drop_feature_cols
    feature_cols: Optional[Sequence[str]] = None,
    # Encoder function you already have (must return (X_tab_df, meta) when return_metadata=True)
    encoder_fn= None,
    # Encoder kwargs (e.g., cat_cols, ord_cols, low_card_max, ordinal_card_max, drop_first, ...)
    encoder_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Encode subject-level covariates from bundle[subject_table_key] and append them to
    bundle["X_raw"] (epoch-level) aligned by bundle["groups"] (group_id per epoch).

    Constraints satisfied:
      - DOES NOT add any new keys to bundle.
      - Only updates: bundle["X_raw"] and bundle["feature_names"].
      - Uses group_id solely for alignment; group_id/label/uuid are NOT appended as features.

    Returns
    -------
    (bundle, meta)
        meta is returned (not stored) so you can inspect feature_names_out, mappings, etc.
    """
    if encoder_fn is None:
        raise ValueError("encoder_fn must be provided (e.g., your encode_categorical_and_ordinal).")
    encoder_kwargs = dict(encoder_kwargs or {})

    # --- Validate bundle ---
    if "X_raw" not in bundle or "groups" not in bundle or "feature_names" not in bundle:
        raise KeyError("bundle must contain 'X_raw', 'groups', and 'feature_names'.")
    if subject_table_key not in bundle:
        raise KeyError(f"bundle must contain '{subject_table_key}' (subject-level table).")

    X_raw = bundle["X_raw"]
    groups = bundle["groups"]

    if X_raw.shape[0] != len(groups):
        raise ValueError(f"X_raw has {X_raw.shape[0]} rows but groups has {len(groups)} entries.")

    subject_table = bundle[subject_table_key]
    if not isinstance(subject_table, pd.DataFrame):
        raise TypeError(f"bundle['{subject_table_key}'] must be a pandas DataFrame.")

    if group_col not in subject_table.columns:
        raise KeyError(f"subject_table must contain '{group_col}' column.")

    # Ensure group_id is int-like for safe alignment with bundle["groups"]
    st = subject_table.copy()
    st[group_col] = st[group_col].astype(int)

    # --- Choose which subject-table columns to encode as features ---
    if feature_cols is None:
        feature_cols = [c for c in st.columns if c not in set(drop_feature_cols)]
    else:
        feature_cols = list(feature_cols)

    # Make sure we did not accidentally include alignment keys
    feature_cols = [c for c in feature_cols if c not in set(drop_feature_cols)]
    if not feature_cols:
        raise ValueError("No feature columns selected to encode/append.")

    # Keep group_id in a separate vector for alignment; do not encode it
    st_keys = st[[group_col]].copy()
    st_feats = st[feature_cols].copy()

    # --- Encode subject-level features ---
    # IMPORTANT: preserve row order by encoding st_feats directly (no sorting/reindexing here)
    X_tab, meta = encoder_fn(st_feats, return_metadata=True, **encoder_kwargs)

    if not isinstance(X_tab, pd.DataFrame):
        raise TypeError("encoder_fn must return a pandas DataFrame when set_output(transform='pandas').")

    # Attach group_id index for alignment (NOT a feature column)
    X_tab = X_tab.copy()
    X_tab.index = st_keys[group_col].values  # index values are group_id aligned to st order

    # If duplicate group_id rows exist (shouldn’t, but safe), keep first
    if X_tab.index.has_duplicates:
        X_tab = X_tab[~X_tab.index.duplicated(keep="first")]

    # --- Broadcast subject rows to epoch rows using groups ---
    # Reindex by epoch group ids; missing group ids become NaN rows
    epoch_gids = pd.Index(pd.Series(groups).astype(int).values, name=group_col)
    X_tab_epoch = X_tab.reindex(epoch_gids)

    # Convert to numpy float32 for concatenation
    X_tab_epoch_np = X_tab_epoch.to_numpy(dtype=np.float32, copy=False)

    # --- Append to X_raw and update feature_names ---
    X_raw_np = np.asarray(X_raw, dtype=np.float32)  # keeps existing if already float32
    X_aug = np.concatenate([X_raw_np, X_tab_epoch_np], axis=1)

    new_feature_names = list(bundle["feature_names"]) + list(meta.get("feature_names_out", X_tab.columns.tolist()))

    # Update ONLY the requested keys
    bundle["X_raw"] = X_aug
    bundle["feature_names"] = new_feature_names

    return bundle, meta



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



def impute_categorical_ordinal_mode(
    X_raw: np.ndarray,
    feature_names: List[str],
    meta: Dict[str, Any],
    *,
    cat_ord_types: Tuple[str, ...] = ("onehot", "ordinal"),
) -> Tuple[np.ndarray, SimpleImputer, List[int]]:
    """
    Mode-impute ONLY the categorical/ordinal columns (as identified by meta).

    Returns:
      - X_imputed: same shape as X_raw
      - imputer: fitted SimpleImputer(strategy="most_frequent") on those columns
      - idx: indices of columns that were imputed
    """
    X = np.asarray(X_raw, dtype=np.float32)
    ots = meta.get("output_to_source", {})

    idx = [
        i for i, name in enumerate(feature_names)
        if (ots.get(name) or {}).get("type") in cat_ord_types
    ]

    if not idx:
        return X.copy(), SimpleImputer(strategy="most_frequent"), []

    X_out = X.copy()
    imputer = SimpleImputer(strategy="most_frequent")
    X_out[:, idx] = imputer.fit_transform(X[:, idx]).astype(np.float32, copy=False)
    return X_out, imputer, idx



def data_preprocessing_pipeline(
    bundle: Dict[str, Any],
    lower_q: float = 0.05,
    upper_q: float = 0.95,
    impute_strategy: str = "median",
    preproc_key: str = "preproc",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Preprocess a bundle's feature matrix in a fixed, reproducible order and
    attach the fitted preprocessing artifacts back onto the bundle.

    This function assumes the bundle represents a single, aligned feature space:
    `bundle["X_raw"]` must have columns ordered exactly as `bundle["feature_names"]`.

    Processing steps (in order)
    ---------------------------
    1) Identify categorical/ordinal columns (if `meta` is provided) using:
         meta["output_to_source"][feature_name]["type"] in {"onehot","ordinal"}

    2) Continuous-feature preprocessing (column-wise) on NON categorical/ordinal columns:
       (a) Winsorization / percentile capping:
           Values below the `lower_q` percentile and above the `upper_q` percentile
           are clipped per feature. The per-feature cap values are stored in `caps_df`.
       (b) Missing-value imputation:
           NaNs are filled using a fitted `SimpleImputer` with the given strategy
           (median by default).
       (c) Standardization:
           A fitted `StandardScaler` is applied to produce zero-mean, unit-variance
           features per column.

       Categorical/ordinal columns are removed before step (a) and added back
       unchanged afterward.

    3) Categorical/ordinal imputation (mode) on ONLY categorical/ordinal columns:
       Missing values in these columns are imputed via "most_frequent".

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
        Strategy passed to `sklearn.impute.SimpleImputer` for CONTINUOUS columns, e.g.
        "median" (default), "mean", "most_frequent", "constant".

    preproc_key : str
        Key under which preprocessing artifacts and configuration are stored
        (default: "preproc").

    meta : dict | None
        Encoder metadata returned by encode_categorical_and_ordinal(..., return_metadata=True).
        If provided, categorical/ordinal columns are detected and excluded from
        continuous preprocessing; then mode-imputed separately.
        If None, the pipeline behaves exactly like the original: all columns are treated
        as continuous and processed the same way.

    Bundle updates (added/overwritten keys)
    ---------------------------------------
    - bundle["X_scaled"] : np.ndarray, shape (n_samples, n_features)
        Result after:
          (continuous columns) capping -> imputation -> standard scaling
          (categorical/ordinal columns) mode imputation
        with columns returned to the original order.

    - bundle["feature_name_to_idx"] : dict[str, int]
        Mapping from feature name to its column index in the full feature space.

    - bundle[preproc_key] : dict
        Contains fitted objects and config needed to reproduce transforms:
          - "feature_names": list[str] (the column order used during fitting)
          - "caps_df": pd.DataFrame with columns ["lower","upper"] indexed by feature name
                       (NaN for categorical/ordinal columns that were not capped)
          - "imputer": fitted SimpleImputer for continuous columns
          - "scaler": fitted StandardScaler for continuous columns
          - "cat_ord_imputer": fitted SimpleImputer(strategy="most_frequent") for cat/ord columns
          - "lower_q", "upper_q", "impute_strategy": config values
          - "n_features_fit": int, number of features fit on
          - "skipped_feature_names": list[str] of categorical/ordinal features (if meta provided)
          - "cat_ord_imputed_feature_names": list[str] of cat/ord features that were mode-imputed

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
    - If `meta` is provided, categorical/ordinal columns are not winsorized or scaled.
      They are only mode-imputed for missingness.
    """
    if "X_raw" not in bundle:
        raise KeyError("bundle must contain key 'X_raw'")
    if "feature_names" not in bundle:
        raise KeyError("bundle must contain key 'feature_names'")

    X_raw = bundle["X_raw"]
    feature_names = list(bundle["feature_names"])

    # mapping (needed later to compute selected_idx_full by name)
    bundle["feature_name_to_idx"] = {name: i for i, name in enumerate(feature_names)}

    n_features = X_raw.shape[1]

    # If no meta, preserve original behavior exactly
    if meta is None:
        X_capped, caps_df = cap_outliers_percentile(
            X_raw, feature_names, lower_q=lower_q, upper_q=upper_q
        )
        X_imputed, imputer = impute_missing_features(X_capped, strategy=impute_strategy)
        X_scaled, scaler = standardize_features(X_imputed)

        bundle["X_scaled"] = X_scaled
        bundle[preproc_key] = {
            "feature_names": feature_names,
            "caps_df": caps_df,
            "imputer": imputer,
            "scaler": scaler,
            "cat_ord_imputer": None,
            "lower_q": lower_q,
            "upper_q": upper_q,
            "impute_strategy": impute_strategy,
            "n_features_fit": int(X_raw.shape[1]),
            "skipped_feature_names": [],
            "cat_ord_imputed_feature_names": [],
        }
        return bundle

    # --- Identify categorical/ordinal feature indices to skip in continuous preprocessing ---
    ots = meta.get("output_to_source", {})
    skip_idx: List[int] = []
    for i, name in enumerate(feature_names):
        t = (ots.get(name) or {}).get("type")
        if t in ("onehot", "ordinal"):
            skip_idx.append(i)

    skip_idx_set = set(skip_idx)
    cont_idx = [i for i in range(n_features) if i not in skip_idx_set]

    # If nothing to skip, just run the original pipeline
    if not skip_idx:
        X_capped, caps_df = cap_outliers_percentile(
            X_raw, feature_names, lower_q=lower_q, upper_q=upper_q
        )
        X_imputed, imputer = impute_missing_features(X_capped, strategy=impute_strategy)
        X_scaled, scaler = standardize_features(X_imputed)

        bundle["X_scaled"] = X_scaled
        bundle[preproc_key] = {
            "feature_names": feature_names,
            "caps_df": caps_df,
            "imputer": imputer,
            "scaler": scaler,
            "cat_ord_imputer": None,
            "lower_q": lower_q,
            "upper_q": upper_q,
            "impute_strategy": impute_strategy,
            "n_features_fit": int(X_raw.shape[1]),
            "skipped_feature_names": [],
            "cat_ord_imputed_feature_names": [],
        }
        return bundle

    # --- Continuous preprocessing on subset only (reusing your exact functions) ---
    X_cont = X_raw[:, cont_idx]
    feature_names_cont = [feature_names[i] for i in cont_idx]

    X_cont_capped, caps_df_cont = cap_outliers_percentile(
        X_cont, feature_names_cont, lower_q=lower_q, upper_q=upper_q
    )
    X_cont_imputed, imputer = impute_missing_features(X_cont_capped, strategy=impute_strategy)
    X_cont_scaled, scaler = standardize_features(X_cont_imputed)

    # --- Recombine (cat/ord columns untouched so far) ---
    X_scaled_full = np.asarray(X_raw, dtype=np.float32).copy()
    X_scaled_full[:, cont_idx] = X_cont_scaled

    # --- Mode-impute categorical/ordinal columns only ---
    X_scaled_full, cat_ord_imputer, cat_ord_idx = impute_categorical_ordinal_mode(
        X_scaled_full, feature_names, meta
    )

    # caps_df for full feature space: NaN for skipped columns
    caps_df_full = pd.DataFrame({"lower": np.nan, "upper": np.nan}, index=feature_names)
    caps_df_full.loc[feature_names_cont, :] = caps_df_cont.loc[feature_names_cont, :]

    bundle["X_scaled"] = X_scaled_full
    bundle[preproc_key] = {
        "feature_names": feature_names,
        "caps_df": caps_df_full,
        "imputer": imputer,
        "scaler": scaler,
        "cat_ord_imputer": cat_ord_imputer,
        "lower_q": lower_q,
        "upper_q": upper_q,
        "impute_strategy": impute_strategy,
        "n_features_fit": int(X_raw.shape[1]),
        "skipped_feature_names": [feature_names[i] for i in skip_idx],
        "cat_ord_imputed_feature_names": [feature_names[i] for i in cat_ord_idx],
    }

    return bundle





# def data_preprocessing_pipeline(
#     bundle: Dict[str, Any],
#     lower_q: float = 0.05,
#     upper_q: float = 0.95,
#     impute_strategy: str = "median",
#     preproc_key: str = "preproc",
# ) -> Dict[str, Any]:
#     """
#     Preprocess a bundle's feature matrix in a fixed, reproducible order and
#     attach the fitted preprocessing artifacts back onto the bundle.

#     This function assumes the bundle represents a single, aligned feature space:
#     `bundle["X_raw"]` must have columns ordered exactly as `bundle["feature_names"]`.

#     Processing steps (in order)
#     ---------------------------
#     1) Winsorization / percentile capping (column-wise):
#        Values below the `lower_q` percentile and above the `upper_q` percentile
#        are clipped per feature. The per-feature cap values are stored in `caps_df`.

#     2) Missing-value imputation (column-wise):
#        NaNs are filled using a fitted `SimpleImputer` with the given strategy
#        (median by default).

#     3) Standardization (column-wise):
#        A fitted `StandardScaler` is applied to produce zero-mean, unit-variance
#        features per column.

#     Inputs
#     ------
#     bundle : dict
#         Must contain:
#           - "X_raw": np.ndarray of shape (n_samples, n_features)
#               Raw feature matrix.
#           - "feature_names": list[str] of length n_features
#               Feature names aligned to the columns of X_raw.

#         Any additional keys (labels, groups, metadata, etc.) are preserved.

#     lower_q, upper_q : float
#         Percentiles for winsorization (0 < lower_q < upper_q < 1). Default 0.05/0.95.

#     impute_strategy : str
#         Strategy passed to `sklearn.impute.SimpleImputer`, e.g. "median", "mean",
#         "most_frequent", "constant".

#     preproc_key : str
#         Key under which preprocessing artifacts and configuration are stored
#         (default: "preproc").

#     Bundle updates (added/overwritten keys)
#     ---------------------------------------
#     - bundle["X_scaled"] : np.ndarray, shape (n_samples, n_features)
#         Result after capping -> imputation -> standard scaling.

#     - bundle["feature_name_to_idx"] : dict[str, int]
#         Mapping from feature name to its column index in the full feature space.
#         This is important for later reduced-feature bundles (e.g., 6 features)
#         so you can recover their indices in the original space and invert scaling
#         for interpretability (e.g., PDP axes).

#     - bundle[preproc_key] : dict
#         Contains fitted objects and config needed to reproduce transforms:
#           - "feature_names": list[str] (the column order used during fitting)
#           - "caps_df": pd.DataFrame with columns ["lower","upper"] indexed by feature name
#           - "imputer": fitted SimpleImputer
#           - "scaler": fitted StandardScaler
#           - "lower_q", "upper_q", "impute_strategy": config values
#           - "n_features_fit": int, number of features fit on

#     Returns
#     -------
#     bundle : dict
#         The same dictionary object, mutated in-place for convenience.

#     Notes
#     -----
#     - The `StandardScaler` inversion (via mean_/scale_) returns values in the
#       *pre-scaled* space (i.e., after capping + imputation). Exact recovery of
#       the original uncapped/unimputed raw values is not possible unless you use
#       `bundle["X_raw"]`.
#     - If you later create a reduced-feature bundle (e.g., 6 features) and only
#       store its scaled matrix, you can map its feature names back to full indices
#       using `bundle["feature_name_to_idx"]` and invert scaling for those columns.
#     """
#     if "X_raw" not in bundle:
#         raise KeyError("bundle must contain key 'X_raw'")
#     if "feature_names" not in bundle:
#         raise KeyError("bundle must contain key 'feature_names'")

#     X_raw = bundle["X_raw"]
#     feature_names = list(bundle["feature_names"])

#     # mapping (needed later to compute selected_idx_full by name)
#     bundle["feature_name_to_idx"] = {name: i for i, name in enumerate(feature_names)}

#     # 1) cap
#     X_capped, caps_df = cap_outliers_percentile(
#         X_raw, feature_names, lower_q=lower_q, upper_q=upper_q
#     )

#     # 2) impute
#     X_imputed, imputer = impute_missing_features(X_capped, strategy=impute_strategy)

#     # 3) scale
#     X_scaled, scaler = standardize_features(X_imputed)

#     bundle["X_scaled"] = X_scaled

#     bundle[preproc_key] = {
#         "feature_names": feature_names,
#         "caps_df": caps_df,
#         "imputer": imputer,
#         "scaler": scaler,
#         "lower_q": lower_q,
#         "upper_q": upper_q,
#         "impute_strategy": impute_strategy,
#         "n_features_fit": int(X_raw.shape[1]),
#     }

#     return bundle



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